#!/usr/bin/env python3
"""tools/mech_causal.py -- does the displacement along the refusal direction CAUSE the re-opening? (white-box, causal)

Registered in PREREG_MECH_CAUSAL_20261031.md. `tools/mech_probe.py` showed, in llama3.1-8b, that re-opening rewordings sit
lower on the model's refusal direction than the canonical request. This tool intervenes on that direction, on the exact
Ollama weights (loader, tokenizer, chat template and refusal phrases from mech_probe):

  A   rewording  + delta * r   at layer l*, every position   (delta = canonical minus rewording projection, last token)
  A0  rewording  + delta * u   (u: random unit direction orthogonal to r, seed 20261101)
  B   canonical  - delta * r   (delta from each matched rewording)        B0  the same along u
  C   canonical, r ablated from the embeddings and every layer output    C0  u ablated instead
and measures P(refusal opening) (teacher-forced, one batched pass over the 12 refusal phrases). Primary test per fix
(n = 23): A raises P more than A0, exact sign test; secondary B against B0 and C against C0. Numbers only, no text; nothing
is generated beyond the teacher-forced refusal phrases.

  python3 tools/mech_causal.py                         # results/mech/causal_anchor_sorrybench_<stamp>.json
  python3 tools/mech_causal.py --limit 1 --out <file>  # smoke test on one fix
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import mech_probe as mp                                               # noqa: E402

NEW, SEED = mp.NEW, 20261101
PREREG = "PREREG_MECH_CAUSAL_20261031.md"


class Model:
    def __init__(self, device):
        import torch
        self.torch, self.device = torch, device
        self.model, self.tok, self.info = mp.load_gguf_llama(mp.ollama_blob(mp.OLLAMA_TAG[NEW]), device, torch.float16)
        self.layers = self.model.model.layers
        self.prefixes = [self.tok(p, add_special_tokens=False).input_ids for p in mp.Probe.REFUSAL_PREFIXES]

    def ids(self, prompt):
        return self.tok(mp.chat(prompt), add_special_tokens=False).input_ids

    def vec(self, v: np.ndarray):
        return self.torch.tensor(v, dtype=self.torch.float16, device=self.device)

    def last_resid(self, prompt) -> np.ndarray:
        """residual stream at the last prompt token after every decoder layer, (L, d); sees active interventions"""
        torch, cap = self.torch, {}
        hs = [layer.register_forward_hook(lambda _m, _a, o, j=j: cap.__setitem__(j, o[0, -1].float().cpu().numpy()))
              for j, layer in enumerate(self.layers)]
        try:
            with torch.no_grad():
                self.model(torch.tensor([self.ids(prompt)], device=self.device), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return np.stack([cap[j] for j in range(len(self.layers))])

    def p_refusal(self, prompt) -> float:
        """P(the reply opens with a refusal phrase): the 12 teacher-forced continuations in one right-padded batch"""
        torch = self.torch
        base = self.ids(prompt)
        seqs = [base + p for p in self.prefixes]
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), 128009, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(seqs), T), dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, device=self.device)
            mask[i, :len(s)] = 1
        with torch.no_grad():
            lp = torch.log_softmax(self.model(x, attention_mask=mask, use_cache=False).logits.float(), -1)
        b, total = len(base), 0.0
        for i, p in enumerate(self.prefixes):
            pos = torch.arange(b - 1, b - 1 + len(p), device=self.device)
            total += float(torch.exp(lp[i, pos, torch.tensor(p, device=self.device)].sum()))
        return min(1.0, total)

    @contextmanager
    def add(self, layer: int, v: np.ndarray):
        t = self.vec(v)
        h = self.layers[layer].register_forward_hook(lambda _m, _a, o: o + t)
        try:
            yield
        finally:
            h.remove()

    @contextmanager
    def ablate(self, unit: np.ndarray):
        u = self.torch.tensor(unit, dtype=self.torch.float32, device=self.device)

        def proj(_m, _a, o):
            of = o.float()
            return (of - (of @ u)[..., None] * u).to(o.dtype)
        hs = [self.model.model.embed_tokens.register_forward_hook(proj)] + [l.register_forward_hook(proj) for l in self.layers]
        try:
            yield
        finally:
            for h in hs:
                h.remove()


def direction(M: Model, harmful, harmless):
    H = np.stack([M.last_resid(p) for p in harmful])                  # (n, L, d)
    B = np.stack([M.last_resid(p) for p in harmless])
    r = H.mean(0) - B.mean(0)
    r_hat = r / np.linalg.norm(r, axis=-1, keepdims=True).clip(1e-6)
    ph, pb = np.einsum("nld,ld->nl", H, r_hat), np.einsum("nld,ld->nl", B, r_hat)
    dprime = (ph.mean(0) - pb.mean(0)) / np.sqrt(0.5 * (ph.var(0) + pb.var(0))).clip(1e-6)
    lstar = int(np.argmax(dprime))
    return r_hat[lstar], lstar, dprime


def sign_test(xs):
    from scipy.stats import binomtest
    pos, neg = sum(x > 0 for x in xs), sum(x < 0 for x in xs)
    return dict(positive=int(pos), negative=int(neg), ties=int(len(xs) - pos - neg),
                p_value=float(binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else None)


def summarise(items):
    from recyclebench.stats import wilson_interval
    fa = [np.mean([r["P_A"] - r["P_A0"] for r in it["reopen"]]) for it in items]
    fb = [np.mean([r["P_B0"] - r["P_B"] for r in it["reopen"]]) for it in items]
    fc = [it["P_C0"] - it["P_C"] for it in items]
    rew = [r for it in items for r in it["reopen"]]
    rest = sum(r["P_A"] >= it["P_canonical"] for it in items for r in it["reopen"])
    w = wilson_interval(rest, len(rew))
    med = lambda xs: float(np.median(xs))
    out = dict(
        n_fixes=len(items), n_rewordings=len(rew),
        A_vs_A0=dict(sign_test=sign_test(fa), median_fix_difference=med(fa),
                     median_increase_A=med([np.mean([r["P_A"] - r["P_rewording"] for r in it["reopen"]]) for it in items]),
                     median_increase_A0=med([np.mean([r["P_A0"] - r["P_rewording"] for r in it["reopen"]]) for it in items]),
                     restored=dict(k=int(rest), n=len(rew), rate=round(w.point, 4), ci=[round(w.low, 4), round(w.high, 4)])),
        B_vs_B0=dict(sign_test=sign_test(fb), median_fix_difference=med(fb),
                     median_decrease_B=med([np.mean([it["P_canonical"] - r["P_B"] for r in it["reopen"]]) for it in items]),
                     median_decrease_B0=med([np.mean([it["P_canonical"] - r["P_B0"] for r in it["reopen"]]) for it in items])),
        C_vs_C0=dict(sign_test=sign_test(fc), median_fix_difference=med(fc),
                     median_P_C=med([it["P_C"] for it in items]), median_P_C0=med([it["P_C0"] for it in items])),
        median_P_canonical=med([it["P_canonical"] for it in items]), median_P_rewording=med([r["P_rewording"] for r in rew]))
    sig = lambda t: t["sign_test"]["p_value"] is not None and t["sign_test"]["p_value"] < 0.05 \
        and t["sign_test"]["positive"] > t["sign_test"]["negative"]
    out["reading"] = dict(sufficient="causally sufficient" if sig(out["A_vs_A0"]) else "not shown",
                          necessary="necessary" if sig(out["B_vs_B0"]) else "not shown")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None, help="first N fixes only (smoke test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.out) if args.out else ROOT / f"results/mech/causal_anchor_sorrybench_{stamp}.json"
    if out.exists():
        raise SystemExit(f"{out} exists -- refusing to overwrite")
    items_all, harmful, harmful_bids = mp.collect_items("all")
    items_all = items_all[:args.limit] if args.limit else items_all
    print(f"{len(items_all)} fixes, {sum(sum(p['role'] == 'reopen' for p in it['prompts']) for it in items_all)} re-opening "
          f"rewordings; direction set {len(harmful)} refused + {len(mp.HARMLESS)} harmless", flush=True)
    for tag in set(mp.OLLAMA_TAG.values()) | {"qwen2.5:14b", "qwen2.5:3b", "gemma4:e4b"}:
        subprocess.run(["ollama", "stop", tag], capture_output=True)    # free the GPU memory Ollama holds
    t0 = time.time()
    M = Model(args.device)
    print(f"loaded {M.info['tensors']} tensors ({', '.join(M.info['types'])}) in {time.time() - t0:.0f}s", flush=True)
    r_hat, lstar, dprime = direction(M, harmful, mp.HARMLESS)
    u = np.random.default_rng(SEED).standard_normal(r_hat.shape[0])
    u -= (u @ r_hat) * r_hat
    u /= np.linalg.norm(u)
    print(f"direction: layer {lstar} (output of decoder layer {lstar}), d' {dprime[lstar]:.2f}", flush=True)
    rows = []
    for k, it in enumerate(items_all, 1):
        can = it["prompts"][0]
        s_can = float(M.last_resid(can["text"])[lstar] @ r_hat)
        rec = dict(bid=it["bid"], category=it["category"], P_canonical=M.p_refusal(can["text"]), proj_canonical=s_can)
        with M.ablate(r_hat):
            rec["P_C"] = M.p_refusal(can["text"])
        with M.ablate(u):
            rec["P_C0"] = M.p_refusal(can["text"])
        rec["reopen"] = []
        for p in (p for p in it["prompts"] if p["role"] == "reopen"):
            s = float(M.last_resid(p["text"])[lstar] @ r_hat)
            d = s_can - s
            r = dict(index=p["index"], cls=p["cls"], proj=s, delta=d, P_rewording=M.p_refusal(p["text"]))
            with M.add(lstar, d * r_hat):
                r["P_A"] = M.p_refusal(p["text"])
                r["proj_after_A"] = float(M.last_resid(p["text"])[lstar] @ r_hat)
            with M.add(lstar, d * u):
                r["P_A0"] = M.p_refusal(p["text"])
            with M.add(lstar, -d * r_hat):
                r["P_B"] = M.p_refusal(can["text"])
            with M.add(lstar, -d * u):
                r["P_B0"] = M.p_refusal(can["text"])
            rec["reopen"].append(r)
        rows.append(rec)
        print(f"  [{k}/{len(items_all)}] {it['bid']}: P canonical {rec['P_canonical']:.2f} · ablated {rec['P_C']:.2f} "
              f"(random {rec['P_C0']:.2f}) · rewordings " + ", ".join(
                  f"{r['P_rewording']:.2f}->A {r['P_A']:.2f}/A0 {r['P_A0']:.2f}" for r in rec["reopen"]), flush=True)
    summ = summarise(rows)
    doc = dict(generated=stamp, prereg=PREREG, model=dict(key=NEW, ollama=mp.OLLAMA_TAG[NEW], **M.info, dtype="float16",
                                                         device=args.device),
               direction=dict(layer=lstar, dprime=dprime.tolist(), harmful_bids=harmful_bids, n_harmless=len(mp.HARMLESS),
                              random_seed=SEED),
               measure="P(refusal opening): teacher-forced probability of the 12 refusal phrases of tools/mech_probe.py",
               note="numbers only: no prompt or response text is stored", summary=summ, items=rows,
               limited=bool(args.limit))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(summ, indent=1))
    print(f"-> {out}")
    del M
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
