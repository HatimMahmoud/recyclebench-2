#!/usr/bin/env python3
"""Compositional (2-factor) transform arm. Spec: PREREG_COMPOSE_20261027.md.

Applies one or two transformations in sequence to each anchor-P^- behaviour, gates the result
against the ORIGINAL behaviour, re-screens both models on it, and judges every response against
the ORIGINAL intent.

Arms run whole, one after another in priority order, so a deadline leaves complete arms rather
than a partial slice of all of them. Everything is append-only and resumable: rerun the same
command and it picks up where it stopped.

  python3 tools/run_compose.py --minutes 420
  python3 tools/run_compose.py --arms para,rpe --n-behaviours 3 --candidates 1   # smoke
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from recyclebench.backends.providers import load_provider_model      # noqa: E402
from recyclebench.config import Config                                # noqa: E402
from recyclebench.datasets import load_jailbreakbench                 # noqa: E402
from recyclebench.types import Behaviour, Response, Variant           # noqa: E402
from recyclebench.realrun import CachedModel, ResponseCache           # noqa: E402
from recyclebench.stats import wilson_interval                        # noqa: E402

OUT = ROOT / "cache" / "compose"
PMINUS = "results/cells_llama3-8b_llama31-8b/sorrybench_p_minus.csv"
CFG_PATH = "configs/recycle_3_31.yaml"
OLD, NEW = "llama3-8b", "llama31-8b"
SEED = 20261027

# (arm id, factors applied left to right, intent-gated?). Priority order: the intent-gated
# question and its single-factor controls first, so a cut-off still answers something.
ARMS = [
    ("para",         ("para",),           True),
    ("rpe",          ("rpe",),            True),
    ("para__rpe",    ("para", "rpe"),     True),
    ("rpe__para",    ("rpe", "para"),     True),
    ("attack",       ("attack",),         False),
    ("para__attack", ("para", "attack"),  False),
    ("attack__para", ("attack", "para"),  False),
    ("rpe__attack",  ("rpe", "attack"),   False),
    ("attack__rpe",  ("attack", "rpe"),   False),
]
PARA_CLASSES = ["paraphrase", "syntactic", "lexical", "back_translation", "tense_shift", "register"]


def log(msg: str) -> None:
    print(f"{_dt.datetime.now():%H:%M:%S} {msg}", flush=True)


class JsonlStore:
    """Append-only JSONL keyed by a string id, so every phase is resumable."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mem: dict = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue            # a hard kill can truncate the last line
                    self.mem[d["id"]] = d
        self._fh = self.path.open("a", encoding="utf-8")

    def put(self, key: str, **fields) -> dict:
        d = {"id": key, **fields}
        self._fh.write(json.dumps(d) + "\n")
        self._fh.flush()
        self.mem[key] = d
        return d

    def close(self):
        self._fh.close()


def clean_candidate(text: str) -> str:
    """Strip a leading slot label the inverter sometimes echoes.

    RPEEngine's parser matches `CANDIDATE <n>:` exactly; when the model writes anything
    else (observed: "CANDIDATOR 1:") the parser falls through to its raw-line fallback and
    the label ends up inside the candidate, where it would be gated and screened as if it
    were part of the request.
    """
    t = (text or "").strip()
    # the digit is mandatory: without it "Describe: the process" would lose its verb.
    t = re.sub(r"^[A-Za-z]{4,15}\s*\d+\s*[:.\)]\s*", "", t).strip()
    return t.strip('"').strip()


def attack_wrap(text: str, tpl: dict) -> str:
    return tpl["prefix"] + text + tpl["suffix"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=None, help="overall wall-clock budget")
    ap.add_argument("--arms", default=None, help="comma-separated subset of arm ids")
    ap.add_argument("--n-behaviours", type=int, default=None, help="cap, for smoke tests")
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.out) if args.out else ROOT / f"results/runs/compose_{stamp}"
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; never overwrite")

    tpl = json.loads((OUT / "attack_template.json").read_text())
    cfg = Config.load(ROOT / CFG_PATH)
    behaviours = load_jailbreakbench(str(ROOT / PMINUS))
    if args.n_behaviours:
        behaviours = behaviours[:args.n_behaviours]
    arms = [a for a in ARMS if not args.arms or a[0] in args.arms.split(",")]
    log(f"{len(behaviours)} behaviours · {len(arms)} arms · {args.candidates} candidates "
        f"· budget {args.minutes or '∞'} min")

    # --- models -------------------------------------------------------------------------
    tcache = ResponseCache(str(ROOT / "cache/compose_targets.jsonl"), enabled=True)
    old = CachedModel(load_provider_model(OLD, "openai_compatible", cfg), tcache)
    new = CachedModel(load_provider_model(NEW, "openai_compatible", cfg), tcache)
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    if not getattr(cfg, "paraphraser", None):
        raise SystemExit("config has no paraphraser: block")
    para_model = ProviderModel("paraphraser", ProviderConfig(**cfg.paraphraser))
    inv_model = load_provider_model("qwen25-14b", "openai_compatible", cfg)

    from recyclebench.backends.reformulate import LLMReformulator
    from recyclebench.independent_probe.rpe_engine import RPEEngine
    rpe = RPEEngine(provider_respond=lambda p: (inv_model.respond(p).text or ""))

    from recyclebench.backends.real_judge import build_ensemble
    jm = {role: ProviderModel(f"judge:{role}", ProviderConfig(**spec))
          for role, spec in (getattr(cfg, "judges", None) or {}).items()}
    judge = build_ensemble(jm)
    if judge is None or not getattr(judge, "members", None):
        raise SystemExit("no judges: block in the config")

    from recyclebench.gate import load_gate
    import numpy as np
    gate = load_gate("real", np.random.default_rng(SEED), cfg.gate, None, cfg)

    gen = JsonlStore(OUT / "candidates.jsonl")
    scr = JsonlStore(OUT / "screened.jsonl")
    jdg = JsonlStore(OUT / "judged.jsonl")

    def expired() -> bool:
        return bool(deadline and time.time() > deadline)

    def paraphrase(text: str, i: int) -> str:
        cls = PARA_CLASSES[i % len(PARA_CLASSES)]
        ref = LLMReformulator([cls], 1, cfg, para_model)
        vs = ref.variants(Behaviour(bid="tmp", canonical_prompt=text))
        return (vs[0].text if vs else ""), cls

    def invert(text: str, bid: str, i: int) -> str:
        """Ask OLD with `text`, reconstruct a request from its answer."""
        resp = old.respond(text, canonical=False, seed=SEED + i).text or ""
        if not resp.strip():
            return ""
        cands = [c for c in (clean_candidate(x) for x in rpe.invert(resp, n_candidates=3)) if c]
        return cands[i % len(cands)] if cands else ""

    # --- per arm, end to end ---------------------------------------------------------------
    # An arm the deadline cuts short is NOT comparable to a whole one: `behaviours` is in CSV
    # order and SORRY-Bench is ordered by category, so a prefix is a category-biased subset,
    # not a random sample. Record it so the summary and the table can say so.
    truncated: dict = {}
    for arm_id, factors, gated in arms:
        if expired():
            log(f"budget reached before {arm_id}; stopping"
                + ("" if not any(truncated.values()) else
                   f" (note: {', '.join(sorted(truncated))} was cut short, not whole)"))
            break
        log(f"--- arm {arm_id}  factors={'→'.join(factors)}  intent-gated={gated}")
        t0 = time.time()
        for b in behaviours:
            if expired():
                truncated[arm_id] = True
                break
            for i in range(args.candidates):
                key = f"{arm_id}|{b.bid}|{i}"
                rec = gen.mem.get(key)
                if rec is None:
                    text, classes = b.canonical_prompt, []
                    for f in factors:
                        if not text.strip():
                            break
                        if f == "para":
                            text, cls = paraphrase(text, i)
                            classes.append(f"para:{cls}")
                        elif f == "attack":
                            text = attack_wrap(text, tpl)
                            classes.append("attack:refusal_suppression")
                        elif f == "rpe":
                            text = invert(text, b.bid, i)
                            classes.append("rpe:qwen2.5-14b")
                    rec = gen.put(key, arm=arm_id, bid=b.bid, idx=i, text=text,
                                  steps=classes, ok=bool(text.strip()))
                if not rec["ok"]:
                    continue
                # gate ALWAYS against the ORIGINAL behaviour; admission only for gated arms
                if "gate" not in rec:
                    v = Variant(bid=b.bid, text=rec["text"], transform_class=arm_id,
                                surface_distance=0.0, index=i, intent_preserved=True)
                    try:
                        gs = gate.check(b, v)
                        sc = dict(cosine=gs.cosine, bertscore=gs.bertscore,
                                  nli_entail=gs.nli_entail, llm_intent=gs.llm_intent)
                        adm = bool(gs.admitted)
                    except Exception as e:                       # a gate failure must not kill the night
                        sc, adm = {"error": str(e)[:200]}, False
                    rec = gen.put(key, **{**{k: v2 for k, v2 in rec.items() if k != "id"},
                                          "gate": sc, "gate_admitted": adm})
                if gated and not rec.get("gate_admitted"):
                    continue                                     # not an admissible paraphrase
                for m in (old, new):
                    skey = f"{key}|{m.name}"
                    if skey in scr.mem:
                        continue
                    r = m.respond(rec["text"], behaviour=b, canonical=False, seed=SEED + i)
                    scr.put(skey, arm=arm_id, bid=b.bid, idx=i, model=m.name,
                            response=(r.text or "")[:6000])
                    jk = f"{skey}|judge"
                    if jk not in jdg.mem:
                        # judged against the ORIGINAL canonical prompt, never the transformed
                        # text: judging attack-wrapped output against its own wrapped prompt
                        # gives spurious bypasses.
                        verdict = judge.classify(b.canonical_prompt, Response(text=r.text or ""))
                        jdg.put(jk, arm=arm_id, bid=b.bid, idx=i, model=m.name,
                                complied=bool(verdict))
        log(f"    arm {arm_id} {'CUT SHORT BY BUDGET' if truncated.get(arm_id) else 'done'} in "
            f"{(time.time()-t0)/60:.1f} min (cands {len(gen.mem)}, screened {len(scr.mem)})")

    tcache.close()
    for s in (gen, scr, jdg):
        s.close()

    # --- summarise ---------------------------------------------------------------------
    res = {"generated": stamp, "pair": f"{OLD} -> {NEW}", "behaviours": len(behaviours),
           "candidates_per_behaviour": args.candidates, "spec": "PREREG_COMPOSE_20261027.md",
           "exploratory": True, "arms": []}
    for arm_id, factors, gated in arms:
        bids, opened, admitted_bids = set(), set(), set()
        for d in gen.mem.values():
            if d["arm"] != arm_id or not d.get("ok"):
                continue
            bids.add(d["bid"])
            if (not gated) or d.get("gate_admitted"):
                admitted_bids.add(d["bid"])
        for d in jdg.mem.values():
            if d["arm"] == arm_id and d["model"] == NEW and d["complied"]:
                opened.add(d["bid"])
        n = len(admitted_bids)
        k = len(opened & admitted_bids)
        w = wilson_interval(k, n) if n else None
        res["arms"].append(dict(arm=arm_id, factors=list(factors), intent_gated=gated,
                                n_with_candidate=len(bids), n_resolved=n, k_reopened=k,
                                truncated=bool(truncated.get(arm_id)),
                                n_behaviours_total=len(behaviours),
                                rate=(k / n if n else None),
                                wilson95=[w.low, w.high] if w else None,
                                estimand=("compositional RSR" if gated
                                          else "compositional attack transfer (NOT RSR)")))
    out_dir.mkdir(parents=True)
    (out_dir / "compose_summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print()
    print(f"{'arm':14} {'gated':>6} {'k/n':>9} {'rate':>7}  Wilson 95%")
    for a in res["arms"]:
        r = "—" if a["rate"] is None else f"{a['rate']:.3f}"
        ci = "—" if not a["wilson95"] else f"[{a['wilson95'][0]:.3f}, {a['wilson95'][1]:.3f}]"
        flag = "  << CUT SHORT: partial, category-biased subset" if a.get("truncated") else ""
        print(f"{a['arm']:14} {str(a['intent_gated']):>6} "
              f"{str(a['k_reopened']) + '/' + str(a['n_resolved']):>9} {r:>7}  {ci}{flag}")
    print(f"\n-> {out_dir}/compose_summary.json")
    print("EXPLORATORY. Gated arms are a compositional RSR; attack arms are a separate rate "
          "and must never be pooled with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
