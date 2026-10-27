"""Is the local PTP failure a SCALE result or an IMPLEMENTATION bug?

Five local runs all fail Gate A, and the best one still scores below a constant
string. Before that is reported as "PTP needs the paper's scale" -- and before any
H100 hours are spent on it -- the failure has to be attributed. Three outcomes are
materially different for the thesis:

  (a) the model cannot reproduce prompts it was explicitly fine-tuned on
      -> something is wrong with the loop, not with the scale. A cluster run would
         reproduce the bug at 20x the cost.
  (b) it reproduces training prompts well but held-out ones badly
      -> the loop is sound and the inverse mapping has not generalised. That is a
         data/scale statement, and it is what justifies the cluster run.
  (c) teacher-forced next-token accuracy on the prompt span is high while free-running
      decoding is poor
      -> the mapping is partly learned and greedy decoding drifts away from it
         (exposure bias). Fixable with decoding, not with GPUs -- so it must be ruled
         out before attributing anything to scale.

So this measures all three, on the same held-out set the runs are scored on:

  memorisation   free-running inversion token F1 on fine-tuning pairs the model SAW,
                 sampled to the same n as the held-out set so the numbers compare
  generalisation the same on the held-out pairs (reproduces the run's reported figure)
  teacher-forced argmax accuracy on the reversed PROMPT span with the true prefix
                 supplied, bucketed by depth into the prompt

Depth buckets matter because in s_rev the prompt is generated last and backwards: the
first prompt token emitted is the prompt's LAST token, conditioned on the whole
response. Accuracy that starts high and decays with depth is drift; accuracy that is
flat at chance was never learned.

No model calls -- responses come from the stored fine-tuning pairs. Pure local compute.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recyclebench.ptp.metrics import token_f1                      # noqa: E402
from recyclebench.ptp.model import PTPConfig, build_model, pick_device   # noqa: E402
from recyclebench.ptp.reversal import build_training_sequence, reverse_ids  # noqa: E402
from recyclebench.ptp.train import invert_ids                      # noqa: E402
from recyclebench.ptp.vocab import CompactVocab                    # noqa: E402

OUT = ROOT / "cache" / "ptp"


def load_into(model, path: Path, device: str) -> bool:
    import torch
    if not path.exists():
        return False
    model.load_state_dict(torch.load(path, map_location=device))
    return True


def free_running(model, pairs, tok, cv, cfg, device, max_new_tokens=40, log=print):
    """Inversion exactly as the eval phase scores it, so figures are comparable."""
    out = []
    for i, r in enumerate(pairs, 1):
        rid = tok(r["response"], add_special_tokens=False)["input_ids"]
        rid = rid[:cfg.max_position_embeddings - 48]
        if cv:
            rid = cv.encode(rid)
        ids = invert_ids(model, rid, device=device, max_new_tokens=max_new_tokens)
        if cv:
            ids = cv.decode(ids)
        got = tok.decode(ids, skip_special_tokens=True)
        out.append(token_f1(got, r["prompt"]))
        if i % 25 == 0:
            log(f"    {i}/{len(pairs)}")
    return sum(out) / max(1, len(out))


def teacher_forced(model, pairs, tok, cv, cfg, device, buckets=(1, 2, 4, 8, 16)):
    """Argmax accuracy on the reversed prompt span, given the true prefix.

    Position j of the prompt span is predicted from logits at j-1, so the model is
    never asked to condition on its own output. `buckets` are depths into the prompt
    measured from the response boundary (bucket 1 = the prompt's last token).
    """
    import torch
    model.to(device).eval()
    hits = {b: [0, 0] for b in buckets}          # depth -> [correct, total]
    allc = alln = 0
    with torch.no_grad():
        for r in pairs:
            n_prompt = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
            seq = build_training_sequence(r["prompt"], r["response"], tok)
            if cv:
                seq = cv.encode(seq)
            seq = seq[:cfg.max_position_embeddings]
            if n_prompt < 2 or len(seq) <= n_prompt + 1:
                continue                          # prompt truncated away
            inp = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(input_ids=inp).logits[0]
            pred = logits.argmax(-1)              # pred[i] predicts seq[i+1]
            start = len(seq) - n_prompt           # first prompt position in s_rev
            for j in range(start, len(seq)):
                ok = int(pred[j - 1]) == seq[j]
                allc += ok
                alln += 1
                d = j - start + 1                 # 1 = prompt's last token
                for b in buckets:
                    if d <= b:
                        hits[b][0] += ok
                        hits[b][1] += 1
                        break
    return {"accuracy": allc / max(1, alln), "n_tokens": alln,
            "by_depth": {f"<={b}": (c / t if t else None) for b, (c, t) in hits.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n", type=int, default=86, help="pairs per condition")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    device = pick_device()

    pairs = [json.loads(l) for l in
             (OUT / "ftpairs_qwen2.5_3b.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    held = json.loads((OUT / "heldout.json").read_text())
    held_prompts = {h["prompt"] for h in held}
    train = [p for p in pairs if p["prompt"] not in held_prompts]
    rng = random.Random(7)
    seen = rng.sample(train, min(args.n, len(train)))
    unseen = held[:args.n]
    print(f"device {device} · {len(train)} training pairs, {len(held)} held out · "
          f"scoring {len(seen)} seen vs {len(unseen)} unseen")

    cv = CompactVocab.load(OUT / "compact_vocab.json") \
        if (OUT / "compact_vocab.json").exists() else None
    vsz = len(cv) if cv else len(tok)

    runs = [("27M (25k seeds)", PTPConfig.local(vsz), OUT / "local27m_25k" / "inverse_finetuned.pt"),
            ("92M (25k seeds)", PTPConfig.mid(vsz), OUT / "inverse_finetuned.pt")]

    report = {"generated": _dt.datetime.now().strftime("%Y%m%d_%H%M"),
              "device": device, "model_vocab": vsz,
              "n_seen": len(seen), "n_unseen": len(unseen), "runs": {}}

    for label, cfg, ckpt in runs:
        print(f"\n=== {label}  ({ckpt.name})")
        model = build_model(cfg)
        if not load_into(model, ckpt, device):
            print(f"    no checkpoint at {ckpt} -- skipped")
            continue
        tf_seen = teacher_forced(model, seen, tok, cv, cfg, device)
        tf_unseen = teacher_forced(model, unseen, tok, cv, cfg, device)
        print(f"  teacher-forced prompt-token accuracy  seen {tf_seen['accuracy']:.3f}  "
              f"unseen {tf_unseen['accuracy']:.3f}")
        print(f"    unseen by depth: " +
              "  ".join(f"{k} {v:.3f}" for k, v in tf_unseen["by_depth"].items() if v is not None))
        print("  free-running inversion (seen):")
        f1_seen = free_running(model, seen, tok, cv, cfg, device, log=lambda s: print(s))
        print("  free-running inversion (unseen):")
        f1_unseen = free_running(model, unseen, tok, cv, cfg, device, log=lambda s: print(s))
        print(f"  token F1   seen {f1_seen:.3f}   unseen {f1_unseen:.3f}")
        report["runs"][label] = {
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "teacher_forced_seen": tf_seen, "teacher_forced_unseen": tf_unseen,
            "token_f1_seen": f1_seen, "token_f1_unseen": f1_unseen,
        }

    # chance level for teacher-forced accuracy, for reference
    report["chance_accuracy"] = 1.0 / vsz
    dest = Path(args.out) if args.out else \
        OUT / f"diagnose_{report['generated']}.json"
    dest.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {dest}")

    print("\n--- reading ---")
    for label, r in report["runs"].items():
        seen_f1, unseen_f1 = r["token_f1_seen"], r["token_f1_unseen"]
        tf = r["teacher_forced_unseen"]["accuracy"]
        if seen_f1 < 0.30:
            verdict = ("cannot reproduce its OWN training pairs -> implementation/optimisation "
                       "problem, not scale")
        elif tf > 0.40 and unseen_f1 < 0.30:
            verdict = ("mapping partly learned but free-running decoding drifts -> decoding "
                       "problem, try sampling/beam before attributing to scale")
        else:
            verdict = "fits what it saw, fails what it did not -> generalisation gap, scale is the lever"
        print(f"  {label}: seen F1 {seen_f1:.3f} · unseen F1 {unseen_f1:.3f} · "
              f"teacher-forced {tf:.3f}\n      {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
