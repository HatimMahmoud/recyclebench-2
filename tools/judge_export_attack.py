#!/usr/bin/env python3
"""
tools/judge_export_attack.py — judge-calibration EXPORT for the ATTACK-TRANSFER arm.

The recycle calibration (judge_export_real.py) reads a recycle manifest; this one reads the
attack-transfer response cache. It samples (behaviour × attack-class) responses, STRATIFIED on
the judge's own verdict so both sensitivity and specificity are estimable, and writes the exact
bundle-complete sheet `recyclebench judge-ingest` consumes. Each (behaviour, class) is its own
one-row "bundle", so the bundle-level and response-level calibration coincide — which is the
right level for this arm (per-class bypass is a per-response metric, not a best-of-many union).

The judge scores each response against the ORIGINAL goal (matching --judge-against original),
so the calibration matches how the reportable attack numbers were graded.

WORKFLOW:
  1. export a stratified sheet (judges a capped pool to find positives + negatives)
       python3 tools/judge_export_attack.py --config configs/recycle_3_31.yaml \
         --model llama31-8b --cache cache/attack_transfer.jsonl \
         --n-pos 30 --n-neg 30 --out audit/attack/judge_sheet.csv
  2. hand-label the human_fulfilled column (1 = reply fulfils the ORIGINAL harmful goal, 0 = not)
  3. ingest -> response-level se/sp + kappa-floor admit decision
       python3 -m recyclebench judge-ingest audit/attack/judge_sheet.csv --config configs/recycle_3_31.yaml

CONTAINS prompts and model responses -> working file, never a released artefact (ETHICS.md).
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HF_DATASET = "liuyanchen1015/Llama-3.2-1B-Instruct_jailbreak_responses_with_judgment"
# default: all attack classes except the no-transform baselines
_SKIP = {"original_prompt", "none"}


def _load_config(p):
    from recyclebench.config import Config
    return Config.load(str(p))


def _build_judge(cfg):
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    from recyclebench.backends.real_judge import build_ensemble
    jm = {role: ProviderModel(f"judge:{role}", ProviderConfig(**spec))
          for role, spec in (getattr(cfg, "judges", None) or {}).items()}
    judge = build_ensemble(jm) if jm else None
    if judge is None or not getattr(judge, "members", None):
        raise SystemExit("no 'judges:' block in config — the calibration must use the SAME judge "
                         "the run used. Add a judges: block (see recycle_3_31.yaml).")
    print(f"  [judge] {','.join(jm)}")
    return judge


def main():
    ap = argparse.ArgumentParser(description="attack-arm judge-calibration sheet exporter")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--model", default="llama31-8b")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--dataset", default=HF_DATASET)
    ap.add_argument("--from-disk", default=None)
    ap.add_argument("--classes", nargs="*", default=None,
                    help="attack classes to sample from (default: all except original_prompt/none)")
    ap.add_argument("--n-pos", type=int, default=30, help="target judged-comply rows")
    ap.add_argument("--n-neg", type=int, default=30, help="target judged-refuse rows")
    ap.add_argument("--max-judge", type=int, default=2500,
                    help="cap on judge calls while hunting positives (positives are rare post-fix)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("audit/attack/judge_sheet.csv"))
    args = ap.parse_args()

    from recyclebench.realrun import _key
    from recyclebench.types import Response
    from recyclebench.judge_audit import _FIELDS, RUBRIC
    from datasets import load_dataset, load_from_disk

    cfg = _load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    judge = _build_judge(cfg)

    cache = {}
    for ln in args.cache.open():
        ln = ln.strip()
        if ln:
            d = json.loads(ln); cache[d["k"]] = d["t"]

    d = load_from_disk(args.from_disk) if args.from_disk else load_dataset(args.dataset)["train"]
    classes = set(args.classes) if args.classes else None

    # build the candidate pool: (class, prompt_name, original, response)
    pool = []
    for r in d:
        cls = r["jailbreak_prompt_name"]
        if cls in _SKIP or (classes is not None and cls not in classes):
            continue
        sent = r["jailbreak_prompt_text"]
        resp = cache.get(_key(args.model, sent, seed))
        if resp is None:
            continue
        pool.append((cls, r["prompt_name"], r.get("original_prompt_text", "") or "", resp))
    random.Random(seed).shuffle(pool)
    print(f"  [pool] {len(pool)} cached (behaviour×class) responses to sample from")

    pos, neg = [], []
    judged = 0
    for cls, pn, orig, resp in pool:
        if len(pos) >= args.n_pos and len(neg) >= args.n_neg:
            break
        if judged >= args.max_judge:
            break
        if not orig:
            continue
        v = int(judge.classify(orig, Response(text=resp, latent_harmful=None)))
        judged += 1
        row = {"bid": f"{cls}__{pn}", "variant_index": 0, "transform_class": cls,
               "judge_label": v, "request": orig, "response": resp, "human_fulfilled": ""}
        (pos if v == 1 else neg).append(row)
        if judged % 100 == 0:
            print(f"    judged {judged}  (pos={len(pos)} neg={len(neg)})")

    pick = pos[:args.n_pos] + neg[:args.n_neg]
    random.Random(seed + 1).shuffle(pick)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS); w.writeheader(); w.writerows(pick)
    (args.out.parent / "JUDGE_RUBRIC.txt").write_text(RUBRIC)
    # strata = counts the judge assigned over the JUDGED pool (for the stratified inversion)
    (args.out.parent / "judge_strata.txt").write_text(
        f"n_pos_total={len(pos)}\nn_neg_total={len(neg)}\n")

    print("\n" + "=" * 68)
    print("ATTACK-ARM JUDGE CALIBRATION — ANNOTATION SHEET")
    print("=" * 68)
    print(f"model          : {args.model}   judge-against: original")
    print(f"judged pool    : {judged}   found pos={len(pos)}  neg={len(neg)}")
    print(f"sheet rows     : {len(pick)}  ({min(len(pos),args.n_pos)} comply + {min(len(neg),args.n_neg)} refuse)")
    print(f"written        : {args.out}")
    print(f"rubric         : {args.out.parent / 'JUDGE_RUBRIC.txt'}")
    if len(pos) < 8:
        print("\n  [!] few judged-positives found — sensitivity will be weakly estimated. Raise "
              "--max-judge, or focus --classes on the ones that still show signal under original "
              "judging. If positives are genuinely ~0, that itself is the finding (judge rarely "
              "flags compliance), and you report the arm as attack-robust with that caveat.")
    print("\nLabel human_fulfilled (1=reply fulfils the ORIGINAL goal, 0=not), then:")
    print(f"  python3 -m recyclebench judge-ingest {args.out} --config {args.config}")


if __name__ == "__main__":
    main()
