"""Prepare SORRY-Bench base (440) as a recyclebench behaviour CSV (Goal, Category).

Why a dedicated script (not --hf-dataset in prepare_behaviours.py):
  * SORRY-Bench stores the instruction in a `turns` LIST, not a flat column, so the
    generic goal-column autodetect misses it.
  * The HF dataset ships the 440 base prompts PLUS 20 linguistic mutations (~9.2K rows).
    We take ONLY the base 440 — the canonical, intent-preservable instructions. The 20
    mutations (ascii/caesar/morse/translate/role_play/…) are paraphrase VARIANTS; your
    recycle arm generates its own intent-gated paraphrases, so the mutations must NOT
    enter the behaviour pool (they would double-count and some aren't meaning-preserving).

Output matches datasets.load_jailbreakbench: a CSV with columns `Goal, Category`,
deduplicated on Goal — identical shape to data/advbench.csv etc., so it drops straight
into `pilot-pminus --behaviours data/sorrybench.csv` with no loader change.

Prereqs on your Mac (SORRY-Bench is gated):
  pip install datasets
  # accept the license once at huggingface.co/datasets/sorry-bench/sorry-bench-202503
  huggingface-cli login

Usage:
  python3 tools/prepare_sorrybench.py --out data/sorrybench.csv                 # from HF
  python3 tools/prepare_sorrybench.py --from-jsonl question.jsonl --out data/sorrybench.csv
  python3 tools/prepare_sorrybench.py --from-csv sorry_bench_202503.csv --out data/sorrybench.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

HF_DATASET = "sorry-bench/sorry-bench-202503"

_GOAL_FLAT = ["question", "prompt", "instruction", "text", "goal", "Goal"]
_CAT_KEYS = ["category", "Category", "fine_category", "category_name"]
_STYLE_KEYS = ["prompt_style", "style", "mutation"]


def _goal_of(rec: dict):
    """Instruction text: SORRY-Bench base keeps it in a one-element `turns` list."""
    t = rec.get("turns")
    if isinstance(t, (list, tuple)) and t:
        return str(t[0]).strip()
    if isinstance(t, str) and t.strip():
        return t.strip()
    for k in _GOAL_FLAT:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _cat_of(rec: dict):
    for k in _CAT_KEYS:
        if k in rec and rec[k] not in (None, ""):
            return str(rec[k]).strip()
    return "unknown"


def _is_base(rec: dict) -> bool:
    """Keep base prompts only. If a style field exists, base == 'base'; if no style
    field exists at all (e.g. a base-only question.jsonl), every row is base."""
    for k in _STYLE_KEYS:
        if k in rec and rec[k] not in (None, ""):
            return str(rec[k]).strip().lower() in ("base", "original", "vanilla")
    return True


_GATED_HELP = """
SORRY-Bench is a GATED dataset — you must authenticate once:

  1. Log in / sign up at huggingface.co
  2. Open the dataset page and click "Agree and access repository":
       https://huggingface.co/datasets/{ds}
     (usually granted instantly; if it says "pending", it's manual approval — wait for the email)
  3. Make a READ token: https://huggingface.co/settings/tokens
  4. Authenticate the CLI (either works):
       huggingface-cli login          # older
       hf auth login                  # newer
     …or just:  export HF_TOKEN=hf_xxx
  5. Re-run this script.

No account / approval pending? Download the base file by hand from the dataset's
"Files and versions" tab (question.jsonl) while logged in, then:
  python3 tools/prepare_sorrybench.py --from-jsonl question.jsonl --out {out}
"""


def _records_from_hf(dataset=HF_DATASET, out="data/sorrybench.csv"):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("needs the datasets library: pip install datasets")
    try:
        ds = load_dataset(dataset, token=True)   # token=True -> use your cached HF login
    except Exception as e:
        msg = str(e)
        if any(w in msg.lower() for w in ("gated", "authenticated", "401", "403", "access")):
            raise SystemExit(_GATED_HELP.format(ds=dataset, out=out))
        raise
    split = "train" if "train" in ds else list(ds.keys())[0]
    return [dict(r) for r in ds[split]]


def _records_from_jsonl(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _records_from_csv(path):
    text = Path(path).read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def build(records):
    out, seen = [], set()
    n_base = 0
    for rec in records:
        if not _is_base(rec):
            continue
        n_base += 1
        goal = _goal_of(rec)
        if not goal or goal in seen:
            continue
        seen.add(goal)
        out.append({"Goal": goal, "Category": _cat_of(rec)})
    return out, n_base


def write(out_path, records):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Goal", "Category"])
        w.writeheader()
        w.writerows(records)
    return len(records)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-jsonl", help="local question.jsonl (base) you downloaded")
    src.add_argument("--from-csv", help="local sorry_bench_*.csv (base) you downloaded")
    ap.add_argument("--out", required=True, help="output CSV (Goal, Category)")
    ap.add_argument("--dataset", default=HF_DATASET,
                    help="HF dataset id (default sorry-bench-202503; use …-202406 if you have that one)")
    args = ap.parse_args(argv)

    if args.from_jsonl:
        recs, origin = _records_from_jsonl(args.from_jsonl), args.from_jsonl
    elif args.from_csv:
        recs, origin = _records_from_csv(args.from_csv), args.from_csv
    else:
        recs, origin = _records_from_hf(args.dataset, args.out), f"hf:{args.dataset}"

    records, n_base = build(recs)
    n = write(args.out, records)
    cats = sorted({r["Category"] for r in records})
    print(f"wrote {n} base behaviours to {args.out}")
    print(f"  source     : {origin}  ({len(recs)} rows scanned, {n_base} base)")
    print(f"  categories : {len(cats)} distinct"
          + ("" if len(cats) != 1 else "  [!] only 1 — category field may be missing"))
    if n != 440:
        print(f"  [!] expected 440 base behaviours, got {n} — check the source/version")
    print(f"  next       : python3 -m recyclebench pilot-pminus --behaviours {args.out} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())