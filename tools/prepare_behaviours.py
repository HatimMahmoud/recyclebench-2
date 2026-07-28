"""Prepare a behaviour CSV for recyclebench from a standard, published safety benchmark.

recyclebench ships NO harmful content. This helper fetches one of the standard, peer-reviewed
red-teaming benchmarks *from its official source at runtime* and rewrites it into the
``Goal`` (+ optional ``Category``) schema that ``datasets.load_jailbreakbench`` reads. The
harmful strings therefore live only in the benchmark you choose to download, on your machine.

Two modes:

  # 1. convert a CSV you already downloaded (any of the known schemas -> Goal/Category)
  python tools/prepare_behaviours.py --from-csv path/to/downloaded.csv --out data/behaviours.csv

  # 2. fetch a named benchmark from its official repo, then convert
  python tools/prepare_behaviours.py --benchmark advbench     --out data/advbench.csv
  python tools/prepare_behaviours.py --benchmark harmbench    --out data/harmbench.csv
  python tools/prepare_behaviours.py --benchmark strongreject --out data/strongreject.csv
  python tools/prepare_behaviours.py --benchmark jbb          --out data/jbb.csv   # needs `datasets`

Benchmarks and their citations (all already in the exposé bibliography):
  advbench      Zou et al. 2023      llm-attacks/llm-attacks
  harmbench     Mazeika et al. 2024  centerforaisafety/HarmBench
  strongreject  Souly et al. 2024    alexandrasouly/strongreject
  jbb           Chao et al. 2024     JailbreakBench/JBB-Behaviors (HuggingFace)

You can pool several: convert each, then `cat` the rows (dedupe on Goal). Pooling across
benchmarks is the recommended way to enlarge P^- for the real run.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

# Official raw-CSV sources (the harmful strings are fetched from here, at runtime, by you).
_SOURCES = {
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
                "data/advbench/harmful_behaviors.csv",
    "harmbench": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
                 "data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "strongreject": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
                    "strongreject_dataset/strongreject_dataset.csv",
}

# Column auto-detection: the "goal" text and (optionally) a category, across known schemas.
_GOAL_CANDIDATES = ["original_prompt_text", "Goal", "goal", "Behavior", "behavior",
                    "forbidden_prompt", "prompt", "question", "text", "jailbreak_prompt_text"]
_CATEGORY_CANDIDATES = ["Category", "category", "SemanticCategory", "semantic_category",
                        "FunctionalCategory", "topic"]


def _pick(fieldnames, candidates):
    lower = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in fieldnames:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def convert(rows, fieldnames):
    """Map any known benchmark schema to a list of {Goal, Category} dicts.

    Deduplicates on Goal (keeping first occurrence), so a dataset that repeats each canonical
    behaviour across many jailbreak variants collapses to one row per behaviour.
    """
    goal_col = _pick(fieldnames, _GOAL_CANDIDATES)
    if goal_col is None:
        raise SystemExit(
            f"could not find a goal/behaviour column in {fieldnames}. "
            f"Expected one of {_GOAL_CANDIDATES}. Pass --goal-col to override.")
    cat_col = _pick(fieldnames, _CATEGORY_CANDIDATES)
    out, seen = [], set()
    for r in rows:
        goal = (r.get(goal_col) or "").strip()
        if not goal or goal in seen:
            continue
        seen.add(goal)
        cat = ((r.get(cat_col) or "").strip() if cat_col else "") or "unknown"
        out.append({"Goal": goal, "Category": cat})
    return out, goal_col, cat_col


def _read_csv_text(text):
    reader = csv.DictReader(io.StringIO(text))
    return list(reader), reader.fieldnames or []


def from_local(path, goal_col=None, cat_col=None):
    text = Path(path).read_text()
    rows, fields = _read_csv_text(text)
    return convert(rows, fields)


def from_url(url):
    with urllib.request.urlopen(url, timeout=60) as resp:   # nosec - official benchmark repo
        text = resp.read().decode("utf-8", errors="replace")
    rows, fields = _read_csv_text(text)
    return convert(rows, fields)


def from_jbb():
    """JailbreakBench via the `datasets` library (its official distribution)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("JBB needs the datasets library: pip install datasets")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["harmful"]
    rows = [dict(r) for r in ds]
    return convert(rows, list(rows[0].keys()) if rows else [])


def from_hf(dataset_name, split="train"):
    """Any HuggingFace dataset -> {Goal, Category}, auto-detecting the goal column and
    deduplicating. For datasets that repeat a canonical behaviour across jailbreak variants
    (e.g. the *_jailbreak_responses_with_judgment sets), this yields the unique behaviours."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("--hf-dataset needs the datasets library: pip install datasets")
    ds = load_dataset(dataset_name)
    if split not in ds:
        split = list(ds.keys())[0]
    ds = ds[split]
    rows = [dict(r) for r in ds]
    return convert(rows, list(rows[0].keys()) if rows else [])


def write(out_path, records):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Goal", "Category"])
        w.writeheader()
        w.writerows(records)
    return len(records)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--benchmark", choices=list(_SOURCES) + ["jbb"],
                     help="fetch a named benchmark from its official repo")
    src.add_argument("--from-csv", help="convert a CSV you already downloaded")
    src.add_argument("--hf-dataset", help="any HuggingFace dataset name (auto-detect + dedup)")
    ap.add_argument("--out", required=True, help="output CSV (Goal, Category)")
    ap.add_argument("--goal-col", default=None, help="override goal-column autodetect")
    args = ap.parse_args(argv)

    if args.from_csv:
        recs, gcol, ccol = from_local(args.from_csv, goal_col=args.goal_col)
        origin = args.from_csv
    elif args.hf_dataset:
        recs, gcol, ccol = from_hf(args.hf_dataset)
        origin = f"hf:{args.hf_dataset}"
    elif args.benchmark == "jbb":
        recs, gcol, ccol = from_jbb()
        origin = "JailbreakBench/JBB-Behaviors"
    else:
        recs, gcol, ccol = from_url(_SOURCES[args.benchmark])
        origin = _SOURCES[args.benchmark]

    n = write(args.out, recs)
    cat_note = f"category<-'{ccol}'" if ccol else "no category column (set to 'unknown')"
    print(f"wrote {n} behaviours to {args.out}")
    print(f"  source : {origin}")
    print(f"  mapping: Goal<-'{gcol}', {cat_note}")
    print(f"  next   : recyclebench pilot-pminus --behaviours {args.out} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
