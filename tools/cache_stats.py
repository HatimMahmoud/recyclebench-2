"""Summarise / split a recyclebench response cache.

New cache lines carry {k, t, model, seed, prompt}. Old lines have only {k, t}.
This tool reports per-model counts (for new-format lines) and can split the cache
into one file per model.

  python3 cache_stats.py cache/llama_gen.jsonl              # summary
  python3 cache_stats.py cache/llama_gen.jsonl --split      # -> cache/by_model/<model>.jsonl
  python3 cache_stats.py cache/llama_gen.jsonl --model llama3.1:8b --grep "bleach"
"""
import argparse, json, sys
from pathlib import Path
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--split", action="store_true", help="write one file per model")
    ap.add_argument("--model", help="only show/count lines for this model")
    ap.add_argument("--grep", help="only lines whose prompt snippet contains this text")
    args = ap.parse_args()

    per_model = Counter()
    legacy = 0
    rows = []
    for line in Path(args.cache).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        m = d.get("model")
        if m is None:
            legacy += 1
            continue
        per_model[m] += 1
        if args.model and m != args.model:
            continue
        if args.grep and args.grep.lower() not in (d.get("prompt","")).lower():
            continue
        rows.append(d)

    print(f"cache: {args.cache}")
    print(f"  total lines with model tag: {sum(per_model.values())}")
    print(f"  legacy lines (old {{k,t}}, no model): {legacy}")
    print("  per-model:")
    for m, n in per_model.most_common():
        print(f"    {m:<16} {n}")

    if args.model or args.grep:
        print(f"\n  matched {len(rows)} lines"
              + (f" for model={args.model}" if args.model else "")
              + (f" grep={args.grep!r}" if args.grep else ""))
        for d in rows[:5]:
            print(f"    [{d.get('model')}] seed={d.get('seed')}  prompt={d.get('prompt','')[:70]!r}")
            print(f"       -> {d.get('t','')[:100]!r}")

    if args.split:
        outdir = Path(args.cache).parent / "by_model"
        outdir.mkdir(exist_ok=True)
        handles = {}
        n = 0
        for line in Path(args.cache).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            m = d.get("model", "_legacy_no_model")
            safe = m.replace("/", "-").replace(":", "-")
            if safe not in handles:
                handles[safe] = (outdir / f"{safe}.jsonl").open("w")
            handles[safe].write(line + "\n"); n += 1
        for h in handles.values():
            h.close()
        print(f"\n  split {n} lines -> {outdir}/  ({len(handles)} files)")

if __name__ == "__main__":
    main()
