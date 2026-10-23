#!/usr/bin/env python3
"""tools/cache_consolidate.py -- append responses that live only in side caches into the shared cache.

Earlier runs wrote some models' responses to their own cache files (llama2-13b -> cache/llama2_13b.jsonl,
llama3.2 1B/3B on HarmBench/AdvBench/StrongREJECT -> cache/ollama_pilot.jsonl). Jobs that combine those models
with cache/llama_gen.jsonl need everything in one file. This copies every record whose key the target lacks,
verbatim, to the end of the target. It never rewrites or reorders existing lines, and never adds a key the
target already has (a key present in both with different text keeps the target's text and is reported).

Keys are sha256(model|prompt|seed), so appending a missing key cannot change any lookup an existing run makes.

  python3 tools/cache_consolidate.py                      # dry run: what would be added
  python3 tools/cache_consolidate.py --apply              # back up the target, then append
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def read_records(path: Path):
    bad = 0
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return out, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="cache/llama_gen.jsonl")
    ap.add_argument("--sources", nargs="+", default=["cache/llama2_13b.jsonl", "cache/ollama_pilot.jsonl"])
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    target = Path(args.target)
    recs, bad = read_records(target)
    have = {r["k"]: r["t"] for r in recs}
    print(f"target  {target}: {len(have):,} keys" + (f" ({bad} malformed lines ignored)" if bad else ""))
    add, conflicts, per_model = [], Counter(), Counter()
    seen = set()
    for src in args.sources:
        srecs, sbad = read_records(Path(src))
        n_add = 0
        for r in srecs:
            k = r["k"]
            if k in have:
                if have[k] != r["t"]:
                    conflicts[src] += 1
                continue
            if k in seen:
                continue
            seen.add(k)
            add.append(r)
            n_add += 1
            per_model[r.get("model", "(no model field)")] += 1
        print(f"source  {src}: {len(srecs):,} records, {n_add:,} new to the target, "
              f"{conflicts[src]} present in both with different text (target kept)" + (f", {sbad} malformed" if sbad else ""))
    print(f"to add  {len(add):,} records: " + ", ".join(f"{m} {c:,}" for m, c in per_model.most_common()))
    if not args.apply:
        print("dry run: nothing written (pass --apply)")
        return 0
    if not add:
        print("nothing to add")
        return 0
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(f"{target.name}.pre_consolidate_{stamp}")
    shutil.copy2(target, backup)
    print(f"backup  {backup}")
    raw = target.read_bytes()
    with target.open("a") as fh:
        if raw and not raw.endswith(b"\n"):
            fh.write("\n")
        for r in add:
            fh.write(json.dumps(r) + "\n")
    after, _ = read_records(target)
    keys_after = {r["k"] for r in after}
    ok = len(keys_after) == len(have) + len(add) and all(r["k"] in keys_after for r in add)
    print(f"after   {target}: {len(keys_after):,} keys -> {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
