#!/usr/bin/env python3
"""audit_progress.py -- how far through an audit sheet, and how much of it is usable.

Bundle-level calibration takes the OR over a `bid` group, so a partially labelled
group is DROPPED. Raw row progress therefore overstates what you can actually use:
stopping mid-bundle wastes every row of that bundle.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

LABEL_COLS = ("human_fulfilled", "human_intent_preserved", "human_same_request")


def main(path: str) -> int:
    import os
    if not os.path.exists(path):
        print(f"no sheet at {path} -- it has not been exported yet.")
        print("Real runs export with tools/judge_export_real.py; the built-in "
              "`judge-export` is the mock twin.")
        return 1
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print("empty sheet")
        return 1
    col = next((c for c in rows[0] if c in LABEL_COLS), None)
    if not col:
        print(f"no label column in {list(rows[0])}")
        return 1
    done = sum(1 for r in rows if (r.get(col) or "").strip())
    print(f"rows      : {done}/{len(rows)} labelled")

    if "bid" not in rows[0]:
        return 0
    groups = defaultdict(list)
    for r in rows:
        groups[r["bid"]].append((r.get(col) or "").strip())
    full = [b for b, v in groups.items() if all(x for x in v)]
    part = [b for b, v in groups.items() if any(x for x in v) and not all(x for x in v)]
    print(f"bundles   : {len(full)}/{len(groups)} complete and usable")
    if part:
        wasted = sum(1 for b in part for x in groups[b] if x)
        print(f"PARTIAL   : {len(part)} bundle(s) started but unfinished -- "
              f"{wasted} labelled row(s) currently unusable")
        print(f"            finish these bids: {', '.join(sorted(part)[:6])}"
              + (" …" if len(part) > 6 else ""))
    if full:
        ones = sum(1 for b in full if any(x == "1" for x in groups[b]))
        print(f"bundle OR : {ones}/{len(full)} fulfilled")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "audit/confirm/judge_sheet.csv"))
