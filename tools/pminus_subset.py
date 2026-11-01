#!/usr/bin/env python3
"""tools/pminus_subset.py -- write the JUDGE-DECIDED P^- subset CSV from a pilot-pminus result.

Why this exists
---------------
``tools/pminus_report.py --dump`` writes its cells using the **offline refusal rule**
(``looks_like_refusal``), because it rebuilds the partition from the response cache and
``run-real`` never stores judge verdicts. On SORRY-Bench that rule disagrees with the
``qwen2.5:14b`` rubric judge (offline P^- 14/2/61/19/33 vs judged 18/3/49/12/13). Feeding the
dumped cell into a recycle run therefore recycles the *wrong* behaviours.

``pilot-pminus`` does use the config's judge, and its JSON carries ``p_minus_bids``. This maps
those bids back to the source CSV and writes the subset, so the recycle arm runs on exactly the
fixes the judge decided.

Caveat it cannot fix: ``load_jailbreakbench`` ignores a ``bid`` column and renumbers rows, so
the ids inside a run built on this subset are **row positions in the subset**, not ids in the
source corpus. Join such a run back to the corpus **on prompt text**. The ``bid`` column here is
provenance for humans, and ``source_bid`` is kept explicitly to make that join checkable.

  python3 tools/pminus_subset.py --pilot <run>/pminus_sciencesafety.json \\
      --behaviours data/sciencesafety.csv --out <run>/sciencesafety_p_minus_judged.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recyclebench.datasets import load_jailbreakbench            # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", required=True, help="pminus_pilot.json from pilot-pminus")
    ap.add_argument("--behaviours", required=True, help="the source behaviour CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default="p_minus_bids",
                    help="which bid list to extract (default: p_minus_bids)")
    args = ap.parse_args(argv)

    pilot = json.loads(Path(args.pilot).read_text())
    bids = list(pilot.get(args.cell) or [])
    behaviours = {b.bid: b for b in load_jailbreakbench(args.behaviours)}

    missing = [b for b in bids if b not in behaviours]
    if missing:
        raise SystemExit(f"{len(missing)} bid(s) in {args.cell} are absent from "
                         f"{args.behaviours} (wrong CSV for this pilot?): {missing[:5]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bid", "source_bid", "Category", "Goal"])
        for bid in bids:                       # pilot order = source order
            b = behaviours[bid]
            w.writerow([b.bid, b.bid, b.category, b.canonical_prompt])

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"pair      : {pilot.get('pair')}")
    print(f"judge     : the config's judge, as used by pilot-pminus (NOT the offline rule)")
    print(f"cell      : {args.cell}  ({len(bids)} behaviours)")
    print(f"of        : {pilot.get('n_behaviours')} screened, breaks_old="
          f"{pilot.get('breaks_old')}, status={pilot.get('status')}")
    print(f"written   : {out}  ({sha[:16]})")
    if not bids:
        print("\nP^- is EMPTY: nothing to recycle. RSR is undefined for this pair "
              "(never 0); report PBR and the 2x2 only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
