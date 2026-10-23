#!/usr/bin/env python3
"""tools/prepare_heldout.py -- build the held-out confirmatory behaviour set (prereg amendment 2026-10-22).

JBB-Behaviors and GuidedBench were never screened on any model. Some of their behaviours repeat ones that
were (JBB reuses AdvBench/HarmBench items), and those are not held out: their outcomes were already seen. This
keeps only behaviours whose normalised text is neither identical to nor a close match (difflib ratio >= --cutoff)
of any behaviour in the already-screened benchmarks, and drops cross-duplicates between the two sources.

Deterministic: same inputs -> same file. Writes the CSV (Goal, Category, Source) and a JSON audit of every drop.

  python3 tools/prepare_heldout.py --out data/heldout_jbb_guided.csv
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

from recyclebench.datasets import load_jailbreakbench

SCREENED = ["advbench", "harmbench", "strongreject", "sorrybench", "llama32jb", "wrapped_rs", "llama_wrapped",
            "harmbench_pminus_3_31"]
SOURCES = ["jbb", "guidedbench"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/heldout_jbb_guided.csv")
    ap.add_argument("--cutoff", type=float, default=0.85)
    args = ap.parse_args(argv)
    d = Path(args.data)
    seen = {}
    for f in SCREENED:
        p = d / f"{f}.csv"
        if p.exists():
            for b in load_jailbreakbench(str(p)):
                seen.setdefault(norm(b.canonical_prompt), f)
    seen_list = list(seen)
    keep, drops, kept_norm = [], [], set()
    for src in SOURCES:
        for b in load_jailbreakbench(str(d / f"{src}.csv")):
            n = norm(b.canonical_prompt)
            if n in seen:
                drops.append(dict(source=src, reason=f"exact duplicate of {seen[n]}", goal_sha=hashlib.sha256(n.encode()).hexdigest()[:12]))
                continue
            m = difflib.get_close_matches(n, seen_list, n=1, cutoff=args.cutoff)
            if m:
                r = difflib.SequenceMatcher(None, n, m[0]).ratio()
                drops.append(dict(source=src, reason=f"near duplicate ({r:.2f}) of {seen[m[0]]}",
                                  goal_sha=hashlib.sha256(n.encode()).hexdigest()[:12]))
                continue
            if n in kept_norm:
                drops.append(dict(source=src, reason="duplicate across held-out sources",
                                  goal_sha=hashlib.sha256(n.encode()).hexdigest()[:12]))
                continue
            kept_norm.add(n)
            keep.append((b.canonical_prompt, b.category, src))
    out = Path(args.out)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Goal", "Category", "Source"])
        w.writerows(keep)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    audit = dict(out=str(out), sha256=digest, n_kept=len(keep), n_dropped=len(drops), cutoff=args.cutoff,
                 kept_by_source={s: sum(1 for k in keep if k[2] == s) for s in SOURCES}, drops=drops)
    out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=1))
    print(f"kept {len(keep)} ({audit['kept_by_source']}), dropped {len(drops)} -> {out}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
