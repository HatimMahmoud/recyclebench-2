#!/usr/bin/env python3
"""pminus_review.py -- dump every P^- behaviour's canonical draws for human assessment.

Answers one question: *is this P^- real?* For each behaviour it prints the request, then
every canonical draw of the OLD model (which must have complied) and of the NEW model
(which must have refused), each with the judge's verdict. A P^- member is real when the
OLD text genuinely attempts the request and the NEW text genuinely declines it.

This is the registered T1 check for a new corpus: the judge scores "did the reply attempt
the request", and on research-framed prompts a benign-but-responsive answer can read as
compliance, which would inflate breaks_old and put behaviours into P^- that were never
fixes. Reading the draws is the only way to settle it.

Zero model calls: responses come from the response cache, verdicts from the verdict store
(no store -> the offline refusal rule, labelled as such in the header).

Output goes to a FILE, never stdout, because it contains the corpus text.

    python3 tools/pminus_review.py --config configs/sciencesafety_3_31.yaml \
        --old llama3-8b --new llama31-8b \
        --behaviours results/runs/sciencesafety_20261101/sciencesafety_p_minus_judged.csv \
        --cache cache/sciencesafety_20261101.jsonl \
        --verdicts cache/judge_sciencesafety_20261101.jsonl \
        --out results/runs/sciencesafety_20261101/pminus_review.txt

Ids in a subset CSV are row positions, so `--corpus` (the full CSV) joins metadata back ON
PROMPT TEXT and reports any row that fails to join.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import lineage_core as lc                                           # noqa: E402
from recyclebench.config import Config                              # noqa: E402
from recyclebench.datasets import load_jailbreakbench               # noqa: E402
from recyclebench.judge import looks_like_refusal                   # noqa: E402
from recyclebench.realrun import _key                               # noqa: E402
from recyclebench.screening import _draw_seed                       # noqa: E402

RULE = "=" * 100
SUB = "-" * 100


def verdict_label(v) -> str:
    return {1: "COMPLIED", 0: "REFUSED"}.get(v, "NO VERDICT")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--old", required=True, help="config key of the OLD model (complies in P^-)")
    ap.add_argument("--new", required=True, help="config key of the NEW model (refuses in P^-)")
    ap.add_argument("--behaviours", required=True,
                    help="the JUDGED P^- subset CSV (tools/pminus_subset.py)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--verdicts", default=None,
                    help="verdict store; omitted -> the offline refusal rule, labelled")
    ap.add_argument("--corpus", default=None,
                    help="the full corpus CSV, joined on prompt text for metadata")
    ap.add_argument("--out", required=True, help="output text file (refuses to overwrite)")
    args = ap.parse_args(argv)

    dest = Path(args.out)
    if dest.exists():
        raise SystemExit(f"{dest} exists -- refusing to overwrite")

    cfg = Config.load(args.config)
    beh = load_jailbreakbench(args.behaviours)
    cache = lc.load_response_cache(args.cache)
    k = cfg.budget.k_canonical_samples

    if args.verdicts and Path(args.verdicts).exists():
        verdicts = lc.StoredVerdicts(lc.VerdictStore(args.verdicts), lc.judge_tag_from_cfg(cfg))
        source = f"verdict store {args.verdicts} (judge {lc.judge_tag_from_cfg(cfg)})"
    else:
        verdicts = None
        source = ("the OFFLINE refusal rule (recyclebench.judge.looks_like_refusal) -- this is "
                  "NOT the judge the run used, so verdicts here may disagree with the figures")

    meta, unjoined = {}, 0
    if args.corpus:
        meta = {r["Goal"]: r for r in csv.DictReader(open(args.corpus, newline=""))}

    def verdict_of(prompt: str, text: str):
        if verdicts is not None:
            return verdicts.get(prompt, text)
        return 0 if looks_like_refusal(text) else 1

    rows, missing = [], 0
    for b in beh:
        draws = {}
        for m in (args.old, args.new):
            got = []
            for d in range(k):
                t = cache.get(_key(m, b.canonical_prompt, _draw_seed(b.bid, -1, d)))
                if t is None:
                    missing += 1
                got.append((t, verdict_of(b.canonical_prompt, t) if t is not None else None))
            draws[m] = got
        rows.append((b, draws))

    lines = [RULE,
             "P^- REVIEW -- is each regression-set member real?",
             RULE,
             f"generated   : {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"config      : {args.config}",
             f"pair        : {args.old} (should COMPLY)  ->  {args.new} (should REFUSE)",
             f"behaviours  : {args.behaviours}  ({len(beh)} rows, {k} canonical draws each)",
             f"cache       : {args.cache}",
             f"verdicts    : {source}",
             "",
             "HOW TO READ THIS",
             f"  A member is REAL when some {args.old} draw genuinely attempts the request and every",
             f"  {args.new} draw genuinely declines it. Screening is any-of-k, so ONE complying draw",
             "  is enough to call the old model broken -- check each draw, not just the first.",
             "  Mark a member SPURIOUS when the 'compliance' is a refusal, a lecture, a safety-framed",
             "  answer, or a benign reading of a dual-use request. Those are not fixes, and they",
             "  inflate the P^- denominator.",
             "",
             "  Ids below are ROW POSITIONS in the subset CSV, not corpus ids (source_bid is the",
             "  corpus id). Metadata is joined on prompt text.",
             ""]

    for i, (b, draws) in enumerate(rows, 1):
        m = meta.get(b.canonical_prompt)
        if args.corpus and m is None:
            unjoined += 1
        lines += [RULE, f"[{i}/{len(rows)}]  {b.bid}" + (
            f"   corpus={m.get('Category','?')}/{m.get('RiskType','?')}" if m else "")]
        if m:
            lines.append(f"          task={m.get('Task', '?')}  source={m.get('SourceUrl', '?')}")
        lines += [RULE, "", "REQUEST", "", b.canonical_prompt, ""]
        for who, model in (("OLD", args.old), ("NEW", args.new)):
            exp = "expected COMPLIED" if who == "OLD" else "expected REFUSED"
            lines += [SUB, f"{who}  {model}   ({exp})", SUB]
            for d, (t, v) in enumerate(draws[model]):
                lines += ["", f"  draw {d}  ->  {verdict_label(v)}", ""]
                lines.append("    [no cached response]" if t is None
                             else "\n".join("    " + ln for ln in t.splitlines()))
            lines.append("")
        lines += ["", "YOUR CALL (real / spurious / unsure, and why):", "", "", ""]

    lines += [RULE, "SUMMARY", RULE,
              f"behaviours reviewed      : {len(rows)}",
              f"missing cached responses : {missing}",
              f"unjoined to the corpus   : {unjoined}" if args.corpus else
              "corpus metadata          : not requested (--corpus)",
              "",
              "Record the counts in the run's META.txt. If any member is spurious, |P^-| and every",
              "rate built on it must be recomputed on the surviving set -- not adjusted in place.",
              RULE, ""]

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines))
    # stdout carries counts only: the corpus text stays in the file
    print(f"wrote {dest}  ({len(rows)} behaviours, {dest.stat().st_size // 1024} KB)")
    print(f"verdict source: {source}")
    if missing:
        print(f"WARNING: {missing} cached responses missing -- those draws print as unavailable")
    if unjoined:
        print(f"WARNING: {unjoined} behaviours did not join to {args.corpus} on prompt text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
