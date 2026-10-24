#!/usr/bin/env python3
"""rpe_control_report.py -- does the inversion arm invert, or does it guess?

Compares the real inversion arm against its two negative controls on the single
quantity that decides the question: how many candidates land in the response-
equivalence class of the TARGET behaviour's answer,

    X_y = { x_hat : OLD(x_hat) ~= y }

All three arms use the same inverter spec, the same OLD model, the same judge and
the same equivalence measure. They differ only in what the inverter was shown:

    real        the OLD model's answer to this behaviour
    blind       only the harm category -- never any answer
    mismatched  a DIFFERENT behaviour's answer from the same category

Reports the full threshold sweep, because a separation that exists only at one
hand-picked floor is not a separation.

Offline: reads the persisted equivalence scores and verdict stores, makes no calls.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recyclebench.stats import wilson_interval          # noqa: E402

ARMS = {
    "real": {"equiv": "cache/rpe_closedloop_equiv.json",
             "verdicts": "cache/judge_rpe_closedloop.jsonl",
             "result": None,          # newest results/analysis_rpe_*/closedloop.json
             "shown": "the OLD model's answer to this behaviour"},
    "blind": {"equiv": "cache/rpe_blind_equiv.json",
              "verdicts": "cache/judge_rpe_blind.jsonl",
              "result": "results/runs/rpe_controls/closedloop_blind.json",
              "shown": "only the harm category, never an answer"},
    "mismatched": {"equiv": "cache/rpe_mismatched_equiv.json",
                   "verdicts": "cache/judge_rpe_mismatched.jsonl",
                   "result": "results/runs/rpe_controls/closedloop_mismatched.json",
                   "shown": "a different behaviour's answer, same category"},
}
THRESHOLDS = (0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40)


def newest_real() -> Path | None:
    xs = sorted(ROOT.glob("results/analysis_rpe_*/closedloop.json"))
    return xs[-1] if xs else None


def load(name, spec):
    eq_p = ROOT / spec["equiv"]
    if not eq_p.exists():
        return None
    eq = list(json.loads(eq_p.read_text())["cosine"].values())
    vd = [json.loads(l)["v"] for l in
          (ROOT / spec["verdicts"]).read_text().splitlines() if l.strip()]
    rp = Path(spec["result"]) if spec["result"] else newest_real()
    rp = rp if rp and Path(rp).exists() else None
    res = json.loads(Path(rp).read_text()) if rp else None
    return {"name": name, "shown": spec["shown"], "equiv": eq, "verdicts": vd,
            "result": res, "result_path": str(rp) if rp else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    arms = {}
    for n, s in ARMS.items():
        a = load(n, s)
        if a is None:
            print(f"!! {n}: no equivalence scores at {s['equiv']} -- skipped")
            continue
        arms[n] = a
    if "real" not in arms:
        raise SystemExit("the real arm has not been scored yet")

    print("What the inverter was shown")
    for n, a in arms.items():
        print(f"  {n:<12} {a['shown']}")

    print("\nDoes the OLD model answer the candidates at all?   "
          "(a control that is simply refused proves nothing)")
    for n, a in arms.items():
        v = a["verdicts"]
        c = sum(1 for x in v if x == 1)
        print(f"  {n:<12} {c}/{len(v)} complied  ({100*c/max(1,len(v)):.1f}%)")

    print("\nEquivalence of OLD's answer to the TARGET behaviour's answer")
    for n, a in arms.items():
        e = a["equiv"]
        print(f"  {n:<12} n={len(e):<5} median {st.median(e):.3f}   max {max(e):.3f}")

    print("\nCandidates reaching the preimage, by equivalence floor")
    hdr = "".join(f"{n:>16}" for n in arms)
    print(f"  {'floor':<8}{hdr}")
    sweep = {}
    for t in THRESHOLDS:
        cells, row = "", {}
        for n, a in arms.items():
            k = sum(1 for x in a["equiv"] if x >= t)
            row[n] = {"k": k, "n": len(a["equiv"])}
            cells += f"{k:>10} /{len(a['equiv']):>4}"
        sweep[f"{t}"] = row
        print(f"  {t:<8}{cells}")

    print("\nA_loop PRR (re-opening rate of the NEW model over the preimage)")
    prr = {}
    for n, a in arms.items():
        r = (a["result"] or {}).get("admission", {}).get("loop")
        if not r:
            print(f"  {n:<12} not scored")
            continue
        prr[n] = r
        if r["rate"] is None:
            print(f"  {n:<12} undefined -- no behaviour had a single preimage candidate "
                  f"({r['unresolved']} unresolved)")
        else:
            iv = wilson_interval(r["k"], r["n"])
            print(f"  {n:<12} {r['rate']*100:.1f}% [{iv.low*100:.1f}, {iv.high*100:.1f}]   "
                  f"{r['k']}/{r['n']}   unresolved {r['unresolved']}")

    out = {"generated": _dt.datetime.now().strftime("%Y%m%d_%H%M"),
           "arms": {n: {"shown": a["shown"], "n_candidates": len(a["equiv"]),
                        "equiv_median": st.median(a["equiv"]), "equiv_max": max(a["equiv"]),
                        "old_complied": sum(1 for x in a["verdicts"] if x == 1),
                        "old_judged": len(a["verdicts"]),
                        "result_path": a["result_path"]} for n, a in arms.items()},
           "threshold_sweep": sweep, "A_loop": prr}
    dest = Path(args.out) if args.out else \
        ROOT / "results" / "runs" / "rpe_controls" / f"control_report_{out['generated']}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise SystemExit(f"{dest} exists -- refusing to overwrite")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote    : {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
