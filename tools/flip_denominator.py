#!/usr/bin/env python3
"""tools/flip_denominator.py -- switch the reported RSR denominator, verify it, rebuild everything.

Amendment 2 §B3 registers the RSR denominator as **re-opened / resolved**: a behaviour with no
gate-admitted rewording is `unresolved` and leaves the denominator, rather than counting against
recovery. The code supports both (`compute_rsr(..., denominator=...)`) and every dashboard and the
evidence layer read ONE switch, `lineage_core.RSR_DENOMINATOR`, which still says "p_minus" until the
amendment is signed.

Flipping it by hand is a one-line edit feeding eight call sites across `study_data`, both dashboards
and `pair_dashboard`, with nothing checking that every reported figure moved together -- so a partial
rebuild leaves a page quoting one convention beside a page quoting the other, and nothing complains.
This does the flip as one reversible, verified step:

  python3 tools/flip_denominator.py --show              # current switch + what each convention gives
  python3 tools/flip_denominator.py --to resolved       # flip, verify, then print the rebuild plan
  python3 tools/flip_denominator.py --to resolved --rebuild    # ... and run the rebuild
  python3 tools/flip_denominator.py --to p_minus        # revert

Guard: it refuses to move to "resolved" while only `prereg_amendment_2_DRAFT.md` exists, because the
convention is registered-but-unsigned until the file is signed (Amendment 1 set the precedent:
`prereg_amendment_1_SIGNED.md`). `--force` overrides, and says so in its output.

No model calls. The verification recomputes the anchor under both conventions from the cache and the
stored verdicts and checks the flipped value against the generated thesis figure, so a silent drift
in either direction fails loudly instead of being published.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

SWITCH = ROOT / "tools/lineage_core.py"
PATTERN = re.compile(r'^(RSR_DENOMINATOR\s*=\s*)"(p_minus|resolved)"', re.M)
VALID = ("p_minus", "resolved")
ANCHOR = "results/runs/recycle_3_31_sorrybench/corrected_sorrybench_recal_n60.json"
# Each convention's own generated figure. The k/n macro shapes differ -- "AnchorResolvedKn" is
# already "23/46" while the p_minus convention stores k and n separately -- so the expected k/n is
# assembled per convention rather than read from one macro.
EXPECT = {"resolved": (("AnchorResolvedKn",), "AnchorResolvedRSRcorr"),
          "p_minus": (("AnchorK", "AnchorPminus"), "AnchorRSRcorr")}


def current() -> str:
    m = PATTERN.search(SWITCH.read_text())
    if not m:
        raise SystemExit(f"could not find RSR_DENOMINATOR in {SWITCH.relative_to(ROOT)}")
    return m.group(2)


def set_to(value: str) -> None:
    s = SWITCH.read_text()
    new, n = PATTERN.subn(lambda m: f'{m.group(1)}"{value}"', s, count=1)
    if n != 1:
        raise SystemExit("refusing to write: the switch did not match exactly once")
    SWITCH.write_text(new)


def call_sites() -> list[str]:
    out = []
    for p in sorted(ROOT.glob("tools/*.py")) + sorted(ROOT.glob("src/recyclebench/*.py")):
        if p in (SWITCH, Path(__file__).resolve()):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "RSR_DENOMINATOR" in line:
                out.append(f"{p.relative_to(ROOT)}:{i}")
    return out


def anchor_both() -> dict:
    """The anchor's RSR under both conventions.

    `k` comes from the stored, calibrated anchor result and is NOT recomputed: the anchor run's frozen
    manifest was re-screened in its own run, and the lineage verdict store holds complete variant
    verdicts for only 2 of its 47 fixes, so a rebuild reproduces P^- exactly but not recovery (it
    reads k = 4). `tools/thesis_numbers.py` takes the same route for the same reason. What the flip
    changes is the DENOMINATOR, so that is what is derived here -- the count of fixes with no admitted
    rewording, rebuilt from the cache -- and the corrected rate is recomputed for both.
    """
    import primary_calibration as pc
    import study_data as sd
    import thesis_numbers as tn
    from recyclebench.stats import corrected_interval, wilson_interval
    a = json.loads((ROOT / str(pc.resolve(ANCHOR))).read_text())
    r = a["RSR"]
    k, n_pm = r["k_recovered"], r["n_p_minus"]
    unres = len(tn.anchor_unresolved())            # asserts the rebuild still reproduces P^- = 47
    jc = sd.anchor_judge_chars(sd.judge_calibrations())
    out = {"unresolved": unres, "n_p_minus": n_pm, "k": k}
    for d, n in (("p_minus", n_pm), ("resolved", n_pm - unres)):
        cr, wr = corrected_interval(k, n, jc), wilson_interval(k, n)
        out[d] = {"k": k, "n": n, "raw": wr.point, "corrected": cr.point,
                  "ci": (cr.low, cr.high)}
    # the stored file is the p_minus convention: it must agree, or one of the two is stale
    if abs(out["p_minus"]["corrected"] - r["RSR_corrected"]) > 5e-4:
        raise SystemExit(f"recomputed p_minus corrected {out['p_minus']['corrected']:.4f} disagrees "
                         f"with the stored {r['RSR_corrected']:.4f} -- investigate before flipping")
    return out


def thesis_value(name):
    f = sorted(glob.glob(str(ROOT / "results/thesis_numbers_*.json")))
    if not f:
        return None
    return json.loads(Path(f[-1]).read_text())["values"].get(name)


REBUILD = [
    ("thesis figures", ["tools/thesis_numbers.py"]),
    ("C4 table (naive vs corrected)", ["tools/naive_vs_corrected.py"]),
    ("exploratory analysis", ["tools/thesis_analysis.py"]),
    ("master dashboard", ["tools/master_dashboard.py"]),
    ("lineage dashboard", ["tools/lineage_dashboard.py"]),
    ("pair dashboards", ["tools/pair_dashboard.py"]),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", choices=VALID, help="the denominator to report")
    ap.add_argument("--show", action="store_true", help="print the state and both figures, change nothing")
    ap.add_argument("--rebuild", action="store_true", help="run the rebuild after a successful flip")
    ap.add_argument("--force", action="store_true", help="flip to 'resolved' before Amendment 2 is signed")
    args = ap.parse_args(argv)
    if not args.to and not args.show:
        ap.error("give --to {p_minus,resolved} or --show")

    now = current()
    signed = (ROOT / "prereg_amendment_2_SIGNED.md").exists()
    print(f"switch      : lineage_core.RSR_DENOMINATOR = {now!r}")
    print(f"read by     : {len(call_sites())} call sites")
    for c in call_sites():
        print(f"              {c}")
    print(f"Amendment 2 : {'SIGNED' if signed else 'DRAFT only (convention registered, not signed)'}")

    a = anchor_both()
    print(f"\nanchor (P^- {a['n_p_minus']}, k {a['k']} from the stored calibrated result; "
          f"unresolved {a['unresolved']} rebuilt from the cache):")
    for d in VALID:
        r = a[d]
        mark = "  <- current" if d == now else ""
        print(f"  {d:9} {r['k']}/{r['n']}  raw {r['raw']:.3f}  corrected {r['corrected']:.3f}{mark}")

    if args.show or args.to == now:
        if args.to == now:
            print(f"\nalready {now!r}; nothing to do")
        return 0

    if args.to == "resolved" and not signed and not args.force:
        print("\nREFUSED: Amendment 2 is a draft, so 'resolved' is registered but not yet signed.")
        print("Sign it (prereg_amendment_2_SIGNED.md, as Amendment 1 was) or pass --force.")
        return 1
    if args.to == "resolved" and not signed and args.force:
        print("\nWARNING: --force used while Amendment 2 is unsigned. Every figure rebuilt from here")
        print("         reports a convention that is not yet signed off. Say so wherever it is quoted.")

    kn_macros, exp_corr = EXPECT[args.to]
    parts = [thesis_value(m) for m in kn_macros]
    want_kn = None if any(x is None for x in parts) else "/".join(parts)
    want_corr = thesis_value(exp_corr)
    got = a[args.to]
    got_kn = f"{got['k']}/{got['n']}"
    got_corr = f"{got['corrected']:.3f}"
    print(f"\nverifying against the generated thesis figures "
          f"({', '.join(kn_macros)}, {exp_corr}):")
    print(f"  recomputed {got_kn} corrected {got_corr}   generated {want_kn} corrected {want_corr}")
    if want_kn is not None and (got_kn != want_kn or got_corr != want_corr):
        print("MISMATCH -- the recomputed figure and the generated one disagree. Not flipping.")
        print("Re-run tools/thesis_numbers.py and investigate before changing the convention.")
        return 1

    set_to(args.to)
    print(f"\nflipped: {now!r} -> {args.to!r} in {SWITCH.relative_to(ROOT)}")
    if current() != args.to:
        raise SystemExit("write-back verification failed")

    if not args.rebuild:
        print("\nrebuild (dashboards and figures still report the old convention until these run):")
        for what, cmd in REBUILD:
            print(f"  python3 {' '.join(cmd)}       # {what}")
        print("\nre-run with --rebuild to execute, or --to "
              f"{now} to revert.")
        return 0

    print("\nrebuilding:")
    for i, (what, cmd) in enumerate(REBUILD, 1):
        print(f"\n=== {i}/{len(REBUILD)} {what}", flush=True)
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        if r.returncode != 0:
            print(f"FAILED at {what} (rc={r.returncode}). The switch is already {args.to!r}; "
                  f"fix and re-run the remaining steps, or revert with --to {now}.")
            return 1
    print(f"\ndone. Every figure now reports the {args.to!r} denominator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
