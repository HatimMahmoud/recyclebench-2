#!/usr/bin/env python3
"""tools/primary_calibration.py -- which SORRY-Bench judge calibration is primary (prereg Amendment 2 §B4.3).

Every corrected figure rests on the 60-bundle SORRY-Bench calibration, `audit/sorrybench/judge_sheet.csv`, which was
labelled with the judge's verdict visible. `judge_sheet_blind_relabel_20261030.csv` holds the same 280 rows for a blind
re-label. Registered rule (§B4.3): once every bundle of the blind re-label is labelled, the BLIND calibration is primary for
every corrected figure and the original is reported as sensitivity; until then the original stays primary.

This module is the one place that decides; consumers never pick a sheet or a corrected file themselves:
  status()        how far the blind re-label is, and which calibration is primary
  sheet()         context manager -> the sheet to calibrate from: the original, or the original rows with the blind labels
                  substituted (written to a private temporary folder beside a copy of the strata file)
  resolve(path)   a registered corrected file (*_recal_n60.json) -> itself, or its *_blind_n60.json twin once the blind
                  calibration is primary; raises if that twin is missing instead of falling back to the original
The twins are written by `apply`: each registered file is first reproduced from the original sheet (or nothing is
written), then recomputed from the same stored k / n with the blind calibration. Rogan-Gladen and `corrected_interval`
are pure functions of (k, n, t_mean, JudgeChars) and the calibration is bundle-level, so no replay is needed.
`tools/blind_relabel_score.py` keeps reading the registered files: it reports original against blind.

  python3 tools/primary_calibration.py status
  python3 tools/primary_calibration.py apply     # blind primary only; never overwrites
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import json
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

ORIGINAL = "audit/sorrybench/judge_sheet.csv"
BLIND = "audit/sorrybench/judge_sheet_blind_relabel_20261030.csv"
CORRECTED = {"anchor": "results/runs/recycle_3_31_sorrybench/corrected_sorrybench_recal_n60.json",
             "heldout": "results/runs/confirm_heldout_3_31/corrected_heldout_recal_n60.json"}
REG_SUFFIX, BLIND_SUFFIX = "_recal_n60.json", "_blind_n60.json"


def _rows(path: Path) -> list:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _blind_labels(root: Path, orig: list) -> dict:
    blind = {(r["bid"], r["variant_index"]): (r.get("human_fulfilled") or "").strip() for r in _rows(root / BLIND)}
    if len(blind) != len(orig) or set(blind) != {(r["bid"], r["variant_index"]) for r in orig}:
        raise ValueError(f"{BLIND} does not cover {ORIGINAL} row for row")
    return blind


def status(root=ROOT) -> dict:
    root = Path(root)
    if not (root / ORIGINAL).exists():
        return dict(primary="none", labelled=0, of=0, bundles_complete=0, bundles=0)
    if not (root / BLIND).exists():
        return dict(primary="original", labelled=0, of=0, bundles_complete=0, bundles=0)
    orig = _rows(root / ORIGINAL)
    blind = _blind_labels(root, orig)
    per = defaultdict(list)
    for r in orig:
        per[r["bid"]].append(blind[(r["bid"], r["variant_index"])] in ("0", "1"))
    done = sum(all(v) for v in per.values())
    return dict(primary="blind" if per and done == len(per) else "original",
                labelled=sum(v in ("0", "1") for v in blind.values()), of=len(blind), bundles_complete=done, bundles=len(per))


def label(root=ROOT) -> str:
    """one phrase naming the primary calibration, for page text"""
    return ("the blind re-label (primary under Amendment 2 §B4.3; the original, unblinded labels are the sensitivity analysis)"
            if status(root)["primary"] == "blind" else "the original sheet (labelled with the judge's verdict visible)")


@contextmanager
def sheet(root=ROOT):
    root = Path(root)
    if status(root)["primary"] != "blind":
        yield root / ORIGINAL
        return
    orig = _rows(root / ORIGINAL)
    blind = _blind_labels(root, orig)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "judge_sheet.csv"
        (Path(td) / "judge_strata.txt").write_text((root / ORIGINAL).with_name("judge_strata.txt").read_text())
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(orig[0]))
            w.writeheader()
            for r in orig:
                w.writerow({**r, "human_fulfilled": blind[(r["bid"], r["variant_index"])]})
        yield p


def resolve(path, root=ROOT) -> Path:
    """the primary version of a registered corrected file, in the form it was given (relative or absolute)"""
    p = Path(path)
    if not p.name.endswith(REG_SUFFIX) or status(root)["primary"] != "blind":
        return p
    twin = p.with_name(p.name[: -len(REG_SUFFIX)] + BLIND_SUFFIX)
    if not (twin if twin.is_absolute() else Path(root) / twin).exists():
        raise RuntimeError(f"the blind re-label is complete, so {twin.name} is primary but does not exist yet: "
                           "run `python3 tools/primary_calibration.py apply`")
    return twin


def apply(root=ROOT) -> list:
    """write the *_blind_n60.json twin of every registered corrected file (blind primary only; never overwrites)"""
    root = Path(root)
    if status(root)["primary"] != "blind":
        return []
    from recalibrate_corrected import KAPPA_FLOOR, PI_STAR, TOL, correct, strata, verdict
    from recyclebench.judge_audit import calibrate_from_human, ingest_judge_audit

    orig = calibrate_from_human(ingest_judge_audit(str(root / ORIGINAL)), *strata(root / ORIGINAL), KAPPA_FLOOR)
    with sheet(root) as sh:
        n_pos, n_neg = strata(sh)
        blind = calibrate_from_human(ingest_judge_audit(str(sh)), n_pos, n_neg, KAPPA_FLOOR)
    if not blind.bundle.admitted:
        raise SystemExit(f"the blind calibration is not admitted (kappa {float(blind.bundle.kappa):.3f} < {KAPPA_FLOOR}); "
                         "§B4.3 does not cover this case, so nothing is written -- decide it with the supervisor")
    jo, jb = orig.bundle.chars, blind.bundle.chars
    todo = []
    for rel in CORRECTED.values():
        src_p = root / rel
        dest = src_p.with_name(src_p.name[: -len(REG_SUFFIX)] + BLIND_SUFFIX)
        if dest.exists():
            raise SystemExit(f"{dest} exists -- refusing to overwrite")
        src = json.loads(src_p.read_text())
        r, p = src["RSR"], src["PBR_new"]
        k_r, n_r, t_r = r["k_recovered"], r["n_p_minus"], float(r.get("mean_admitted") or 1.0)
        k_p, n_p = p["k_bypassed"], p["n_resolved"]
        # 1. the original calibration must reproduce the registered file, or nothing is written
        got = correct(k_r, n_r, jo, t_r) + correct(k_p, n_p, jo, 1.0)
        want = (r["RSR_corrected"], *r["CI"], p["PBR_corrected"], *p["CI"])
        miss = [(round(a, 6), b) for a, b in zip(got, want) if abs(a - b) > TOL]
        if miss or orig.bundle.n != src["judge"]["n_bundles"]:
            raise SystemExit(f"cannot reproduce {rel} from {ORIGINAL}: n {orig.bundle.n} vs {src['judge']['n_bundles']}, {miss}")
        # 2. the same k / n under the blind calibration
        rr, pp = correct(k_r, n_r, jb, t_r), correct(k_p, n_p, jb, 1.0)
        out = copy.deepcopy(src)
        for block, kind, (pt, lo, hi) in ((out["RSR"], "RSR", rr), (out["PBR_new"], "PBR", pp)):
            block[f"{kind}_corrected"], block["CI"] = round(pt, 4), [round(lo, 4), round(hi, 4)]
            if "corrected" in block:
                block["corrected"] = round(pt, 4)
            if "corrected_ci95_delta" in block:
                block["corrected_ci95_delta"] = [round(lo, 4), round(hi, 4)]
        if "verdict" in out["RSR"]:
            out["RSR"]["verdict"] = verdict(rr[1], rr[2])
        P, mech = out["PBR_new"], verdict(pp[1], pp[2])
        if "verdict_reported" in P:
            # the registered file overrode a threshold-fragile mechanical verdict by hand; carry that reading over and
            # flag it, rather than let a recomputation decide a reporting question
            P["verdict_mechanical"] = mech
            P["verdict_warning"] = (f"Under the blind calibration the mechanical verdict is '{mech}' (95% CI [{pp[1]:.4f}, "
                                    f"{pp[2]:.4f}] against pi* = {PI_STAR}). The registered file reported this PBR as "
                                    f"'{p['verdict_reported']}' because its verdict was threshold-fragile; that reading is carried "
                                    "over unchanged and is to be re-read with both calibrations in the write-up (Amendment 2 §B4.3).")
            P["verdict"] = P["verdict_reported"]
        else:
            P["verdict"] = mech
        cal = dict(se=round(jb.se, 6), sp=round(jb.sp, 6), kappa=round(float(blind.bundle.kappa), 6),
                   admitted=bool(blind.bundle.admitted), n_bundles=blind.bundle.n,
                   se_ci=[blind.se_ci.low, blind.se_ci.high] if blind.se_ci else None,
                   sp_ci=[blind.sp_ci.low, blind.sp_ci.high] if blind.sp_ci else None,
                   strata=[n_pos, n_neg], response_level=bool(jb.response_level))
        out["judge"] = cal
        if "calibration" in out:
            out["calibration"] = dict(sheet=BLIND, rows_from=ORIGINAL, **cal,
                                      note="blind labels substituted into the original rows; design-weighted strata")
        out.pop("previous", None)                                   # described the pre-repair file; see sensitivity_original
        out.update(
            what=f"{src_p.name} recomputed with the BLIND re-label calibration ({blind.bundle.n} bundles, design-weighted "
                 f"strata {n_pos}/{n_neg}), primary under prereg Amendment 2 §B4.3.",
            supersedes=f"{src_p.name} (same run dir), which stays as the sensitivity analysis",
            why="The original calibration sheet was labelled with the judge's verdict visible. Amendment 2 §B4.3 registers "
                "that a complete blind re-label of the same 280 rows becomes primary for every corrected figure, with the "
                "original reported as sensitivity. Exact: the registered file is reproduced from the original sheet first, "
                "and the correction is a pure function of the stored k / n and the calibration.",
            generated=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            tool="tools/primary_calibration.py apply",
            sensitivity_original=dict(judge=src["judge"], RSR_corrected=r["RSR_corrected"], RSR_CI=r["CI"],
                                      PBR_corrected=p["PBR_corrected"], PBR_CI=p["CI"], PBR_verdict=p.get("verdict")))
        todo.append((dest, out))
    for dest, out in todo:                                          # every file checked before any is written
        dest.write_text(json.dumps(out, indent=2))
    return [d for d, _ in todo]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("status", "apply"))
    args = ap.parse_args(argv)
    st = status()
    print(f"blind re-label: {st['labelled']}/{st['of']} rows, {st['bundles_complete']}/{st['bundles']} bundles complete "
          f"-> primary calibration: {st['primary']}")
    if args.cmd == "apply":
        if st["primary"] != "blind":
            print("nothing to write: the original calibration stays primary until every bundle is labelled")
            return 0
        for d in apply():
            print(f"wrote {d.relative_to(ROOT) if d.is_relative_to(ROOT) else d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
