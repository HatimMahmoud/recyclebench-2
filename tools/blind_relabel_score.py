#!/usr/bin/env python3
"""tools/blind_relabel_score.py -- score the BLIND re-label of the SORRY-Bench judge-calibration sheet.

The 60-bundle calibration behind every corrected figure (`audit/sorrybench/judge_sheet.csv`) was labelled with the judge's
verdict visible. `judge_sheet_blind_relabel_20261030.csv` holds the same 280 rows, shuffled, without the judge column. This
tool joins the two on (bid, variant_index) and reports, as registered in prereg Amendment 2 §B2.3:

  anchoring   agreement and Cohen's κ between the original and the blind labels (row and bundle level), and how often
              each label set agrees with the judge -- if the original labels side with the judge more often than the
              blind ones do, the unblinded labelling was anchored
  calibration the judge's bundle-level se / sp / κ recomputed from the BLIND labels, design-weighted with the sheet's
              population strata (`judge_strata.txt`), through the same `calibrate_from_human` as every other calibration
  headline    the anchor's and the confirmatory replication's corrected RSR and PBR recomputed with that calibration from
              their stored k / n / t_mean (pure arithmetic, no replay), beside the registered values, with each verdict

Partial labelling is allowed and reported; the calibration is computed only over complete bundles.

  python3 tools/blind_relabel_score.py        # -> results/analysis_blind_relabel_<stamp>/blind_relabel.{json,txt}
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from recalibrate_corrected import KAPPA_FLOOR, correct, strata, verdict   # noqa: E402
from recyclebench.judge_audit import calibrate_from_human, ingest_judge_audit   # noqa: E402

SHEET = ROOT / "audit/sorrybench/judge_sheet.csv"
BLIND = ROOT / "audit/sorrybench/judge_sheet_blind_relabel_20261030.csv"
TARGETS = {"anchor": ROOT / "results/runs/recycle_3_31_sorrybench/corrected_sorrybench_recal_n60.json",
           "confirmatory": ROOT / "results/runs/confirm_heldout_3_31/corrected_heldout_recal_n60.json"}


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def load():
    orig = list(csv.DictReader(SHEET.open(newline="")))
    blind = {(r["bid"], r["variant_index"]): (r.get("human_fulfilled") or "").strip()
             for r in csv.DictReader(BLIND.open(newline=""))}
    if len(blind) != len(orig) or set(blind) != {(r["bid"], r["variant_index"]) for r in orig}:
        raise SystemExit("blind sheet does not cover the calibration sheet row for row")
    return orig, blind


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    orig, blind = load()
    rows = [dict(bid=r["bid"], o=int(r["human_fulfilled"]), b=int(blind[(r["bid"], r["variant_index"])]),
                 j=int(r["judge_label"])) for r in orig if blind[(r["bid"], r["variant_index"])] in ("0", "1")]
    res = dict(created=_dt.datetime.now().isoformat(timespec="seconds"), labelled=len(rows), of=len(orig))
    if rows:
        res["row"] = dict(agreement=round(sum(r["o"] == r["b"] for r in rows) / len(rows), 4),
                          kappa=round(kappa([r["o"] for r in rows], [r["b"] for r in rows]), 4),
                          original_agrees_with_judge=round(sum(r["o"] == r["j"] for r in rows) / len(rows), 4),
                          blind_agrees_with_judge=round(sum(r["b"] == r["j"] for r in rows) / len(rows), 4),
                          flips_original_to_blind={"1->0": sum(r["o"] == 1 and r["b"] == 0 for r in rows),
                                                   "0->1": sum(r["o"] == 0 and r["b"] == 1 for r in rows)})
        per = defaultdict(list)
        for r in orig:
            per[r["bid"]].append(blind[(r["bid"], r["variant_index"])])
        complete = {b for b, v in per.items() if all(x in ("0", "1") for x in v)}
        bo, bb = defaultdict(int), defaultdict(int)
        for r in rows:
            if r["bid"] in complete:
                bo[r["bid"]] |= r["o"]; bb[r["bid"]] |= r["b"]
        ks = sorted(complete)
        res["bundle"] = dict(complete=len(ks), of=len(per),
                             agreement=round(sum(bo[k] == bb[k] for k in ks) / len(ks), 4) if ks else None,
                             kappa=round(kappa([bo[k] for k in ks], [bb[k] for k in ks]), 4) if ks and kappa([bo[k] for k in ks], [bb[k] for k in ks]) is not None else None)
    # calibration from the blind labels (complete bundles only; ingest drops the others)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "judge_sheet.csv"
        (Path(td) / "judge_strata.txt").write_text((SHEET.parent / "judge_strata.txt").read_text())
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(orig[0]))
            w.writeheader()
            for r in orig:
                w.writerow({**r, "human_fulfilled": blind[(r["bid"], r["variant_index"])]})
        n_pos, n_neg = strata(tmp)
        try:
            cal = calibrate_from_human(ingest_judge_audit(str(tmp)), n_pos, n_neg, KAPPA_FLOOR)
        except Exception as e:                                        # noqa: BLE001
            cal = None
            res["calibration"] = dict(error=f"{type(e).__name__}: {e}")
    if cal is not None:
        jc = cal.bundle.chars
        res["calibration"] = dict(bundles=cal.bundle.n, se=round(jc.se, 4), sp=round(jc.sp, 4),
                                  kappa=round(float(cal.bundle.kappa), 4), admitted=bool(cal.bundle.admitted),
                                  se_ci=[round(cal.se_ci.low, 4), round(cal.se_ci.high, 4)] if cal.se_ci else None,
                                  sp_ci=[round(cal.sp_ci.low, 4), round(cal.sp_ci.high, 4)] if cal.sp_ci else None)
        res["headline"] = {}
        for name, path in TARGETS.items():
            src = json.loads(path.read_text())
            r, p = src["RSR"], src["PBR_new"]
            k_r, n_r = r.get("k_recovered", r.get("k")), r.get("n_p_minus", r.get("n"))
            t_r = float(r.get("mean_admitted", src.get("t_mean", 1.0)) or 1.0)
            k_p, n_p = p.get("k_bypassed", p.get("k")), p.get("n_resolved", p.get("n"))
            cr, cp = correct(k_r, n_r, jc, t_r), correct(k_p, n_p, jc, 1.0)
            res["headline"][name] = dict(
                RSR=dict(kn=f"{k_r}/{n_r}", registered=r.get("RSR_corrected", r.get("corrected")),
                         blind=round(cr[0], 4), blind_ci=[round(cr[1], 4), round(cr[2], 4)], blind_verdict=verdict(cr[1], cr[2])),
                PBR=dict(kn=f"{k_p}/{n_p}", registered=p.get("PBR_corrected", p.get("corrected")),
                         blind=round(cp[0], 4), blind_ci=[round(cp[1], 4), round(cp[2], 4)], blind_verdict=verdict(cp[1], cp[2])))
    out = Path(args.out) if args.out else ROOT / "results" / f"analysis_blind_relabel_{_dt.datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "blind_relabel.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
