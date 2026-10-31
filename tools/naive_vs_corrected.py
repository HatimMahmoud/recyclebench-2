#!/usr/bin/env python3
"""tools/naive_vs_corrected.py -- contribution C4: the gap between the naïve and the corrected estimate.

For the SORRY-Bench lineage (frozen manifest, cached responses, no model calls) every pair's RSR and every model's
PBR is computed three ways, each through the package estimators (`compute_rsr`, `compute_pbr`):

  naïve      the offline refusal-prefix rule throughout (its own P⁻ and C_M) -- what a string-matching pipeline reports
  judge raw  the stored qwen2.5:14b verdicts (`cache/judge_lineage_sorrybench.jsonl`) -- the run's own judge, uncorrected
  corrected  judge raw, Rogan–Gladen corrected at bundle level with the 60-bundle SORRY-Bench calibration (delta-method CI)

plus the anchor's confirmatory replication (held-out set) from its registered corrected file. The calibration was
labelled on llama3.1-8b responses; applying it to the other models is an assumption, stated in the output.

  python3 tools/naive_vs_corrected.py            # -> results/analysis_c4_<stamp>/c4.{json,txt}
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import lineage_core as lc                                      # noqa: E402
import study_data as sd                                        # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.datasets import load_jailbreakbench          # noqa: E402
from recyclebench.manifest import Manifest                     # noqa: E402
from recyclebench.pbr import compute_pbr                       # noqa: E402
from recyclebench.recycling import compute_rsr                 # noqa: E402
from recyclebench.screening import regression_set              # noqa: E402
from recyclebench.stats import corrected_interval, wilson_interval  # noqa: E402

CFG = "configs/real_llama_lineage_all.yaml"
RUN = "results/runs/lineage_sorrybench_20261020"
VERDICTS = "cache/judge_lineage_sorrybench.jsonl"
HELDOUT = "results/runs/confirm_heldout_3_31/corrected_heldout_recal_n60.json"


def _r(x, d=4):
    return None if x is None else round(float(x), d)


def _wil(k, n):
    w = wilson_interval(k, n)
    return [_r(w.low), _r(w.high)]


def rows_for(runs, models, jc, pi_star):
    from recyclebench.stats import JudgeChars
    ident = JudgeChars(se=1.0, sp=1.0, response_level=False)
    rsr, pbr = {}, {}
    for o, n in zip(models, models[1:]):
        pm = regression_set(runs[o], runs[n])
        r = compute_rsr(runs[o], runs[n], pm, jc or ident, 30) if pm else None
        if r is None:
            rsr[(o, n)] = None
            continue
        # registered convention (exposé §5, Amendment 1 §A2): unresolved fixes (no admitted rewording) are excluded
        # from the denominator and reported; |P⁻| as denominator is kept as the conservative variant
        n_res = sum(1 for b in pm if runs[n].recovery[b].n_admitted > 0)
        c = corrected_interval(r.k_recovered, n_res, jc) if jc and n_res else None
        rsr[(o, n)] = dict(p_minus=r.n_p_minus, n=n_res, unresolved=r.n_p_minus - n_res, k=r.k_recovered,
                           raw=_r(r.k_recovered / n_res) if n_res else None, wilson=_wil(r.k_recovered, n_res) if n_res else None,
                           corrected=_r(c.point) if c else None, corrected_ci=[_r(c.low), _r(c.high)] if c else None,
                           over_p_minus=dict(raw=_r(r.rsr_raw), corrected=_r(r.rsr_corrected) if jc else None,
                                             corrected_ci=[_r(r.corrected_ci.low), _r(r.corrected_ci.high)] if jc else None))
    for m in models:
        p = compute_pbr(runs[m], jc or ident, pi_star)
        pbr[m] = dict(n=p.n_resolved, k=p.k_bypassed, raw=_r(p.pbr_raw), wilson=_wil(p.k_bypassed, p.n_resolved),
                      corrected=_r(p.pbr_corrected) if jc else None,
                      corrected_ci=[_r(p.corrected_ci.low), _r(p.corrected_ci.high)] if jc else None,
                      verdict=p.verdict if jc else None)
    return rsr, pbr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = Config.load(ROOT / CFG)
    models = list(cfg.lineage)
    behaviours = load_jailbreakbench(ROOT / "data/sorrybench.csv")
    manifest = Manifest.load(ROOT / RUN / "manifest.jsonl")
    cache = lc.load_response_cache(ROOT / "cache/llama_gen.jsonl")
    tag = lc.judge_tag_from_cfg(cfg)
    jc = sd.anchor_judge_chars(sd.judge_calibrations())
    if jc is None:
        raise SystemExit("no SORRY-Bench calibration found (audit/sorrybench/judge_sheet.csv)")
    pi = cfg.decision.pi_star
    _, off_runs = lc.build_runs(cfg, behaviours, manifest, cache, lc.OfflineVerdicts(), models)
    _, q_runs = lc.build_runs(cfg, behaviours, manifest, cache, lc.StoredVerdicts(lc.VerdictStore(ROOT / VERDICTS), tag), models)
    off_rsr, off_pbr = rows_for(off_runs, models, None, pi)
    q_rsr, q_pbr = rows_for(q_runs, models, jc, pi)

    pairs = []
    for o, n in zip(models, models[1:]):
        a, b = off_rsr[(o, n)], q_rsr[(o, n)]
        pairs.append(dict(pair=f"{o} -> {n}", status=lc.pair_status(o, n)["kind"], naive=a, judge=b,
                          gap_corrected_minus_naive=_r(b["corrected"] - a["raw"]) if a and b and b["corrected"] is not None else None,
                          gap_corrected_minus_judge_raw=_r(b["corrected"] - b["raw"]) if b and b["corrected"] is not None else None))
    mods = [dict(model=m, naive=off_pbr[m], judge=q_pbr[m],
                 gap_corrected_minus_naive=_r(q_pbr[m]["corrected"] - off_pbr[m]["raw"]),
                 gap_corrected_minus_judge_raw=_r(q_pbr[m]["corrected"] - q_pbr[m]["raw"])) for m in models]
    held = json.loads((ROOT / HELDOUT).read_text()) if (ROOT / HELDOUT).exists() else None
    res = dict(created=_dt.datetime.now().isoformat(timespec="seconds"), benchmark="SORRY-Bench (440)", run=RUN,
               judge=tag, calibration=dict(se=_r(jc.se), sp=_r(jc.sp), bundles=60, labelled_on="llama31-8b responses"),
               assumption="the calibration is applied to every model; it was measured on llama3.1-8b responses only",
               pairs=pairs, models=mods,
               confirmatory=None if not held else dict(RSR=held.get("RSR"), PBR_new=held.get("PBR_new")))
    lines = [f"C4 · naïve (offline rule) → judge raw (qwen2.5:14b) → corrected (se {jc.se:.3f}, sp {jc.sp:.3f}) · SORRY-Bench", ""]
    fmt = lambda x: "—" if not x else f"{x['k']}/{x['n']} = {x['raw']:.3f}"
    lines.append("RSR over the RESOLVED fixes (registered convention; unresolved excluded and counted in brackets)")
    lines.append(f"{'RSR pair':<28}{'naïve':>18}{'judge raw':>18}{'corrected [95%]':>26}{'Δ vs naïve':>12}")
    for p in pairs:
        b = p["judge"]
        c = "—" if not b else f"{b['corrected']:.3f} [{b['corrected_ci'][0]:.2f},{b['corrected_ci'][1]:.2f}]"
        lines.append(f"{p['pair'] + (' (+' + str(b['unresolved']) + ')' if b and b['unresolved'] else ''):<28}{fmt(p['naive']):>18}{fmt(b):>18}{c:>26}{p['gap_corrected_minus_naive'] if p['gap_corrected_minus_naive'] is not None else '—':>12}")
    lines += ["", f"{'PBR model':<28}{'naïve':>18}{'judge raw':>18}{'corrected [95%]':>26}{'Δ vs naïve':>12}"]
    for m in mods:
        b = m["judge"]
        c = f"{b['corrected']:.3f} [{b['corrected_ci'][0]:.2f},{b['corrected_ci'][1]:.2f}]"
        lines.append(f"{m['model']:<28}{fmt(m['naive']):>18}{fmt(b):>18}{c:>26}{m['gap_corrected_minus_naive']:>12}")
    if held:
        r = held["RSR"]
        lines += ["", f"confirmatory (held-out, registered): RSR {r['k_recovered']}/{r['n_p_minus']} raw {r['RSR_raw']:.3f} "
                      f"→ corrected {r['RSR_corrected']:.3f} {r['CI']}"]
    out = Path(args.out) if args.out else ROOT / "results" / f"analysis_c4_{_dt.datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "c4.json").write_text(json.dumps(res, indent=1))
    (out / "c4.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
