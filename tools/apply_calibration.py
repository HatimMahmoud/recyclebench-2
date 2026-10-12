#!/usr/bin/env python3
"""
tools/apply_calibration.py

Close the last gap in the real pipeline: judge-ingest prints the calibrated bundle-level
(se, sp) but persists nothing, and run-real computes RSR/PBR with an IDENTITY judge
(se=sp=1 -> corrected==raw). This step takes the human-labelled calibration sheet and the
run's own cache+manifest, rebuilds the two model runs (served from cache, so no new target
calls), and recomputes RSR/PBR through the repo's OWN compute_rsr / compute_pbr using the
calibrated JudgeChars. Nothing is re-estimated here by hand — the estimator, the Rogan-Gladen
correction and the CIs are all the package's.

RUN (after judge-ingest has shown the judge is admitted):

    python3 tools/apply_calibration.py \
      --config     configs/recycle_3_31.yaml \
      --old        llama3-8b --new llama31-8b \
      --behaviours results/cells_llama3-8b_llama31-8b/sorrybench_p_minus.csv \
      --manifest   results/runs/recycle_3_31_sorrybench/manifest_sorrybench.jsonl \
      --cache      cache/llama_gen.jsonl \
      --sheet      audit/sorrybench/judge_sheet.csv \
      --out        results/runs/recycle_3_31_sorrybench/corrected_sorrybench.json

Prints RAW vs CORRECTED RSR/PBR side by side and writes the JSON. If the sheet's judge is
NOT admitted (kappa below cfg.decision.kappa_floor) it refuses to emit a corrected headline,
per the protocol — swap the judge and re-calibrate instead.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_config(p: Path):
    from recyclebench.config import Config
    return Config.load(str(p))


def _build_judge(cfg):
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    from recyclebench.backends.real_judge import build_ensemble
    jm = {role: ProviderModel(f"judge:{role}", ProviderConfig(**spec))
          for role, spec in (getattr(cfg, "judges", None) or {}).items()}
    judge = build_ensemble(jm) if jm else None
    if judge is None or not getattr(judge, "members", None):
        raise SystemExit("no 'judges:' block — the runs must be rebuilt with the SAME judge "
                         "that was calibrated. Add a judges: block (see recycle_3_31.yaml).")
    return judge


# reuse the exact manifest loader the export tool uses, to keep bid alignment identical
def _load_manifest(manifest_jsonl: Path):
    import importlib.util
    src = Path(__file__).resolve().parent / "judge_export_real.py"
    s = importlib.util.spec_from_file_location("jer", str(src))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m._load_manifest(manifest_jsonl)


def main():
    ap = argparse.ArgumentParser(description="apply a calibrated judge to RSR/PBR")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--old", required=True, help="cfg.real key for M_old")
    ap.add_argument("--new", required=True, help="cfg.real key for M_new (headline model)")
    ap.add_argument("--behaviours", required=True, type=Path, help="P- cells CSV")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--sheet", required=True, type=Path, help="the labelled judge_sheet.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true",
                    help="emit corrected numbers even if the judge is below the kappa floor")
    args = ap.parse_args()

    from recyclebench.datasets import load_jailbreakbench
    from recyclebench.backends.providers import load_provider_model
    from recyclebench.realrun import ResponseCache, CachedModel
    from recyclebench.screening import run_model, regression_set
    from recyclebench.recycling import compute_rsr
    from recyclebench.pbr import compute_pbr
    from recyclebench.judge_audit import ingest_judge_audit, calibrate_from_human
    from recyclebench.stats import JudgeChars

    cfg = _load_config(args.config)
    behaviours = load_jailbreakbench(str(args.behaviours))
    manifest = _load_manifest(args.manifest)
    judge = _build_judge(cfg)

    # --- 1. calibrate from the labelled sheet (the repo's own estimator) ---
    labels = ingest_judge_audit(str(args.sheet))
    strata = args.sheet.parent / "judge_strata.txt"
    n_pos = n_neg = None
    if strata.exists():
        kv = dict(line.split("=") for line in strata.read_text().split())
        n_pos, n_neg = int(kv["n_pos_total"]), int(kv["n_neg_total"])
    cal = calibrate_from_human(labels, n_pos, n_neg, cfg.decision.kappa_floor)
    jc: JudgeChars = cal.bundle.chars
    admitted = bool(cal.bundle.admitted)
    print(f"  [calibration] se={jc.se:.4f} sp={jc.sp:.4f} kappa={cal.bundle.kappa:.4f} "
          f"admitted={admitted}  (n={cal.bundle.n} complete bundles)")
    if not admitted and not args.force:
        raise SystemExit(f"judge NOT admitted (kappa {cal.bundle.kappa:.3f} < "
                         f"{cfg.decision.kappa_floor}); refusing to emit a corrected headline. "
                         f"Swap the judge and re-calibrate, or pass --force to override.")

    # --- 2. rebuild the two runs from cache with the SAME judge (no new target calls) ---
    cache = ResponseCache(str(args.cache), enabled=True)
    old = CachedModel(load_provider_model(args.old, "openai_compatible", cfg), cache)
    new = CachedModel(load_provider_model(args.new, "openai_compatible", cfg), cache)
    old_run = run_model(old, behaviours, manifest, judge, cfg,
                        collect_calib=False, store_responses=False)
    new_run = run_model(new, behaviours, manifest, judge, cfg,
                        collect_calib=False, store_responses=False)
    print(f"  [replay] old: calls={old.calls} hits={old.hits} | "
          f"new: calls={new.calls} hits={new.hits} (judge ran fresh)")
    cache.close()

    # --- 3. recompute RSR + PBR through the package, once identity and once calibrated ---
    p_minus = regression_set(old_run, new_run)
    ident = JudgeChars(se=1.0, sp=1.0, response_level=False)

    rsr_raw = compute_rsr(old_run, new_run, p_minus, ident, cfg.decision.min_p_minus)
    rsr_cal = compute_rsr(old_run, new_run, p_minus, jc, cfg.decision.min_p_minus)
    pbr_raw = compute_pbr(new_run, ident, cfg.decision.pi_star)
    pbr_cal = compute_pbr(new_run, jc, cfg.decision.pi_star)

    def ci(x):
        return f"[{x.low:.4f}, {x.high:.4f}]" if x is not None else "--"

    print("\n" + "=" * 70)
    print(f"CALIBRATED HEADLINE — {args.old} -> {args.new}   (P-={len(p_minus)})")
    print("=" * 70)
    print(f"RSR  raw={rsr_raw.rsr_raw:.4f} {ci(rsr_raw.raw_ci)}   "
          f"corrected={rsr_cal.rsr_corrected:.4f} {ci(rsr_cal.corrected_ci)}   "
          f"(McNemar p={rsr_cal.mcnemar_p:.2e}, t_mean={rsr_cal.mean_admitted:.2f})")
    print(f"PBR({args.new})  raw={pbr_raw.pbr_raw:.4f} {ci(pbr_raw.corrected_ci)}   "
          f"corrected={pbr_cal.pbr_corrected:.4f} {ci(pbr_cal.corrected_ci)}   "
          f"{pbr_cal.verdict}")

    out = {
        "pair": f"{args.old} -> {args.new}",
        "n_p_minus": len(p_minus),
        "judge": {"se": jc.se, "sp": jc.sp, "kappa": cal.bundle.kappa,
                  "admitted": admitted, "n_bundles": cal.bundle.n,
                  "se_ci": [cal.se_ci.low, cal.se_ci.high] if cal.se_ci else None,
                  "sp_ci": [cal.sp_ci.low, cal.sp_ci.high] if cal.sp_ci else None},
        "RSR": rsr_cal.summary(),
        "PBR_new": pbr_cal.summary(),
        "RSR_raw": rsr_raw.summary(),
        "PBR_new_raw": pbr_raw.summary(),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.out}")
    print("\nReport the CORRECTED RSR/PBR with their CIs as the headline; the judge line "
          "(se, sp, kappa) is the validation that admits them.")


if __name__ == "__main__":
    main()
