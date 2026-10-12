#!/usr/bin/env python3
"""
tools/judge_export_real.py

Judge-calibration EXPORT for a REAL run (the built-in `judge-export` is wired to the mock
lineage/MockJudge path and cannot read a real cache+manifest — this is its real-run twin).

It reconstructs the frozen manifest from a recycle run's manifest JSONL, replays the NEW
model over the admitted variants straight out of the response cache (no new target calls),
grades each reply with the SAME decorrelated judge configured under `judges:` in the config,
and writes the bundle-complete annotation sheet the repo's own `judge-ingest` consumes
unchanged. Reuses recyclebench.judge_audit.{sample_bundles,export_judge_audit}, so the sheet
schema and the any-success (OR) bundle rule are the repo's, not reinvented here.

WORKFLOW (both arms calibrate the same way — point --manifest at each arm's manifest):

  1. export the sheet (cheap: model calls hit cache, only the judge runs fresh)
       python3 tools/judge_export_real.py \
         --config     configs/recycle_3_31.yaml \
         --new-model  llama31-8b \
         --behaviours results/cells_llama3-8b_llama31-8b/sorrybench_p_minus.csv \
         --manifest   results/runs/recycle_3_31_sorrybench/manifest_sorrybench.jsonl \
         --cache      cache/llama_gen.jsonl \
         --n-bundles  60 \
         --out        audit/sorrybench/judge_sheet.csv

  2. hand-label the `human_fulfilled` column (1 = reply fulfils the request, 0 = refuses/
     deflects/empty). Label EVERY row of each bid group — a partly-labelled group is dropped.
     The rubric is written next to the sheet as JUDGE_RUBRIC.txt.

  3. ingest -> bundle-level (se, sp) with CIs + the kappa-floor admit decision:
       python3 -m recyclebench judge-ingest audit/sorrybench/judge_sheet.csv \
         --config configs/recycle_3_31.yaml

If kappa clears cfg.decision.kappa_floor the judge is admitted and the (se, sp) it prints are
what convert RAW RSR/PBR into the corrected headline; if not, the judge must be swapped or the
human-audited subset reported directly. Until then every RSR/PBR stays RAW.

CONTAINS prompts and model responses -> working file, never a released artefact (ETHICS.md).
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

# make the installed package importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_config(p: Path):
    from recyclebench.config import Config
    return Config.load(str(p))


def _load_behaviours(csv_path: Path, goal_col: str, cat_col: str):
    """P- behaviours via the PACKAGE loader — critical for bid alignment.

    recyclebench.datasets.load_jailbreakbench IGNORES any 'bid' column and assigns
    bid = f"jbb-{row_index:04d}". The manifest was frozen from behaviours loaded the same
    way on the same CSV, so its keys are those jbb-NNNN bids. Reading the CSV's own 'bid'
    column here would mis-key every lookup (the 8/61 symptom). So we delegate to the exact
    loader run-real used, on the exact same CSV in the same row order.
    """
    from recyclebench.datasets import load_jailbreakbench
    return load_jailbreakbench(str(csv_path), goal_col=goal_col, category_col=cat_col)


def _load_manifest(manifest_jsonl: Path):
    """Rebuild a Manifest (admitted variants + gate audit) from a recycle manifest JSONL.

    Consumes the two line types the recycle run writes:
      {"type":"admitted","bid":...,"variants":[{bid,text,transform_class,surface_distance,
                                                index,intent_preserved}, ...]}
      {"type":"audit","bid":...,"index":...,"transform_class":...,"intent_preserved":...,
       "admitted":...,"cosine":...,"bertscore":...,"nli_entail":...,"llm_intent":...,...}
    Only `admitted` is needed to replay variants; `audit` is carried through if present.
    """
    from recyclebench.types import Variant
    from recyclebench.manifest import Manifest
    try:
        from recyclebench.manifest import GateAuditRecord
    except Exception:  # noqa: BLE001
        GateAuditRecord = None

    admitted: dict[str, list] = {}
    audit: list = []
    with manifest_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("type")
            if t == "admitted":
                vs = []
                for v in d.get("variants", []):
                    vd = dict(v)
                    vd.setdefault("bid", d["bid"])
                    vs.append(Variant.from_dict(vd))
                admitted[d["bid"]] = vs
            elif t == "audit" and GateAuditRecord is not None:
                try:
                    audit.append(GateAuditRecord(
                        d["bid"], int(d.get("index", 0)), d.get("transform_class", "unknown"),
                        d.get("intent_preserved"), bool(d.get("admitted", False)),
                        float(d.get("cosine", 0.0)), float(d.get("bertscore", 0.0)),
                        float(d.get("nli_entail", 0.0)), float(d.get("llm_intent", 0.0))))
                except Exception:  # noqa: BLE001 — audit is optional for export
                    pass
    m = Manifest(admitted=admitted, audit=audit)
    return m


def _build_judge(cfg):
    """Decorrelated judge from cfg.judges — identical to cli._build_real / the RPE probe."""
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    from recyclebench.backends.real_judge import build_ensemble
    judge_models = {role: ProviderModel(f"judge:{role}", ProviderConfig(**spec))
                    for role, spec in (getattr(cfg, "judges", None) or {}).items()}
    judge = build_ensemble(judge_models) if judge_models else None
    if judge is None or not getattr(judge, "members", None):
        raise SystemExit(
            "no 'judges:' block in the config — calibration needs the SAME decorrelated judge "
            "the run used. Add a judges: block (see configs/recycle_3_31.yaml) and re-run the "
            "arm so its manifest/cache reflect that judge, then export.")
    print(f"  [judge] {','.join(judge_models)}")
    return judge


def main():
    ap = argparse.ArgumentParser(description="real-run judge-calibration sheet exporter")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--new-model", required=True,
                    help="cfg.real key of the model whose RSR/PBR you will correct (e.g. llama31-8b)")
    ap.add_argument("--behaviours", required=True, type=Path,
                    help="the P- cells CSV (columns bid, Category, Goal)")
    ap.add_argument("--manifest", required=True, type=Path,
                    help="the arm's manifest JSONL (admitted variants to replay)")
    ap.add_argument("--cache", required=True, type=Path,
                    help="the run's response cache JSONL (model replies served from here)")
    ap.add_argument("--goal-col", default="Goal")
    ap.add_argument("--cat-col", default="Category")
    ap.add_argument("--n-bundles", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("audit/judge_sheet.csv"))
    args = ap.parse_args()

    from recyclebench.backends.providers import load_provider_model
    from recyclebench.realrun import ResponseCache, CachedModel
    from recyclebench.screening import run_model
    from recyclebench.judge_audit import sample_bundles, export_judge_audit

    cfg = _load_config(args.config)
    behaviours = _load_behaviours(args.behaviours, args.goal_col, args.cat_col)
    manifest = _load_manifest(args.manifest)
    judge = _build_judge(cfg)

    # restrict to behaviours that actually have admitted variants in this manifest
    behaviours = [b for b in behaviours if manifest.R(b.bid)]
    if not behaviours:
        raise SystemExit("no behaviours in --behaviours have admitted variants in --manifest "
                         "(check the manifest matches this pair/benchmark).")
    n_adm = sum(len(manifest.R(b.bid)) for b in behaviours)
    print(f"  [manifest] {len(behaviours)} behaviours, {n_adm} admitted variants to replay")

    # NEW model served from the SAME cache the run wrote -> replay is free; only the judge runs.
    cache = ResponseCache(args.cache, enabled=True)
    new = CachedModel(load_provider_model(args.new_model, "openai_compatible", cfg), cache)

    run = run_model(new, behaviours, manifest, judge, cfg,
                    collect_calib=False, store_responses=True)
    print(f"  [replay] target calls={new.calls} cache_hits={new.hits} "
          f"(judge calls ran fresh); {len(run.responses)} responses captured")
    if new.calls:
        print("  [note] some target replies were NOT in cache and were queried now — expected "
              "only if this manifest/cache pair differs from the original run.")
    cache.close()

    samp = sample_bundles(run, behaviours, manifest, run.responses,
                          n_bundles=args.n_bundles, seed=cfg.seed)
    out = export_judge_audit(samp, args.out, judge_labels=run.judge_labels)

    print("\n" + "=" * 68)
    print("JUDGE CALIBRATION — ANNOTATION SHEET (real run)")
    print("=" * 68)
    print(f"model            : {new.name}")
    print(f"bundles selected : {samp.n_bundles}   rows: {len(samp)}")
    print(f"strata (full run): judge-bypassed={samp.n_pos_total}  judge-blocked={samp.n_neg_total}")
    print(f"written          : {out}")
    print(f"rubric           : {Path(out).parent / 'JUDGE_RUBRIC.txt'}")
    print("\nLabel EVERY row of each bid group (bundle truth = OR over its rows; a partly-"
          "labelled group is dropped). Then:")
    print(f"  python3 -m recyclebench judge-ingest {out} --config {args.config}")


if __name__ == "__main__":
    main()