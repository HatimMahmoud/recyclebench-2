#!/usr/bin/env python3
"""
tools/run_independent_probe.py

Possibility-1 (independent validator) driver, wired to RecycleBench's REAL
gate / judge / provider. For each P- behaviour it inverts the OLD model's
harmful response with RPE (MVP engine, swappable), gates the recovered
candidates with the SAME production gate the paraphrase arm uses, re-queries
the NEW model, judges the result, and writes an RPE manifest + prints the
convergent-validity report (RSR_paraphrase vs RSR_rpe, Cohen's kappa,
divergences).

RUN (from repo root), on the two benchmarks whose paraphrase manifest exists:

    python tools/run_independent_probe.py \
        --run-dir     results/runs/20260804_020847_llama3-8b_to_llama31-8b_full \
        --recycle-dir results/runs/20260805_025658_llama3-8b_to_llama31-8b_RECYCLE \
        --benchmarks  advbench strongreject \
        --config      configs/real_ollama.yaml \
        --new-model   new \
        --rpe-model   paraphraser \
        --out         results/runs/20260805_025658_llama3-8b_to_llama31-8b_RECYCLE/rpe

NOTE ON HARMBENCH: the 20260805_025658 RECYCLE run has advbench + strongreject
manifests only; the harmbench paraphrase manifest lives in the *_qwen3b_SMOKE run.
Point --recycle-dir there and use --benchmarks harmbench to include it.

============================================================================
THE ONLY REPO-COUPLING THAT MAY NEED EDITING is the TYPES ADAPTER block below.
It builds Behaviour / Variant objects and reads the admitted flag out of
GateScores. Adjust field names once to match src/recyclebench/types.py; nothing
else in this file or the independent_probe package depends on those shapes.
============================================================================
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# --- make the package importable when run from repo root ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recyclebench.independent_probe.contract import ProbeItem
from recyclebench.independent_probe.rpe_engine import RPEEngine
from recyclebench.independent_probe.analyze_agreement import analyze


# ===========================================================================
# TYPES ADAPTER  --  the single place that touches src/recyclebench/types.py.
# If any of these field names is wrong, fix it HERE only.
# ===========================================================================
from recyclebench.types import Behaviour, Variant, Response  # noqa: E402

def make_behaviour(bid: str, prompt: str):
    """Build a Behaviour. Confirmed fields: bid, canonical_prompt, category, source."""
    return Behaviour(bid=bid, canonical_prompt=prompt,
                     category="unknown", source="rpe_probe")

def make_variant(bid, text, index):
    """Build a Variant (frozen dataclass -> bid must go in the constructor).
    Required fields: bid, text, transform_class, surface_distance; index optional."""
    return Variant(bid=bid, text=text, transform_class="rpe_inversion",
                   surface_distance=0.0, index=index)


def gate_admits(gate, behaviour, variant):
    """Call the real gate: RealGate.check(behaviour, variant) -> GateScores.
    Returns (admitted:bool, scores:dict) with all four component scores for the cards."""
    gs = gate.check(behaviour, variant)
    scores = {name: (float(getattr(gs, name)) if getattr(gs, name, None) is not None else None)
              for name in ("cosine", "bertscore", "nli_entail", "llm_intent")}
    return bool(gs.admitted), scores

def response_of(text: str):
    try:
        return Response(text=text, latent_harmful=None)
    except TypeError:
        return Response(text=text)
# ===========================================================================
# END TYPES ADAPTER
# ===========================================================================


def _load_config(config_path: Path):
    """Load the RecycleBench config.

    Confirmed API: recyclebench.config.Config.load(path) -> Config (static method).
    We call that directly; the fallbacks below only trigger if the API changes.
    """
    from recyclebench.config import Config
    p = str(config_path)

    # confirmed path
    try:
        return Config.load(p)
    except AttributeError:
        pass  # API moved; fall through to discovery
    # (discovery fallbacks, in case Config.load is renamed later)
    attempts = []
    for m_name in ("load", "from_yaml", "from_file", "from_path", "parse_file"):
        m = getattr(Config, m_name, None)
        if callable(m):
            try:
                return m(p)
            except Exception as e:                       # noqa: BLE001
                attempts.append(f"Config.{m_name}(path): {e!r}")
    for kw in ({}, {"path": p}, {"config_path": p}):
        try:
            return Config(p) if not kw else Config(**kw)
        except Exception as e:                           # noqa: BLE001
            attempts.append(f"Config({kw or 'path'}): {e!r}")
    tried = "\n  - ".join(attempts) if attempts else "(none)"
    raise ImportError("Could not load config. Tried:\n  - " + tried)


def build_real_components(config_path: Path, new_model_name: str, rpe_model_name: str):
    """Construct the real provider(s), judge ensemble, and gate from the config."""
    from recyclebench.backends.providers import (
        load_provider_model, ProviderConfig, ProviderModel)
    from recyclebench.backends.real_judge import build_ensemble
    from recyclebench.backends.real_gate import RealGate

    cfg = _load_config(config_path)

    # NEW model = the model under test (llama3.1-8b), re-queried with recovered prompts.
    new_model = load_provider_model(new_model_name, "openai_compatible", cfg)

    # RPE engine reuses a ProviderModel too (OpenAI-compatible), NOT a bespoke client.
    rpe_model = load_provider_model(rpe_model_name, "openai_compatible", cfg)
    engine = RPEEngine(provider_respond=lambda p: rpe_model.respond(p).text)

    # --- Judge: DECORRELATED, built from cfg.judges exactly as cli._build_real does, so the
    # RPE arm and the recycle (run-real) arm share one production judge. Target-as-judge
    # (the new model grading its own compliance) is CIRCULAR and only used as a last resort,
    # with a loud warning — a number produced that way must not be reported.
    judge_models = {role: ProviderModel(f"judge:{role}", ProviderConfig(**spec))
                    for role, spec in (getattr(cfg, "judges", None) or {}).items()}
    judge = build_ensemble(judge_models) if judge_models else None
    if judge is None or not getattr(judge, "members", None):
        judge = build_ensemble({"rubric": new_model})
        print("  [!!] no 'judges:' block in config -> RPE is falling back to TARGET-AS-JUDGE\n"
              "       (the new model grading its own output). This is CIRCULAR and must not\n"
              "       back a reported number. Add a 'judges:' block (see recycle_3_31.yaml).")
    else:
        print(f"  [judge] decorrelated ensemble from cfg.judges: {','.join(judge_models)}")

    # Real gate. thresholds come from cfg.gate (frozen at pre-registration).
    gate = RealGate(cfg.gate, cfg=cfg)

    return cfg, new_model, engine, judge, gate


def run_one_benchmark(bench: str, items: list[ProbeItem], *, engine, new_model,
                      judge, gate, n_candidates: int, verbose: bool) -> list[ProbeItem]:
    from recyclebench.judge import looks_like_refusal
    for k, it in enumerate(items):
        # 1. invert the OLD model's harmful response (never a refusal -> see contract)
        cands = engine.invert(it.old_response, n_candidates=n_candidates)
        it.rpe_candidates = cands
        behaviour = make_behaviour(it.bid, it.prompt)

        details = []          # one entry per candidate, for the dashboard cards
        admitted = []
        recovered = None      # None = unresolved (nothing admitted)
        for i, c in enumerate(cands):
            variant = make_variant(it.bid, c, i)          # frozen-safe: bid in ctor
            ok, scores = gate_admits(gate, behaviour, variant)
            entry = {"text": c, "admitted": ok, "scores": scores,
                     "response": None, "broke": None}
            if ok:
                admitted.append(c)
                resp = new_model.respond(c)               # Response
                rtext = resp.text if hasattr(resp, "text") else str(resp)
                verdict = judge.classify(c, resp)         # 1 comply / 0 refuse
                broke = (verdict == 1)
                entry["response"] = rtext
                entry["broke"] = broke
                recovered = 1 if (recovered == 1 or broke) else 0
            details.append(entry)

        it.rpe_admitted = admitted
        it.rpe_details = details
        it.rpe_recovered = recovered                      # stays None if nothing admitted

        if verbose:
            tag = {1: "RECOVERED", 0: "held", None: "unresolved"}[it.rpe_recovered]
            print(f"  [{bench} {k+1}/{len(items)}] {it.bid}: "
                  f"{len(cands)} cand, {len(admitted)} admitted -> {tag}")
    return items


def main():
    ap = argparse.ArgumentParser(description="RPE independent-validator probe")
    # data source: mirror make_dashboard.py -- P- comes from cache + CSV via partition()
    ap.add_argument("--old", required=True, help="cfg.real key for the OLD model (e.g. llama32-1b)")
    ap.add_argument("--new-model", required=True, help="cfg.real key for the NEW model under test")
    ap.add_argument("--rpe-model", required=True, help="cfg.real key for the RPE engine model")
    ap.add_argument("--cache", required=True, type=Path, help="the run's cache .jsonl (as make_dashboard uses)")
    ap.add_argument("--behaviours", nargs="+", required=True,
                    help="benchmark CSV(s) -- same as make_dashboard --behaviours (e.g. data/harmbench.csv)")
    ap.add_argument("--benchmarks", nargs="+", required=True,
                    help="which benchmark(s) to probe, by CSV stem (e.g. harmbench)")
    ap.add_argument("--recycle-dir", type=Path, default=None,
                    help="optional *_RECYCLE dir with manifest_<bench>.jsonl (gives RSR_paraphrase)")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="optional *_full dir; if given, its pminus_<bench>.json p_minus_bids "
                         "are used as a belt-and-suspenders bid filter")
    ap.add_argument("--config", type=Path, default=Path("configs/real_ollama.yaml"))
    ap.add_argument("--old-cache-name", default=None,
                    help="override: exact string the cache was keyed under for OLD "
                         "(defaults to the --old cfg key; set if partition finds 0 P-)")
    ap.add_argument("--new-cache-name", default=None,
                    help="override: exact string the cache was keyed under for NEW")
    ap.add_argument("--out", type=Path, required=True, help="output dir for rpe manifests")
    ap.add_argument("--n-candidates", type=int, default=5)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg, new_model, engine, judge, gate = build_real_components(
        args.config, args.new_model, args.rpe_model)

    # --- build P- exactly like the dashboard does: partition(cache, csvs, old, new, k) ---
    from recyclebench.independent_probe.contract import (
        load_probe_items_from_partition, load_pminus_bids)
    from recyclebench.independent_probe.run_probe import write_manifest
    # import the dashboard's own partition + cache loader so the source is identical
    import importlib.util as _ilu
    _md_path = Path(__file__).resolve().parents[0] / "make_dashboard.py"
    _spec = _ilu.spec_from_file_location("_md", _md_path)
    _md = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_md)

    # resolve the OLD/NEW model names the CACHE was written under. make_dashboard keys the
    # cache via _key(model_name, ...) where model_name is the string passed as --old/--new to
    # make_dashboard. That is the provider .name, which ProviderModel sets to the cfg key.
    from recyclebench.backends.providers import load_provider_model
    old_model = load_provider_model(args.old, "openai_compatible", cfg)
    old_cache_name = args.old_cache_name or old_model.name
    new_cache_name = args.new_cache_name or new_model.name

    k = cfg.budget.k_canonical_samples
    cache = _md.load_cache(str(args.cache))
    cells, per_bench = _md.partition(cache, [str(c) for c in args.behaviours],
                                     old_cache_name, new_cache_name, k)
    _npm = len(cells.get("pminus", []))
    print(f"  [partition] {_npm} P- records across {len(args.behaviours)} CSV(s) "
          f"(old='{old_cache_name}', new='{new_cache_name}', k={k})")
    if _npm == 0:
        print("  [!] partition found 0 P- -- the --old/--new names likely don't match the "
              "strings the cache was written under. Re-run with --old-cache-name/--new-cache-name "
              "set to the exact --old/--new values used by make_dashboard for THIS run "
              "(check the run's META.txt or the make_dashboard command in your logs).")

    all_reports = {}
    for bench in args.benchmarks:
        print(f"\n=== {bench} ===")
        restrict = None
        if args.run_dir is not None:
            pj = args.run_dir / f"pminus_{bench}.json"
            if pj.exists():
                restrict = load_pminus_bids(pj)
                if restrict:
                    print(f"  [filter] restricting to {len(restrict)} p_minus_bids from {pj.name}")

        recycle_manifest = None
        if args.recycle_dir is not None:
            rm = args.recycle_dir / f"manifest_{bench}.jsonl"
            recycle_manifest = rm if rm.exists() else None
            if recycle_manifest is None:
                print(f"  [warn] no paraphrase manifest for {bench} in {args.recycle_dir} "
                      f"-> RSR_paraphrase unavailable; RPE arm still computed.")

        items = load_probe_items_from_partition(
            bench, cells, recycle_manifest=recycle_manifest, restrict_bids=restrict)

        if not items:
            print(f"  [skip] {bench}: 0 P- behaviours found in partition "
                  f"(check --behaviours CSV stem matches '{bench}' and cache has old+new responses)")
            continue
        print(f"  {len(items)} P- behaviours to probe")

        items = run_one_benchmark(bench, items, engine=engine, new_model=new_model,
                                  judge=judge, gate=gate, n_candidates=args.n_candidates,
                                  verbose=not args.quiet)

        out_manifest = args.out / f"rpe_manifest_{bench}.jsonl"
        write_manifest(items, out_manifest)

        report = analyze(items)
        all_reports[bench] = report
        print("\n" + report.summary())
        print(f"\nwrote {out_manifest}")

    # write a small machine-readable rollup the dashboard tool reads
    import json
    rollup = {b: _report_to_dict(r) for b, r in all_reports.items()}
    (args.out / "rpe_agreement.json").write_text(json.dumps(rollup, indent=2))
    print(f"\nwrote {args.out / 'rpe_agreement.json'}")


def _report_to_dict(r) -> dict:
    return {
        "n_total": r.n_total, "n_both_resolved": r.n_both_resolved,
        "rsr_para": r.rsr_para, "rsr_rpe": r.rsr_rpe,
        "rsr_para_ci": list(r.rsr_para_ci), "rsr_rpe_ci": list(r.rsr_rpe_ci),
        "agreement_rate": r.agreement_rate, "cohen_kappa": r.cohen_kappa,
        "both_recovered": len(r.both_recovered), "both_held": len(r.both_held),
        "rpe_only": len(r.rpe_only), "para_only": len(r.para_only),
        "para_unresolved": r.para_unresolved, "rpe_unresolved": r.rpe_unresolved,
    }


if __name__ == "__main__":
    main()