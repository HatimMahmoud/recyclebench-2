#!/usr/bin/env python3
"""RPE independent-validator probe (partition-sourced, detail-capturing)."""
from __future__ import annotations
import argparse, sys, json, importlib.util as _ilu
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recyclebench.independent_probe.contract import (
    ProbeItem, load_probe_items_from_partition, load_pminus_bids)
from recyclebench.independent_probe.rpe_engine import RPEEngine
from recyclebench.independent_probe.run_probe import write_manifest
from recyclebench.independent_probe.analyze_agreement import analyze
from recyclebench.types import Behaviour, Variant, Response


def make_behaviour(bid, prompt):
    return Behaviour(bid=bid, canonical_prompt=prompt, category="unknown", source="rpe_probe")

def make_variant(bid, text, index):
    return Variant(bid=bid, text=text, transform_class="rpe_inversion",
                   surface_distance=0.0, index=index)

def gate_admits(gate, behaviour, variant):
    gs = gate.check(behaviour, variant)
    scores = {n: (float(getattr(gs, n)) if getattr(gs, n, None) is not None else None)
              for n in ("cosine", "bertscore", "nli_entail", "llm_intent")}
    return bool(gs.admitted), scores


def _load_config(config_path):
    from recyclebench.config import Config
    return Config.load(str(config_path))


def build_real_components(config_path, new_model_name, rpe_model_name):
    from recyclebench.backends.providers import load_provider_model
    from recyclebench.backends.real_judge import build_ensemble
    from recyclebench.backends.real_gate import RealGate
    cfg = _load_config(config_path)
    new_model = load_provider_model(new_model_name, "openai_compatible", cfg)
    rpe_model = load_provider_model(rpe_model_name, "openai_compatible", cfg)
    engine = RPEEngine(provider_respond=lambda p: rpe_model.respond(p).text)
    # Build the judge from cfg.judges (decorrelated), NOT the target model.
    from recyclebench.backends.providers import ProviderModel, ProviderConfig
    _judges = getattr(cfg, "judges", None) or {}
    _members = {}
    for _role in ("rubric", "guard", "harmbench"):
        _spec = _judges.get(_role) if isinstance(_judges, dict) else None
        if _spec:
            _members[_role] = ProviderModel(f"judge-{_role}", ProviderConfig(**_spec))
    if _members:
        judge = build_ensemble(_members)
        print(f"  [judge] using cfg.judges: {list(_members)}")
    else:
        judge = build_ensemble({"rubric": new_model})
        print("  [judge] WARNING: no cfg.judges block -> falling back to target-as-judge (circular)")
    gate = RealGate(cfg.gate, cfg=cfg)
    return cfg, new_model, engine, judge, gate


def run_one_benchmark(bench, items, *, engine, new_model, judge, gate, n_candidates, verbose):
    for k, it in enumerate(items):
        cands = engine.invert(it.old_response, n_candidates=n_candidates)
        it.rpe_candidates = cands
        behaviour = make_behaviour(it.bid, it.prompt)
        details, admitted, recovered = [], [], None
        for i, c in enumerate(cands):
            variant = make_variant(it.bid, c, i)
            ok, scores = gate_admits(gate, behaviour, variant)
            entry = {"text": c, "admitted": ok, "scores": scores, "response": None, "broke": None}
            if ok:
                admitted.append(c)
                resp = new_model.respond(c)
                rtext = resp.text if hasattr(resp, "text") else str(resp)
                broke = (judge.classify(c, resp) == 1)
                entry["response"] = rtext
                entry["broke"] = broke
                recovered = 1 if (recovered == 1 or broke) else 0
            details.append(entry)
        it.rpe_admitted = admitted
        it.rpe_details = details
        it.rpe_recovered = recovered
        if verbose:
            tag = {1: "RECOVERED", 0: "held", None: "unresolved"}[it.rpe_recovered]
            print(f"  [{bench} {k+1}/{len(items)}] {it.bid}: {len(cands)} cand, "
                  f"{len(admitted)} admitted -> {tag}")
    return items


def _report_to_dict(r):
    return dict(n_total=r.n_total, n_both_resolved=r.n_both_resolved,
        rsr_para=r.rsr_para, rsr_rpe=r.rsr_rpe,
        rsr_para_ci=list(r.rsr_para_ci), rsr_rpe_ci=list(r.rsr_rpe_ci),
        agreement_rate=r.agreement_rate, cohen_kappa=r.cohen_kappa,
        both_recovered=len(r.both_recovered), both_held=len(r.both_held),
        rpe_only=len(r.rpe_only), para_only=len(r.para_only),
        para_unresolved=r.para_unresolved, rpe_unresolved=r.rpe_unresolved)


def main():
    ap = argparse.ArgumentParser(description="RPE independent-validator probe")
    ap.add_argument("--old", required=True)
    ap.add_argument("--new-model", required=True)
    ap.add_argument("--rpe-model", required=True)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--behaviours", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--recycle-dir", type=Path, default=None)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/real_llama_lineage.yaml"))
    ap.add_argument("--old-cache-name", default=None)
    ap.add_argument("--new-cache-name", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-candidates", type=int, default=5)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg, new_model, engine, judge, gate = build_real_components(
        args.config, args.new_model, args.rpe_model)

    _md_path = Path(__file__).resolve().parents[0] / "make_dashboard.py"
    _spec = _ilu.spec_from_file_location("_md", _md_path)
    _md = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_md)

    from recyclebench.backends.providers import load_provider_model
    old_model = load_provider_model(args.old, "openai_compatible", cfg)
    old_cache_name = args.old_cache_name or old_model.name
    new_cache_name = args.new_cache_name or new_model.name

    k = cfg.budget.k_canonical_samples
    cache = _md.load_cache(str(args.cache))
    cells, per_bench = _md.partition(cache, [str(c) for c in args.behaviours],
                                     old_cache_name, new_cache_name, k)
    _npm = len(cells.get("pminus", []))
    print(f"  [partition] {_npm} P- records (old='{old_cache_name}', new='{new_cache_name}', k={k})")
    if _npm == 0:
        print("  [!] 0 P- -- set --old-cache-name/--new-cache-name to the exact strings the cache used.")

    all_reports = {}
    for bench in args.benchmarks:
        print(f"\n=== {bench} ===")
        restrict = None
        if args.run_dir is not None:
            pj = args.run_dir / f"pminus_{bench}.json"
            if pj.exists():
                restrict = load_pminus_bids(pj) or None
                if restrict:
                    print(f"  [filter] restricting to {len(restrict)} p_minus_bids from {pj.name}")
        recycle_manifest = None
        if args.recycle_dir is not None:
            rm = args.recycle_dir / f"manifest_{bench}.jsonl"
            recycle_manifest = rm if rm.exists() else None
            if recycle_manifest is None:
                print(f"  [warn] no paraphrase manifest for {bench} in {args.recycle_dir}")
        items = load_probe_items_from_partition(
            bench, cells, recycle_manifest=recycle_manifest, restrict_bids=restrict)
        if not items:
            print(f"  [skip] {bench}: 0 P- behaviours (check CSV stem == '{bench}')")
            continue
        print(f"  {len(items)} P- behaviours to probe")
        items = run_one_benchmark(bench, items, engine=engine, new_model=new_model,
                                  judge=judge, gate=gate, n_candidates=args.n_candidates,
                                  verbose=not args.quiet)
        out_manifest = args.out / f"rpe_manifest_{bench}.jsonl"
        write_manifest(items, out_manifest)
        report = analyze(items); all_reports[bench] = report
        print("\n" + report.summary())
        print(f"\nwrote {out_manifest}")

    rollup = {b: _report_to_dict(r) for b, r in all_reports.items()}
    (args.out / "rpe_agreement.json").write_text(json.dumps(rollup, indent=2))
    print(f"\nwrote {args.out / 'rpe_agreement.json'}")


if __name__ == "__main__":
    main()
