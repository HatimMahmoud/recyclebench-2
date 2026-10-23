#!/usr/bin/env python3
"""rpe_repair.py -- rebuild the convergent-validity comparison between the recycle
(paraphrase) arm and the independent-probe (RPE / output-inversion) arm.

Why this exists
---------------
Every stored `rpe_agreement.json` is invalid, for four separate reasons:

  1. the paraphrase side was read with `parse_recycle_manifest`, which hunted for a
     `recovered`/`bypassed` field. A `run-real` manifest has no such field -- it holds
     the frozen variants and the gate audit -- so every behaviour defaulted to 0 and
     one side of the comparison was a constant;
  2. the RPE run was anchored on the OFFLINE-rule P-minus (61 behaviours), while the
     headline anchor uses the qwen-judged P-minus (49). The arms were not even
     compared on the same behaviours;
  3. the RPE candidates were judged by the target model (circular);
  4. no test of the difference in re-opening *rate* was reported, only kappa.

This tool rebuilds both sides from first principles -- frozen manifest + response
cache + judge verdict store, through `lineage_core.observe` and the package
estimators -- and reports the coverage it actually had, so a thin result reads as
thin rather than as a confident zero.

Verdict algebra is three-valued throughout (`lineage_core._any`): 1 if any admitted
variant complied, 0 only if every admitted variant has a verdict and none complied,
None otherwise. A behaviour with no admitted variant is *unresolved*, not robust.

Read-only: writes a single dated JSON, never overwrites, makes no model calls.
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

import lineage_core as lc                                         # noqa: E402
from recyclebench.config import Config                            # noqa: E402
from recyclebench.datasets import load_jailbreakbench             # noqa: E402
from recyclebench.independent_probe.analyze_agreement import analyze  # noqa: E402
from recyclebench.independent_probe.contract import ProbeItem     # noqa: E402
from recyclebench.manifest import Manifest                        # noqa: E402
from recyclebench.stats import wilson_interval                    # noqa: E402


# ---------------------------------------------------------------------------
# the two arms
# ---------------------------------------------------------------------------
def para_side(cfg, behaviours, manifest, cache, verdicts, old, new):
    """-> (p_minus, {bid: 0/1/None}, coverage) rebuilt exactly as the recycle arm scores it."""
    obs = lc.observe(cfg, behaviours, manifest, cache, verdicts, [old, new])
    runs = {m: lc.to_model_run(obs[m], behaviours) for m in (old, new)}
    part = lc.partition(runs[old], runs[new])
    p_minus = list(part["pminus"])

    recovered, complete = {}, 0
    for bid in p_minus:
        vs = obs[new].variants.get(bid, [])
        if not vs:
            recovered[bid] = None                 # no admitted variant => unresolved
            continue
        recovered[bid] = lc._any([v.verdict for v in vs])
        complete += obs[new].recovery_complete(bid)
    cov = {
        "n_p_minus": len(p_minus),
        "behaviours_fully_judged": complete,
        "missing_variant_verdicts": obs[new].missing_verdicts["variant"],
        "missing_responses": obs[new].missing_responses,
        "partition": {k: len(v) for k, v in part.items()},
    }
    return p_minus, recovered, cov


def rpe_side(rpe_manifest: Path, verdicts, use_stored: bool = False):
    """-> ({bid: 0/1/None}, {bid: [candidate rows]}, coverage) from the probe manifest.

    Re-judges each admitted candidate's stored reply through `verdicts` rather than
    trusting the manifest's own `broke`, which came from the target model itself.
    """
    recovered, cands, missing, n_adm = {}, {}, 0, 0
    for line in Path(rpe_manifest).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        bid = rec["bid"]
        rows, vals = [], []
        for d in rec.get("details") or []:
            if not d.get("admitted") or d.get("response") is None:
                rows.append({"text": d.get("text"), "admitted": bool(d.get("admitted")),
                             "verdict": None, "scores": d.get("scores")})
                continue
            n_adm += 1
            v = d.get("broke") if use_stored else verdicts.get(d["text"], d["response"])
            if v is None:
                missing += 1
            vals.append(v)
            rows.append({"text": d["text"], "admitted": True, "verdict": v,
                         "scores": d.get("scores")})
        cands[bid] = rows
        recovered[bid] = lc._any(vals) if vals else None     # nothing admitted => unresolved
    cov = {"n_behaviours": len(recovered), "n_admitted_candidates": n_adm,
           "missing_candidate_verdicts": missing,
           "n_no_admitted": sum(1 for v in recovered.values() if v is None)}
    return recovered, cands, cov


def provenance(cfg, behaviours, manifest, cache, verdicts, old, new, rpe_bids):
    """Do the two arms even cover the same behaviours?

    The frozen recycle manifest names the behaviours the paraphrase arm ran on; the
    probe manifest names the ones the inversion arm ran on. If those sets differ, no
    agreement statistic between the arms means anything, however it is computed --
    so this is checked before any rate is reported.
    """
    obs = lc.observe(cfg, behaviours, manifest, cache, verdicts, [old, new])
    runs = {m: lc.to_model_run(obs[m], behaviours) for m in (old, new)}
    part = lc.partition(runs[old], runs[new])
    man_bids, pm = set(manifest.admitted), set(part["pminus"])
    cell = {k: len(set(v) & man_bids) for k, v in part.items()}
    return {
        "recycle_manifest_bids": len(man_bids),
        "rpe_manifest_bids": len(rpe_bids),
        "overlap_recycle_rpe": len(man_bids & set(rpe_bids)),
        "rebuilt_p_minus": len(pm),
        "rpe_bids_in_rebuilt_p_minus": len(set(rpe_bids) & pm),
        "recycle_manifest_bids_in_rebuilt_p_minus": len(man_bids & pm),
        "recycle_manifest_bids_by_partition_cell_today": cell,
        "old_canonical_verdict_on_recycle_manifest_bids": {
            "complied": sum(1 for b in man_bids if obs[old].canonical.get(b) == 1),
            "refused": sum(1 for b in man_bids if obs[old].canonical.get(b) == 0),
            "unresolved": sum(1 for b in man_bids if obs[old].canonical.get(b) is None)},
        "comparable": len(man_bids & set(rpe_bids) & pm),
    }


def _rate(d, keys):
    vals = [d.get(b) for b in keys]
    res = [v for v in vals if v is not None]
    if not res:
        return {"k": 0, "n": 0, "rate": None, "ci": [None, None], "unresolved": len(vals)}
    k, n = sum(res), len(res)
    iv = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci": [iv.low, iv.high], "unresolved": len(vals) - n}


def compare(label, p_minus, para, rpe, para_cov, rpe_cov):
    """Join the two arms on the P-minus behaviours only and run the agreement analysis."""
    shared = [b for b in p_minus if b in rpe]
    items = []
    for b in shared:
        it = ProbeItem(bid=b, benchmark="sorrybench", prompt="", old_response="")
        it.para_recovered, it.rpe_recovered = para.get(b), rpe.get(b)
        items.append(it)
    rep = analyze(items)
    return {
        "judge": label,
        "p_minus": len(p_minus),
        "p_minus_covered_by_rpe_run": len(shared),
        "p_minus_missing_from_rpe_run": sorted(set(p_minus) - set(rpe)),
        "RSR_paraphrase": _rate(para, shared),
        "PRR_rpe": _rate(rpe, shared),
        "agreement": {
            "n_both_resolved": rep.n_both_resolved,
            "rate": rep.agreement_rate,
            "cohen_kappa": rep.cohen_kappa,
            "mcnemar_p": rep.mcnemar_p,
            "both_recovered": rep.both_recovered,
            "both_held": rep.both_held,
            "rpe_only": rep.rpe_only,
            "para_only": rep.para_only,
        },
        "coverage": {"paraphrase": para_cov, "rpe": rpe_cov},
    }


def fmt(r) -> str:
    def pc(x):
        return "undefined" if x is None else f"{x*100:.1f}%"

    def ci(c):
        return "" if c[0] is None else f" [{c[0]*100:.1f}, {c[1]*100:.1f}]"
    a, p, q = r["agreement"], r["RSR_paraphrase"], r["PRR_rpe"]
    L = [
        f"=== {r['judge']}",
        f"  P^-                        : {r['p_minus']}  "
        f"(covered by the RPE run: {r['p_minus_covered_by_rpe_run']})",
        f"  RSR  paraphrase arm        : {pc(p['rate'])}{ci(p['ci'])}   "
        f"{p['k']}/{p['n']}   unresolved {p['unresolved']}",
        f"  PRR  inversion arm         : {pc(q['rate'])}{ci(q['ci'])}   "
        f"{q['k']}/{q['n']}   unresolved {q['unresolved']}",
        f"  resolved by BOTH arms      : {a['n_both_resolved']}",
        f"  agreement / kappa          : {pc(a['rate'])} / "
        + ("n/a" if a["cohen_kappa"] is None else f"{a['cohen_kappa']:.3f}"),
        f"  McNemar exact p            : "
        + ("n/a" if a["mcnemar_p"] is None else f"{a['mcnemar_p']:.4g}"),
        f"  cells  both={len(a['both_recovered'])}  neither={len(a['both_held'])}  "
        f"rpe-only={len(a['rpe_only'])}  para-only={len(a['para_only'])}",
        f"  coverage: {r['coverage']['paraphrase']['missing_variant_verdicts']} variant verdicts "
        f"missing of the paraphrase arm; {r['coverage']['rpe']['missing_candidate_verdicts']} "
        f"candidate verdicts missing of {r['coverage']['rpe']['n_admitted_candidates']} admitted",
    ]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults are the LINEAGE run, not the anchor run: the anchor manifest
    # (results/runs/recycle_3_31_sorrybench/manifest_sorrybench.jsonl) names 61
    # behaviours that overlap the rebuilt P-minus by only 8, and on which the OLD
    # model refuses 52 of 61 under the current cache -- so it cannot be joined to
    # anything. The lineage manifest covers all 440 behaviours, overlaps the probe
    # run 61/61, and has a complete qwen verdict set. Pass --manifest to override.
    ap.add_argument("--config", default="configs/real_llama_lineage_all.yaml")
    ap.add_argument("--behaviours", default="data/sorrybench.csv")
    ap.add_argument("--manifest",
                    default="results/runs/lineage_sorrybench_20261020/manifest.jsonl")
    ap.add_argument("--cache", default="cache/llama_gen.jsonl")
    ap.add_argument("--verdicts", default="cache/judge_lineage_sorrybench.jsonl")
    ap.add_argument("--rpe-verdicts", default="cache/judge_rpe_qwen.jsonl")
    ap.add_argument("--rpe-manifest",
                    default="results/runs/rpe_3_31_loosegate/out/rpe_manifest_sorrybench.jsonl")
    ap.add_argument("--old", default="llama3-8b")
    ap.add_argument("--new", default="llama31-8b")
    ap.add_argument("--out", default=None, help="default: results/analysis_rpe_<stamp>/rpe_repair.json")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    tag = lc.judge_tag_from_cfg(cfg)
    behaviours = load_jailbreakbench(args.behaviours)
    manifest = Manifest.load(args.manifest)
    cache = lc.load_response_cache(args.cache)
    print(f"pair     : {args.old} -> {args.new}")
    print(f"judge    : {tag}")
    print(f"manifest : {args.manifest}  ({len(manifest.admitted)} behaviours, "
          f"{sum(len(v) for v in manifest.admitted.values())} admitted variants)")
    print(f"rpe      : {args.rpe_manifest}\n")

    # the qwen store covers the two arms through two different caches
    qwen_para = lc.StoredVerdicts(lc.VerdictStore(args.verdicts), tag)
    qwen_rpe = lc.StoredVerdicts(lc.VerdictStore(args.rpe_verdicts), tag)
    offline = lc.OfflineVerdicts()

    out = {"generated": _dt.datetime.now().strftime("%Y%m%d_%H%M"),
           "pair": [args.old, args.new], "judge_tag": tag,
           "inputs": {k: getattr(args, k) for k in
                      ("config", "behaviours", "manifest", "cache", "verdicts",
                       "rpe_verdicts", "rpe_manifest")},
           "views": []}

    rpe_bids = [json.loads(l)["bid"] for l in
                Path(args.rpe_manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    prov = provenance(cfg, behaviours, manifest, cache, offline, args.old, args.new, rpe_bids)
    out["provenance"] = prov
    print("--- provenance: do the arms cover the same behaviours?")
    for k, v in prov.items():
        print(f"    {k:<48} {v}")
    if prov["comparable"] < 20:
        print(f"\n!!! the two arms share only {prov['comparable']} behaviours in the rebuilt P^-.\n"
              "    No agreement statistic between them is interpretable. Both arms must be\n"
              "    re-run over ONE frozen behaviour set before this comparison exists.\n")

    for label, pv, rv in [(f"{tag} (judged)", qwen_para, qwen_rpe),
                          ("offline refusal rule", offline, offline)]:
        p_minus, para, pcov = para_side(cfg, behaviours, manifest, cache, pv, args.old, args.new)
        rpe, cands, rcov = rpe_side(Path(args.rpe_manifest), rv)
        r = compare(label, p_minus, para, rpe, pcov, rcov)
        r["per_behaviour"] = {b: {"para": para.get(b), "rpe": rpe.get(b)} for b in p_minus}
        out["views"].append(r)
        print(fmt(r), "\n")

    # what the stored (circular, target-as-judge) numbers claimed, for the record
    _, _, scov = rpe_side(Path(args.rpe_manifest), offline, use_stored=True)
    out["stored_target_as_judge_coverage"] = scov

    stamp = out["generated"]
    dest = Path(args.out) if args.out else ROOT / "results" / f"analysis_rpe_{stamp}" / "rpe_repair.json"
    if dest.exists():
        raise SystemExit(f"{dest} exists -- refusing to overwrite")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote    : {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
