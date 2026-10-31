#!/usr/bin/env python3
"""tools/thesis_numbers.py -- one generated source for every figure the thesis cites.

Reads the registered result files (never recomputes an estimate a result file already holds) and writes
  results/thesis_numbers_<stamp>.json   flat {name: value} with the source file of each value
  results/thesis_numbers_<stamp>.tex    \\newcommand{\\rbName}{value} for every entry
so the text never carries a hand-copied number. Re-run after any audit sheet is labelled; an audit that is not yet
labelled yields the macro text "[pending]". Fails loudly when a registered source is missing.

  python3 tools/thesis_numbers.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

PENDING = "[pending]"
SOURCES = dict(
    anchor="results/runs/recycle_3_31_sorrybench/corrected_sorrybench_recal_n60.json",
    heldout="results/runs/confirm_heldout_3_31/corrected_heldout_recal_n60.json",
    lineage="results/runs/lineage_sorrybench_20261020/summary_real.json",
    gate="audit/gate/gate_audit_result.json",
    rpe="results/analysis_rpe_audit_20261029_1849/audit.json",
)
LATEST = dict(  # newest dated output of each analysis tool
    analysis="results/analysis_2*/analysis.json",
    c4="results/analysis_c4_*/c4.json",
    power="results/analysis_power_*/power.json",
    compose_anchor="results/runs/compose_v2_2*/compose_v2_summary.json",
    compose_second="results/runs/compose_v2_llama2-7b__to__llama31-8b_*/compose_v2_summary.json",
)


def _load(rel):
    p = ROOT / rel
    if not p.exists():
        raise SystemExit(f"missing registered source: {rel}")
    return json.loads(p.read_text())


def _latest(pattern):
    c = sorted(ROOT.glob(pattern))
    if not c:
        raise SystemExit(f"no output matches {pattern} -- run its tool first")
    return c[-1]


def _half_up(x, d):
    """round half away from zero on the value as stored (0.7645 -> 0.765), not on its binary float"""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def pct(x, d=1):
    return f"{_half_up(100 * x, d)}\\%"


def num(x, d=3):
    return _half_up(x, d)


def ci(lo, hi, d=3):
    return f"[{_half_up(lo, d)}, {_half_up(hi, d)}]"


def pval(p):
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def anchor_unresolved():
    """fixes of the anchor run's judged P⁻ with no admitted rewording: rebuild the run from the cache with the stored
    qwen verdicts (reproduces P⁻ = 47), no model calls"""
    import lineage_core as lc
    from recyclebench.config import Config
    from recyclebench.datasets import load_jailbreakbench
    from recyclebench.manifest import Manifest
    from recyclebench.screening import regression_set
    cfg = Config.load(ROOT / "configs/recycle_3_31.yaml")
    beh = load_jailbreakbench(ROOT / "results/cells_llama3-8b_llama31-8b/sorrybench_p_minus.csv")
    man = Manifest.load(ROOT / "results/runs/recycle_3_31_sorrybench/manifest_sorrybench.jsonl")
    cache = lc.load_response_cache(ROOT / "cache/llama_gen.jsonl")
    ver = lc.StoredVerdicts(lc.VerdictStore(ROOT / "cache/judge_lineage_sorrybench.jsonl"), lc.judge_tag_from_cfg(cfg))
    _, runs = lc.build_runs(cfg, beh, man, cache, ver, ["llama3-8b", "llama31-8b"])
    pm = regression_set(runs["llama3-8b"], runs["llama31-8b"])
    if len(pm) != 47:
        raise SystemExit(f"anchor rebuild gives P- {len(pm)}, not the stored 47 -- refusing to derive the resolved figure")
    return [b for b in pm if runs["llama31-8b"].recovery[b].n_admitted == 0]


class Numbers:
    def __init__(self):
        self.v, self.src = {}, {}

    def put(self, name, value, src):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro names are letters only: {name}")
        if name in self.v:
            raise ValueError(f"duplicate name {name}")
        self.v[name], self.src[name] = value, src


def collect() -> Numbers:
    N = Numbers()
    # anchor (dedicated recycle run, 60-bundle calibration)
    a = _load(SOURCES["anchor"]); s = SOURCES["anchor"]
    r, j = a["RSR"], a["judge"]
    N.put("AnchorPminus", str(r["n_p_minus"]), s)
    N.put("AnchorK", str(r["k_recovered"]), s)
    N.put("AnchorRSRraw", num(r["RSR_raw"]), s)
    N.put("AnchorRSRrawCI", ci(*r["raw_wilson95"]), s)
    N.put("AnchorRSRcorr", num(r["RSR_corrected"]), s)
    N.put("AnchorRSRcorrCI", ci(*r["CI"]), s)
    N.put("AnchorMcNemar", pval(r["mcnemar_p"]), s)
    # registered RSR convention (Amendment 2 §B3): unresolved fixes leave the denominator
    unres = anchor_unresolved()
    import study_data as sd
    from recyclebench.stats import corrected_interval, wilson_interval
    jc = sd.anchor_judge_chars(sd.judge_calibrations())   # the same calibration the stored file used
    n_res = r["n_p_minus"] - len(unres)
    cr, wr = corrected_interval(r["k_recovered"], n_res, jc), wilson_interval(r["k_recovered"], n_res)
    N.put("AnchorUnresolved", str(len(unres)), "rebuilt from cache + stored qwen verdicts")
    N.put("AnchorResolvedKn", f"{r['k_recovered']}/{n_res}", s)
    N.put("AnchorResolvedRSRraw", num(r["k_recovered"] / n_res), s)
    N.put("AnchorResolvedRSRrawCI", ci(wr.low, wr.high), s)
    N.put("AnchorResolvedRSRcorr", num(cr.point), s)
    N.put("AnchorResolvedRSRcorrCI", ci(cr.low, cr.high), s)
    p = a["PBR_new"]
    N.put("AnchorPBRraw", num(p["PBR_raw"]), s)
    N.put("AnchorPBRk", f"{p['k_bypassed']}/{p['n_resolved']}", s)
    N.put("AnchorPBRcorr", num(p["PBR_corrected"]), s)
    N.put("AnchorPBRcorrCI", ci(*p["CI"]), s)
    N.put("JudgeSe", num(j["se"]), s); N.put("JudgeSeCI", ci(*j["se_ci"]), s)
    N.put("JudgeSp", num(j["sp"]), s); N.put("JudgeSpCI", ci(*j["sp_ci"]), s)
    N.put("JudgeKappa", num(j["kappa"]), s); N.put("JudgeBundles", str(j["n_bundles"]), s)
    # confirmatory replication
    h = _load(SOURCES["heldout"]); s = SOURCES["heldout"]
    r, p = h["RSR"], h["PBR_new"]
    N.put("ConfKn", f"{r['k']}/{r['n']}", s)
    N.put("ConfRSRraw", num(r["raw"]), s); N.put("ConfRSRrawCI", ci(*r["raw_wilson95"]), s)
    N.put("ConfRSRcorr", num(r["corrected"]), s); N.put("ConfRSRcorrCI", ci(*r["corrected_ci95_delta"]), s)
    N.put("ConfMcNemar", pval(r["mcnemar_p"]), s); N.put("ConfVerdict", r["verdict"], s)
    N.put("ConfPBRkn", f"{p['k']}/{p['n']}", s); N.put("ConfPBRcorr", num(p["corrected"]), s)
    N.put("ConfPBRcorrCI", ci(*p["corrected_ci95_delta"]), s); N.put("ConfPBRverdict", p["verdict_reported"], s)
    # gate audit (D5)
    g = _load(SOURCES["gate"]); s = SOURCES["gate"]
    c = g["contamination_R"]
    N.put("GateChangedR", f"{c['k']}/{c['n']}", s); N.put("GateChangedRrate", pct(c["rate"]), s)
    N.put("GateChangedRCI", ci(100 * c["lo"], 100 * c["hi"], 1), s)
    N.put("GateChangedWeighted", pct(g["contamination_admitted_weighted"], 2), s)
    N.put("GateAuditLabelled", str(g["labelled"]), s)
    # gate false rejection (compose v2 singles) and benign judge check -- pending until labelled
    import gate_rejection_audit as gra
    import overrefusal_judge_audit as oja
    fr = gra.score(ROOT / "audit/gate_rejection"); s = "audit/gate_rejection"
    done = fr and fr["labelled"] == fr["of"] and "false_rejection_P(rejected|preserved)" in fr
    N.put("GateFalseRejection", pct(fr["false_rejection_P(rejected|preserved)"]["rate"]) if done else PENDING, s)
    N.put("GateFalseRejectionCI", ci(*[100 * x for x in fr["false_rejection_P(rejected|preserved)"]["ci"]], 1) if done else PENDING, s)
    bj = oja.score(ROOT / "audit/overrefusal"); s = "audit/overrefusal"
    done = bj and bj["labelled"] == bj["of"] and "judge_on_benign" in bj
    N.put("BenignJudgeSe", num(bj["judge_on_benign"]["se"]) if done else PENDING, s)
    N.put("BenignJudgeSp", num(bj["judge_on_benign"]["sp"]) if done else PENDING, s)
    relabel = ROOT / "audit/sorrybench/judge_sheet_blind_relabel_20261030.csv"
    import csv
    rows = list(csv.DictReader(relabel.open()))
    n_lab = sum(1 for r in rows if (r.get("human_fulfilled") or "").strip() in ("0", "1"))
    N.put("BlindRelabelProgress", f"{n_lab}/{len(rows)}", str(relabel.relative_to(ROOT)))
    # lineage + C4 + power + pooled + RQ3 + RPE + compose
    c4p = _latest(LATEST["c4"]); c4 = json.loads(c4p.read_text()); s = str(c4p.relative_to(ROOT))
    names = {"llama2-7b -> llama2-13b": "LinA", "llama2-13b -> llama3-8b": "LinB", "llama3-8b -> llama31-8b": "LinC",
             "llama31-8b -> llama32-1b": "LinD", "llama32-1b -> llama32-3b": "LinE"}
    for p in c4["pairs"]:
        k, b, n0 = names[p["pair"]], p["judge"], p["naive"]
        N.put(f"{k}Pminus", str(b["p_minus"]), s); N.put(f"{k}Kn", f"{b['k']}/{b['n']}", s)
        N.put(f"{k}RSRcorr", num(b["corrected"]), s); N.put(f"{k}NaiveKn", f"{n0['k']}/{n0['n']}", s)
    mnames = {"llama2-7b": "TwoSeven", "llama2-13b": "TwoThirteen", "llama3-8b": "ThreeEight", "llama31-8b": "ThreeOneEight",
              "llama32-1b": "ThreeTwoOne", "llama32-3b": "ThreeTwoThree"}
    for m in c4["models"]:
        k = mnames[m["model"]]
        N.put(f"PBR{k}Naive", num(m["naive"]["raw"]), s); N.put(f"PBR{k}Raw", num(m["judge"]["raw"]), s)
        N.put(f"PBR{k}Corr", num(m["judge"]["corrected"]), s)
    pw = _latest(LATEST["power"]); P = json.loads(pw.read_text()); s = str(pw.relative_to(ROOT))
    words = {9: "Nine", 23: "TwentyThree", 49: "FortyNine", 82: "EightyTwo"}
    N.put("McNemarKmin", str(P["by_n"][0]["mcnemar_k_min"]), s)
    for r in P["by_n"]:
        if r["n"] in words:
            w = words[r["n"]]
            N.put(f"PowerSBk{w}", str(r["surface_bound_k_min"]), s)
            N.put(f"PowerMDE{w}", num(r["mde_surface_bound"], 2) if r["mde_surface_bound"] is not None else "--", s)
    an = _latest(LATEST["analysis"]); A = json.loads(an.read_text()); s = str(an.relative_to(ROOT))
    po = A["headline"]["pooled"]
    N.put("PooledKn", f"{po['k']}/{po['n_pminus']}", s); N.put("PooledCorr", num(po["corrected"]), s)
    N.put("PooledCorrCI", ci(po["corrected_lo"], po["corrected_hi"]), s)
    rq3 = {t["term"]: t for t in A["rq3_drivers"]["anchor_pooled"]["table"]}
    for term, nm in (("C(transform_class)[T.paraphrase]", "Paraphrase"), ("C(transform_class)[T.register]", "Register")):
        if term in rq3:
            N.put(f"RQThreeOR{nm}", num(rq3[term]["odds_ratio"], 2), s)
    rp = _load(SOURCES["rpe"]); s = SOURCES["rpe"]
    pr = next(x for x in rp["harm"]["pairs"] if x["pair"] == "llama3-8b->llama31-8b" and x["benchmark"] == "sorrybench")
    N.put("RPEPRRkn", f"{pr['lower']['k']}/{pr['lower']['n']}", s)
    N.put("RPEHarmKept", f"{rp['harm']['candidate_preserved']['k']}/{rp['harm']['candidate_preserved']['n']}", s)
    for key, nm in (("compose_anchor", "ComposeA"), ("compose_second", "ComposeB")):
        cp = _latest(LATEST[key]); C = json.loads(cp.read_text()); s = str(cp.relative_to(ROOT))
        N.put(f"{nm}MinHolm", num(min(x["p_holm"] for x in C["primary_pair_vs_either_single"]), 2), s)
        N.put(f"{nm}Fixes", str(C["p_minus"]), s)
    # 2026-10-31 controls
    import controls_section as ctl
    agg = ctl.aggregate(ROOT)
    o = agg["overrefusal"]; s = str(o["path"].relative_to(ROOT))
    for m, k in mnames.items():
        if m in o["complete"]:
            N.put(f"OR{k}", pct(o["data"]["models"][m]["safe_refusal_judged"]["rate"]), s)
    anc = ctl._pair_line(o["data"], "llama3-8b", "llama31-8b")
    N.put("ORAnchorSwap", f"{anc['newly_refused']}/{anc['newly_answered']}", s)
    N.put("ORAnchorP", pval(ctl.mc_p(anc)), s)
    x = agg["xfamily"]; s = str(x["dir"].relative_to(ROOT))
    N.put("XfamPminus", str(x["screen"]["p_minus"]), s); N.put("XfamNegFlips", str(x["screen"]["negative_flips"]), s)
    rr = x["summary"]["RSR_primary_RAW"][0]
    N.put("XfamKn", f"{rr['k_recovered']}/{rr['n_p_minus']}", s)
    ct = agg["contamination"]["data"]; s = str(agg["contamination"]["path"].relative_to(ROOT))
    for m, k in (("llama3-8b", "ThreeEight"), ("llama31-8b", "ThreeOneEight"), ("llama2-7b", "TwoSeven")):
        h = ct["models"][m]["harmbench"]
        N.put(f"ContamHarm{k}", f"{h['reproduced']}/{h['n']}", s)
    return N


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="stem; default results/thesis_numbers_<stamp>")
    args = ap.parse_args(argv)
    N = collect()
    stem = Path(args.out) if args.out else ROOT / "results" / f"thesis_numbers_{_dt.datetime.now():%Y%m%d_%H%M}"
    if stem.with_suffix(".json").exists():
        raise SystemExit(f"{stem}.json exists -- refusing to overwrite")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(dict(created=_dt.datetime.now().isoformat(timespec="seconds"),
                                                         values=N.v, sources=N.src), indent=1))
    tex = [f"% generated by tools/thesis_numbers.py on {_dt.datetime.now():%Y-%m-%d %H:%M} -- do not edit by hand"]
    tex += [f"\\newcommand{{\\rb{k}}}{{{v}}}  % {N.src[k]}" for k, v in N.v.items()]
    stem.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    pend = [k for k, v in N.v.items() if v == PENDING]
    print(f"wrote {stem.relative_to(ROOT)}.{{json,tex}} · {len(N.v)} values · pending: {', '.join(pend) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
