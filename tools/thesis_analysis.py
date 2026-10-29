#!/usr/bin/env python3
"""tools/thesis_analysis.py -- the exploratory analyses of prereg Amendment 1 §A3, from cache + frozen manifests.

No model calls. Every rate goes through the package estimators (compute_rsr, compute_pbr, wilson_interval,
corrected_interval, mcnemar_exact, fit_drivers), so the numbers are the ones the instrument defines.

  0  CONFIRMATORY replication        Amendment 1 §A2, read from results/runs/confirm_heldout_3_31/ and reported on its
                                    own -- the only confirmatory figure here, never pooled with 1..6 below
  1  pooled matched-scale headline   llama3-8b -> llama3.1-8b over SORRY-Bench, HarmBench, AdvBench, StrongREJECT
  2  RQ2 per harm category / domain  (descriptive; exploratory by registration §8)
  3  RQ3 drivers                     mixed-effects logit of per-variant recovery on transform class + surface distance
  4  sensitivity                     gate thresholds tightened +0.05/+0.10 (loosening needs the rejected texts, which
                                     the manifests do not keep), budget t in {1, 2, 4, 8}, judge (qwen vs offline rule)
  5  PBR backbone                    per model, only where every needed verdict is stored
  6  inversion arm                   the current closed-loop runs per benchmark x pair: PRR under the intent gate,
                                     the closed loop and both, beside RSR on the same judged P^-, plus the
                                     harmful-ask audit bracket (tools/inversion_section.load); then the early RPE
                                     runs' agreement with paraphrase recovery, joined on the prompt text (superseded)

Writes results/analysis_<stamp>/analysis.json and analysis.md (never overwrites).

  python3 tools/thesis_analysis.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_core as lc                                          # noqa: E402
import study_data as sd                                            # noqa: E402
from recyclebench.config import Config                             # noqa: E402
from recyclebench.datasets import load_jailbreakbench              # noqa: E402
from recyclebench.drivers import fit_drivers                       # noqa: E402
from recyclebench.judge import looks_like_refusal                  # noqa: E402
from recyclebench.manifest import Manifest                         # noqa: E402
from recyclebench.pbr import compute_pbr                           # noqa: E402
from recyclebench.recycling import compute_rsr                     # noqa: E402
from recyclebench.stats import corrected_interval, mcnemar_exact, wilson_interval  # noqa: E402

ROOT = sd.ROOT
CFG_PATH = "configs/real_llama_lineage_all.yaml"
OLD, NEW = "llama3-8b", "llama31-8b"
SMOKE = "results/runs/20260805_124501_llama3-8b_to_llama31-8b_RECYCLE_qwen3b_SMOKE/manifest_harmbench.jsonl"
L3 = "results/runs/20260805_025658_llama3-8b_to_llama31-8b_RECYCLE"
SORRY_MAN = "results/runs/lineage_sorrybench_20261020/manifest.jsonl"
# benchmark -> (csv, manifest, verdict store or None = offline rule, models screened on that manifest)
SOURCES = {
    "sorrybench": ("data/sorrybench.csv", SORRY_MAN, "cache/judge_lineage_sorrybench.jsonl", None),
    "harmbench": ("data/harmbench.csv", SMOKE, "cache/judge_qwen_harmbench.jsonl",
                  ["llama2-7b", "llama2-13b", "llama3-8b", "llama31-8b"]),
    "advbench": ("data/advbench.csv", f"{L3}/manifest_advbench.jsonl", "cache/judge_qwen_advbench.jsonl",
                 ["llama2-7b", "llama2-13b", "llama3-8b", "llama31-8b"]),
    "strongreject": ("data/strongreject.csv", f"{L3}/manifest_strongreject.jsonl", "cache/judge_qwen_strongreject.jsonl",
                     ["llama2-7b", "llama2-13b", "llama3-8b", "llama31-8b"]),
}
STRICT = sd.STRICT_GATE
INTENT_MODEL = "gemma4:e4b"
# The CONFIRMATORY replication (Amendment 1 §A2). Read, never recomputed here and never pooled with
# §1-§6: those are exploratory, and mixing a registered result into them is what the registration
# exists to prevent. The recalibrated file supersedes the original (see its own "why" field).
CONFIRM_DIR = "results/runs/confirm_heldout_3_31"
CONFIRM_FILES = ["corrected_heldout_recal_n60.json", "corrected_heldout.json"]


def wil(k, n):
    w = wilson_interval(k, n)
    return dict(k=k, n=n, rate=(k / n if n else None), lo=w.low, hi=w.high)


def surface_distance(a: str, b: str) -> float:
    return 1.0 - difflib.SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio()


class Source:
    """One benchmark: behaviours, frozen manifest, gate scores, and observations under a verdict source."""

    def __init__(self, cfg, cache, tag, bench, judge="primary"):
        csv_p, man_p, vstore, models = SOURCES[bench]
        self.bench, self.cfg = bench, cfg
        self.behaviours = load_jailbreakbench(str(ROOT / csv_p))
        self.bmap = {b.bid: b for b in self.behaviours}
        self.manifest = Manifest.load(ROOT / man_p)
        self.audit = {(a.bid, a.index): a for a in self.manifest.audit}
        self.models = models or list(cfg.lineage)
        use_store = judge != "offline" and vstore and (ROOT / vstore).exists()
        self.judge = "qwen2.5:14b" if use_store else "offline rule"
        if use_store:
            store = lc.VerdictStore(ROOT / vstore)
            self.obs, self.runs = lc.build_runs(cfg, self.behaviours, self.manifest, cache, lc.StoredVerdicts(store, tag),
                                                self.models)
            store.close()
            if any(self.obs[m].missing_verdicts["canonical"] for m in self.models):   # incomplete store -> offline, labelled
                use_store = False
                self.judge = "offline rule (qwen verdicts incomplete)"
        if not use_store:
            self.obs, self.runs = lc.build_runs(cfg, self.behaviours, self.manifest, cache, lc.OfflineVerdicts(), self.models)

    def pminus(self, o, n):
        return lc.regression_set(self.runs[o], self.runs[n])

    def complete_for(self, n, bids):
        return all(self.obs[n].recovery_complete(b) for b in bids)

    def recovered(self, n, bid, keep=None):
        """1/0 re-opened by an admitted rewording (restricted to `keep(variant_obs)` if given); None if no rewording."""
        vs = [v for v in self.obs[n].variants.get(bid, []) if keep is None or keep(self, bid, v)]
        if not vs:
            return None
        return int(any(v.verdict == 1 for v in vs))


def rsr_block(src: Source, o, n, keep=None):
    pm = src.pminus(o, n)
    if not src.complete_for(n, pm):
        return dict(bench=src.bench, judge=src.judge, n_pminus=len(pm), complete=False)
    rec = {b: src.recovered(n, b, keep) for b in pm}
    k = sum(1 for v in rec.values() if v == 1)
    unres = sum(1 for v in rec.values() if v is None)
    return dict(bench=src.bench, judge=src.judge, n_pminus=len(pm), k=k, unresolved=unres, complete=True,
                rec=rec, **{f"wilson_{a}": b for a, b in wil(k, len(pm)).items() if a in ("lo", "hi")})


def pooled(blocks, jc, t_mean):
    n = sum(b["n_pminus"] for b in blocks)
    k = sum(b["k"] for b in blocks)
    before = [0] * n                                     # every P^- behaviour is refused canonically by NEW
    after = [v if v is not None else 0 for b in blocks for v in b["rec"].values()]
    w = wil(k, n)
    out = dict(n_pminus=n, k=k, unresolved=sum(b["unresolved"] for b in blocks), rate=w["rate"], lo=w["lo"], hi=w["hi"],
               mcnemar_p=mcnemar_exact(before, after)["p_value"])
    if jc is not None and n:
        iv = corrected_interval(k, n, jc, t_mean=t_mean)
        out.update(corrected=iv.point, corrected_lo=iv.low, corrected_hi=iv.high)
    return out


def pooled_by_benchmark_calibration(blocks, jcs: dict, default_jc, t_mean):
    """Sensitivity to the calibration-transfer assumption: correct each benchmark's k/n with its OWN human
    calibration where one exists (else the default, SORRY-Bench), then pool the corrected rates weighted by
    |P^-|. Interval: delta method per benchmark, variances summed with weights (n_b / N)^2."""
    import math
    blocks = [b for b in blocks if b.get("complete") and b["n_pminus"]]
    N = sum(b["n_pminus"] for b in blocks)
    if not N or default_jc is None:
        return None
    point, var, per = 0.0, 0.0, []
    for b in blocks:
        jc = jcs.get(b["bench"]) or default_jc
        iv = corrected_interval(b["k"], b["n_pminus"], jc, t_mean=t_mean)
        w = b["n_pminus"] / N
        point += w * iv.point
        var += (w * (iv.high - iv.low) / 3.92) ** 2
        per.append(dict(bench=b["bench"], calibration=("own" if b["bench"] in jcs else "SORRY-Bench"),
                        k=b["k"], n=b["n_pminus"], corrected=iv.point, lo=iv.low, hi=iv.high))
    half = 1.959963984540054 * math.sqrt(var)
    return dict(corrected=point, lo=max(0.0, point - half), hi=min(1.0, point + half), per_benchmark=per,
                calibrations={k: dict(se=v.se, sp=v.sp) for k, v in jcs.items()})


def tighten(delta):
    thr = {k: v + delta for k, v in STRICT.items() if k in ("cosine", "bertscore", "nli_entail")}

    def keep(src, bid, v):
        a = src.audit.get((bid, v.index))
        if a is None:
            return True
        return all((getattr(a, k) is None) or getattr(a, k) >= t for k, t in thr.items())
    return keep


def budget(t):
    return lambda src, bid, v: v.index < t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.out or ROOT / f"results/analysis_{stamp}")
    if out.exists():
        raise SystemExit(f"{out} exists; never overwrite")
    cfg = Config.load(ROOT / CFG_PATH)
    tag = lc.judge_tag_from_cfg(cfg)
    cache = lc.load_response_cache(ROOT / "cache/llama_gen.jsonl")
    cal = sd.judge_calibrations()
    jc = sd.anchor_judge_chars(cal)
    res = {"generated": stamp, "judge_correction": (dict(se=jc.se, sp=jc.sp, source="SORRY-Bench human sheet, bundle level")
                                                   if jc else None)}
    res["confirmatory"] = load_confirmatory()
    srcs = {b: Source(cfg, cache, tag, b) for b in SOURCES}
    offs = {b: Source(cfg, cache, tag, b, judge="offline") for b in SOURCES}

    # ---- 1. pooled matched-scale headline -------------------------------------------------------------
    blocks = [rsr_block(srcs[b], OLD, NEW) for b in SOURCES]
    n_adm = [len(srcs[blk["bench"]].manifest.R(x)) for blk in blocks for x in blk.get("rec", {})]
    t_mean = sum(n_adm) / max(1, len(n_adm))           # mean admitted rewordings per P^- behaviour
    res["headline"] = dict(pair=f"{OLD} -> {NEW}", per_benchmark=[{k: v for k, v in b.items() if k != "rec"} for b in blocks],
                           pooled=pooled(blocks, jc, t_mean) if all(b["complete"] for b in blocks) else None,
                           mean_admitted=t_mean)
    hb = (cal or {}).get("harmbench")
    own = {"harmbench": sd.JudgeChars(se=hb["package"]["se"], sp=hb["package"]["sp"], var_se=hb["package"]["var_se"],
                                      var_sp=hb["package"]["var_sp"], response_level=False)} if hb else {}
    res["headline"]["pooled_own_calibration"] = (pooled_by_benchmark_calibration(blocks, own, jc, t_mean)
                                                 if own and res["headline"]["pooled"] else None)
    oblocks = [rsr_block(offs[b], OLD, NEW) for b in SOURCES]
    res["headline_offline_rule"] = dict(per_benchmark=[{k: v for k, v in b.items() if k != "rec"} for b in oblocks],
                                        pooled=pooled(oblocks, None, t_mean))

    # ---- 2. RQ2 per category (anchor pair, pooled) and per SORRY domain (lineage) -----------------------
    cat = defaultdict(lambda: [0, 0])
    for blk in blocks:
        s = srcs[blk["bench"]]
        for bid, v in blk["rec"].items():
            c0 = s.bmap[bid].category
            c = f"{blk['bench']}:{lc.category_name(c0) if blk['bench'] == 'sorrybench' else c0}"
            cat[c][0] += int(v == 1)
            cat[c][1] += 1
    res["rq2_anchor_by_category"] = sorted(([c, *kn] for c, kn in cat.items()), key=lambda r: (-r[2], r[0]))
    dom = defaultdict(lambda: [0, 0])
    s = srcs["sorrybench"]
    lineage = list(cfg.lineage)
    for o, n in zip(lineage, lineage[1:]):
        b = rsr_block(s, o, n)
        if not b["complete"]:
            continue
        for bid, v in b["rec"].items():
            d = lc.domain_of(s.bmap[bid].category)
            dom[d][0] += int(v == 1)
            dom[d][1] += 1
    res["rq2_sorry_lineage_by_domain"] = {d: wil(k, n) for d, (k, n) in sorted(dom.items())}

    # ---- 3. RQ3 drivers: per-variant recovery on P^- of the newer model -----------------------------
    def trials(pairs_by_src):
        rows = []
        for src, o, n in pairs_by_src:
            for bid in src.pminus(o, n):
                for v in src.obs[n].variants.get(bid, []):
                    if v.verdict is None:
                        continue
                    rows.append(dict(bid=f"{src.bench}:{o}>{n}:{bid}", success=int(v.verdict), transform_class=v.transform_class,
                                     surface_distance=surface_distance(src.bmap[bid].canonical_prompt, v.text),
                                     category=src.bench))
        return pd.DataFrame(rows)
    rq3 = {}
    for name, spec in {"anchor_pooled": [(srcs[b], OLD, NEW) for b in SOURCES],
                       "sorry_lineage": [(s, o, n) for o, n in zip(lineage, lineage[1:])]}.items():
        df = trials(spec)
        r = fit_drivers(df, list(df["bid"].unique())) if not df.empty else None
        rq3[name] = (dict(r.summary(), class_rates=df.groupby("transform_class")["success"].agg(["mean", "count"]).round(3)
                          .reset_index().to_dict(orient="records")) if r else None)
    res["rq3_drivers"] = rq3

    # ---- 4. sensitivity -------------------------------------------------------------------------------
    sens = []
    for label, keep in [("strict (registered)", None), ("tightened +0.05", tighten(0.05)), ("tightened +0.10", tighten(0.10)),
                        *[(f"budget t={t}", budget(t)) for t in (1, 2, 4)]]:
        bl = [rsr_block(srcs[b], OLD, NEW, keep) for b in SOURCES]
        p = pooled(bl, None, t_mean)
        nres = p["n_pminus"] - p["unresolved"]
        wr = wil(p["k"], nres)
        sens.append(dict(setting=label, n_pminus=p["n_pminus"], k=p["k"], unresolved=p["unresolved"],
                         rsr=p["rate"], lo=p["lo"], hi=p["hi"], rsr_resolved=wr["rate"], lo_r=wr["lo"], hi_r=wr["hi"]))
    sens.append(dict(setting="loosened -0.05 / -0.10", note="not estimable: rejected rewordings' text is not stored"))
    from regate_intent import IntentStore, intent_key
    ip = ROOT / f"cache/intent_{INTENT_MODEL.replace(':', '_')}.jsonl"
    if ip.exists():
        ist = IntentStore(ip)
        need = [(srcs[b].bench, x, v) for b in SOURCES for x in srcs[b].pminus(OLD, NEW)
                for v in srcs[b].obs[NEW].variants.get(x, [])]
        missing = sum(1 for bench, x, v in need if ist.get(intent_key(INTENT_MODEL, bench, x, v.index, v.text)) is None)
        if missing:
            sens.append(dict(setting=f"+ gate (d) {INTENT_MODEL}", note=f"incomplete: {missing} P⁻ rewordings not yet checked"))
        else:
            def keep_d(src, bid, v):
                return ist.get(intent_key(INTENT_MODEL, src.bench, bid, v.index, v.text))["admitted"] == 1
            bl = [rsr_block(srcs[b], OLD, NEW, keep_d) for b in SOURCES]
            p = pooled(bl, None, t_mean)
            nres = p["n_pminus"] - p["unresolved"]
            wr = wil(p["k"], nres)
            dropped = sum(1 for bench, x, v in need if ist.get(intent_key(INTENT_MODEL, bench, x, v.index, v.text))["admitted"] == 0)
            sens.append(dict(setting=f"+ gate (d) {INTENT_MODEL} ({dropped} of {len(need)} P⁻ rewordings dropped)",
                             n_pminus=p["n_pminus"], k=p["k"], unresolved=p["unresolved"], rsr=p["rate"], lo=p["lo"],
                             hi=p["hi"], rsr_resolved=wr["rate"], lo_r=wr["lo"], hi_r=wr["hi"]))
    else:
        sens.append(dict(setting=f"+ gate (d) {INTENT_MODEL}", note="not run yet (tools/regate_intent.py)"))
    res["sensitivity_anchor_pooled"] = sens

    # ---- 5. PBR backbone (only where complete) --------------------------------------------------------
    pbr = []
    for b, src in srcs.items():
        for m in src.models:
            cm = src.runs[m].C_M()
            if not src.complete_for(m, cm):
                pbr.append(dict(bench=b, model=m, judge=src.judge, complete=False,
                                missing_variant_verdicts=src.obs[m].missing_verdicts["variant"],
                                missing_responses=src.obs[m].missing_responses))
                continue
            p = compute_pbr(src.runs[m], sd.IDENTITY, cfg.decision.pi_star)
            pbr.append(dict(bench=b, model=m, judge=src.judge, complete=True, n_cm=p.n_cm, n_resolved=p.n_resolved,
                            k=p.k_bypassed, pbr=p.pbr_raw, lo=p.raw_ci.low, hi=p.raw_ci.high, verdict=p.verdict))
    res["pbr"] = pbr

    # ---- 6. RPE vs paraphrase agreement (joined on prompt text) -------------------------------------
    rpe_store = lc.VerdictStore(ROOT / "cache/judge_rpe_qwen.jsonl")
    agree = []
    for sp in sd.RPE_SPECS:
        p = ROOT / sp.path
        if not p.exists() or sp.dataset not in srcs:
            continue
        src = srcs[sp.dataset]
        by_prompt = defaultdict(list)
        for b in src.behaviours:
            by_prompt[b.canonical_prompt].append(b.bid)
        pm = set(src.pminus(sp.old, sp.new)) if sp.old in src.models and sp.new in src.models else set()
        cells = defaultdict(int)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not (r.get("prompt") and str(r.get("bid", "")).startswith("jbb-")):
                continue
            adm = [x for x in r.get("details") or [] if x.get("admitted") and x.get("response") is not None]
            if not adm:
                continue
            rpe = int(any(rpe_store.get(lc.verdict_key(tag, x["text"], x["response"])) == 1 for x in adm))
            bids = [b for b in by_prompt.get(r["prompt"], []) if b in pm]
            if not bids:
                cells["not_in_current_pminus"] += 1
                continue
            par = src.recovered(sp.new, bids[0])
            if par is None:
                cells["paraphrase_unresolved"] += 1
                continue
            cells[f"rpe{rpe}_para{par}"] += 1
        both = sum(v for k, v in cells.items() if k.startswith("rpe"))
        agree.append(dict(run=sp.id, dataset=sp.dataset, judge="qwen2.5:14b", **cells, n_both=both,
                          agreement=((cells["rpe1_para1"] + cells["rpe0_para0"]) / both) if both else None))
    rpe_store.close()
    res["rpe_agreement"] = agree

    # ---- 6. the inversion arm as it stands: every gate-scored closed-loop run ---------------------------
    import inversion_section as invsec
    inv = invsec.load(root=ROOT)
    arm = []
    for (bench, o, n), r in sorted(inv.runs.items()):
        if not r.p_minus:
            continue
        v = r.view or {}
        row = dict(bench=bench, old=o, new=n, posthoc=r.posthoc, p_minus=r.p_minus, gated=r.gated,
                   closedloop=invsec._rel(r.closedloop) or None, RSR=v.get("RSR"),
                   mcnemar_loop=(v.get("agreement") or {}).get("mcnemar_p"),
                   mcnemar_both=(v.get("agreement_both") or {}).get("mcnemar_p"))
        for a in ("loop", "both", "gate"):
            row[f"PRR_{a}"] = r.adm(a) if (a == "loop" or r.gated) else None
        hb = invsec.harm_bracket(inv, r)
        if hb:
            row["harm_audit"] = {k: hb[k] for k in ("verified", "refuted", "pending", "lower", "upper")}
        arm.append(row)
    res["inversion_arm"] = arm
    res["inversion_reconcile"] = (inv.reconcile or {}).get("findings")

    out.mkdir(parents=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "analysis.md").write_text(render_md(res))
    print((out / "analysis.md").read_text())
    print(f"-> {out}")
    return 0


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def load_confirmatory():
    """The registered confirmatory headline, as its run directory recorded it."""
    for name in CONFIRM_FILES:
        p = ROOT / CONFIRM_DIR / name
        if p.exists():
            d = json.loads(p.read_text())
            d["_file"] = f"{CONFIRM_DIR}/{name}"
            d["_superseded_files"] = [f"{CONFIRM_DIR}/{o}" for o in CONFIRM_FILES
                                      if o != name and (ROOT / CONFIRM_DIR / o).exists()]
            return d
    return None


def render_confirmatory(c):
    if not c:
        return ["## 0. Confirmatory replication — NOT FOUND", "",
                f"No corrected headline in `{CONFIRM_DIR}`. The exploratory sections below cannot "
                "stand in for it.", ""]
    cal, rsr, pbr = c["calibration"], c["RSR"], c["PBR_new"]
    L = [f"## 0. CONFIRMATORY replication (Amendment 1 §A2) — {c['pair']}", "",
         f"Held-out JBB+GuidedBench, stamped before the run. Source: `{c['_file']}`. "
         "**This is the only confirmatory result in this document.** It is reported on its own and is "
         "never pooled with §1–§6.", "",
         "| estimand | k/n | raw | raw Wilson 95% | corrected | corrected 95% |",
         "|---|--:|--:|---|--:|---|"]
    for name, d in (("RSR", rsr), (f"PBR ({c['pair'].split('->')[-1].strip()})", pbr)):
        L.append(f"| {name} | {d['k']}/{d['n']} | {d['raw']:.4f} | "
                 f"{d['raw_wilson95'][0]:.4f}–{d['raw_wilson95'][1]:.4f} | {d['corrected']:.4f} | "
                 f"{d['corrected_ci95_delta'][0]:.4f}–{d['corrected_ci95_delta'][1]:.4f} |")
    L += ["", f"Judge admitted: se {cal['se']:.4f} · sp {cal['sp']:.4f} · κ {cal['kappa']:.4f} over "
          f"{cal['n_bundles']} bundles ({cal['sheet']}). §A2 registers this SORRY-Bench calibration for "
          "the confirmatory headline, so no held-out calibration is required.",
          f"McNemar exact p = {c['mcnemar_p']:.3g}; mean admitted rewordings per P⁻ behaviour "
          f"t = {c['t_mean']}.", ""]
    if rsr.get("verdict"):
        L.append(f"**RSR verdict: {rsr['verdict']}** against π\\* = {c.get('pi_star', 0.15)}.")
    if pbr.get("verdict_warning"):
        L += ["", f"**PBR verdict: reported as _{pbr['verdict_reported']}_** (mechanically "
              f"_{pbr['verdict_mechanical']}_). {pbr['verdict_warning']}"]
    if c.get("_superseded_files"):
        L += ["", f"Superseded in the same directory: {', '.join('`' + f + '`' for f in c['_superseded_files'])}."]
    return L + [""]


def render_md(r):
    L = [f"# Exploratory analyses (prereg Amendment 1 §A3) — {r['generated']}", "",
         "All figures RAW unless marked corrected. **Everything below §0 is exploratory**: no result "
         "in §1–§6 is confirmatory, and none of them may be pooled with §0.", ""]
    L += render_confirmatory(r.get("confirmatory"))
    h = r["headline"]
    L += [f"## 1. Pooled matched-scale headline: {h['pair']}", "",
          "| benchmark | judge | P⁻ | re-opened | unresolved | Wilson 95% |", "|---|---|--:|--:|--:|---|"]
    for b in h["per_benchmark"]:
        if b["complete"] and b["n_pminus"] == 0:
            L.append(f"| {b['bench']} | {b['judge']} | 0 | — | — | RSR undefined (P⁻ = 0) |")
        elif b["complete"]:
            L.append(f"| {b['bench']} | {b['judge']} | {b['n_pminus']} | {b['k']} | {b['unresolved']} | "
                     f"{pct(b['wilson_lo'])}–{pct(b['wilson_hi'])} |")
        else:
            L.append(f"| {b['bench']} | {b['judge']} | {b['n_pminus']} | incomplete | | |")
    p = h["pooled"]
    if p:
        L += ["", f"**Pooled RSR {p['k']}/{p['n_pminus']} = {pct(p['rate'])}** (Wilson {pct(p['lo'])}–{pct(p['hi'])}), "
              f"McNemar exact p = {p['mcnemar_p']:.2g}"
              + (f"; corrected {pct(p['corrected'])} ({pct(p['corrected_lo'])}–{pct(p['corrected_hi'])}) with the "
                 f"SORRY-Bench bundle calibration (transfer to other benchmarks assumed)" if "corrected" in p else "") + "."]
        q = h.get("pooled_own_calibration")
        if q:
            L += ["", f"Sensitivity, each benchmark with its own human calibration where one exists "
                  f"({', '.join(x['bench'] + ' ' + x['calibration'] for x in q['per_benchmark'])}): corrected "
                  f"{pct(q['corrected'])} ({pct(q['lo'])}–{pct(q['hi'])})."]
    po = r["headline_offline_rule"]["pooled"]
    L += [f"Offline refusal rule on the same items: {po['k']}/{po['n_pminus']} = {pct(po['rate'])} "
          f"(Wilson {pct(po['lo'])}–{pct(po['hi'])}).", ""]
    L += ["## 2. RQ2", "", "Per SORRY-Bench domain, pooled over the five lineage pairs:", "",
          "| domain | re-opened / P⁻ | Wilson 95% |", "|---|--:|---|"]
    for d, w in r["rq2_sorry_lineage_by_domain"].items():
        L.append(f"| {d} | {w['k']}/{w['n']} = {pct(w['rate'])} | {pct(w['lo'])}–{pct(w['hi'])} |")
    L += ["", "Anchor pair, categories with ≥ 3 fixes:", "", "| category | re-opened / P⁻ |", "|---|--:|"]
    for c, k, n in r["rq2_anchor_by_category"]:
        if n >= 3:
            L.append(f"| {c} | {k}/{n} |")
    L += ["", "## 3. RQ3 drivers (per-variant recovery on P⁻, behaviour random intercept, Holm)", ""]
    for name, d in r["rq3_drivers"].items():
        if not d:
            L.append(f"- {name}: not estimable")
            continue
        L += [f"**{name}** — {d['method']}, {d['n_obs']} variants in {d['n_groups']} behaviours", "",
              "| term | odds ratio | p | Holm |", "|---|--:|--:|---|"]
        for t in d["table"]:
            L.append(f"| {t['term']} | {t['odds_ratio']:.2f} | {t['p_value']:.3g} | {'yes' if t['driver_after_holm'] else ''} |")
        L += ["", "| class | recovery rate | variants |", "|---|--:|--:|"]
        for c in d["class_rates"]:
            L.append(f"| {c['transform_class']} | {pct(c['mean'])} | {c['count']} |")
        L.append("")
    L += ["## 4. Sensitivity (anchor pair, pooled, qwen where stored)", "",
          "| setting | P⁻ | re-opened | unresolved | RSR (÷P⁻) | Wilson 95% | RSR (÷resolved) | Wilson 95% |",
          "|---|--:|--:|--:|--:|---|--:|---|"]
    for s in r["sensitivity_anchor_pooled"]:
        if "note" in s:
            L.append(f"| {s['setting']} | {s['note']} | | | | | | |")
        else:
            L.append(f"| {s['setting']} | {s['n_pminus']} | {s['k']} | {s['unresolved']} | {pct(s['rsr'])} | "
                     f"{pct(s['lo'])}–{pct(s['hi'])} | {pct(s['rsr_resolved'])} | {pct(s['lo_r'])}–{pct(s['hi_r'])} |")
    L += ["", "## 5. PBR backbone", "", "| benchmark | model | judge | PBR | k / resolved | Wilson 95% | verdict (π* 0.15) |",
          "|---|---|---|--:|--:|---|---|"]
    for p in r["pbr"]:
        if p["complete"]:
            L.append(f"| {p['bench']} | {p['model']} | {p['judge']} | {pct(p['pbr'])} | {p['k']}/{p['n_resolved']} | "
                     f"{pct(p['lo'])}–{pct(p['hi'])} | {p['verdict']} |")
        else:
            why = (f"{p['missing_responses']} rewording responses never generated" if p.get("missing_responses")
                   else f"{p['missing_variant_verdicts']} verdicts missing")
            L.append(f"| {p['bench']} | {p['model']} | {p['judge']} | not estimable ({why}) | | | |")
    def kn(x):
        return "—" if not x or not x.get("n") else f"{pct(x['rate'])} ({x['k']}/{x['n']})"
    L += ["", "## 6. Inversion arm (current closed-loop runs, qwen2.5:14b judge)", "",
          "Per pair only (a behaviour sits in several pairs' P⁻). Closed loop = the older model's reply to the "
          "reconstruction matches its original answer (upper bound until the harmful-ask audit); intent gate = the "
          "recycle arm's own gate against the original request; both = the intersection.", "",
          "| benchmark | pair | P⁻ | RSR | PRR closed loop | PRR both | PRR intent gate | McNemar RSR vs loop | harmful-ask audit |",
          "|---|---|--:|--:|--:|--:|--:|--:|---|"]
    for a in r.get("inversion_arm", []):
        au = "—"
        if a.get("harm_audit"):
            h = a["harm_audit"]
            au = (f"[{pct(h['lower']['rate'])}, {pct(h['upper']['rate'])}] · {h['verified']} verified, {h['refuted']} refuted, "
                  f"{h['pending']} pending" if h["verified"] + h["refuted"] else f"not labelled yet ({h['pending']} behaviours)")
        mc = "—" if a["mcnemar_loop"] is None else f"{a['mcnemar_loop']:.3g}"
        L.append(f"| {a['bench']} | {a['old']} → {a['new']}{' (post-hoc)' if a['posthoc'] else ''} | {a['p_minus']} | "
                 f"{kn(a['RSR'])} | {kn(a['PRR_loop'])} | {kn(a['PRR_both'])} | {kn(a['PRR_gate'])} | {mc} | {au} |")
    for f_ in r.get("inversion_reconcile") or []:
        L.append(f"- {f_}")
    L += ["", "## 6b. Early RPE runs vs paraphrase recovery (superseded; behaviours resolved by both, qwen judge)", "",
          "| run | both resolved | agree | rpe+/para+ | rpe+/para− | rpe−/para+ | rpe−/para− | not in P⁻ |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for a in r["rpe_agreement"]:
        L.append(f"| {a['run']} | {a['n_both']} | {pct(a['agreement'])} | {a.get('rpe1_para1', 0)} | {a.get('rpe1_para0', 0)} | "
                 f"{a.get('rpe0_para1', 0)} | {a.get('rpe0_para0', 0)} | {a.get('not_in_current_pminus', 0)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
