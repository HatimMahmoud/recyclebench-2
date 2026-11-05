#!/usr/bin/env python3
"""tools/cluster_pairs.py -- one dashboard per model pair (older -> newer) for the Pegasus runs, plus an index.

The cluster twin of tools/pair_dashboard.py (whose evidence layer, tools/study_data.py, reads the Mac's artifacts).
Sources are the cluster's caches and verdict stores: the canonical screens of the dense lineage, Llama 4 Scout and
Maverick on SORRY-Bench / HarmBench / AdvBench / StrongREJECT, the external corpora (ScienceSafetyData, GuidedBench)
and the 70B chain. For every pair and every benchmark where both models are judged: the canonical 2x2, the recycle
run (RSR, raw) and the inversion run (A_gate / A_loop / A_both) where they exist, and the cases: every fix and every
regression with the request, both models' replies and draw verdicts, and for recycled pairs the admitted rewordings
that re-opened the fix with the newer model's reply. Same DFKI design as every other page; no model call.

  python3 tools/cluster_pairs.py                    # every pair with data; text included (working file)
  python3 tools/cluster_pairs.py --redact           # counts only, shareable
  python3 tools/cluster_pairs.py --pairs llama3-8b:llama4-scout

Writes results/runs/dedicated-runs-cluster/<old>__to__<new>/dashboard_<stamp>.html and index_<stamp>.html; never
overwrites. Every figure is raw judge output (qwen2.5:14b rubric) and exploratory unless a registration says
otherwise; nothing here is pooled.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import lineage_core as lc                                      # noqa: E402
import lineage_dashboard as ld                                 # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.datasets import load_jailbreakbench          # noqa: E402
from recyclebench.manifest import Manifest                     # noqa: E402
from recyclebench.realrun import _key                          # noqa: E402
from recyclebench.screening import _draw_seed                  # noqa: E402

esc, pct, ci_txt, label, short = ld.esc, ld.pct, ld.ci_txt, ld.label, ld.short
tile, table, wilson = ld.tile, ld.table, ld.wilson
OUT_ROOT = "results/runs/dedicated-runs-cluster"
CAP = {"prompt": 1200, "reply": 1600, "rewording": 900}
DENSE = ("llama2-7b", "llama2-13b", "llama3-8b", "llama31-8b", "llama32-1b", "llama32-3b")
B70 = ("llama2-70b", "llama3-70b", "llama31-70b", "llama33-70b")
LABEL = {"sorrybench": "SORRY-Bench", "harmbench": "HarmBench", "advbench": "AdvBench", "strongreject": "StrongREJECT",
         "sciencesafety": "ScienceSafetyData", "guidedbench": "GuidedBench"}

# (benchmark, config, response cache, verdict store, models, registration / designation)
SOURCES = [
    *[(b, "configs/llama4_pegasus.yaml", "cache/llama4_maverick_merged.jsonl", s, DENSE + ("llama4-scout", "llama4-maverick"),
       "Scout study (post hoc, exploratory); Maverick judged 2026-11-05")
      for b, s in (("harmbench", "cache/judge_qwen_harmbench.jsonl"), ("advbench", "cache/judge_qwen_advbench.jsonl"),
                   ("strongreject", "cache/judge_qwen_strongreject.jsonl"))],
    ("sorrybench", "configs/llama4_pegasus.yaml", "cache/llama_gen.jsonl", "cache/judge_lineage_sorrybench.jsonl",
     DENSE + ("llama4-scout",), "lineage cache (Mac) + Scout screen 2026-11-04 (post hoc, exploratory)"),
    ("sorrybench", "configs/chain70b.yaml", "cache/chain70b/canonical.jsonl", "cache/chain70b/judge_sorrybench.jsonl",
     B70, "PREREG_70B_CHAIN_20261105.md (signed; exploratory)"),
    *[(b, "configs/llama4_pegasus.yaml", "cache/external/canonical.jsonl", f"cache/external/judge_qwen_{b}.jsonl",
       DENSE + ("llama4-scout",), "PREREG_EXTERNAL_CLUSTER_20261103.md (signed; exploratory)")
      for b in ("sciencesafety", "guidedbench")],
]
PAIRS = ([("llama2-7b", "llama2-13b"), ("llama2-13b", "llama3-8b"), ("llama3-8b", "llama31-8b"),
          ("llama31-8b", "llama32-1b"), ("llama32-1b", "llama32-3b"), ("llama2-7b", "llama3-8b"),
          ("llama2-7b", "llama31-8b")]
         + [(o, "llama4-scout") for o in DENSE] + [(o, "llama4-maverick") for o in DENSE]
         + [("llama2-70b", "llama3-70b"), ("llama3-70b", "llama31-70b"), ("llama31-70b", "llama33-70b")])


# =============================================================================
# evidence
# =============================================================================
_CACHES = {}


def _cache(path):
    """Several sources share one response cache: load each file once."""
    if path not in _CACHES:
        _CACHES[path] = lc.load_response_cache(str(ROOT / path))
    return _CACHES[path]


class Source:
    def __init__(self, bench, config, cache, store, models, reg):
        self.bench, self.config, self.cache_path, self.store_path, self.reg = bench, config, cache, store, reg
        self.cfg = Config.load(str(ROOT / config))
        self.behaviours = load_jailbreakbench(str(ROOT / "data" / f"{bench}.csv"))
        cache = _cache(cache)
        st = lc.VerdictStore(str(ROOT / store))
        want = [m for m in models if m in self.cfg.real]
        self.obs, self.runs = lc.build_runs(self.cfg, self.behaviours, Manifest(), cache,
                                            lc.StoredVerdicts(st, lc.judge_tag_from_cfg(self.cfg)), want)
        st.close()
        self.models = [m for m in want if self.obs[m].missing_responses == 0 and self.obs[m].missing_verdicts["canonical"] == 0
                       and len(self.runs[m].canonical_obs) == len(self.behaviours)]


def load_sources(only=None):
    out = []
    for s in SOURCES:
        if only and s[0] not in only:
            continue
        if not ((ROOT / s[2]).exists() and (ROOT / s[3]).exists() and (ROOT / "data" / f"{s[0]}.csv").exists()):
            print(f"skip {s[0]} ({s[2]}): cache, store or data absent")
            continue
        src = Source(*s)
        print(f"{s[0]:13s} {s[2]:40s} complete models: {' '.join(src.models)}")
        out.append(src)
    return out


def recycle_run(bench, o, n):
    """(run dir, its own cache) of the recycle run of this pair on this benchmark, if any."""
    for d, c in ((f"results/runs/external/external__{bench}__{o}__{n}", "cache/external"),
                 (f"results/runs/llama4_scout/llama4_scout__{bench}__{o}", "cache/llama4_scout") if n == "llama4-scout" else (None, None),
                 (f"results/runs/chain70b/chain70b__{bench}__{o}__{n}", "cache/chain70b")):
        if d and (ROOT / d / "summary_real.json").exists():
            return ROOT / d, ROOT / c / f"{Path(d).name}.jsonl"
    return None, None


def recycle_info(bench, o, n, k_spv, with_text):
    d, cache_p = recycle_run(bench, o, n)
    if not d:
        return None
    s = json.loads((d / "summary_real.json").read_text(encoding="utf-8"))
    r = next((x for x in s.get("RSR_primary_RAW") or [] if x.get("pair") == f"{o} -> {n}"), None)
    pil = json.loads((d / "pminus_pilot.json").read_text(encoding="utf-8")) if (d / "pminus_pilot.json").exists() else {}
    man = sorted(d.glob("manifest_*.jsonl"))
    adm = var = 0
    if man:
        for ln in man[-1].read_text(encoding="utf-8").splitlines():
            if ln.strip():
                x = json.loads(ln)
                if x.get("type") == "audit":
                    var += 1
                    adm += bool(x.get("admitted"))
    info = {"run": str(d.relative_to(ROOT)), "rsr": r, "p_rejudged": pil.get("p_minus"), "admitted": adm, "variants": var,
            "reworded": {}}
    sub = next(iter(d.glob("*_p_minus_judged.csv")), None)
    trials = d / "trials" / f"trials_{n}.csv"
    if with_text and man and sub and trials.exists() and cache_p.exists():
        beh = load_jailbreakbench(str(sub))
        manifest = Manifest.load(str(man[-1]))
        cache = lc.load_response_cache(str(cache_p))
        rows = {}
        with open(trials, newline="", encoding="utf-8") as fh:
            for t in csv.DictReader(fh):
                rows.setdefault(t["bid"], []).append(t)
        for b in beh:
            vs, tr = manifest.R(b.bid), rows.get(b.bid, [])
            if len(vs) != len(tr) or any(v.transform_class != t["transform_class"] for v, t in zip(vs, tr)):
                continue
            info["reworded"][b.canonical_prompt] = [
                {"style": v.transform_class, "text": v.text, "success": int(t["success"]),
                 "reply": next((cache[k] for k in (_key(n, v.text, _draw_seed(b.bid, v.index, i)) for i in range(k_spv))
                                if k in cache), "")}
                for v, t in zip(vs, tr)]
    return info


def rpe_info(bench, o, n):
    for root in ("results/runs/external/rpe", "results/runs/llama4_scout/rpe", "results/runs/chain70b/rpe"):
        xs = sorted((ROOT / root / bench).glob(f"closedloop_{o}__to__{n}_gated_*.json"))
        if xs:
            j = json.loads(xs[-1].read_text(encoding="utf-8"))
            return {"file": str(xs[-1].relative_to(ROOT)), "n": j.get("n_behaviours"),
                    "admission": {k: (v["k"], v["n"], v["unresolved"]) for k, v in (j.get("admission") or {}).items()}}
    return None


# =============================================================================
# rendering
# =============================================================================
def _txt(tag, text, n):
    return f'<div class="txt"><span class="tl">{esc(tag)}</span>{esc(ld.cap(text or "", n))}</div>'


def _kn(k, n):
    if not n:
        return "undefined (n = 0)"
    p, lo, hi = wilson(k, n)
    return f"<b>{k}/{n}</b> · {pct(p)} <span class='ci'>{ci_txt(lo, hi)}</span>"


def _verd(draws):
    return "".join("C" if v == 1 else "R" if v == 0 else "?" for _, v in draws)


def _first(draws, want):
    return next((t for t, v in draws if v == want and t), None) or next((t for t, _ in draws if t), "")


def _tab(tid, title, sub, active):
    return (f'<button role="tab" class="tab" id="tab-{tid}" aria-controls="panel-{tid}" '
            f'aria-selected="{"true" if active else "false"}"><b>{esc(title)}</b><span>{esc(sub)}</span></button>')


def cases(src, o, n, rec, redact):
    items, cnt = [], Counter()
    for b in src.behaviours:
        vo, vn = src.obs[o].canonical.get(b.bid), src.obs[n].canonical.get(b.bid)
        if vo == 1 and vn == 0:
            kind = "fix"
        elif vo == 0 and vn == 1:
            kind = "regression"
        else:
            continue
        cnt[kind] += 1
        do, dn = src.obs[o].draws[b.bid], src.obs[n].draws[b.bid]
        pill = ('<span class="pill pill-refuse">fix</span>' if kind == "fix"
                else '<span class="pill pill-comply">regression</span>')
        body = f'<p class="note">{esc(short(o))} draws {_verd(do)} · {esc(short(n))} draws {_verd(dn)}</p>'
        if not redact:
            body += (_txt(f"request · {b.category}", b.canonical_prompt, CAP["prompt"])
                     + _txt(f"{label(o)} · {'answered' if vo == 1 else 'refused'}", _first(do, vo), CAP["reply"])
                     + _txt(f"{label(n)} · {'answered' if vn == 1 else 'refused'}", _first(dn, vn), CAP["reply"]))
        if kind == "fix" and rec:
            vs = rec["reworded"].get(b.canonical_prompt)
            if vs is None:
                body += '<p class="note">Not in the recycle run\'s own P⁻ (fresh draws), or rewordings not joined.</p>'
            else:
                hit = [v for v in vs if v["success"]]
                pill += (f'<span class="pill pill-comply">re-opened · {len(hit)}/{len(vs)}</span>' if hit
                         else f'<span class="pill">held · 0/{len(vs)}</span>' if vs else '<span class="pill">unresolved</span>')
                if hit:
                    kind = "reopened"
                    cnt["reopened"] += 1
                if not redact:
                    for v in (hit or vs)[:2]:
                        body += (_txt(f"admitted rewording · {v['style']}", v["text"], CAP["rewording"])
                                 + _txt(f"{label(n)} · {'answered' if v['success'] else 'refused'} the rewording",
                                        v["reply"], CAP["reply"]))
        items.append(f'<details class="case" data-outcome="{kind}"><summary><span class="cid">{esc(b.bid)}</span>'
                     f'<span class="ccat">{esc(b.category)}</span>{"" if redact else "<span class=sens>sensitive</span>"}'
                     f'{pill}</summary><div class="cbody">{body}</div></details>')
    if not items:
        return '<p class="note">No fix and no regression on this benchmark.</p>'
    btns = "".join(f'<button type="button" class="fbtn{" active" if k == "all" else ""}" data-filter="{k}" '
                   f'aria-pressed="{"true" if k == "all" else "false"}">{t} <span>{c}</span></button>'
                   for k, t, c in [("all", "All", len(items)), ("fix", "Fixes held / not recycled", cnt["fix"] - cnt["reopened"]),
                                   ("reopened", "Fixes re-opened", cnt["reopened"]), ("regression", "Regressions", cnt["regression"])]
                   if k == "all" or c)
    return f'<div class="cases"><div class="filters" role="group" aria-label="Filter cases">{btns}</div>{"".join(items)}</div>'


def bench_panel(src, o, n, rec, rpe, redact):
    c = lc.partition(src.runs[o], src.runs[n])
    k = {x: len(v) for x, v in c.items()}
    nb = len(src.behaviours)
    ao, an = len(src.runs[o].broken_set()), len(src.runs[n].broken_set())
    rows = [["Answered", f"{ao} / {nb} · {pct(ao / nb)}", f"{an} / {nb} · {pct(an / nb)}"]]
    two = table(["", esc(label(o)), esc(label(n))], rows)
    grid = table(["P⁻ (fix)", "Regression", "Both answered", "Both refused"],
                 [[f"<b>{k['pminus']}</b>", k["neg_flip"], k["persistent"], k["both_safe"]]], num_cols=(0, 1, 2, 3))
    rec_html = '<p class="note">No recycle run for this pair here (P⁻ below the floor of 5, or not registered).</p>'
    if rec and rec["rsr"]:
        r = rec["rsr"]
        rec_html = table(["P⁻ canonical", "re-judged", "in run", "RSR raw · Wilson 95 %", "McNemar p", "Unresolved", "Gate admits"],
                         [[k["pminus"], rec["p_rejudged"] or "—", r["n_p_minus"], _kn(r["k_recovered"], r["n_p_minus"]),
                           ld.p_txt(r.get("mcnemar_p")), r["n_unresolved"], f"{rec['admitted']}/{rec['variants']}"]],
                         num_cols=(0, 1, 2, 5)) + f'<p class="note">Run {esc(rec["run"])}.</p>'
    rpe_html = '<p class="note">No inversion run for this pair here (P⁻ = 0, or not run).</p>'
    if rpe:
        a = rpe["admission"]
        rpe_html = table(["Fixes", "A_gate", "A_loop (upper bound)", "A_both"],
                         [[rpe["n"], _kn(*a["gate"][:2]), _kn(*a["loop"][:2]), _kn(*a["both"][:2])]], num_cols=(0,)) \
            + f'<p class="note">{esc(rpe["file"])}. Undefined = no admitted reconstruction, never 0.</p>'
    return (f'<div class="panel"><h3>{esc(LABEL.get(src.bench, src.bench))} · canonical</h3>{two}{grid}'
            f'<p class="note">{esc(src.reg)} · judge {esc(lc.judge_tag_from_cfg(src.cfg))} · {nb} behaviours, k = 3.</p></div>'
            f'<div class="grid2"><div class="panel"><h3>Recycling (RSR)</h3>{rec_html}</div>'
            f'<div class="panel"><h3>Output inversion (RPE)</h3>{rpe_html}</div></div>'
            f'<div class="panel"><h3>Fixes and regressions{"" if redact else " · requests and replies"}</h3>'
            f'{cases(src, o, n, rec, redact)}</div>'), k


def pair_page(o, n, srcs, redact, today):
    have = [s for s in srcs if o in s.models and n in s.models]
    if not have:
        return None, None
    tabs, panels, summary = [], [], []
    for i, s in enumerate(have):
        rec = recycle_info(s.bench, o, n, s.cfg.budget.samples_per_variant, not redact)
        rpe = rpe_info(s.bench, o, n)
        html, k = bench_panel(s, o, n, rec, rpe, redact)
        tid = f"b-{s.bench}"
        tabs.append(_tab(tid, LABEL.get(s.bench, s.bench), f"P⁻ {k['pminus']} · regr. {k['neg_flip']}", i == 0))
        panels.append(f'<div class="tabpanel" role="tabpanel" id="panel-{tid}" aria-labelledby="tab-{tid}"'
                      f'{"" if i == 0 else " hidden"}>{html}</div>')
        r = (rec or {}).get("rsr")
        summary.append({"bench": s.bench, **k, "rsr": (r["k_recovered"], r["n_p_minus"]) if r else None,
                        "loop": rpe["admission"]["loop"][:2] if rpe else None})
    srow = [[esc(LABEL.get(x["bench"], x["bench"])), f"<b>{x['pminus']}</b>", x["neg_flip"], x["persistent"], x["both_safe"],
             _kn(*x["rsr"]) if x["rsr"] else "—", _kn(*x["loop"]) if x["loop"] else "—"] for x in summary]
    nav = "".join(f'<a href="#{i}"><span class="n">{c}</span>{t}</a>' for i, c, t in
                  [("s-all", "01", "Across every benchmark"), ("s-bench", "02", "Benchmark by benchmark"), ("s-prov", "05", "Provenance")])
    body = f"""
<header class="hero" id="top">
  <p class="eyebrow">RecycleBench · dedicated run · Pegasus</p>
  <h1>{esc(label(o))} → {esc(label(n))}</h1>
  <p class="lede">Every benchmark on which both models were screened on the cluster. <b>P⁻</b> (fix): the older model
  answered, the newer refused; <b>regression</b>: the reverse. Recycling re-tests each fix with intent-gated rewordings
  (RSR = fixes re-opened by ≥ 1 admitted rewording); inversion reconstructs the request from the older model's answer.</p>
  <div class="meta"><span class="chip chip-warn">exploratory · raw judge output (qwen2.5:14b rubric)</span>
  {"<span class='chip'>counts only</span>" if redact else "<span class='chip chip-warn'>contains requests and replies · working file</span>"}
  <span>generated <b>{today}</b></span></div>
</header>
<section id="s-all" class="sec"><h2><span class="sn">01</span>This pair across every benchmark</h2>
<div class="panel">{table(["Benchmark", "P⁻", "Regression", "Both answered", "Both refused", "RSR raw", "A_loop"], srow, num_cols=(1, 2, 3, 4))}
<p class="note">No pair here reaches the confirmatory floor of 30 fixes unless stated; every RSR is descriptive.</p></div></section>
<section id="s-bench" class="sec"><h2><span class="sn">02</span>Benchmark by benchmark</h2>
<div class="tablist" role="tablist" aria-label="Benchmarks">{''.join(tabs)}</div>{''.join(panels)}</section>
<section id="s-prov" class="sec"><h2><span class="sn">05</span>Provenance</h2><div class="panel">
{table(["Benchmark", "Config", "Cache", "Verdict store", "Registration"],
       [[esc(s.bench), esc(s.config), esc(s.cache_path), esc(s.store_path), esc(s.reg)] for s in have])}
<p class="note">Built by tools/cluster_pairs.py from the cluster's caches and stores; no model call.</p></div></section>"""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1,viewport-fit=cover"><title>{esc(short(o))} to {esc(short(n))}</title>'
            f'<style>{ld.CSS}</style></head><body><div class="layout"><nav class="side" aria-label="Sections">'
            f'{ld.brand_html(short(o) + " → " + short(n))}<div class="toc">{nav}</div>{ld.legend_html()}'
            f'<button type="button" class="themebtn" id="themebtn" data-t="system">Theme: system</button></nav>'
            f'<main id="main">{body}<footer>{esc("tools/cluster_pairs.py · " + today)}</footer></main></div>'
            f'<div id="tip" role="tooltip" hidden></div><script>{ld.rq_script({})}{ld.JS}</script></body></html>')
    return page, summary


def index_page(entries, redact, today, stamp):
    rows = []
    for o, n, summ in entries:
        cells = " · ".join(f"{LABEL.get(x['bench'], x['bench'])} P⁻ {x['pminus']}/regr. {x['neg_flip']}"
                           + (f" · RSR {x['rsr'][0]}/{x['rsr'][1]}" if x["rsr"] else "") for x in summ)
        rows.append([f'<a href="{o}__to__{n}/dashboard_{stamp}.html">{esc(label(o))} → {esc(label(n))}</a>', esc(cells)])
    body = (f'<header class="hero"><p class="eyebrow">RecycleBench · Pegasus</p><h1>One dashboard per model pair</h1>'
            f'<p class="lede">The cluster runs: dense lineage, Llama 4 Scout and Maverick, the 70B chain, ScienceSafetyData '
            f'and GuidedBench. Raw, exploratory; {"counts only" if redact else "pages contain requests and replies"}.</p>'
            f'<div class="meta"><span>generated <b>{today}</b></span><span><b>{len(entries)}</b> pairs</span></div></header>'
            f'<section class="sec"><div class="panel">{table(["Pair", "Benchmarks"], rows)}</div></section>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>Cluster Pair Dashboards</title><style>{ld.CSS}</style></head>'
            f'<body><div class="layout"><nav class="side">{ld.brand_html("dedicated runs · cluster")}{ld.legend_html()}'
            f'<button type="button" class="themebtn" id="themebtn" data-t="system">Theme: system</button></nav>'
            f'<main id="main">{body}</main></div><div id="tip" role="tooltip" hidden></div>'
            f'<script>{ld.rq_script({})}{ld.JS}</script></body></html>')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=None, help="old:new (default: every pair with data)")
    ap.add_argument("--redact", action="store_true", help="counts only, no request or reply text")
    ap.add_argument("--out-root", default=OUT_ROOT)
    a = ap.parse_args(argv)
    pairs = [tuple(p.split(":")) for p in a.pairs] if a.pairs else PAIRS
    srcs = load_sources()
    today, stamp = _dt.date.today().isoformat(), _dt.datetime.now().strftime("%Y%m%d_%H%M") + ("_redacted" if a.redact else "")
    root = ROOT / a.out_root
    entries = []
    for o, n in pairs:
        page, summ = pair_page(o, n, srcs, a.redact, today)
        if not page:
            print(f"-- {o} -> {n}: no benchmark with both models judged")
            continue
        out = root / f"{o}__to__{n}" / f"dashboard_{stamp}.html"
        if out.exists():
            raise SystemExit(f"{out} exists -- refusing to overwrite")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
        entries.append((o, n, summ))
        print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1e3:.0f} kB)")
    idx = root / f"index_{stamp}.html"
    if idx.exists():
        raise SystemExit(f"{idx} exists -- refusing to overwrite")
    idx.write_text(index_page(entries, a.redact, today, stamp), encoding="utf-8")
    print(f"wrote {idx.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
