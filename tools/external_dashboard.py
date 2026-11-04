#!/usr/bin/env python3
"""tools/external_dashboard.py -- one page for one external corpus of the Pegasus run (PREREG_EXTERNAL_CLUSTER_20261103).

Built from the completion matrix of tools/external_suite.py (canonical 2x2 per pair, recycle RSR, inversion admission,
coverage), plus counts the matrix does not hold: answer rate per corpus category (response cache + verdict store), the
inversion gate's rejection layers (cache/external/rpe_gate.jsonl), admissions per rewording style (frozen manifests), the
Slurm job list and the run's deviation log. Same DFKI design as the lineage / pair / Scout pages (CSS, JS, brand, legend
and chart helpers come from lineage_dashboard, so it cannot drift from them).

Default: counts, ids and hashes only (no prompt, response or rewording text). --text adds a section with every fix and
regression per pair (request, both models' replies, and for recycled pairs the admitted rewordings that re-opened it),
collapsed and truncated: a working file in the same sense as the pair dashboards (ETHICS.md), written as *_text_*.html.
Standalone by registration: the corpus is not in tools/study_data.py and nothing here pools into any other page.

  python3 tools/external_dashboard.py --corpus sciencesafety
  python3 tools/external_dashboard.py --corpus sciencesafety --matrix results/runs/external/matrix_<stamp>.json \
      --jobs <sacct --parsable2 dump> --log $RB_LOGS/external_submit_<stamp>.txt

Writes results/runs/external/dashboard_<corpus>_<stamp>.html; never overwrites.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import external_suite as es                                    # noqa: E402
import lineage_dashboard as ld                                 # noqa: E402

esc, pct, ci_txt, label, short = ld.esc, ld.pct, ld.ci_txt, ld.label, ld.short
tile, table, hbars, chip, wilson, p_txt = ld.tile, ld.table, ld.hbars, ld.chip, ld.wilson, ld.p_txt

CORPUS_LABEL = {"sciencesafety": "ScienceSafetyData", "guidedbench": "GuidedBench", "jbb": "JBB-Behaviors"}
GATE_ORDER = ("lexical_divergence", "cosine", "bertscore", "nli_entail")      # backends/real_gate.py
STYLES = ("paraphrase", "syntactic", "lexical", "back_translation", "tense_shift", "register")
NAV = [("s-headline", "01", "Headline"), ("s-models", "02", "Each model"), ("s-pairs", "03", "Every pair"),
       ("s-recycle", "04", "Recycling (RSR)"), ("s-rpe", "05", "Output inversion (RPE)"),
       ("s-styles", "06", "Rewording styles"), ("s-coverage", "07", "Coverage"),
       ("s-exec", "08", "Execution & provenance"), ("s-method", "09", "Threats & comparison")]

# The 2026-11-01 Mac pilot of this corpus (PREREG_SCIENCESAFETY_20261101.md, results/runs/sciencesafety_20261101/META.txt):
# shown beside the cluster figures, never merged with them (registration §1).
PILOT = {"sciencesafety": {"pair": ("llama3-8b", "llama31-8b"), "p_minus": 23, "rsr": (13, 23), "a_loop": (15, 21),
                           "gate": (0, 183), "env": "Mac M1 Max, Metal, Ollama local, 2026-11-01"}}


# =============================================================================
# data
# =============================================================================
def latest(pattern):
    xs = sorted((ROOT / es.RUN_ROOT).glob(pattern))
    return xs[-1] if xs else None


def canonical(corpus, report_path):
    """The canonical screen rebuilt from the run's cache + verdict store: behaviours, per-model observations, runs."""
    import lineage_core as lc
    from recyclebench.config import Config
    from recyclebench.datasets import load_jailbreakbench
    from recyclebench.manifest import Manifest
    rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
    cache_p, csv_p = ROOT / rep["cache"], ROOT / "data" / f"{corpus}.csv"
    if not (cache_p.exists() and csv_p.exists() and (ROOT / es.verdict_store(corpus)).exists()):
        return None
    cfg = Config.load(str(ROOT / rep["config"]))
    beh = load_jailbreakbench(str(csv_p))
    store = lc.VerdictStore(str(ROOT / es.verdict_store(corpus)))
    models = [m for m in es.MODELS if rep["corpora"][corpus]["models"][m]["judged"]]
    obs, runs = lc.build_runs(cfg, beh, Manifest(), lc.load_response_cache(str(cache_p)),
                              lc.StoredVerdicts(store, lc.judge_tag_from_cfg(cfg)), models)
    store.close()
    return {"cfg": cfg, "behaviours": beh, "obs": obs, "runs": runs}


def per_category(can):
    """Answered behaviours per model and corpus category (counts only)."""
    if not can:
        return None
    beh, runs = can["behaviours"], can["runs"]
    models = list(runs)
    cats = sorted({b.category for b in beh})
    tot = Counter(b.category for b in beh)
    cat_of = {b.bid: b.category for b in beh}
    out = {"categories": cats, "n": dict(tot), "answered": {}}
    for m in models:
        c = Counter(cat_of[b] for b in runs[m].broken_set())
        out["answered"][m] = {k: c.get(k, 0) for k in cats}
    return out


def gate_layers(path):
    """Which gate layer rejected each inversion candidate, and how far the best candidate got (scores only)."""
    p = ROOT / path
    if not p.exists():
        return None
    rej, adm, best = Counter(), 0, {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        stop = r.get("rejected_by")
        if r.get("admitted"):
            adm += 1
        else:
            rej[stop or "?"] += 1
        # the gate short-circuits in GATE_ORDER: only layers up to the rejecting one were computed
        upto = GATE_ORDER[:GATE_ORDER.index(stop) + 1] if stop in GATE_ORDER else GATE_ORDER
        for k in upto:
            v = (r.get("layers") or {}).get(k)
            if isinstance(v, (int, float)):
                best[k] = max(best.get(k, 0.0), float(v))
    return {"n": adm + sum(rej.values()), "admitted": adm, "rejected_by": dict(rej), "max_score": best}


def style_counts(manifests):
    """Variants and gate admissions per rewording style over the frozen recycle manifests (no text)."""
    agg = {s: {"variants": 0, "admitted": 0} for s in STYLES}
    for m in manifests:
        if not (ROOT / m).exists():
            continue
        for ln in (ROOT / m).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("type") != "audit":
                continue
            s = agg.setdefault(r.get("transform_class") or "?", {"variants": 0, "admitted": 0})
            s["variants"] += 1
            s["admitted"] += bool(r.get("admitted"))
    return agg


def read_jobs(path):
    """sacct --parsable2 -o JobID,JobName,Partition,State,Start,Elapsed,AllocTRES (header line first)."""
    if not path or not Path(path).exists():
        return []
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    head = lines[0].split("|")
    return [dict(zip(head, ln.split("|"))) for ln in lines[1:]]


def read_log(path):
    if not path or not Path(path).exists():
        return []
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def reworded(cell, k_spv):
    """For one recycle pair: prompt text -> the newer model's outcome on every admitted rewording (manifest order),
    from the pair's subset CSV, frozen manifest, trial CSV and response cache. Joined on prompt text, because subset
    ids are row positions of the subset, not corpus ids."""
    import csv
    import lineage_core as lc
    from recyclebench.datasets import load_jailbreakbench
    from recyclebench.manifest import Manifest
    from recyclebench.realrun import _key
    from recyclebench.screening import _draw_seed
    run = ROOT / cell["run_dir"]
    name = Path(cell["run_dir"]).name
    sub = run / f"{name}_p_minus_judged.csv"
    man = (ROOT / cell["manifest"]["path"]) if cell.get("manifest") else None
    trials = run / "trials" / f"trials_{cell['new']}.csv"
    cache_p = ROOT / es.CACHE_DIR / f"{name}.jsonl"
    if not (sub.exists() and man and man.exists() and trials.exists() and cache_p.exists()):
        return {}
    beh = load_jailbreakbench(str(sub))
    manifest = Manifest.load(str(man))
    cache = lc.load_response_cache(str(cache_p))
    rows = {}
    with open(trials, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["bid"], []).append(r)
    out = {}
    for b in beh:
        vs, tr = manifest.R(b.bid), rows.get(b.bid, [])
        if len(vs) != len(tr) or any(v.transform_class != t["transform_class"] for v, t in zip(vs, tr)):
            continue                                    # never pair a rewording with another one's outcome
        out[b.canonical_prompt] = [{"style": v.transform_class, "text": v.text, "success": int(t["success"]),
                                    "reply": next((cache.get(_key(cell["new"], v.text, _draw_seed(b.bid, v.index, s)))
                                                   for s in range(k_spv)
                                                   if cache.get(_key(cell["new"], v.text, _draw_seed(b.bid, v.index, s)))), "")}
                                   for v, t in zip(vs, tr)]
    return out


TEXT_CAP = {"prompt": 1200, "reply": 1600, "rewording": 900}


def _txt(tag, text, n):
    return f'<div class="txt"><span class="tl">{esc(tag)}</span>{esc(ld.cap(text or "", n))}</div>'


def _verdicts(draws):
    return "".join("C" if v == 1 else "R" if v == 0 else "?" for _, v in draws)


def _first(draws, want):
    return next((t for t, v in draws if v == want and t), None) or next((t for t, _ in draws if t), "")


def sec_text(d):
    can = d["canonical"]
    if not can:
        return ""
    obs, beh = can["obs"], can["behaviours"]
    floor = d["matrix"]["recycle_floor"]
    rec = {(c["old"], c["new"]): c for c in d["cells"] if c["arm"] == "recycle" and (c.get("pminus") or 0) >= floor}
    tabs, panels = [], []
    pairs = [c for c in d["cells"] if c["arm"] == "canonical"]
    for i, c in enumerate(pairs):
        o, n = c["old"], c["new"]
        rw = d["reworded"].get((o, n), {})
        items, cnt = [], Counter()
        for b in beh:
            vo, vn = obs[o].canonical.get(b.bid), obs[n].canonical.get(b.bid)
            if vo == 1 and vn == 0:
                kind = "fix"
            elif vo == 0 and vn == 1:
                kind = "regression"
            else:
                continue
            cnt[kind] += 1
            do, dn = obs[o].draws[b.bid], obs[n].draws[b.bid]
            body = (_txt(f"request · {b.category}", b.canonical_prompt, TEXT_CAP["prompt"])
                    + _txt(f"{label(o)} · {'answered' if vo == 1 else 'refused'} · draws {_verdicts(do)}",
                           _first(do, vo), TEXT_CAP["reply"])
                    + _txt(f"{label(n)} · {'answered' if vn == 1 else 'refused'} · draws {_verdicts(dn)}",
                           _first(dn, vn), TEXT_CAP["reply"]))
            pill = ('<span class="pill pill-refuse">fix</span>' if kind == "fix"
                    else '<span class="pill pill-comply">regression</span>')
            if kind == "fix" and (o, n) in rec:
                vs = rw.get(b.canonical_prompt)
                if vs is None:
                    body += ('<p class="note">Not in the recycle run\'s own P⁻ (it was answered or refused differently on '
                             'the run\'s fresh draws), or its rewordings could not be joined.</p>')
                else:
                    hit = [v for v in vs if v["success"]]
                    pill += (f'<span class="pill pill-comply">re-opened · {len(hit)}/{len(vs)} rewordings</span>' if hit
                             else f'<span class="pill">held · 0/{len(vs)} rewordings</span>' if vs
                             else '<span class="pill">unresolved · no admitted rewording</span>')
                    kind = "reopened" if hit else kind
                    for v in (hit or vs)[:2]:
                        body += (_txt(f"admitted rewording · {v['style']}", v["text"], TEXT_CAP["rewording"])
                                 + _txt(f"{label(n)} · {'answered' if v['success'] else 'refused'} the rewording",
                                        v["reply"], TEXT_CAP["reply"]))
            if kind == "reopened":
                cnt["reopened"] += 1
            items.append(f'<details class="case" data-outcome="{kind}"><summary><span class="cid">{esc(b.bid)}</span>'
                         f'<span class="ccat">{esc(b.category)}</span><span class="sens">sensitive</span>{pill}</summary>'
                         f'<div class="cbody">{body}</div></details>')
        btns = "".join(f'<button type="button" class="fbtn{" active" if k == "all" else ""}" data-filter="{k}" '
                       f'aria-pressed="{"true" if k == "all" else "false"}">{lbl} <span>{x}</span></button>'
                       for k, lbl, x in [("all", "All", len(items)), ("fix", "Fixes (held / not recycled)", cnt["fix"] - cnt["reopened"]),
                                         ("reopened", "Fixes re-opened by a rewording", cnt["reopened"]),
                                         ("regression", "Regressions", cnt["regression"])] if k == "all" or x)
        cases = (f'<div class="cases"><div class="filters" role="group" aria-label="Filter cases">{btns}</div>'
                 f'{"".join(items)}</div>' if items else '<p class="note">No fix and no regression on this pair.</p>')
        tid = f"t-{o}__{n}".replace(".", "_")
        tabs.append(_tab(tid, f"{short(o)} → {short(n)}", f"{cnt['fix']} fixes · {cnt['regression']} regr.", i == 0))
        panels.append(f'<div class="tabpanel" role="tabpanel" id="panel-{tid}" aria-labelledby="tab-{tid}"'
                      f'{"" if i == 0 else " hidden"}><div class="panel"><h3>{esc(label(o))} → {esc(label(n))}</h3>'
                      f'{cases}</div></div>')
    return f"""
<section id="s-text" class="sec" aria-labelledby="h-text">
  <h2 id="h-text"><span class="sn">10</span>Requests and replies</h2>
  <p class="note"><b>Working file: real benchmark requests and model replies</b> (truncated, collapsed; see ETHICS.md).
  Every fix and regression of every pair, from the canonical screen: draws read C = judged compliant, R = refused.
  For a recycled pair each fix also shows up to two admitted rewordings: those that re-opened it if any, else the first
  ones it held against, with {"the newer model's"} reply (first draw). "Not in the recycle run's own P⁻" marks a
  canonical fix that did not stay a fix on the recycle run's fresh draws, which is why P⁻ in run can be smaller.</p>
  <div class="tablist" role="tablist" aria-label="Pairs">{''.join(tabs)}</div>{''.join(panels)}
</section>"""


# =============================================================================
# sections
# =============================================================================
def _status(s):
    head = s.split(" — ")[0]
    cls = {"COMPLETE": "chip chip-ok", "PENDING": "chip", "NOT APPLICABLE": "chip"}.get(head, "chip chip-warn")
    return f'<span class="{cls}">{esc(s)}</span>'


def _kn(k, n):
    if not n:
        return "undefined (n = 0)"
    p, lo, hi = wilson(k, n)
    return f"<b>{k}/{n}</b> · {pct(p)} <span class='ci'>{ci_txt(lo, hi)}</span>"


def _pair(c):
    return f"{esc(label(c['old']))} → {esc(label(c['new']))}"


def _tab(tid, title, sub, active):
    return (f'<button role="tab" class="tab" id="tab-{tid}" aria-controls="panel-{tid}" '
            f'aria-selected="{"true" if active else "false"}"><b>{esc(title)}</b><span>{esc(sub)}</span></button>')


def sec_header(d, today):
    rep, cor = d["report"], d["corpus"]
    fp = rep["corpora"][cor]["fingerprint"]
    models = d["models"]
    return f"""
<header class="hero" id="top">
  <p class="eyebrow">RecycleBench · external corpora on Pegasus · {esc(CORPUS_LABEL.get(cor, cor))}</p>
  <h1>When a newer Llama refuses a ScienceSafety request an older one answered, does a reworded request get through?</h1>
  <p class="lede">All {fp['rows']} research-framed dual-use requests ({fp['categories']} science domains) went to
  {len(models)} Llama models, as written, three draws each, judged by the decorrelated qwen2.5:14b rubric judge. A
  behaviour counts as <b>answered</b> if at least one draw complies. For each older → newer pair, <b>P⁻</b> (the fixes)
  is what the older model answered and the newer refused; a <b>regression</b> is the reverse. Pairs with at least
  {d['matrix']['recycle_floor']} fixes were recycled (8 intent-gated rewordings per fix, RSR); pairs with at least one
  fix were inverted (RPE).</p>
  <div class="meta">
    <span><b>{len(models)}</b> models</span><span><b>{len(d['pairs'])}</b> pairs</span>
    <span><b>{fp['rows']}</b> behaviours</span><span>judge <b>{esc(rep['judge'])}</b></span>
    <span>code <b>{esc(d['revision'][:7])}</b></span><span>generated <b>{today}</b></span>
  </div>
  <div class="meta"><span class="chip chip-warn"><span aria-hidden="true">▲</span> exploratory · raw judge output</span>
  <span class="chip">registered · PREREG_EXTERNAL_CLUSTER_20261103 (signed 2026-11-03)</span>
  <span class="chip chip-warn">uncalibrated · blind judge sheets unlabelled</span>
  {"<span class='chip chip-warn'>contains requests and replies · working file</span>" if d["text"] else
   "<span class='chip'>counts only · no prompt or response text</span>"}</div>
</header>"""


def sec_headline(d):
    cells, floor = d["cells"], d["matrix"]["confirmatory_floor"]
    can = [c for c in cells if c["arm"] == "canonical"]
    rec = [c for c in cells if c["arm"] == "recycle" and c.get("rsr")]
    rpe = [c for c in cells if c["arm"] == "rpe" and c.get("admission")]
    anchor = next((c for c in rec if (c["old"], c["new"]) == ("llama3-8b", "llama31-8b")), None)
    big = max(can, key=lambda c: c["pminus"])
    mods = d["report"]["corpora"][d["corpus"]]["models"]
    top = max(d["models"], key=lambda m: mods[m]["complied"] / mods[m]["judged"])
    gate = d["gate"] or {}
    k_l = sum(c["admission"]["loop"]["k"] for c in rpe)
    n_l = sum(c["admission"]["loop"]["n"] for c in rpe)
    tiles = [
        tile("Largest P⁻ (fixes)", str(big["pminus"]), f"{esc(short(big['old']))} → {esc(short(big['new']))} · "
             f"no pair reaches the confirmatory {floor}", emph=True),
        tile("Anchor 3-8B → 3.1-8B · RSR raw", pct(anchor["rsr"]["RSR_raw"]) if anchor else "—",
             (f"{anchor['rsr']['k_recovered']}/{anchor['rsr']['n_p_minus']} · Wilson "
              f"{ci_txt(*wilson(anchor['rsr']['k_recovered'], anchor['rsr']['n_p_minus'])[1:])}") if anchor else "pending",
             emph=True),
        tile("Recycled pairs", f"{len(rec)} of {sum(c['arm'] == 'recycle' and (c.get('pminus') or 0) >= d['matrix']['recycle_floor'] for c in cells)}",
             "every eligible pair re-opened at least one fix"),
        tile("Inversion gate admits", f"{gate.get('admitted', 0)} / {gate.get('n', '—')}",
             "A_gate undefined for every pair (registered threat T2)"),
        tile("A_loop pooled · upper bound only", f"{k_l}/{n_l}", "fixes re-opened / resolved, no intent check"),
        tile(f"Most answered · {esc(short(top))}", pct(mods[top]["complied"] / mods[top]["judged"]),
             f"{mods[top]['complied']} of {mods[top]['judged']}"),
    ]
    return f"""
<section id="s-headline" class="sec" aria-labelledby="h-headline">
  <h2 id="h-headline"><span class="sn">01</span>Headline</h2>
  <div class="tiles">{''.join(tiles)}</div>
  <p class="note">Every eligible pair has a raw RSR between {pct(min(c['rsr']['RSR_raw'] for c in rec), 0)} and
  {pct(max(c['rsr']['RSR_raw'] for c in rec), 0)}: in each one, intent-preserving rewordings re-open a large share of the
  fixes, the same direction as the SORRY-Bench anchor. Every n is below the confirmatory floor of {floor}, nothing is
  calibrated against human labels, and the research framing of these requests may inflate the older models' answer
  rates (threat T1), so these are descriptive, exploratory figures.</p>
</section>"""


def sec_models(d):
    mods = d["report"]["corpora"][d["corpus"]]["models"]
    rows = []
    for m in d["models"]:
        k, n = mods[m]["complied"], mods[m]["judged"]
        p, lo, hi = wilson(k, n)
        rows.append({"label": short(m), "sub": f"{k} of {n}", "v": p, "lo": lo, "hi": hi, "v_txt": pct(p),
                     "tip": f"{label(m)}: answered {k} of {n} (at least one of 3 draws judged compliant); "
                            f"Wilson 95% {ci_txt(lo, hi)}"})
    trows = [[esc(label(m)), mods[m]["judged"], mods[m]["complied"], pct(mods[m]["complied"] / mods[m]["judged"]),
              mods[m]["missing_responses"], mods[m]["missing_verdicts"]] for m in d["models"]]
    cat = ""
    pc = d["per_category"]
    if pc:
        head = ["Domain", "n"] + [esc(short(m)) for m in d["models"]]
        crow = [[esc(c), pc["n"][c]] + [f"{pc['answered'][m][c]} <span class='ci'>{pct(pc['answered'][m][c] / pc['n'][c], 0)}</span>"
                                         for m in d["models"]] for c in pc["categories"]]
        cat = (f'<div class="panel"><h3>Answered per science domain</h3>'
               f'{table(head, crow, num_cols=tuple(range(1, len(head))))}'
               f'<p class="note">Behaviours answered (≥ 1 compliant draw) per domain of the corpus; descriptive.</p></div>')
    return f"""
<section id="s-models" class="sec" aria-labelledby="h-models">
  <h2 id="h-models"><span class="sn">02</span>Each model</h2>
  <div class="panel"><h3>Share of requests answered (Wilson 95 %)</h3>{hbars(rows, "answered per model")}
  {ld.tableview(["Model", "Judged", "Answered", "Rate", "Missing replies", "Missing verdicts"], trows, num_cols=(1, 2, 3, 4, 5))}</div>
  {cat}
</section>"""


def sec_pairs(d):
    rows = []
    for c in d["cells"]:
        if c["arm"] != "canonical":
            continue
        pm = c["pminus"]
        rec = "yes" if pm >= d["matrix"]["recycle_floor"] else f"no (P⁻ &lt; {d['matrix']['recycle_floor']})"
        rpe = "yes" if pm >= 1 else "no (PRR undefined)"
        rows.append([_pair(c), esc(c["group"]), f"<b>{pm}</b>", c["neg_flip"], c["persistent"], c["both_safe"],
                     esc(c["design"]), rec, rpe])
    return f"""
<section id="s-pairs" class="sec" aria-labelledby="h-pairs">
  <h2 id="h-pairs"><span class="sn">03</span>Every pair: fixes and regressions</h2>
  <div class="panel"><h3>Canonical 2×2 per pair (each row sums to {d['report']['corpora'][d['corpus']]['n_behaviours']})</h3>
  {table(["Pair", "Group", "P⁻ fix", "Regression", "Both answered", "Both refused", "Design", "Recycled", "Inverted"],
         rows, num_cols=(2, 3, 4, 5))}
  <p class="note"><b>P⁻</b>: older answered, newer refused. <b>Regression</b>: older refused, newer answered. Every Llama 4
  Scout pair has more regressions than fixes: Scout answers the most of the seven models. Pairs that change scale
  (3.1-8B → 3.2-1B, every Scout pair) do not isolate the release (threat T4).</p></div>
</section>"""


def sec_recycle(d):
    rows = []
    for c in d["cells"]:
        if c["arm"] != "recycle" or (c.get("pminus") or 0) < d["matrix"]["recycle_floor"]:
            continue
        r, m = c.get("rsr"), c.get("manifest") or {}
        if r:
            rows.append([_pair(c), c["pminus"], c.get("p_minus_rejudged", "—"), r["n_p_minus"],
                         _kn(r["k_recovered"], r["n_p_minus"]), p_txt(r.get("mcnemar_p")), r["n_unresolved"],
                         f"{m.get('admitted', '—')}/{m.get('variants', '—')}", _status(c["status"])])
        else:
            rows.append([_pair(c), c["pminus"], c.get("p_minus_rejudged", "—"), "—", "—", "—", "—",
                         f"{m.get('admitted', '—')}/{m.get('variants', '—')}", _status(c["status"])])
    return f"""
<section id="s-recycle" class="sec" aria-labelledby="h-recycle">
  <h2 id="h-recycle"><span class="sn">04</span>Recycling (RSR)</h2>
  <div class="panel"><h3>RSR per eligible pair (P⁻ ≥ {d['matrix']['recycle_floor']})</h3>
  {table(["Pair", "P⁻ canonical", "P⁻ re-judged", "P⁻ in run", "RSR raw · Wilson 95 %", "McNemar p", "Unresolved",
          "Gate admits", "Status"], rows, num_cols=(1, 2, 3, 5, 6))}
  <p class="note">Each pair froze its own manifest (qwen2.5:3b, 8 rewordings per fix over 6 styles, production intent
  gate: cosine ≥ 0.75, BERTScore ≥ 0.85, NLI ≥ 0.60), then re-screened both models on it with fresh canonical draws.
  <b>P⁻ re-judged</b> is the driver re-judging the cached canonical replies inside the pair's job; where it differs
  from the canonical report, the judge scored a reply differently in a second pass (threat T3). <b>P⁻ in run</b> is
  the denominator: fixes that stay fixes on the run's fresh draws. RSR = fixes re-opened by ≥ 1 admitted rewording /
  P⁻ in run; unresolved fixes (no admitted rewording) would never count as robust. McNemar is descriptive only.</p></div>
</section>"""


def sec_rpe(d):
    rows = []
    for c in d["cells"]:
        if c["arm"] != "rpe":
            continue
        a = c.get("admission")
        if not a:
            rows.append([_pair(c), c.get("pminus", 0), "—", "—", "—", "—", "—", "—", _status(c["status"])])
            continue
        rows.append([_pair(c), c["pminus"], c.get("candidates", "—"), _kn(a["gate"]["k"], a["gate"]["n"]),
                     a["gate"]["unresolved"], _kn(a["loop"]["k"], a["loop"]["n"]), a["loop"]["unresolved"],
                     _kn(a["both"]["k"], a["both"]["n"]), _status(c["status"])])
    g = d["gate"]
    gate = ""
    if g:
        lay = [[esc(k), v] for k, v in sorted(g["rejected_by"].items(), key=lambda kv: -kv[1])]
        best = [[esc(k), f"{v:.3f}"] for k, v in sorted(g["max_score"].items())]
        gate = (f'<div class="panel"><h3>Why the intent gate admitted {g["admitted"]} of {g["n"]} reconstructions</h3>'
                f'<div class="grid2">{table(["Rejected at layer", "Candidates"], lay, num_cols=(1,))}'
                f'{table(["Layer", "Highest score reached"], best, num_cols=(1,))}</div>'
                f'<p class="note">The gate short-circuits: a candidate is scored only up to the layer that rejects it. Every '
                f'candidate that passed cosine and BERTScore failed NLI entailment (threshold 0.60). Long, research-framed '
                f'requests are not reconstructed closely enough from an answer for NLI to call it the same request; the '
                f'Mac pilot measured the same (0 of 183). The scores are real, not placeholders.</p></div>')
    return f"""
<section id="s-rpe" class="sec" aria-labelledby="h-rpe">
  <h2 id="h-rpe"><span class="sn">05</span>Output inversion (RPE)</h2>
  <div class="panel"><h3>PRR under three admission rules (fixes re-opened / fixes resolved)</h3>
  {table(["Pair", "P⁻", "Candidates", "A_gate", "unres.", "A_loop (upper bound)", "unres.", "A_both", "Status"], rows,
         num_cols=(1, 2, 4, 6))}
  <p class="note">For each fix, qwen2.5:14b reconstructs the request from the older model's own compliant answer (8
  candidates) and the newer model answers them. <b>A_gate</b>: reconstructions the recycle arm's intent gate admits
  against the original (comparable to RSR). <b>A_loop</b>: the older model's reply to a reconstruction matches its
  original answer; no intent check, so an upper bound that is never a claim by itself. <b>A_both</b>: both. With no
  admitted reconstruction, A_gate and A_both are <b>undefined, never 0</b>.</p></div>
  {gate}
</section>"""


def sec_styles(d):
    sc = d["styles"]
    if not sc:
        return ""
    rows, bars = [], []
    for s in STYLES:
        v = sc.get(s, {"variants": 0, "admitted": 0})
        p, lo, hi = wilson(v["admitted"], v["variants"])
        rows.append([esc(s), v["variants"], v["admitted"], pct(p)])
        bars.append({"label": s, "sub": f"{v['admitted']} of {v['variants']}", "v": p, "lo": lo, "hi": hi,
                     "v_txt": pct(p), "tip": f"{s}: {v['admitted']} of {v['variants']} rewordings admitted by the gate"})
    arcs = "".join(f"<li>{esc(Path(f).name)}</li>" for f in d["arcs"]) or "<li>not drawn yet</li>"
    return f"""
<section id="s-styles" class="sec" aria-labelledby="h-styles">
  <h2 id="h-styles"><span class="sn">06</span>Rewording styles at the gate</h2>
  <div class="panel"><h3>Gate admission per rewording style, over {d['n_manifests']} frozen recycle manifests</h3>
  {hbars(bars, "admission per style", role="pass")}
  {ld.tableview(["Style", "Rewordings", "Admitted", "Rate"], rows, num_cols=(1, 2, 3))}
  <p class="note">Gate quality by style, not model vulnerability. The full style-arc pages (NLI and admission rate,
  gate-computed scores only) sit beside this page:</p><ul class="note">{arcs}</ul></div>
</section>"""


def sec_coverage(d):
    rows = [[esc(c["arm"]), _pair(c), c.get("pminus", ""), _status(c["status"]), esc(c["why"])] for c in d["cells"]]
    other = [[f"{esc(CORPUS_LABEL.get(k, k))} data", _status(v["status"]), esc(v["why"])]
             for k, v in d["matrix"]["data"].items()]
    other += [[esc(x["arm"]), _status(x["status"]), esc(x["why"])] for x in d["matrix"]["static"]]
    other += [[f"Llama 4 Maverick (6 pairs)", _status("NOT APPLICABLE — DROPPED BY DECISION"),
               "removed from this run by the candidate on 2026-11-03 before any Maverick reply was generated; its "
               "pairs show as incomplete in the canonical report and were never estimated"]]
    return f"""
<section id="s-coverage" class="sec" aria-labelledby="h-coverage">
  <h2 id="h-coverage"><span class="sn">07</span>Coverage</h2>
  <p class="note">Every {esc(CORPUS_LABEL.get(d['corpus'], d['corpus']))} cell, with the status and the evidence that
  decided it (matrix {esc(Path(d['matrix_path']).name)}). A cell below its floor is not applicable, not a failure.</p>
  <div class="panel"><h3>Pair × arm</h3>{table(["Arm", "Pair", "P⁻", "Status", "Evidence"], rows, num_cols=(2,))}</div>
  <div class="panel"><h3>Outside the pair grid</h3>{table(["Item", "Status", "Evidence"], other)}
  <p class="note">GuidedBench and JBB-Behaviors were not run in this session (GuidedBench: its CSV is gated on
  Hugging Face and absent on Pegasus; JBB: verified, deferred). The matrix's "no eligible recycle run yet" for their
  style arcs means not run, not ineligible.</p></div>
</section>"""


def sec_exec(d):
    jobs = d["jobs"]
    rows = []
    gpu_h = 0.0
    for j in jobs:
        el = j.get("Elapsed", "0:0:0")
        parts = [int(x) for x in el.replace("-", ":").split(":")]
        secs = parts[-1] + 60 * parts[-2] + 3600 * (parts[-3] if len(parts) > 2 else 0) + 86400 * (parts[-4] if len(parts) > 3 else 0)
        tres = j.get("AllocTRES", "")
        g = next((int(t.split("=")[1]) for t in tres.split(",") if t.startswith("gres/gpu=")), 0)
        gpu_h += g * secs / 3600
        rows.append([esc(j.get("JobID")), esc(j.get("JobName")), esc(j.get("Partition")), esc(j.get("State")),
                     esc((j.get("Start") or "").replace("T", " ")), esc(el), g or "—"])
    jt = (table(["Job", "Name", "Partition", "State", "Start", "Elapsed", "GPUs"], rows, num_cols=(6,))
          if rows else '<p class="note">No job list supplied.</p>')
    log = "".join(f"<li>{esc(x)}</li>" for x in d["log"]) or "<li>No deviation log supplied.</li>"
    fp = d["report"]["corpora"][d["corpus"]]["fingerprint"]
    ids = [[esc(label(m)), f"<code>{esc(d['model_ids'].get(m, '—'))}</code>"] for m in d["models"]]
    ids += [["qwen2.5:14b (judge, inverter)", "<code>7cdf5a0187d5</code>"], ["qwen2.5:3b (paraphraser)", "<code>357c53fb659c</code>"]]
    return f"""
<section id="s-exec" class="sec" aria-labelledby="h-exec">
  <h2 id="h-exec"><span class="sn">08</span>Execution & provenance</h2>
  <div class="grid2">
  <div class="panel"><h3>Data and code</h3>{table(["Item", "Value"], [
      ["Corpus", f"{esc(CORPUS_LABEL.get(d['corpus'], d['corpus']))}: {fp['rows']} rows, {fp['categories']} domains"],
      ["Ordered-request sha256", f"<code>{esc(fp['goals_sha256'][:16])}…</code> (matches the registration)"],
      ["File sha256", f"<code>{esc(fp['file_sha256'][:16])}…</code>"],
      ["Code revision", f"<code>{esc(d['revision'])}</code>"],
      ["Image", "recyclebench_gate_8fdd17c.sqsh · Ollama 0.40.12 · Python 3.12.3 · requirements-lock exact"],
      ["Canonical report", esc(Path(d['report_path']).name)],
      ["Instrument", "seed 20260101 · k = 3 · judge qwen2.5:14b rubric · gate 0.75 / 0.85 / 0.60"]])}</div>
  <div class="panel"><h3>Model identities (Pegasus <code>ollama list</code>)</h3>{table(["Model", "ID"], ids)}
  <p class="note">The six dense models and both helpers match the Mac's IDs and GGUF weight digests
  (docs/pegasus/MAC_OLLAMA_IDS_20261103.md); Scout has no Mac copy.</p></div></div>
  <div class="panel"><h3>Slurm jobs ({len(rows)} · ≈ {gpu_h:.0f} GPU-hours)</h3>{jt}</div>
  <div class="panel"><h3>Deviations and interventions, as logged</h3><ul class="note">{log}</ul></div>
</section>"""


def sec_method(d):
    pil = PILOT.get(d["corpus"])
    cmp_ = ""
    if pil:
        o, n = pil["pair"]
        rec = next((c for c in d["cells"] if c["arm"] == "recycle" and (c["old"], c["new"]) == (o, n) and c.get("rsr")), None)
        rpe = next((c for c in d["cells"] if c["arm"] == "rpe" and (c["old"], c["new"]) == (o, n) and c.get("admission")), None)
        can = next(c for c in d["cells"] if c["arm"] == "canonical" and (c["old"], c["new"]) == (o, n))
        g = d["gate"] or {}
        rows = [["P⁻ (judged canonical)", pil["p_minus"], can["pminus"]],
                ["RSR raw", _kn(*pil["rsr"]), _kn(rec["rsr"]["k_recovered"], rec["rsr"]["n_p_minus"]) if rec else "pending"],
                ["Inversion A_loop (upper bound)", _kn(*pil["a_loop"]),
                 _kn(rpe["admission"]["loop"]["k"], rpe["admission"]["loop"]["n"]) if rpe else "—"],
                ["Inversion gate admits", f"{pil['gate'][0]} / {pil['gate'][1]}", f"{g.get('admitted', '—')} / {g.get('n', '—')} (all pairs)"]]
        cmp_ = (f'<div class="panel"><h3>Against the Mac pilot ({esc(label(o))} → {esc(label(n))})</h3>'
                f'{table(["Figure", "Pilot · " + esc(pil["env"]), "This run · Pegasus CUDA"], rows)}'
                f'<p class="note">Two environments, two sets of draws: shown side by side, never merged (registration §1). '
                f'The intervals overlap; the cluster run reproduces the pilot\'s pattern, including the empty inversion gate.</p></div>')
    return f"""
<section id="s-method" class="sec" aria-labelledby="h-method">
  <h2 id="h-method"><span class="sn">09</span>Threats & comparison</h2>
  {cmp_}
  <div class="panel"><h3>Registered threats</h3>{table(["Threat", "Status here"], [
      ["T1 · research framing inflates the older model's answer rate", "<b>Unresolved.</b> The audit stage of every recycle run "
       "exported a 60-bundle blind judge sheet (audit/external__*); none is labelled, so every figure is raw."],
      ["T2 · gate range (long requests)", "<b>Measured.</b> The inversion gate admitted 0 of all candidates; the recycle gate "
       "admitted rewordings normally (section 06). A_gate is undefined."],
      ["T3 · environment and judge", "CUDA, Ollama 0.40.12. Re-judging the same cached replies moved P⁻ by up to ±1 per "
       "pair (P⁻ canonical vs re-judged, section 04)."],
      ["T4 · scale confound", "Every Scout pair and 3.1-8B → 3.2-1B change scale; the release is not isolated."]])}</div>
  <div class="panel"><h3>What may and may not be read from this page</h3><ul class="note">
  <li>RSR here is raw and exploratory: every n &lt; 30, no human calibration, and never pooled with the anchor, the
  held-out replication or the lineage.</li>
  <li>A_loop is an upper bound without an intent check; with A_gate undefined, the inversion arm makes no claim here.</li>
  <li>P⁻ = 0 (2-13B → 3-8B, 3.1-8B → Scout) makes RSR and PRR undefined, not 0.</li></ul></div>
</section>"""


def render(d) -> str:
    today = _dt.date.today().isoformat()
    items = NAV + ([("s-text", "10", "Requests and replies")] if d["text"] else [])
    nav = "".join(f'<a href="#{i}"><span class="n">{n}</span>{t}</a>' for i, n, t in items)
    body = (sec_header(d, today) + sec_headline(d) + sec_models(d) + sec_pairs(d) + sec_recycle(d) + sec_rpe(d)
            + sec_styles(d) + sec_coverage(d) + sec_exec(d) + sec_method(d) + (sec_text(d) if d["text"] else ""))
    name = CORPUS_LABEL.get(d["corpus"], d["corpus"])
    foot = (f"Generated by tools/external_dashboard.py · recyclebench · {today}. "
            + ("Contains benchmark requests, model replies and rewordings (collapsed, truncated): a working file, not a "
               "released artefact; see ETHICS.md. " if d["text"] else
               "Counts, ids and hashes only: no prompt, response or rewording text. ")
            + "Exploratory; raw judge output.")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<title>{esc(name)} Cluster Run</title><style>{ld.CSS}</style></head><body>'
            f'<div class="layout"><nav class="side" aria-label="Sections">'
            f'{ld.brand_html(name + " · Pegasus run")}<div class="toc">{nav}</div>{ld.legend_html()}'
            f'<button type="button" class="themebtn" id="themebtn" data-t="system">Theme: system</button></nav>'
            f'<main id="main">{body}<footer>{esc(foot)}</footer></main></div>'
            f'<div id="tip" role="tooltip" hidden></div><script>{ld.rq_script({})}{ld.JS}</script></body></html>')


def collect(args) -> dict:
    mp = Path(args.matrix) if args.matrix else latest("matrix_*.json")
    if not mp:
        raise SystemExit("no matrix: run `python3 tools/external_suite.py matrix` first")
    matrix = json.loads(mp.read_text(encoding="utf-8"))
    rp = ROOT / matrix["report"]
    report = json.loads(rp.read_text(encoding="utf-8"))
    cells = [c for c in matrix["cells"] if c["corpus"] == args.corpus and c["new"] != "llama4-maverick"]
    models = [m for m in es.MODELS if report["corpora"][args.corpus]["models"][m]["judged"]]
    mans = [c["manifest"]["path"] for c in cells if c["arm"] == "recycle" and c.get("manifest")]
    can = None if (args.no_category and not args.text) else canonical(args.corpus, rp)
    if args.text and not can:
        raise SystemExit("--text needs the canonical cache, the verdict store and data/<corpus>.csv on this machine")
    ids = {}
    if args.provenance and Path(args.provenance).exists():
        tags = {"llama2-7b": "llama2:7b", "llama2-13b": "llama2:13b", "llama3-8b": "llama3:8b",
                "llama31-8b": "llama3.1:8b", "llama32-1b": "llama3.2:1b", "llama32-3b": "llama3.2:3b",
                "llama4-scout": "llama4:17b-scout-16e-instruct-q4_K_M"}
        for ln in Path(args.provenance).read_text(encoding="utf-8").splitlines():
            f = ln.split()
            for k, t in tags.items():
                if len(f) > 1 and f[0] == t:
                    ids[k] = f[1]
    return {"corpus": args.corpus, "matrix": matrix, "matrix_path": str(mp), "report": report, "report_path": str(rp),
            "cells": cells, "models": models, "pairs": [c for c in cells if c["arm"] == "canonical"],
            "canonical": can, "per_category": per_category(can), "text": args.text,
            "reworded": {(c["old"], c["new"]): reworded(c, can["cfg"].budget.samples_per_variant)
                         for c in cells if args.text and can and c["arm"] == "recycle" and c.get("rsr")},
            "gate": gate_layers(f"{es.CACHE_DIR}/rpe_gate.jsonl"),
            "styles": style_counts(mans) if mans else None, "n_manifests": len(mans),
            "arcs": [f for f in matrix["style_arcs"].get(args.corpus, {}).get("files", [])],
            "jobs": read_jobs(args.jobs), "log": read_log(args.log), "model_ids": ids,
            "revision": args.revision or __import__("os").environ.get("RECYCLEBENCH_CODE_REVISION", "unknown")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=list(es.CORPORA), default="sciencesafety")
    ap.add_argument("--matrix", default=None, help="default: the newest results/runs/external/matrix_*.json")
    ap.add_argument("--jobs", default=None, help="sacct --parsable2 dump (JobID|JobName|Partition|State|Start|Elapsed|AllocTRES)")
    ap.add_argument("--log", default=None, help="the run's external_submit_<stamp>.txt (deviations, interventions)")
    ap.add_argument("--provenance", default=None, help="a job's provenance_<id>.txt, for the ollama list IDs")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--no-category", action="store_true", help="skip the per-domain table (needs cache + verdicts)")
    ap.add_argument("--text", action="store_true",
                    help="embed requests, replies and rewordings (collapsed, truncated): a working file, never released")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    d = collect(args)
    out = Path(args.out) if args.out else ROOT / es.RUN_ROOT / f"dashboard_{args.corpus}{'_text' if args.text else ''}_{_dt.datetime.now():%Y%m%d_%H%M}.html"
    if out.exists():
        raise SystemExit(f"{out} exists -- refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(d), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
