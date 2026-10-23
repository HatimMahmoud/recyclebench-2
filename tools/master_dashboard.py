#!/usr/bin/env python3
"""tools/master_dashboard.py -- every RecycleBench result in one page.

All models, all benchmarks, every recycle run, every RPE run, the style arcs, both attack-transfer
models, both human judge calibrations, the intent gate, the exploratory framing runs, and an
integrity registry of every artifact on disk. Built entirely from files (tools/study_data.py does
the discovery, integrity checks and recomputation); no model is called. Cells with no data say
"not yet finished". The page embeds no prompt or response text, so it can be shared as is; the
case-level evidence for SORRY-Bench lives in the lineage dashboard, linked from here.

  python3 tools/master_dashboard.py
  python3 tools/master_dashboard.py --attack-70b-responses "~/Downloads/all_local_results_llama3_1_70b 1.jsonl"

Writes results/dashboard_master_<YYYYmmdd_HHMM>.html and never overwrites an existing file.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_core as lc                       # noqa: E402
import lineage_dashboard as ld                  # noqa: E402  (shared chart/table primitives, CSS, JS)
import study_data as sd                         # noqa: E402
from recyclebench.stats import wilson_interval  # noqa: E402

esc, pct, ci_txt, p_txt, num = ld.esc, ld.pct, ld.ci_txt, ld.p_txt, ld.num
table, tableview, tile, heat = ld.table, ld.tableview, ld.tile, ld.heat

DS_ORDER = ["sorrybench", "harmbench", "advbench", "strongreject", "llama32jb", "jbb", "guidedbench"]
NOT_YET = "not yet finished"


def label(m):
    return lc.meta(m)["label"]


def short(m):
    return lc.meta(m)["short"]


def pair_short(o, n):
    return f"{short(o)} → {short(n)}"


def dsl(s, k):
    return s.datasets[k].label if k in s.datasets else k


def wil(k, n):
    if not n:
        return None, None, None
    w = wilson_interval(k, n)
    return w.point, w.low, w.high


def judge_chip(j: str) -> str:
    j = j or "unknown"
    if j.startswith("offline"):
        return '<span class="jchip j-off" title="offline refusal-prefix rule">offline rule</span>'
    if j.startswith("llm"):
        return '<span class="jchip j-llm" title="qwen2.5:14b rubric judge">qwen2.5:14b</span>'
    if j.startswith("target"):
        return '<span class="jchip j-circ" title="target model grading itself">target-as-judge ✕</span>'
    if j.startswith("human"):
        return '<span class="jchip j-hum">human</span>'
    if "llm" in j or "qwen" in j:
        return f'<span class="jchip j-llm">{esc(j)}</span>'
    return f'<span class="jchip">{esc(j)}</span>'


STATUS_STYLE = {
    "valid": ("ok", "●", "valid"), "complete": ("ok", "●", "complete"), "from-log": ("ok", "●", "valid (log)"), "judge-replay": ("ok", "●", "judge replay"),
    "superseded": ("muted", "◌", "superseded"), "stale-copy": ("warn", "▲", "stale copy"),
    "invalid": ("crit", "✕", "invalid"), "circular-judge": ("crit", "✕", "circular judge"),
    "incomplete": ("warn", "◐", "incomplete"), "empty": ("muted", "○", "empty"), "mock": ("muted", "◌", "mock"),
    "missing": ("muted", "○", "missing"), "partial": ("warn", "◐", "partial"), "none": ("muted", "○", NOT_YET),
    "conditional": ("warn", "◐", "conditional"), "na": ("muted", "–", "n/a"),
}


def status_chip(st: str, text: str = None) -> str:
    cls, icon, lab = STATUS_STYLE.get(st, ("muted", "•", st))
    return f'<span class="st st-{cls}"><span aria-hidden="true">{icon}</span> {esc(text or lab)}</span>'


def path_html(p) -> str:
    """Compact, wrap-safe path: file name on one line, its folder (breakable at / and _) below."""
    p = str(p)
    d, _, f = p.rpartition("/")
    dd = esc(d).replace("/", "/<wbr>").replace("_", "_<wbr>")
    return (f'<span class="pathc" title="{esc(p)}"><span class="pf">{esc(f)}</span>'
            + (f'<span class="pd">{dd}/</span>' if d else "") + "</span>")


def mtable(head, rows, num=(), nowrap=(), wide=(), cls="tbl"):
    def c(i, tag):
        k = [x for x, on in (("num", i in num), ("nw", i in nowrap), ("wide", i in wide)) if on]
        attr = f' class="{" ".join(k)}"' if k else ""
        return f"<{tag}{attr}" + (' scope="col"' if tag == "th" else "") + ">"
    h = "".join(f"{c(i, 'th')}{x}</th>" for i, x in enumerate(head))
    b = "".join("<tr>" + "".join(f"{c(i, 'td')}{x}</td>" for i, x in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tscroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def tview(summary, head, rows, **kw):
    return f'<details class="tv"><summary>{esc(summary)}</summary>{mtable(head, rows, **kw)}</details>'


def section(sid, num_, title, body, lede=""):
    lede_html = f'<p class="note lede2">{lede}</p>' if lede else ""
    return (f'<section id="{sid}" class="sec" aria-labelledby="h-{sid}"><h2 id="h-{sid}"><span class="sn">{num_}</span>'
            f'{title}</h2>{lede_html}{body}</section>')


def panel(title, body, note=""):
    n = f'<p class="note">{note}</p>' if note else ""
    return f'<div class="panel"><h3>{title}</h3>{n}{body}</div>'


# =============================================================================
# Derived figures (pure functions of the Study)
# =============================================================================
def recycle_cells(s):
    """One recycle result per (pair, dataset): full-scope runs before subsets, never an incomplete pair.
    -> {(o, n, ds): (RecycleRun, pair_dict)}"""
    order = sorted(s.recycle, key=lambda r: (r.spec.behaviour_scope != "full", r.spec.id))
    out = {}
    for rr in order:
        for (o, n), pr in rr.pairs.items():
            if not pr["complete"] or pr["recon"] is None:
                continue
            out.setdefault((o, n, rr.spec.dataset), (rr, pr))
    return out


def pooled_offline(s):
    cells = recycle_cells(s)
    k = sum(pr["recon"]["k"] for _, pr in cells.values())
    n = sum(pr["recon"]["n"] for _, pr in cells.values())
    return k, n, len(cells), sorted({ds for (_, _, ds) in cells})


def _max_transfer(s):
    cl = (s.attack.get("8b_original") or {}).get("classes") or {}
    return max((v.get("transfer_on_refused") or 0) for v in cl.values()) if cl else None


def judge_gap(s):
    """(#pairs where the offline pilot P^- exceeds the qwen-judged run's, #pairs compared) on SORRY-Bench."""
    lin = lineage_rows(s)
    more = tot = 0
    for (o, n), r in lin.items():
        c = s.cells.get((o, n, "sorrybench"))
        if c and c.status in ("complete", "p-minus-only"):
            tot += 1
            more += int(len(c.p_minus) > r["n_p_minus"])
    return more, tot


def lineage_rows(s):
    lin = s.lineage or {}
    return {tuple(x.strip() for x in r["pair"].split("->")): r for r in lin.get("RSR_primary_RAW", [])}


def anchor(s):
    rr = next((r for r in s.recycle if r.spec.id == "anchor-sorrybench"), None)
    return rr, (rr.calibrated if rr else None)


# =============================================================================
# Charts specific to this page
# =============================================================================
def forest_grouped(groups, title, x_max=1.0, ref=None, ref_label=""):
    """groups: [(group_label, [row dicts])]; row: label, sub, point, lo, hi, corr, emph, tip, v_txt, n_txt, hollow."""
    W, rh, gh, top = 860, 36, 30, 16
    x0, x1 = 300, 720
    n_rows = sum(len(r) for _, r in groups)
    H = top + n_rows * rh + len(groups) * gh + 30
    y_bot = H - 30
    s = [ld.svg_open(W, H, title)]
    ticks = [t / 4 * x_max for t in range(5)]
    for t in ticks:
        x = x0 + (x1 - x0) * t / x_max
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{y_bot}" class="grid"/>'
                 f'<text x="{x:.1f}" y="{y_bot + 16}" class="ax" text-anchor="middle">{int(round(t * 100))}%</text>')
    if ref is not None:
        xr = x0 + (x1 - x0) * ref / x_max
        s.append(f'<line x1="{xr:.1f}" x2="{xr:.1f}" y1="{top - 6}" y2="{y_bot}" class="refline"/>'
                 f'<text x="{xr + 4:.1f}" y="{top + 4}" class="ax">{esc(ref_label)}</text>')
    y = top

    def X(v):
        return x0 + (x1 - x0) * max(0.0, min(v, x_max)) / x_max
    for g, rows in groups:
        s.append(f'<text x="8" y="{y + 20}" class="glabel">{esc(g)}</text>'
                 f'<line x1="8" x2="{W - 8}" y1="{y + 26}" y2="{y + 26}" class="axis"/>')
        y += gh
        for r in rows:
            cy = y + rh / 2
            cls = "emph" if r.get("emph") else "muted"
            s.append(f'<g class="row {cls}" tabindex="0" data-tip="{esc(r["tip"])}">'
                     f'<rect x="0" y="{y}" width="{W}" height="{rh}" class="hit"/>'
                     f'<text x="16" y="{cy - 3:.1f}" class="lbl strong">{esc(r["label"])}</text>'
                     f'<text x="16" y="{cy + 11:.1f}" class="ax">{esc(r.get("sub", ""))}</text>')
            if r.get("point") is not None:
                if r.get("lo") is not None and r.get("hi") is not None:
                    a, b = X(r["lo"]), X(r["hi"])
                    s.append(f'<line x1="{a:.1f}" x2="{b:.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="whisk"/>'
                             f'<line x1="{a:.1f}" x2="{a:.1f}" y1="{cy - 4:.1f}" y2="{cy + 4:.1f}" class="whisk"/>'
                             f'<line x1="{b:.1f}" x2="{b:.1f}" y1="{cy - 4:.1f}" y2="{cy + 4:.1f}" class="whisk"/>')
                cls_pt = "pt-hollow" if r.get("hollow") else "pt"
                s.append(f'<circle cx="{X(r["point"]):.1f}" cy="{cy:.1f}" r="5.5" class="{cls_pt}"/>')
                if r.get("corr") is not None:
                    xc = X(r["corr"])
                    s.append(f'<path d="M{xc:.1f},{cy - 6:.1f} L{xc + 6:.1f},{cy:.1f} L{xc:.1f},{cy + 6:.1f} '
                             f'L{xc - 6:.1f},{cy:.1f} Z" class="corr"/>')
            else:
                s.append(f'<text x="{x0}" y="{cy + 4:.1f}" class="ax">{esc(r.get("empty", NOT_YET))}</text>')
            s.append(f'<text x="{x1 + 16}" y="{cy - 3:.1f}" class="val">{esc(r.get("v_txt", ""))}</text>'
                     f'<text x="{x1 + 16}" y="{cy + 11:.1f}" class="ax">{esc(r.get("n_txt", ""))}</text></g>')
            y += rh
    s.append("</svg>")
    return "".join(s)


def arcs_svg(classes, title, show_naive=True):
    """Style arcs: one dot per rewording style on concentric semicircles; radius = mean NLI entailment
    of the rewordings the gate actually scored (placeholders excluded); dot area ~ count. A hollow ring
    marks the placeholder-inflated mean the old dashboard drew, joined by a thin line."""
    W, H = 460, 236
    cx, base = W / 2, 200
    r0, R = 36, 150
    lo, hi = 0.4, 1.0

    def rad(v):
        return r0 + (R - r0) * (max(lo, min(hi, v)) - lo) / (hi - lo)
    s = [ld.svg_open(W, H, title)]
    for g in (0.6, 0.8, 1.0):
        rr = rad(g)
        s.append(f'<path d="M{cx - rr:.1f},{base} A{rr:.1f},{rr:.1f} 0 0,1 {cx + rr:.1f},{base}" class="arcgrid"/>'
                 f'<text x="{cx + rr + 3:.1f}" y="{base + 12}" class="ax">{g:.1f}</text>')
    s.append(f'<line x1="{cx - R - 6}" x2="{cx + R + 6}" y1="{base}" y2="{base}" class="axis"/>')
    items = [c for c in classes if c["n_eval"] or c["n_naive"]]
    m = len(items)
    nmax = max([c["n_eval"] for c in items] + [1])
    for i, c in enumerate(items):
        ang = math.radians(180 - (i + 0.5) * 180 / max(m, 1))
        ux, uy = math.cos(ang), math.sin(ang)
        if c["nli_eval"] is not None:
            rr = rad(c["nli_eval"])
            x, y = cx + rr * ux, base - rr * uy
            size = 4 + 9 * math.sqrt(c["n_eval"] / nmax)
            if show_naive and c["nli_naive"] is not None and abs(c["nli_naive"] - c["nli_eval"]) > 0.005:
                rn = rad(c["nli_naive"])
                xn, yn = cx + rn * ux, base - rn * uy
                s.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{xn:.1f}" y2="{yn:.1f}" class="arclink"/>'
                         f'<circle cx="{xn:.1f}" cy="{yn:.1f}" r="4.5" class="arcnaive"/>')
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" class="arcdot" tabindex="0" data-tip="'
                     f'{esc(c["name"])}\nmean NLI of scored rewordings {c["nli_eval"]:.3f} (n={c["n_eval"]})'
                     + (f'\nplaceholder-inflated mean {c["nli_naive"]:.3f} (n={c["n_naive"]})' if c["nli_naive"] is not None else "")
                     + f'\nadmitted {c["adm"]}/{c["gen"]} ({pct(c["adm"] / c["gen"] if c["gen"] else None, 0)})"/>')
        lx, ly = cx + (R + 14) * ux, base - (R + 14) * uy
        anchor_ = "end" if ux < -0.2 else ("start" if ux > 0.2 else "middle")
        s.append(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" class="lbl" text-anchor="{anchor_}">{esc(c["name"].replace("_", " "))}</text>')
    s.append("</svg>")
    return "".join(s)


def dumbbell(rows, title, x_max, series):
    """rows: label, sub, vals {series_key: (rate, k, n) | None}, flag; series: [(key, css, legend)]."""
    W, rh, top = 860, 24, 14
    x0, x1 = 230, 770
    H = top + rh * len(rows) + 34
    y_bot = H - 30
    s = [ld.svg_open(W, H, title)]
    step = next(st for st in (0.05, 0.1, 0.2, 0.25) if x_max / st <= 6)
    t = 0.0
    while t <= x_max + 1e-9:
        x = x0 + (x1 - x0) * t / x_max
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{y_bot}" class="grid"/>'
                 f'<text x="{x:.1f}" y="{y_bot + 16}" class="ax" text-anchor="middle">{t * 100:.0f}%</text>')
        t += step

    def X(v):
        return x0 + (x1 - x0) * max(0.0, min(v, x_max)) / x_max
    for i, r in enumerate(rows):
        y = top + i * rh
        cy = y + rh / 2
        tip = [r["label"] + (" · low-readability cipher class" if r.get("flag") else "")]
        pts = []
        for key, css, leg in series:
            v = r["vals"].get(key)
            if v is None:
                tip.append(f"{leg}: {NOT_YET}")
                continue
            rate, k, n = v
            tip.append(f"{leg}: {pct(rate)} ({k}/{n})")
            pts.append((X(rate), css))
        s.append(f'<g class="row" tabindex="0" data-tip="{esc(chr(10).join(tip))}">'
                 f'<rect x="0" y="{y}" width="{W}" height="{rh}" class="hit"/>'
                 f'<text x="8" y="{cy + 4:.1f}" class="lbl{" strong" if not r.get("flag") else ""}">'
                 f'{esc(r["label"])}{" ⚠" if r.get("flag") else ""}</text>')
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            s.append(f'<line x1="{min(xs):.1f}" x2="{max(xs):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="dbline"/>')
        for x, css in pts:
            if css == "db-o70":
                s.append(f'<path d="M{x:.1f},{cy - 6:.1f} L{x + 6:.1f},{cy:.1f} L{x:.1f},{cy + 6:.1f} L{x - 6:.1f},{cy:.1f} Z" class="{css}"/>')
            else:
                s.append(f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="5" class="{css}"/>')
        for key, css, leg in series:                     # off-scale values: an arrow at the edge + the value
            v = r["vals"].get(key)
            if v is not None and v[0] > x_max:
                s.append(f'<path d="M{x1 + 4:.1f},{cy - 5:.1f} L{x1 + 12:.1f},{cy:.1f} L{x1 + 4:.1f},{cy + 5:.1f} Z" class="{css}"/>'
                         f'<text x="{x1 + 16:.1f}" y="{cy + 4:.1f}" class="ax">{pct(v[0], 0)}</text>')
        s.append("</g>")
    s.append("</svg>")
    return "".join(s)


def stacked_bars(rows, keys, title):
    """rows: label, sub, parts {key: count}, total; keys: [(key, css, legend)] -> horizontal 100% stacks."""
    W, rh, top = 860, 34, 10
    x0, x1 = 250, 800
    H = top + rh * len(rows) + 10
    s = [ld.svg_open(W, H, title)]
    for i, r in enumerate(rows):
        y = top + i * rh
        tot = max(r["total"], 1)
        s.append(f'<text x="8" y="{y + 15}" class="lbl strong">{esc(r["label"])}</text>'
                 f'<text x="8" y="{y + 28}" class="ax">{esc(r.get("sub", ""))}</text>')
        x = x0
        for key, css, leg in keys:
            v = r["parts"].get(key, 0)
            if not v:
                continue
            w = (x1 - x0) * v / tot
            s.append(f'<rect x="{x:.1f}" y="{y + 6}" width="{max(w - 2, 1):.1f}" height="18" rx="3" class="{css}" '
                     f'tabindex="0" data-tip="{esc(r["label"])} · {esc(leg)}: {v:,} of {r["total"]:,} ({pct(v / tot)})"/>')
            if w > 34:
                s.append(f'<text x="{x + 6:.1f}" y="{y + 19}" class="inbar">{v:,}</text>')
            x += w
    s.append("</svg>")
    return "".join(s)


def confusion(c, title):
    """2x2 judge-vs-human grid from a rates dict (tp/fn/tn/fp)."""
    return (f'<div class="cm" role="img" aria-label="{esc(title)}"><div class="cm-t">{esc(title)}</div>'
            f'<div></div><div class="cm-h">judge: complied</div><div class="cm-h">judge: refused</div>'
            f'<div class="cm-h cm-r">human: complied</div><div class="cm-c cm-ok"><b>{c["tp"]}</b>true pos.</div>'
            f'<div class="cm-c cm-bad"><b>{c["fn"]}</b>missed</div>'
            f'<div class="cm-h cm-r">human: refused</div><div class="cm-c cm-bad"><b>{c["fp"]}</b>false alarm</div>'
            f'<div class="cm-c cm-ok"><b>{c["tn"]}</b>true neg.</div></div>')


# =============================================================================
# Sections
# =============================================================================
def sec_header(s, today):
    n_resp = sum(v for k, v in s.overlap["entries"].items())
    n_runs = len({a.path.split("/")[2] for a in s.registry if a.path.startswith("results/runs/")})
    return f"""
<header class="hero" id="top">
  <p class="eyebrow">RecycleBench · master results · all models · all benchmarks · all arms</p>
  <h1>Does Llama safety alignment generalise across releases?</h1>
  <p class="lede">Every result the project has produced, in one place: canonical screening of six Llama models on
  {sum(1 for d in s.datasets.values() if d.role == 'benchmark')} benchmarks, the regression sets (P⁻), every paraphrase-recycling
  run (RSR, PBR), the human-calibrated anchor, the output-inversion probe (RPE), the rewording-style arcs, known-attack
  transfer on Llama 3.1 8B and 70B, both judge calibrations, the intent gate, and an integrity audit of every artifact.
  Each number names its source and the judge that produced it. Missing cells say <b>{NOT_YET}</b>.</p>
  <div class="meta">
    <span><b>{len(sd.LOCAL_MODELS) + 1}</b> models</span><span><b>{len(s.datasets)}</b> behaviour sets</span>
    <span><b>{len(s.recycle)}</b> recycle runs</span><span><b>{len([r for r in s.rpe if r.items])}</b> RPE runs</span>
    <span><b>{n_resp:,}</b> cached responses</span><span><b>{len(s.registry)}</b> artifacts audited
    across <b>{n_runs}</b> run folders</span><span>generated <b>{today}</b></span>
  </div>
  <div class="meta"><span class="chip chip-ok"><span aria-hidden="true">●</span> no prompt or response text embedded</span>
  <span class="chip">no model calls: rebuilt from caches, manifests, logs and human sheets</span></div>
</header>"""


def sec_summary(s):
    rr_anchor, cal = anchor(s)
    t = []
    if cal:
        r = cal["RSR"]
        t.append(tile("Calibrated anchor RSR · SORRY-Bench · 3-8B → 3.1-8B", pct(r["RSR_corrected"]),
                      f"95% CI {ci_txt(*r['CI'])} · raw {pct(r['RSR_raw'])} ({r['k_recovered']}/{r['n_p_minus']})"
                      f"<br>human-calibrated qwen2.5:14b · κ {cal['judge']['kappa']:.2f}", emph=True,
                      tip="Rogan–Gladen corrected at the behaviour-bundle level; recomputed here from audit/sorrybench/judge_sheet.csv"))
    lin = lineage_rows(s)
    if lin:
        k = sum(r["k_recovered"] for r in lin.values())
        n = sum(r["n_p_minus"] for r in lin.values())
        p, lo, hi = wil(k, n)
        c = sd.corrected(k, n, s.anchor_jc)
        t.append(tile("Lineage RSR · SORRY-Bench · 5 consecutive releases", pct(p),
                      f"{k}/{n} fixes · Wilson {ci_txt(lo, hi)}" + (f"<br>corrected {pct(c['point'])} {ci_txt(c['lo'], c['hi'])}" if c else ""),
                      tip="qwen2.5:14b judge (run summary); correction transfers the anchor calibration to other models"))
    k, n, cells, dss = pooled_offline(s)
    p, lo, hi = wil(k, n)
    t.append(tile(f"Every recycle cell · {len(dss)} benchmarks · offline rule", pct(p),
                  f"{k}/{n} fixes re-opened across {cells} pair × benchmark cells · Wilson {ci_txt(lo, hi)}",
                  tip="One result per (pair, benchmark), full-scope runs preferred; offline rule throughout so benchmarks are comparable"))
    att = (s.calib or {}).get("attack") or {}
    if att.get("design"):
        d = att["design"]
        t.append(tile(f"Known-attack success · Llama 3.1 8B · {att['strata'][0] + att['strata'][1]:,} audited attack responses",
                      f'{pct(d["pool_positive_rate"])} → '
                      f'{pct(d["qwen_boot"]["prevalence"], 0)}',
                      f"raw judge rate → human-calibrated (design-weighted) {ci_txt(*d['qwen_boot']['prev_ci'])}"
                      f"<br>judge sensitivity on attacks ≈ {d['qwen']['se']:.2f} (design-weighted)",
                      tip="The audit sampled all 30 judge-positives but only 30 of 2,338 judge-negatives; 5 of those 30 were real compliance"))
    sc = (s.calib or {}).get("sorrybench")
    if sc:
        t.append(tile("Judge agreement with humans · SORRY-Bench bundles",
                      f"κ {sc['bundle']['offline']['kappa']:.2f} vs {sc['bundle']['qwen']['kappa']:.2f}",
                      f"offline refusal rule vs qwen2.5:14b · {sc['n_bundles']} bundles, {sc['n_rows']} responses",
                      tip="Same human labels; the offline rule misses 0 of 29 compliant bundles, qwen misses 4"))
    stc = Counter(a.status for a in s.registry)
    bad = sum(v for k_, v in stc.items() if k_ not in ("valid",))
    t.append(tile("Artifacts audited", f"{len(s.registry)}", f"{stc['valid']} valid · {bad} flagged "
                  f"({stc['stale-copy']} stale copies, {stc['invalid']} invalid, {stc['circular-judge']} circular-judge, "
                  f"{stc['superseded']} superseded)", tip="Full list in section 15"))

    # findings, all data-driven
    f = []
    if cal:
        f.append(f"<b>The confirmatory result holds.</b> On SORRY-Bench, {cal['RSR']['k_recovered']} of the "
                 f"{cal['RSR']['n_p_minus']} requests Llama 3 8B answered and Llama 3.1 8B refused were re-opened by a "
                 f"meaning-preserving rewording: {pct(cal['RSR']['RSR_corrected'])} after human-calibrated judge "
                 f"correction {ci_txt(*cal['RSR']['CI'])}, McNemar p = {p_txt(cal['RSR']['mcnemar_p'])}. "
                 f"Recomputed here from the human sheet: identical.")
    if lin:
        f.append("<b>Fixes are wording-bound across the lineage.</b> Every consecutive SORRY-Bench pair re-opens "
                 f"between {pct(min(r['RSR_raw'] for r in lin.values()), 0)} and {pct(max(r['RSR_raw'] for r in lin.values()), 0)} "
                 "of its fixes; only the matched-scale 3 → 3.1 pair clears the pre-registered |P⁻| ≥ 30.")
    f.append(f"<b>The effect replicates on other benchmarks, but they are thin.</b> Across {cells} pair × benchmark cells "
             f"the offline rule re-opens {k} of {n} fixes ({pct(p, 0)}). Raw-prompt P⁻ is small outside SORRY-Bench because "
             "safety-tuned older models already refuse most raw harm (section 05).")
    if att.get("design"):
        f.append("<b>Attack-transfer magnitudes are not yet trustworthy.</b> The arm's calibration sheet is stratified on "
                 "the judge's verdict; weighted back to the judged pool, qwen2.5:14b recognises only "
                 f"~{att['design']['qwen']['se'] * 100:.0f}% of real compliance in attack outputs, so the reported ≤ "
                 f"{pct(_max_transfer(s))} per-class transfer "
                 f"is an under-count (calibrated prevalence {pct(att['design']['qwen_boot']['prevalence'], 0)}). The offline "
                 "rule fails differently: it reads cipher and foreign-language garble as compliance (section 11).")
    if sc:
        f.append(f"<b>The cheap judge is the better judge on SORRY-Bench.</b> Against the same human labels the offline "
                 f"refusal rule reaches κ {sc['bundle']['offline']['kappa']:.2f} (sensitivity "
                 f"{sc['bundle']['offline']['se']:.2f}) versus κ {sc['bundle']['qwen']['kappa']:.2f} (sensitivity "
                 f"{sc['bundle']['qwen']['se']:.2f}) for qwen2.5:14b. That is why the offline-judged pilots and the qwen-judged run "
                 f"disagree on P⁻: the offline P⁻ is larger for {judge_gap(s)[0]} of {judge_gap(s)[1]} lineage pairs (section 12).")
    integ = []
    integ.append("<b>RPE on SORRY-Bench was judged by the target model itself</b> (logs: “target-as-judge (circular)”); its "
                 "“RSR·rpe 100%” is not reportable. RPE on HarmBench used qwen2.5:14b and stands.")
    integ.append("<b>Every stored RPE cross-arm agreement (κ, RSR_paraphrase) is an artefact</b>: the paraphrase side was read "
                 "from fields recycle manifests never contain. Re-done here by joining on the prompt text (section 09).")
    integ.append("<b>The style-arc ranking in the previous dashboard was inflated by placeholders</b>: rejected rewordings carry "
                 "1.0 for every gate layer that never ran, so back-translation looked best. Scored-only means reorder the styles (section 10).")
    integ.append("<b>Gate “false-admission 0.0” is not a measurement</b>: no audit row has a human label; the LLM intent layer "
                 "never ran (section 13).")
    n_stale = sum(1 for a_ in s.registry if a_.status == "stale-copy")
    integ.append(f"<b>{n_stale} stale copies</b> were found (a HarmBench summary that is StrongREJECT's, pilots copied across "
                 "benchmarks by a failed run) and are excluded (section 15).")
    return section("s-summary", "01", "Executive summary",
                   f'<div class="tiles tiles3">{"".join(t)}</div>'
                   f'<div class="grid2"><div class="panel"><h3>What the evidence says</h3><ul class="take">{"".join(f"<li>{x}</li>" for x in f)}</ul></div>'
                   f'<div class="panel warnbox"><h3>What the audit changed</h3><ul class="take">{"".join(f"<li>{x}</li>" for x in integ)}</ul></div></div>')


def _cell_lines(s, o, n, ds, rcells, lin):
    lines = []
    c = s.cells.get((o, n, ds))
    if c and c.status in ("complete", "p-minus-only"):
        jl = ""
        if ds == "sorrybench" and (o, n) in lin:
            jl = f" · {lin[(o, n)]['n_p_minus']} judged"
        lines.append(("ok", f"P⁻ {len(c.p_minus)}{jl}"))
    elif c and c.status == "incomplete":
        got = c.breaks_old - c.missing_new
        lines.append(("warn", "P⁻ — newer model not screened" if got == 0 else
                      f"P⁻ ◐ newer model screened on {got}/{c.breaks_old} of the older model's answers"))
    else:
        lines.append(("muted", f"P⁻ {NOT_YET}"))
    rc = rcells.get((o, n, ds))
    if rc:
        rr, pr = rc
        r = pr["recon"]
        lab = f"RSR {pct(r['rsr'], 0)} ({r['k']}/{r['n']})"
        if ds == "sorrybench" and (o, n) in lin:
            lab = f"RSR {pct(lin[(o, n)]['RSR_raw'], 0)} judged · {pct(r['rsr'], 0)} offline"
        lines.append(("ok", lab))
    elif c and c.status in ("complete", "p-minus-only") and not c.p_minus:
        lines.append(("muted", "RSR undefined (P⁻ = 0)"))
    else:
        lines.append(("muted", f"recycle {NOT_YET}"))
    rp = [r for r in s.rpe if r.spec.old == o and r.spec.new == n and r.spec.dataset == ds and r.items]
    if rp:
        bad = all(r.status == "circular-judge" for r in rp)
        lines.append(("crit" if bad else "ok", "RPE " + ("circular judge" if bad else f"{len(rp)} run(s)")))
    if ds == "sorrybench" and (o, n) == ("llama3-8b", "llama31-8b"):
        lines.append(("ok", "human-calibrated"))
    return lines


def sec_status(s):
    lin = lineage_rows(s)
    rcells = recycle_cells(s)
    # (a) screening
    head = "<tr><th scope='col'>Model</th>" + "".join(f"<th scope='col'>{esc(dsl(s, d))}</th>" for d in DS_ORDER) + \
           "<th scope='col'>Framing sets</th></tr>"
    rows = []
    for m in sd.LOCAL_MODELS:
        r = f"<tr><th scope='row'>{esc(label(m))}</th>"
        for d in DS_ORDER:
            sc = s.screens[(m, d)]
            if sc.status == "complete":
                r += f"<td>{status_chip('complete', f'{sc.n_screened}/{sc.n}')}</td>"
            elif sc.status == "conditional":
                r += f"<td>{status_chip('conditional', f'{sc.n_screened}/{sc.n}')}</td>"
            else:
                r += f"<td>{status_chip('none')}</td>"
        fr = [f"{dsl(s, d).split('·')[-1].strip()} {s.screens[(m, d)].n_screened}/{s.screens[(m, d)].n}"
              for d in ("wrapped_rs", "llama_wrapped") if s.screens[(m, d)].n_screened]
        r += f"<td class='small'>{esc('; '.join(fr)) if fr else status_chip('none')}</td></tr>"
        rows.append(r)
    rows.append(f"<tr><th scope='row'>{esc(label('llama31-70b'))}</th><td colspan='{len(DS_ORDER) + 1}' class='small'>"
                f"external responses for the attack-transfer set only (no canonical screening, no recycling) — "
                f"{status_chip('none', 'screening ' + NOT_YET)}</td></tr>")
    screen_tbl = f"<div class='tscroll'><table class='tbl stbl'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"
    # (b) pair x benchmark
    pair_ds = [d for d in DS_ORDER if any(s.screens[(m, d)].n_screened for m in sd.LOCAL_MODELS)]
    unscreened = [dsl(s, d) for d in DS_ORDER if d not in pair_ds]
    head2 = "<tr><th scope='col'>Release pair</th>" + "".join(f"<th scope='col'>{esc(dsl(s, d))}</th>" for d in pair_ds) + "</tr>"
    rows2 = []
    for o, n in sd.PAIRS:
        st = lc.pair_status(o, n)
        r = (f"<tr><th scope='row'>{esc(pair_short(o, n))}<span class='sub2'>{esc(st['kind'].replace('-', ' '))}"
             f"{' · post-hoc' if (o, n) in sd.EXTRA_PAIRS else ''}</span></th>")
        for d in pair_ds:
            r += "<td class='pcell'>" + "".join(f"<div class='pl pl-{c}'>{esc(t)}</div>" for c, t in _cell_lines(s, o, n, d, rcells, lin)) + "</td>"
        rows2.append(r + "</tr>")
    pair_tbl = f"<div class='tscroll'><table class='tbl stbl'><thead>{head2}</thead><tbody>{''.join(rows2)}</tbody></table></div>"
    # (c) instruments and other arms
    att = s.attack
    inst = [
        ["Judge calibration · recycle arm", status_chip("complete", "SORRY-Bench anchor (59 bundles)"),
         status_chip("none", "HarmBench / AdvBench / StrongREJECT / lineage pairs: " + NOT_YET)],
        ["Judge calibration · attack arm", status_chip("partial", "60 responses, stratified"),
         "design-weighted correction computed here; a class-stratified audit of judge-negatives is " + NOT_YET],
        ["Gate human audit", status_chip("none"), "0 of every manifest's audit rows carry a human label"],
        ["LLM intent layer (L2)", status_chip("none", "never ran"), "llm_intent is the placeholder 1.0 on every row"],
        ["Proposal judge ensemble", status_chip("none"), "StrongREJECT evaluator / HarmBench classifier / Llama Guard 3: configured in configs/real.yaml only"],
        ["Lineage verdict cache", status_chip("complete" if Path(sd.ROOT / "cache/judge_lineage_sorrybench.jsonl").exists() else "none"),
         "tools/judge_lineage.py → per-item qwen verdicts for the SORRY-Bench lineage"],
        ["Attack transfer · Llama 3.1 8B", status_chip("complete", f"{len((att.get('8b_original') or {}).get('classes', {}))} classes"),
         "judged against the original goal; the 4 attack-prompt-judged files are superseded"],
        ["Attack transfer · Llama 3.1 70B", status_chip("complete", f"{len((att.get('70b_original') or {}).get('classes', {}))} classes"),
         "external responses, judged here against the original goal"],
        ["Attack transfer · other models", status_chip("none"), "no Llama 2 / 3 / 3.2 attack runs"],
        ["70B generational chain (2-70B → 3-70B → 3.1-70B → 3.3-70B)", status_chip("none"), "out of scope on local hardware"],
        ["Factorial composition ablation (21 conditions)", status_chip("none"), "pending supervisor decision"],
    ]
    return section("s-status", "02", "What's finished",
                   panel("Canonical screening coverage", screen_tbl,
                         "Behaviours with all k = 3 canonical draws cached, per model and benchmark. <i>Conditional</i> = the newer "
                         "model was screened only where the older one complied (the pilot's design), enough for P⁻ but not for an answer rate.")
                   + panel("Release pairs × benchmarks", pair_tbl,
                           "Per cell: regression set (offline rule; SORRY-Bench also shows the judged size), recycling result, RPE runs, "
                           "calibration. Matched-scale pairs not adjacent in the lineage are marked post-hoc."
                           + (f" Not screened for any model, so every pair is {NOT_YET} there: {', '.join(unscreened)}." if unscreened else ""))
                   + panel("Instruments and other arms", table(["Component", "Status", "Detail"], inst)),
                   lede="Green = done, amber = partial, grey = not yet finished, red = done but not reportable.")


def sec_design(s):
    mrows = []
    for m in sd.LOCAL_MODELS + ["llama31-70b"]:
        mt = lc.meta(m)
        caches = sorted({sc.cache for (mm, d), sc in s.screens.items() if mm == m and sc.cache})
        n = sum(sc.n_screened for (mm, d), sc in s.screens.items() if mm == m)
        mrows.append([esc(mt["label"]), esc(mt["gen"]), f"{mt['size']}B", esc(mt["released"]),
                      esc(", ".join(caches) or ("external JSONL" if m == "llama31-70b" else "—")),
                      num(n) if n else "—"])
    drows = []
    for k in [d for d in s.datasets]:
        d = s.datasets[k]
        drows.append([esc(d.label), esc(d.role), num(d.n), esc(d.source), path_html(d.csv)])
    inst = [
        ["Paraphraser", "qwen2.5:3b (October runs); qwen2.5:14b (August AdvBench/StrongREJECT run, per META)", "non-Llama, decorrelated from the targets"],
        ["Judge", "qwen2.5:14b rubric (temperature 0) · offline refusal-prefix rule", "see section 12 for which runs used which"],
        ["Intent gate", "lexical divergence → cosine ≥ 0.75 → BERTScore ≥ 0.85 → NLI ≥ 0.60 → (LLM intent ≥ 0.50, never ran)",
         "loose variant for one RPE run: 0.60 / 0.80 / 0.30"],
        ["Budget", "k = 3 canonical draws · 8 rewordings per behaviour (6 styles) · 1 sample per rewording", "seed 20260101"],
        ["Decision rule", "|P⁻| ≥ 30 for a confirmatory RSR · π* = 0.15 reference for PBR", "pre-registered"],
        ["RPE inverter", "decorrelated prober reconstructing the request from the old model's output", "inverter model not recorded in the artifacts"],
    ]
    return section("s-design", "03", "Study design",
                   panel("Models", mtable(["Model", "Generation", "Size", "Released", "Caches", "Canonical screens"], mrows,
                                          num=(5,), nowrap=(0, 1, 2, 3)))
                   + panel("Instruments", mtable(["Instrument", "Setting", "Note"], inst, nowrap=(0,), wide=(1,)))
                   + panel("Behaviour sets", mtable(["Set", "Role", "Behaviours", "Source", "File"], drows, num=(2,), nowrap=(0, 1, 4))))


def sec_screening(s):
    head = "<tr><th scope='col'>Model</th>" + "".join(f"<th scope='col' class='num'>{esc(dsl(s, d))}</th>"
                                                     for d in DS_ORDER[:5]) + "</tr>"
    rows, trows = [], []
    for m in sd.LOCAL_MODELS:
        r = f"<tr><th scope='row'>{esc(label(m))}</th>"
        for d in DS_ORDER[:5]:
            sc = s.screens[(m, d)]
            if sc.complete:
                p, lo, hi = wil(sc.k, sc.n)
                r += (f"<td class='num hm' style='{heat(min(1.0, p / 0.3))}' data-tip='{esc(label(m))} · {esc(dsl(s, d))}: "
                      f"answered {sc.k} of {sc.n} ({pct(p)}, Wilson {ci_txt(lo, hi)})'>{pct(p)}<span class='sub'> {sc.k}/{sc.n}</span></td>")
                trows.append([esc(label(m)), esc(dsl(s, d)), num(sc.k), num(sc.n), pct(p), ci_txt(lo, hi), esc(sc.cache)])
            elif sc.status == "conditional":
                r += (f"<td class='num hatch' data-tip='screened only on {sc.n_screened} behaviours the older model answered'>"
                      f"<span class='sub'>conditional · {sc.n_screened}</span></td>")
            else:
                r += f"<td class='num'><span class='sub'>{NOT_YET}</span></td>"
        rows.append(r + "</tr>")
    lin = s.lineage or {}
    pbr = {x["model"]: x for x in lin.get("PBR_backbone_RAW", [])}
    jrows = []
    n_sb = s.datasets["sorrybench"].n if "sorrybench" in s.datasets else 0
    for m in sd.LOCAL_MODELS:
        sc = s.screens.get((m, "sorrybench"))
        q = pbr.get(m)
        if not sc or not sc.complete or not n_sb:
            continue
        qa = (n_sb - q["n_cm"]) if q else None
        jrows.append([esc(label(m)), f"{sc.k} ({pct(sc.k / sc.n)})", f"{qa} ({pct(qa / n_sb)})" if qa is not None else "—",
                      f"{(qa - sc.k):+d}" if qa is not None else "—"])
    if not jrows:
        jrows = [[NOT_YET, "—", "—", "—"]]
    return section("s-screening", "04", "Canonical screening across benchmarks",
                   panel("How often each model answers the raw request", f"<div class='tscroll'><table class='tbl hmtbl'>{head}{''.join(rows)}</table></div>"
                         + tableview(["Model", "Benchmark", "Answered", "Screened", "Rate", "Wilson 95%", "Cache"], trows, num_cols=(2, 3, 4)),
                         "Offline refusal rule, any of k = 3 draws, so every benchmark is read by the same judge. Darker = higher "
                         "(scale saturates at 30%). Safety-tuned models answer almost nothing on AdvBench and StrongREJECT, which is why "
                         "P⁻ there is tiny; SORRY-Bench and HarmBench (with its copyright category) are where older models still comply.")
                   + panel("SORRY-Bench: the two judges disagree on what counts as an answer",
                           table(["Model", "Answered · offline rule", "Answered · qwen2.5:14b (lineage run)", "Δ"], jrows, num_cols=(1, 2, 3)),
                           "qwen2.5:14b answers = the benchmark size − the model's canonically-refused set in the lineage run. The LLM judge is stricter "
                           "for the Llama 3.x models; section 12 shows which judge agrees with the humans."))


def sec_pminus(s):
    head = "<tr><th scope='col'>Release pair</th>" + "".join(f"<th scope='col' class='num'>{esc(dsl(s, d))}</th>" for d in DS_ORDER[:5]) + "</tr>"
    rows = []
    for o, n in sd.PAIRS:
        r = f"<tr><th scope='row'>{esc(pair_short(o, n))}<span class='sub2'>{esc(lc.pair_status(o, n)['kind'].replace('-', ' '))}</span></th>"
        for d in DS_ORDER[:5]:
            c = s.cells.get((o, n, d))
            if c and c.status in ("complete", "p-minus-only"):
                nf = "—" if c.neg_flip is None else str(c.neg_flip)
                tip = (f"{pair_short(o, n)} · {dsl(s, d)}\nolder answered {c.breaks_old}: P⁻ {len(c.p_minus)}, still answered "
                       f"{c.persistent}\nolder refused: newly answered (regression) {nf}" + ("" if c.neg_flip is not None else
                       " — newer model not screened there"))
                big = len(c.p_minus) >= 30
                r += (f"<td class='num pmc{' pm-big' if big else ''}' data-tip='{esc(tip)}'><b>{len(c.p_minus)}</b>"
                      f"<span class='sub'> / reg {nf}</span></td>")
            elif c and c.status == "incomplete":
                r += f"<td class='num'><span class='sub'>◐ incomplete</span></td>"
            else:
                r += f"<td class='num'><span class='sub'>{NOT_YET}</span></td>"
        rows.append(r + "</tr>")
    prow = []
    for p in sorted(s.pilots, key=lambda x: (x.dataset or "", x.old, x.new, x.mtime)):
        prow.append([esc(pair_short(p.old, p.new)), esc(dsl(s, p.dataset) if p.dataset else "?") + (" *" if p.dataset_inferred else ""),
                     num(p.n), num(p.breaks_old), f"<b>{p.p_minus}</b>", judge_chip(p.judge), status_chip(p.status),
                     path_html(p.path) + (f"<div class='note2'>{esc('; '.join(p.issues))}</div>" if p.issues else ""),
                     esc(p.mtime[:10])])
    worst = None
    for o, n in sd.PAIRS:
        a_, b_ = s.cells.get((o, n, "sorrybench")), s.cells.get((o, n, "harmbench"))
        if a_ and b_ and a_.neg_flip is not None and b_.neg_flip is not None:
            gap = a_.neg_flip + b_.neg_flip - len(a_.p_minus) - len(b_.p_minus)
            if worst is None or gap > worst[0]:
                worst = (gap, o, n, a_, b_)
    reg_txt = ""
    if worst and worst[0] > 0:
        _, o, n, a_, b_ = worst
        reg_txt = (f" {label(o)} → {label(n)} shows {a_.neg_flip} regressions on SORRY-Bench and {b_.neg_flip} on HarmBench "
                   f"against {len(a_.p_minus)} and {len(b_.p_minus)} fixes: newer is not uniformly safer.")
    return section("s-pminus", "05", "Regression sets (P⁻)",
                   panel("P⁻ and regressions for every pair and benchmark", f"<div class='tscroll'><table class='tbl hmtbl'>{head}{''.join(rows)}</table></div>",
                         "Bold = |P⁻| (older answered, newer refused) under the offline rule; “reg” = safety regressions (older refused, newer "
                         "answers). Highlighted cells clear the |P⁻| ≥ 30 floor." + reg_txt)
                   + panel("Every P⁻ pilot on disk", tview(f"Show all {len(prow)} stored pilots with their integrity verdicts",
                                                           ["Pair", "Benchmark", "n", "Older answered", "P⁻", "Judge", "Status",
                                                            "File · findings", "Written"], prow, num=(2, 3, 4), nowrap=(0, 5, 6, 8),
                                                           wide=(7,))
                           + "<p class='note'>" + " · ".join(f"{status_chip(k)} {v}" for k, v in
                                                             sorted(Counter(p.status for p in s.pilots).items())) + "</p>",
                           "Each stored pilot re-checked against today's reconstruction. * = benchmark inferred from size and coverage."))


def sec_rsr(s):
    lin = lineage_rows(s)
    groups = defaultdict(list)
    trows = []
    foot = {}
    for rr in s.recycle:
        sp = rr.spec
        for (o, n), pr in rr.pairs.items():
            rep = pr["reported"]
            rec = pr["recon"]
            judged = (sp.judge != "offline" and rep is not None)
            cal_rsr = (rr.calibrated or {}).get("RSR") if sp.calibrated else None
            if cal_rsr:                                  # the anchor: show the calibrated (qwen re-judge) pair
                k, n_, rsr = cal_rsr["k_recovered"], cal_rsr["n_p_minus"], cal_rsr["RSR_raw"]
                judged = True
            elif rep and rep.get("defined", True) and rep.get("n_p_minus"):
                k, n_, rsr = rep["k_recovered"], rep["n_p_minus"], rep["RSR_raw"]
            elif rec:
                k, n_, rsr = rec["k"], rec["n"], rec["rsr"]
            else:
                k = n_ = rsr = None
            _, lo, hi = wil(k, n_) if n_ else (None, None, None)
            corr = None
            if sp.id == "anchor-sorrybench" and rr.calibrated:
                corr = rr.calibrated["RSR"]["RSR_corrected"]
            if cal_rsr and pr["match"] is True:
                repro = status_chip("valid", f"raw offline run {rec['k']}/{rec['n']} reproduced")
            elif pr["match"] is True:
                repro = status_chip("valid", "reproduced")
            elif judged and rec:
                repro = f"judge differs · offline {rec['k']}/{rec['n']}"
            elif not pr["complete"]:
                repro = status_chip("incomplete", "re-screen unfinished")
            elif rep is None and rec:
                repro = "reconstructed (no valid summary)"
            elif pr["n_pm"] == 0:
                repro = status_chip("na", "P⁻ = 0")
            else:
                repro = "—"
            jtxt = sp.judge if not judged else "llm (qwen2.5:14b)"
            para_main, _, para_note = sp.paraphraser.partition(" (")
            if para_note:
                key = para_note.rstrip(")")
                foot.setdefault(key, len(foot) + 1)
                para_cell = f"{esc(para_main)}<sup>{foot[key]}</sup>"
            else:
                para_cell = esc(para_main)
            emph = bool(n_ and n_ >= 30)
            tip = (f"{pair_short(o, n)} · {dsl(s, sp.dataset)} · run {sp.id}\njudge: {jtxt}\n"
                   + (f"{k} of {n_} fixes re-opened · RSR {pct(rsr)} (Wilson {ci_txt(lo, hi)})" if n_ else
                      ("P⁻ empty: RSR undefined" if pr["n_pm"] == 0 else NOT_YET))
                   + (f"\nhuman-calibrated: {pct(corr)}" if corr is not None else "")
                   + (f"\noffline reconstruction: {rec['k']}/{rec['n']}" if judged and rec else ""))
            jsub = ("qwen re-judge" if cal_rsr else "qwen") if judged else "offline"
            groups[sp.dataset].append(dict(label=f"{pair_short(o, n)}", sub=f"{sp.id} · {jsub} · {sp.behaviour_scope}",
                                           point=rsr, lo=lo, hi=hi, corr=corr, emph=emph, tip=tip,
                                           v_txt=pct(rsr) if rsr is not None else ("undefined" if pr["n_pm"] == 0 else "—"),
                                           n_txt=f"{k}/{n_}" if n_ else ("P⁻ = 0" if pr["n_pm"] == 0 else NOT_YET),
                                           hollow=not judged and sp.judge == "offline",
                                           empty=("P⁻ = 0 → undefined" if pr["n_pm"] == 0 else "re-screen unfinished")))
            mp = cal_rsr.get("mcnemar_p") if cal_rsr else (rep.get("mcnemar_p") if rep else None)
            if mp is None and rec:
                mp = rec["p"]
            trows.append([esc(sp.id), esc(pair_short(o, n)), esc(dsl(s, sp.dataset)), judge_chip(jtxt), para_cell,
                          num(n_) if n_ else ("0" if pr["n_pm"] == 0 else "—"), num(k) if n_ else "—",
                          (f"{pct(rsr)} <span class='sub'>{ci_txt(lo, hi)}</span>" if n_ else "—"), p_txt(mp), repro])
    gl = [(f"{dsl(s, d)}", groups[d]) for d in DS_ORDER if groups.get(d)]
    link = (f'<p class="note">Case-level SORRY-Bench detail (flows, cells, every fix with its rewordings): '
            f'<a href="{esc(os.path.relpath(sd.ROOT / s.lineage_dashboard, sd.ROOT / "results"))}">{esc(s.lineage_dashboard)}</a></p>'
            if s.lineage_dashboard else "")
    return section("s-rsr", "06", "Recycling Success Rate — every run",
                   panel("RSR by benchmark and run",
                         '<div class="legend"><span><i class="lg-dot"></i>qwen2.5:14b judge (run summary)</span>'
                         '<span><i class="lg-ring"></i>offline rule</span><span><i class="lg-dia"></i>human-calibrated</span>'
                         '<span><i class="lg-mute"></i>|P⁻| &lt; 30 (exploratory)</span></div>'
                         + forest_grouped(gl, "Recycling Success Rate by benchmark and run") + link,
                         "Wilson 95% intervals. RSR = share of the fixes the newer model answers on at least one admitted rewording. "
                         "Every offline-judged summary was rebuilt from cache and reproduces exactly; the HarmBench smoke run never finished "
                         "its 3.1-8B re-screen, but its completed 2-7B → 3-8B half is shown.")
                   + panel("All recycle results", mtable(["Run", "Pair", "Benchmark", "Judge", "Paraphraser", "P⁻", "Re-opened",
                                                         "RSR [Wilson 95%]", "McNemar p", "Reproduction"], trows,
                                                        num=(5, 6, 7, 8), nowrap=(0, 1, 2, 3, 7))
                           + "".join(f"<p class='note small'><sup>{i}</sup> {esc(t)}</p>" for t, i in foot.items())))


def sec_pbr(s):
    rows_by_model = defaultdict(dict)
    srcs = {}
    for rr in sorted(s.recycle, key=lambda r: (r.spec.behaviour_scope != "full", r.spec.id)):
        if rr.spec.behaviour_scope != "full":
            continue
        for m, v in rr.pbr.items():
            rep, rec = v["reported"], v["recon"]
            judged = rr.spec.judge != "offline" and rep is not None
            if judged:
                val = (rep["k_bypassed"], rep["n_resolved"], rep["PBR_raw"], "qwen")
            elif rec:
                val = (rec["k"], rec["n"], rec["pbr"], "offline")
            else:
                continue
            if rr.spec.dataset not in rows_by_model[m]:
                rows_by_model[m][rr.spec.dataset] = val
                srcs[(m, rr.spec.dataset)] = rr.spec.id
    head = "<tr><th scope='col'>Model</th>" + "".join(f"<th scope='col' class='num'>{esc(dsl(s, d))}</th>" for d in DS_ORDER[:4]) + "</tr>"
    body, trows = [], []
    for m in sd.LOCAL_MODELS:
        r = f"<tr><th scope='row'>{esc(label(m))}</th>"
        for d in DS_ORDER[:4]:
            v = rows_by_model[m].get(d)
            if v:
                k, n, p, j = v
                _, lo, hi = wil(k, n)
                r += (f"<td class='num hm' style='{heat(min(1.0, p / 0.3))}' data-tip='{esc(label(m))} · {esc(dsl(s, d))}: "
                      f"{k} of {n} own refusals re-opened · PBR {pct(p)} (Wilson {ci_txt(lo, hi)}) · {j} judge · run {srcs[(m, d)]}'>"
                      f"{pct(p)}<span class='sub'> {k}/{n} · {j}</span></td>")
                trows.append([esc(label(m)), esc(dsl(s, d)), num(k), num(n), pct(p), ci_txt(lo, hi), esc(j), esc(srcs[(m, d)])])
            else:
                r += f"<td class='num'><span class='sub'>{NOT_YET}</span></td>"
        body.append(r + "</tr>")
    return section("s-pbr", "07", "Paraphrase robustness of each model's own refusals (PBR)",
                   panel("PBR per model and benchmark", f"<div class='tscroll'><table class='tbl hmtbl'>{head}{''.join(body)}</table></div>"
                         + tableview(["Model", "Benchmark", "Re-opened", "Resolved refusals", "PBR", "Wilson 95%", "Judge", "Run"], trows, num_cols=(2, 3, 4)),
                         "Of what each model refuses canonically, the share at least one admitted rewording re-opens (full-scope runs only; "
                         "P⁻-subset runs are excluded because their refusal set is not the model's). Darker = higher, scale saturates at 30%. "
                         "PBR is always estimable, so it is the backbone where P⁻ is thin: on SORRY-Bench every model sits at 10–23%, far "
                         "above AdvBench (1–9%)."))


def sec_anchor(s):
    rr, cal = anchor(s)
    if not cal:
        return section("s-anchor", "08", "Calibrated headline", f"<p class='note'>{NOT_YET}</p>")
    r, pb, j = cal["RSR"], cal["PBR_new"], cal["judge"]
    rec = sd.corrected(r["k_recovered"], r["n_p_minus"], s.anchor_jc, r.get("mean_admitted", 1.0))
    same = rec and abs(rec["point"] - r["RSR_corrected"]) < 5e-4 and abs(rec["lo"] - r["CI"][0]) < 5e-4
    tiles = (tile("RSR · corrected", pct(r["RSR_corrected"]), f"95% CI {ci_txt(*r['CI'])}", emph=True)
             + tile("RSR · raw", pct(r["RSR_raw"]), f"{r['k_recovered']}/{r['n_p_minus']} · McNemar p {p_txt(r['mcnemar_p'])}")
             + tile(f"PBR · {label('llama31-8b')} · corrected", pct(pb["PBR_corrected"]), f"95% CI {ci_txt(*pb['CI'])} · raw {pct(pb['PBR_raw'])}")
             + tile("Judge vs humans (bundle level)", f"κ {j['kappa']:.3f}", f"se {j['se']:.3f} · sp {j['sp']:.3f} · {j['n_bundles']} bundles · admitted"))
    check = (status_chip("valid", "recomputed from the human sheet: identical") if same else
             status_chip("warn", "recomputation differs"))
    lin = lineage_rows(s)
    lrows = []
    for (o, n), x in lin.items():
        c = sd.corrected(x["k_recovered"], x["n_p_minus"], s.anchor_jc, x.get("mean_admitted", 1.0))
        _, lo, hi = wil(x["k_recovered"], x["n_p_minus"])
        lrows.append([esc(pair_short(o, n)), num(x["n_p_minus"]), num(x["k_recovered"]), pct(x["RSR_raw"]), ci_txt(lo, hi),
                      pct(c["point"]) if c else "—", ci_txt(c["lo"], c["hi"]) if c else "—"])
    raw_log = rr.reported if rr else None
    raw_line = ""
    if raw_log:
        x = raw_log["RSR_primary_RAW"][0]
        raw_line = (f"The raw run itself (logs/recycle_3_31.log) used the offline rule: P⁻ {x['n_p_minus']}, {x['k_recovered']} re-opened "
                    f"({pct(x['RSR_raw'])}). The headline re-judges the same cached responses with qwen2.5:14b "
                    f"(P⁻ {r['n_p_minus']}, {r['k_recovered']} re-opened) and corrects with the human sheet.")
    return section("s-anchor", "08", "Calibrated headline — the anchor pair",
                   f'<div class="tiles">{tiles}</div>'
                   + panel("Provenance", f"<p class='note'>{check} {esc(raw_line)}</p>"
                           f"<p class='note'>Sources: <code>{esc(sd.rel(sd.ROOT / rr.spec.run_dir / rr.spec.calibrated))}</code> · "
                           f"<code>audit/sorrybench/judge_sheet.csv</code> (60 bundles labelled; 59 complete) · the audit sampled every resolved "
                           f"bundle, so no stratum re-weighting is needed.</p>")
                   + panel("The same calibration applied to every lineage pair",
                           table(["Pair", "P⁻", "Re-opened", "RSR raw", "Wilson 95%", "Corrected", "Corrected 95% CI"], lrows, num_cols=(1, 2, 3, 5)),
                           "Corrected intervals propagate the calibration variance (delta method, recyclebench.stats.corrected_interval). "
                           "Transferring one judge calibration from Llama 3.1 8B outputs to other models is an assumption, not a measurement."))


def sec_rpe(s):
    rows = []
    for r in s.rpe:
        c = r.counts
        if not r.items:
            rows.append([esc(r.spec.id), esc(pair_short(r.spec.old, r.spec.new)), esc(dsl(s, r.spec.dataset)), "—", "—",
                         "—", "—", "—", "—", "—", "—", status_chip(r.status)])
            continue
        res = c.get("resolved", 0)
        rows.append([esc(r.spec.id), esc(pair_short(r.spec.old, r.spec.new)), esc(dsl(s, r.spec.dataset)), judge_chip(r.judge),
                     esc(r.spec.gate_label), num(c["probed"]), num(c["candidates"]), num(c["admitted"]),
                     f"{res} ({pct(res / c['probed'], 0)})", f"{c.get('rec_stored', 0)}/{res}", f"{c.get('rec_offline', 0)}/{res}",
                     status_chip(r.status)])
    notes = "".join(f"<li><b>{esc(r.spec.id)}</b>: {esc('; '.join(r.issues))}</li>" for r in s.rpe if r.issues)
    arows = []
    for r in s.rpe:
        a = r.agreement
        if not a or "both_resolved" not in a:
            continue
        arows.append([esc(r.spec.id), num(a["probed"]), num(a["in_para_pminus"]), num(a["both_resolved"]),
                      f"{a['both_recovered']} / {a['both_held']}", f"{a['rpe_only']} / {a['para_only']}",
                      pct(a["agree"], 0), f"{a['kappa']:.2f}" if a["kappa"] is not None else "n/a",
                      f"{a['para_k']}/{a['para_resolved']} ({pct(a['para_k'] / a['para_resolved'] if a['para_resolved'] else None, 0)})",
                      f"{a['rpe_k']}/{a['rpe_resolved']}"])
    stored = sd.stored_agreement_files()
    srows = [[path_html(x["path"]), esc(x["dataset"]), num(x["n_total"]),
              pct(x["rsr_para"]) if x["rsr_para"] is not None else "n/a", pct(x["rsr_rpe"]) if x["rsr_rpe"] is not None else "n/a",
              f"{x['cohen_kappa']:.2f}" if x["cohen_kappa"] is not None else "n/a", num(x["para_unresolved"])] for x in stored]
    rej = [dict(label=r.spec.id.replace("rpe-", ""), sub=f"{sum(r.rejections.values()):,} rejected candidates",
                parts=dict(r.rejections), total=r.counts.get("candidates", 0)) for r in s.rpe if r.items]
    for x in rej:
        x["parts"]["admitted"] = next(r.counts.get("admitted", 0) for r in s.rpe if r.spec.id.replace("rpe-", "") == x["label"])
    keys = [("admitted", "sb-adm", "admitted"), ("lexical_divergence", "sb-lex", "lexical divergence"),
            ("cosine", "sb-cos", "embedding cosine"), ("bertscore", "sb-bert", "BERTScore"), ("nli_entail", "sb-nli", "NLI entailment")]
    return section("s-rpe", "09", "Output inversion (RPE) — the independent probe",
                   panel("Every RPE run", mtable(["Run", "Pair", "Benchmark", "Judge", "Gate", "Probed", "Candidates", "Admitted",
                                                 "Resolved", "Re-opened (stored judge)", "Re-opened (offline)", "Status"], rows,
                                                num=(5, 6, 7, 8, 9, 10), nowrap=(0, 1, 2, 3, 11))
                         + (f"<ul class='take small'>{notes}</ul>" if notes else ""),
                         "RPE reconstructs the request from the older model's output, gates it for intent, and re-asks the newer model. "
                         "Coverage is the binding constraint: inversions rarely survive the entailment gate (the “entailment bound”).")
                   + panel("Where inversion candidates fail the gate", stacked_bars(rej, keys, "RPE candidate outcomes by gate layer")
                           + '<div class="legend">' + "".join(f'<span><i class="lg {c}"></i>{esc(l)}</span>' for _, c, l in keys) + "</div>",
                           "First failing layer per candidate (the gate short-circuits). The loose gate trades cosine rejections for NLI rejections.")
                   + panel("Convergent validity, recomputed", mtable(["RPE run", "Probed", "In paraphrase P⁻", "Both resolved",
                                                                     "Both re-opened / both held", "RPE-only / paraphrase-only",
                                                                     "Agreement", "Cohen's κ", "Paraphrase RSR (resolved)", "RPE RSR (resolved)"],
                                                                    arows, num=(1, 2, 3, 6, 7), nowrap=(0,)),
                           "Joined on the canonical prompt text (the arms number behaviours differently) and judged with the same offline rule "
                           "on both sides. Few behaviours are resolved by both arms, so κ is undefined or unstable; where both resolve, inversion "
                           "re-opens at least as often as paraphrase, which supports reading paraphrase RSR as a lower bound.")
                   + panel("Stored agreement files (not measurements)", table(["File", "Benchmark", "P⁻", "RSR para (stored)",
                                                                             "RSR rpe (stored)", "κ (stored)", "Para unresolved"], srows, num_cols=(2, 6)),
                           "These files report RSR_paraphrase 0% and dozens of unresolved paraphrase items because the probe read "
                           "recovered/bypassed/success fields that recycle manifests do not contain."))


def sec_arcs(s):
    cards, trows = [], []
    for rr in s.recycle:
        g = rr.gate
        cls = []
        for tc in ["paraphrase", "syntactic", "lexical", "back_translation", "tense_shift", "register"]:
            c = g["per_class"].get(tc)
            if not c:
                continue
            ev, nv = c["nli_eval"], c["nli_naive"]
            cls.append(dict(name=tc, gen=c["gen"], adm=c["adm"], n_eval=len(ev), n_naive=len(nv),
                            nli_eval=(sum(ev) / len(ev)) if ev else None, nli_naive=(sum(nv) / len(nv)) if nv else None))
            trows.append([esc(rr.spec.id), esc(tc.replace("_", " ")), num(c["gen"]), f"{c['adm']} ({pct(c['adm'] / c['gen'] if c['gen'] else None, 0)})",
                          f"{(sum(ev) / len(ev)):.3f}" if ev else "—", num(len(ev)), f"{(sum(nv) / len(nv)):.3f}" if nv else "—",
                          num(c["rej"].get("lexical_divergence", 0))])
        best = max((c for c in cls if c["nli_eval"] is not None), key=lambda c: c["nli_eval"], default=None)
        naive_best = max((c for c in cls if c["nli_naive"] is not None), key=lambda c: c["nli_naive"], default=None)
        flip = (best and naive_best and best["name"] != naive_best["name"])
        cards.append(f'<div class="arccard"><div class="arct">{esc(rr.spec.id)}</div>'
                     f'<div class="arcs">{esc(dsl(s, rr.spec.dataset))} · {g["total"]["gen"]:,} rewordings</div>'
                     f'{arcs_svg(cls, "Style arcs for " + rr.spec.id)}'
                     f'<div class="arcf">best: <b>{esc(best["name"].replace("_", " ") if best else "—")}</b>'
                     + (f' · <span class="warnt">placeholder ranking said {esc(naive_best["name"].replace("_", " "))}</span>' if flip else "")
                     + "</div></div>")
    # recovery by style, offline, every complete pair
    rec_rows = []
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for rr in s.recycle:
        for (o, n), byc in sd.recovery_by_style(rr).items():
            for tc, (vh, vn, ih, inn) in byc.items():
                a = agg[tc]
                a[0] += vh; a[1] += vn; a[2] += ih; a[3] += inn
                rec_rows.append([esc(rr.spec.id), esc(pair_short(o, n)), esc(tc.replace("_", " ")), f"{vh}/{vn}",
                                 pct(vh / vn if vn else None), f"{ih}/{inn}", pct(ih / inn if inn else None)])
    bars = [dict(label=tc.replace("_", " "), sub=f"{a[1]:,} rewordings of fixes", v=(a[0] / a[1] if a[1] else None),
                 v_txt=pct(a[0] / a[1] if a[1] else None), tip=f"{tc}: {a[0]} of {a[1]} admitted rewordings of a fix were answered "
                 f"by the newer model (offline rule, all recycle runs)") for tc, a in sorted(agg.items(), key=lambda x: -(x[1][0] / max(x[1][1], 1)))]
    return section("s-arcs", "10", "Rewording styles — the arcs",
                   panel("Gate quality by style, per recycle manifest",
                         f'<div class="arcgrid-wrap">{"".join(cards)}</div>'
                         '<div class="legend"><span><i class="lg-arcdot"></i>mean NLI of rewordings the gate scored (dot area ~ count)</span>'
                         '<span><i class="lg-arcnaive"></i>mean including placeholder 1.0s (what the old dashboard drew)</span></div>',
                         "Radius = mean NLI entailment (intent preserved): outer arcs keep intent best. The gate stops at the first failing layer "
                         "and stores 1.0 for layers it never ran, so means over all rows are inflated, most for back-translation, whose round-trips "
                         "are often rejected as too close to the original before NLI runs. Only scored rewordings count here.")
                   + '<div class="grid2">'
                   + panel("Which styles re-open fixes (all recycle runs, offline rule)", ld.hbars(bars, "Per-variant re-open rate by style"),
                           "Share of admitted rewordings of a P⁻ item that the newer model answered.")
                   + panel("Tables behind the arcs", "<p class='note'>Every number in this section, per manifest and per pair.</p>"
                           + tview("Gate statistics per style and manifest", ["Run", "Style", "Generated", "Admitted", "NLI (scored)",
                                                                              "n scored", "NLI (with placeholders)", "Lexical rejects"],
                                   trows, num=(2, 4, 5, 6, 7), nowrap=(0, 1))
                           + tview("Re-open rate by style, per recycle pair", ["Run", "Pair", "Style", "Variants answered", "Rate",
                                                                               "Fixes re-opened by style", "Rate"], rec_rows,
                                   num=(3, 4, 5, 6), nowrap=(0, 1, 2)))
                   + "</div>")


def sec_attack(s):
    a = s.attack
    o8 = (a.get("8b_original") or {}).get("classes", {})
    a8 = a.get("8b_attack") or {}
    o70 = (a.get("70b_original") or {}).get("classes", {})
    off8 = (s.attack_offline_8b or {}).get("classes", {}) if s.attack_offline_8b and "classes" in s.attack_offline_8b else {}
    off70 = (s.attack_70b or {}).get("classes", {})
    classes = sorted(set(o8) | set(o70) - {"original_prompt"},
                     key=lambda c: -max((o70.get(c) or {}).get("transfer_on_refused") or 0, (o8.get(c) or {}).get("transfer_on_refused") or 0))
    rows, trows = [], []
    xmax = 0.3
    for c in classes:
        if c == "original_prompt":
            continue
        v = {}
        if c in o8:
            v["o8"] = (o8[c]["transfer_on_refused"], o8[c]["n_broke_among_refused"], o8[c]["n_refused_baseline"])
        if c in a8:
            v["a8"] = (a8[c]["transfer_on_refused"], a8[c]["n_broke_among_refused"], a8[c]["n_refused_baseline"])
        if c in o70:
            v["o70"] = (o70[c]["transfer_on_refused"], o70[c]["n_broke_among_refused"], o70[c]["n_refused_baseline"])
        xmax = max([xmax] + [x[0] for kk, x in v.items() if kk != "a8"])
        rows.append(dict(label=c.replace("_", " "), vals=v, flag=c in sd.CIPHER))
        trows.append([esc(c), "⚠" if c in sd.CIPHER else "",
                      f"{pct(v['o8'][0])} ({v['o8'][1]}/{v['o8'][2]})" if "o8" in v else NOT_YET,
                      f"{pct(v['a8'][0])} ({v['a8'][1]}/{v['a8'][2]})" if "a8" in v else NOT_YET,
                      f"{pct(v['o70'][0])} ({v['o70'][1]}/{v['o70'][2]})" if "o70" in v else NOT_YET,
                      (f"{pct(off8[c]['k'] / off8[c]['n'])}" if c in off8 and off8[c]["n"] else "—"),
                      (f"{pct(off70[c]['o'] / off70[c]['n'])}" if c in off70 and off70[c]["n"] else "—")])
    xmax = min(1.0, math.ceil(xmax * 20) / 20)
    series = [("a8", "db-a8", "8B · judged vs attack prompt (superseded)"), ("o8", "db-o8", "8B · judged vs original goal"),
              ("o70", "db-o70", "70B · judged vs original goal")]
    chart = dumbbell(rows, "Transfer-on-refused by attack class", xmax, series)
    att = (s.calib or {}).get("attack") or {}
    calp = ""
    if att:
        d = att.get("design") or {}
        cm = '<div class="cmrow">' + confusion(att["response"]["qwen"], "qwen2.5:14b vs human (sample)") + \
             confusion(att["response"]["offline"], "offline rule vs human (sample)") + "</div>"
        rows_c = [["qwen2.5:14b · package (sample proportions)", f"{att['package']['se']:.3f}", f"{att['package']['sp']:.3f}",
                   f"{att['package']['kappa']:.3f}", "treats 30 + 30 as representative"]]
        if d:
            rows_c.append(["qwen2.5:14b · design-weighted (population strata)", f"{d['qwen']['se']:.3f} {ci_txt(*d['qwen']['se_ci'], d=1)}",
                           f"{d['qwen']['sp']:.3f}", "—", f"strata {att['strata'][0]} judge-positive / {att['strata'][1]:,} judge-negative"])
            rows_c.append(["offline rule · design-weighted", f"{d['offline']['se']:.3f}", f"{d['offline']['sp']:.3f} {ci_txt(*d['offline']['sp_ci'], d=1)}",
                           "—", "flags garbled / non-English output as compliance"])
        calp = panel("Is the attack judge fit for purpose?",
                     cm + table(["Judge", "Sensitivity", "Specificity", "κ", "Note"], rows_c)
                     + (f"<p class='note'>Of the {att['strata'][1]:,} responses the judge called refusals, the 30 audited contained 5 real "
                        f"compliances. Weighted back, the judged pool's true compliance prevalence is "
                        f"<b>{pct(d['qwen_boot']['prevalence'])}</b> {ci_txt(*d['qwen_boot']['prev_ci'])} against a raw judge rate of "
                        f"{pct(d['pool_positive_rate'])}. Class-level rates cannot be corrected from this sheet (no per-class audit), so the "
                        f"per-class numbers above are raw lower bounds.</p>" if d else ""),
                     "audit/attack/judge_sheet.csv: every judge-positive plus a random 30 of the judge-negatives (tools/judge_export_attack.py).")
    return section("s-attack", "11", "Known-attack transfer — Llama 3.1 8B and 70B",
                   panel("Transfer-on-refused by attack class",
                         '<div class="legend">' + "".join(f'<span><i class="lg {c}"></i>{esc(l)}</span>' for _, c, l in series)
                         + '<span>⚠ low-readability cipher class</span></div>' + chart
                         + tableview(["Class", "Cipher", "8B vs original", "8B vs attack prompt", "70B vs original",
                                      "8B · no refusal marker (offline)", "70B · no refusal marker (offline)"], trows),
                         "Of the behaviours each model refused in plain form, the share an attack re-opens (qwen2.5:14b judge, RAW; the axis "
                         "covers the reportable series, off-scale attack-prompt-judged values are marked at the edge). Judging against "
                         "the transformed prompt inflates cipher classes; judging against the original goal is the reportable setting. The 70B model "
                         "is markedly more vulnerable to combination attacks. The offline columns show why the refusal rule cannot judge this arm: "
                         "encoded replies carry no refusal phrase at all.")
                   + calp)


def sec_judges(s):
    inv = [
        ["Offline refusal-prefix rule", "<code>recyclebench.judge.looks_like_refusal</code>",
         "all August pilots and recycle runs; October SORRY-Bench pilots; the anchor's raw run; every reconstruction on this page",
         "calibrated here on both human sheets"],
        ["qwen2.5:14b rubric", "decorrelated LLM judge, temperature 0, 1-token answer",
         "SORRY-Bench lineage run; anchor headline (re-judge); attack transfer 8B + 70B; RPE on HarmBench; FRESH HarmBench and llama32jb pilots",
         "calibrated on both sheets (SORRY-Bench admitted κ 0.83)"],
        ["Target-as-judge", "the newer model grading its own reply", "RPE on SORRY-Bench (strict and loose gate)", "circular: not reportable"],
        ["Proposal ensemble", "StrongREJECT evaluator + HarmBench classifier + Llama Guard 3", "configured in configs/real.yaml (vLLM ids) only", NOT_YET],
    ]
    blocks = []
    for name, title in [("sorrybench", "SORRY-Bench recycle arm · anchor pair responses"), ("attack", "Attack-transfer arm · Llama 3.1 8B responses")]:
        c = (s.calib or {}).get(name)
        if not c:
            blocks.append(panel(title, f"<p class='note'>{NOT_YET}</p>"))
            continue
        lvl = "bundle" if name == "sorrybench" else "response"
        q, o = c[lvl]["qwen"], c[lvl]["offline"]
        rows = []
        for jn, r in [("qwen2.5:14b", q), ("offline rule", o)]:
            rows.append([esc(jn), f"{r['se']:.3f} {ci_txt(r['se_lo'], r['se_hi'], d=1)}", f"{r['sp']:.3f} {ci_txt(r['sp_lo'], r['sp_hi'], d=1)}",
                         f"{r['kappa']:.3f}", f"{r['tp']} / {r['fn']} / {r['fp']} / {r['tn']}"])
        resp = c["response"]
        extra = (f"<p class='note'>Response level ({resp['qwen']['n']} responses): qwen κ {resp['qwen']['kappa']:.3f} "
                 f"(se {resp['qwen']['se']:.3f}), offline κ {resp['offline']['kappa']:.3f} (se {resp['offline']['se']:.3f}). "
                 f"Package calibration (used for the headline): se {c['package']['se']:.3f}, sp {c['package']['sp']:.3f}, "
                 f"κ {c['package']['kappa']:.3f}, {c['package']['n']} complete bundles"
                 + (f", {c['dropped_bundles']} dropped as incomplete" if c.get("dropped_bundles") else "") + ".</p>")
        if name == "sorrybench":
            extra += ("<p class='note'>The sheet labels every resolved bundle of the anchor pair (strata 26 judge-positive + 34 "
                      "judge-negative = the whole population), so these rates need no re-weighting.</p>")
        else:
            d = c.get("design") or {}
            if d:
                extra += (f"<p class='note warnt'>Sampled 30 of 30 judge-positives but 30 of {c['strata'][1]:,} judge-negatives. "
                          f"<code>calibrate_from_human</code> passes the sample stratum sizes to <code>invert_stratified</code>, so its "
                          f"se {c['package']['se']:.3f} assumes a 50/50 population. With the population strata the judge's sensitivity is "
                          f"{d['qwen']['se']:.3f} {ci_txt(*d['qwen']['se_ci'], d=1)}.</p>")
        blocks.append(panel(title, '<div class="cmrow">' + confusion(q, f"qwen2.5:14b vs human ({lvl})")
                            + confusion(o, f"offline rule vs human ({lvl})") + "</div>"
                            + table(["Judge", "Sensitivity (95%)", "Specificity (95%)", "Cohen's κ", "TP / FN / FP / TN"], rows) + extra))
    return section("s-judges", "12", "Judges — who decided what counts as an answer",
                   panel("Judge inventory", table(["Judge", "What it is", "Used by", "Calibration"], inv))
                   + '<div class="grid2">' + "".join(blocks) + "</div>"
                   + panel("What this means", "<ul class='take'>"
                           "<li>On SORRY-Bench the offline rule agrees with humans better than qwen2.5:14b, mainly through sensitivity: qwen "
                           "labels some compliant answers as refusals. The offline-judged pilots and the qwen-judged run therefore disagree on "
                           f"P⁻ (the offline P⁻ is larger for {judge_gap(s)[0]} of {judge_gap(s)[1]} lineage pairs); because qwen under-counts "
                           "compliance, its RSR is the more conservative of the two.</li>"
                           "<li>On attack outputs neither judge is adequate: qwen under-detects compliance in transformed replies, the offline rule "
                           "over-detects it in garbled ones. A per-class human audit of judge-negatives is the missing piece.</li>"
                           "<li>Every corrected figure on this page uses the package's bundle-level Rogan–Gladen correction with the SORRY-Bench "
                           "calibration; it is never applied to the attack arm.</li></ul>"))


def sec_gate(s):
    rows, keys = [], [("adm", "sb-adm", "admitted"), ("lexical_divergence", "sb-lex", "lexical divergence"),
                      ("cosine", "sb-cos", "embedding cosine"), ("bertscore", "sb-bert", "BERTScore"), ("nli_entail", "sb-nli", "NLI entailment")]
    trows = []
    for rr in s.recycle:
        t = rr.gate["total"]
        parts = {"adm": t["adm"], **dict(t["rej"])}
        rows.append(dict(label=rr.spec.id, sub=f"{dsl(s, rr.spec.dataset)} · {t['gen']:,} generated", parts=parts, total=t["gen"]))
        trows.append([esc(rr.spec.id), esc(dsl(s, rr.spec.dataset)), num(t["gen"]), f"{t['adm']:,} ({pct(t['adm'] / t['gen'] if t['gen'] else None, 0)})",
                      num(t["rej"].get("lexical_divergence", 0)), num(t["rej"].get("cosine", 0)), num(t["rej"].get("bertscore", 0)),
                      num(t["rej"].get("nli_entail", 0)), num(rr.gate["human_labelled"]), "no" if not rr.gate["intent_ran"] else "yes"])
    return section("s-gate", "13", "The intent gate",
                   panel("Where each manifest's rewordings were stopped", stacked_bars(rows, keys, "Gate outcomes per manifest")
                         + '<div class="legend">' + "".join(f'<span><i class="lg {c}"></i>{esc(l)}</span>' for _, c, l in keys) + "</div>"
                         + tableview(["Manifest", "Benchmark", "Generated", "Admitted", "Lexical", "Cosine", "BERTScore", "NLI",
                                      "Human labels", "L2 ran"], trows, num_cols=(2, 4, 5, 6, 7, 8)),
                         "First failing layer per rejected rewording. The lexical-divergence floor (rewording too close to the original) is the "
                         "largest single rejecter on several manifests; its score is not persisted. No manifest has a human-labelled audit row, and "
                         "the LLM intent layer never ran, so the gate's false-admission rate is still unmeasured.")
                   , lede="Thresholds: cosine ≥ 0.75, BERTScore ≥ 0.85, NLI ≥ 0.60 (strict) — ANDed after a lexical-divergence floor.")


def sec_explore(s):
    rows = []
    for d in ["wrapped_rs", "llama_wrapped", "llama32jb"]:
        for o, n in [("llama2-7b", "llama3-8b"), ("llama3-8b", "llama31-8b")]:
            c = s.cells.get((o, n, d)) or sd._adhoc_cell(o, n, s.datasets[d], s.screens)
            pil = [p for p in s.pilots if p.dataset == d and (p.old, p.new) == (o, n) and p.status == "valid"]
            if not c and not pil:
                continue
            jl = {"offline": "offline rule"}
            rep = [(p.p_minus, jl.get(p.judge, p.judge.replace("llm (", "").rstrip(")"))) for p in pil]
            pm_known = [x for x, _ in rep] + ([len(c.p_minus)] if c and c.status in ("complete", "p-minus-only") else [])
            if pm_known and max(pm_known) == 0:
                rec = "undefined (P⁻ = 0)"
            elif d == "llama32jb" and (o, n) == ("llama3-8b", "llama31-8b"):
                rec = "not run (pilot P⁻ = 3 < 30)"
            else:
                rec = NOT_YET
            rows.append([esc(dsl(s, d)), esc(pair_short(o, n)),
                         f"{c.breaks_old}/{s.datasets[d].n}" if c and c.breaks_old is not None else "—",
                         (str(len(c.p_minus)) if c and c.status in ("complete", "p-minus-only") else ("◐ incomplete" if c and c.status == "incomplete" else NOT_YET)),
                         "; ".join(f"{x} ({j})" for x, j in rep) or "—", rec])
    return section("s-explore", "14", "Exploratory framing runs",
                   panel("Fixed jailbreak templates and the llama32jb set", table(["Set", "Pair", "Older answered", "P⁻ (offline)",
                                                                                 "P⁻ reported (judge)", "Recycle"], rows),
                         "Wrapping behaviours in a fixed template was the August plan to grow P⁻ when raw prompts were refused. Llama 2 7B "
                         "answered 5/300 under refusal suppression and 7/300 under AIM, so the lever did not move the older model; the judge was "
                         "never validated on wrapped output (the open “case A vs case B” question). The llama32jb anchor pilot found P⁻ = 3 and the "
                         "recycle run was not carried out."))


def sec_registry(s):
    order = {"circular-judge": 0, "invalid": 1, "stale-copy": 2, "incomplete": 3, "superseded": 4, "empty": 5, "mock": 6, "missing": 7, "valid": 8}
    reg = sorted(s.registry, key=lambda a: (order.get(a.status, 9), a.kind, a.path))
    cnt = Counter(a.status for a in reg)
    n_flag = sum(v for k, v in cnt.items() if k != "valid")
    btns = "".join(f'<button type="button" class="fbtn{" active" if k == "flagged" else ""}" data-filter="{k}" '
                   f'aria-pressed="{"true" if k == "flagged" else "false"}">{esc(l)} <span>{c}</span></button>'
                   for k, l, c in [("flagged", "Flagged", n_flag), ("all", "All", len(reg))]
                   + [(k, STATUS_STYLE.get(k, ("", "", k))[2], v) for k, v in sorted(cnt.items(), key=lambda x: order.get(x[0], 9))])
    trs = "".join(f'<tr class="regrow" data-outcome="{esc(a.status)}"{" hidden" if a.status == "valid" else ""}>'
                  f'<td class="nw">{status_chip(a.status)}</td><td>{esc(a.kind)}</td>'
                  f'<td>{path_html(a.path)}</td><td>{esc(a.scope)}</td><td class="nw">{judge_chip(a.judge) if a.judge not in ("—", "") else "—"}</td>'
                  f'<td class="nw">{esc(a.mtime[:10])}</td><td class="small wide">{esc("; ".join(a.issues))}</td></tr>' for a in reg)
    ov = s.overlap
    sh = "; ".join(f"{a} ∩ {b}: {v:,} shared keys, {ov['differ'].get((a, b), 0)} with different text" for (a, b), v in ov["shared"].items())
    return section("s-registry", "15", "Run registry & integrity",
                   panel("Every artifact, with a verdict",
                         f'<div class="cases"><div class="filters" role="group" aria-label="Filter artifacts">{btns}</div>'
                         f'<div class="tscroll"><table class="tbl regtbl"><thead><tr><th class="nw">Status</th><th>Kind</th><th>Path</th>'
                         f'<th>Scope</th><th>Judge</th><th class="nw">Written</th><th>Findings</th></tr></thead><tbody>{trs}</tbody></table></div></div>',
                         "Showing flagged artifacts; use the filters for the valid ones. Checks: summary vs its own manifest, pilot pair "
                         "and size vs its run's META, identical content across files, "
                         "re-screen completeness from the cache, the judge recorded in each RPE log, and reproduction of every offline-judged "
                         "number from the cache.")
                   + panel("Caches", f"<p class='note'>{esc(sh)}. Each reconstruction uses the cache its run wrote "
                                     f"(regenerated responses differ slightly between caches).</p>"
                           + mtable(["Cache", "Responses"], [[path_html(sd.CACHE_FILES[k]), num(v)] for k, v in ov["entries"].items()],
                                    num=(1,))))


def sec_appendix(s):
    try:
        import positioning_section as ps
        rr, cal = anchor(s)
        frag = ps.render_fragment(anchor_pair="llama3-8b → llama31-8b",
                                  mcnemar_p=p_txt(cal["RSR"]["mcnemar_p"]) if cal else "—")
    except Exception as e:                                         # noqa: BLE001
        frag = f"<p class='note'>positioning_section unavailable ({esc(type(e).__name__)})</p>"
    cmds = ("python3 tools/master_dashboard.py --attack-70b-responses \"~/Downloads/all_local_results_llama3_1_70b 1.jsonl\"\n"
            "                                                       # this page (the flag adds the per-item 70B checks)\n"
            "python3 tools/lineage_dashboard.py                     # SORRY-Bench lineage, case level\n"
            "python3 tools/judge_lineage.py --scope all             # per-item qwen verdicts for the lineage (Ollama)\n"
            "python3 tools/apply_calibration.py ...                 # anchor headline (see its docstring)\n"
            "python3 -m pytest tests/ -q")
    errata = ("<ul class='take'><li>Risk register row “4-signal intent gate … report its false-admission rate — built”: the L2 intent "
              "signal never ran and no audit row is human-labelled, so the gate is 3-signal + lexical floor and its error is unmeasured.</li>"
              f"<li>The McNemar p shown in 07·g is the anchor pair's ({p_txt(anchor(s)[1]['RSR']['mcnemar_p']) if anchor(s)[1] else '—'}), "
              "not the HarmBench pilot's 6×10⁻⁵ used by the previous dashboard.</li></ul>")
    return section("s-appendix", "16", "Reproduce · positioning appendix",
                   panel("Rebuild everything", f"<pre class='cmd'>{esc(cmds)}</pre>")
                   + panel("Errata against current evidence", errata)
                   + f'<div class="panel posbox"><details><summary class="possum">Positioning (reference content from the previous '
                     f'dashboard, unchanged) — show</summary><div class="posfix">{frag}</div></details></div>')


NAV = [("Overview", [("s-rq", "00", "Research questions"), ("s-summary", "01", "Executive summary"), ("s-status", "02", "What's finished"), ("s-design", "03", "Study design")]),
       ("Results", [("s-screening", "04", "Screening"), ("s-pminus", "05", "Regression sets"), ("s-rsr", "06", "Recycling (RSR)"),
                    ("s-pbr", "07", "Robustness (PBR)"), ("s-anchor", "08", "Calibrated anchor"), ("s-rpe", "09", "Output inversion"),
                    ("s-arcs", "10", "Style arcs"), ("s-attack", "11", "Attack transfer")]),
       ("Instruments", [("s-judges", "12", "Judges"), ("s-gate", "13", "Intent gate")]),
       ("Provenance", [("s-explore", "14", "Exploratory runs"), ("s-registry", "15", "Registry & integrity"),
                       ("s-appendix", "16", "Reproduce · appendix")])]


RQ_MAP = {"s-pminus": ["RQ1"], "s-rsr": ["RQ1", "RQ2"], "s-anchor": ["RQ1"], "s-rpe": ["RQ1"], "s-arcs": ["RQ3"], "s-pbr": ["RQ4"]}
RQ_WHERE = {"RQ1": [("s-pminus", "Regression sets"), ("s-rsr", "Recycling"), ("s-anchor", "Calibrated anchor"), ("s-rpe", "Output inversion")],
            "RQ2": [("s-rsr", "Recycling by category")], "RQ3": [("s-arcs", "Style arcs")], "RQ4": [("s-pbr", "Robustness (PBR)")]}


def _latest(pattern):
    xs = sorted(sd.ROOT.glob(pattern))
    return xs[-1] if xs else None


def _sign_p(w, n):
    """two-sided exact sign test."""
    k = max(w, n - w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def mech_sign(path):
    """per behaviour: mean last-layer refusal-direction score of re-opening vs still-refused rewordings."""
    import json
    m = json.loads(path.read_text())
    out = {}
    for mod, r in m.get("results", {}).items():
        w = n = 0
        for it in r.get("items", []):
            ro = [x["score"][-1] for x in it["prompts"] if x["role"] == "reopen"]
            he = [x["score"][-1] for x in it["prompts"] if x["role"] == "held"]
            if ro and he:
                n += 1
                w += sum(ro) / len(ro) < sum(he) / len(he)
        out[mod] = (w, n, _sign_p(w, n) if n else None, len(r.get("items", [])))
    return out


def _pf(p):
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}" if p < 0.01 else f"p = {p:.2f}"


_MLAB = {"llama3-8b": "Llama 3 8B", "llama31-8b": "Llama 3.1 8B"}
_BLAB = {"sorrybench": "SORRY-Bench", "harmbench": "HarmBench", "advbench": "AdvBench", "strongreject": "StrongREJECT"}


def rq_answers():
    """current answers, recomputed from the newest dated analysis and white-box files (exploratory until the confirmatory run)."""
    import json
    a_path, m_path = _latest("results/analysis_*/analysis.json"), _latest("results/mech/anchor_*_all_*.json")
    ans, src = {}, []
    if a_path:
        a = json.loads(a_path.read_text())
        src.append(str(a_path.relative_to(sd.ROOT)))
        h = a.get("headline", {})
        po = h.get("pooled") or {}
        if po.get("n_pminus"):
            per = ", ".join(f"{ld.esc(_BLAB.get(x['bench'], x['bench']))} {x['k']}/{x['n_pminus']}" for x in h.get("per_benchmark", []) if x.get("n_pminus"))
            ans["RQ1"] = (f"<b>Yes, often.</b> Anchor {ld.esc(h.get('pair', ''))}, {len(h.get('per_benchmark', []))} benchmarks, one judge: "
                          f"<b>{po['k']}/{po['n_pminus']}</b> fixes re-opened, RSR <b>{po['rate']:.1%}</b> [{po['lo']:.1%}, {po['hi']:.1%}]"
                          + (f", judge-corrected {po['corrected']:.1%} [{po['corrected_lo']:.1%}, {po['corrected_hi']:.1%}]" if po.get("corrected") is not None else "")
                          + f"; McNemar {_pf(po['mcnemar_p'])}. Per benchmark: {per}. "
                          "Exploratory until the registered held-out replication (Amendment 1).")
        dom = a.get("rq2_sorry_lineage_by_domain") or {}
        if dom:
            xs = sorted(dom.items(), key=lambda kv: -kv[1]["rate"])
            hi, lo = xs[0], xs[-1]
            ans["RQ2"] = (f"<b>Brittleness is uneven.</b> SORRY-Bench lineage by domain: highest in {ld.esc(hi[0].lower())} "
                          f"({hi[1]['k']}/{hi[1]['n']}, {hi[1]['rate']:.0%}), lowest in {ld.esc(lo[0].lower())} "
                          f"({lo[1]['k']}/{lo[1]['n']}, {lo[1]['rate']:.0%}). Re-opening in every domain argues against a single "
                          "intent-level fix; the wide intervals argue against ranking the domains.")
        d3 = (a.get("rq3_drivers") or {}).get("anchor_pooled") or {}
        if d3.get("table"):
            sig = [r for r in d3["table"] if r["term"].startswith("C(transform_class)") and r.get("driver_after_holm")]
            sd_ = next((r for r in d3["table"] if r["term"] == "surface_distance"), None)
            nm = lambda t: t.split("[T.")[-1].rstrip("]").replace("_", " ")
            ans["RQ3"] = ((f"<b>{', '.join(nm(r['term']) for r in sig)}</b> rewordings drive recovery "
                           + "(" + "; ".join(f"OR {r['odds_ratio']:.2f}" for r in sig) + ", vs back-translation, Holm-corrected)"
                           if sig else "No transform class survives Holm correction")
                          + (f"; surface distance itself does not (OR {sd_['odds_ratio']:.2f}, p = {sd_['p_value']:.2f})" if sd_ else "")
                          + f". Anchor, {d3.get('n_obs')} rewordings of {d3.get('n_groups')} fixes, behaviour random intercept.")
        pb = [r for r in a.get("pbr", []) if r.get("complete") and r.get("pbr") is not None]
        if pb:
            by = defaultdict(list)
            for r in pb:
                by[r["bench"]].append(r)
            parts = []
            for b, rs in by.items():
                c = Counter(r["verdict"] for r in rs)
                parts.append(f"{ld.esc(_BLAB.get(b, b))}: " + ", ".join(f"{v} {k}" for k, v in c.most_common()))
            ans["RQ4"] = ("<b>Depends on the benchmark.</b> PBR vs π* = 0.15 · " + " · ".join(parts)
                          + ". On SORRY-Bench the newer 8B models are the more brittle; on HarmBench the order reverses within overlapping intervals.")
    if m_path:
        src.append(str(m_path.relative_to(sd.ROOT)))
        ms = mech_sign(m_path)
        bits = [f"{ld.esc(_MLAB.get(k, k))} {w}/{n} ({_pf(p)})" for k, (w, n, p, _) in ms.items() if n]
        if bits:
            tot = max(v[3] for v in ms.values())
            ans["Beyond the RQs"] = (f"<b>Mostly, in the newer model.</b> Anchor SORRY-Bench, {tot} fixes: the re-opening rewording sits lower "
                                     "on the refusal direction than a still-refused one in " + "; ".join(bits)
                                     + ". Correlational, last prompt token, reproduced under the server's decoding (repeat penalty 1.1/64).")
    return ans, src


def sec_rq(s):
    ans, src = rq_answers()
    where = dict(RQ_WHERE)
    if "Beyond the RQs" in ans:
        where["Beyond the RQs"] = []
    status = ("Current answers are recomputed at build time from " + ", ".join(f"<code>{ld.esc(x)}</code>" for x in src)
              + ". Every answer is exploratory: the confirmatory replication is registered in Amendment 1 and has not run.") if src else ""
    return ld.rq_section(answers=ans, where=where, frame=ld.RQ_FRAME, status=status)


EXTRA_CSS = r"""
.lede2{margin-top:-6px}
.tiles3{grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))}
.tbl .nw{white-space:nowrap}
.tbl .wide{min-width:260px}
.pathc{display:inline-block;font-family:var(--mono);font-size:11.5px;line-height:1.35;max-width:300px}
.pathc .pf{display:block;white-space:nowrap;font-weight:600;color:var(--ink)}
.pathc .pd{display:block;color:var(--muted);font-size:10.5px;word-break:normal;overflow-wrap:normal}
p.small{font-size:12px;margin:4px 0}
.meta .chip{white-space:normal}
@media (max-width:980px){nav.side .grp{width:100%;margin:8px 4px 2px}}
.glabel{fill:var(--ink);font-weight:650;font-size:12.5px;letter-spacing:.02em}
.chart .pt-hollow{fill:var(--card);stroke:var(--comply);stroke-width:2.2}
.chart .row.muted .pt-hollow{stroke:var(--none)}
.lg-ring{background:transparent!important;border:2px solid var(--comply);border-radius:50%!important}
nav.side .grp{font-size:10.5px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:14px 10px 4px}
.st{display:inline-flex;align-items:center;gap:4px;font-size:11.5px;font-weight:600;white-space:nowrap}
.st-ok{color:var(--good-ink)}.st-warn{color:var(--warn-ink)}.st-crit{color:var(--crit)}.st-muted{color:var(--muted)}
.jchip{display:inline-block;font-size:11px;font-weight:600;padding:1px 8px;border-radius:999px;border:1px solid var(--grid);white-space:nowrap}
.j-off{border-color:var(--refuse)}.j-llm{border-color:var(--fix-ink)}.j-circ{border-color:var(--crit);color:var(--crit)}.j-hum{border-color:var(--ink)}
.stbl td{vertical-align:top}
.sub2{display:block;font-size:11px;color:var(--muted);font-weight:400}
.pcell{min-width:150px}
.pl{font-size:11.5px;line-height:1.35;padding-left:9px;position:relative;margin:2px 0}
.pl::before{content:"";position:absolute;left:0;top:.42em;width:5px;height:5px;border-radius:50%;background:var(--muted)}
.pl-ok::before{background:var(--good)}.pl-warn::before{background:var(--warn)}.pl-crit::before{background:var(--crit)}
.pl-muted{color:var(--muted)}.pl-crit{color:var(--crit);font-weight:600}
td.small,.small{font-size:12px;color:var(--ink2)}
td.hatch{background-image:repeating-linear-gradient(45deg,transparent 0 6px,color-mix(in oklab,var(--warn) 22%,transparent) 6px 8px)}
td.pmc b{font-size:15px}
td.pm-big{box-shadow:inset 0 0 0 2px var(--fix)}
.arcgrid-wrap{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,400px),1fr));gap:12px}
.arccard{border:1px solid var(--ring);border-radius:12px;padding:10px 12px;background:var(--surface)}
.arct{font-weight:650;font-size:13px}.arcs{font-size:11.5px;color:var(--muted)}
.arcf{font-size:12px;color:var(--ink2);margin-top:2px}.warnt{color:var(--warn-ink)}
.chart .arcgrid{fill:none;stroke:var(--grid);stroke-width:1}
.chart .arcdot{fill:var(--fix);fill-opacity:.85;stroke:var(--card);stroke-width:1.5}
.chart .arcnaive{fill:none;stroke:var(--muted);stroke-width:1.5;stroke-dasharray:none}
.chart .arclink{stroke:var(--muted);stroke-width:1}
.lg-arcdot{background:var(--fix)!important;border-radius:50%!important}
.lg-arcnaive{background:transparent!important;border:1.5px solid var(--muted);border-radius:50%!important}
.chart .dbline{stroke:var(--axis);stroke-width:2}
.chart .db-o8{fill:var(--comply)}.chart .db-o70{fill:var(--ink);fill-opacity:.85}
.chart .db-a8{fill:var(--card);stroke:var(--comply);stroke-width:1.8}
.lg.db-o8{background:var(--comply);border-radius:50%}.lg.db-o70{background:var(--ink);transform:rotate(45deg) scale(.8)}
.lg.db-a8{background:transparent;border:1.8px solid var(--comply);border-radius:50%}
.chart .sb-adm{fill:var(--fix)}.chart .sb-lex{fill:var(--muted)}.chart .sb-cos{fill:var(--refuse)}
.chart .sb-bert{fill:var(--ink2)}.chart .sb-nli{fill:var(--comply)}
.lg.sb-adm{background:var(--fix)}.lg.sb-lex{background:var(--muted)}.lg.sb-cos{background:var(--refuse)}
.lg.sb-bert{background:var(--ink2)}.lg.sb-nli{background:var(--comply)}
.chart .inbar{fill:#fff;font-size:11px;font-weight:600}
.cmrow{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 10px}
.cm{display:grid;grid-template-columns:auto 92px 92px;gap:4px;font-size:11.5px}
.cm-t{grid-column:1/-1;font-weight:650;font-size:12.5px;margin-bottom:2px}
.cm-h{color:var(--muted);display:flex;align-items:flex-end;justify-content:center;text-align:center}
.cm-h.cm-r{justify-content:flex-end;align-items:center;padding-right:4px}
.cm-c{border-radius:8px;padding:8px;text-align:center;border:1px solid var(--ring);color:var(--ink2)}
.cm-c b{display:block;font-size:18px;color:var(--ink);font-variant-numeric:tabular-nums}
.cm-ok{background:color-mix(in oklab,var(--fix) 14%,var(--card))}
.cm-bad{background:color-mix(in oklab,var(--comply) 14%,var(--card))}
.regtbl td{vertical-align:top}.regtbl code{font-size:11px}
.regtbl td:nth-child(2){max-width:130px}.regtbl td:nth-child(4){max-width:170px}
.regtbl td.wide{min-width:220px}
.note2{font-size:11.5px;color:var(--ink2);margin-top:3px;max-width:420px;white-space:normal}
tr.regrow[hidden]{display:none}
.posbox .posfix{color-scheme:light;background:#fff;border-radius:10px;padding:6px;overflow-x:auto;margin-top:10px}
.possum{cursor:pointer;font-weight:650;font-size:14px}
"""

EXTRA_JS = r"""
(function(){
  // registry filter: rows carry data-outcome like the case filter
  Array.prototype.forEach.call(document.querySelectorAll('.cases'), function(box){
    var rows = box.querySelectorAll('tr.regrow');
    if (!rows.length) return;
    Array.prototype.forEach.call(box.querySelectorAll('.fbtn'), function(b){
      b.addEventListener('click', function(){
        var f = b.getAttribute('data-filter');
        Array.prototype.forEach.call(rows, function(r){
          var o = r.getAttribute('data-outcome');
          r.hidden = !(f === 'all' || o === f || (f === 'flagged' && o !== 'valid'));
        });
      });
    });
  });
})();
"""


def render(s) -> str:
    today = _dt.date.today().isoformat()
    nav = ""
    for g, items in NAV:
        nav += f'<div class="grp">{esc(g)}</div>' + "".join(f'<a href="#{i}"><span class="n">{n}</span>{esc(t)}</a>' for i, n, t in items)
    body = (sec_header(s, today) + sec_rq(s) + sec_summary(s) + sec_status(s) + sec_design(s) + sec_screening(s) + sec_pminus(s)
            + sec_rsr(s) + sec_pbr(s) + sec_anchor(s) + sec_rpe(s) + sec_arcs(s) + sec_attack(s) + sec_judges(s)
            + sec_gate(s) + sec_explore(s) + sec_registry(s) + sec_appendix(s))
    foot = ("Generated by tools/master_dashboard.py from files only (study_data.py). No prompt or response text is embedded. "
            "RAW = judge output as produced; corrected figures name their calibration.")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<title>RecycleBench Master Results</title><style>{ld.CSS}{EXTRA_CSS}</style></head><body>'
            f'<div class="layout"><nav class="side" aria-label="Sections">'
            f'{ld.brand_html("Master results · all arms")}<div class="toc">{nav}</div>{ld.legend_html()}'
            f'<button type="button" class="themebtn" id="themebtn" data-t="system">Theme: system</button></nav>'
            f'<main id="main">{body}<footer>{esc(foot)}</footer></main></div>'
            f'<div id="tip" role="tooltip" hidden></div><script>{ld.rq_script(RQ_MAP)}{ld.JS}{EXTRA_JS}</script></body></html>')


def default_out() -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    base = sd.ROOT / "results" / f"dashboard_master_{stamp}.html"
    out, i = base, 2
    while out.exists():
        out = base.with_name(f"{base.stem}_{i}.html")
        i += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack-70b-responses", default=None,
                    help="the llama-3.1-70B responses JSONL judged by tools/judge_70b_jsonl.py (enables per-item 70B checks)")
    ap.add_argument("--out", default=None, help="default: results/dashboard_master_<stamp>.html (never overwrites)")
    args = ap.parse_args(argv)
    jsonl = os.path.expanduser(args.attack_70b_responses) if args.attack_70b_responses else None
    study = sd.build_study(jsonl)
    out = Path(args.out) if args.out else default_out()
    if out.exists():
        raise SystemExit(f"{out} exists -- refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(study))
    st = Counter(a.status for a in study.registry)
    print(f"wrote {sd.rel(out)}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"  artifacts audited: {len(study.registry)}  " + "  ".join(f"{k}={v}" for k, v in sorted(st.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
