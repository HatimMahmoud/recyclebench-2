#!/usr/bin/env python3
"""tools/compose_v2_section.py -- the 21-condition style factorial (compose v2) on the DFKI dashboards.

    aggregate(root)      newest results/runs/compose_v2_*/compose_v2_summary.json + the per-behaviour verdicts
                         rebuilt from cache/compose_v2/ through run_compose_v2.verdict_table (the same code `score` uses)
    section_body(agg)    tiles, then three tabs: the 6 x 6 style matrix, the registered pair-vs-singles
                         comparisons, and the per-behaviour detail view (49 fixes x 21 conditions)

Text-free: behaviour ids and SORRY-Bench categories only. Adds no <script>; the page's shared JS drives the
tooltips and the tabs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lineage_dashboard as ld                  # noqa: E402

esc, pct, num = ld.esc, ld.pct, ld.num
ABBR = {"paraphrase": "P", "syntactic": "S", "lexical": "L", "back_translation": "B", "tense_shift": "T", "register": "R"}
LEDE = ("Does rewording a fixed request with <b>two</b> styles in a row re-open fixes that neither style re-opens alone? "
        "Six styles and their fifteen pairs, three candidates each, on the 49 judged fixes of the anchor pair "
        "(Llama 3 8B → 3.1 8B), every candidate gated against the original request. Registered before the run "
        "(<code>PREREG_COMPOSE_V2_20261029.md</code>); exploratory.")


def label(c: str) -> str:
    return " + ".join(x.replace("_", " ") for x in c.split("+"))


def short(c: str) -> str:
    return "·".join(ABBR[x] for x in c.split("+"))


ANCHOR = ("llama3-8b", "llama31-8b")


def _pair_of(summ):
    return tuple(x.strip() for x in summ.get("pair", "").split("->"))


def latest(root: Path, pair=ANCHOR):
    """Newest compose v2 summary for `pair` (runs of other pairs live beside it)."""
    best = None
    for q in sorted((Path(root) / "results" / "runs").glob("compose_v2*/compose_v2_summary.json")):
        try:
            if _pair_of(json.loads(q.read_text())) == tuple(pair):
                best = q
        except (OSError, ValueError):
            continue
    return best


def aggregate(root: Path = ROOT, pair=ANCHOR):
    root = Path(root)
    p = latest(root, pair)
    if p is None:
        return None
    summ = json.loads(p.read_text())
    import lineage_core as lc
    import run_compose_v2 as rc
    from recyclebench.config import Config
    from recyclebench.datasets import load_jailbreakbench
    from recyclebench.realrun import ResponseCache
    store = root / (summ.get("store") or "cache/compose_v2")     # the anchor's summary predates the field
    V = None
    if (store / "candidates.jsonl").exists():
        gen = {}                                             # read-only; last write wins, as in JsonlStore
        for line in (store / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            gen[d["id"]] = d
        bids = sorted({d["bid"] for d in gen.values()})
        tag = lc.judge_tag_from_cfg(Config.load(ROOT / rc.CFG))
        V = rc.verdict_table(gen, ResponseCache(store / "replies.jsonl", enabled=True), lc.VerdictStore(store / "verdicts.jsonl"),
                             tag, bids, summ["candidate_levels_used"] - 1)
    cats = {}
    csv_p = root / "data" / "sorrybench.csv"
    if csv_p.exists():
        cats = {b.bid: lc.category_name(b.category) for b in load_jailbreakbench(str(csv_p))}
    return {"summary": summ, "path": str(p.relative_to(root)), "verdicts": V, "categories": cats}


def _matrix(summ) -> str:
    conds = {c["condition"]: c for c in summ["conditions"]}
    prim = {r["pair"]: r for r in summ["primary_pair_vs_either_single"]}
    classes = [c["condition"] for c in summ["conditions"] if c["kind"] == "single"]
    cw, ch, x0, y0 = 104, 58, 132, 40
    W, H = x0 + cw * len(classes) + 8, y0 + ch * len(classes) + 10
    s = [ld.svg_open(W, H, "Re-opening rate per style and style pair")]
    for j, c in enumerate(classes):
        s.append(f'<text x="{x0 + j * cw + cw / 2:.0f}" y="{y0 - 12}" class="ax strong" text-anchor="middle">'
                 f'{esc(c.replace("_", " "))}</text>')
    for i, a in enumerate(classes):
        s.append(f'<text x="{x0 - 10}" y="{y0 + i * ch + ch / 2 + 4:.0f}" class="ax strong" text-anchor="end">'
                 f'{esc(a.replace("_", " "))}</text>')
        for j, b in enumerate(classes):
            x, y = x0 + j * cw, y0 + i * ch
            if j < i:
                s.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cw - 4}" height="{ch - 4}" rx="6" class="cv2-void"/>')
                continue
            key = a if i == j else f"{a}+{b}"
            c = conds.get(key)
            if not c:
                continue
            r = c["rsr"]
            tip = [label(key) + (" (single style)" if i == j else " (pair: first, then second)"),
                   f"re-opened {c['reopened']} of {c['resolved']} resolved fixes"
                   + (f" ({pct(r)}, Wilson {ld.ci_txt(*c['wilson95'])})" if r is not None else ""),
                   f"gate admitted {c['admitted']} of {c['candidates']} candidates; {c['unresolved']} fixes unresolved"]
            pr = prim.get(key)
            if pr:
                tip.append(f"vs either single, {pr['n_shared']} shared fixes: pair {pr['k_pair']}, singles {pr['k_either_single']},"
                           f" pair-only {pr['composite_only']}; McNemar p {ld.p_txt(pr['p'])}, Holm {ld.p_txt(pr['p_holm'])}")
            fill = ld.heat(r).replace("background:", "fill:") if r is not None else ""
            s.append(f'<g class="cv2-cell{" cv2-diag" if i == j else ""}" tabindex="0" data-tip="{esc(chr(10).join(tip))}">'
                     f'<rect x="{x + 2}" y="{y + 2}" width="{cw - 4}" height="{ch - 4}" rx="6" style="{fill}"/>'
                     f'<text x="{x + cw / 2:.0f}" y="{y + ch / 2 + 1:.0f}" class="val" text-anchor="middle">'
                     f'{pct(r, 0) if r is not None else "—"}</text>'
                     f'<text x="{x + cw / 2:.0f}" y="{y + ch / 2 + 16:.0f}" class="ax" text-anchor="middle">'
                     f'{c["reopened"]}/{c["resolved"]}</text></g>')
    s.append("</svg>")
    return "".join(s)


def _arc(cx, cy, r, f0, f1, sweep=0.75):
    """SVG path of an arc on radius r from fraction f0 to f1 of `sweep` turns, clockwise from 12 o'clock."""
    import math
    a0, a1 = (-90 + 360 * sweep * f0), (-90 + 360 * sweep * f1)
    x0, y0 = cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0))
    x1, y1 = cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1))
    large = 1 if (a1 - a0) > 180 else 0
    return f"M{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 1 {x1:.1f},{y1:.1f}"


def _radial(summ, top=10) -> str:
    """The `top` conditions by re-opening rate as concentric arcs, best outermost. Arc length = RSR on a
    three-quarter turn; the faint whisker under it is the Wilson 95% interval; singles and pairs differ in colour."""
    import math
    cs = sorted((c for c in summ["conditions"] if c["rsr"] is not None),
                key=lambda c: (-c["rsr"], -c["resolved"]))[:top]
    if not cs:
        return ""
    n = len(cs)
    r_in, step = 74, 25
    R = r_in + step * (n - 1)
    cx, cy = 380, R + 34
    W, H = 760, cy + R + 44
    s = [ld.svg_open(W, int(H), f"Top {n} conditions by re-opening rate")]
    for t in (0, .25, .5, .75, 1):
        ang = math.radians(-90 + 270 * t)
        tx, ty = cx + (R + 26) * math.cos(ang), cy + (R + 26) * math.sin(ang)
        s.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx + (R + 12) * math.cos(ang):.1f}" '
                 f'y2="{cy + (R + 12) * math.sin(ang):.1f}" class="grid"/>'
                 f'<text x="{tx:.1f}" y="{ty + 4:.1f}" class="ax" text-anchor="middle">{int(t * 100)}%</text>')
    for k, c in enumerate(cs):
        r = R - k * step
        pair = c["kind"] == "pair"
        lo, hi = c["wilson95"]
        tip = (f"#{k + 1} {label(c['condition'])} ({'style pair' if pair else 'single style'})\n"
               f"re-opened {c['reopened']} of {c['resolved']} resolved fixes ({pct(c['rsr'])}, Wilson {ld.ci_txt(lo, hi)})\n"
               f"gate admitted {c['admitted']} of {c['candidates']} candidates")
        s.append(f'<g class="cv2-ring" tabindex="0" data-tip="{esc(tip)}">'
                 f'<path d="{_arc(cx, cy, r, 0, 1)}" class="cv2-track"/>'
                 f'<path d="{_arc(cx, cy, r, 0, max(c["rsr"], 0.004))}" class="{"cv2-arc-pair" if pair else "cv2-arc-single"}"/>'
                 f'<path d="{_arc(cx, cy, r, lo, max(hi, lo + 0.002))}" class="cv2-ci"/>'
                 f'<text x="{cx - 8}" y="{cy - r + 4}" class="lbl{" strong" if k == 0 else ""}" text-anchor="end">'
                 f'{esc(label(c["condition"]))} · {pct(c["rsr"], 0)} '
                 f'<tspan class="ax">{c["reopened"]}/{c["resolved"]}</tspan></text></g>')
    s.append(f'<text x="{cx}" y="{cy + 5}" class="ax strong" text-anchor="middle">top {n} of 21</text>')
    s.append("</svg>")
    singles_in = sum(1 for c in cs if c["kind"] == "single")
    return ("".join(s) + "<div class='legend'><span><i class='lg cv2-lg-single'></i>single style</span>"
            "<span><i class='lg cv2-lg-pair'></i>style pair (first, then second)</span>"
            "<span><i class='lg cv2-lg-ci'></i>Wilson 95% interval</span></div>"
            f"<p class='note'>Ranked by re-opening rate; {singles_in} of the top {n} are single styles. Rates rest on "
            f"{min(c['resolved'] for c in cs)}–{max(c['resolved'] for c in cs)} resolved fixes each, so neighbouring rings "
            "overlap within their intervals: the ranking is "
            "descriptive, and the registered test (next tab) finds no pair better than its singles.</p>")


def _comparisons(summ) -> str:
    rows = []
    for r in sorted(summ["primary_pair_vs_either_single"], key=lambda r: (r["p"] is None, r["p"] or 1)):
        d = r["k_pair"] - r["k_either_single"]
        rows.append([esc(label(r["pair"])), num(r["n_shared"]), num(r["k_pair"]), num(r["k_either_single"]),
                     f'<span class="{"warnt" if d > 0 else "sub"}">{d:+d}</span>', num(r["composite_only"]),
                     ld.p_txt(r["p"]), ld.p_txt(r["p_holm"])])
    prim = ld.table(["style pair", "shared fixes", "pair re-opens", "either single re-opens", "difference",
                     "pair only", "McNemar p", "Holm p"], rows, num_cols=(1, 2, 3, 5))
    srows = []
    for r in sorted(summ["secondary_pair_vs_each_single"], key=lambda r: (r["p"] is None, r["p"] or 1)):
        srows.append([esc(label(r["pair"])), esc(r["single"].replace("_", " ")), num(r["n_shared"]), num(r["k_pair"]),
                      num(r["k_single"]), ld.p_txt(r["p"]), ld.p_txt(r["p_holm"])])
    sec = ld.table(["style pair", "against single", "shared fixes", "pair", "single", "McNemar p", "Holm p"],
                   srows, num_cols=(2, 3, 4))
    n_sig = sum(1 for r in summ["primary_pair_vs_either_single"] if r["p_holm"] is not None and r["p_holm"] < 0.05)
    return (f"<div class='panel'><h3>Primary: each pair against either of its singles</h3>{prim}"
            "<p class='note'>On the fixes all three conditions resolved, does the pair re-open a fix that neither single "
            f"re-opened? {n_sig} of 15 pairs differ after Holm–Bonferroni. “Either single” pools both singles' 3 candidates "
            "(6 in all) against the pair's 3, which favours the singles; the equal-budget comparison below finds no "
            "difference either.</p></div>"
            f"<details class='tv'><summary>Secondary: each pair against each single, equal budget (30 tests)</summary>{sec}</details>")


def _detail(summ, V, cats) -> str:
    if not V:
        return f"<p class='note'>Per-behaviour verdicts are not on disk (cache/compose_v2/).</p>"
    conds = [c["condition"] for c in summ["conditions"]]
    bids = sorted(next(iter(V.values())).keys(), key=lambda b: (-sum(1 for c in conds if V[c][b] == 1), b))
    cw, rh, x0, y0 = 26, 16, 268, 78
    W, H = x0 + cw * len(conds) + 52, y0 + rh * len(bids) + 8
    s = [ld.svg_open(W, H, "Every fix under every condition")]
    for j, c in enumerate(conds):
        x = x0 + j * cw + cw / 2
        s.append(f'<text x="{x:.0f}" y="{y0 - 8}" class="ax strong" text-anchor="start" '
                 f'transform="rotate(-60 {x:.0f} {y0 - 8})"><title>{esc(label(c))}</title>{esc(short(c))}</text>')
    xs = x0 + 6 * cw
    s.append(f'<line x1="{xs - 1}" x2="{xs - 1}" y1="{y0 - 4}" y2="{H - 6}" class="axis"/>')
    s.append(f'<text x="{x0 + cw * len(conds) + 8}" y="{y0 - 8}" class="ax strong">re-opened</text>')
    for i, b in enumerate(bids):
        y = y0 + i * rh
        cat = cats.get(b, "")
        n_on = sum(1 for c in conds if V[c][b] == 1)
        s.append(f'<text x="8" y="{y + 11}" class="ax"><tspan class="strong">{esc(b)}</tspan> '
                 f'{esc(cat[:30])}</text>')
        for j, c in enumerate(conds):
            v = V[c][b]
            cls, st = ({1: ("cv2-on", "re-opened"), 0: ("cv2-off", "held")}.get(v, ("cv2-na", "no admitted rewording")))
            s.append(f'<rect x="{x0 + j * cw + 2}" y="{y + 2}" width="{cw - 4}" height="{rh - 4}" rx="2" class="{cls}" '
                     f'tabindex="-1" data-tip="{esc(f"{b} · {cat}{chr(10)}{label(c)}: {st}")}"/>')
        s.append(f'<text x="{x0 + cw * len(conds) + 8}" y="{y + 11}" class="ax">{n_on}/21</text>')
    s.append("</svg>")
    key = " · ".join(f"{v} = {k.replace('_', ' ')}" for k, v in ABBR.items())
    return (f"<div class='panel'><h3>Every fix under every condition</h3><div class='scroll'>{''.join(s)}</div>"
            "<div class='legend'><span><i class='lg cv2-on-lg'></i>re-opened (≥ 1 admitted candidate answered)</span>"
            "<span><i class='lg cv2-off-lg'></i>held</span><span><i class='lg cv2-na-lg'></i>no admitted rewording</span></div>"
            f"<p class='note'>Rows: the 49 judged fixes, most-often re-opened first; ids and SORRY-Bench categories only. "
            f"Columns: the 6 single styles, then the 15 pairs (first style, then second). {esc(key)}.</p></div>")


def section_body(agg, uid="cv2", second=None) -> str:
    if not agg:
        return "<p class='note'>Not run: no results/runs/compose_v2_*/compose_v2_summary.json.</p>"
    summ = agg["summary"]
    cs = summ["conditions"]
    singles = [c for c in cs if c["kind"] == "single" and c["rsr"] is not None]
    pairs = [c for c in cs if c["kind"] == "pair" and c["rsr"] is not None]
    bs, bp = max(singles, key=lambda c: c["rsr"]), max(pairs, key=lambda c: c["rsr"])
    adm, cand = sum(c["admitted"] for c in cs), sum(c["candidates"] for c in cs)
    n_sig = sum(1 for r in summ["primary_pair_vs_either_single"] if r["p_holm"] is not None and r["p_holm"] < 0.05)
    tiles = (ld.tile("Pairs beating their singles", f"{n_sig} of 15", "registered primary test, Holm-corrected", emph=True)
             + ld.tile("Best single style", pct(bs["rsr"], 0), f"{label(bs['condition'])} · {bs['reopened']}/{bs['resolved']}")
             + ld.tile("Best style pair", pct(bp["rsr"], 0), f"{label(bp['condition'])} · {bp['reopened']}/{bp['resolved']}")
             + ld.tile("Gate admitted", f"{adm:,}", f"of {cand:,} candidates · {summ['candidate_levels_used']}/3 levels"))
    views = [("matrix", "Style matrix", f"<div class='panel'><h3>Re-opening rate per condition</h3>{_matrix(summ)}"
              "<p class='note'>Diagonal: one style. Above it: the pair, row style first. Share of the fixes the condition "
              "resolved (≥ 1 admitted candidate) that llama3.1-8b answered, judged by qwen2.5:14b against the candidate. "
              "Hover a cell for counts, gate admission and its test.</p></div>"),
             ("radial", "Top 10 · radial", f"<div class='panel'><h3>The ten conditions that re-open most</h3>{_radial(summ)}</div>"),
             ("tests", "Pair vs singles", _comparisons(summ)),
             ("detail", "Per behaviour", _detail(summ, agg.get("verdicts"), agg.get("categories", {})))]
    if second:
        s2 = second["summary"]
        o2, n2 = _pair_of(s2)
        n2sig = sum(1 for r in s2["primary_pair_vs_either_single"] if r["p_holm"] is not None and r["p_holm"] < 0.05)
        views.append(("second", f"{o2} → {n2}",
                      f"<p class='note'>The same registered design on a second pair "
                      f"(<code>PREREG_COMPOSE_V2_ADDENDUM_20261030.md</code>): {s2['behaviours']} judged fixes; "
                      f"{n2sig} of 15 style pairs beat their singles after Holm. Few fixes, so little power.</p>"
                      f"<div class='panel'><h3>Re-opening rate per condition</h3>{_matrix(s2)}</div>"
                      f"<div class='panel'><h3>Top 10 · radial</h3>{_radial(s2)}</div>{_comparisons(s2)}"))
    tabs, panels = [], []
    for k, (v, name, body) in enumerate(views):
        on = k == 0
        tabs.append(f'<button type="button" role="tab" id="tab-{uid}-{v}" aria-controls="panel-{uid}-{v}" '
                    f'aria-selected="{"true" if on else "false"}" tabindex="{0 if on else -1}" class="tab">{esc(name)}</button>')
        panels.append(f'<div role="tabpanel" id="panel-{uid}-{v}" aria-labelledby="tab-{uid}-{v}"{"" if on else " hidden"}>{body}</div>')
    return (f'<div class="tiles">{tiles}</div><div class="tablist" role="tablist" aria-label="Style factorial view">'
            f'{"".join(tabs)}</div>{"".join(panels)}'
            f"<p class='note'>Source: <code>{esc(agg['path'])}</code>; per-behaviour cells rebuilt from "
            "<code>cache/compose_v2/</code> by the same code that wrote the summary.</p>")


CSS = """
.chart .cv2-void{fill:var(--wash)}
.chart .cv2-cell rect{stroke:var(--ring);stroke-width:1}
.chart .cv2-diag rect{stroke:var(--ink2);stroke-width:1.5}
.chart .cv2-cell:hover rect,.chart .cv2-cell:focus rect{stroke:var(--ink);stroke-width:2}
.chart .cv2-track{fill:none;stroke:var(--wash);stroke-width:13;stroke-linecap:round}
.chart .cv2-ci{fill:none;stroke:var(--ink);stroke-width:2;stroke-linecap:round;opacity:.45}
.chart .cv2-arc-single{fill:none;stroke:var(--brand);stroke-width:11;stroke-linecap:round}
.chart .cv2-arc-pair{fill:none;stroke:var(--comply);stroke-width:11;stroke-linecap:round}
.chart .cv2-ring:hover .cv2-track,.chart .cv2-ring:focus .cv2-track{stroke:var(--grid)}
.lg.cv2-lg-single{background:var(--brand)}.lg.cv2-lg-pair{background:var(--comply)}.lg.cv2-lg-ci{background:var(--ink);opacity:.45;height:2px;vertical-align:middle}
.chart .cv2-on{fill:var(--comply)}.chart .cv2-off{fill:var(--refuse);fill-opacity:.55}
.chart .cv2-na{fill:none;stroke:var(--axis);stroke-width:1;stroke-dasharray:2 2}
.lg.cv2-on-lg{background:var(--comply)}.lg.cv2-off-lg{background:var(--refuse);opacity:.55}
.lg.cv2-na-lg{background:transparent;border:1px dashed var(--axis)}
"""
