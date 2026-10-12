#!/usr/bin/env python3
"""
tools/style_arcs.py -- concentric semicircular-arc diagram of paraphrase STYLES.

OPTION A (gate quality, no re-run): reads the recycle manifests' `audit` rows, which
carry per-variant {transform_class, admitted, nli_entail, cosine, bertscore}. For each
paraphrase style it computes how well that style PRESERVES INTENT (mean NLI) and how
often its variants pass the gate (admission rate), and places one dot per style on
concentric semicircle tracks.

Encoding (matches the reference image):
  - one DOT per transform_class (paraphrase style)
  - RADIUS  = the chosen metric (default: mean NLI). Outer = better intent-preservation.
              (swap --metric admission_rate to show gate pass-rate instead.)
  - ANGLE   = styles spread evenly across the semicircle
  - DOT SIZE = variant count n for that style (confidence/volume)
  - STAR dot = the best style on the metric
  - DARK/hollow dot = the worst style on the metric
  - BASE ARC (blue) = the overall value of the metric across all variants (a gauge)
  - concentric dashed tracks = metric gridlines (0 .. 1)

NOTE ON LABELLING: this is "intent-preservation / gate quality by style", NOT
"model vulnerability by style". The per-variant BYPASS outcome was not persisted in
these runs (audit rows have success=None), so model-vulnerability (Option B) needs a
re-run with trials logging. This diagram is built only on the deterministic gate
signals (NLI/cosine/admission), which are sound without judge calibration.

Standalone:
    python3 tools/style_arcs.py \
        --manifests "results/runs/*RECYCLE*/manifest_*.jsonl" \
        --metric nli --out results/runs/.../style_arcs.html
It also exposes render_fragment(...) for embedding in the dashboard (CSS 'sa-' scoped).
"""
from __future__ import annotations
import argparse, glob, json, math, html
from collections import defaultdict

METRICS = {
    "nli":            ("mean_nli", "mean NLI entailment", "intent preserved"),
    "admission_rate": ("adm_rate", "gate admission rate", "passed the gate"),
    "cosine":         ("mean_cos", "mean embedding cosine", "surface similarity"),
}


def esc(s):
    return html.escape(str(s if s is not None else ""))


def aggregate(manifest_globs):
    agg = defaultdict(lambda: {"n": 0, "adm": 0, "nli": 0.0, "cos": 0.0, "bert": 0.0})
    files = []
    for g in manifest_globs:
        files.extend(glob.glob(g))
    for f in sorted(set(files)):
        try:
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if r.get("type") != "audit":
                    continue
                tc = r.get("transform_class")
                if not tc:
                    continue
                a = agg[tc]
                a["n"] += 1
                a["adm"] += 1 if r.get("admitted") else 0
                a["nli"] += r.get("nli_entail") or 0.0
                a["cos"] += r.get("cosine") or 0.0
                a["bert"] += r.get("bertscore") or 0.0
        except Exception:
            pass
    styles = {}
    for tc, a in agg.items():
        n = max(1, a["n"])
        styles[tc] = {
            "n": a["n"],
            "adm_rate": a["adm"] / n,
            "mean_nli": a["nli"] / n,
            "mean_cos": a["cos"] / n,
            "mean_bert": a["bert"] / n,
        }
    return styles


# ---- geometry: semicircle from 180deg (left) to 0deg (right) ----
def _pt(cx, cy, r, deg):
    a = math.radians(deg)
    return (cx + r * math.cos(a), cy - r * math.sin(a))


INK = "#1a2740"; PAPER = "#f4f6fb"; SLATE = "#6b7896"; GRID = "#c9d2e3"
BLUE = "#2f4b82"; TEAL = "#2e6e68"; AMBER = "#d98a1f"; BRICK = "#b23a3a"; WASH = "#eef1f7"


def build_svg(styles, metric="nli", W=860, title_metric=None):
    key, label, meaning = METRICS[metric]
    cx, cy = W / 2, 430
    R = 300                      # outer radius (metric = 1.0) -- smaller so labels fit
    r0 = 70                      # inner radius (metric = 0.0)
    def radius_for(v):           # map metric value 0..1 -> r0..R
        return r0 + max(0.0, min(1.0, v)) * (R - r0)

    svg = [f'<svg viewBox="0 0 {W} {cy+40}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="IBM Plex Mono, monospace">']

    # concentric dashed tracks at 0,.25,.5,.75,1
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        rr = radius_for(g)
        x1, y1 = _pt(cx, cy, rr, 180); x2, y2 = _pt(cx, cy, rr, 0)
        svg.append(f'<path d="M {x1:.1f} {y1:.1f} A {rr:.1f} {rr:.1f} 0 0 1 {x2:.1f} {y2:.1f}" '
                   f'fill="none" stroke="{GRID}" stroke-dasharray="3 5" stroke-width="1"/>')
        svg.append(f'<text x="{cx+ (0 if g<1 else 4)}" y="{cy-rr-4:.1f}" font-size="9" '
                   f'fill="{SLATE}" text-anchor="middle">{int(g*100)}</text>')

    # order styles by the metric, place dots across the semicircle
    items = sorted(styles.items(), key=lambda kv: kv[1][key])
    names = [k for k, _ in items]
    if not names:
        svg.append(f'<text x="{cx}" y="{cy-120}" text-anchor="middle" fill="{SLATE}" '
                   f'font-size="13">no audit rows found</text></svg>')
        return "".join(svg)
    best = max(items, key=lambda kv: kv[1][key])[0]
    worst = min(items, key=lambda kv: kv[1][key])[0]
    nmax = max(s["n"] for s in styles.values()) or 1

    m = len(items)
    # spread angles within [160, 20] so dots aren't at the very edges
    lo, hi = 162, 18
    for i, (name, s) in enumerate(items):
        ang = lo + (hi - lo) * (i / (m - 1) if m > 1 else 0.5)
        val = s[key]
        rr = radius_for(val)
        x, y = _pt(cx, cy, rr, ang)
        # radial guide from inner to the dot
        xi, yi = _pt(cx, cy, r0, ang)
        svg.append(f'<line x1="{xi:.1f}" y1="{yi:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        rad = 6 + 10 * (s["n"] / nmax)       # dot size by volume
        if name == best:
            fill, stroke = AMBER, INK
        elif name == worst:
            fill, stroke = "#fff", SLATE     # hollow = worst
        else:
            fill, stroke = TEAL, "none"
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{fill}" '
                   f'stroke="{stroke}" stroke-width="1.5"/>')
        if name == best:   # star glyph
            svg.append(f'<text x="{x:.1f}" y="{y+3.5:.1f}" font-size="11" text-anchor="middle" '
                       f'fill="#fff">\u2605</text>')
        # label outside the dot, clamped to canvas so long names never clip
        lx, ly = _pt(cx, cy, rr + rad + 14, ang)
        anchor = "end" if ang > 90 else "start"
        lx = max(70, min(W - 8, lx))   # keep inside [8, W-8] with margin
        svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11" fill="{INK}" '
                   f'text-anchor="{anchor}">{esc(name)}</text>')
        svg.append(f'<text x="{lx:.1f}" y="{ly+12:.1f}" font-size="9.5" fill="{SLATE}" '
                   f'text-anchor="{anchor}">{val:.2f} &middot; n={s["n"]}</text>')

    # base blue arc = overall metric value (gauge at the bottom)
    overall = sum(styles[k][key]*styles[k]["n"] for k in styles) / max(1, sum(styles[k]["n"] for k in styles))
    base_r = r0 - 34
    x1, y1 = _pt(cx, cy, base_r, 180)
    xov, yov = _pt(cx, cy, base_r, 180 - overall*180)
    svg.append(f'<path d="M {x1:.1f} {y1:.1f} A {base_r:.1f} {base_r:.1f} 0 0 1 '
               f'{_pt(cx,cy,base_r,0)[0]:.1f} {_pt(cx,cy,base_r,0)[1]:.1f}" fill="none" '
               f'stroke="{WASH}" stroke-width="10"/>')
    large = 1 if overall > 0.5 else 0
    svg.append(f'<path d="M {x1:.1f} {y1:.1f} A {base_r:.1f} {base_r:.1f} 0 {large} 1 '
               f'{xov:.1f} {yov:.1f}" fill="none" stroke="{BLUE}" stroke-width="10" '
               f'stroke-linecap="round"/>')
    svg.append(f'<text x="{cx:.1f}" y="{cy-6:.1f}" font-size="22" font-weight="700" '
               f'text-anchor="middle" fill="{INK}">{overall*100:.0f}%</text>')
    svg.append(f'<text x="{cx:.1f}" y="{cy+12:.1f}" font-size="10" text-anchor="middle" '
               f'fill="{SLATE}">overall {esc(label)}</text>')
    svg.append("</svg>")
    return "".join(svg)


CSS = f"""
.sa-wrap{{font-family:'Space Grotesk',system-ui,sans-serif;color:{INK};background:{PAPER};
border:1px solid {GRID};border-radius:12px;padding:20px 22px}}
.sa-wrap h3{{margin:0 0 2px;font-size:16px;font-weight:700}}
.sa-sub{{color:{SLATE};font-size:12.5px;margin:0 0 6px;max-width:72ch;line-height:1.5}}
.sa-legend{{color:{SLATE};font-size:11px;font-family:'IBM Plex Mono',monospace;margin-top:8px}}
.sa-legend b{{color:{AMBER}}} .sa-legend i{{color:{TEAL};font-style:normal}}
.sa-note{{display:inline-block;margin-top:10px;padding:3px 10px;border:1px solid {GRID};
border-radius:16px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:{SLATE}}}
"""


def render_fragment(styles, metric="nli", section_no="01\u00b7f"):
    key, label, meaning = METRICS[metric]
    svg = build_svg(styles, metric)
    return (f'<h2><span class="step-n">{section_no}</span> Paraphrase styles &mdash; which preserve intent</h2>'
            f'<div class="sa-wrap"><style>{CSS}</style>'
            f'<h3>How each reword style holds up at the intent gate</h3>'
            f'<p class="sa-sub">One dot per paraphrase style. Radius = <b>{esc(label)}</b> '
            f'({esc(meaning)}); styles on the outer arcs preserve intent best. Dot size = how '
            f'many variants used that style. This is gate quality by style (NLI/cosine are '
            f'deterministic, no judge needed), not model vulnerability &mdash; the latter needs '
            f'per-variant bypass logging (a re-run).</p>'
            f'{svg}'
            f'<div class="sa-legend">radius = {esc(label)} &middot; '
            f'<b>&#9733; best style</b> &middot; <i>filled = mid</i> &middot; hollow = worst &middot; '
            f'blue base arc = overall</div>'
            f'<div class="sa-note">OPTION A &middot; gate quality by style &middot; not model vulnerability</div>'
            f'</div>')


def render_page(styles, metric="nli"):
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Paraphrase style arcs</title>'
            f'<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono&display=swap" rel="stylesheet">'
            f'<style>body{{margin:0;background:{PAPER};padding:24px;max-width:900px;margin:0 auto}}'
            f'h2{{font-family:"Space Grotesk";font-weight:600;font-size:13px;letter-spacing:.16em;'
            f'text-transform:uppercase;color:{SLATE};display:flex;gap:12px;align-items:center}}'
            f'h2::after{{content:"";flex:1;height:1px;background:{GRID}}}'
            f'.step-n{{font-family:"IBM Plex Mono";color:{AMBER}}}</style></head><body>'
            f'{render_fragment(styles, metric)}</body></html>')


def main():
    ap = argparse.ArgumentParser(description="Per-style paraphrase gate-quality arc diagram")
    ap.add_argument("--manifests", nargs="+", required=True,
                    help="glob(s) for recycle manifest_*.jsonl files")
    ap.add_argument("--metric", choices=list(METRICS), default="nli")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    styles = aggregate(args.manifests)
    print(f"styles found: {list(styles)}")
    for k, s in sorted(styles.items(), key=lambda kv: -kv[1][METRICS[args.metric][0]]):
        print(f"  {k:16} n={s['n']:5d}  nli={s['mean_nli']:.3f}  adm={s['adm_rate']:.3f}  cos={s['mean_cos']:.3f}")
    open(args.out, "w", encoding="utf-8").write(render_page(styles, args.metric))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
