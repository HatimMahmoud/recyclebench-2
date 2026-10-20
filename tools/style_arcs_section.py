"""
style_arcs_section.py -- Section 01.f of the RecycleBench dashboard:
"Paraphrase styles -- which preserve intent" (concentric semicircular arcs,
Option A = gate quality by style).

One dot per paraphrase style. Radius = mean NLI entailment (intent preserved),
so styles on the OUTER arcs survive the intent gate best. Dot size = how many
variants used that style. This is gate quality (NLI is deterministic, no judge),
not model vulnerability -- the per-style BYPASS view (Option B) needs per-variant
verdict logging, which is the heavy-run work.

Self-contained: it aggregates straight from the in-memory `admitted` dict that
make_dashboard.render() already holds ({bid: [variant, ...]}), so it needs no
manifest re-read and no change to render()'s signature. It also accepts manifest
path(s)/globs for standalone use. Field-name tolerant, and it degrades to a
count-only arc (with an honest note) when a run didn't persist NLI.

Public API (mirrors positioning_section.py / rpe_log.py):
    aggregate(source)            -> dict   # source = admitted dict | [variant dicts] | [manifest globs]
    render_fragment(agg=None, source=None) -> str   # <style> + <section class="sa-root">..</section>
    render_page(agg=None, source=None)     -> str   # full standalone HTML document
    inject(html_doc, source)     -> str    # splice the fragment before the final </body>

Wire into make_dashboard.render(), right where `doc` is finished (before the
positioning block, so order reads 01.f then 02):

    from style_arcs_section import inject as add_style_arcs
    doc = add_style_arcs(doc, admitted)          # admitted is already in scope

Run directly to preview from a manifest:
    python3 style_arcs_section.py run/paraphrase_manifest.jsonl   # -> style_arcs_board.html
"""
from __future__ import annotations

import glob as _glob
import html as _html
import json as _json
import math as _math
from pathlib import Path

__all__ = ["aggregate", "render_fragment", "render_page", "inject"]

# ---- palette (matches the results dashboard) ----
_PAPER = "#FAFAFA"; _PANEL = "#FFFFFF"; _INK = "#14202E"; _SLATE = "#6B7A88"
_GRID = "#E4E2DB"; _HELD = "#107C6E"; _FIX = "#E3A008"; _REGRESS = "#B5522E"

_STYLE_KEYS = ("transform_class", "tc", "transform", "style", "tclass", "family")
_NLI_KEYS = ("nli_entail", "nli", "nli_entailment", "entailment")


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------
def _style_of(d):
    if not isinstance(d, dict):
        return None
    for k in _STYLE_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _nli_of(d):
    if not isinstance(d, dict):
        return None
    for k in _NLI_KEYS:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    sc = d.get("scores")
    if isinstance(sc, dict):
        for k in _NLI_KEYS:
            v = sc.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _variants_in_obj(obj):
    """Yield the variant dicts carried by one manifest JSON object, without
    double-counting (prefer an admitted line's `variants`, else `details`,
    else the object itself if it looks like a variant)."""
    if not isinstance(obj, dict):
        return
    if isinstance(obj.get("variants"), list):
        for v in obj["variants"]:
            if isinstance(v, dict):
                yield v
    elif isinstance(obj.get("details"), list):
        for v in obj["details"]:
            if isinstance(v, dict):
                yield v
    elif _style_of(obj) is not None:
        yield obj


def _iter_variants(source):
    """Normalise any accepted source into a stream of variant dicts."""
    if isinstance(source, dict):                      # admitted: {bid: [variant,...]}
        for vs in source.values():
            if isinstance(vs, list):
                for v in vs:
                    if isinstance(v, dict):
                        yield v
        return
    if isinstance(source, (list, tuple)):
        if source and all(isinstance(x, str) for x in source):   # manifest globs/paths
            seen_files = []
            for pat in source:
                seen_files += sorted(_glob.glob(pat)) or ([pat] if Path(pat).exists() else [])
            for fp in seen_files:
                try:
                    for line in Path(fp).read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = _json.loads(line)
                        except Exception:
                            continue
                        yield from _variants_in_obj(obj)
                except Exception:
                    continue
            return
        for v in source:                             # already a list of variant dicts
            if isinstance(v, dict):
                yield v


def aggregate(source) -> dict:
    """source = admitted dict | list of variant dicts | list of manifest globs.
    Returns {"styles":[{name,n,nli}], "overall_nli", "total_n", "has_nli"}."""
    per: dict[str, dict] = {}
    for v in _iter_variants(source):
        name = _style_of(v)
        if name is None:
            continue
        slot = per.setdefault(name, {"n": 0, "nlis": []})
        slot["n"] += 1
        nli = _nli_of(v)
        if nli is not None:
            slot["nlis"].append(nli)
    styles, all_nli = [], []
    for name, slot in per.items():
        nlis = slot["nlis"]
        mean = sum(nlis) / len(nlis) if nlis else None
        styles.append({"name": name, "n": slot["n"], "nli": mean})
        all_nli += nlis
    has_nli = len(all_nli) > 0
    # order best-first when NLI exists, else by volume
    styles.sort(key=lambda s: (s["nli"] if s["nli"] is not None else -1, s["n"]), reverse=True)
    return {
        "styles": styles,
        "overall_nli": (sum(all_nli) / len(all_nli)) if has_nli else None,
        "total_n": sum(s["n"] for s in styles),
        "has_nli": has_nli,
    }


# ----------------------------------------------------------------------
# svg
# ----------------------------------------------------------------------
def _pt(cx, base_y, r, deg):
    a = _math.radians(deg)
    return (cx + r * _math.cos(a), base_y - r * _math.sin(a))


def build_svg(agg: dict) -> str:
    styles = agg["styles"]
    has_nli = agg["has_nli"]
    W, H = 760, 400
    cx, base_y = W / 2, 330.0
    r_min, r_max = 70.0, 300.0
    # radial scale: NLI domain -> radius. When NLI is absent, everything sits mid-arc.
    if has_nli:
        vals = [s["nli"] for s in styles if s["nli"] is not None]
        d0 = min(0.30, min(vals)) if vals else 0.30
        d1 = 1.0
    else:
        d0, d1 = 0.0, 1.0

    def R(nli):
        if nli is None or d1 <= d0:
            return (r_min + r_max) / 2
        t = max(0.0, min(1.0, (nli - d0) / (d1 - d0)))
        return r_min + t * (r_max - r_min)

    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="IBM Plex Mono, monospace" role="img" '
           f'aria-label="Paraphrase styles by mean NLI entailment">']

    # guide arcs
    grid = [0.5, 0.7, 0.9, 1.0] if has_nli else []
    for g in grid:
        if g < d0:
            continue
        r = R(g)
        x0, y0 = _pt(cx, base_y, r, 180)
        x1, y1 = _pt(cx, base_y, r, 0)
        svg.append(f'<path d="M {x0:.1f} {y0:.1f} A {r:.1f} {r:.1f} 0 0 1 {x1:.1f} {y1:.1f}" '
                   f'fill="none" stroke="{_GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{cx - r - 6:.1f}" y="{base_y + 4:.1f}" text-anchor="end" '
                   f'font-size="10" fill="{_SLATE}">{g:.1f}</text>')

    # overall-mean arc
    if has_nli and agg["overall_nli"] is not None:
        r = R(agg["overall_nli"])
        x0, y0 = _pt(cx, base_y, r, 180)
        x1, y1 = _pt(cx, base_y, r, 0)
        svg.append(f'<path d="M {x0:.1f} {y0:.1f} A {r:.1f} {r:.1f} 0 0 1 {x1:.1f} {y1:.1f}" '
                   f'fill="none" stroke="{_HELD}" stroke-width="1.5" stroke-dasharray="3 4" opacity="0.6"/>')
        mx, my = _pt(cx, base_y, r, 90)
        svg.append(f'<text x="{mx:.1f}" y="{my - 8:.1f}" text-anchor="middle" font-size="10" '
                   f'fill="{_HELD}">overall {agg["overall_nli"]:.2f}</text>')

    # baseline
    svg.append(f'<line x1="{cx - r_max - 10:.1f}" y1="{base_y:.1f}" x2="{cx + r_max + 10:.1f}" '
               f'y2="{base_y:.1f}" stroke="{_GRID}" stroke-width="1"/>')
    svg.append(f'<circle cx="{cx:.1f}" cy="{base_y:.1f}" r="3" fill="{_SLATE}"/>')

    # style dots, fanned across the arc, best-first from the left
    n = len(styles)
    a_lo, a_hi = 158.0, 22.0
    best = max((s["nli"] for s in styles if s["nli"] is not None), default=None)
    worst = min((s["nli"] for s in styles if s["nli"] is not None), default=None)
    ns = [s["n"] for s in styles] or [1]
    n_max = max(ns)
    for i, s in enumerate(styles):
        deg = a_lo if n == 1 else a_lo + (a_hi - a_lo) * i / (n - 1)
        r = R(s["nli"])
        x, y = _pt(cx, base_y, r, deg)
        dot = 5 + 10 * _math.sqrt(s["n"] / n_max)
        is_best = has_nli and s["nli"] is not None and abs(s["nli"] - best) < 1e-9
        is_worst = has_nli and s["nli"] is not None and abs(s["nli"] - worst) < 1e-9 and n > 1
        # spoke
        sx, sy = _pt(cx, base_y, r_min - 8, deg)
        svg.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
                   f'stroke="{_GRID}" stroke-width="1" opacity="0.7"/>')
        if is_best:
            svg.append(_star(x, y, dot + 3, _FIX))
        elif is_worst:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{dot:.1f}" fill="{_PANEL}" '
                       f'stroke="{_REGRESS}" stroke-width="2"/>')
        else:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{dot:.1f}" fill="{_HELD}" '
                       f'fill-opacity="0.85"/>')
        # label, offset outward along the spoke
        lx, ly = _pt(cx, base_y, r + dot + 8, deg)
        anchor = "end" if deg > 90 else "start"
        nli_txt = f'{s["nli"]:.2f}' if s["nli"] is not None else "--"
        svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="11" '
                   f'fill="{_INK}" font-weight="600">{_html.escape(s["name"])}</text>')
        svg.append(f'<text x="{lx:.1f}" y="{ly + 13:.1f}" text-anchor="{anchor}" font-size="9.5" '
                   f'fill="{_SLATE}">NLI {nli_txt} · n={s["n"]}</text>')

    svg.append("</svg>")
    return "".join(svg)


def _star(cx, cy, r, fill):
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.45
        a = _math.radians(-90 + i * 36)
        pts.append(f'{cx + rr * _math.cos(a):.1f},{cy + rr * _math.sin(a):.1f}')
    return f'<polygon points="{" ".join(pts)}" fill="{fill}"/>'


# ----------------------------------------------------------------------
# fragment / page / inject
# ----------------------------------------------------------------------
_CSS = """<style>
.sa-root{--sa-paper:#FAFAFA;--sa-panel:#FFFFFF;--sa-ink:#14202E;--sa-slate:#6B7A88;
  --sa-grid:#E4E2DB;--sa-held:#107C6E;--sa-fix:#E3A008;--sa-regress:#B5522E;
  background:var(--sa-paper);color:var(--sa-ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.55}
.sa-root *{box-sizing:border-box}
.sa-wrap{max-width:1040px;margin:0 auto;padding:10px 28px 40px}
.sa-root h2{font-family:"Space Grotesk",sans-serif;font-weight:600;font-size:21px;
  letter-spacing:-.01em;margin:40px 0 4px;padding-top:26px;border-top:1px solid var(--sa-grid)}
.sa-root h2 .n{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--sa-slate);
  font-weight:500;margin-right:10px;letter-spacing:.04em}
.sa-sub{color:var(--sa-slate);font-size:14px;margin:4px 0 14px;max-width:74ch}
.sa-sub b{color:var(--sa-ink);font-weight:600}
.sa-card{background:var(--sa-panel);border:1px solid var(--sa-grid);border-radius:12px;padding:18px 20px 10px}
.sa-legend{display:flex;flex-wrap:wrap;gap:16px;margin:4px 0 2px;font-family:"IBM Plex Mono",monospace;
  font-size:11px;color:var(--sa-slate)}
.sa-legend span{display:inline-flex;align-items:center;gap:6px}
.sa-sw{width:12px;height:12px;border-radius:50%;display:inline-block}
.sa-note{display:inline-block;margin:6px 0 2px;padding:3px 10px;border:1px solid var(--sa-grid);
  border-radius:16px;font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--sa-slate)}
</style>"""


def _body(agg: dict) -> str:
    svg = build_svg(agg)
    if agg["has_nli"]:
        sub = ('One dot per paraphrase <b>style</b>. Radius = <b>mean NLI entailment</b> '
               '(intent preserved) &mdash; styles on the <b>outer</b> arcs clear the intent gate best. '
               'Dot size = how many variants used that style. This is gate quality by style '
               '(NLI is deterministic, no judge), not model vulnerability.')
        note = f'gate quality (Option A) · {agg["total_n"]} gated variants across {len(agg["styles"])} styles'
    else:
        sub = ('One dot per paraphrase <b>style</b>; dot size = how many variants used it. '
               'This run did not persist per-variant <b>NLI</b>, so radius is uniform (counts only). '
               'Re-run with gate scores persisted to get the mean-NLI radius.')
        note = f'counts only · NLI not in this run · {agg["total_n"]} variants across {len(agg["styles"])} styles'
    legend = (
        '<div class="sa-legend">'
        f'<span><i class="sa-sw" style="background:{_HELD}"></i>style (dot = variant count)</span>'
        f'<span><svg width="14" height="14" viewBox="0 0 14 14">{_star(7,7,6,_FIX)}</svg>best intent-preserving</span>'
        f'<span><i class="sa-sw" style="background:{_PANEL};border:2px solid {_REGRESS}"></i>weakest</span>'
        f'<span><i class="sa-sw" style="background:{_HELD};opacity:.6;border-radius:2px"></i>overall mean arc</span>'
        '</div>')
    return (
        '<section class="sa-root"><div class="sa-wrap">'
        '<h2><span class="n">05</span>Paraphrase styles — which preserve intent</h2>'
        f'<p class="sa-sub">{sub}</p>'
        f'<div class="sa-card">{svg}{legend}</div>'
        f'<span class="sa-note">{_html.escape(note)}</span>'
        '</div></section>'
    )


def _resolve(agg, source):
    if agg is None:
        agg = aggregate(source if source is not None else {})
    return agg


def render_fragment(agg: dict | None = None, source=None) -> str:
    """Scoped <style> + <section class="sa-root">..</section>; safe to inline."""
    return _CSS + "\n" + _body(_resolve(agg, source))


_HEAD = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
    '<title>Paraphrase style arcs</title>\n'
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700'
    '&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" '
    'rel="stylesheet">\n'
)


def render_page(agg: dict | None = None, source=None) -> str:
    """Full standalone HTML document."""
    return (_HEAD + _CSS + "</head>\n<body>\n"
            + _body(_resolve(agg, source)) + "\n</body></html>\n")


def inject(html_doc: str, source) -> str:
    """Splice the arc section into an existing dashboard HTML string, just
    before the final </body>. Self-contained (own .sa-wrap column), so it need
    not sit inside the host's .wrap."""
    agg = aggregate(source)
    if not agg.get("styles"):
        return html_doc                       # nothing to draw -> leave the doc untouched
    frag = render_fragment(agg)
    idx = html_doc.rfind("</body>")
    if idx == -1:
        return html_doc + "\n" + frag
    return html_doc[:idx] + frag + "\n" + html_doc[idx:]


if __name__ == "__main__":
    import sys
    globs = sys.argv[1:] or []
    out = "style_arcs_board.html"
    agg = aggregate(globs)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render_page(agg))
    print("wrote", out, "|", agg["total_n"], "variants across", len(agg["styles"]), "styles",
          "| NLI:" , "yes" if agg["has_nli"] else "no (counts only)")