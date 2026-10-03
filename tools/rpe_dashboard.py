#!/usr/bin/env python3
"""
tools/rpe_dashboard.py

Render the independent-probe (RPE) convergent-validity result as a self-contained
HTML panel matching RecycleBench's dashboard aesthetic (navy ink on cool paper,
Space Grotesk + IBM Plex, hand-built inline SVG, no JS deps, emailable).

Reads the rollup written by tools/run_independent_probe.py:
    <out>/rpe_agreement.json   ->  {bench: {rsr_para, rsr_rpe, cohen_kappa, ...}}

Two uses:
    1. Standalone panel:
         python tools/rpe_dashboard.py \
             --agreement results/runs/.../rpe/rpe_agreement.json \
             --out       results/runs/.../rpe/rpe_panel.html
    2. Fragment for embedding: pass --fragment to emit just the <section>...</section>
       (no <html> wrapper), which make_dashboard.py can inject into
       dashboard_all3_with_filter_view.html. See EMBEDDING note at the bottom.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path


# --- palette lifted from make_dashboard.py's measurement-instrument look ---
INK = "#1a2740"
PAPER = "#f4f6fb"
AMBER = "#d98a1f"       # the P- "fix" hero colour
TEAL = "#2e6e68"
MUTE = "#6b7896"
LINE = "#c9d2e3"
GOOD = "#2e6e68"
BAD = "#b23a3a"


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _pct(x):
    return "n/a" if x is None else f"{x*100:.1f}%"


def _ci(c):
    if not c or c[0] is None:
        return ""
    return f"[{c[0]*100:.0f}\u2013{c[1]*100:.0f}]"


def _kappa_word(k):
    if k is None:
        return ("n/a", MUTE)
    if k >= 0.8:  return (f"{k:.2f} (almost perfect)", GOOD)
    if k >= 0.6:  return (f"{k:.2f} (substantial)", GOOD)
    if k >= 0.4:  return (f"{k:.2f} (moderate)", AMBER)
    if k >= 0.2:  return (f"{k:.2f} (fair)", AMBER)
    return (f"{k:.2f} (poor)", BAD)


def _bar_svg(rsr_para, rsr_rpe, width=260, h=54):
    """Two horizontal bars: RSR_paraphrase vs RSR_rpe, same axis."""
    def bar(y, val, colour, label):
        w = 0 if val is None else int(val * (width - 70))
        txt = _pct(val)
        return (
            f'<text x="0" y="{y+11}" font-size="10" fill="{MUTE}" '
            f'font-family="IBM Plex Mono, monospace">{label}</text>'
            f'<rect x="60" y="{y}" width="{width-70}" height="14" rx="3" '
            f'fill="none" stroke="{LINE}"/>'
            f'<rect x="60" y="{y}" width="{w}" height="14" rx="3" fill="{colour}"/>'
            f'<text x="{width-6}" y="{y+11}" font-size="10" fill="{INK}" '
            f'text-anchor="end" font-family="IBM Plex Mono, monospace">{txt}</text>'
        )
    return (
        f'<svg width="{width}" height="{h}" viewBox="0 0 {width} {h}">'
        + bar(4, rsr_para, TEAL, "para")
        + bar(30, rsr_rpe, AMBER, "rpe")
        + "</svg>"
    )


def _agreement_glyph(both_rec, both_held, rpe_only, para_only, size=54):
    """A tiny 2x2 confusion glyph: agreement on diagonal, divergence off-diagonal."""
    total = max(1, both_rec + both_held + rpe_only + para_only)
    def cell(x, y, n, colour):
        op = 0.15 + 0.85 * (n / total)
        return (f'<rect x="{x}" y="{y}" width="24" height="24" rx="3" '
                f'fill="{colour}" fill-opacity="{op:.2f}" stroke="{LINE}"/>'
                f'<text x="{x+12}" y="{y+16}" font-size="11" text-anchor="middle" '
                f'fill="{INK}" font-family="IBM Plex Mono, monospace">{n}</text>')
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        + cell(2, 2, both_rec, GOOD)      # both recovered
        + cell(28, 2, para_only, BAD)     # para only
        + cell(2, 28, rpe_only, BAD)      # rpe only
        + cell(28, 28, both_held, GOOD)   # both held
        + "</svg>"
    )


def _card(bench, d) -> str:
    kappa_txt, kappa_col = _kappa_word(d.get("cohen_kappa"))
    bars = _bar_svg(d.get("rsr_para"), d.get("rsr_rpe"))
    glyph = _agreement_glyph(d.get("both_recovered", 0), d.get("both_held", 0),
                             d.get("rpe_only", 0), d.get("para_only", 0))
    return f"""
    <div class="rpe-card">
      <div class="rpe-card-head">{esc(bench)}</div>
      <div class="rpe-card-body">
        <div class="rpe-bars">{bars}</div>
        <div class="rpe-agree">
          <div class="rpe-glyph">{glyph}</div>
          <div class="rpe-stats">
            <div><span class="k">agreement</span> {_pct(d.get('agreement_rate'))}</div>
            <div><span class="k">kappa</span> <span style="color:{kappa_col}">{esc(kappa_txt)}</span></div>
            <div><span class="k">n (both)</span> {esc(d.get('n_both_resolved'))}</div>
            <div><span class="k">rpe-only / para-only</span> {esc(d.get('rpe_only'))} / {esc(d.get('para_only'))}</div>
            <div><span class="k">unresolved p/r</span> {esc(d.get('para_unresolved'))} / {esc(d.get('rpe_unresolved'))}</div>
          </div>
        </div>
      </div>
    </div>"""


STYLE = f"""
<style>
  .rpe-section {{ font-family:'Space Grotesk',system-ui,sans-serif; color:{INK};
                 background:{PAPER}; padding:22px; border-radius:10px;
                 border:1px solid {LINE}; margin:22px 0; }}
  .rpe-section h2 {{ font-size:18px; margin:0 0 4px; letter-spacing:.2px; }}
  .rpe-section .sub {{ color:{MUTE}; font-size:12.5px; margin:0 0 16px;
                       max-width:70ch; line-height:1.5; }}
  .rpe-grid {{ display:flex; flex-wrap:wrap; gap:16px; }}
  .rpe-card {{ background:#fff; border:1px solid {LINE}; border-radius:8px;
               min-width:300px; flex:1; }}
  .rpe-card-head {{ background:{INK}; color:#fff; padding:6px 12px;
                    font-size:12px; letter-spacing:.6px; text-transform:uppercase;
                    border-radius:8px 8px 0 0; font-family:'IBM Plex Mono',monospace; }}
  .rpe-card-body {{ padding:12px; display:flex; gap:16px; align-items:flex-start;
                    flex-wrap:wrap; }}
  .rpe-agree {{ display:flex; gap:10px; align-items:flex-start; }}
  .rpe-stats {{ font-size:11.5px; line-height:1.7; }}
  .rpe-stats .k {{ color:{MUTE}; font-family:'IBM Plex Mono',monospace;
                   display:inline-block; min-width:120px; }}
  .rpe-legend {{ color:{MUTE}; font-size:11px; margin-top:14px;
                 font-family:'IBM Plex Mono',monospace; }}
  .rpe-legend b {{ color:{TEAL}; }} .rpe-legend i {{ color:{AMBER}; font-style:normal; }}
</style>"""


def render_section(rollup: dict) -> str:
    cards = "\n".join(_card(b, d) for b, d in rollup.items())
    return f"""{STYLE}
<section class="rpe-section" id="independent-probe">
  <h2>Independent-probe validation &mdash; does an inversion method reproduce the paraphrase result?</h2>
  <p class="sub">Convergent-validity check (Possibility&nbsp;1). For each apparent fix in
  P&minus;, a mechanically independent generator&mdash;Reverse&nbsp;Prompt&nbsp;Engineering
  (RPE, an MVP inverter, to be replaced by output2prompt)&mdash;recovers candidate prompts
  from the <b>old model's</b> harmful response; the same intent gate and judge then decide
  recovery on the new model. High agreement / &kappa; means the recycling effect is not an
  artifact of the paraphraser. Divergences where RPE recovers a fix the paraphraser missed
  tighten the lower bound.</p>
  <div class="rpe-grid">
    {cards}
  </div>
  <div class="rpe-legend">
    bars: <b>para</b> = RSR&#8320; paraphrase arm &nbsp;&middot;&nbsp; <i>rpe</i> = RSR RPE arm &nbsp;|&nbsp;
    glyph: diagonal = agreement (both recovered / both held), off-diagonal = divergence (rpe-only / para-only)
  </div>
</section>"""


def render_page(rollup: dict) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RecycleBench &mdash; Independent-probe validation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=IBM+Plex+Mono&display=swap" rel="stylesheet">
</head><body style="margin:0;background:{PAPER};padding:24px;">
{render_section(rollup)}
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Render the RPE convergent-validity panel")
    ap.add_argument("--agreement", required=True, type=Path,
                    help="rpe_agreement.json from run_independent_probe.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fragment", action="store_true",
                    help="emit only the <section> (for embedding), not a full page")
    args = ap.parse_args()

    rollup = json.loads(args.agreement.read_text())
    htmltext = render_section(rollup) if args.fragment else render_page(rollup)
    args.out.write_text(htmltext, encoding="utf-8")
    print(f"wrote {args.out}  ({len(rollup)} benchmark(s))")


if __name__ == "__main__":
    main()


# ===========================================================================
# EMBEDDING into dashboard_all3_with_filter_view.html
# ---------------------------------------------------------------------------
# The dashboards are GENERATED by make_dashboard.py, so hand-editing the .html
# gets overwritten on the next regen. Two clean options:
#
# (A) Standalone (zero coupling): open rpe_panel.html on its own, or link to it
#     from the main dashboard with an <a>. Nothing in make_dashboard.py changes.
#
# (B) Injected section (survives regen): in make_dashboard.py, where the final
#     page string is assembled, add:
#
#         from importlib import import_module
#         rpe = import_module("tools.rpe_dashboard")   # or a direct import
#         agreement = Path(run_dir) / "rpe" / "rpe_agreement.json"
#         rpe_html = rpe.render_section(json.loads(agreement.read_text())) \
#                    if agreement.exists() else ""
#         # then splice rpe_html into the body, e.g. right before the closing </body>
#         page = page.replace("</body>", rpe_html + "</body>")
#
#     render_section() is import-safe (no CLI side effects) precisely so it can be
#     called this way. Paste the ~10 lines around make_dashboard.py's final
#     assembly and I'll give you the exact splice point.
# ===========================================================================
