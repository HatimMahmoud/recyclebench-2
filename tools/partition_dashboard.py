"""Standalone PARTITION dashboard for a screening run (no recycle needed).

Visualises the full 2x2 partition -- P^-, negative-flip, both-refused, both-complied --
across every benchmark, straight from the `responses/<bench>_<cell>.txt` dumps that
inspect_cache.py writes. Built for a run where recycling has NOT been done yet, so there is
no RSR and no before->after paraphrase cards -- just the screening outcome, richly shown.

Usage:
  python3 partition_dashboard.py \
      --responses results/runs/<run>/responses \
      --old llama2-13b --new llama3-8b \
      --benchmarks harmbench advbench strongreject \
      --out results/runs/<run>/partition_dashboard.html

It parses each dump into (prompt, old_response, new_response) records. The inspector's format
is blocks separated by a rule line, each block containing the prompt then each model's reply
under a header line. The parser is defensive: if a file is header-only (no records) it counts 0.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

CELLS = ["only-pminus", "only-negflip", "refusals-only", "compliances-only"]
CELL_LABEL = {
    "only-pminus": "P⁻ · old answered, new refused",
    "only-negflip": "negative flip · old refused, new answered",
    "refusals-only": "old refused",
    "compliances-only": "old complied",
}
CELL_SHORT = {"only-pminus": "P⁻ (fixes)", "only-negflip": "neg-flip (regressions)",
              "refusals-only": "old refused", "compliances-only": "old complied"}


def esc(s: str) -> str:
    return html.escape(s or "")


def parse_dump(path: Path):
    """Return a list of {prompt, old, new} records from an inspect_cache dump file.

    Defensive parser: splits on long rule lines, then within each block pulls the prompt and
    the two model responses by looking for the model-name header markers the inspector emits.
    Falls back to counting '[bench:bid]' block markers if the fine structure differs.
    """
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    # blocks are separated by a run of '=' or '-' (the inspector prints a rule between behaviours)
    blocks = re.split(r"\n[=\-]{20,}\n", text)
    recs = []
    for blk in blocks:
        blk = blk.strip()
        if not blk or blk.lower().startswith(("inspecting", "showing", "filter", "cache")):
            continue
        # a real block has a [something:bid] marker or a PROMPT: line
        if "PROMPT" not in blk and "prompt" not in blk and "[" not in blk:
            continue
        prompt = _grab(blk, [r"PROMPT[: ]+(.*?)(?:\n\n|\n---|\Z)",
                             r"prompt[: ]+(.*?)(?:\n\n|\n---|\Z)"])
        # model responses: look for "--- <name> ---" or "<name> replied/answered/refused:" markers
        old = _grab(blk, [r"---[^\n]*\n(.*?)(?:\n---|\Z)"])  # first response block
        new = _grab_second(blk)
        if prompt or old or new:
            recs.append({"prompt": prompt or "(prompt not parsed)",
                         "old": old or "", "new": new or ""})
    return recs


def _grab(blk, patterns):
    for p in patterns:
        m = re.search(p, blk, re.S)
        if m:
            return m.group(1).strip()
    return ""


def _grab_second(blk):
    # the SECOND '--- ... ---' delimited section (the new model's reply)
    parts = re.split(r"\n---[^\n]*---?\n", blk)
    if len(parts) >= 3:
        return parts[2].strip()[:1200]
    return ""


def count_records(path: Path) -> int:
    """Fast count of behaviours in a dump without full parse (block markers)."""
    if not path.exists():
        return 0
    t = path.read_text(errors="replace")
    n = len(re.findall(r"\[[^\]\n]+:[^\]\n]+\]", t))  # [bench:bid] markers
    if n == 0:
        n = len(parse_dump(path))
    return n


CSS = """
:root{--navy:#14202E;--paper:#F5F4F0;--ink:#1b2430;--muted:#5b6675;--line:#dcd9d2;
--amber:#E0A100;--red:#C6413A;--teal:#2E8B84;--card:#ffffff;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:40px 28px 80px}
.kicker{font-family:"Space Grotesk","IBM Plex Sans",sans-serif;letter-spacing:.18em;
text-transform:uppercase;font-size:11.5px;color:var(--muted)}
h1{font-family:"Space Grotesk","IBM Plex Sans",sans-serif;font-size:34px;line-height:1.12;
margin:.35em 0 .5em;max-width:20ch}
.sub{color:var(--muted);max-width:60ch;font-size:15px}
.meta{font-size:12.5px;color:var(--muted);margin-top:14px}
.meta b{color:var(--ink)}
h2{font-family:"Space Grotesk","IBM Plex Sans",sans-serif;font-size:19px;margin:44px 0 14px;
border-top:1px solid var(--line);padding-top:22px}
.step-n{color:var(--muted);font-size:14px;margin-right:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card .n{font-family:"Space Grotesk",sans-serif;font-size:30px;font-weight:600}
.card .l{font-size:12.5px;color:var(--muted);margin-top:2px}
.bars{display:flex;flex-direction:column;gap:14px;margin-top:6px}
.brow{}
.brow .top{display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px}
.brow .bn{font-weight:600}
.track{height:26px;background:#ece9e2;border-radius:6px;overflow:hidden;display:flex}
.seg{height:100%;display:flex;align-items:center;padding:0 7px;font-size:11px;color:#fff;white-space:nowrap}
.seg.pm{background:var(--amber)} .seg.nf{background:var(--red)}
.seg.ref{background:var(--teal)} .seg.cmp{background:#7a8699}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:12px}
.legend i{width:11px;height:11px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:-1px}
.callout{background:#fbf3d9;border:1px solid #ecd98f;border-radius:10px;padding:14px 16px;
font-size:13.5px;margin:18px 0}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 12px}
.tab{padding:6px 12px;border:1px solid var(--line);border-radius:20px;font-size:12.5px;
cursor:pointer;background:#fff;color:var(--muted)}
.tab.active{background:var(--navy);color:#fff;border-color:var(--navy)}
details{background:#fff;border:1px solid var(--line);border-radius:10px;margin:8px 0;padding:2px 14px}
summary{cursor:pointer;padding:10px 0;font-size:13.5px;list-style:none}
summary::-webkit-details-marker{display:none}
.pill{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;background:#eef;margin-left:8px}
.pill.nf{background:#f8e0de;color:#8f2b25} .pill.pm{background:#faf0cf;color:#8a6a00}
.sens{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:#b08900;
background:#fbf3d9;border-radius:4px;padding:1px 6px;margin-left:8px}
.tl{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
display:block;margin:8px 0 2px}
.txt{background:#faf9f6;border:1px solid var(--line);border-radius:7px;padding:9px 11px;
font-size:12.5px;white-space:pre-wrap;word-break:break-word;max-height:220px;overflow:auto}
.hide{display:none}
.foot{margin-top:40px;font-size:11.5px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px}
"""

JS = """
function showBench(b){
 document.querySelectorAll('[data-bench]').forEach(function(e){e.classList.add('hide')});
 document.querySelectorAll('[data-bench="'+b+'"]').forEach(function(e){e.classList.remove('hide')});
 document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active')});
 document.getElementById('tab-'+b).classList.add('active');
}
"""


def render(responses_dir, old, new, benchmarks, out):
    rdir = Path(responses_dir)
    # counts per (bench, cell)
    counts = {}
    for b in benchmarks:
        for c in CELLS:
            counts[(b, c)] = count_records(rdir / f"{b}_{c}.txt")

    # totals
    tot = {c: sum(counts[(b, c)] for b in benchmarks) for c in CELLS}
    total_pm = tot["only-pminus"]
    total_nf = tot["only-negflip"]

    # ---- top cards ----
    cards = "".join(
        f'<div class="card"><div class="n">{tot[c]}</div><div class="l">{CELL_SHORT[c]}</div></div>'
        for c in CELLS)

    # ---- per-benchmark stacked bars ----
    bars = ""
    for b in benchmarks:
        vals = {c: counts[(b, c)] for c in CELLS}
        total = max(sum(vals.values()), 1)
        seg = ""
        for c, cls in [("only-pminus", "pm"), ("only-negflip", "nf"),
                       ("refusals-only", "ref"), ("compliances-only", "cmp")]:
            w = vals[c] / total * 100
            if w <= 0:
                continue
            lbl = f"{CELL_SHORT[c]} {vals[c]}" if w > 12 else str(vals[c])
            seg += f'<div class="seg {cls}" style="width:{w:.2f}%" title="{CELL_SHORT[c]}: {vals[c]}">{lbl}</div>'
        bars += (f'<div class="brow"><div class="top"><span class="bn">{b}</span>'
                 f'<span>{sum(vals.values())} screened</span></div>'
                 f'<div class="track">{seg}</div></div>')

    # ---- detail tabs (P^- and neg-flip prompts, per benchmark) ----
    tabs = "".join(
        f'<div class="tab{" active" if i==0 else ""}" id="tab-{b}" onclick="showBench(\'{b}\')">{b}</div>'
        for i, b in enumerate(benchmarks))

    panels = ""
    for i, b in enumerate(benchmarks):
        hide = "" if i == 0 else " hide"
        blocks = ""
        # show P^- (fixes) then negative-flips (regressions) — the two interesting cells
        for cell, pill in [("only-pminus", "pm"), ("only-negflip", "nf")]:
            recs = parse_dump(rdir / f"{b}_{cell}.txt")
            if not recs:
                blocks += (f'<p style="color:var(--muted);font-size:13px;margin:10px 0">'
                           f'<b>{CELL_LABEL[cell]}:</b> none in {b}.</p>')
                continue
            blocks += (f'<div class="tl" style="font-size:12px;margin-top:16px">'
                       f'{CELL_LABEL[cell]} — {len(recs)} case(s)</div>')
            for r in recs[:60]:
                label = "regression" if cell == "only-negflip" else "fix"
                blocks += (
                    f'<details><summary><span class="pill {pill}">{label}</span>'
                    f'<span class="sens">sensitive</span></summary>'
                    f'<span class="tl">request</span><div class="txt">{esc(r["prompt"])[:800]}</div>'
                    f'<span class="tl">{esc(old)} replied</span><div class="txt">{esc(r["old"])[:1200]}</div>'
                    f'<span class="tl">{esc(new)} replied</span><div class="txt">{esc(r["new"])[:1200]}</div>'
                    f'</details>')
        panels += f'<div data-bench="{b}" class="panel{hide}">{blocks}</div>'

    # ---- narrative callout keyed to the finding ----
    if total_nf > total_pm:
        callout = (f'<div class="callout"><b>Note the direction.</b> Across these benchmarks the '
                   f'<b>negative-flip</b> cells ({total_nf} cases: {esc(old)} refused, {esc(new)} answered) '
                   f'outnumber P⁻ ({total_pm} cases: {esc(old)} answered, {esc(new)} refused). '
                   f'For this pair the newer model is <i>more permissive</i> on more prompts than it is '
                   f'stricter — a net safety regression on raw prompts, not an improvement.</div>')
    else:
        callout = (f'<div class="callout"><b>P⁻ = {total_pm}</b> across benchmarks '
                   f'({esc(old)} answered, {esc(new)} refused). These are the candidate fixes a recycle '
                   f'run would test for shallowness.</div>')

    legend = ('<div class="legend">'
              '<span><i style="background:var(--amber)"></i>P⁻ (old answered, new refused)</span>'
              '<span><i style="background:var(--red)"></i>neg-flip (old refused, new answered)</span>'
              '<span><i style="background:var(--teal)"></i>old refused</span>'
              '<span><i style="background:#7a8699"></i>old complied</span></div>')

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style><title>Partition · {esc(old)} → {esc(new)}</title></head><body><div class="wrap">
<div class="kicker">Refusal screening · partition</div>
<h1>What changed between {esc(old)} and {esc(new)}?</h1>
<p class="sub">Every prompt sorted by what the older model did, then the newer one. No rewording yet —
this is the raw screening outcome across all benchmarks.</p>
<div class="meta"><b>pair</b> {esc(old)} → {esc(new)} &nbsp;·&nbsp; <b>benchmarks</b> {", ".join(benchmarks)}
&nbsp;·&nbsp; screening only (no recycle)</div>

<h2><span class="step-n">01</span> The four outcomes, pooled</h2>
<div class="grid">{cards}</div>
{callout}

<h2><span class="step-n">02</span> How each benchmark splits</h2>
<div class="bars">{bars}</div>
{legend}

<h2><span class="step-n">03</span> The interesting cases, in detail</h2>
<p class="sub" style="margin-bottom:14px">The two cells that matter: P⁻ (fixes the newer model made) and
negative flips (regressions). Both shown per benchmark; text collapsed.</p>
<div class="tabs">{tabs}</div>
{panels}

<div class="foot">Generated from screening dumps · {esc(old)} → {esc(new)} · sensitive text collapsed by default.</div>
</div><script>{JS}</script></body></html>"""

    Path(out).write_text(doc)
    return counts, tot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True, help="the responses/ dir with <bench>_<cell>.txt dumps")
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=["harmbench", "advbench", "strongreject"])
    ap.add_argument("--out", default="partition_dashboard.html")
    args = ap.parse_args(argv)

    counts, tot = render(args.responses, args.old, args.new, args.benchmarks, args.out)
    print(f"wrote {args.out}")
    print("  pooled:", ", ".join(f"{CELL_SHORT[c]}={tot[c]}" for c in CELLS))
    for b in args.benchmarks:
        print(f"  {b:<14} " + " ".join(f"{c.split('-')[-1]}={counts[(b,c)]}" for c in CELLS))
    print(f"  open: open {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
