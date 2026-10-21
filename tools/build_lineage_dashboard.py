#!/usr/bin/env python3
"""tools/build_lineage_dashboard.py — consolidate the SORRY-Bench lineage into ONE file.

make_dashboard.py is single-pair, so this runs it once per adjacent pair against the
lineage run's frozen manifest + cache, then embeds each page as an iframe tab in one
self-contained HTML. Experiment-grade: the tabs ARE the real make_dashboard pages.

  python3 tools/build_lineage_dashboard.py
  python3 tools/build_lineage_dashboard.py --config configs/real_llama_lineage_all.yaml
  python3 tools/build_lineage_dashboard.py --use-existing   # skip generation, wrap newest per-rung files

Writes results/runs/lineage_sorrybench_20261020/dashboard_lineage_consolidated_<stamp>.html
(dated; never overwrites). Per-rung helper files are named _rung_<old>_to_<new>_<stamp>.html.
"""
from __future__ import annotations
import argparse, html, os, subprocess, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKE = ROOT / "tools/make_dashboard.py"
DEFAULT_RUN = "results/runs/lineage_sorrybench_20261020"
PAIRS = [
    ("llama2-7b", "llama2-13b"),
    ("llama2-13b", "llama3-8b"),
    ("llama3-8b", "llama31-8b"),   # anchor
    ("llama31-8b", "llama32-1b"),
    ("llama32-1b", "llama32-3b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/recycle_3_31.yaml")
    ap.add_argument("--cache", default="cache/llama_gen.jsonl")
    ap.add_argument("--behaviours", default="data/sorrybench.csv")
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--manifest", default=None)       # defaults to <run>/manifest.jsonl
    ap.add_argument("--out", default=None)
    ap.add_argument("--use-existing", action="store_true",
                    help="wrap the newest dashboard_<old>_to_<new>_*.html already in the run folder")
    ap.add_argument("--keep-rungs", action="store_true", help="keep the per-rung helper files")
    args = ap.parse_args()

    os.chdir(ROOT)
    if not MAKE.exists():
        sys.exit(f"{MAKE} not found — run from inside the repo.")
    run = Path(args.run)
    if not run.is_dir():
        sys.exit(f"run folder not found: {run}")
    manifest = args.manifest or str(run / "manifest.jsonl")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    frames, tmp = [], []
    for old, new in PAIRS:
        label = f"{old}→{new}"
        if args.use_existing:
            cands = sorted(run.glob(f"dashboard_{old}_to_{new}_*.html"))
            if not cands:
                print(f"SKIP  {label}: no existing dashboard_{old}_to_{new}_*.html")
                continue
            page = cands[-1]
        else:
            page = run / f"_rung_{old}_to_{new}_{stamp}.html"
            cmd = [sys.executable, str(MAKE), "--config", args.config,
                   "--old", old, "--new", new, "--cache", args.cache,
                   "--behaviours", args.behaviours, "--manifest", manifest,
                   "--meta", f"lineage · {label} · sorrybench", "--out", str(page)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            pm = next((l.strip() for l in (r.stdout + "\n" + r.stderr).splitlines() if "P^-" in l), "")
            if r.returncode != 0 or not page.exists():
                print(f"FAIL  {label}: rc={r.returncode}  {(r.stderr or r.stdout).strip()[-200:]}")
                continue
            print(f"BUILD {label}: {pm}")
            tmp.append(page)
        frames.append((label, page.read_text(encoding="utf-8")))

    if not frames:
        sys.exit("no rung pages produced — nothing to consolidate (check --config / cache).")

    out = Path(args.out) if args.out else run / f"dashboard_lineage_consolidated_{stamp}.html"
    out.write_text(render(frames, stamp), encoding="utf-8")
    if not args.keep_rungs:
        for p in tmp:
            try: p.unlink()
            except OSError: pass
    print(f"\nwrote {out}   ({len(frames)} tabs)")
    print(f"open: open {out}")


def render(frames, stamp):
    tabs, panes = [], []
    for i, (label, doc) in enumerate(frames):
        act = " active" if i == 0 else ""
        tabs.append(f'<button class="tab{act}" data-i="{i}" type="button">{html.escape(label)}</button>')
        panes.append(f'<iframe class="pane{act}" data-i="{i}" loading="lazy" '
                     f'srcdoc="{html.escape(doc, quote=True)}"></iframe>')
    tabs_html = "\n".join(tabs)
    panes_html = "\n".join(panes)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lineage (consolidated) — SORRY-Bench</title>
<style>
 :root{{--bg:#eef1f5;--bar:#fff;--ink:#15181d;--muted:#5a626e;--line:#d8dde5;--accent:#2f6df0;--accent-soft:#e7eefc}}
 @media(prefers-color-scheme:dark){{:root{{--bg:#0d1014;--bar:#141920;--ink:#e7eaef;--muted:#9aa3b0;--line:#232a33;--accent:#6ea0ff;--accent-soft:#182439}}}}
 *{{box-sizing:border-box}} html,body{{margin:0;height:100%}}
 body{{background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;display:flex;flex-direction:column;height:100vh}}
 .bar{{background:var(--bar);border-bottom:1px solid var(--line);padding:9px 14px;display:flex;gap:7px;align-items:center;flex-wrap:wrap;flex:0 0 auto}}
 .bar .ttl{{font-size:.84rem;font-weight:700;margin-right:12px;letter-spacing:-.01em}}
 .bar .ttl small{{font-weight:400;color:var(--muted)}}
 .tab{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.76rem;padding:5px 11px;border:1px solid var(--line);background:transparent;color:var(--muted);border-radius:7px;cursor:pointer}}
 .tab:hover{{color:var(--ink);border-color:var(--muted)}}
 .tab.active{{background:var(--accent-soft);color:var(--accent);border-color:var(--accent)}}
 .tab.anchor::after{{content:" ★";color:var(--accent)}}
 .frames{{flex:1 1 auto;position:relative;min-height:0}}
 .pane{{position:absolute;inset:0;width:100%;height:100%;border:0;display:none;background:#fff}}
 .pane.active{{display:block}}
</style></head><body>
<div class="bar">
 <span class="ttl">RecycleBench · lineage <small>SORRY-Bench · 5 adjacent pairs · experiment view · {stamp}</small></span>
 {tabs_html}
</div>
<div class="frames">
{panes_html}
</div>
<script>
 var tabs=document.querySelectorAll('.tab'), panes=document.querySelectorAll('.pane');
 // mark the anchor tab (3-8b->31-8b)
 tabs.forEach(function(t){{ if(t.textContent.indexOf('llama3-8b→llama31-8b')===0) t.classList.add('anchor'); }});
 tabs.forEach(function(t){{ t.addEventListener('click', function(){{
   var i=t.getAttribute('data-i');
   tabs.forEach(function(x){{x.classList.toggle('active', x.getAttribute('data-i')===i)}});
   panes.forEach(function(x){{x.classList.toggle('active', x.getAttribute('data-i')===i)}});
 }}); }});
</script></body></html>"""


if __name__ == "__main__":
    main()
