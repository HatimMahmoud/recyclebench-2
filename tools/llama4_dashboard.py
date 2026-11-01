#!/usr/bin/env python3
"""tools/llama4_dashboard.py -- one page for a Llama 4 model's canonical screen against the six-model lineage.

Every benchmark that has a CSV in data/ and a verdict store (tools/llama4_report.STORES): each lineage model's
judged compliance with Wilson intervals, the 2x2 of every earlier model -> the Llama 4 model, the behaviours where
the verdicts differ (fixes, regressions, still answered) with the replies, and where the offline refusal rule
disagrees with the judge on the Llama 4 replies. Same design system as tools/lineage_dashboard.py; no model is
called. Canonical only: Llama 4 is not in any frozen manifest, so there is no RSR here.

  python3 tools/llama4_dashboard.py --run-dir results/runs/20261031_1816_llama4_scout_canonical
  python3 tools/llama4_dashboard.py --new llama4-maverick --run-dir <its run dir>
  python3 tools/llama4_dashboard.py --redact ...            # no prompt or response text

Writes <run-dir>/dashboard_llama4_<stamp>.html; never overwrites.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lineage_core as lc                                      # noqa: E402
import lineage_dashboard as ld                                 # noqa: E402
from llama4_report import STORES                               # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.datasets import load_jailbreakbench          # noqa: E402
from recyclebench.judge import looks_like_refusal              # noqa: E402
from recyclebench.manifest import Manifest                     # noqa: E402

esc, pct, ci_txt, num, cap, label, short = ld.esc, ld.pct, ld.ci_txt, ld.num, ld.cap, ld.label, ld.short
tile, table, hbars, chip, wilson = ld.tile, ld.table, ld.hbars, ld.chip, ld.wilson

BENCH_LABEL = {"sorrybench": "SORRY-Bench", "harmbench": "HarmBench", "advbench": "AdvBench",
               "strongreject": "StrongREJECT"}
TEXT_CAP = {"prompt": 900, "reply": 1600}
NAV = [("s-headline", "01", "Headline"), ("s-models", "02", "Each model"), ("s-pairs", "03", "Every older model →"),
       ("s-cases", "04", "Where verdicts differ"), ("s-judge", "05", "Judge vs refusal rule"),
       ("s-method", "06", "Provenance & method")]


class Bench:
    def __init__(self, name, behaviours, obs, runs, cells):
        self.name, self.behaviours, self.obs, self.runs, self.cells = name, behaviours, obs, runs, cells
        self.bmap = {b.bid: b for b in behaviours}


def analyse(cfg, new, benches, cache):
    models = list(cfg.lineage)[:list(cfg.lineage).index(new) + 1]
    out = []
    for name in benches:
        behaviours = load_jailbreakbench(f"data/{name}.csv")
        store = lc.VerdictStore(STORES[name])
        verdicts = lc.StoredVerdicts(store, lc.judge_tag_from_cfg(cfg))
        obs, runs = lc.build_runs(cfg, behaviours, Manifest(), cache, verdicts, models)
        store.close()
        cells = {o: lc.partition(runs[o], runs[new]) for o in models[:-1]}
        out.append(Bench(name, behaviours, obs, runs, cells))
    return models, out


def read_provenance(run_dir: Path) -> dict:
    p = next(iter(sorted(run_dir.glob("provenance_*.txt"))), None)
    info = {}
    if p:
        for line in p.read_text().splitlines():
            if " : " in line:
                k, v = line.split(" : ", 1)
                info.setdefault(k.strip(), v.strip())
            elif line.startswith("llama4:") or line.startswith("qwen2.5:"):
                parts = line.split()
                info.setdefault("digests", []).append(f"{parts[0]} {parts[1]}")
        info["file"] = str(p)
    return info


def complied(bench, m):
    run = bench.runs[m]
    return len(run.broken_set()), len(run.canonical_obs)


# =============================================================================
# sections
# =============================================================================
def sec_header(new, models, benches, prov, args, today):
    n_beh = sum(len(b.behaviours) for b in benches)
    return f"""
<header class="hero" id="top">
  <p class="eyebrow">RecycleBench · canonical screen · {esc(" · ".join(BENCH_LABEL.get(b.name, b.name) for b in benches))}</p>
  <h1>Does {esc(label(new))} refuse what earlier Llamas answered?</h1>
  <p class="lede">Every benchmark request is sent to {esc(label(new))} as written, three draws each, and judged by the
  same decorrelated judge as the earlier lineage (qwen2.5:14b rubric). A behaviour counts as <b>answered</b> if at least
  one draw complies. For each earlier model, <b>P⁻</b> is the set it answered and {esc(label(new))} refused (the fixes);
  a <b>regression</b> is the reverse. Recycling (RSR) needs {esc(label(new))} in a frozen rewording manifest and is not
  on this page.</p>
  <div class="meta">
    <span><b>{len(models)}</b> models</span><span><b>{n_beh:,}</b> behaviours · <b>{len(benches)}</b> benchmarks</span>
    <span>judge <b>{esc(lc.judge_tag_from_cfg(args.cfg) or 'none')}</b></span>
    <span>GPU <b>{esc(prov.get('gpu', '?').split(',')[0])}</b></span>
    <span>Ollama <b>{esc(prov.get('ollama', '?').replace('ollama version is ', ''))}</b></span>
    <span>generated <b>{today}</b></span>
  </div>
  <div class="meta"><span class="chip chip-warn"><span aria-hidden="true">▲</span> post hoc · not pre-registered</span>
  {chip("scale-confounded")}<span class="chip">judged · raw (no calibration on {esc(short(new))} replies)</span>
  {"<span class='chip chip-warn'>redacted · no prompt or response text</span>" if args.redact else ""}</div>
</header>"""


def sec_headline(new, models, benches, floor):
    tiles = []
    for b in benches:
        k, n = complied(b, new)
        p, lo, hi = wilson(k, n)
        best = max(models[:-1], key=lambda o: len(b.cells[o]["pminus"]))
        tiles.append(tile(f"{esc(BENCH_LABEL.get(b.name, b.name))} · {esc(short(new))} answered", pct(p),
                          f"{k} of {n} · Wilson {ci_txt(lo, hi)}<br>largest P⁻: {len(b.cells[best]['pminus'])} "
                          f"from {esc(short(best))}", emph=True))
    worst = sorted(((len(b.cells[o]["neg_flip"]), BENCH_LABEL.get(b.name, b.name), o)
                    for b in benches for o in models[:-1]), reverse=True)[0]
    max_pm = max(len(b.cells[o]["pminus"]) for b in benches for o in models[:-1])
    return f"""
<section id="s-headline" class="sec" aria-labelledby="h-headline">
  <h2 id="h-headline"><span class="sn">01</span>Headline</h2>
  <div class="tiles">{''.join(tiles)}</div>
  <p class="note">P⁻ is at most {max_pm} for any older model on any benchmark: {esc(label(new))} refuses few of the requests
  the earlier models answered, and answers many they refused. The largest regression count is {worst[0]} ({esc(worst[1])},
  {esc(short(worst[2]))} → {esc(short(new))}). All figures are raw judge output; with P⁻ this small none of them is
  estimable against the {floor}-fix floor (decision.min_p_minus).</p>
</section>"""


def sec_models(new, models, benches):
    tabs, panels = [], []
    for i, b in enumerate(benches):
        rows = []
        for m in models:
            k, n = complied(b, m)
            p, lo, hi = wilson(k, n)
            miss = b.obs[m].missing_responses
            rows.append(dict(label=label(m), sub=f"{k} of {n}" + (f" · {miss} draws missing" if miss else ""),
                             v=p, lo=lo, hi=hi, v_txt=pct(p),
                             tip=f"{label(m)} answered {k} of {n} {BENCH_LABEL.get(b.name, b.name)} requests "
                                 f"(≥1 of 3 draws judged compliant) · Wilson {ci_txt(lo, hi)}"))
        tid = f"m-{b.name}"
        tabs.append(_tab(tid, BENCH_LABEL.get(b.name, b.name), f"{len(b.behaviours)} requests", i == 0))
        panels.append(f'<div class="tabpanel" role="tabpanel" id="panel-{tid}" aria-labelledby="tab-{tid}"'
                      f'{"" if i == 0 else " hidden"}><div class="panel"><h3>Share of {esc(BENCH_LABEL.get(b.name, b.name))} '
                      f'answered, per model</h3>{hbars(rows, "Compliance per model")}</div></div>')
    return f"""
<section id="s-models" class="sec" aria-labelledby="h-models">
  <h2 id="h-models"><span class="sn">02</span>Each model</h2>
  <p class="note">Share of requests each model answered on at least one of three draws, with 95% Wilson intervals.</p>
  <div class="tablist" role="tablist" aria-label="Benchmarks">{''.join(tabs)}</div>{''.join(panels)}
</section>"""


def _tab(tid, title, sub, active):
    return (f'<button type="button" role="tab" id="tab-{tid}" aria-controls="panel-{tid}" '
            f'aria-selected="{"true" if active else "false"}" tabindex="{0 if active else -1}" class="tab">'
            f'<span class="tt">{esc(title)}</span><span class="ts">{esc(sub)}</span></button>')


def sec_pairs(new, models, benches):
    head = ["Older model", "Design", "P⁻ · fixes", "Still answered", "Regressions", "Refused by both"]
    tabs, panels = [], []
    for i, b in enumerate(benches):
        rows = []
        for o in models[:-1]:
            c = b.cells[o]
            st = lc.pair_status(o, new)
            rows.append([f"{esc(label(o))} → {esc(short(new))}", chip(st["kind"]),
                         f"<b>{len(c['pminus'])}</b>", len(c["persistent"]), len(c["neg_flip"]), len(c["both_safe"])])
        tid = f"p-{b.name}"
        tabs.append(_tab(tid, BENCH_LABEL.get(b.name, b.name), f"max P⁻ {max(len(b.cells[o]['pminus']) for o in models[:-1])}", i == 0))
        panels.append(f'<div class="tabpanel" role="tabpanel" id="panel-{tid}" aria-labelledby="tab-{tid}"'
                      f'{"" if i == 0 else " hidden"}><div class="panel"><h3>{esc(BENCH_LABEL.get(b.name, b.name))}: every '
                      f'older model → {esc(short(new))}</h3>{table(head, rows, num_cols=(2, 3, 4, 5))}</div></div>')
    return f"""
<section id="s-pairs" class="sec" aria-labelledby="h-pairs">
  <h2 id="h-pairs"><span class="sn">03</span>Every older model → {esc(short(new))}</h2>
  <p class="note">The canonical 2×2 per pair: <b>P⁻</b> older answered, {esc(short(new))} refused; <b>regressions</b>
  older refused, {esc(short(new))} answered. Every pair is scale-confounded ({esc(label(new))} is a
  {lc.meta(new)['size']}B-parameter mixture of experts, 17B active).</p>
  <div class="tablist" role="tablist" aria-label="Benchmarks">{''.join(tabs)}</div>{''.join(panels)}
</section>"""


def _reply_block(who, text):
    return f'<div class="txt"><span class="tl">{esc(who)}</span>{esc(cap(text, TEXT_CAP["reply"]))}</div>'


def _cases(new, models, b, redact):
    olds = models[:-1]
    items, cnt = [], Counter()
    for beh in b.behaviours:
        bid = beh.bid
        vn = b.obs[new].canonical.get(bid)
        vo = {o: b.obs[o].canonical.get(bid) for o in olds}
        answered_old = [o for o in olds if vo[o] == 1]
        if vn is None or (vn == 0 and not answered_old):
            continue
        if vn == 0:
            kind, pill = "fix", '<span class="pill pill-refuse">fixed by ' + esc(short(new)) + "</span>"
        elif not answered_old:
            kind, pill = "regression", '<span class="pill pill-comply">regression</span>'
        else:
            kind, pill = "persistent", '<span class="pill">still answered</span>'
        cnt[kind] += 1
        dots = "".join(f'<span class="vtc" title="{esc(label(m))}">{esc(short(m))} '
                       f'{"✔" if (vn if m == new else vo[m]) == 1 else "✘" if (vn if m == new else vo[m]) == 0 else "?"}</span>'
                       for m in models)
        if redact:
            body = f'<div class="vrow">{dots}</div><p class="note">Text redacted.</p>'
        else:
            draws_new = b.obs[new].draws[bid]
            new_t = next((t for t, v in draws_new if v == 1), None) or next((t for t, _ in draws_new if t), "")
            verdicts = "".join("C" if v == 1 else "R" if v == 0 else "?" for _, v in draws_new)
            body = (f'<div class="vrow">{dots}</div>'
                    f'<div class="txt"><span class="tl">request</span>{esc(cap(beh.canonical_prompt, TEXT_CAP["prompt"]))}</div>'
                    + _reply_block(f"{label(new)} · {'answered' if vn == 1 else 'refused'} · draws {verdicts}", new_t)
                    + "".join(_reply_block(f"{label(o)} · answered",
                                           next((t for t, v in b.obs[o].draws[bid] if v == 1), ""))
                              for o in answered_old))
        items.append(f'<details class="case" data-outcome="{kind}"><summary><span class="cid">{esc(bid)}</span>'
                     f'<span class="ccat">{esc(cap(beh.category, 40))}</span>'
                     f'{"" if redact else "<span class=sens>sensitive</span>"}{pill}</summary>'
                     f'<div class="cbody">{body}</div></details>')
    if not items:
        return "<p class='note'>Every model refused every request here.</p>"
    btns = "".join(f'<button type="button" class="fbtn{" active" if k == "all" else ""}" data-filter="{k}" '
                   f'aria-pressed="{"true" if k == "all" else "false"}">{lbl} <span>{c}</span></button>'
                   for k, lbl, c in [("all", "All", len(items)), ("fix", f"Fixed by {short(new)}", cnt["fix"]),
                                     ("regression", "Regressions", cnt["regression"]),
                                     ("persistent", "Still answered", cnt["persistent"])] if k == "all" or c)
    return f'<div class="cases"><div class="filters" role="group" aria-label="Filter cases">{btns}</div>{"".join(items)}</div>'


def sec_cases(new, models, benches, redact):
    tabs, panels = [], []
    for i, b in enumerate(benches):
        tid = f"c-{b.name}"
        tabs.append(_tab(tid, BENCH_LABEL.get(b.name, b.name), "", i == 0))
        panels.append(f'<div class="tabpanel" role="tabpanel" id="panel-{tid}" aria-labelledby="tab-{tid}"'
                      f'{"" if i == 0 else " hidden"}><div class="panel"><h3>{esc(BENCH_LABEL.get(b.name, b.name))}: '
                      f'requests any model answered</h3>{_cases(new, models, b, redact)}</div></div>')
    return f"""
<section id="s-cases" class="sec" aria-labelledby="h-cases">
  <h2 id="h-cases"><span class="sn">04</span>Where verdicts differ</h2>
  <p class="note">Every request at least one model answered. ✔ answered, ✘ refused (judge, any of three draws).
  {"Text is redacted in this build." if redact else
   "Real request and reply text, truncated and collapsed: the first compliant draw of each model that answered, else "
   + esc(short(new)) + "'s first draw."}</p>
  <div class="tablist" role="tablist" aria-label="Benchmarks">{''.join(tabs)}</div>{''.join(panels)}
</section>"""


def sec_judge(new, benches, redact):
    rows, cases = [], []
    for b in benches:
        agree, j_c, r_c, n = 0, 0, 0, 0
        for beh in b.behaviours:
            for t, v in b.obs[new].draws[beh.bid]:
                if t is None or v is None:
                    continue
                rule = int(not looks_like_refusal(t))
                n += 1
                agree += rule == v
                if rule == 1 and v == 0:
                    r_c += 1
                    kind = "rule says answered, judge says refused"
                elif rule == 0 and v == 1:
                    j_c += 1
                    kind = "rule says refused, judge says answered"
                else:
                    continue
                if len(cases) < 40:
                    body = ("<p class='note'>Text redacted.</p>" if redact else
                            f'<div class="txt"><span class="tl">request</span>{esc(cap(beh.canonical_prompt, TEXT_CAP["prompt"]))}</div>'
                            + _reply_block(label(new), t))
                    cases.append(f'<details class="case"><summary><span class="cid">{esc(beh.bid)}</span>'
                                 f'<span class="ccat">{esc(BENCH_LABEL.get(b.name, b.name))}</span>'
                                 f'<span class="pill">{esc(kind)}</span></summary><div class="cbody">{body}</div></details>')
        rows.append([esc(BENCH_LABEL.get(b.name, b.name)), f"{n:,}", pct(agree / n if n else None),
                     r_c, j_c])
    return f"""
<section id="s-judge" class="sec" aria-labelledby="h-judge">
  <h2 id="h-judge"><span class="sn">05</span>Judge vs refusal rule on {esc(short(new))}</h2>
  <p class="note">{esc(label(new))} refuses in its own words ("I can't help with that…", or a hedge offering a safer topic),
  so the offline refusal-prefix rule (recyclebench.judge.looks_like_refusal) misreads some of its replies. Every figure on
  this page uses the judge; this table shows how far the rule would move them. Per draw, not per behaviour.</p>
  {table(["Benchmark", "Draws", "Agreement", "Rule: answered · judge: refused", "Rule: refused · judge: answered"], rows, num_cols=(1, 2, 3, 4))}
  <div class="cases">{''.join(cases)}</div>
</section>"""


def sec_method(new, models, benches, prov, args):
    stores = "".join(f"<li>{esc(BENCH_LABEL.get(b.name, b.name))}: <code>data/{esc(b.name)}.csv</code> "
                     f"({len(b.behaviours)} behaviours) · verdicts <code>{esc(STORES[b.name])}</code></li>" for b in benches)
    missing = [n for n in STORES if n not in {b.name for b in benches}]
    digests = "".join(f"<li><code>{esc(d)}</code></li>" for d in prov.get("digests", []))
    return f"""
<section id="s-method" class="sec" aria-labelledby="h-method">
  <h2 id="h-method"><span class="sn">06</span>Provenance &amp; method</h2>
  <div class="grid2">
    <div class="panel"><h3>Inputs</h3><ul>{stores}
      <li>responses <code>{esc(args.cache)}</code> (config keys of <code>{esc(args.config)}</code>)</li>
      <li>run <code>{esc(args.run_dir)}</code> · provenance <code>{esc(prov.get('file', '—'))}</code></li></ul>
      <h3>Where {esc(short(new))} ran</h3><ul><li>job {esc(prov.get('job', '?'))}</li>
      <li>GPU {esc(prov.get('gpu', '?'))}</li><li>{esc(prov.get('ollama', '?'))}</li>{digests}</ul></div>
    <div class="panel"><h3>Read with care</h3><ul>
      <li>{esc(label(new))} ran on DFKI Pegasus (CUDA), the earlier models on the M1 Mac (Metal): same Ollama tags and
      seeds, different hardware and Ollama build. Treat it as a separate environment.</li>
      <li>Post hoc: not in the pre-registration or Amendment 1. Every pair changes scale, so no pair here is a matched-scale
      generational comparison. Exploratory only.</li>
      <li>Verdicts are raw qwen2.5:14b rubric output. The judge was calibrated on SORRY-Bench replies of the 8B pair, not on
      {esc(short(new))}'s replies; section 05 shows its refusal style is different.</li>
      <li>Temperature 0, three seeded draws: draws can still differ (non-deterministic GPU kernels), so "any of three" matters.</li>
      <li>Not on this page: {esc(", ".join(BENCH_LABEL.get(n, n) for n in missing) or "—")} (no CSV/verdicts here),
      rewording recycling (RSR) and PBR, which need {esc(short(new))} in a frozen manifest.</li></ul></div>
  </div>
</section>"""


def render(new, models, benches, prov, args) -> str:
    today = _dt.date.today().isoformat()
    nav = "".join(f'<a href="#{i}"><span class="n">{n}</span>{t}</a>' for i, n, t in NAV)
    body = (sec_header(new, models, benches, prov, args, today) + sec_headline(new, models, benches, args.cfg.decision.min_p_minus)
            + sec_models(new, models, benches) + sec_pairs(new, models, benches)
            + sec_cases(new, models, benches, args.redact) + sec_judge(new, benches, args.redact)
            + sec_method(new, models, benches, prov, args))
    title = f"{label(new)} Canonical Screen"
    foot = ("Generated by tools/llama4_dashboard.py · recyclebench. "
            + ("Prompt and response text is omitted in this build. " if args.redact else
               "Contains benchmark prompts and model responses (collapsed, truncated): a working file, not a released "
               "artefact — see ETHICS.md. ")
            + "All figures are raw judge output.")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<title>{esc(title)}</title><style>{ld.CSS}</style></head><body>'
            f'<div class="layout"><nav class="side" aria-label="Sections">'
            f'{ld.brand_html(label(new) + " · canonical screen")}<div class="toc">{nav}</div>{ld.legend_html()}'
            f'<button type="button" class="themebtn" id="themebtn" data-t="system">Theme: system</button></nav>'
            f'<main id="main">{body}<footer>{esc(foot)}</footer></main></div>'
            f'<div id="tip" role="tooltip" hidden></div><script>{ld.rq_script({})}{ld.JS}</script></body></html>')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/llama4_pegasus.yaml")
    ap.add_argument("--new", default="llama4-scout")
    ap.add_argument("--cache", default="cache/llama_gen.jsonl")
    ap.add_argument("--benchmarks", nargs="*",
                    default=[b for b in ("sorrybench", "harmbench", "advbench", "strongreject")
                             if Path(f"data/{b}.csv").exists() and Path(STORES[b]).exists()])
    ap.add_argument("--run-dir", required=True, help="the llama4_canonical.sh run dir (provenance, output)")
    ap.add_argument("--redact", action="store_true", help="omit every prompt and response text")
    ap.add_argument("--out", default=None, help="default: <run-dir>/dashboard_llama4_<stamp>.html (never overwrites)")
    args = ap.parse_args(argv)

    args.cfg = cfg = Config.load(args.config)
    if args.new not in cfg.lineage:
        raise SystemExit(f"{args.new} is not in the lineage of {args.config}")
    if not args.benchmarks:
        raise SystemExit("no benchmark has both data/<bench>.csv and its verdict store")
    cache = lc.load_response_cache(args.cache)
    models, benches = analyse(cfg, args.new, args.benchmarks, cache)
    for b in benches:
        miss = sum(b.obs[m].missing_verdicts["canonical"] for m in models)
        k, n = complied(b, args.new)
        print(f"{b.name:13s} {args.new} answered {k}/{n}" + (f"  ({miss} draws unjudged)" if miss else ""))
    run_dir = Path(args.run_dir)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.out) if args.out else run_dir / f"dashboard_llama4{'_redacted' if args.redact else ''}_{stamp}.html"
    if out.exists():
        raise SystemExit(f"{out} exists -- refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(args.new, models, benches, read_provenance(run_dir), args), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
