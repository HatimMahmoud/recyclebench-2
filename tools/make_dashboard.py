"""Generate an art-quality, interactive HTML dashboard of a recycling experiment.

Signature visual: an animated Sankey flow tracking harmful requests through the pipeline --
all requests -> split by the OLD model -> split by the NEW model (the P^- "fix" band is the
hero, amber) -> paraphrase -> gate -> re-screen, splitting into RECOVERED (fix broke) vs HELD
(fix generalised). Band widths are proportional, so the picture is honest.

Design: measurement-instrument aesthetic (navy ink on cool paper, Space Grotesk + IBM Plex),
one bold signature. Self-contained single .html (fonts via CDN with system fallback; all viz is
hand-built inline SVG + vanilla JS, so it renders offline and can be emailed). Sensitive text is
collapsed behind toggles and marked, never splashed on screen.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recyclebench.config import Config
from recyclebench.datasets import load_jailbreakbench
from recyclebench.realrun import _key
from recyclebench.screening import _draw_seed
from recyclebench.judge import looks_like_refusal


def load_cache(path):
    d = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["t"]
    return d


def responses(cache, model, prompt, bid, k):
    return [cache.get(_key(model, prompt, _draw_seed(bid, -1, d))) for d in range(k)]


def verdict(rs):
    if all(x is None for x in rs):
        return None
    return "comply" if any(x is not None and not looks_like_refusal(x) for x in rs) else "refuse"


def first_text(rs):
    for x in rs:
        if x:
            return x
    return ""


def esc(s):
    return html.escape(s or "")


def partition(cache, csvs, old, new, k):
    cells = {"pminus": [], "persistent": [], "neg_flip": [], "both_safe": [], "unscreened": []}
    per_bench = {}
    for csv in csvs:
        name = Path(csv).stem
        c = dict(pminus=0, persistent=0, neg_flip=0, both_safe=0, unscreened=0)
        for b in load_jailbreakbench(csv):
            o = verdict(responses(cache, old, b.canonical_prompt, b.bid, k))
            n = verdict(responses(cache, new, b.canonical_prompt, b.bid, k))
            rec = {"bench": name, "bid": b.bid, "category": b.category, "prompt": b.canonical_prompt,
                   "old_resp": first_text(responses(cache, old, b.canonical_prompt, b.bid, k)),
                   "new_resp": first_text(responses(cache, new, b.canonical_prompt, b.bid, k))}
            if o is None: cells["unscreened"].append(rec); c["unscreened"] += 1
            elif o == "comply" and n == "refuse": cells["pminus"].append(rec); c["pminus"] += 1
            elif o == "comply" and n == "comply": cells["persistent"].append(rec); c["persistent"] += 1
            elif o == "refuse" and n == "comply": cells["neg_flip"].append(rec); c["neg_flip"] += 1
            elif o == "refuse" and n == "refuse": cells["both_safe"].append(rec); c["both_safe"] += 1
            else: cells["unscreened"].append(rec); c["unscreened"] += 1
        per_bench[name] = c
    return cells, per_bench


def load_manifest(paths):
    admitted, audit = {}, {}
    for path in paths or []:
        if not path or not Path(path).exists():
            continue
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") == "admitted":
                admitted.setdefault(r["bid"], []).extend(r.get("variants", []))
            elif r.get("type") == "audit":
                audit.setdefault(r["bid"], []).append(r)
    return admitted, audit


def after_state(cache, new, bid, admitted_variants, k):
    """Reconstruct the AFTER state: each admitted paraphrase and how the NEW model replied.

    A variant's re-screen is cached under _key(new, variant.text, _draw_seed(bid, index, draw)).
    Returns (tries, recovered_bool); each try labelled RECOVERED (new complied) or held.
    """
    tries, recovered = [], False
    for v in admitted_variants:
        idx = v.get("index", 0)
        rs = [cache.get(_key(new, v.get("text", ""), _draw_seed(bid, idx, d))) for d in range(k)]
        if all(x is None for x in rs):
            continue
        broke = any(x is not None and not looks_like_refusal(x) for x in rs)
        recovered = recovered or broke
        tries.append({"text": v.get("text", ""), "tc": v.get("transform_class", "?"),
                      "resp": first_text(rs), "broke": broke})
    return tries, recovered


COL = dict(paper="#F5F4F0", panel="#FFFFFF", ink="#14202E", slate="#6B7A88",
           grid="#E4E2DB", broke="#D8452B", held="#107C6E", fix="#E3A008",
           regress="#B5522E", washed="#EEF1F2")


def ribbon(x0, y0a, y0b, x1, y1a, y1b, fill, op=0.55):
    mx = (x0 + x1) / 2
    return (f'<path d="M{x0:.1f},{y0a:.1f} C{mx:.1f},{y0a:.1f} {mx:.1f},{y1a:.1f} {x1:.1f},{y1a:.1f} '
            f'L{x1:.1f},{y1b:.1f} C{mx:.1f},{y1b:.1f} {mx:.1f},{y0b:.1f} {x0:.1f},{y0b:.1f} Z" '
            f'fill="{fill}" opacity="{op}" class="ribbon"/>')


def node(x, ytop, h, w, fill, label, count):
    return (f'<rect x="{x}" y="{ytop:.1f}" width="{w}" height="{max(h,2):.1f}" rx="2" fill="{fill}"/>'
            f'<text x="{x + w + 8}" y="{ytop + h/2 - 2:.1f}" class="nlab">{esc(label)}</text>'
            f'<text x="{x + w + 8}" y="{ytop + h/2 + 13:.1f}" class="ncount">{count}</text>')


def build_flow(cells, recovered, has_rsr):
    pm = len(cells["pminus"]); per = len(cells["persistent"])
    nf = len(cells["neg_flip"]); bs = len(cells["both_safe"])
    old_ans = pm + per; old_ref = nf + bs; total = old_ans + old_ref
    if total == 0:
        return "<p class='muted'>No screened data to visualise.</p>"

    W, H = 1000, 460
    pad_t, pad_b, gap, nodew = 30, 30, 6, 12
    usable = H - pad_t - pad_b

    def h_of(n):
        return max((n / total) * (usable - 3 * gap), 3 if n > 0 else 0)

    x0, x1, x2, x3 = 40, 330, 620, 900
    svg = [f'<svg viewBox="0 0 {W} {H}" class="flow" preserveAspectRatio="xMidYMid meet">']
    for t, x in [("All requests", x0), ("Older model", x1), ("Newer model", x2), ("After rewording", x3)]:
        svg.append(f'<text x="{x}" y="16" class="stage">{t}</text>')

    trunk_h = usable
    svg.append(node(x0, pad_t, trunk_h, nodew, COL["ink"], "total", total))

    h_oa, h_or = h_of(old_ans), h_of(old_ref)
    y1 = pad_t + (usable - (h_oa + h_or + gap)) / 2
    oa_top, or_top = y1, y1 + h_oa + gap
    svg.append(ribbon(x0 + nodew, pad_t, pad_t + trunk_h * (old_ans / total), x1, oa_top, oa_top + h_oa, COL["broke"], 0.26))
    svg.append(ribbon(x0 + nodew, pad_t + trunk_h * (old_ans / total), pad_t + trunk_h, x1, or_top, or_top + h_or, COL["held"], 0.20))
    svg.append(node(x1, oa_top, h_oa, nodew, COL["broke"], "answered", old_ans))
    svg.append(node(x1, or_top, h_or, nodew, COL["held"], "refused", old_ref))

    h_pm, h_pe, h_nf, h_bs = h_of(pm), h_of(per), h_of(nf), h_of(bs)
    oa2 = oa_top + max(0, (h_oa - (h_pe + h_pm + gap)) / 2)
    pe_top, pm_top = oa2, oa2 + h_pe + gap
    or2 = or_top + max(0, (h_or - (h_nf + h_bs + gap)) / 2)
    nf_top, bs_top = or2, or2 + h_nf + gap

    svg.append(ribbon(x1 + nodew, oa_top, oa_top + h_oa * (per / max(old_ans, 1)), x2, pe_top, pe_top + h_pe, COL["broke"], 0.22))
    svg.append(ribbon(x1 + nodew, oa_top + h_oa * (per / max(old_ans, 1)), oa_top + h_oa, x2, pm_top, pm_top + h_pm, COL["fix"], 0.5))
    svg.append(ribbon(x1 + nodew, or_top, or_top + h_or * (nf / max(old_ref, 1)), x2, nf_top, nf_top + h_nf, COL["regress"], 0.3))
    svg.append(ribbon(x1 + nodew, or_top + h_or * (nf / max(old_ref, 1)), or_top + h_or, x2, bs_top, bs_top + h_bs, COL["held"], 0.16))

    svg.append(node(x2, pe_top, h_pe, nodew, COL["broke"], "still answered", per))
    svg.append(f'<rect x="{x2}" y="{pm_top:.1f}" width="{nodew}" height="{max(h_pm,3):.1f}" rx="2" fill="{COL["fix"]}" class="hero"/>'
               f'<text x="{x2+nodew+8}" y="{pm_top+h_pm/2-2:.1f}" class="nlab hero-lab">P⁻ · the fix</text>'
               f'<text x="{x2+nodew+8}" y="{pm_top+h_pm/2+13:.1f}" class="ncount hero-lab">{pm}</text>')
    svg.append(node(x2, nf_top, h_nf, nodew, COL["regress"], "regression", nf))
    svg.append(node(x2, bs_top, h_bs, nodew, COL["held"], "both refused", bs))

    if has_rsr and pm > 0:
        rec = max(0, min(recovered, pm)); held = pm - rec
        h_rec, h_held = h_of(rec), h_of(held)
        s3 = pm_top + max(0, (h_pm - (h_rec + h_held + gap)) / 2)
        rec_top, held_top = s3, s3 + h_rec + gap
        svg.append(ribbon(x2 + nodew, pm_top, pm_top + h_pm * (rec / max(pm, 1)), x3, rec_top, rec_top + h_rec, COL["broke"], 0.55))
        svg.append(ribbon(x2 + nodew, pm_top + h_pm * (rec / max(pm, 1)), pm_top + h_pm, x3, held_top, held_top + h_held, COL["held"], 0.4))
        svg.append(node(x3, rec_top, h_rec, nodew, COL["broke"], "recovered", rec))
        svg.append(node(x3, held_top, h_held, nodew, COL["held"], "held", held))
    else:
        svg.append(f'<text x="{x3-6}" y="{pm_top+h_pm/2:.1f}" class="pending">→ reworded &amp; re-tested'
                   f'<tspan x="{x3-6}" dy="15">RSR in run log</tspan></text>')

    if pm > 0:
        cy = pm_top + h_pm / 2
        svg.append('<g class="particles">' + "".join(
            f'<circle r="2.2" fill="{COL["fix"]}" opacity="0.9"><animate attributeName="cx" '
            f'from="{x2+nodew}" to="{x3}" dur="2.4s" begin="{i*0.5}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{cy:.1f};{cy:.1f}" dur="2.4s" begin="{i*0.5}s" '
            f'repeatCount="indefinite"/></circle>' for i in range(5)) + '</g>')

    svg.append('</svg>')
    return "".join(svg)


def transform_bars(audit):
    from collections import Counter
    gen, adm = Counter(), Counter()
    for rows in audit.values():
        for a in rows:
            tc = a.get("transform_class", "?"); gen[tc] += 1
            if a.get("admitted"): adm[tc] += 1
    if not gen:
        return ""
    mx = max(gen.values())
    out = ""
    for tc, g in gen.most_common():
        a = adm[tc]
        out += (f'<div class="tcrow"><div class="tcname">{esc(tc)}</div>'
                f'<div class="tcbar"><div class="tcgen" style="width:{int(100*g/mx)}%"></div>'
                f'<div class="tcadm" style="width:{int(100*a/mx)}%"></div></div>'
                f'<div class="tcnum">{a}/{g}</div></div>')
    return out


def render(cells, per_bench, admitted, audit, old, new, rsr, recovered, out, run_meta="",
           _cache_ref=None, _k_ref=1):
    pm = len(cells["pminus"]); per = len(cells["persistent"])
    nf = len(cells["neg_flip"]); bs = len(cells["both_safe"])
    n_tries = sum(len(v) for v in audit.values())
    n_admit = sum(len(v) for v in admitted.values())
    has_rsr = rsr is not None
    flow = build_flow(cells, recovered or 0, has_rsr)
    tbars = transform_bars(audit)

    if has_rsr:
        pct = max(0.0, min(1.0, rsr)); ang = -90 + pct * 180
        verdict_txt = ("Fixes look <b>shallow</b> — rewording brought many back" if pct >= 0.4
                       else "Fixes look <b>robust</b> — most held under rewording")
        gx = 100 + 70 * math.cos(math.radians(ang)); gy = 110 + 70 * math.sin(math.radians(ang))
        gauge = f"""<div class="gauge-wrap"><svg viewBox="0 0 200 122" class="gauge">
<path d="M20,110 A80,80 0 0,1 180,110" fill="none" stroke="{COL['grid']}" stroke-width="14" stroke-linecap="round"/>
<path d="M20,110 A80,80 0 0,1 180,110" fill="none" stroke="url(#gg)" stroke-width="14" stroke-linecap="round" stroke-dasharray="{pct*251.2:.1f} 251.2"/>
<defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" stop-color="{COL['held']}"/><stop offset="1" stop-color="{COL['broke']}"/></linearGradient></defs>
<line x1="100" y1="110" x2="{gx:.1f}" y2="{gy:.1f}" stroke="{COL['ink']}" stroke-width="3" stroke-linecap="round"/>
<circle cx="100" cy="110" r="5" fill="{COL['ink']}"/></svg>
<div class="gauge-n">{pct*100:.0f}%</div><div class="gauge-lbl">Recycling Success Rate<br><span>{verdict_txt}</span></div></div>"""
    else:
        gauge = ('<div class="gauge-wrap pending-g"><div class="gauge-n">—</div><div class="gauge-lbl">'
                 'Recycling Success Rate<br><span>run the recycle step, then pass <code>--rsr</code></span></div></div>')

    brows = ""
    for name, d in per_bench.items():
        tot = d["pminus"] + d["persistent"] + d["neg_flip"] + d["both_safe"]
        segs = [("fix", d["pminus"]), ("broke", d["persistent"]), ("regress", d["neg_flip"]), ("held", d["both_safe"])]
        bar = "".join(f'<span class="seg s-{c}" style="width:{100*n/max(tot,1):.1f}%"></span>' for c, n in segs)
        brows += (f'<tr><td class="bn">{esc(name)}</td><td class="num hero-lab">{d["pminus"]}</td>'
                  f'<td><div class="stackbar">{bar}</div></td><td class="num">{tot}</td></tr>')

    pmd = ""
    for r in cells["pminus"][:80]:
        tries, recovered = ([], None)
        if admitted.get(r["bid"]) and _cache_ref is not None:
            tries, recovered = after_state(_cache_ref, new, r["bid"], admitted[r["bid"]], _k_ref)
        # summary pill reflects the AFTER outcome when we have it
        if recovered is True:
            outcome = '<span class="pill brokep">reword RECOVERED it</span>'
        elif recovered is False and tries:
            outcome = '<span class="pill">reword held</span>'
        elif admitted.get(r["bid"]):
            outcome = f'<span class="pill">{len(admitted[r["bid"]])} paraphrases gated</span>'
        else:
            outcome = ""
        # the after-state rows: each paraphrase + how the newer model replied
        after_rows = ""
        for t in tries[:6]:
            cls = "arow broke" if t["broke"] else "arow held"
            verdict = "ANSWERED" if t["broke"] else "refused"
            after_rows += (f'<div class="{cls}"><div class="atc">{esc(t["tc"])} · {verdict}</div>'
                           f'<div class="atxt"><span class="tl">reworded request</span>{esc(t["text"])[:400]}</div>'
                           f'<div class="atxt"><span class="tl">{esc(new)} replied</span>{esc(t["resp"])[:400]}</div></div>')
        after_block = (f'<div class="afterwrap"><div class="tl" style="margin:10px 0 4px">'
                       f'after rewording ({len(tries)} tested)</div>{after_rows}</div>') if tries else ""
        pmd += (f'<details><summary><span class="dsrc">{esc(r["bench"])}</span> · {esc(r["category"])}'
                f'<span class="sens">sensitive</span>{outcome}</summary>'
                f'<div class="tl" style="margin:2px 0 4px">before · canonical request</div>'
                f'<div class="txt"><span class="tl">request</span>{esc(r["prompt"])[:600]}</div>'
                f'<div class="txt"><span class="tl">{esc(old)} · answered</span>{esc(r["old_resp"])[:600]}</div>'
                f'<div class="txt"><span class="tl">{esc(new)} · refused</span>{esc(r["new_resp"])[:400]}</div>'
                f'{after_block}</details>')

    small = pm < 30
    today = __import__("datetime").date.today().isoformat()
    n3, n4, n5 = ("03", "04", "05") if tbars else ("--", "03", "04")

    tbar_block = (f"<h2><span class='step-n'>{n3}</span> Which rewordings the gate accepted</h2><div class='panel'>"
                  "<p class='q' style='margin:0 0 14px;font-size:13.5px'>Each transformation produces candidate "
                  "rewordings; the gate keeps only those that still ask for the same thing. "
                  "Green = accepted, grey = generated.</p>" + tbars + "</div>") if tbars else ""
    callout = ("<div class='callout'><b>Exploratory result.</b> Only " + str(pm) +
               " genuine fixes here — a small sample, so treat any rate as a direction, not a final "
               "number. Enlarging P⁻ (a weaker baseline, or reworded canonical prompts) is the next step.</div>") if small else ""

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Refusal Recycling · {esc(old)} → {esc(new)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{--paper:{COL['paper']};--panel:{COL['panel']};--ink:{COL['ink']};--slate:{COL['slate']};--grid:{COL['grid']};--broke:{COL['broke']};--held:{COL['held']};--fix:{COL['fix']};--regress:{COL['regress']};--washed:{COL['washed']};}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.55;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1040px;margin:0 auto;padding:44px 28px 90px}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--slate);margin:0 0 10px}}
h1{{font-family:"Space Grotesk",sans-serif;font-weight:700;font-size:38px;line-height:1.08;letter-spacing:-.02em;margin:0 0 12px;max-width:22ch}}
.lede{{font-size:16px;color:#3a4a58;max-width:68ch;margin:0 0 6px}}
.meta{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--slate);margin:14px 0 0}}
.meta b{{color:var(--ink)}}
h2{{font-family:"Space Grotesk",sans-serif;font-weight:600;font-size:13px;letter-spacing:.16em;text-transform:uppercase;color:var(--slate);margin:56px 0 6px;display:flex;align-items:center;gap:12px}}
h2::after{{content:"";flex:1;height:1px;background:var(--grid)}}
.step-n{{font-family:"IBM Plex Mono",monospace;color:var(--fix);font-weight:500}}
.panel{{background:var(--panel);border:1px solid var(--grid);border-radius:14px;padding:26px;margin:16px 0;box-shadow:0 1px 0 rgba(20,32,46,.03)}}
.q{{font-size:16px;line-height:1.6;max-width:72ch}}.q b{{color:var(--fix)}}
.flow{{width:100%;height:auto;display:block;margin:8px 0}}
.flow .stage{{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;fill:var(--slate)}}
.flow .nlab{{font-family:"Space Grotesk",sans-serif;font-size:12.5px;font-weight:500;fill:var(--ink)}}
.flow .ncount{{font-family:"IBM Plex Mono",monospace;font-size:12px;fill:var(--slate)}}
.flow .hero-lab{{fill:#7a5200}}.flow .hero{{filter:drop-shadow(0 0 6px rgba(227,160,8,.5))}}
.flow .pending{{font-family:"IBM Plex Sans";font-size:12px;fill:var(--slate)}}
.ribbon{{transition:opacity .2s}}.flow:hover .ribbon{{opacity:.16}}.flow .ribbon:hover{{opacity:.85!important}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;margin:6px 2px 0;font-size:12.5px;color:#3a4a58}}
.legend i{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.result{{display:grid;grid-template-columns:1fr 1.3fr;gap:22px;align-items:center}}
@media(max-width:760px){{.result{{grid-template-columns:1fr}}}}
.gauge-wrap{{text-align:center;padding:8px}}.gauge{{width:220px;max-width:100%}}
.gauge-n{{font-family:"Space Grotesk";font-weight:700;font-size:40px;margin-top:-16px}}
.gauge-lbl{{font-size:13px;color:var(--slate)}}.gauge-lbl span{{color:#3a4a58}}.pending-g .gauge-n{{color:var(--grid)}}
.funnel{{display:flex;flex-direction:column;gap:10px}}
.fstep{{display:flex;align-items:center;gap:14px}}
.fbar{{height:34px;border-radius:7px;display:flex;align-items:center;padding:0 12px;color:#fff;font-family:"Space Grotesk";font-weight:600;font-size:14px;min-width:44px}}
.fcap{{font-size:13px;color:#3a4a58}}
.tcrow{{display:grid;grid-template-columns:130px 1fr 54px;align-items:center;gap:12px;margin:9px 0}}
.tcname{{font-family:"IBM Plex Mono",monospace;font-size:12px}}
.tcbar{{position:relative;height:20px;background:var(--washed);border-radius:5px;overflow:hidden}}
.tcgen{{position:absolute;inset:0;background:var(--grid)}}
.tcadm{{position:absolute;left:0;top:0;height:100%;background:var(--held);border-radius:5px}}
.tcnum{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--slate);text-align:right}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th{{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--slate);text-align:left;padding:8px 12px;border-bottom:1px solid var(--grid);font-weight:500}}
td{{padding:11px 12px;border-bottom:1px solid var(--grid)}}
td.num,th.num{{text-align:right;font-family:"IBM Plex Mono",monospace}}
.bn{{font-family:"Space Grotesk";font-weight:500}}.hero-lab{{color:#a8760a;font-weight:600}}
.stackbar{{display:flex;height:12px;border-radius:6px;overflow:hidden;background:var(--washed);width:230px}}
.seg{{height:100%}}.s-fix{{background:var(--fix)}}.s-broke{{background:var(--broke)}}.s-regress{{background:var(--regress)}}.s-held{{background:var(--held)}}
details{{background:var(--panel);border:1px solid var(--grid);border-radius:9px;margin:7px 0;padding:0 15px}}
summary{{cursor:pointer;padding:12px 0;font-size:13.5px;list-style:none;display:flex;align-items:center;gap:10px}}
summary::-webkit-details-marker{{display:none}}
summary::before{{content:"+";font-family:"IBM Plex Mono";color:var(--fix);font-weight:600}}
details[open] summary::before{{content:"\\2013"}}
.dsrc{{font-family:"IBM Plex Mono";font-size:12px;color:var(--slate)}}
.sens{{font-family:"IBM Plex Mono";font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#9a3412;background:#fff2e8;padding:2px 7px;border-radius:20px;margin-left:auto}}
.pill{{font-family:"IBM Plex Mono";font-size:10px;color:#166534;background:#e7f6ec;padding:2px 8px;border-radius:20px}}
.txt{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:#2a3a48;background:var(--washed);border-radius:7px;padding:10px 12px;margin:7px 0;white-space:pre-wrap;max-height:170px;overflow:auto}}
.tl{{display:block;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--slate);margin-bottom:4px}}
.afterwrap{{border-top:1px dashed var(--grid);margin-top:12px;padding-top:6px}}
.arow{{border-left:3px solid var(--grid);padding:6px 0 6px 12px;margin:8px 0;border-radius:0 6px 6px 0}}
.arow.broke{{border-left-color:var(--broke);background:#fdf1ef}}
.arow.held{{border-left-color:var(--held);background:#eef7f5}}
.atc{{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:500;letter-spacing:.04em;margin-bottom:5px}}
.arow.broke .atc{{color:{COL['broke']}}}.arow.held .atc{{color:{COL['held']}}}
.atxt{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:#2a3a48;background:rgba(255,255,255,.6);border-radius:6px;padding:8px 10px;margin:5px 0;white-space:pre-wrap;max-height:150px;overflow:auto}}
.pill.brokep{{color:#991b1b;background:#fee2e2}}
.callout{{background:#fff7ec;border:1px solid #fbe3bf;border-radius:12px;padding:16px 20px;font-size:13.5px;color:#7a4e12;margin:14px 0}}
.muted{{color:var(--slate)}}
.foot{{color:var(--slate);font-size:12px;margin-top:60px;border-top:1px solid var(--grid);padding-top:16px;font-family:"IBM Plex Mono",monospace;line-height:1.7}}
code{{font-family:"IBM Plex Mono";background:var(--washed);padding:1px 5px;border-radius:4px;font-size:.92em}}
@media(prefers-reduced-motion:reduce){{.particles{{display:none}}}}
</style></head><body><div class="wrap">

<p class="eyebrow">Refusal Recycling · behavioural measurement</p>
<h1>When a newer model fixes a jailbreak, does the fix survive a reword?</h1>
<p class="lede">A safety fix is only real if it recognises the <em>intent</em>, not the exact words.
We take the requests <b style="color:{COL['ink']}">{esc(new)}</b> refuses but
<b style="color:{COL['ink']}">{esc(old)}</b> answered, reword them while preserving meaning, and test
whether the newer model can be made to answer again.</p>
<p class="meta">pair <b>{esc(old)} → {esc(new)}</b> &nbsp;·&nbsp; generated {today}{(' &nbsp;·&nbsp; ' + esc(run_meta)) if run_meta else ''}</p>

<h2><span class="step-n">01</span> The journey of every request</h2>
<div class="panel">{flow}
<div class="legend">
<span><i style="background:{COL['broke']}"></i>answered (harmful content produced)</span>
<span><i style="background:{COL['held']}"></i>refused (safe)</span>
<span><i style="background:{COL['fix']}"></i>P⁻ — the fix we test</span>
<span><i style="background:{COL['regress']}"></i>regression</span></div></div>
<p class="q"><b>Read it left to right.</b> All requests split by what the <em>older</em> model did,
then by the <em>newer</em> model. The amber <b>P⁻</b> band — older answered, newer refused — is the
set of genuine fixes, which flow into the recycling test.</p>

<h2><span class="step-n">02</span> Did the fixes hold?</h2>
<div class="panel result">{gauge}
<div class="funnel">
<div class="fstep"><div class="fbar" style="background:{COL['fix']};width:100%">{pm}</div><div class="fcap">fixes to test (P⁻)</div></div>
<div class="fstep"><div class="fbar" style="background:{COL['slate']};width:{100 if n_tries else 6}%">{n_tries}</div><div class="fcap">rewordings generated</div></div>
<div class="fstep"><div class="fbar" style="background:{COL['held']};width:{int(100*n_admit/max(n_tries,1)) if n_tries else 4}%">{n_admit}</div><div class="fcap">kept the same meaning (passed the gate)</div></div>
</div></div>
{callout}
{tbar_block}

<h2><span class="step-n">{n4}</span> By source benchmark</h2>
<div class="panel"><table><tr><th class="bn">Benchmark</th><th class="num">P⁻ fixed</th><th>distribution</th><th class="num">screened</th></tr>
{brows}</table>
<div class="legend" style="margin-top:14px">
<span><i style="background:{COL['fix']}"></i>P⁻ fix</span><span><i style="background:{COL['broke']}"></i>both answered</span>
<span><i style="background:{COL['regress']}"></i>regression</span><span><i style="background:{COL['held']}"></i>both refused</span></div></div>

<h2><span class="step-n">{n5}</span> The fixed requests, in detail</h2>
<p class="q" style="font-size:13px"><span class="sens" style="margin:0">sensitive</span>&nbsp; Real prompt and response text, collapsed by default. Expand any case.</p>
{pmd if pmd else "<p class='muted'>None in this run.</p>"}

<div class="foot">Generated by recyclebench · {esc(old)} vs {esc(new)}<br>
Response labels come from an automated judge; small-sample results are exploratory. Sensitive content is collapsed and truncated by design. Band widths are proportional to counts.</div>
</div></body></html>"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(doc)
    return pm, n_tries, n_admit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/real_ollama.yaml")
    ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--behaviours", nargs="+", required=True)
    ap.add_argument("--manifest", nargs="*", default=None)
    ap.add_argument("--rsr", type=float, default=None)
    ap.add_argument("--recovered", type=int, default=None)
    ap.add_argument("--meta", default="")
    ap.add_argument("--out", default="results/dashboard.html")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config); k = cfg.budget.k_canonical_samples
    cache = load_cache(args.cache)
    cells, per_bench = partition(cache, args.behaviours, args.old, args.new, k)
    admitted, audit = load_manifest(args.manifest)
    rec = args.recovered
    if rec is None and args.rsr is not None:
        rec = round(args.rsr * len(cells["pminus"]))
    pm, tries, adm = render(cells, per_bench, admitted, audit, args.old, args.new, args.rsr, rec,
                            args.out, args.meta, _cache_ref=cache, _k_ref=k)
    print(f"wrote {args.out}")
    print(f"  P^-={pm}  paraphrase_tries={tries}  gated_admitted={adm}" + (f"  RSR={args.rsr}" if args.rsr is not None else ""))
    print(f"  open: open {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())