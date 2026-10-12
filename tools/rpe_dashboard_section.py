"""
rpe_dashboard_section.py -- Section 01.e of the RecycleBench dashboard:
"Independent probe (inversion)" -- the RPE arm's per-behaviour log.

Reads an RPE manifest (rpe_manifest_<benchmark>.jsonl, one JSON record per P- behaviour)
and renders, per behaviour: the canonical request, the OLD output that was inverted, the
reconstructed candidate requests with their gate scores, and llama-new's reply to the
admitted ones -- plus a summary bar. Self-contained and scoped under .rpe-root, so it opens
standalone AND drops into make_dashboard like style_arcs_section.

Manifest record schema (per line):
  {bid, benchmark, rpe_recovered (1|0|null), para_recovered, n_candidates, n_admitted,
   admitted:[...], prompt, old_response,
   details:[{text, admitted, scores:{cosine,bertscore,nli_entail,llm_intent}, response, broke}]}

Outcome:
  rpe_recovered == 1   -> RECOVERED (an intent-admitted inversion re-elicited compliance)
  rpe_recovered == 0   -> HELD      (admitted inversion, but new model still refused)
  rpe_recovered is None -> UNRESOLVED (no candidate passed the intent gate)

API (mirrors style_arcs_section / positioning_section):
  render_fragment(source) -> str    # <style> + <section class="rpe-root">..</section>
  render_page(source)     -> str    # standalone HTML document
  inject(html_doc, source)-> str    # splice before the final </body>
  source = manifest path | list of records

Run standalone:
  python3 tools/rpe_dashboard_section.py results/runs/rpe_3_31_sorrybench/rpe_panel.html/rpe_manifest_sorrybench.jsonl
"""
from __future__ import annotations

import html as _html
import json as _json
from pathlib import Path

__all__ = ["aggregate", "render_fragment", "render_page", "inject"]

_INK = "#14202E"; _SLATE = "#6B7A88"; _GRID = "#E4E2DB"; _PAPER = "#F5F4F0"; _PANEL = "#FFFFFF"
_HELD = "#107C6E"; _BROKE = "#D8452B"; _FIX = "#E3A008"


def _records(source):
    if isinstance(source, (list, tuple)):
        return list(source)
    recs = []
    for line in Path(source).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(_json.loads(line))
            except Exception:
                pass
    return recs


def aggregate(source) -> dict:
    recs = _records(source)
    recovered = sum(1 for r in recs if r.get("rpe_recovered") == 1)
    held = sum(1 for r in recs if r.get("rpe_recovered") == 0)
    unresolved = sum(1 for r in recs if r.get("rpe_recovered") is None)
    n = len(recs)
    cand = sum(int(r.get("n_candidates") or 0) for r in recs)
    adm = sum(int(r.get("n_admitted") or 0) for r in recs)
    resolved = recovered + held
    return {
        "records": recs, "n": n, "recovered": recovered, "held": held,
        "unresolved": unresolved, "resolved": resolved,
        "rsr_resolved": (recovered / resolved) if resolved else None,
        "rsr_overall": (recovered / n) if n else None,
        "cand": cand, "admitted": adm,
        "admit_rate": (adm / cand) if cand else 0.0,
    }


def _esc(s, n=None):
    s = "" if s is None else str(s)
    if n is not None and len(s) > n:
        s = s[:n] + "…"
    return _html.escape(s)


def _score_pills(sc):
    if not isinstance(sc, dict):
        return ""
    order = [("cos", "cosine"), ("bert", "bertscore"), ("nli", "nli_entail"), ("intent", "llm_intent")]
    out = []
    for lab, key in order:
        v = sc.get(key)
        if isinstance(v, (int, float)):
            # NLI is the gate's decisive axis -> tint by value
            cls = "rpe-sc"
            if key == "nli_entail":
                cls += " rpe-sc-nli-hi" if v >= 0.5 else " rpe-sc-nli-lo"
            out.append(f'<span class="{cls}">{lab} {v:.2f}</span>')
    return "".join(out)


def _card(r):
    rec = r.get("rpe_recovered")
    if rec == 1:
        badge, bcls = "recovered", "rpe-b-rec"
    elif rec == 0:
        badge, bcls = "held", "rpe-b-held"
    else:
        badge, bcls = "unresolved", "rpe-b-unres"
    bid = _esc(r.get("bid"))
    n_adm = int(r.get("n_admitted") or 0); n_cand = int(r.get("n_candidates") or 0)

    cand_rows = []
    for d in (r.get("details") or []):
        adm = bool(d.get("admitted"))
        mark = '<span class="rpe-adm rpe-adm-y">admitted</span>' if adm else '<span class="rpe-adm rpe-adm-n">rejected</span>'
        reply = ""
        if adm and d.get("response") is not None:
            broke = d.get("broke")
            rcls = "rpe-reply-broke" if broke else "rpe-reply-held"
            rlab = "new model complied" if broke else "new model refused"
            reply = (f'<div class="rpe-reply {rcls}"><span class="rpe-tl">{rlab}</span>'
                     f'{_esc(d.get("response"), 400)}</div>')
        cand_rows.append(
            f'<div class="rpe-cand{" rpe-cand-adm" if adm else ""}">'
            f'<div class="rpe-cand-top">{mark}{_score_pills(d.get("scores"))}</div>'
            f'<div class="rpe-cand-txt">{_esc(d.get("text"), 300)}</div>{reply}</div>')

    return (
        f'<div class="rpe-card">'
        f'<div class="rpe-card-hd"><span class="rpe-bid">{bid}</span>'
        f'<span class="rpe-badge {bcls}">{badge}</span>'
        f'<span class="rpe-adm-count">{n_adm}/{n_cand} admitted</span></div>'
        f'<div class="rpe-step"><span class="rpe-tl">① canonical request</span>'
        f'<div class="rpe-txt">{_esc(r.get("prompt"), 500)}</div></div>'
        f'<div class="rpe-step"><span class="rpe-tl">② old output (inverted)</span>'
        f'<div class="rpe-txt rpe-old">{_esc(r.get("old_response"), 500)}</div></div>'
        f'<div class="rpe-step"><span class="rpe-tl">③ reconstructed candidates → gate</span>'
        f'{"".join(cand_rows) or "<div class=rpe-txt>—</div>"}</div>'
        f'</div>')


_CSS = """<style>
.rpe-root{--ink:#14202E;--slate:#6B7A88;--grid:#E4E2DB;--paper:#F5F4F0;--panel:#FFFFFF;
  --held:#107C6E;--broke:#D8452B;--fix:#E3A008;
  background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.5}
.rpe-root *{box-sizing:border-box}
.rpe-wrap{max-width:1040px;margin:0 auto;padding:10px 28px 40px}
.rpe-root h2{font-family:"Space Grotesk",sans-serif;font-weight:600;font-size:21px;letter-spacing:-.01em;
  margin:40px 0 4px;padding-top:26px;border-top:1px solid var(--grid)}
.rpe-root h2 .n{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--slate);font-weight:500;margin-right:10px}
.rpe-sub{color:var(--slate);font-size:14px;margin:4px 0 14px;max-width:76ch}
.rpe-sub b{color:var(--ink)}
.rpe-tiles{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:0 0 16px}
.rpe-tile{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:12px 14px}
.rpe-tile .k{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--slate);margin:0 0 6px}
.rpe-tile .v{font-family:"Space Grotesk",sans-serif;font-weight:700;font-size:22px;line-height:1}
.rpe-tile .d{font-size:11px;color:var(--slate);margin-top:3px}
.rpe-note{background:#FBEFD2;border:1px solid #f0d89a;border-radius:10px;padding:10px 14px;font-size:12.5px;color:#7a5600;margin:0 0 16px}
.rpe-card{background:var(--panel);border:1px solid var(--grid);border-radius:12px;padding:14px 16px;margin:0 0 12px}
.rpe-card-hd{display:flex;align-items:center;gap:10px;margin:0 0 8px}
.rpe-bid{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:13px}
.rpe-badge{font-family:"IBM Plex Mono",monospace;font-size:10.5px;padding:2px 9px;border-radius:999px}
.rpe-b-rec{background:#F4DdD7;color:#A1321E}
.rpe-b-held{background:#Dff0ec;color:#0C6657}
.rpe-b-unres{background:#EDEDE9;color:#6B7A88}
.rpe-adm-count{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--slate)}
.rpe-step{margin:8px 0}
.rpe-tl{display:block;font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--slate);margin:0 0 3px}
.rpe-txt{background:var(--paper);border:1px solid var(--grid);border-radius:7px;padding:8px 10px;font-size:13px;white-space:pre-wrap;word-break:break-word}
.rpe-old{color:#55636f}
.rpe-cand{border:1px solid var(--grid);border-radius:7px;padding:7px 9px;margin:5px 0;background:var(--paper)}
.rpe-cand-adm{border-color:#Bfe0d9;background:#F1F8F6}
.rpe-cand-top{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 4px}
.rpe-cand-txt{font-size:12.5px;white-space:pre-wrap;word-break:break-word}
.rpe-adm{font-family:"IBM Plex Mono",monospace;font-size:10px;padding:1px 7px;border-radius:999px}
.rpe-adm-y{background:#Dff0ec;color:#0C6657}
.rpe-adm-n{background:#EDEDE9;color:#6B7A88}
.rpe-sc{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--slate);background:var(--panel);border:1px solid var(--grid);border-radius:999px;padding:1px 7px}
.rpe-sc-nli-hi{color:#0C6657;border-color:#Bfe0d9}
.rpe-sc-nli-lo{color:#A1321E;border-color:#e7c3bb}
.rpe-reply{margin-top:5px;border-radius:7px;padding:7px 9px;font-size:12.5px;white-space:pre-wrap;word-break:break-word}
.rpe-reply-broke{background:#FBE4DF;border:1px solid #e7c3bb}
.rpe-reply-held{background:#E7F2F0;border:1px solid #Bfe0d9}
</style>"""


def _body(agg):
    rsr_res = agg["rsr_resolved"]
    rsr_res_s = f'{rsr_res*100:.0f}%' if rsr_res is not None else '—'
    rsr_all_s = f'{agg["rsr_overall"]*100:.0f}%' if agg["rsr_overall"] is not None else '—'
    tiles = (
        '<div class="rpe-tiles">'
        f'<div class="rpe-tile"><p class="k">P⁻ probed</p><p class="v">{agg["n"]}</p></div>'
        f'<div class="rpe-tile"><p class="k">recovered</p><p class="v" style="color:{_BROKE}">{agg["recovered"]}</p><p class="d">intent-admitted inversion re-elicited</p></div>'
        f'<div class="rpe-tile"><p class="k">held</p><p class="v" style="color:{_HELD}">{agg["held"]}</p></div>'
        f'<div class="rpe-tile"><p class="k">unresolved</p><p class="v" style="color:{_SLATE}">{agg["unresolved"]}</p><p class="d">no candidate passed the intent gate</p></div>'
        f'<div class="rpe-tile"><p class="k">RSR·rpe</p><p class="v">{rsr_res_s}</p><p class="d">{agg["recovered"]}/{agg["resolved"]} resolved · {rsr_all_s} of all P⁻</p></div>'
        '</div>')
    note = ('<div class="rpe-note"><b>Independent validator, RAW.</b> The inverter (decorrelated) '
            'reconstructs a request from the old model’s output; the intent gate admits only '
            'intent-preserving reconstructions, then the new model is re-queried. '
            f'Here {agg["unresolved"]}/{agg["n"]} behaviours had <b>no</b> candidate pass the gate — inversion could not '
            'reconstruct an intent-preserving request from the output (the entailment bound). '
            'Verdicts are judge-uncorrected; cross-arm agreement vs the paraphrase RSR is reported separately.</div>')
    cards = "".join(_card(r) for r in agg["records"])
    return (
        '<section class="rpe-root"><div class="rpe-wrap">'
        '<h2><span class="n">01·e</span>Independent probe — inversion (RPE)</h2>'
        '<p class="rpe-sub">One card per P⁻ behaviour. The RPE arm <b>inverts the old model’s output</b> '
        'to reconstruct the request (a decorrelated prober, not a paraphraser), gates it for intent, and '
        're-queries the new model — an independent check that the recycling result isn’t a paraphraser artifact.</p>'
        f'{tiles}{note}{cards}'
        '</div></section>')


_HEAD = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
    '<title>RPE independent probe</title>\n'
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700'
    '&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">\n')


def render_fragment(source) -> str:
    return _CSS + "\n" + _body(aggregate(source))


def render_page(source) -> str:
    return _HEAD + _CSS + "</head>\n<body>\n" + _body(aggregate(source)) + "\n</body></html>\n"


def inject(html_doc: str, source) -> str:
    agg = aggregate(source)
    if not agg["records"]:
        return html_doc
    frag = _CSS + "\n" + _body(agg)
    i = html_doc.rfind("</body>")
    return (html_doc[:i] + frag + "\n" + html_doc[i:]) if i != -1 else html_doc + frag


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: rpe_dashboard_section.py <rpe_manifest.jsonl> [out.html]")
    out = sys.argv[2] if len(sys.argv) > 2 else "rpe_section.html"
    Path(out).write_text(render_page(sys.argv[1]), encoding="utf-8")
    a = aggregate(sys.argv[1])
    print(f"wrote {out} | {a['n']} probed · {a['recovered']} recovered · "
          f"{a['held']} held · {a['unresolved']} unresolved")
