"""RPE convergent-validity section (01.c) + per-behaviour detail cards (01.d),
styled to match section 05 (uses the dashboard's own CSS classes). Reads
rpe_agreement.json + rpe_manifest_<bench>.jsonl from --rpe-out. Import-safe."""
from __future__ import annotations
import html, json
from pathlib import Path


def _esc(s):
    return html.escape(str(s if s is not None else ""))

def _pct(x):
    return "n/a" if x is None else f"{x*100:.0f}%"

def _load_agreement(out_dir):
    if not out_dir:
        return {}
    p = Path(out_dir) / "rpe_agreement.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _load_manifest(out_dir, bench):
    if not out_dir:
        return []
    p = Path(out_dir) / f"rpe_manifest_{bench}.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def _bar(label, val, n_txt, colour):
    w = 0 if val is None else int(val * 100)
    return ('<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
        '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
        'color:var(--slate);min-width:90px">' + _esc(label) + '</span>'
        '<span style="flex:1;max-width:340px;height:16px;border:1px solid var(--grid);'
        'border-radius:4px;position:relative;background:#fff">'
        '<span style="position:absolute;left:0;top:0;bottom:0;width:' + str(w) + '%;'
        'background:' + colour + ';border-radius:3px"></span></span>'
        '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
        'color:var(--ink);min-width:120px">' + _pct(val) +
        ' <span style="color:var(--slate)">' + _esc(n_txt) + '</span></span></div>')

def _score_str(scores):
    if not scores:
        return ""
    parts = []
    for k in ("cosine", "bertscore", "nli_entail", "llm_intent"):
        v = scores.get(k)
        if v is not None:
            parts.append(f"{k[:3]} {v:.2f}")
    return " &middot; ".join(parts)

def _score_line(scores):
    s = _score_str(scores)
    return (" &middot; gate: " + s) if s else ""


def _behaviour_card(row, old="older", new="newer"):
    """Mirror section 05's card: before (canonical request + old answer) then
    after (one .arow per RPE candidate). Uses the dashboard's own CSS classes."""
    rec = row.get("rpe_recovered")
    if rec == 1:
        outcome = '<span class="pill brokep">RPE RECOVERED it</span>'
    elif rec == 0:
        outcome = '<span class="pill">RPE held</span>'
    else:
        outcome = '<span class="pill">unresolved &middot; nothing gated</span>'

    details = row.get("details", [])
    n_c = row.get("n_candidates", len(details))
    n_a = row.get("n_admitted", sum(1 for d in details if d.get("admitted")))

    after_rows = ""
    for d in details:
        if not d.get("admitted"):
            continue
        broke = d.get("broke")
        cls = "arow broke" if broke else "arow held"
        verdict = "ANSWERED" if broke else "refused"
        after_rows += (
            '<div class="' + cls + '"><div class="atc">RPE candidate &middot; ' + verdict
            + _score_line(d.get("scores")) + '</div>'
            '<div class="atxt"><span class="tl">reworded request</span>'
            + _esc(d.get("text", ""))[:400] + '</div>'
            '<div class="atxt"><span class="tl">' + _esc(new) + ' replied</span>'
            + _esc(d.get("response") or "")[:400] + '</div></div>'
        )
    rejected = [d for d in details if not d.get("admitted")]
    rej_rows = ""
    for d in rejected:
        rej_rows += (
            '<div class="arow held" style="opacity:.7"><div class="atc">'
            'RPE candidate &middot; gate REJECTED' + _score_line(d.get("scores")) + '</div>'
            '<div class="atxt"><span class="tl">rejected request</span>'
            + _esc(d.get("text", ""))[:400] + '</div></div>'
        )
    after_inner = after_rows + rej_rows
    after_block = ('<div class="afterwrap"><div class="tl" style="margin:10px 0 4px">'
                   'RPE candidates (' + str(n_a) + ' gated of ' + str(n_c) + ' generated)</div>'
                   + after_inner + '</div>') if details else ""

    bench = _esc(row.get("benchmark", ""))
    bid = _esc(row.get("bid", ""))
    return (
        '<details class="pcard"><summary><span class="dsrc">' + bench + '</span> &middot; ' + bid +
        '<span class="sens">sensitive</span>' + outcome + '</summary>'
        '<div class="tl" style="margin:2px 0 4px">before &middot; canonical request</div>'
        '<div class="txt"><span class="tl">request</span>' + _esc(row.get("prompt", ""))[:600] + '</div>'
        '<div class="txt"><span class="tl">' + _esc(old) + ' &middot; answered</span>'
        + _esc(row.get("old_response", ""))[:600] + '</div>'
        + after_block + '</details>'
    )


def render_rpe_cards(bench_name, out_dir, section_no="01\u00b7d", old="older", new="newer"):
    rows = _load_manifest(out_dir, bench_name)
    if not rows:
        return ""
    order = {1: 0, 0: 1, None: 2}
    rows.sort(key=lambda r: order.get(r.get("rpe_recovered"), 3))
    cards = "".join(_behaviour_card(r, old=old, new=new) for r in rows)
    return ('<h2><span class="step-n">' + section_no + '</span> RPE requests, in detail</h2>'
            '<p class="q" style="margin:0 0 8px">Real prompt and response text, collapsed by default. '
            'Expand any case. Each shows the canonical request, the older model\'s answer, and every '
            'RPE-recovered candidate with its gate scores and the newer model\'s reply.</p>'
            '<div class="pcards">' + cards + '</div>')


def render_rpe_section(bench_name, out_dir, section_no="01\u00b7c"):
    agree = _load_agreement(out_dir)
    d = agree.get(bench_name)
    if d is None and bench_name == "all" and len(agree) == 1:
        d = next(iter(agree.values()))
    if d is None:
        return ('<h2><span class="step-n">' + section_no + '</span> Does an independent probe agree?</h2>'
            '<div class="panel"><p class="q" style="margin:0">No independent-probe run for '
            'this view yet.</p></div>')
    rsr_p = d.get("rsr_para"); rsr_r = d.get("rsr_rpe")
    n_total = d.get("n_total", 0); rpe_unres = d.get("rpe_unresolved", 0)
    both = d.get("n_both_resolved", 0); kappa = d.get("cohen_kappa")
    rpe_only = d.get("rpe_only", 0); para_only = d.get("para_only", 0)
    agr = d.get("agreement_rate")
    rpe_resolved = n_total - rpe_unres
    para_n = "(" + str(int(round((rsr_p or 0)*n_total))) + "/" + str(n_total) + ")" if rsr_p is not None else ""
    rpe_n = "(" + str(int(round((rsr_r or 0)*rpe_resolved))) + "/" + str(rpe_resolved) + " admitted)" if rsr_r is not None else ""
    kappa_txt = "n/a" if kappa is None else f"{kappa:.2f}"
    warn = ""
    if rpe_unres:
        warn = ('<div style="margin:8px 0 0;font-family:\'IBM Plex Mono\',monospace;'
            'font-size:12px;color:var(--broke)">&#9888; ' + str(rpe_unres) + '/' + str(n_total) +
            ' unresolved &mdash; the RPE MVP could not produce a gate-admitted candidate '
            '(measures inverter weakness, not fix robustness).</div>')
    banner = ('<div style="display:inline-block;margin:10px 0 0;padding:3px 10px;'
        'border:1px solid var(--grid);border-radius:20px;'
        'font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:var(--slate)">'
        'SMOKE &middot; MVP inverter &middot; not headline</div>')
    return ('<h2><span class="step-n">' + section_no + '</span> Does an independent probe agree?</h2>'
        '<div class="panel">'
        '<p class="q" style="margin:0 0 14px;font-size:13.5px">An independent method '
        '(<b>RPE inversion</b>) recovers candidate prompts from the <em>older</em> model\'s '
        'response and re-tests the newer model through the <b>same gate</b>. Agreement with '
        'the paraphrase arm means the fix\'s fragility isn\'t an artifact of one reword method.</p>'
        + _bar("paraphrase", rsr_p, para_n, "var(--broke)")
        + _bar("RPE (MVP)", rsr_r, rpe_n, "var(--fix)")
        + warn
        + '<div style="margin:14px 0 0;font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:var(--slate)">'
          'agreement (both resolved): <b style="color:var(--ink)">' + _pct(agr) + '</b> &middot; '
          '&kappa; <b style="color:var(--ink)">' + kappa_txt + '</b> &middot; '
          'resolved by both: <b style="color:var(--ink)">' + str(both) + '</b> &middot; '
          'RPE-only: <b style="color:var(--ink)">' + str(rpe_only) + '</b> &middot; '
          'para-only: <b style="color:var(--ink)">' + str(para_only) + '</b></div>'
        + banner + '</div>')
