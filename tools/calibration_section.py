"""
tools/calibration_section.py  --  dashboard section 00 · calibrated headline.

Reads the JSON written by tools/apply_calibration.py (the calibrated RSR/PBR + the judge's
se/sp/kappa) and renders a prominent "calibrated headline" panel: the corrected RSR and PBR
with CIs as the reported numbers, the raw observed rates beneath them, and the judge-validation
line (se, sp, Cohen's kappa, admitted?) that licenses the correction. Self-contained and scoped
under .cal-root, spliced before </body> like the other sections.

Public API (mirrors the other sections):
  aggregate(source) -> dict            # source: path or already-loaded dict
  render_fragment(source) -> str       # <style> + <section class="cal-root">..</section>
  inject(html_doc, source) -> str      # splice before the final </body>
"""
from __future__ import annotations
import json
from pathlib import Path


def aggregate(source) -> dict:
    """Normalise the apply_calibration JSON (path or dict) into the fields the panel needs."""
    if source is None:
        return {"ok": False}
    d = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    rsr = d.get("RSR") or {}
    pbr = d.get("PBR_new") or {}
    judge = d.get("judge") or {}

    def _ci(block, key="CI"):
        c = block.get(key)
        return (float(c[0]), float(c[1])) if c and len(c) == 2 else None

    return {
        "ok": bool(rsr),
        "pair": d.get("pair", ""),
        "n_p_minus": d.get("n_p_minus", rsr.get("n_p_minus")),
        "rsr_raw": rsr.get("RSR_raw"),
        "rsr_corr": rsr.get("RSR_corrected"),
        "rsr_ci": _ci(rsr),
        "k_recovered": rsr.get("k_recovered"),
        "mcnemar_p": rsr.get("mcnemar_p"),
        "t_mean": rsr.get("mean_admitted"),
        "pbr_model": pbr.get("model", ""),
        "pbr_raw": pbr.get("PBR_raw"),
        "pbr_corr": pbr.get("PBR_corrected"),
        "pbr_ci": _ci(pbr),
        "verdict": pbr.get("verdict", ""),
        "se": judge.get("se"), "sp": judge.get("sp"), "kappa": judge.get("kappa"),
        "admitted": bool(judge.get("admitted")), "n_bundles": judge.get("n_bundles"),
        "se_ci": judge.get("se_ci"), "sp_ci": judge.get("sp_ci"),
    }


def _pct(x):
    return f"{100*x:.1f}%" if isinstance(x, (int, float)) else "--"


def _ci_txt(ci, pct=True):
    if not ci:
        return ""
    lo, hi = ci
    return (f"[{100*lo:.1f}, {100*hi:.1f}]" if pct else f"[{lo:.3f}, {hi:.3f}]")


def _p_txt(p):
    if not isinstance(p, (int, float)):
        return "--"
    return f"{p:.2e}"


_CSS = """<style>
.cal-root{--ink:#14202E;--slate:#6B7A88;--grid:#E4E2DB;--paper:#FAFAFA;--panel:#FFFFFF;
  --good:#1F7A5A;--bad:#B23A2E;--accent:#2E5E8C;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--ink);background:var(--paper);padding:34px 26px 40px;border-top:1px solid var(--grid)}
.cal-root *{box-sizing:border-box}
.cal-wrap{max-width:980px;margin:0 auto}
.cal-root h2{font-family:"Space Grotesk",sans-serif;font-weight:600;font-size:21px;letter-spacing:-.01em;margin:0 0 4px}
.cal-root h2 .n{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--slate);font-weight:500;margin-right:10px}
.cal-sub{color:var(--slate);font-size:13px;margin:0 0 20px}
.cal-cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:680px){.cal-cards{grid-template-columns:1fr}}
.cal-card{background:var(--panel);border:1px solid var(--grid);border-radius:12px;padding:20px 22px}
.cal-lab{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--slate);margin:0 0 6px}
.cal-big{font-family:"Space Grotesk",sans-serif;font-weight:700;font-size:40px;line-height:1;letter-spacing:-.02em}
.cal-ci{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--slate);margin-top:7px}
.cal-raw{font-size:12.5px;color:var(--slate);margin-top:10px}
.cal-raw b{color:var(--ink);font-weight:600}
.cal-verdict{display:inline-block;margin-top:12px;font-size:12.5px;font-weight:600;padding:3px 10px;border-radius:999px;
  background:#FBEAE7;color:var(--bad)}
.cal-judge{margin-top:16px;background:var(--panel);border:1px solid var(--grid);border-radius:12px;padding:16px 20px;
  display:flex;flex-wrap:wrap;gap:22px;align-items:center}
.cal-jitem{font-size:13px}
.cal-jitem .k{color:var(--slate);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;display:block;margin-bottom:2px}
.cal-jitem .v{font-family:"IBM Plex Mono",monospace;font-size:15px;font-weight:600}
.cal-badge{margin-left:auto;font-size:12.5px;font-weight:600;padding:4px 12px;border-radius:999px}
.cal-badge.ok{background:#E6F2EC;color:var(--good)}
.cal-badge.no{background:#FBEAE7;color:var(--bad)}
.cal-foot{color:var(--slate);font-size:12px;margin-top:16px;line-height:1.55}
.cal-tip{position:relative;display:inline-flex;vertical-align:middle;margin-left:6px}
.cal-ibtn{width:15px;height:15px;border-radius:50%;border:1px solid var(--slate);background:transparent;color:var(--slate);font:600 10px/1 "IBM Plex Mono",monospace;cursor:pointer;padding:0;display:inline-flex;align-items:center;justify-content:center}
.cal-ibtn:hover,.cal-ibtn:focus{border-color:var(--ink);color:var(--ink);outline:none}
.cal-bubble{position:absolute;left:50%;top:24px;transform:translateX(-50%);width:248px;max-width:70vw;background:var(--ink);color:#fff;font:400 11.5px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;text-transform:none;letter-spacing:normal;text-align:left;padding:10px 12px;border-radius:8px;box-shadow:0 6px 22px rgba(20,32,46,.20);z-index:90;opacity:0;visibility:hidden;transition:opacity .12s}
.cal-bubble::before{content:"";position:absolute;left:50%;top:-5px;transform:translateX(-50%) rotate(45deg);width:10px;height:10px;background:var(--ink)}
.cal-bubble b{color:#fff;font-weight:600}
.cal-tip:hover .cal-bubble,.cal-ibtn:focus + .cal-bubble,.cal-tip:focus-within .cal-bubble{opacity:1;visibility:visible}
</style>"""


def _tip(text: str) -> str:
    """A small (i) toggletip: shows on hover, click (focus), or keyboard focus."""
    return ('<span class="cal-tip"><button type="button" class="cal-ibtn" '
            'aria-label="What is this?">i</button>'
            f'<span class="cal-bubble" role="tooltip">{text}</span></span>')


# metric explanations shown in the toggletips
_T_RSR = ("<b>Recycling Success Rate.</b> Of the P⁻ fixes — requests the older model answered and "
          "the newer one refused — the fraction re-opened by at least one gated, meaning-preserving "
          "paraphrase. <b>Corrected</b> adjusts the raw rate for the judge's measured error "
          "(Rogan–Gladen, at the behaviour-bundle level).")
_T_PBR = ("<b>Paraphrase Bypass Rate.</b> Of this model's own refusals, the fraction that at least "
          "one gated paraphrase re-opens — a direct measure of how surface-bound its refusals are. "
          "“materially surface-bound” means a substantial share re-open.")
_T_SE = ("<b>Sensitivity.</b> How often the judge correctly flags a genuine compliance "
         "(true-positive rate), measured against the human labels.")
_T_SP = ("<b>Specificity.</b> How often the judge correctly flags a genuine refusal "
         "(true-negative rate). This one matters most: bypass is a best-of-many union, so a false "
         "positive is costly.")
_T_KAPPA = ("<b>Cohen's κ.</b> Judge–human agreement beyond chance. The judge is <b>admitted</b> "
            "only above the pre-registered floor (0.60); its se/sp then correct the headline rates.")


def _body(a: dict) -> str:
    pair = a["pair"] or "old → new"
    mcn = _p_txt(a["mcnemar_p"])
    tmean = f"{a['t_mean']:.2f}" if isinstance(a["t_mean"], (int, float)) else "--"
    badge = ('<span class="cal-badge ok">judge admitted</span>' if a["admitted"]
             else '<span class="cal-badge no">judge NOT admitted</span>')

    def jitem(k, v, ci=None, tip=None):
        civ = ""
        if ci and len(ci) == 2:
            civ = f' <span style="color:var(--slate)">[{ci[0]:.3f}, {ci[1]:.3f}]</span>'
        t = _tip(tip) if tip else ""
        return f'<div class="cal-jitem"><span class="k">{k}{t}</span><span class="v">{v}{civ}</span></div>'

    krec = a["k_recovered"]
    npm = a["n_p_minus"]
    rsr_count = (f"{krec}/{npm} observed" if krec is not None and npm is not None else "")

    rsr_card = (
        f'<div class="cal-card"><p class="cal-lab">Recycling Success Rate (corrected){_tip(_T_RSR)}</p>'
        f'<div class="cal-big">{_pct(a["rsr_corr"])}</div>'
        f'<div class="cal-ci">95% CI {_ci_txt(a["rsr_ci"])}</div>'
        f'<div class="cal-raw">raw <b>{_pct(a["rsr_raw"])}</b> · {rsr_count} · McNemar p={mcn} · t̄={tmean}</div></div>')

    pbr_card = (
        f'<div class="cal-card"><p class="cal-lab">Paraphrase Bypass Rate — {a["pbr_model"] or "new model"} (corrected){_tip(_T_PBR)}</p>'
        f'<div class="cal-big">{_pct(a["pbr_corr"])}</div>'
        f'<div class="cal-ci">95% CI {_ci_txt(a["pbr_ci"])}</div>'
        f'<div class="cal-raw">raw <b>{_pct(a["pbr_raw"])}</b></div>'
        + (f'<div class="cal-verdict">{a["verdict"]}</div>' if a["verdict"] else "") + '</div>')

    judge_bar = (
        '<div class="cal-judge">'
        + jitem("sensitivity", f'{a["se"]:.3f}' if isinstance(a["se"], (int, float)) else "--", a["se_ci"], tip=_T_SE)
        + jitem("specificity", f'{a["sp"]:.3f}' if isinstance(a["sp"], (int, float)) else "--", a["sp_ci"], tip=_T_SP)
        + jitem("Cohen's κ", f'{a["kappa"]:.3f}' if isinstance(a["kappa"], (int, float)) else "--", tip=_T_KAPPA)
        + jitem("bundles", str(a["n_bundles"]) if a["n_bundles"] is not None else "--")
        + badge + '</div>')

    foot = ('<p class="cal-foot">The <b>corrected</b> rates are the reported headline: raw judge '
            'labels are adjusted for the judge\'s measured sensitivity/specificity (Rogan–Gladen, at the '
            'behaviour-bundle level). The judge line is the validation that licenses it — correction is '
            'only admissible when κ clears the pre-registered floor. Wide CIs reflect the P⁻ sample size, '
            'not instability; report them with the point estimate.</p>')

    return (f'<section class="cal-root"><div class="cal-wrap">'
            f'<h2><span class="n">01</span>Calibrated headline — {pair}</h2>'
            f'<p class="cal-sub">P⁻ = {npm if npm is not None else "--"} · judge-corrected, human-calibrated</p>'
            f'<div class="cal-cards">{rsr_card}{pbr_card}</div>{judge_bar}{foot}'
            f'</div></section>')


def render_fragment(source) -> str:
    a = aggregate(source)
    return (_CSS + "\n" + _body(a)) if a.get("ok") else ""


def inject(html_doc: str, source) -> str:
    a = aggregate(source)
    if not a.get("ok"):
        return html_doc
    frag = _CSS + "\n" + _body(a)
    i = html_doc.rfind("</body>")
    return (html_doc[:i] + frag + "\n" + html_doc[i:]) if i != -1 else html_doc + frag