"""
tools/attack_transfer_section.py  --  dashboard section 01·g · known-attack transfer (Arm 3).

Reads one or more result JSONs written by tools/run_attack_transfer.py and renders a ranked
horizontal-bar view of each attack class's transfer-on-refused rate (the share of behaviours
the model refused canonically that the attack re-opens), with CIs and a ⚠ flag on the
low-readability cipher classes whose judge labels are pending manual verification. Scoped under
.atk-root, spliced before </body> like the other sections.

Public API (mirrors the others):
  aggregate(sources)            -> dict
  render_fragment(sources)      -> str
  inject(html_doc, sources)     -> str

`sources` is a path, a dict, or a list of them; class dicts are merged across files (later
files win on a class collision), so you can pass all three batch JSONs at once.
"""
from __future__ import annotations
import json
from pathlib import Path

# cipher / low-readability classes where the judge may miscount garbled output -> ⚠ until
# the outputs are read by eye. Overridable via aggregate(..., flag=...).
DEFAULT_FLAG = {"rot13", "disemvowel", "base64", "base64_raw", "base64_input_only",
                "base64_output_only", "obfuscation", "leetspeak"}


def _load_one(src):
    return src if isinstance(src, dict) else json.loads(Path(src).read_text())


def aggregate(sources, flag=None) -> dict:
    if sources is None:
        return {"ok": False, "classes": []}
    if isinstance(sources, (str, Path, dict)):
        sources = [sources]
    flag = DEFAULT_FLAG if flag is None else set(flag)

    merged = {}
    model = None
    n_ref = None
    judge = []
    baseline_class = None
    for s in sources:
        d = _load_one(s)
        model = d.get("model", model)
        n_ref = d.get("n_refused_baseline", n_ref)
        judge = d.get("judge", judge) or judge
        baseline_class = d.get("baseline_class", baseline_class)
        for cls, rec in (d.get("classes") or {}).items():
            merged[cls] = rec

    rows = []
    for cls, rec in merged.items():
        tr = rec.get("transfer_on_refused")
        rows.append({
            "cls": cls,
            "transfer": tr,
            "transfer_ci": rec.get("transfer_ci"),
            "n_broke_ref": rec.get("n_broke_among_refused"),
            "n_ref": rec.get("n_refused_baseline"),
            "bypass": rec.get("bypass_rate"),
            "flag": cls in flag,
        })
    # rank by transfer desc (None last)
    rows.sort(key=lambda r: (-1 if r["transfer"] is None else r["transfer"]), reverse=True)
    return {"ok": bool(rows), "classes": rows, "model": model or "model",
            "n_ref": n_ref, "judge": judge, "baseline_class": baseline_class,
            "any_flagged": any(r["flag"] and (r["transfer"] or 0) > 0 for r in rows)}


_CSS = """<style>
.atk-root{--ink:#14202E;--slate:#6B7A88;--grid:#E4E2DB;--paper:#F5F4F0;--panel:#FFFFFF;
  --bar:#2E5E8C;--bar2:#9BB4CC;--warn:#B8791F;--warnbg:#FBF0DC;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);
  background:var(--paper);padding:34px 26px 40px;border-top:1px solid var(--grid)}
.atk-root *{box-sizing:border-box}
.atk-wrap{max-width:980px;margin:0 auto}
.atk-root h2{font-family:"Space Grotesk",sans-serif;font-weight:600;font-size:21px;letter-spacing:-.01em;margin:0 0 4px}
.atk-root h2 .n{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--slate);font-weight:500;margin-right:10px}
.atk-sub{color:var(--slate);font-size:13px;margin:0 0 20px}
.atk-row{display:grid;grid-template-columns:168px 1fr 150px;gap:14px;align-items:center;padding:7px 0;border-bottom:1px solid var(--grid)}
.atk-name{font-family:"IBM Plex Mono",monospace;font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.atk-name .w{color:var(--warn);font-weight:700;margin-left:5px}
.atk-track{position:relative;height:16px;background:#ECEAE4;border-radius:4px;overflow:hidden}
.atk-fill{position:absolute;left:0;top:0;bottom:0;background:var(--bar);border-radius:4px}
.atk-fill.flag{background:repeating-linear-gradient(45deg,var(--warn),var(--warn) 5px,#CF9A47 5px,#CF9A47 10px)}
.atk-val{font-size:12.5px;text-align:right}
.atk-val b{font-family:"Space Grotesk",sans-serif;font-size:15px}
.atk-val .ci{color:var(--slate);font-size:11px;font-family:"IBM Plex Mono",monospace}
.atk-foot{color:var(--slate);font-size:12px;margin-top:16px;line-height:1.55}
.atk-foot .wk{color:var(--warn);font-weight:700}
.atk-chip{display:inline-block;background:var(--warnbg);color:var(--warn);font-size:11.5px;font-weight:600;
  padding:2px 9px;border-radius:999px;margin-left:8px}
</style>"""


def _pct(x):
    return f"{x:.1f}%" if isinstance(x, (int, float)) else "--"


def _body(a: dict) -> str:
    rows = a["classes"]
    scale = max([(r["transfer"] or 0) for r in rows] + [1.0])  # bar scale = max observed
    judge = ("judge " + "/".join(a["judge"])) if a.get("judge") else "judge uncalibrated"
    sub = (f"model <b>{a['model']}</b> · denominator = {a['n_ref']} behaviours refused "
           f"canonically (baseline {a.get('baseline_class','original_prompt')}) · {judge} · RAW")

    line = []
    for r in rows:
        tr = r["transfer"]
        w = max(1.5, 100 * (tr or 0) / scale)
        ci = ""
        if r["transfer_ci"]:
            ci = f'<span class="ci"> [{100*r["transfer_ci"][0]:.1f},{100*r["transfer_ci"][1]:.1f}]</span>'
        cnt = f'{r["n_broke_ref"]}/{r["n_ref"]}' if r["n_broke_ref"] is not None and r["n_ref"] else ""
        flagmark = '<span class="w">⚠</span>' if r["flag"] and (tr or 0) > 0 else ""
        line.append(
            f'<div class="atk-row"><div class="atk-name">{r["cls"]}{flagmark}</div>'
            f'<div class="atk-track"><div class="atk-fill{" flag" if r["flag"] and (tr or 0)>0 else ""}" '
            f'style="width:{w:.1f}%"></div></div>'
            f'<div class="atk-val"><b>{_pct(tr)}</b> <span class="ci">{cnt}</span>{ci}</div></div>')

    foot = ('<p class="atk-foot">Bars are <b>transfer-on-refused</b>: of the behaviours the model '
            'refused in plaintext, the share each attack re-opens. This arm is NOT intent-gated '
            '(gating an attack is a category error) and is distinct from the recycle arm (benign '
            'paraphrase). ')
    if a.get("any_flagged"):
        foot += ('<span class="wk">⚠ hatched</span> = low-readability cipher classes whose judge '
                 'labels are <b>pending manual verification</b>: the model may emit garbled output '
                 'the judge miscounts as compliance. Treat those as unconfirmed until the responses '
                 'are read by eye. ')
    foot += 'All rates RAW (judge-uncalibrated for this arm).</p>'

    chip = '<span class="atk-chip">⚠ 2 classes pending verification</span>' if a.get("any_flagged") else ""
    return (f'<section class="atk-root"><div class="atk-wrap">'
            f'<h2><span class="n">01·g</span>Known-attack transfer{chip}</h2>'
            f'<p class="atk-sub">{sub}</p>{"".join(line)}{foot}'
            f'</div></section>')


def render_fragment(sources, flag=None) -> str:
    a = aggregate(sources, flag=flag)
    return (_CSS + "\n" + _body(a)) if a.get("ok") else ""


def inject(html_doc: str, sources, flag=None) -> str:
    a = aggregate(sources, flag=flag)
    if not a.get("ok"):
        return html_doc
    frag = _CSS + "\n" + _body(a)
    i = html_doc.rfind("</body>")
    return (html_doc[:i] + frag + "\n" + html_doc[i:]) if i != -1 else html_doc + frag
