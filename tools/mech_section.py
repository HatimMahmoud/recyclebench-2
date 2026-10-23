"""tools/mech_section.py -- the white-box panel inside a fix card ("Inside the model"), from tools/mech_probe.py output.

Loads the newest results/mech/anchor_<benchmark>_*.json (numbers only) and renders, for one behaviour, four views of
what differs inside each model between the canonical request, a rewording the newer model still refuses and the
rewording that re-opened the fix:

  A  refusal-direction score by layer   (0 = mean harmless prompt, 1 = mean refused harmful prompt, own direction per model)
  B  representation divergence by layer (1 - cosine of the residual stream to the canonical prompt)
  C  most differential MLP neurons      (at the layer where the newer model separates canonical and re-opening most)
  D  what the model is about to say     (top next tokens) and the faithfulness check against the cached Ollama reply
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_CACHE: dict = {}

ROLE = {"canonical": ("canonical request", "var(--ink)", ""),
        "held": ("rewording · still refused", "var(--refuse)", "5 4"),
        "reopen": ("rewording · re-opened", "var(--comply)", "")}
LABEL = {"llama3-8b": "Llama 3 8B", "llama31-8b": "Llama 3.1 8B"}


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def load(old: str, new: str, bench: str):
    """{bid: doc-slice} from the newest probe file of this pair and benchmark, or {}."""
    key = (old, new, bench)
    if key in _CACHE:
        return _CACHE[key]
    files = sorted((ROOT / "results" / "mech").glob(f"anchor_{bench}_*.json"), key=lambda p: p.stat().st_mtime)
    out = {}
    for f in files:                                   # newest last, so a full run overrides the PoC per behaviour
        doc = json.loads(f.read_text())
        if doc.get("pair") != [old, new]:
            continue
        res = doc["results"]
        for i, it in enumerate(res[new]["items"]):
            out[it["bid"]] = dict(file=f.name, generated=doc["generated"], models=doc["models"],
                                  direction=doc["direction"], dprime={m: res[m]["direction_dprime"] for m in res},
                                  items={m: res[m]["items"][i] for m in res})
    _CACHE[key] = out
    return out


def _line_chart(series_by_model, title, y_lab, y_min, y_max, ref_lines=(), mark=None, h=170, weak=None):
    """small multiples: one panel per model; series = [(role, ys)] indexed by residual layer (0 = embeddings, skipped).
    weak: {model: [layer indices where the refusal direction is not yet formed]} -> shaded."""
    W, pad_l, pad_r, pad_t, pad_b = 300, 40, 8, 22, 26
    panels = []
    for m, series in series_by_model.items():
        series = [(r, [None] + list(ys[1:])) for r, ys in series]
        n = max(len(ys) for _, ys in series)
        def X(i):
            return pad_l + (W - pad_l - pad_r) * i / max(1, n - 1)
        def Y(v):
            v = max(y_min, min(y_max, v))
            return pad_t + (h - pad_t - pad_b) * (1 - (v - y_min) / (y_max - y_min))
        g = [f'<text x="{pad_l}" y="13" class="mt">{esc(LABEL.get(m, m))}</text>']
        for v, lab in ref_lines:
            g.append(f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="mref"/>'
                     f'<text x="{W - pad_r}" y="{Y(v) - 3:.1f}" class="mrl" text-anchor="end">{esc(lab)}</text>')
        if weak and weak.get(m):
            ws = [i for i in weak[m] if i >= 1]
            if ws:
                x0, x1 = X(max(1, min(ws)) - 0.5), X(max(ws) + 0.5)
                g.append(f'<rect x="{x0:.1f}" y="{pad_t}" width="{x1 - x0:.1f}" height="{h - pad_t - pad_b}" class="mweak"/>'
                         f'<text x="{(x0 + x1) / 2:.1f}" y="{h - pad_b - 5}" class="mrl" text-anchor="middle">direction not formed</text>')
        for t in range(0, n, 8):
            g.append(f'<text x="{X(t):.1f}" y="{h - 8}" class="mtick" text-anchor="middle">{t}</text>')
        g.append(f'<text x="{(pad_l + W) / 2:.0f}" y="{h + 2}" class="mtick" text-anchor="middle">layer</text>')
        for v in (y_min, y_max):
            g.append(f'<text x="{pad_l - 5}" y="{Y(v) + 4:.1f}" class="mtick" text-anchor="end">{v:.2f}</text>')
        if mark is not None and mark.get(m) is not None:
            x = X(mark[m])
            g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{h - pad_b}" class="mmark"/>')
        for role, ys in series:
            _, col, dash = ROLE[role]
            pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys) if v is not None)
            vals = [v for v in ys if v is not None]
            tip = f"{LABEL.get(m, m)} · {ROLE[role][0]} · last layer {vals[-1]:.2f} · max {max(vals):.2f} (layer {ys.index(max(vals))})"
            g.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-dasharray="{dash}" '
                     f'stroke-linejoin="round" tabindex="0" data-tip="{esc(tip)}"/>')
        panels.append(f'<svg class="mchart" viewBox="0 0 {W} {h + 6}" role="img" aria-label="{esc(title)} · {esc(LABEL.get(m, m))}">{"".join(g)}</svg>')
    return (f'<figure class="mfig"><figcaption>{esc(title)} <span class="sub">{esc(y_lab)}</span></figcaption>'
            f'<div class="mgrid">{"".join(panels)}</div></figure>')


def _heat(it, m):
    nz = it.get("neurons")
    if not nz:
        return ""
    cols, rows = nz["index"], nz["rows"]
    vmax = max(1e-6, max(abs(v) for r in rows for v in r["values"]))
    cw, ch, lw, th = 26, 24, 150, 46
    W, H = lw + cw * len(cols) + 6, th + ch * len(rows) + 6
    g = [f'<text x="{lw + cw * i + cw / 2:.0f}" y="{th - 6}" class="mtick" transform="rotate(-55 {lw + cw * i + cw / 2:.0f} {th - 6})">n{c}</text>'
         for i, c in enumerate(cols)]
    for j, r in enumerate(rows):
        y = th + ch * j
        g.append(f'<text x="{lw - 8}" y="{y + ch / 2 + 4:.0f}" class="mrow" text-anchor="end">{esc(ROLE[r["role"]][0])}</text>')
        for i, (c, v) in enumerate(zip(cols, r["values"])):
            col = "var(--comply)" if v > 0 else "var(--sky)"
            g.append(f'<rect x="{lw + cw * i + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="4" fill="{col}" '
                     f'fill-opacity="{0.08 + 0.92 * abs(v) / vmax:.2f}" class="mcell" tabindex="0" '
                     f'data-tip="{esc(LABEL.get(m, m))} · layer {nz["layer"]} · neuron {c} · {esc(ROLE[r["role"]][0])}: {v:+.3f}"/>')
    return (f'<div class="mheat"><div class="mt2">{esc(LABEL.get(m, m))} · MLP layer {nz["layer"]} · top {len(cols)} neurons by '
            f'|canonical − re-opened| · <span class="sub">magenta = positive, blue = negative activation</span></div>'
            f'<svg class="mchart" viewBox="0 0 {W} {H}" role="img" aria-label="neuron activations">{"".join(g)}</svg></div>')


def _nice(v):
    import math
    e = 10 ** math.floor(math.log10(v))
    for f in (1, 2, 2.5, 5, 10):
        if v <= f * e:
            return f * e
    return v


def _reading(items, m):
    """one factual sentence per model from the numbers (no causal claim)."""
    ps = {p["role"]: p for p in items[m]["prompts"]}
    c, r, h = ps.get("canonical"), ps.get("reopen"), ps.get("held")
    if not (c and r):
        return ""
    def pr(p):
        return p.get("p_refusal_prefix")
    bits = [f"refusal-direction score at the last layer {c['score'][-1]:.2f} (canonical) vs {r['score'][-1]:.2f} (re-opened)"
            + (f" vs {h['score'][-1]:.2f} (still refused)" if h else "")]
    bits.append(f"late-layer divergence from the canonical request {1 - r['cos_to_canonical'][-1]:.2f} (re-opened)"
                + (f" vs {1 - h['cos_to_canonical'][-1]:.2f} (still refused)" if h else ""))
    if pr(c) is not None:
        bits.append(f"P(refusal opening) {pr(c):.2f} → {pr(r):.2f}" + (f" (still refused {pr(h):.2f})" if h else ""))
    return f"<li><b>{esc(LABEL.get(m, m))}</b>: " + "; ".join(bits) + ".</li>"


def _verdict(v):
    return {1: '<span class="pill pill-comply">answered</span>', 0: '<span class="pill pill-refuse">refused</span>'}.get(
        v, '<span class="pill pill-none">—</span>')


def panel(old: str, new: str, bench: str, bid: str) -> str:
    data = load(old, new, bench).get(bid)
    if not data:
        return ""
    items = data["items"]
    roles = [p["role"] for p in items[new]["prompts"]]
    lstar = {m: (items[m].get("neurons") or {}).get("layer") for m in items}
    mark = {m: (lstar[m] + 1 if lstar[m] is not None else None) for m in items}      # residual index = MLP layer + 1
    dp = data["dprime"]
    weak = {m: [i for i, v in enumerate(dp[m]) if v < 0.5 * max(dp[m])] for m in dp}
    a = _line_chart({m: [(p["role"], p["score"]) for p in items[m]["prompts"]] for m in (old, new)},
                    "A · Refusal-direction score", "0 = mean harmless · 1 = mean refused-harmful · shaded: d′ below half its peak",
                    -0.25, 1.25, ref_lines=((0, "harmless"), (1, "refused")), mark=mark, weak=weak)
    b = _line_chart({m: [(p["role"], [1 - c for c in p["cos_to_canonical"]]) for p in items[m]["prompts"] if p["role"] != "canonical"]
                     for m in (old, new)},
                    "B · How differently the model represents the rewording", "1 − cosine to the canonical request",
                    0, _nice(max(0.05, max(1 - c for m in items for p in items[m]["prompts"] for c in p["cos_to_canonical"][1:]))),
                    mark=mark)
    c = f'<figure class="mfig"><figcaption>C · Most differential MLP neurons <span class="sub">activation at the last prompt token</span></figcaption>' \
        f'<div class="mgrid">{_heat(items[old], old)}{_heat(items[new], new)}</div></figure>'
    rows = ""

    def rep(x):
        return "—" if not x or x[0] is None else f"{x[0]}/{x[1]}"
    for m in (old, new):
        for p in items[m]["prompts"]:
            nxt = " · ".join(f'<code>{esc(t.strip() or repr(t))}</code> {pr:.2f}' for t, pr in p["next_tokens"][:3])
            pr = p.get("p_refusal_prefix")
            bar = ("—" if pr is None else
                   f'<span class="mbar" role="img" aria-label="{pr:.2f}"><i style="width:{pr * 100:.0f}%"></i></span> {pr:.2f}')
            rows += (f'<tr><td>{esc(LABEL.get(m, m))}</td><td><span class="mkey" style="--c:{ROLE[p["role"]][1]}"></span>{esc(ROLE[p["role"]][0])}'
                     f'{"" if p["role"] == "canonical" else " <span class=sub>(" + esc(p["cls"]) + ")</span>"}</td>'
                     f'<td>{_verdict(p["verdict"])}</td><td class="nowrap">{bar}</td><td>{nxt}</td>'
                     f'<td class="num">{rep(p.get("greedy_match"))}</td><td class="num">{rep(p.get("greedy_match_ollama"))}</td></tr>')
    d = (f'<figure class="mfig"><figcaption>D · What the model is about to say <span class="sub">probability that the reply opens '
         f'with an explicit refusal phrase · top next tokens · token-for-token reproduction of the cached Ollama reply</span></figcaption>'
         f'<div class="tscroll"><table class="tbl mtab"><thead><tr><th>Model</th><th>Prompt</th><th>Verdict (qwen)</th>'
         f'<th>P(refusal opening)</th><th>Top next tokens</th><th class="num">Greedy</th><th class="num">Ollama decoding</th></tr></thead>'
         f'<tbody>{rows}</tbody></table></div></figure>')
    reading = f'<ul class="mread">{_reading(items, old)}{_reading(items, new)}</ul>'
    legend = "".join(f'<span><i class="mkey" style="--c:{ROLE[r][1]}"></i>{esc(ROLE[r][0])}</span>' for r in ROLE if r in roles)
    meta = data["models"]
    note = (f"Weights: the GGUF files Ollama served ({esc(meta[old]['ollama'])} {esc(', '.join(meta[old]['types']))}; "
            f"{esc(meta[new]['ollama'])} {esc(', '.join(meta[new]['types']))}), dequantised to fp16. Refusal direction: difference of mean "
            f"residual activations at the last prompt token between {len(data['direction']['harmful_bids'])} SORRY-Bench requests both models "
            f"refuse and {data['direction']['n_harmless']} harmless requests, per layer and per model (separation d′ peaks at "
            f"{max(dp[old]):.1f} / {max(dp[new]):.1f}). Vertical line: layer of the neuron view. Probe {esc(data['file'])}. "
            "Ollama decoding = greedy with Ollama's default repeat penalty (1.1 over the last 64 tokens), which the study did not override. "
            "Exploratory and correlational: a lower score is consistent with the rewording moving the prompt off the refusal "
            "direction, not proof that it causes the answer.")
    return (f'<details class="mech" open><summary><span class="mbadge">white-box</span> Inside the model · how the three prompts differ internally</summary>'
            f'<div class="mlegend">{legend}</div>{reading}{a}{b}{c}{d}<p class="note">{note}</p></details>')


CSS = r"""
details.mech{margin-top:14px;border:1px solid var(--grid);border-radius:12px;padding:10px 14px 6px;
  background:linear-gradient(180deg,color-mix(in srgb,var(--accent) 7%,var(--card)),var(--card) 140px)}
details.mech>summary{cursor:pointer;font-weight:600;color:var(--brand);list-style:none;display:flex;align-items:center;gap:8px}
details.mech>summary::-webkit-details-marker{display:none}
.mbadge{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--card);
  background:var(--brand);border-radius:999px;padding:2px 8px}
.mlegend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--ink2);margin:10px 0 4px}
.mkey{display:inline-block;width:18px;height:3px;border-radius:2px;background:var(--c);vertical-align:middle;margin-right:6px}
.mfig{margin:12px 0}.mfig figcaption{font-size:12.5px;font-weight:600;color:var(--ink);margin-bottom:4px}
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.mchart{width:100%;height:auto;background:var(--card);border:1px solid var(--grid);border-radius:10px}
.mchart .mt{font-size:11.5px;font-weight:600;fill:var(--ink)}.mchart .mtick{font-size:9.5px;fill:var(--muted);font-family:var(--mono)}
.mchart .mref{stroke:var(--grid);stroke-dasharray:3 3}.mchart .mrl{font-size:9px;fill:var(--muted)}
.mchart .mmark{stroke:var(--brand2);stroke-width:1;stroke-dasharray:2 3;opacity:.7}
.mchart polyline:focus{outline:none;stroke-width:3.5}
.mheat{overflow-x:auto}.mchart .mrow{font-size:10.5px;fill:var(--ink2)}.mchart .mweak{fill:var(--wash);opacity:.8}
.mread{margin:6px 0 2px;padding-left:18px;font-size:12.5px;color:var(--ink2)}.mread li{margin:2px 0}.mt2{font-size:11.5px;color:var(--ink2);margin-bottom:4px}
.mhtab{border-collapse:separate;border-spacing:2px;font-size:10.5px}
.mhtab th.mn{font-family:var(--mono);font-weight:400;color:var(--muted);writing-mode:vertical-rl;transform:rotate(180deg);padding:2px 0}
.mhtab th.mrow{text-align:right;font-weight:500;color:var(--ink2);padding-right:6px;white-space:nowrap}
.mhtab td.mh{width:22px;height:20px;border-radius:4px;border:1px solid var(--grid)}
.mtab td{font-size:12px}.mtab code{font-size:11px}.mtab .nowrap{white-space:nowrap}
.mbar{display:inline-block;width:64px;height:8px;border-radius:4px;background:var(--wash);vertical-align:middle;overflow:hidden;border:1px solid var(--grid)}
.mbar i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--brand2))}
"""
