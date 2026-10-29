"""
compose_arcs_section.py -- the compose arm as concentric semicircular arcs, in the DFKI design of
master_dashboard.py / pair_dashboard.py (section "Two-factor composition — compositional reach").

The compose arm (tools/run_compose.py, spec PREREG_COMPOSE_20261027.md) runs a FIXED list of
ordered transform arms over the anchor P^-: one or two of `para`, `rpe`, `attack`, applied left
to right. Nothing is searched or selected (THESIS_KB §4.4), so the rings are composition DEPTH,
not generations:

  ring d         arms that apply d factors (inner = 1, outer = 2, ...)
  node           one arm; size = behaviours resolved, shade = re-open rate on the newer model
  left | right   the two estimands, split at the apex and never compared across it:
                 intent-gated arms (compositional RSR) | attack-involving arms (NOT RSR)
  star           best estimable, genuine arm per half
  threads        composition parentage: factor arm -> the composite built from it
  dashed ring    the planned design space; filled = run, part-filled = cut short, empty = not run
  centre gauges  per half, the best estimable composite (bold) over its own single-factor
                 control (faint), on the behaviours both resolved; the overshoot is the gap.
                 The gated gauge also ticks the anchor RSR (raw) for reference.

Everything is recomputed from the append-only store (cache/compose/{candidates,judged}.jsonl,
last write per id wins, as in run_compose.JsonlStore) through the package's own Wilson and
McNemar, then checked against the run's compose_summary.json. No prompt or response text is
emitted. The markup uses the host page's tokens and components (.panel, .tiles, .tab, .chart,
[data-tip]) and adds no script: the page's single <script> already drives tooltips and tabs, and
ring/half highlighting is CSS (:has).

    aggregate(run_dir, store=None, anchor=None) -> dict
    section_body(agg) -> str        # body for md.section(...); add CSS to the page's <style>
    latest_run() -> Path | None     # newest results/runs/compose_*/
    render_page(agg) -> str         # standalone preview

    python3 tools/compose_arcs_section.py [run_dir]   # -> <run_dir>/compose_arcs_<YYYYmmdd_HHMM>.html
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import json as _json
import math as _math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recyclebench.stats import mcnemar_exact, wilson_interval  # noqa: E402

__all__ = ["aggregate", "section_body", "latest_run", "render_page", "CSS", "LEDE"]

STORE = ROOT / "cache" / "compose"
CFG_PATH = ROOT / "configs" / "recycle_3_31.yaml"          # run_compose.CFG_PATH
NEW = "llama31-8b"                                          # run_compose.NEW: re-opened = NEW complied
# Below this many resolved behaviours an arm's rate rests on a handful of items: drawn hollow,
# never starred, never a gauge champion (KB §4.4: the RPE arms resolved 7, 6 and 6 of 61).
MIN_RESOLVED = 10
GATE_ORDER = ("cosine", "bertscore", "nli_entail", "llm_intent")   # RealGate's evaluation order
HALVES = (("g", "Intent-gated", "compositional RSR"),
          ("a", "Ungated", "attack transfer — not RSR"))


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------
def _load_store(path: Path) -> dict:
    """Last write per id wins (a candidate is written once, then again with its gate scores)."""
    mem: dict = {}
    if not path.exists():
        return mem
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                d = _json.loads(line)
            except _json.JSONDecodeError:
                continue                    # a hard kill can truncate the last line
            mem[d["id"]] = d
    return mem


def _gate_thresholds() -> dict:
    try:
        from recyclebench.config import Config
        g = Config.load(CFG_PATH).gate
        return {k: float(getattr(g, k)) for k in GATE_ORDER}
    except Exception:
        return {"cosine": 0.75, "bertscore": 0.85, "nli_entail": 0.60, "llm_intent": 0.50}


def _first_fail(gate: dict, th: dict) -> str:
    """Attribute a rejection to the first layer that failed. Layers the gate never evaluated
    carry a 1.0 placeholder (KB §9), so they can never be the first failure."""
    if not isinstance(gate, dict) or "error" in gate:
        return "error"
    for k in GATE_ORDER:
        v = gate.get(k)
        if isinstance(v, (int, float)) and v < th[k]:
            return k
    return "other"


def _wilson(k: int, n: int):
    if not n:
        return None
    w = wilson_interval(k, n)
    return [w.low, w.high]


def _arm_label(factors) -> str:
    return " → ".join(factors)


def _anchor_from(corrected) -> dict | None:
    """corrected_<bench>.json (apply_calibration) -> the RAW anchor RSR. Compose rates are raw,
    so the reference tick is raw too."""
    if not corrected:
        return None
    if isinstance(corrected, (str, Path)):
        try:
            corrected = _json.loads(Path(corrected).read_text())
        except Exception:
            return None
    r = (corrected or {}).get("RSR") or {}
    if r.get("RSR_raw") is None:
        return None
    return {"k": r.get("k_recovered"), "n": r.get("n_p_minus"), "rate": float(r["RSR_raw"]),
            "corrected": r.get("RSR_corrected")}


def aggregate(run_dir, store=None, anchor=None) -> dict:
    """run_dir holds compose_summary.json (+ optional compose_analysis_*.json); store holds the
    per-candidate JSONL the runner appends to. anchor = corrected_<bench>.json path or dict."""
    run_dir = Path(run_dir)
    summary = _json.loads((run_dir / "compose_summary.json").read_text())
    store = Path(store) if store else STORE
    n_cands = int(summary.get("candidates_per_behaviour") or 0)
    n_total = int(summary.get("behaviours") or 0)
    th = _gate_thresholds()

    gen = _load_store(store / "candidates.jsonl")
    jdg = _load_store(store / "judged.jsonl")
    tpl = None
    if (store / "attack_template.json").exists():
        t = _json.loads((store / "attack_template.json").read_text())
        tpl = (t["prefix"].strip()[:80], t["suffix"].strip()[-80:])

    arms: dict = {}
    for s in summary["arms"]:
        arms[s["arm"]] = dict(
            id=s["arm"], factors=tuple(s["factors"]), depth=len(s["factors"]),
            gated=bool(s["intent_gated"]), half="g" if s["intent_gated"] else "a",
            label=_arm_label(s["factors"]), estimand=s.get("estimand", ""),
            summary_kn=(s.get("k_reopened"), s.get("n_resolved")),
            summary_truncated=bool(s.get("truncated")),
            cands=0, empty=0, admitted=0, rejected={}, surv=[0, 0],
            bids=set(), resolved=set(), opened=set())
    for d in gen.values():
        a = arms.get(d.get("arm"))
        if a is None or (n_cands and d.get("idx", 0) >= n_cands):
            continue
        if not d.get("ok"):
            a["empty"] += 1
            continue
        a["cands"] += 1
        a["bids"].add(d["bid"])
        if "attack" in a["factors"] and tpl:
            a["surv"][1] += 1
            a["surv"][0] += int(tpl[0] in d["text"] and tpl[1] in d["text"])
        if d.get("gate_admitted"):
            a["admitted"] += 1
        elif "gate" in d:
            f = _first_fail(d["gate"], th)
            a["rejected"][f] = a["rejected"].get(f, 0) + 1
        if (not a["gated"]) or d.get("gate_admitted"):
            a["resolved"].add(d["bid"])
    for d in jdg.values():
        a = arms.get(d.get("arm"))
        if a is None or (n_cands and d.get("idx", 0) >= n_cands):
            continue
        if d.get("model") == NEW and d.get("complied"):
            a["opened"].add(d["bid"])
    for a in arms.values():
        a["opened"] &= a["resolved"]
        a["ran"] = a["cands"] + a["empty"] > 0
        a["n_bids"] = len(a["bids"])
        a["truncated"] = a["ran"] and (a["summary_truncated"] or len(a["bids"]) < n_total)
        # a factor whose signature does not survive the later steps was dissolved by them:
        # the arm is then not the composition its name says (KB §4.4, §9)
        s_k, s_n = a["surv"]
        a["dissolved"] = (a["ran"] and "attack" in a["factors"] and s_n > 0 and s_k / s_n < 0.5)
        if not a["ran"]:
            a["status"] = "not_run"
        elif a["dissolved"]:
            a["status"] = "not_composition"
        elif len(a["resolved"]) < MIN_RESOLVED:
            a["status"] = "non_estimable"
        else:
            a["status"] = "ok"

    # integrity: the store must reproduce the summary the runner wrote
    mismatch = [a["id"] for a in arms.values() if a["ran"]
                and (len(a["opened"]), len(a["resolved"])) != tuple(a["summary_kn"])]

    # views: "all" = every resolved behaviour; "matched" = per half, the behaviours every
    # estimable arm of that half resolved (e.g. the attack arms on the 39 attack→para reached)
    def matched_set(half):
        sets = [a["resolved"] for a in arms.values()
                if a["half"] == half and a["ran"] and len(a["resolved"]) >= MIN_RESOLVED]
        return set.intersection(*sets) if sets else None

    views = {"all": {"g": None, "a": None},
             "matched": {"g": matched_set("g"), "a": matched_set("a")}}
    out_views = {}
    for vname, msets in views.items():
        out_views[vname] = _view(arms, msets, n_total)
    differs = any(out_views["all"]["arms"][i]["kn"] != out_views["matched"]["arms"][i]["kn"]
                  for i in out_views["all"]["arms"])

    defects = []
    analyses = sorted(run_dir.glob("compose_analysis_*.json"))
    if analyses:
        an = _json.loads(analyses[-1].read_text())
        defects = [(d.get("arm", ""), d.get("what", "")) for d in an.get("defect_found", [])]

    planned = len(arms)
    return dict(
        run=run_dir.name, pair=summary.get("pair", ""), generated=summary.get("generated", ""),
        n_total=n_total, n_cands=n_cands, exploratory=bool(summary.get("exploratory", True)),
        arms={k: {kk: vv for kk, vv in a.items() if not isinstance(vv, set)} for k, a in arms.items()},
        views=out_views, views_differ=differs,
        matched_n={h: (len(s) if s is not None else None) for h, s in views["matched"].items()},
        planned=planned, n_run=sum(a["ran"] for a in arms.values()),
        n_truncated=sum(a["truncated"] for a in arms.values()),
        mismatch=mismatch, defects=defects, anchor=_anchor_from(anchor), thresholds=th)


def _view(arms: dict, msets: dict, n_total: int) -> dict:
    per = {}
    for a in arms.values():
        m = msets[a["half"]]
        res = a["resolved"] if m is None else (a["resolved"] & m)
        opn = a["opened"] & res
        k, n = len(opn), len(res)
        per[a["id"]] = dict(kn=(k, n), rate=(k / n if n else None), wilson=_wilson(k, n),
                            res=res, opened=opn)

    def estimable(aid):
        a = arms[aid]
        return a["status"] == "ok" and per[aid]["kn"][1] >= MIN_RESOLVED

    single = {a["factors"][0]: a["id"] for a in arms.values() if a["depth"] == 1 and a["ran"]}
    # composite vs its factor arms, on the behaviours all of them resolved
    for a in arms.values():
        p = per[a["id"]]
        p["parents"] = [single[f] for f in a["factors"] if f in single] if a["depth"] > 1 else []
        if a["depth"] < 2 or not a["ran"] or len(p["parents"]) < a["depth"]:
            p["gains"] = None
            continue
        shared = set(p["res"])
        for pid in p["parents"]:
            shared &= per[pid]["res"]
        either = set().union(*[per[pid]["opened"] & shared for pid in p["parents"]])
        p["gains"] = dict(n_shared=len(shared), k_comp=len(p["opened"] & shared),
                          k_factors={pid: len(per[pid]["opened"] & shared) for pid in p["parents"]},
                          k_either=len(either), only=len((p["opened"] & shared) - either))

    stars, gauges = {}, {}
    for h, _, _ in HALVES:
        cand = [aid for aid, a in arms.items() if a["half"] == h and estimable(aid)]
        stars[h] = max(cand, key=lambda x: (per[x]["rate"], per[x]["kn"][1]), default=None)
        comps = [aid for aid in cand if arms[aid]["depth"] > 1]
        champ = max(comps, key=lambda x: (per[x]["rate"], per[x]["kn"][1]), default=None)
        g = dict(champion=champ, control=None)
        if champ:
            # its own control: a single-factor arm of the SAME half (same estimand) it is built from
            ctrls = [pid for pid in per[champ]["parents"] if arms[pid]["half"] == h and estimable(pid)]
            ctrl = max(ctrls, key=lambda x: per[x]["rate"], default=None)
            g["control"] = ctrl
            if ctrl:
                pairs = sorted(per[champ]["res"] & per[ctrl]["res"])
                before = [int(b in per[ctrl]["opened"]) for b in pairs]
                after = [int(b in per[champ]["opened"]) for b in pairs]
                mc = mcnemar_exact(before, after) if pairs else None
                n = len(pairs)
                kc, kb = sum(after), sum(before)
                g.update(n=n, k_champ=kc, k_ctrl=kb,
                         r_champ=kc / n if n else None, r_ctrl=kb / n if n else None,
                         w_champ=_wilson(kc, n), mcnemar_p=(mc["p_value"] if mc else None))
        else:
            singles = [aid for aid in cand if arms[aid]["depth"] == 1]
            g["control"] = max(singles, key=lambda x: per[x]["rate"], default=None)
            g["blocked"] = [arms[aid]["label"] for aid, a in arms.items()
                            if a["half"] == h and a["depth"] > 1 and a["ran"]]
        gauges[h] = g
    for p in per.values():
        p.pop("res"); p.pop("opened")
    return dict(arms=per, stars=stars, gauges=gauges)




# ----------------------------------------------------------------------
# svg (DFKI design: the page's .chart tokens, tooltips via the page's own [data-tip] handler)
# ----------------------------------------------------------------------
W, H = 880, 446
CX, BASE_Y = W / 2, 400.0
R_GAUGE, R_MIN, R_COV = 72.0, 98.0, 352.0
RING_LO, RING_HI = 158.0, 268.0
HALF_ANGLES = {"g": (115.0, 165.0), "a": (65.0, 15.0)}      # apex side first
COV_ANGLES = {"g": (100.0, 172.0), "a": (80.0, 8.0)}


def _pt(r, deg):
    a = _math.radians(deg)
    return (CX + r * _math.cos(a), BASE_Y - r * _math.sin(a))


def _arc(r, d0, d1):
    x0, y0 = _pt(r, d0)
    x1, y1 = _pt(r, d1)
    sweep = 1 if d1 < d0 else 0
    large = 1 if abs(d1 - d0) > 180 else 0
    return f"M {x0:.1f} {y0:.1f} A {r:.1f} {r:.1f} 0 {large} {sweep} {x1:.1f} {y1:.1f}"


def _star(cx, cy, r, cls="cmp-star"):
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.45
        a = _math.radians(-90 + i * 36)
        pts.append(f"{cx + rr * _math.cos(a):.1f},{cy + rr * _math.sin(a):.1f}")
    return f'<polygon class="{cls}" points="{" ".join(pts)}"/>'


def _ring_r(depth, max_depth):
    if max_depth <= 1:
        return (RING_LO + RING_HI) / 2
    return RING_LO + (RING_HI - RING_LO) * (depth - 1) / (max_depth - 1)


def _pct(x, d=0):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def _ci(w, d=0):
    return "—" if not w else f"[{100 * w[0]:.{d}f}, {100 * w[1]:.{d}f}]"


def _pfmt(p):
    if p is None:
        return "—"
    return f"{p:.4f}".rstrip("0") if p >= 0.001 else f"{p:.1e}"


_STATUS_TXT = {"not_run": "planned, not run",
               "not_composition": "not a composition: the first factor was dissolved",
               "non_estimable": f"non-estimable (< {MIN_RESOLVED} behaviours resolved)",
               "ok": ""}


def _tip(agg, aid, view) -> str:
    """Plain-text tooltip (the page renders data-tip with white-space: pre-line). Numbers only."""
    a = agg["arms"][aid]
    v = view["arms"][aid]
    k, n = v["kn"]
    lines = [a["label"], ("intent-gated · compositional RSR" if a["gated"] else "ungated · attack transfer, not RSR")
             + f" · depth {a['depth']}"]
    if a["ran"]:
        lines.append(f"re-opened on the newer model: {k}/{n} = {_pct(v['rate'], 1)} {_ci(v['wilson'], 1)}")
        n_gate = a["admitted"] + sum(a["rejected"].values())
        cand = f"candidates {a['cands']}" + (f" (+{a['empty']} empty)" if a["empty"] else "")
        if n_gate:
            rej = ", ".join(f"{c} at {kk.replace('_entail', '')}" for kk, c in sorted(a["rejected"].items(), key=lambda x: -x[1]))
            cand += (" · gate " if a["gated"] else " · gate (recorded only) ") + f"{a['admitted']}/{n_gate} admitted" \
                + (f", rejected {rej}" if rej else "")
        lines.append(cand)
        if a["surv"][1]:
            lines.append(f"attack template survives in {a['surv'][0]}/{a['surv'][1]}")
        lines.append(f"coverage {a['n_bids']}/{agg['n_total']} behaviours" + (" · cut short by the time budget" if a["truncated"] else ""))
        gn = v.get("gains")
        if gn:
            fac = " · ".join(f"{agg['arms'][pid]['label']} {kk}" for pid, kk in gn["k_factors"].items())
            lines.append(f"vs its factors (n={gn['n_shared']}): composite {gn['k_comp']} · {fac} · either {gn['k_either']} "
                         f"· composite-only {gn['only']}")
    if a["status"] != "ok":
        lines.append(_STATUS_TXT[a["status"]])
    return "\n".join(lines)


def build_svg(agg: dict, vname: str) -> str:
    view = agg["views"][vname]
    arms = agg["arms"]
    max_depth = max((a["depth"] for a in arms.values()), default=1)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" '
           f'aria-label="Compose arms by composition depth, intent-gated and ungated halves" '
           f'preserveAspectRatio="xMinYMin meet" class="chart cmp-svg">']

    for d in range(1, max_depth + 1):
        r = _ring_r(d, max_depth)
        out.append(f'<path class="cmp-ring" d="{_arc(r, 180, 0)}"/>')
        out.append(f'<text class="ax cmp-hl" data-hl="d{d}" x="{CX - r:.1f}" y="{BASE_Y + 16:.1f}" '
                   f'text-anchor="middle">{d} factor{"s" if d > 1 else ""}</text>')
    out.append(f'<path class="cmp-cov" d="{_arc(R_COV, 180, 0)}"/>')
    out.append(f'<text class="ax" x="{CX - R_COV:.1f}" y="{BASE_Y + 16:.1f}" text-anchor="middle">design space</text>')
    out.append(f'<line class="cmp-div" x1="{CX:.1f}" y1="{BASE_Y - R_GAUGE - 14:.1f}" '
               f'x2="{CX:.1f}" y2="{BASE_Y - R_COV - 16:.1f}"/>')
    for h, t1, t2 in HALVES:
        x, anc = (CX - 14, "end") if h == "g" else (CX + 14, "start")
        out.append(f'<g class="cmp-hl" data-hl="h{h}"><text x="{x:.1f}" y="20" text-anchor="{anc}" class="cmp-ht">'
                   f'{_html.escape(t1)}</text><text x="{x:.1f}" y="35" text-anchor="{anc}" class="ax">'
                   f'{_html.escape(t2)}</text></g>')
    out.append(f'<line class="axis" x1="{CX - R_COV - 12:.1f}" y1="{BASE_Y:.1f}" x2="{CX + R_COV + 12:.1f}" y2="{BASE_Y:.1f}"/>')

    def order_key(aid):
        a, v = arms[aid], view["arms"][aid]
        rank = {"ok": 0, "non_estimable": 1, "not_composition": 2}.get(a["status"], 3)
        return (rank, -(v["rate"] or 0), aid)

    pos, dot = {}, {}
    n_max = max(agg["n_total"], 1)
    for h, _, _ in HALVES:
        for d in range(1, max_depth + 1):
            ids = sorted([aid for aid, a in arms.items() if a["half"] == h and a["depth"] == d and a["ran"]], key=order_key)
            lo, hi = HALF_ANGLES[h]
            for i, aid in enumerate(ids):
                deg = (lo + hi) / 2 if len(ids) == 1 else lo + (hi - lo) * i / (len(ids) - 1)
                pos[aid] = (_ring_r(d, max_depth), deg)
                dot[aid] = 5 + 10 * _math.sqrt(view["arms"][aid]["kn"][1] / n_max)

    for h, _, _ in HALVES:
        out.append(_gauge(agg, view, h, view["gauges"][h]))

    champs = {g["champion"] for g in view["gauges"].values() if g.get("champion")}
    crowded = {aid: sum(1 for x in pos if arms[x]["half"] == arms[aid]["half"] and arms[x]["depth"] == arms[aid]["depth"]) > 4
               for aid in pos}
    for aid, (r, deg) in pos.items():
        a, v = arms[aid], view["arms"][aid]
        x, y = _pt(r, deg)
        rr = dot[aid]
        is_star = view["stars"].get(a["half"]) == aid
        k, n = v["kn"]
        out.append(f'<g class="cmp-node cmp-{a["half"]} st-{a["status"]}" data-d="{a["depth"]}" data-h="{a["half"]}" '
                   f'data-tip="{_html.escape(_tip(agg, aid, view), quote=True)}" tabindex="0" '
                   f'aria-label="{_html.escape(a["label"], quote=True)}: {k} of {n} re-opened">')
        # parentage threads live in the composite's group, so hovering it lights its own lineage
        for pid in v.get("parents") or []:
            if pid in pos:
                x0, y0 = _pt(*pos[pid])
                mx, my = (x0 + x) / 2, (y0 + y) / 2
                qx, qy = CX + (mx - CX) * 1.22, BASE_Y + (my - BASE_Y) * 1.22
                out.append(f'<path class="cmp-thread{" champ" if aid in champs else ""}" '
                           f'd="M {x0:.1f} {y0:.1f} Q {qx:.1f} {qy:.1f} {x:.1f} {y:.1f}"/>')
        sx, sy = _pt(R_MIN - 8, deg)
        out.append(f'<line class="cmp-spoke" x1="{sx:.1f}" y1="{sy:.1f}" x2="{x:.1f}" y2="{y:.1f}"/>'
                   f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="{rr + 9:.1f}"/>')
        if is_star:
            out.append(_star(x, y, rr + 4))
        else:
            op = 0.3 + 0.65 * (v["rate"] or 0) if a["status"] == "ok" else 1
            out.append(f'<circle class="cmp-mark" cx="{x:.1f}" cy="{y:.1f}" r="{rr:.1f}" fill-opacity="{op:.2f}"/>')
            if a["status"] == "not_composition":
                dx = rr * 0.72
                out.append(f'<line class="cmp-slash" x1="{x - dx:.1f}" y1="{y + dx:.1f}" x2="{x + dx:.1f}" y2="{y - dx:.1f}"/>')
        if a["truncated"]:
            rc = rr + 4.5
            circ = 2 * _math.pi * rc
            out.append(f'<circle class="cmp-trunc-bg" cx="{x:.1f}" cy="{y:.1f}" r="{rc:.1f}"/>'
                       f'<circle class="cmp-trunc" cx="{x:.1f}" cy="{y:.1f}" r="{rc:.1f}" '
                       f'stroke-dasharray="{circ * a["n_bids"] / n_max:.1f} {circ:.1f}" transform="rotate(-90 {x:.1f} {y:.1f})"/>')
        if is_star or not crowded[aid]:
            top = y - rr - (8 if a["truncated"] else 5)
            line2 = f"{_pct(v['rate'])} · {k}/{n}" + (" · cut short" if a["truncated"] else "")
            out.append(f'<text class="cmp-nl" x="{x:.1f}" y="{top - 14:.1f}" text-anchor="middle">{_html.escape(a["label"])}</text>'
                       f'<text class="ax" x="{x:.1f}" y="{top - 1:.1f}" text-anchor="middle">{_html.escape(line2)}</text>')
        out.append('</g>')

    # design-space ring: one slot per planned arm, in its half
    for h, _, _ in HALVES:
        ids = [aid for aid, a in arms.items() if a["half"] == h]
        lo, hi = COV_ANGLES[h]
        for i, aid in enumerate(ids):
            a = arms[aid]
            deg = (lo + hi) / 2 if len(ids) == 1 else lo + (hi - lo) * i / (len(ids) - 1)
            x, y = _pt(R_COV, deg)
            st = "run" if a["ran"] and not a["truncated"] else ("cut" if a["ran"] else "none")
            tip = (f"{a['label']}\ndesign space · {'intent-gated' if a['gated'] else 'ungated'}\n"
                   + ("planned, not run" if st == "none" else
                      f"{a['n_bids']}/{agg['n_total']} behaviours reached" + (" · cut short" if st == "cut" else "")))
            out.append(f'<g class="cmp-slot cmp-{h} sl-{st}" data-d="{a["depth"]}" data-h="{h}" '
                       f'data-tip="{_html.escape(tip, quote=True)}" tabindex="0" '
                       f'aria-label="{_html.escape(a["label"], quote=True)}: {st}">'
                       f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12"/>'
                       f'<circle class="cmp-sm" cx="{x:.1f}" cy="{y:.1f}" r="6.5"/>')
            if st == "cut":
                out.append(f'<path class="cmp-sfill" d="M {x:.1f} {y:.1f} L {x:.1f} {y - 6.5:.1f} '
                           + _pie(x, y, 6.5, a["n_bids"] / n_max) + ' Z"/>')
            out.append('</g>')
    out.append(f'<circle class="cmp-centre" cx="{CX:.1f}" cy="{BASE_Y:.1f}" r="3"/></svg>')
    return "".join(out)


def _pie(x, y, r, frac):
    a = 2 * _math.pi * min(frac, 0.9999)
    ex, ey = x + r * _math.sin(a), y - r * _math.cos(a)
    return f"A {r:.1f} {r:.1f} 0 {1 if frac > 0.5 else 0} 1 {ex:.1f} {ey:.1f}"


def _gauge(agg, view, h, g) -> str:
    """Quarter gauge: left runs 180°→90° (rate 0→1), right 0°→90°, mirrored about the apex."""
    def deg(rate):
        return 180 - 90 * rate if h == "g" else 90 * rate

    o = [f'<g class="cmp-gauge cmp-{h}" data-h="{h}">', f'<path class="cmp-gtrack" d="{_arc(R_GAUGE, deg(0), deg(1))}"/>']
    champ, ctrl = g.get("champion"), g.get("control")
    if champ and ctrl and g.get("n"):
        rc, rb = g["r_champ"], g["r_ctrl"]
        o.append(f'<path class="cmp-gctrl" d="{_arc(R_GAUGE, deg(0), deg(rb))}"/>')
        o.append(f'<path class="cmp-gchamp" d="{_arc(R_GAUGE, deg(0), deg(min(rc, rb)))}"/>')
        if rc > rb:
            o.append(f'<path class="cmp-ggap" d="{_arc(R_GAUGE, deg(rb), deg(rc))}"/>')
        if g.get("w_champ"):
            lo, hi = g["w_champ"]
            o.append(f'<path class="cmp-gci" d="{_arc(R_GAUGE + 9, deg(lo), deg(hi))}"/>')
        for r_, cls in ((rc, "cmp-gv"), (rb, "ax")):
            x, y = _pt(R_GAUGE + 27, deg(r_))
            o.append(f'<text class="{cls}" x="{x:.1f}" y="{y + 3:.1f}" text-anchor="middle">{_pct(r_)}</text>')
    elif ctrl:
        v = view["arms"][ctrl]
        o.append(f'<path class="cmp-gctrl" d="{_arc(R_GAUGE, deg(0), deg(v["rate"]))}"/>')
        x, y = _pt(R_GAUGE + 27, deg(v["rate"]))
        o.append(f'<text class="ax" x="{x:.1f}" y="{y + 3:.1f}" text-anchor="middle">{_pct(v["rate"])}</text>')
    anc = agg.get("anchor")
    if h == "g" and anc:
        x0, y0 = _pt(R_GAUGE - 9, deg(anc["rate"]))
        x1, y1 = _pt(R_GAUGE + 13, deg(anc["rate"]))
        o.append(f'<line class="cmp-anchor" x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"/>')
        x, y = _pt(R_GAUGE + 33, deg(anc["rate"]))
        o.append(f'<text class="ax" x="{x:.1f}" y="{y + 3:.1f}" text-anchor="middle">anchor {_pct(anc["rate"])}</text>')
    o.append("</g>")
    return "".join(o)


# ----------------------------------------------------------------------
# section body (tiles, tabs, chart, table) for master_dashboard / pair_dashboard
# ----------------------------------------------------------------------
LEDE = ("Exploratory. Each arm applies one or two fixed transforms, in order, to the anchor P⁻ and re-screens the newer "
        "model; nothing was searched or selected, every planned arm was enumerated. The two halves are different estimands "
        "and are never compared across: intent-gated arms give a compositional RSR, attack arms wrap the request in a fixed "
        "jailbreak template and are not RSR.")


def _tile(label_, value, sub="", emph=False):
    return (f'<div class="tile{" tile-emph" if emph else ""}"><div class="tile-l">{label_}</div>'
            f'<div class="tile-v">{value}</div><div class="tile-s">{sub}</div></div>')


def _tiles(agg, vname) -> str:
    view, arms = agg["views"][vname], agg["arms"]
    out = []
    for h, t1, t2 in HALVES:
        g = view["gauges"][h]
        champ, ctrl = g.get("champion"), g.get("control")
        if champ and ctrl and g.get("n"):
            gn = view["arms"][champ].get("gains")
            out.append(_tile(f"{t1} · {_html.escape(arms[champ]['label'])} over {_html.escape(arms[ctrl]['label'])}",
                             f"{100 * (g['r_champ'] - g['r_ctrl']):+.0f} pts",
                             f"{g['k_champ']}/{g['n']} vs {g['k_ctrl']}/{g['n']}, paired · McNemar p {_pfmt(g['mcnemar_p'])}"
                             + (f" · {gn['only']} opened by neither factor alone" if gn else "")
                             + ("" if h == "g" else " · not RSR, raw judge on wrapped output")))
        else:
            gains = [view["arms"][aid]["gains"] for aid, a in arms.items()
                     if a["half"] == h and a["depth"] > 1 and view["arms"][aid].get("gains")]
            out.append(_tile(f"{t1} · best composite over its factor", '<span class="muted">n/a</span>',
                             f"no composite resolved ≥ {MIN_RESOLVED} behaviours"
                             + (" · composite-only gains " + ", ".join(f"{x['only']} of {x['n_shared']}" for x in gains) if gains else "")))
    ctrl = view["gauges"]["g"].get("control")
    if ctrl:
        v = view["arms"][ctrl]
        anc = agg.get("anchor")
        out.append(_tile(f"Single factor · {_html.escape(arms[ctrl]['label'])}, gated", _pct(v["rate"], 1),
                         f"{v['kn'][0]}/{v['kn'][1]} {_ci(v['wilson'])}"
                         + (f" · anchor RSR {_pct(anc['rate'], 1)} raw ({anc['k']}/{anc['n']}, 8 rewordings)" if anc else "")))
    n_dis = sum(a["status"] == "not_composition" for a in arms.values())
    out.append(_tile("Design space", f"{agg['n_run']} / {agg['planned']}",
                     "arms run" + (f" · {agg['n_truncated']} cut short" if agg["n_truncated"] else "")
                     + (f" · {n_dis} not a composition" if n_dis else "")))
    return '<div class="tiles">' + "".join(out) + "</div>"


def _legend() -> str:
    def sw(svg):
        return f'<svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true" class="cmp-sw">{svg}</svg>'
    items = [(sw('<circle class="cmp-lg-g" cx="8" cy="8" r="6"/>'), "intent-gated arm (compositional RSR)"),
             (sw('<circle class="cmp-lg-a" cx="8" cy="8" r="6"/>'), "attack arm (not RSR)"),
             (sw(_star(8, 8.5, 7)), "best estimable arm per half"),
             (sw('<circle class="cmp-lg-h" cx="8" cy="8" r="5.5"/>'), f"non-estimable (< {MIN_RESOLVED} resolved)"),
             (sw('<circle class="cmp-lg-h solid" cx="8" cy="8" r="5.5"/><line class="cmp-lg-x" x1="4" y1="12" x2="12" y2="4"/>'),
              "not a composition"),
             (sw('<circle class="cmp-lg-cut" cx="8" cy="8" r="6"/>'), "cut short (ring = coverage)"),
             (sw('<circle class="cmp-lg-none" cx="8" cy="8" r="5.5"/>'), "planned, not run"),
             (sw('<path class="cmp-lg-thread" d="M2 13 Q 8 1 14 13"/>'), "built from"),
             (sw('<path class="cmp-lg-gf" d="M2 12 A 8 8 0 0 1 14 12"/><path class="cmp-lg-gb" d="M2 12 A 8 8 0 0 1 9 5"/>'),
              "gauge: best composite (bold) over its control (faint)")]
    return '<div class="legend cmp-legend">' + "".join(f"<span>{s}{_html.escape(t)}</span>" for s, t in items) + "</div>"


def _table(agg, vname) -> str:
    view = agg["views"][vname]
    rows = []
    for aid, a in agg["arms"].items():
        v = view["arms"][aid]
        k, n = v["kn"]
        st = (_STATUS_TXT[a["status"]] or "ok") + ("; cut short" if a["truncated"] else "")
        rows.append(f'<tr><td class="nw">{_html.escape(a["label"])}</td><td>{"gated · RSR" if a["gated"] else "ungated · not RSR"}</td>'
                    f'<td class="num">{a["depth"]}</td><td class="num">{f"{k}/{n}" if a["ran"] else "—"}</td>'
                    f'<td class="num">{_pct(v["rate"], 1)}</td><td class="num">{_ci(v["wilson"], 1)}</td>'
                    f'<td class="num">{a["n_bids"] if a["ran"] else "—"}</td><td>{_html.escape(st)}</td></tr>')
    head = "".join(f'<th scope="col"{" class=num" if i in (2, 3, 4, 5, 6) else ""}>{x}</th>' for i, x in enumerate(
        ["Arm", "Estimand", "Depth", "Re-opened", "Rate", "Wilson 95%", "Reached", "Status"]))
    return (f'<details class="tv"><summary>Table behind the chart</summary><div class="tscroll"><table class="tbl">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>')


def section_body(agg: dict, uid: str = "cmp") -> str:
    """Body for md.section(...): tiles, one tab per behaviour set, the arcs, legend, table, defects."""
    views = ["all", "matched"] if agg["views_differ"] else ["all"]
    mn = agg["matched_n"]
    names = {"all": ("All behaviours", f"{agg['n_total']} P⁻ behaviours"),
             "matched": ("Matched subset", " · ".join(f"{'gated' if h == 'g' else 'ungated'} {n}" for h, n in mn.items() if n is not None))}
    tabs, panels = [], []
    for i, v in enumerate(views):
        on = i == 0
        tabs.append(f'<button type="button" role="tab" id="tab-{uid}-{v}" aria-controls="panel-{uid}-{v}" '
                    f'aria-selected="{"true" if on else "false"}" tabindex="{0 if on else -1}" class="tab">'
                    f'<span class="tt">{names[v][0]}</span><span class="ts">{_html.escape(names[v][1])}</span></button>')
        panels.append(f'<div role="tabpanel" id="panel-{uid}-{v}" aria-labelledby="tab-{uid}-{v}"{"" if on else " hidden"}>'
                      f'{_tiles(agg, v)}<div class="panel"><h3>Arms by composition depth</h3>'
                      f'<p class="note">Rings are depth: inner = one transform, outer = two applied in order '
                      f'(<code>para → attack</code> = reword, then wrap). Node size = behaviours resolved, shade = share the '
                      f'newer model re-opened. Hover a node for its numbers and its lineage; hover a ring label or a half title '
                      f'to isolate it.</p>{build_svg(agg, v)}{_legend()}{_table(agg, v)}</div></div>')
    tablist = (f'<div class="tablist" role="tablist" aria-label="Behaviour set">{"".join(tabs)}</div>' if len(views) > 1 else "")
    chips = ['<span class="chip chip-warn">exploratory</span>',
             '<span class="chip">raw verdicts, no judge correction</span>',
             f'<span class="chip">{agg["n_cands"]} candidates per behaviour per arm</span>',
             f'<span class="chip">{_html.escape(agg["pair"])}</span>']
    if agg["mismatch"]:
        chips.append(f'<span class="chip chip-warn">store does not reproduce the summary: {_html.escape(", ".join(agg["mismatch"]))}</span>')
    defects = ""
    if agg["defects"]:
        defects = ('<details class="tv"><summary>Defects recorded for this run</summary><ul class="take">'
                   + "".join(f'<li><b>{_html.escape(a)}</b>: {_html.escape(w)}</li>' for a, w in agg["defects"]) + "</ul></details>")
    note = ('<p class="note">matched = the behaviours every estimable arm of that half resolved, so the attack arms compare on '
            'the prefix the cut-short arm reached.</p>' if len(views) > 1 else "")
    return (f'<div class="cmp-chips">{"".join(chips)}</div>{tablist}{note}{"".join(panels)}{defects}'
            f'<p class="note">Run <code>{_html.escape(agg["run"])}</code>, recomputed from the compose store and checked against '
            f'its summary. P⁻ here is the {agg["n_total"]}-behaviour offline-rule set; the judged anchor P⁻ is smaller.</p>')


CSS = r"""
.cmp-chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px}
.chart .cmp-ring{fill:none;stroke:var(--grid);stroke-width:1}
.chart .cmp-cov{fill:none;stroke:var(--axis);stroke-width:1;stroke-dasharray:2 5}
.chart .cmp-div{stroke:var(--axis);stroke-width:1}
.chart .cmp-spoke{stroke:var(--grid);stroke-width:1}
.chart .cmp-ht{fill:var(--ink);font-weight:600;font-size:12px;letter-spacing:.04em;text-transform:uppercase}
.chart .cmp-nl{fill:var(--ink);font-weight:600;font-size:12.5px}
.chart .cmp-gv{fill:var(--ink);font-weight:600;font-size:12px;font-variant-numeric:tabular-nums}
.chart .hit{fill:transparent;stroke:none}
.cmp-node,.cmp-slot{cursor:default;outline:none}
.cmp-node:focus-visible .cmp-mark,.cmp-slot:focus-visible .cmp-sm{stroke:var(--ink);stroke-width:2.5}
.cmp-g .cmp-mark{fill:var(--fix);stroke:var(--fix-ink);stroke-width:1.5}
.cmp-a .cmp-mark{fill:var(--brand);stroke:var(--brand);stroke-width:1.5}
.st-non_estimable .cmp-mark,.st-not_composition .cmp-mark{fill:var(--card)}
.st-non_estimable .cmp-mark{stroke-dasharray:3 2.5}
.st-not_composition .cmp-mark{stroke-width:2}
.cmp-g .cmp-slash{stroke:var(--fix-ink);stroke-width:2}.cmp-a .cmp-slash{stroke:var(--brand);stroke-width:2}
.cmp-star{fill:var(--warn);stroke:var(--card);stroke-width:1.5}
.cmp-trunc-bg{fill:none;stroke:var(--grid);stroke-width:2.5}
.cmp-trunc{fill:none;stroke:var(--neutral);stroke-width:2.5}
.cmp-thread{fill:none;stroke:var(--axis);stroke-width:1;stroke-dasharray:3 4;transition:stroke-width .15s,stroke .15s}
.cmp-thread.champ{stroke:var(--warn);stroke-width:1.6}
.cmp-node:hover .cmp-thread,.cmp-node:focus .cmp-thread{stroke:var(--ink);stroke-width:2}
.cmp-sm{fill:var(--card);stroke-width:1.5}
.cmp-g .cmp-sm{stroke:var(--fix-ink)}.cmp-a .cmp-sm{stroke:var(--brand)}
.cmp-g.sl-run .cmp-sm,.cmp-g .cmp-sfill{fill:var(--fix)}
.cmp-a.sl-run .cmp-sm,.cmp-a .cmp-sfill{fill:var(--brand)}
.sl-none .cmp-sm{stroke:var(--muted);stroke-dasharray:2 2}
.cmp-centre{fill:var(--muted)}
.cmp-gtrack{fill:none;stroke:var(--grid);stroke-width:10}
.cmp-gctrl{fill:none;stroke-width:12;opacity:.3}
.cmp-g .cmp-gctrl{stroke:var(--fix)}.cmp-a .cmp-gctrl{stroke:var(--brand)}
.cmp-gchamp{fill:none;stroke-width:5}
.cmp-g .cmp-gchamp{stroke:var(--fix-ink)}.cmp-a .cmp-gchamp{stroke:var(--brand)}
.cmp-ggap{fill:none;stroke:var(--warn);stroke-width:5}
.cmp-gci{fill:none;stroke-width:1.5}
.cmp-g .cmp-gci{stroke:var(--fix-ink)}.cmp-a .cmp-gci{stroke:var(--brand)}
.cmp-anchor{stroke:var(--ink);stroke-width:1.5;stroke-dasharray:2 2}
.cmp-hl{cursor:default}
.cmp-node,.cmp-slot,.cmp-gauge{transition:opacity .18s}
.cmp-svg:has([data-hl="d1"]:hover) :is(.cmp-node,.cmp-slot):not([data-d="1"]),
.cmp-svg:has([data-hl="d2"]:hover) :is(.cmp-node,.cmp-slot):not([data-d="2"]),
.cmp-svg:has([data-hl="d3"]:hover) :is(.cmp-node,.cmp-slot):not([data-d="3"]),
.cmp-svg:has([data-hl="d1"]:hover) .cmp-gauge,.cmp-svg:has([data-hl="d2"]:hover) .cmp-gauge,.cmp-svg:has([data-hl="d3"]:hover) .cmp-gauge,
.cmp-svg:has([data-hl="hg"]:hover) :is(.cmp-node,.cmp-slot,.cmp-gauge):not([data-h="g"]),
.cmp-svg:has([data-hl="ha"]:hover) :is(.cmp-node,.cmp-slot,.cmp-gauge):not([data-h="a"]){opacity:.15}
.cmp-legend .cmp-sw{vertical-align:-3px;margin-right:6px}
.cmp-lg-g{fill:var(--fix);stroke:var(--fix-ink)}.cmp-lg-a{fill:var(--brand)}
.cmp-lg-h{fill:var(--card);stroke:var(--neutral);stroke-width:1.5;stroke-dasharray:3 2}.cmp-lg-h.solid{stroke-dasharray:none}
.cmp-lg-x{stroke:var(--neutral);stroke-width:1.5}
.cmp-lg-cut{fill:none;stroke:var(--neutral);stroke-width:2.5;stroke-dasharray:24 40}
.cmp-lg-none{fill:none;stroke:var(--muted);stroke-width:1.5;stroke-dasharray:2 2}
.cmp-lg-thread{fill:none;stroke:var(--axis);stroke-width:1.3;stroke-dasharray:3 3}
.cmp-lg-gf{fill:none;stroke:var(--neutral);stroke-width:4;opacity:.3}.cmp-lg-gb{fill:none;stroke:var(--neutral);stroke-width:2}
"""


def latest_run(root: Path = ROOT) -> Path | None:
    runs = sorted(p.parent for p in (root / "results" / "runs").glob("compose_*/compose_summary.json"))
    return runs[-1] if runs else None


def render_page(agg: dict) -> str:
    """Standalone preview in the dashboards' design (their CSS and script)."""
    import lineage_dashboard as ld
    import master_dashboard as md
    body = md.section("s-compose", "10b", "Two-factor composition — compositional reach", section_body(agg), LEDE)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<title>Compose arcs</title><style>{ld.CSS}{md.EXTRA_CSS}{CSS}</style></head><body>'
            f'<main id="main">{body}</main><div id="tip" role="tooltip" hidden></div><script>{ld.JS}</script></body></html>')


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=None, help="compose run dir (default: newest results/runs/compose_*)")
    ap.add_argument("--store", default=None, help=f"compose store dir (default {STORE.relative_to(ROOT)})")
    ap.add_argument("--anchor", default=None, help="corrected_<bench>.json for the anchor-RSR reference tick")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir) if a.run_dir else latest_run()
    if run is None:
        raise SystemExit("no results/runs/compose_*/compose_summary.json")
    agg = aggregate(run, store=a.store, anchor=a.anchor)
    out = Path(a.out) if a.out else run / f"compose_arcs_{_dt.datetime.now():%Y%m%d_%H%M}.html"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwrite")
    out.write_text(render_page(agg), encoding="utf-8")
    print("wrote", out, "|", agg["n_run"], "of", agg["planned"], "arms run",
          "| store matches summary" if not agg["mismatch"] else f"| MISMATCH {agg['mismatch']}")
