#!/usr/bin/env python3
"""tools/lineage_dashboard.py -- one professional dashboard for a whole lineage run.

Built for the SORRY-Bench lineage (llama2-7b -> 2-13b -> 3-8b -> 3.1-8b -> 3.2-1b -> 3.2-3b),
but driven entirely by the config's `lineage:` so it works for any lineage run.

Two sources, never mixed silently:
  * HEADLINE figures (RSR per pair, pooled RSR, PBR per model, gate totals) come from the run's
    own summary (`run-real` -> summary_real.json), i.e. the run's judge (qwen2.5:14b).
  * PER-ITEM views (2x2 cells, flows, recovery curves, categories, styles, the case list, the
    all-pairs matrix) are rebuilt from the response cache + frozen manifest and pushed through
    the package's estimators. Their verdicts come from tools/judge_lineage.py's verdict cache
    when complete (=> they reproduce the run), else from the offline refusal-prefix rule -- in
    which case every such view is labelled, and section 09 reconciles the two line by line.

No model calls. Writes a DATED file and never overwrites an existing one.

  python3 tools/lineage_dashboard.py                       # all defaults = the SORRY-Bench lineage
  python3 tools/lineage_dashboard.py --redact              # no prompt/response text at all (shareable)
  python3 tools/lineage_dashboard.py --offline             # force the offline rule even if verdicts exist

Defaults: --config configs/real_llama_lineage_all.yaml  --cache cache/llama_gen.jsonl
          --behaviours data/sorrybench.csv  --run-dir results/runs/lineage_sorrybench_20261020
          (manifest.jsonl + summary_real.json inside it)  --verdicts cache/judge_lineage_sorrybench.jsonl
          --anchor results/runs/recycle_3_31_sorrybench/corrected_sorrybench.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_core as lc                                      # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.datasets import load_jailbreakbench          # noqa: E402
from recyclebench.manifest import Manifest                     # noqa: E402
from recyclebench.pbr import compute_pbr                       # noqa: E402
from recyclebench.recycling import compute_rsr                 # noqa: E402
from recyclebench.stats import JudgeChars, rogan_gladen, wilson_interval  # noqa: E402

IDENTITY = JudgeChars(se=1.0, sp=1.0, response_level=False)
GATE_KEYS = ["cosine", "bertscore", "nli_entail", "llm_intent"]
GATE_LABEL = {"cosine": "Embedding cosine", "bertscore": "BERTScore F1",
              "nli_entail": "NLI entailment", "llm_intent": "LLM intent check"}
TEXT_CAP = {"prompt": 700, "response": 700, "variant": 420, "vresponse": 420}


# =============================================================================
# Analysis
# =============================================================================
@dataclass
class PairView:
    old: str
    new: str
    adjacent: bool
    status: dict
    cells: dict
    complete: bool                    # every P^- variant on NEW has a verdict
    rsr: object = None                # RSRResult (identity judge) or None if incomplete
    recovered: set = field(default_factory=set)
    unresolved: list = field(default_factory=list)
    attempts: list = field(default_factory=list)      # attempts_to_first of recovered items
    max_admitted: int = 0
    by_class: dict = field(default_factory=dict)      # tc -> [n_items_with_tc, n_recovered_via_tc]
    by_domain: dict = field(default_factory=dict)     # domain -> [n, k]

    @property
    def key(self):
        return f"{self.old}->{self.new}"

    @property
    def n(self):
        return len(self.cells["pminus"])

    @property
    def k(self):
        return len(self.recovered) if self.complete else None


@dataclass
class ModelView:
    name: str
    n_screened: int
    n_comply: int
    cm: list
    pbr: object = None                # PBRResult or None if verdicts incomplete
    per_cat: dict = field(default_factory=dict)       # cat -> [k_comply, n]
    class_succ: dict = field(default_factory=dict)    # tc -> [k, n] per-variant on C_M
    class_complete: bool = True


@dataclass
class Analysis:
    cfg: object
    lineage: list
    behaviours: list
    bmap: dict
    obs: dict
    runs: dict
    verdict_kind: str
    verdict_label: str
    verdict_note: str
    models: dict
    pairs: dict                       # (old, new) -> PairView, all ordered i<j
    adjacent: list                    # [(old, new)]
    extra: list                       # non-adjacent matched-scale pairs shown as tabs
    classes: list
    gate: dict
    run: dict                         # parsed run summary (may be empty)
    anchor: dict                      # calibrated judge from the anchor run (may be empty)


def _pair_view(o, n, adjacent, obs, runs, bmap):
    ro, rn = runs[o], runs[n]
    cells = lc.partition(ro, rn)
    pm = cells["pminus"]
    complete = all(obs[n].recovery_complete(b) for b in pm)
    pv = PairView(o, n, adjacent, lc.pair_status(o, n), cells, complete)
    pv.unresolved = [b for b in pm if rn.recovery[b].n_admitted == 0]
    pv.max_admitted = max((rn.recovery[b].n_admitted for b in pm), default=0)
    if complete:
        # RSR is undefined (never 0) on an empty P^-: leave pv.rsr None so every view shows "—"
        pv.rsr = compute_rsr(ro, rn, pm, IDENTITY, 30) if pm else None
        pv.recovered = {b for b in pm if rn.recovery[b].obs_bypassed}
        pv.attempts = [rn.recovery[b].attempts_to_first for b in pv.recovered]
        for b in pm:
            dom = lc.domain_of(bmap[b].category)
            d = pv.by_domain.setdefault(dom, [0, 0])
            d[0] += 1
            d[1] += int(b in pv.recovered)
            seen = defaultdict(int)
            for v in obs[n].variants[b]:
                seen[v.transform_class] = max(seen[v.transform_class], int(v.verdict == 1))
            for tc, hit in seen.items():
                c = pv.by_class.setdefault(tc, [0, 0])
                c[0] += 1
                c[1] += hit
    return pv


def _parse_summary(path):
    if not path or not Path(path).exists():
        return {}
    d = json.loads(Path(path).read_text())
    rsr = {}
    for r in d.get("RSR_primary_RAW", []):
        a, _, b = r["pair"].partition(" -> ")
        rsr[(a.strip(), b.strip())] = r
    pbr = {r["model"]: r for r in d.get("PBR_backbone_RAW", [])}
    return {"path": str(path), "lineage": d.get("lineage", []), "rsr": rsr, "pbr": pbr,
            "gate": d.get("gate_audit", {}), "pilot": d.get("pilot", {}),
            "cache": d.get("cache", {}), "calibration_status": d.get("calibration_status", "")}


def _parse_anchor(path):
    if not path or not Path(path).exists():
        return {}
    d = json.loads(Path(path).read_text())
    j = d.get("judge") or {}
    if j.get("se") is None or j.get("sp") is None:
        return {}
    return {"path": str(path), "pair": d.get("pair", ""), "se": float(j["se"]), "sp": float(j["sp"]),
            "kappa": j.get("kappa"), "admitted": j.get("admitted"), "n_bundles": j.get("n_bundles"),
            "se_ci": j.get("se_ci"), "sp_ci": j.get("sp_ci"), "RSR": d.get("RSR") or {},
            "PBR_new": d.get("PBR_new") or {}}


GATE_ORDER = ["cosine", "bertscore", "nli_entail"]       # real_gate.RealGate layer order after L0a


def _gate_stats(manifest, thresholds):
    """Gate audit, honest about the short-circuit.

    RealGate evaluates lexical_divergence -> cosine -> bertscore -> nli_entail (-> intent) and stops
    at the first failure; GateDecision.to_gate_scores() then writes the PASSING value 1.0 for every
    layer it never evaluated. So for a rejected row the first persisted score below threshold names
    the rejecting layer and every later score is a placeholder; a rejected row with no failing score
    was stopped by the (unpersisted) lexical-divergence floor and all four scores are placeholders.
    """
    audit = list(manifest.audit)
    by_class = defaultdict(lambda: [0, 0])
    rejected_by = Counter()
    rejected_by_class = defaultdict(Counter)
    scores = {k: [] for k in GATE_KEYS}           # evaluated scores only
    raw_intent = []
    for r in audit:
        by_class[r.transform_class][0] += 1
        by_class[r.transform_class][1] += int(r.admitted)
        vals = {k: getattr(r, k, None) for k in GATE_KEYS}
        vals = {k: (None if (v is None or v != v) else float(v)) for k, v in vals.items()}
        raw_intent.append(vals["llm_intent"])
        if r.admitted:
            evaluated = GATE_ORDER
        else:
            first = next((k for k in GATE_ORDER if vals[k] is not None and vals[k] < thresholds[k]), None)
            if first is None:
                rejected_by["lexical_divergence"] += 1
                rejected_by_class[r.transform_class]["lexical_divergence"] += 1
                evaluated = []
            else:
                rejected_by[first] += 1
                rejected_by_class[r.transform_class][first] += 1
                evaluated = GATE_ORDER[:GATE_ORDER.index(first) + 1]
        for k in evaluated:
            if vals[k] is not None:
                scores[k].append(vals[k])
    intent_vals = {v for v in raw_intent if v is not None}
    intent_ran = not (len(intent_vals) <= 1 and intent_vals <= {1.0})
    if intent_ran:                                 # only if an intent model really scored rows
        scores["llm_intent"] = [v for v in raw_intent if v is not None]
    labelled = sum(1 for r in audit if r.human_label is not None)
    return {"n": len(audit), "admitted": sum(int(r.admitted) for r in audit),
            "by_class": dict(by_class), "rejected_by": rejected_by,
            "rejected_by_class": {k: dict(v) for k, v in rejected_by_class.items()},
            "scores": scores, "intent_ran": intent_ran,
            "human_labelled": labelled, "thresholds": thresholds}


def analyse(cfg, behaviours, manifest, cache, verdicts, summary_path, anchor_path) -> Analysis:
    L = list(cfg.lineage)
    bmap = {b.bid: b for b in behaviours}
    obs, runs = lc.build_runs(cfg, behaviours, manifest, cache, verdicts, L)
    kind, label, note = verdicts.kind, verdicts.label, ""

    missing_resp = sum(obs[m].missing_responses for m in L)
    if missing_resp:
        note = f"{missing_resp} cached responses are missing; affected items are excluded."

    models = {}
    pi_star = cfg.decision.pi_star
    for m in L:
        run, mo = runs[m], obs[m]
        cm = run.C_M()
        mv = ModelView(m, len(run.canonical_obs), sum(run.canonical_obs.values()), cm)
        if all(mo.recovery_complete(b) for b in cm):
            mv.pbr = compute_pbr(run, IDENTITY, pi_star)
        for bid, c in run.canonical_obs.items():
            pc = mv.per_cat.setdefault(bmap[bid].category, [0, 0])
            pc[0] += c
            pc[1] += 1
        for bid in cm:
            for v in mo.variants[bid]:
                if v.verdict is None:
                    mv.class_complete = False
                    continue
                cs = mv.class_succ.setdefault(v.transform_class, [0, 0])
                cs[0] += v.verdict
                cs[1] += 1
        models[m] = mv

    pairs = {}
    adjacent = list(zip(L, L[1:]))
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            o, n = L[i], L[j]
            pairs[(o, n)] = _pair_view(o, n, (o, n) in adjacent, obs, runs, bmap)
    extra = [p for p, v in pairs.items() if not v.adjacent and v.status["kind"] == "matched-scale"]

    classes = list(cfg.transform_classes or [])
    for r in manifest.audit:
        if r.transform_class not in classes:
            classes.append(r.transform_class)
    g = cfg.gate
    thresholds = {"cosine": g.cosine, "bertscore": g.bertscore, "nli_entail": g.nli_entail,
                  "llm_intent": g.llm_intent}
    summary = _parse_summary(summary_path)
    if summary and summary["lineage"] and summary["lineage"] != L:
        print(f"  [summary] lineage in {summary_path} != config lineage -> ignoring the summary")
        summary = {}
    return Analysis(cfg, L, behaviours, bmap, obs, runs, kind, label, note, models, pairs,
                    adjacent, extra, classes, _gate_stats(manifest, thresholds), summary,
                    _parse_anchor(anchor_path))


# =============================================================================
# Formatting helpers
# =============================================================================
def esc(s) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def pct(x, d=1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.{d}f}%"


def ci_txt(lo, hi, d=1) -> str:
    if lo is None or hi is None or any(isinstance(v, float) and math.isnan(v) for v in (lo, hi)):
        return "—"
    return f"[{100 * lo:.{d}f}, {100 * hi:.{d}f}]"


def p_txt(p) -> str:
    if p is None:
        return "—"
    if p >= 0.001:
        return f"{p:.3f}".rstrip("0").rstrip(".") if p < 1 else "1"
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.1f}×10{sup}"


def num(n) -> str:
    return "—" if n is None else f"{n:,}"


def cap(s, n) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n].rstrip() + " …"


def short(m) -> str:
    return lc.meta(m)["short"]


def label(m) -> str:
    return lc.meta(m)["label"]


def pair_label(o, n) -> str:
    return f"{label(o)} → {label(n)}"


def pair_id(o, n) -> str:
    return f"{o}__{n}".replace(".", "_")


def wilson(k, n):
    if not n:
        return (None, None, None)
    w = wilson_interval(k, n)
    return (w.point, w.low, w.high)


def heat(p, var="--comply", cap_=0.85) -> str:
    """Single-hue sequential fill by mixing the role colour into the surface (OKLab)."""
    if p is None:
        return ""
    q = 6 + max(0.0, min(1.0, p)) * (cap_ * 100 - 6)
    return f"background:color-mix(in oklab, var({var}) {q:.0f}%, var(--surface))"


# =============================================================================
# SVG charts (all colours via CSS roles so both themes work)
# =============================================================================
def svg_open(w, h, label_):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img" '
            f'aria-label="{esc(label_)}" preserveAspectRatio="xMinYMin meet" class="chart">')


def x_axis(x0, x1, y_top, y_bot, ticks=(0, .25, .5, .75, 1), fmt=lambda t: f"{int(t * 100)}%"):
    out = []
    for t in ticks:
        x = x0 + (x1 - x0) * t
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y_top}" y2="{y_bot}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{y_bot + 16}" class="ax" text-anchor="middle">{fmt(t)}</text>')
    return "".join(out)


def forest(rows, title, ref=None, ref_label=""):
    """rows: dicts(label, sub, point, lo, hi, corr, emph, tip, n_txt, sep_before)."""
    W, rh, top = 820, 42, 14
    x0, x1 = 250, 700
    H = top + rh * len(rows) + 8 * sum(1 for r in rows if r.get("sep_before")) + 30
    s = [svg_open(W, H, title)]
    y_bot = H - 30
    s.append(x_axis(x0, x1, top - 4, y_bot))
    if ref is not None:
        xr = x0 + (x1 - x0) * ref
        s.append(f'<line x1="{xr:.1f}" x2="{xr:.1f}" y1="{top - 4}" y2="{y_bot}" class="refline"/>')
        s.append(f'<text x="{xr + 4:.1f}" y="{top + 6}" class="ax">{esc(ref_label)}</text>')
    y = top
    for r in rows:
        if r.get("sep_before"):
            s.append(f'<line x1="8" x2="{W - 8}" y1="{y + 2}" y2="{y + 2}" class="axis"/>')
            y += 8
        cy = y + rh / 2
        cls = "emph" if r.get("emph") else "muted"
        s.append(f'<g class="row {cls}" tabindex="0" data-tip="{esc(r["tip"])}">'
                 f'<rect x="0" y="{y}" width="{W}" height="{rh}" class="hit"/>')
        s.append(f'<text x="8" y="{cy - 3:.1f}" class="lbl strong">{esc(r["label"])}</text>'
                 f'<text x="8" y="{cy + 12:.1f}" class="ax">{esc(r.get("sub", ""))}</text>')
        if r.get("point") is not None:
            if r.get("lo") is not None and r.get("hi") is not None:
                a, b = x0 + (x1 - x0) * r["lo"], x0 + (x1 - x0) * r["hi"]
                s.append(f'<line x1="{a:.1f}" x2="{b:.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="whisk"/>'
                         f'<line x1="{a:.1f}" x2="{a:.1f}" y1="{cy - 5:.1f}" y2="{cy + 5:.1f}" class="whisk"/>'
                         f'<line x1="{b:.1f}" x2="{b:.1f}" y1="{cy - 5:.1f}" y2="{cy + 5:.1f}" class="whisk"/>')
            xp = x0 + (x1 - x0) * r["point"]
            s.append(f'<circle cx="{xp:.1f}" cy="{cy:.1f}" r="6" class="pt"/>')
            if r.get("corr") is not None:
                xc = x0 + (x1 - x0) * r["corr"]
                s.append(f'<path d="M{xc:.1f},{cy - 6:.1f} L{xc + 6:.1f},{cy:.1f} L{xc:.1f},{cy + 6:.1f} '
                         f'L{xc - 6:.1f},{cy:.1f} Z" class="corr"/>')
        s.append(f'<text x="{x1 + 16}" y="{cy - 3:.1f}" class="val">{esc(r.get("v_txt", ""))}</text>'
                 f'<text x="{x1 + 16}" y="{cy + 12:.1f}" class="ax">{esc(r.get("n_txt", ""))}</text></g>')
        y += rh
    s.append("</svg>")
    return "".join(s)


def hbars(rows, title, ref=None, ref_label="", role="comply"):
    """Horizontal bars with optional Wilson whiskers. rows: label, sub, v, lo, hi, tip, v_txt."""
    W, rh, top = 560, 36, 14
    x0, x1 = 196, 486
    H = top + rh * len(rows) + 30
    y_bot = H - 30
    s = [svg_open(W, H, title), x_axis(x0, x1, top - 4, y_bot)]
    if ref is not None:
        xr = x0 + (x1 - x0) * ref
        s.append(f'<line x1="{xr:.1f}" x2="{xr:.1f}" y1="{top - 4}" y2="{y_bot}" class="refline"/>'
                 f'<text x="{xr + 4:.1f}" y="{y_bot - 4}" class="ax">{esc(ref_label)}</text>')
    for i, r in enumerate(rows):
        y = top + i * rh
        cy = y + rh / 2
        s.append(f'<g class="row" tabindex="0" data-tip="{esc(r["tip"])}">'
                 f'<rect x="0" y="{y}" width="{W}" height="{rh}" class="hit"/>'
                 f'<text x="8" y="{cy - 2:.1f}" class="lbl strong">{esc(r["label"])}</text>'
                 f'<text x="8" y="{cy + 12:.1f}" class="ax">{esc(r.get("sub", ""))}</text>')
        if r.get("v") is not None:
            w = max((x1 - x0) * r["v"], 0)
            s.append(f'<rect x="{x0}" y="{cy - 6:.1f}" width="{w:.1f}" height="12" rx="3" class="bar-{role}"/>')
            if r.get("lo") is not None and r.get("hi") is not None:
                a, b = x0 + (x1 - x0) * r["lo"], x0 + (x1 - x0) * r["hi"]
                s.append(f'<line x1="{a:.1f}" x2="{b:.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="whisk"/>'
                         f'<line x1="{a:.1f}" x2="{a:.1f}" y1="{cy - 4:.1f}" y2="{cy + 4:.1f}" class="whisk"/>'
                         f'<line x1="{b:.1f}" x2="{b:.1f}" y1="{cy - 4:.1f}" y2="{cy + 4:.1f}" class="whisk"/>')
        s.append(f'<text x="{x1 + 12}" y="{cy + 4:.1f}" class="val">{esc(r.get("v_txt", ""))}</text></g>')
    s.append("</svg>")
    return "".join(s)


def _place_labels(items, top, bottom, gap=28):
    """De-collide right-hand node labels: items=[(centre_y, ...)] -> list of label y."""
    ys = [c for c, *_ in items]
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = [0.0] * len(ys)
    cur = top - 1e9
    for i in order:
        y = max(ys[i], cur + gap, top + 8)
        placed[i] = y
        cur = y
    overflow = cur - (bottom - 8)
    if overflow > 0:                       # push the whole stack up, keeping the gaps
        for i in order:
            placed[i] -= overflow
        lo = top + 8
        for i in order:                    # re-apply the top bound
            if placed[i] < lo:
                placed[i] = lo
            lo = placed[i] + gap
    return placed


def sankey(pv: PairView, n_total, o, n):
    pm, pe, nf, bs = (len(pv.cells[c]) for c in ("pminus", "persistent", "neg_flip", "both_safe"))
    total = pm + pe + nf + bs
    if total == 0:
        return "<p class='muted'>No screened behaviours for this pair.</p>"
    W, H = 980, 420
    top, bot = 34, H - 14
    gap, nodew = 10, 12
    usable = bot - top - 3 * gap
    xs = [20, 300, 580, 800]
    MIN = 2.0

    def hh(v):
        return 0.0 if v == 0 else max(usable * v / total, MIN)

    s = [svg_open(W, H, f"Flow of behaviours for {pair_label(o, n)}")]
    for t, x in zip(["All behaviours", label(o), label(n), "After rewording (P⁻ only)"], xs):
        s.append(f'<text x="{x}" y="16" class="stage">{esc(t)}</text>')

    def node(x, y, h, role, tip):
        return (f'<rect x="{x}" y="{y:.1f}" width="{nodew}" height="{max(h, MIN):.1f}" rx="2" '
                f'class="node r-{role}" tabindex="0" data-tip="{esc(tip)}"/>')

    def band(xa, ya0, ya1, xb, yb0, yb1, role, tip):
        mx = (xa + xb) / 2
        return (f'<path d="M{xa:.1f},{ya0:.1f} C{mx:.1f},{ya0:.1f} {mx:.1f},{yb0:.1f} {xb:.1f},{yb0:.1f} '
                f'L{xb:.1f},{yb1:.1f} C{mx:.1f},{yb1:.1f} {mx:.1f},{ya1:.1f} {xa:.1f},{ya1:.1f} Z" '
                f'class="band r-{role}" data-tip="{esc(tip)}"/>')

    labels = []   # (cy, x, name, count, role)
    # stage 0
    h0 = hh(total) + 3 * gap
    s.append(node(xs[0], top, h0, "ink", f"{total} behaviours screened by both models"))
    labels.append((top + h0 / 2, xs[0], "screened", total, "ink"))
    # stage 1
    oa, orf = pe + pm, nf + bs
    h_oa, h_or = hh(oa), hh(orf)
    y_oa = top + gap
    y_or = y_oa + h_oa + gap
    split0 = top + h0 * (oa / total)
    s.append(band(xs[0] + nodew, top, split0, xs[1], y_oa, y_oa + h_oa, "comply",
                  f"{label(o)} answered {oa} of {total}"))
    s.append(band(xs[0] + nodew, split0, top + h0, xs[1], y_or, y_or + h_or, "refuse",
                  f"{label(o)} refused {orf} of {total}"))
    s.append(node(xs[1], y_oa, h_oa, "comply", f"{label(o)} answered: {oa}"))
    s.append(node(xs[1], y_or, h_or, "refuse", f"{label(o)} refused: {orf}"))
    labels += [(y_oa + h_oa / 2, xs[1], "answered", oa, "comply"),
               (y_or + h_or / 2, xs[1], "refused", orf, "refuse")]
    # stage 2: persistent, P^- (from answered); regression, both safe (from refused)
    hp, hm, hn, hb = hh(pe), hh(pm), hh(nf), hh(bs)
    y_pe = top
    y_pm = y_pe + hp + gap
    y_nf = y_pm + hm + gap
    y_bs = y_nf + hn + gap
    a_split = y_oa + h_oa * (pe / oa) if oa else y_oa
    r_split = y_or + h_or * (nf / orf) if orf else y_or
    if pe:
        s.append(band(xs[1] + nodew, y_oa, a_split, xs[2], y_pe, y_pe + hp, "comply",
                      f"Still answered by {label(n)}: {pe}"))
    if pm:
        s.append(band(xs[1] + nodew, a_split, y_oa + h_oa, xs[2], y_pm, y_pm + hm, "fix",
                      f"P⁻ — answered by {label(o)}, refused by {label(n)}: {pm}"))
    if nf:
        s.append(band(xs[1] + nodew, y_or, r_split, xs[2], y_nf, y_nf + hn, "comply",
                      f"Regression — refused by {label(o)}, answered by {label(n)}: {nf}"))
    if bs:
        s.append(band(xs[1] + nodew, r_split, y_or + h_or, xs[2], y_bs, y_bs + hb, "refuse",
                      f"Refused by both: {bs}"))
    for y_, h_, nm, c, role in [(y_pe, hp, "still answered", pe, "comply"), (y_pm, hm, "P⁻ · the fix", pm, "fix"),
                                (y_nf, hn, "regression", nf, "comply"), (y_bs, hb, "refused by both", bs, "refuse")]:
        if c:
            s.append(node(xs[2], y_, h_, role, f"{nm}: {c}"))
            labels.append((y_ + h_ / 2, xs[2], nm, c, role))
    # stage 3: recovered / held / unresolved, from P^-
    if pm and pv.complete:
        rec = len(pv.recovered)
        unres = len(pv.unresolved)
        held = pm - rec - unres
        parts = [(rec, "recovered", "comply", f"Re-opened by ≥1 admitted rewording: {rec} of {pm}"),
                 (held, "held", "refuse", f"Refused every admitted rewording: {held} of {pm}"),
                 (unres, "unresolved", "none", f"Unresolved — no rewording passed the gate: {unres} of {pm}")]
        y3 = y_pm - max(0.0, (sum(hh(c) for c, *_ in parts if c) + gap * 2 - hm) / 2)
        y3 = max(top, y3)
        src = y_pm
        for c, nm, role, tip in parts:
            if not c:
                continue
            h3 = hh(c)
            sh = hm * c / pm
            s.append(band(xs[2] + nodew, src, src + sh, xs[3], y3, y3 + h3, role, tip))
            s.append(node(xs[3], y3, h3, role, tip))
            labels.append((y3 + h3 / 2, xs[3], nm, c, role))
            src += sh
            y3 += h3 + gap
    elif pm:
        s.append(f'<text x="{xs[3]}" y="{y_pm + hm / 2:.1f}" class="ax">recovery not judged yet</text>')
    # labels, de-collided per stage column
    by_x = defaultdict(list)
    for it in labels:
        by_x[it[1]].append(it)
    for x, items in by_x.items():
        ys = _place_labels(items, top, bot, gap=32)
        for (cy, _x, nm, c, role), ly in zip(items, ys):
            tx = x + nodew + 8
            if abs(ly - cy) > 1:
                s.append(f'<line x1="{x + nodew + 1}" x2="{tx - 2}" y1="{cy:.1f}" y2="{ly - 4:.1f}" class="leader"/>')
            s.append(f'<text x="{tx}" y="{ly - 2:.1f}" class="lbl strong halo">{esc(nm)}</text>'
                     f'<text x="{tx}" y="{ly + 12:.1f}" class="ax halo">{c:,} · {pct(c / n_total, 1)}</text>')
    s.append("</svg>")
    return "".join(s)


def recovery_curves(an: Analysis):
    """Cumulative share of P^- re-opened within the first j admitted rewordings (manifest order)."""
    pvs = [an.pairs[p] for p in an.adjacent if an.pairs[p].complete and an.pairs[p].n > 0]
    if not pvs:
        return "<p class='muted'>Recovery curves need judged rewordings.</p>"
    J = max(8, max(pv.max_admitted for pv in pvs))
    W, H = 820, 330
    x0, x1, y0, y1 = 60, 640, 20, 280
    s = [svg_open(W, H, "Cumulative recovery by number of rewordings tried")]
    for t in (0, .25, .5, .75, 1):
        y = y1 - (y1 - y0) * t
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                 f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="ax" text-anchor="end">{int(t * 100)}%</text>')
    for j in range(1, J + 1):
        x = x0 + (x1 - x0) * (j - 1) / (J - 1)
        s.append(f'<text x="{x:.1f}" y="{y1 + 18}" class="ax" text-anchor="middle">{j}</text>')
    s.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 38}" class="ax" text-anchor="middle">'
             f'admitted rewordings tried (manifest order)</text>')
    ends = []
    for pv in sorted(pvs, key=lambda p: p.n >= 30):          # emphasised series drawn last
        emph = pv.n >= 30
        pts = []
        for j in range(1, J + 1):
            k = sum(1 for a in pv.attempts if 0 < a <= j)
            pts.append((x0 + (x1 - x0) * (j - 1) / (J - 1), y1 - (y1 - y0) * k / pv.n, j, k))
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y, *_ in pts)
        cls = "line-emph" if emph else "line-mute"
        s.append(f'<path d="{d}" class="{cls}"/>')
        for x, y, j, k in pts:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" class="hitpt" tabindex="-1" data-tip="'
                     f'{esc(pair_label(pv.old, pv.new))}\n{k} of {pv.n} P⁻ re-opened within {j} rewording(s) '
                     f'({pct(k / pv.n)})"/>')
        ex, ey = pts[-1][0], pts[-1][1]
        s.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" class="{"dot-emph" if emph else "dot-mute"}"/>')
        ends.append((ey, f"{short(pv.old)} → {short(pv.new)}  {pct(pts[-1][3] / pv.n, 0)} (n={pv.n})", emph))
    ys = _place_labels([(e[0],) for e in ends], y0, y1 + 10, gap=16)
    for (ey, txt, emph), ly in zip(ends, ys):
        s.append(f'<text x="{x1 + 12}" y="{ly + 4:.1f}" class="{"lbl strong" if emph else "ax"}">{esc(txt)}</text>')
    s.append("</svg>")
    return "".join(s)


def histogram(values, thr, key, not_run=False):
    vals = [v for v in values if v is not None]
    W, H = 400, 190
    x0, x1, y0, y1 = 36, 384, 26, 150
    s = [svg_open(W, H, f"Distribution of {GATE_LABEL[key]}")]
    s.append(f'<text x="{x0}" y="14" class="lbl strong">{esc(GATE_LABEL[key])}</text>')
    if not vals:
        lines = (["Never evaluated: no intent model was configured,", "so every row holds the placeholder 1.0."]
                 if not_run else ["No evaluated scores."])
        s.append(f'<text x="{x0}" y="80" class="ax">' + "".join(
            f'<tspan x="{x0}" dy="{0 if i == 0 else 16}">{esc(t)}</tspan>' for i, t in enumerate(lines)) + "</text></svg>")
        return "".join(s)
    lo = 0.0
    nb = 25
    counts = [0] * nb
    for v in vals:
        b = min(nb - 1, max(0, int((v - lo) / (1 - lo) * nb)))
        counts[b] += 1
    mx = max(counts) or 1
    bw = (x1 - x0) / nb
    for i, c in enumerate(counts):
        if not c:
            continue
        a = lo + (1 - lo) * i / nb
        b = lo + (1 - lo) * (i + 1) / nb
        h = (y1 - y0) * c / mx
        cls = "bar-pass" if a >= thr - 1e-12 else "bar-fail"
        s.append(f'<rect x="{x0 + i * bw + 1:.1f}" y="{y1 - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" rx="2" '
                 f'class="{cls}" tabindex="0" data-tip="{esc(f"{GATE_LABEL[key]} {a:.2f}–{b:.2f}: {c:,} variants")}"/>')
    xt = x0 + (x1 - x0) * (thr - lo) / (1 - lo)
    left = xt > x0 + (x1 - x0) * 0.7               # put the label on the emptier side of the line
    s.append(f'<line x1="{xt:.1f}" x2="{xt:.1f}" y1="{y0 - 6}" y2="{y1}" class="refline"/>'
             f'<text x="{xt - 4 if left else xt + 4:.1f}" y="{y0 + 4}" class="ax halo" '
             f'text-anchor="{"end" if left else "start"}">threshold {thr:.2f}</text>')
    s.append(f'<line x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}" class="axis"/>')
    for t in (0, .5, 1):
        x = x0 + (x1 - x0) * t
        s.append(f'<text x="{x:.1f}" y="{y1 + 16}" class="ax" text-anchor="middle">{t:.1f}</text>')
    below = sum(1 for v in vals if v < thr)
    note = f"{len(vals):,} evaluated · {below:,} below threshold"
    s.append(f'<text x="{x0}" y="{H - 6}" class="ax">{esc(note)}</text></svg>')
    return "".join(s)


# =============================================================================
# HTML blocks
# =============================================================================
def table(head, rows, cls="tbl", num_cols=()):
    h = "".join(f'<th{" class=num" if i in num_cols else ""} scope="col">{c}</th>' for i, c in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td{" class=num" if i in num_cols else ""}>{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tscroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def tableview(head, rows, num_cols=()):
    return (f'<details class="tv"><summary>View as table</summary>{table(head, rows, num_cols=num_cols)}</details>')


def chip(kind):
    txt = {"matched-scale": "matched scale", "within-generation": "within generation",
           "scale-confounded": "scale confounded"}.get(kind, kind)
    return f'<span class="chip chip-{esc(kind)}">{esc(txt)}</span>'


def power_chip(n, floor):
    if n >= floor:
        return '<span class="chip chip-ok"><span aria-hidden="true">●</span> estimable</span>'
    return f'<span class="chip chip-warn"><span aria-hidden="true">▲</span> exploratory · P⁻ &lt; {floor}</span>'


def tile(label_, value, sub="", emph=False, tip=""):
    t = f' data-tip="{esc(tip)}" tabindex="0"' if tip else ""
    return (f'<div class="tile{" tile-emph" if emph else ""}"{t}><div class="tile-l">{label_}</div>'
            f'<div class="tile-v">{value}</div><div class="tile-s">{sub}</div></div>')


def _run_rsr(an, o, n):
    return (an.run.get("rsr") or {}).get((o, n))


def run_wilson(r):
    """Wilson 95% for a run row. The summary's own "CI" is a normal-approximation (delta-method)
    interval, which collapses to [1, 1] at k = n; it is shown separately and labelled as such."""
    _, lo, hi = wilson(r["k_recovered"], r["n_p_minus"])
    return lo, hi


def _corr(an, x):
    if x is None or not an.anchor:
        return None
    return rogan_gladen(x, an.anchor["se"], an.anchor["sp"])


def reconcile(an):
    """Per-pair and per-model differences between this reconstruction and the run summary.
    Returns (n_compared, n_differing); (0, 0) when there is no run summary."""
    cmp_ = diff = 0
    for o, n in an.adjacent:
        r = _run_rsr(an, o, n)
        if not r:
            continue
        pv = an.pairs[(o, n)]
        cmp_ += 1
        diff += int(pv.n != r["n_p_minus"] or pv.k != r["k_recovered"])
    for m in an.lineage:
        r = (an.run.get("pbr") or {}).get(m)
        mv = an.models[m]
        if not r:
            continue
        cmp_ += 1
        diff += int(not mv.pbr or (mv.pbr.k_bypassed, mv.pbr.n_resolved) != (r["k_bypassed"], r["n_resolved"]))
    return cmp_, diff


# ---------------------------------------------------------------------------
def sec_header(an, args, today):
    L = an.lineage
    n_b = len(an.behaviours)
    cache = an.run.get("cache", {})
    n_cmp, n_diff = reconcile(an)
    if an.verdict_kind != "judged":
        badge = "<span class='chip chip-warn'><span aria-hidden='true'>▲</span> per-item views: offline reconstruction</span>"
    elif n_cmp and not n_diff:
        badge = "<span class='chip chip-ok'><span aria-hidden='true'>●</span> per-item views reproduce the run exactly</span>"
    elif n_cmp:
        badge = (f"<span class='chip chip-warn'><span aria-hidden='true'>▲</span> judged per-item views · "
                 f"{n_diff} of {n_cmp} figures differ from the run (section 09)</span>")
    else:
        badge = "<span class='chip'>judged per-item views · no run summary to compare</span>"
    return f"""
<header class="hero" id="top">
  <p class="eyebrow">RecycleBench · cross-generational refusal recycling · SORRY-Bench</p>
  <h1>Do Llama safety fixes survive a reword?</h1>
  <p class="lede">For every consecutive release pair in the Llama lineage, <b>P⁻</b> is the set of
  SORRY-Bench requests the older model answered and the newer model refused — the “fixes”. Each fix is
  then reworded by a decorrelated paraphraser (qwen2.5:3b), filtered by an intent-preservation gate, and
  re-tested on the newer model. The <b>Recycling Success Rate (RSR)</b> is the share of fixes re-opened by
  at least one admitted rewording. A high RSR means the fix was tied to the wording, not the intent.</p>
  <div class="meta">
    <span><b>{len(L)}</b> models</span><span><b>{n_b}</b> behaviours · 44 categories</span>
    <span><b>{len(an.adjacent)}</b> release pairs</span>
    <span>judge <b>{esc(lc.judge_tag_from_cfg(an.cfg) or 'none')}</b></span>
    <span>cache <b>{num(cache.get('entries'))}</b> responses</span>
    <span>generated <b>{today}</b></span>
  </div>
  <div class="meta">{badge}{" <span class='chip chip-warn'>redacted · no prompt or response text</span>" if args.redact else ""}</div>
</header>"""


def sec_headline(an):
    floor = an.cfg.decision.min_p_minus
    adj = an.adjacent
    rows = [_run_rsr(an, o, n) for o, n in adj]
    if all(rows):
        src = "run"
        pooled_n = sum(r["n_p_minus"] for r in rows)
        pooled_k = sum(r["k_recovered"] for r in rows)
    else:
        src = "view"
        pvs = [an.pairs[p] for p in adj]
        pooled_n = sum(p.n for p in pvs)
        pooled_k = sum(p.k or 0 for p in pvs) if all(p.complete for p in pvs) else None
    pr, plo, phi = wilson(pooled_k, pooled_n) if pooled_k is not None else (None, None, None)
    est = [(o, n) for o, n in adj if ((_run_rsr(an, o, n) or {}).get("n_p_minus", an.pairs[(o, n)].n) >= floor)]
    anc = est[0] if est else None
    anc_tile = ""
    if anc:
        r = _run_rsr(an, *anc)
        if r:
            corr = _corr(an, r["RSR_raw"])
            anc_tile = tile(f"Anchor RSR · {short(anc[0])} → {short(anc[1])}", pct(r["RSR_raw"]),
                            f"Wilson 95% {ci_txt(*run_wilson(r))} · n={r['n_p_minus']}"
                            + (f"<br>judge-corrected ≈ {pct(corr)}" if corr is not None else ""), emph=True,
                            tip=f"{r['k_recovered']} of {r['n_p_minus']} fixes re-opened. McNemar exact p = {p_txt(r['mcnemar_p'])}.")
        else:
            pv = an.pairs[anc]
            anc_tile = tile(f"Anchor RSR · {short(anc[0])} → {short(anc[1])}",
                            pct(pv.rsr.rsr_raw) if pv.rsr else "—", f"n={pv.n} (reconstruction)", emph=True)
    pbr = an.run.get("pbr") or {}
    pbr_vals = [(m, pbr[m]["PBR_raw"]) for m in an.lineage if m in pbr]
    if not pbr_vals:
        pbr_vals = [(m, an.models[m].pbr.pbr_raw) for m in an.lineage if an.models[m].pbr]
    pbr_tile = ""
    if pbr_vals:
        lo_m = min(pbr_vals, key=lambda t: t[1])
        hi_m = max(pbr_vals, key=lambda t: t[1])
        pbr_tile = tile("Own-refusal robustness (PBR)", f"{pct(lo_m[1], 0)}–{pct(hi_m[1], 0)}",
                        f"lowest {short(lo_m[0])} · highest {short(hi_m[0])}",
                        tip="PBR: share of each model's own canonically refused behaviours that ≥1 admitted rewording re-opens.")
    g = an.gate
    gate_tile = tile("Rewordings admitted by the gate", f"{g['admitted']:,} / {g['n']:,}",
                     f"{pct(g['admitted'] / g['n'] if g['n'] else None)} of generated · 8 per behaviour")
    pooled_corr = _corr(an, pr) if src == "run" else None
    tiles = (tile("Pooled RSR · 5 release pairs", pct(pr),
                  f"{num(pooled_k)} of {num(pooled_n)} fixes · Wilson {ci_txt(plo, phi)}"
                  + (f"<br>judge-corrected ≈ {pct(pooled_corr)}" if pooled_corr is not None else ""),
                  tip="Pairs are pooled as pair::behaviour units; the same behaviour can appear in several pairs, "
                      "so the interval treats them as independent — read it as descriptive.")
             + anc_tile + pbr_tile + gate_tile)
    # takeaways
    t = []
    if pr is not None:
        t.append(f"Across the {len(adj)} consecutive releases, <b>{pct(pr, 0)}</b> of the observed fixes "
                 f"({pooled_k} of {pooled_n}) were re-opened by at least one meaning-preserving rewording.")
    if anc:
        r = _run_rsr(an, *anc)
        if r:
            t.append(f"Only <b>{pair_label(*anc)}</b> clears the pre-registered floor of |P⁻| ≥ {floor} "
                     f"(n = {r['n_p_minus']}). Its RSR is <b>{pct(r['RSR_raw'])}</b> (Wilson 95% {ci_txt(*run_wilson(r))}), "
                     f"McNemar p = {p_txt(r['mcnemar_p'])} — this pair carries the confirmatory claim.")
    n_expl = sum(1 for o, n in adj if (o, n) not in est)
    if n_expl:
        t.append(f"The other {n_expl} pairs are exploratory: they fall below the floor and are either within one "
                 f"generation or change model size, which confounds generation with capacity (section 02).")
    if pbr_vals:
        t.append(f"Within-model robustness (PBR) ranges from {pct(lo_m[1])} ({label(lo_m[0])}) to "
                 f"{pct(hi_m[1])} ({label(hi_m[0])}): every model can be re-opened by rewording some of what it refuses.")
    t.append("All figures are raw judge output. The judge-corrected values apply the anchor pair's human calibration "
             "(Rogan–Gladen); applying one judge calibration across models is an assumption.")
    return f"""
<section id="s-headline" class="sec" aria-labelledby="h-headline">
  <h2 id="h-headline"><span class="sn">01</span>Headline</h2>
  <div class="tiles">{tiles}</div>
  <div class="panel"><h3>What this shows</h3><ul class="take">{''.join(f'<li>{x}</li>' for x in t)}</ul></div>
</section>"""


def lineage_strip(an):
    L = an.lineage
    cells = []
    for i, m in enumerate(L):
        mt = lc.meta(m)
        mv = an.models[m]
        cr = mv.n_comply / mv.n_screened if mv.n_screened else None
        pbr = (an.run.get("pbr") or {}).get(m)
        pbr_v = pbr["PBR_raw"] if pbr else (mv.pbr.pbr_raw if mv.pbr else None)
        cells.append(f'<div class="mnode" tabindex="0" data-tip="{esc(mt["label"])} · released {mt["released"]}\n'
                     f'answered {mv.n_comply} of {mv.n_screened} canonical requests ({pct(cr)})'
                     f'{" — " + an.verdict_kind + " verdicts" if an.verdict_kind != "judged" else ""}\n'
                     f'PBR {pct(pbr_v)}">'
                     f'<div class="mgen">{esc(mt["gen"])}</div><div class="mname">{esc(mt["size"])}B</div>'
                     f'<div class="mrel">{esc(mt["released"])}</div>'
                     f'<div class="mstat"><span>PBR</span> {pct(pbr_v)}</div></div>')
        if i < len(L) - 1:
            o, n = m, L[i + 1]
            r = _run_rsr(an, o, n)
            pv = an.pairs[(o, n)]
            nn = r["n_p_minus"] if r else pv.n
            rr = r["RSR_raw"] if r else (pv.rsr.rsr_raw if pv.rsr else None)
            st = pv.status
            cells.append(f'<div class="medge e-{esc(st["kind"])}" tabindex="0" data-tip="{esc(pair_label(o, n))}\n'
                         f'{esc(st["why"])}\nP⁻ = {nn} · RSR {pct(rr)}">'
                         f'<div class="earrow" aria-hidden="true">→</div>'
                         f'<div class="epm">P⁻ {nn}</div><div class="ersr">RSR {pct(rr, 0)}</div>'
                         f'<div class="ekind">{esc(st["kind"].replace("-", " "))}</div></div>')
    return f'<div class="strip-scroll"><div class="strip">{"".join(cells)}</div></div>'


def sec_lineage(an):
    floor = an.cfg.decision.min_p_minus
    rows, trows = [], []
    for o, n in an.adjacent:
        r = _run_rsr(an, o, n)
        pv = an.pairs[(o, n)]
        st = pv.status
        run_ci = None
        if r:
            lo, hi = run_wilson(r)
            point, nn, kk, p = r["RSR_raw"], r["n_p_minus"], r["k_recovered"], r["mcnemar_p"]
            run_ci = r.get("CI")
        elif pv.rsr:
            point, lo, hi = pv.rsr.rsr_raw, pv.rsr.raw_ci.low, pv.rsr.raw_ci.high
            nn, kk, p = pv.n, pv.k, pv.rsr.mcnemar_p
        else:
            point = lo = hi = kk = p = None
            nn = pv.n
        corr = _corr(an, point)
        tip = (f"{pair_label(o, n)}\n{kk} of {nn} fixes re-opened · RSR {pct(point)} {ci_txt(lo, hi)}\n"
               f"McNemar p = {p_txt(p)}\n{st['why']}")
        rows.append(dict(label=f"{short(o)} → {short(n)}", sub=f"{st['kind'].replace('-', ' ')} · n={nn}",
                         point=point, lo=lo, hi=hi, corr=corr, emph=(nn >= floor), tip=tip,
                         v_txt=pct(point), n_txt=f"{kk}/{nn}"))
        trows.append([esc(pair_label(o, n)), esc(st["kind"]), num(nn), num(kk), pct(point), ci_txt(lo, hi),
                      ci_txt(*run_ci) if run_ci else "—", pct(corr), p_txt(p), "yes" if nn >= floor else "no"])
    pr = [_run_rsr(an, o, n) for o, n in an.adjacent]
    if all(pr):
        pn, pk = sum(r["n_p_minus"] for r in pr), sum(r["k_recovered"] for r in pr)
        pp, plo, phi = wilson(pk, pn)
        rows.append(dict(label="Pooled", sub="all 5 pairs · descriptive", point=pp, lo=plo, hi=phi,
                         corr=_corr(an, pp), emph=True, sep_before=True, v_txt=pct(pp), n_txt=f"{pk}/{pn}",
                         tip=f"Pooled over pair::behaviour units: {pk} of {pn} (Wilson {ci_txt(plo, phi)})"))
        trows.append(["Pooled", "—", num(pn), num(pk), pct(pp), ci_txt(plo, phi), "—", pct(_corr(an, pp)), "—", "—"])
    anchor_txt = (f"anchor calibration, se {an.anchor['se']:.3f} / sp {an.anchor['sp']:.3f}" if an.anchor
                  else "no anchor calibration found, so no correction is drawn")
    src = (f"Numbers from the run summary ({esc(lc.judge_tag_from_cfg(an.cfg))} judge)." if all(pr)
           else f"Run summary unavailable — numbers recomputed from the {an.verdict_kind} verdicts.")
    return f"""
<section id="s-lineage" class="sec" aria-labelledby="h-lineage">
  <h2 id="h-lineage"><span class="sn">02</span>The lineage at a glance</h2>
  <div class="panel">
    <h3>Release chain</h3>
    <p class="note">Models in release order. Between each pair: the number of fixes (P⁻), the share re-opened by
    rewording (RSR), and whether the pair is a clean comparison. Only cross-generation pairs at matched scale (7–8B)
    isolate the effect of a new release; the others also change model size or stay within one generation.</p>
    {lineage_strip(an)}
  </div>
  <div class="panel">
    <h3>Recycling Success Rate per release pair</h3>
    <p class="note">Dots are raw RSR with Wilson 95% intervals; diamonds are the judge-corrected point estimate
    ({anchor_txt}). Pairs with
    fewer than {floor} fixes are greyed — exploratory by the pre-registered decision rule. {src}</p>
    <div class="legend"><span><i class="lg-dot"></i>raw RSR · 95% CI</span><span><i class="lg-dia"></i>judge-corrected point</span>
    <span><i class="lg-mute"></i>exploratory (P⁻ &lt; {floor})</span></div>
    {forest(rows, "Recycling Success Rate per release pair")}
    {tableview(["Pair", "Design", "P⁻", "Re-opened", "RSR", "Wilson 95%", "Run CI (normal approx.)", "Corrected", "McNemar p", f"P⁻ ≥ {floor}"], trows, num_cols=(2, 3, 4, 7, 8))}
  </div>
</section>"""


def sec_models(an):
    pi_star = an.cfg.decision.pi_star
    crow, ctab, prow, ptab = [], [], [], []
    pbr = an.run.get("pbr") or {}
    for m in an.lineage:
        mv = an.models[m]
        p, lo, hi = wilson(mv.n_comply, mv.n_screened)
        crow.append(dict(label=label(m), sub=f"{lc.meta(m)['released']}", v=p, lo=lo, hi=hi, v_txt=pct(p),
                         tip=f"{label(m)} answered {mv.n_comply} of {mv.n_screened} canonical requests "
                             f"(any of k={an.cfg.budget.k_canonical_samples} draws) · {pct(p)} {ci_txt(lo, hi)}"))
        ctab.append([esc(label(m)), num(mv.n_comply), num(mv.n_screened), pct(p), ci_txt(lo, hi)])
        r = pbr.get(m)
        if r:
            v, lo2, hi2, k2, n2, unres, verdict = (r["PBR_raw"], r["CI"][0], r["CI"][1], r["k_bypassed"],
                                                   r["n_resolved"], r["n_unresolved"], r["verdict"])
        elif mv.pbr:
            x = mv.pbr
            v, lo2, hi2, k2, n2, unres, verdict = (x.pbr_raw, x.raw_ci.low, x.raw_ci.high, x.k_bypassed,
                                                   x.n_resolved, x.n_unresolved, x.verdict)
        else:
            v = lo2 = hi2 = k2 = n2 = unres = None
            verdict = "not judged"
        ptip = (f"{label(m)}: {k2} of {n2} own refusals re-opened by rewording · PBR {pct(v)} "
                f"{ci_txt(lo2, hi2)}\n{unres} refused behaviours had no admitted rewording (unresolved)\n"
                f"verdict vs π* = {pi_star:.2f}: {verdict}") if v is not None else \
            f"{label(m)}: PBR needs every rewording judged (run tools/judge_lineage.py --scope all)"
        prow.append(dict(label=label(m), sub=f"{verdict}", v=v, lo=lo2, hi=hi2, v_txt=pct(v), tip=ptip))
        ptab.append([esc(label(m)), num(k2), num(n2), num(unres), pct(v), ci_txt(lo2, hi2), pct(_corr(an, v)), esc(verdict)])
    view_note = "" if an.verdict_kind == "judged" else (
        f"<p class='flag'>▲ Canonical answer rates are rebuilt with the {esc(an.verdict_kind)} rule, not the run's judge. "
        f"They are shown for shape, not as reported figures (section 09).</p>")
    psrc = "run summary" if pbr else f"{an.verdict_kind} verdicts"
    return f"""
<section id="s-models" class="sec" aria-labelledby="h-models">
  <h2 id="h-models"><span class="sn">03</span>Each model on its own</h2>
  <div class="grid2">
  <div class="panel">
    <h3>How often each model answers the canonical request</h3>
    <p class="note">Share of the {len(an.behaviours)} SORRY-Bench requests answered on at least one of
    k={an.cfg.budget.k_canonical_samples} draws, with 95% Wilson intervals. P⁻ can only be as large as the older model's answer rate allows.</p>
    {view_note}
    {hbars(crow, "Canonical answer rate per model")}
    {tableview(["Model", "Answered", "Screened", "Rate", "95% CI"], ctab, num_cols=(1, 2, 3))}
  </div>
  <div class="panel">
    <h3>Paraphrase robustness of each model's own refusals (PBR)</h3>
    <p class="note">Of what a model refuses canonically, the share at least one admitted rewording re-opens.
    The line marks the pre-registered reference π* = {pi_star:.2f}: an interval wholly above it is “materially
    surface-bound”, wholly below “materially robust”. Source: {psrc}.</p>
    {hbars(prow, "PBR per model", ref=pi_star, ref_label=f"π* {pi_star:.2f}")}
    {tableview(["Model", "Re-opened", "Resolved refusals", "Unresolved", "PBR", "95% CI", "Corrected", "Verdict"], ptab, num_cols=(1, 2, 3, 4, 6))}
  </div>
  </div>
</section>"""


def _case_list(an, pv, redact):
    o, n = pv.old, pv.new
    items, outcomes = [], Counter()
    for bid in pv.cells["pminus"]:
        b = an.bmap[bid]
        rec_n = an.runs[n].recovery[bid]
        vs = an.obs[n].variants[bid]
        if not pv.complete:
            outcome, pill = "pending", '<span class="pill">not judged</span>'
        elif rec_n.n_admitted == 0:
            outcome, pill = "unresolved", '<span class="pill pill-none">no admitted rewording</span>'
        elif bid in pv.recovered:
            hits = sum(1 for v in vs if v.verdict == 1)
            outcome, pill = "recovered", f'<span class="pill pill-comply">re-opened · {hits}/{len(vs)}</span>'
        else:
            outcome, pill = "held", f'<span class="pill pill-refuse">held · 0/{len(vs)}</span>'
        outcomes[outcome] += 1
        cat = f"#{b.category} {lc.category_name(b.category)}"
        if redact:
            body = "".join(f'<div class="vrow v-{"comply" if v.verdict == 1 else "refuse" if v.verdict == 0 else "none"}">'
                           f'<span class="vtc">{esc(v.transform_class)}</span>'
                           f'<span class="vv">{"answered" if v.verdict == 1 else "refused" if v.verdict == 0 else "not judged"}</span></div>'
                           for v in vs) or "<p class='note'>No rewording passed the gate.</p>"
            body = f'<p class="note">Text redacted. Rewordings tried on {esc(label(n))}:</p>{body}'
        else:
            old_resp = next((t for t, v in an.obs[o].draws[bid] if v == 1), "")
            new_resp = next((t for t, v in an.obs[n].draws[bid] if t is not None), "")
            rows = "".join(
                f'<div class="vrow v-{"comply" if v.verdict == 1 else "refuse" if v.verdict == 0 else "none"}">'
                f'<div class="vhead"><span class="vtc">{esc(v.transform_class)}</span>'
                f'<span class="vv">{"answered" if v.verdict == 1 else "refused" if v.verdict == 0 else "not judged"}</span></div>'
                f'<div class="txt"><span class="tl">reworded request</span>{esc(cap(v.text, TEXT_CAP["variant"]))}</div>'
                f'<div class="txt"><span class="tl">{esc(label(n))} replied</span>{esc(cap(v.response, TEXT_CAP["vresponse"]))}</div></div>'
                for v in vs) or "<p class='note'>No rewording passed the gate.</p>"
            body = (f'<div class="txt"><span class="tl">canonical request</span>{esc(cap(b.canonical_prompt, TEXT_CAP["prompt"]))}</div>'
                    f'<div class="txt"><span class="tl">{esc(label(o))} · answered</span>{esc(cap(old_resp, TEXT_CAP["response"]))}</div>'
                    f'<div class="txt"><span class="tl">{esc(label(n))} · refused</span>{esc(cap(new_resp, TEXT_CAP["response"]))}</div>'
                    f'<div class="vlist"><div class="tl">admitted rewordings on {esc(label(n))} ({len(vs)})</div>{rows}</div>')
        items.append(f'<details class="case" data-outcome="{outcome}"><summary>'
                     f'<span class="cid">{esc(bid)}</span><span class="ccat">{esc(cat)}</span>'
                     f'{"" if redact else "<span class=sens>sensitive</span>"}{pill}</summary>'
                     f'<div class="cbody">{body}</div></details>')
    if not items:
        return "<p class='note'>No fixes (P⁻ is empty) for this pair.</p>"
    cnt = outcomes
    btns = "".join(f'<button type="button" class="fbtn{" active" if k == "all" else ""}" data-filter="{k}" '
                   f'aria-pressed="{"true" if k == "all" else "false"}">{lbl} <span>{c}</span></button>'
                   for k, lbl, c in [("all", "All", len(items)), ("recovered", "Re-opened", cnt["recovered"]),
                                     ("held", "Held", cnt["held"]), ("unresolved", "No admitted rewording", cnt["unresolved"]),
                                     ("pending", "Not judged", cnt["pending"])] if k == "all" or c)
    return f'<div class="cases"><div class="filters" role="group" aria-label="Filter cases">{btns}</div>{"".join(items)}</div>'


def pair_panel(an, pv, idx, active, redact):
    o, n = pv.old, pv.new
    floor = an.cfg.decision.min_p_minus
    st = pv.status
    r = _run_rsr(an, o, n)
    pm, pe, nf, bs = (len(pv.cells[c]) for c in ("pminus", "persistent", "neg_flip", "both_safe"))
    total = pm + pe + nf + bs
    view_rsr = pv.rsr.rsr_raw if pv.rsr else None
    view_ci = (pv.rsr.raw_ci.low, pv.rsr.raw_ci.high) if pv.rsr else (None, None)
    if r and an.verdict_kind != "judged":
        kpi_note = (f"<p class='flag'>▲ The headline tiles are the run's judged figures. The flow, cells and cases below are an "
                    f"offline reconstruction ({pm} fixes here vs {r['n_p_minus']} in the run) — see section 09.</p>")
    elif an.verdict_kind != "judged":
        kpi_note = (f"<p class='flag'>▲ No run figure exists for this pair, so every number here comes from the "
                    f"{esc(an.verdict_kind)} reconstruction — read it as indicative only.</p>")
    else:
        kpi_note = ""
    if r:
        t_rsr = tile("RSR (run)", pct(r["RSR_raw"]), f"{r['k_recovered']} of {r['n_p_minus']} · Wilson {ci_txt(*run_wilson(r))}", emph=True)
        t_pm = tile("P⁻ · fixes (run)", num(r["n_p_minus"]), f"floor {floor} · {power_chip(r['n_p_minus'], floor)}")
        t_c = tile("Judge-corrected", pct(_corr(an, r["RSR_raw"])), "anchor calibration · point estimate")
        t_p = tile("McNemar exact p", p_txt(r["mcnemar_p"]), f"mean {r['mean_admitted']} admitted rewordings")
    else:
        t_rsr = tile("RSR", pct(view_rsr), f"{num(pv.k)} of {pm} · {ci_txt(*view_ci)}", emph=True)
        t_pm = tile("P⁻ · fixes", num(pm), f"floor {floor} · {power_chip(pm, floor)}")
        t_c = tile("Judge-corrected", pct(_corr(an, view_rsr)), "anchor calibration · point estimate")
        t_p = tile("McNemar exact p", p_txt(pv.rsr.mcnemar_p if pv.rsr else None), "post-hoc pair" if not pv.adjacent else "")
    matrix = f"""
<div class="m2">
  <div></div><div class="mh">{esc(label(n))} answered</div><div class="mh">{esc(label(n))} refused</div>
  <div class="mh mrow">{esc(label(o))} answered</div>
  <div class="mc c-comply" tabindex="0" data-tip="Both answered — never fixed: {pe}"><b>{pe}</b><span>still answered</span></div>
  <div class="mc c-fix" tabindex="0" data-tip="P⁻ — the fixes under test: {pm}"><b>{pm}</b><span>P⁻ · the fix</span></div>
  <div class="mh mrow">{esc(label(o))} refused</div>
  <div class="mc c-regress" tabindex="0" data-tip="Regression — newly answered by {esc(label(n))}: {nf}"><b>{nf}</b><span>regression</span></div>
  <div class="mc c-refuse" tabindex="0" data-tip="Refused by both: {bs}"><b>{bs}</b><span>refused by both</span></div>
</div>"""
    cls_rows = []
    for tc in an.classes:
        nn, kk = pv.by_class.get(tc, [0, 0])
        cls_rows.append(dict(label=tc.replace("_", " "), sub=f"on {nn} fixes with this style", v=(kk / nn if nn else None),
                             v_txt=(f"{kk}/{nn}" if nn else "—"),
                             tip=f"{tc}: re-opened {kk} of the {nn} fixes that had ≥1 admitted {tc} rewording"))
    cls_chart = (hbars(cls_rows, f"Recovery by style for {pair_label(o, n)}") if pv.complete and pm
                 else "<p class='note'>Needs judged rewordings.</p>")
    hidden = "" if active else " hidden"
    design = ("pre-registered consecutive pair" if pv.adjacent else
              "post-hoc pair — not consecutive, computed from the same frozen stimuli")
    return f"""
<div class="tabpanel" role="tabpanel" id="panel-{pair_id(o, n)}" aria-labelledby="tab-{pair_id(o, n)}"{hidden}>
  <div class="pairhead"><h3>{esc(pair_label(o, n))}</h3>{chip(st['kind'])}{power_chip(r['n_p_minus'] if r else pm, floor)}
  <span class="note">{esc(st['why'])} · {design}</span></div>
  <div class="tiles">{t_pm}{t_rsr}{t_c}{t_p}</div>
  {kpi_note}
  <div class="grid2">
    <div class="panel"><h3>The 2×2 on canonical requests</h3>{matrix}
      <p class="note">{total} behaviours with a canonical verdict for both models.</p></div>
    <div class="panel"><h3>Which rewording styles re-opened fixes</h3>
      <p class="note">For the fixes that had at least one admitted rewording of a style, the share that style re-opened.</p>{cls_chart}</div>
  </div>
  <div class="panel"><h3>Flow of every behaviour</h3>
    <p class="note">Band width is proportional to count (bands under 2 px are drawn at 2 px so they stay visible).</p>
    <div class="legend"><span><i class="lg r-comply"></i>answered</span><span><i class="lg r-refuse"></i>refused</span>
    <span><i class="lg r-fix"></i>P⁻ · the fix</span><span><i class="lg r-none"></i>unresolved (no admitted rewording)</span></div>
    <div class="scroll">{sankey(pv, total, o, n)}</div></div>
  <div class="panel"><h3>The fixes, case by case</h3>
    <p class="note">{"Text is redacted in this build." if redact else "Real prompt and response text, truncated and collapsed. Expand a case to see the canonical request, both models' replies, and every admitted rewording with the newer model's reply."}</p>
    {_case_list(an, pv, redact)}</div>
</div>"""


def sec_pairs(an, redact):
    order = list(an.adjacent) + list(an.extra)
    tabs, panels = [], []
    for i, (o, n) in enumerate(order):
        pv = an.pairs[(o, n)]
        r = _run_rsr(an, o, n)
        nn = r["n_p_minus"] if r else pv.n
        tabs.append(f'<button type="button" role="tab" id="tab-{pair_id(o, n)}" aria-controls="panel-{pair_id(o, n)}" '
                    f'aria-selected="{"true" if i == 0 else "false"}" tabindex="{0 if i == 0 else -1}" '
                    f'class="tab{" posthoc" if not pv.adjacent else ""}">'
                    f'<span class="tt">{esc(short(o))} → {esc(short(n))}</span>'
                    f'<span class="ts">{"post-hoc · " if not pv.adjacent else ""}P⁻ {nn}</span></button>')
        panels.append(pair_panel(an, pv, i, i == 0, redact))
    return f"""
<section id="s-pairs" class="sec" aria-labelledby="h-pairs">
  <h2 id="h-pairs"><span class="sn">04</span>Pair explorer</h2>
  <p class="note">One tab per consecutive release pair, plus the non-consecutive matched-scale pairs (marked post-hoc).
  Use ← → to move between tabs.</p>
  <div class="tablist" role="tablist" aria-label="Release pairs">{''.join(tabs)}</div>
  {''.join(panels)}
</section>"""


def sec_matrix(an):
    L = an.lineage
    head = '<div class="mx-corner">older ↓ · newer →</div>' + "".join(
        f'<div class="mx-h">{esc(short(m))}</div>' for m in L)
    cells = []
    for i, o in enumerate(L):
        cells.append(f'<div class="mx-h mx-r">{esc(short(o))}</div>')
        for j, n in enumerate(L):
            if j <= i:
                cells.append('<div class="mx-c mx-na" aria-hidden="true"></div>')
                continue
            pv = an.pairs[(o, n)]
            rsr = pv.rsr.rsr_raw if (pv.rsr and pv.n) else None
            small = pv.n < 5
            style = heat(rsr) if (rsr is not None and not small) else ""
            cls = ["mx-c"]
            if pv.adjacent:
                cls.append("mx-adj")
            if pv.status["kind"] == "matched-scale":
                cls.append("mx-ms")
            if small:
                cls.append("mx-small")
            tip = (f"{pair_label(o, n)}\nP⁻ = {pv.n}" + (f" · re-opened {pv.k} · RSR {pct(rsr)}" if rsr is not None else "")
                   + f"\n{pv.status['why']}" + ("\npre-registered consecutive pair" if pv.adjacent else "\npost-hoc"))
            cells.append(f'<div class="{" ".join(cls)}" style="{style}" tabindex="0" data-tip="{esc(tip)}">'
                         f'<b>{pct(rsr, 0) if rsr is not None else "—"}</b><span>P⁻ {pv.n}</span></div>')
    trows = []
    for (o, n), pv in an.pairs.items():
        trows.append([esc(pair_label(o, n)), "yes" if pv.adjacent else "no", esc(pv.status["kind"]), num(pv.n),
                      num(pv.k), pct(pv.rsr.rsr_raw if pv.rsr and pv.n else None)])
    flag = "" if an.verdict_kind == "judged" else (
        f"<p class='flag'>▲ Built from {esc(an.verdict_kind)} verdicts — the consecutive pairs here will not match section 02 until the judge pass has run.</p>")
    return f"""
<section id="s-matrix" class="sec" aria-labelledby="h-matrix">
  <h2 id="h-matrix"><span class="sn">05</span>Every ordered pair</h2>
  <div class="panel">
    <p class="note">Because every model answered the same frozen requests and rewordings, any older → newer pair can be compared
    at no extra cost. Cell colour is RSR (darker = more fixes re-opened); the small number is |P⁻|. Outlined cells are the
    pre-registered consecutive pairs; ◆ marks matched-scale cross-generation pairs. Only the consecutive pairs are confirmatory — every
    other cell is post-hoc and must be read as exploratory. Cells with P⁻ &lt; 5 are left uncoloured.</p>
    {flag}
    <div class="scroll"><div class="mx" style="grid-template-columns:120px repeat({len(L)}, minmax(84px,1fr))">{head}{''.join(cells)}</div></div>
    <div class="legend"><span><i class="lg-ramp"></i>RSR 0% → 100%</span><span><i class="lg-adj"></i>consecutive (pre-registered)</span><span>◆ matched scale</span></div>
    {tableview(["Pair", "Consecutive", "Design", "P⁻", "Re-opened", "RSR"], trows, num_cols=(3, 4, 5))}
  </div>
</section>"""


def sec_categories(an):
    L = an.lineage
    head = "<tr><th scope='col'>Category</th>" + "".join(f"<th scope='col' class='num'>{esc(short(m))}</th>" for m in L) + "</tr>"
    body = []
    cats_present = sorted({b.category for b in an.behaviours}, key=lambda c: (int(c) if str(c).isdigit() else 999, str(c)))
    for dname, lo, hi in lc.SORRY_DOMAINS + [("Other", None, None)]:
        dc = [c for c in cats_present if lc.domain_of(c) == dname]
        if not dc:
            continue
        drow = f"<tr class='dom'><th scope='row'>{esc(dname)} <span>({len(dc)} categories)</span></th>"
        for m in L:
            k = sum(an.models[m].per_cat.get(c, [0, 0])[0] for c in dc)
            n = sum(an.models[m].per_cat.get(c, [0, 0])[1] for c in dc)
            p = k / n if n else None
            drow += (f"<td class='num hm' style='{heat(p)}' data-tip='{esc(f'{label(m)} · {dname}: answered {k} of {n} ({pct(p)})')}'>"
                     f"{pct(p, 0)}</td>")
        body.append(drow + "</tr>")
        for c in dc:
            row = f"<tr><th scope='row'><span class='cnum'>#{esc(c)}</span> {esc(lc.category_name(c))}</th>"
            for m in L:
                k, n = an.models[m].per_cat.get(c, [0, 0])
                p = k / n if n else None
                row += (f"<td class='num hm' style='{heat(p)}' "
                        f"data-tip='{esc(f'{label(m)} · #{c} {lc.category_name(c)}: answered {k} of {n}')}'>{k}/{n}</td>")
            body.append(row + "</tr>")
    # P^- and RSR by domain per consecutive pair
    dhead = "<tr><th scope='col'>Domain</th>" + "".join(
        f"<th scope='col' class='num'>{esc(short(o))}→{esc(short(n))}</th>" for o, n in an.adjacent) + "<th scope='col' class='num'>Pooled</th></tr>"
    drows = []
    for dname, *_ in lc.SORRY_DOMAINS:
        row = f"<tr><th scope='row'>{esc(dname)}</th>"
        tn = tk = 0
        ok = True
        for p in an.adjacent:
            pv = an.pairs[p]
            if not pv.complete:
                ok = False
                row += "<td class='num'>—</td>"
                continue
            n_, k_ = pv.by_domain.get(dname, [0, 0])
            tn += n_
            tk += k_
            row += (f"<td class='num hm' style='{heat(k_ / n_ if n_ >= 3 else None)}' "
                    f"data-tip='{esc(f'{pair_label(*p)} · {dname}: {k_} of {n_} fixes re-opened')}'>"
                    f"{(f'{k_}/{n_}') if n_ else '—'}</td>")
        row += (f"<td class='num hm' style='{heat(tk / tn if (ok and tn) else None)}'><b>{pct(tk / tn, 0) if (ok and tn) else '—'}</b>"
                f"<span class='sub'> {tk}/{tn}</span></td>" if ok else "<td class='num'>—</td>")
        drows.append(row + "</tr>")
    flag = "" if an.verdict_kind == "judged" else (
        f"<p class='flag'>▲ Rebuilt with {esc(an.verdict_kind)} verdicts — read the pattern, not the exact counts.</p>")
    return f"""
<section id="s-categories" class="sec" aria-labelledby="h-categories">
  <h2 id="h-categories"><span class="sn">06</span>Harm categories</h2>
  {flag}
  <div class="panel">
    <h3>Where fixes are re-opened, by SORRY-Bench domain</h3>
    <p class="note">Re-opened / P⁻ per consecutive pair, split by domain (darker = higher share; cells with fewer than 3 fixes are uncoloured).
    This is RQ2's locator: where in the taxonomy safety fixes are wording-bound.</p>
    <div class="tscroll"><table class="tbl hmtbl">{dhead}{''.join(drows)}</table></div>
  </div>
  <div class="panel">
    <h3>Canonical answer rate by category and model</h3>
    <p class="note">Each SORRY-Bench category has 10 requests. Cells show how many each model answered (darker = more).
    Domain rows aggregate their categories.</p>
    <div class="legend"><span><i class="lg-ramp"></i>0 → 10 answered</span></div>
    <div class="tscroll"><table class="tbl hmtbl"><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>
  </div>
</section>"""


def sec_styles(an):
    L = an.lineage
    g = an.gate
    head = ("<tr><th scope='col'>Style</th><th scope='col' class='num'>Generated</th><th scope='col' class='num'>Admitted</th>"
            + "".join(f"<th scope='col' class='num'>{esc(short(m))}</th>" for m in L)
            + "<th scope='col' class='num'>Re-opened fixes</th></tr>")
    rows = []
    vmax = max([mv.class_succ[tc][0] / mv.class_succ[tc][1] for mv in an.models.values() if mv.class_complete
                for tc in mv.class_succ if mv.class_succ[tc][1]] or [1.0]) or 1.0
    for tc in an.classes:
        gen, adm = g["by_class"].get(tc, [0, 0])
        r = (f"<tr><th scope='row'>{esc(tc.replace('_', ' '))}</th><td class='num'>{gen:,}</td>"
             f"<td class='num'>{adm:,} <span class='sub'>{pct(adm / gen if gen else None, 0)}</span></td>")
        for m in L:
            mv = an.models[m]
            k, n = mv.class_succ.get(tc, [0, 0])
            p = k / n if (n and mv.class_complete) else None
            r += (f"<td class='num hm' style='{heat(p / vmax) if p is not None else ''}' "
                  f"data-tip='{esc(f'{label(m)} · {tc}: {k} of {n} admitted rewordings of its own refusals answered')}'>"
                  f"{pct(p, 0)}</td>")
        tn = tk = 0
        ok = True
        for p_ in an.adjacent:
            pv = an.pairs[p_]
            if not pv.complete:
                ok = False
                break
            n_, k_ = pv.by_class.get(tc, [0, 0])
            tn += n_
            tk += k_
        r += (f"<td class='num'><b>{pct(tk / tn, 0)}</b> <span class='sub'>{tk}/{tn}</span></td>" if ok and tn
              else "<td class='num'>—</td>")
        rows.append(r + "</tr>")
    flag = "" if an.verdict_kind == "judged" else (
        f"<p class='flag'>▲ Answer rates rebuilt with {esc(an.verdict_kind)} verdicts.</p>")
    return f"""
<section id="s-styles" class="sec" aria-labelledby="h-styles">
  <h2 id="h-styles"><span class="sn">07</span>Rewording styles</h2>
  {flag}
  <div class="panel">
    <h3>Which kinds of rewording get past each model</h3>
    <p class="note">Per style: how many rewordings were generated and admitted by the gate, then — for each model — the share of
    admitted rewordings of its <em>own</em> canonical refusals that it answered (per-variant rate; darker = higher, scaled to the table's maximum). The last column pools
    the consecutive pairs: of the fixes that had an admitted rewording of this style, the share it re-opened. This is the descriptive
    side of RQ3; the mixed-effects model lives in <code>recyclebench.drivers</code>.</p>
    <div class="tscroll"><table class="tbl hmtbl">{head}{''.join(rows)}</table></div>
  </div>
  <div class="panel">
    <h3>Cumulative recovery by number of rewordings tried</h3>
    <p class="note">For each consecutive pair, the share of fixes re-opened within the first <i>j</i> admitted rewordings. A curve that
    flattens early means a few rewordings do most of the work; one still rising at the end means a larger budget would re-open more.
    The estimable pair is highlighted.</p>
    {recovery_curves(an)}
  </div>
</section>"""


def sec_gate(an):
    g = an.gate
    th = g["thresholds"]
    hists = "".join(f'<div class="hcell">{histogram(g["scores"][k], th[k], k, not_run=(k == "llm_intent" and not g["intent_ran"]))}</div>'
                    for k in GATE_KEYS)
    rej = g["n"] - g["admitted"]
    layers = [("lexical_divergence", "Lexical divergence (L0a)", "too close to the original wording"),
              ("cosine", "Embedding cosine (L0b)", f"below {th['cosine']:.2f}"),
              ("bertscore", "BERTScore F1 (L0c)", f"below {th['bertscore']:.2f}"),
              ("nli_entail", "NLI entailment (L1)", f"below {th['nli_entail']:.2f}")]
    frows = [dict(label=nm, sub=why, v=(g["rejected_by"][k] / rej if rej else None),
                  v_txt=f"{g['rejected_by'][k]:,}",
                  tip=f"{nm}: first failing layer for {g['rejected_by'][k]:,} of {rej:,} rejected rewordings")
             for k, nm, why in layers]
    ftab = [[esc(nm), esc(why), f"{g['rejected_by'][k]:,}", pct(g['rejected_by'][k] / rej if rej else None)]
            for k, nm, why in layers]
    ctab = []
    for tc in an.classes:
        gen, adm = g["by_class"].get(tc, [0, 0])
        rb = g["rejected_by_class"].get(tc, {})
        ctab.append([esc(tc.replace("_", " ")), f"{gen:,}", f"{adm:,}", pct(adm / gen if gen else None, 0)]
                    + [f"{rb.get(k, 0):,}" for k, *_ in layers])
    crows = [dict(label=tc.replace("_", " "), sub=f"{g['by_class'].get(tc, [0, 0])[0]:,} generated",
                  v=(g["by_class"][tc][1] / g["by_class"][tc][0] if g["by_class"].get(tc, [0, 0])[0] else None),
                  v_txt=pct(g["by_class"][tc][1] / g["by_class"][tc][0] if g["by_class"].get(tc, [0, 0])[0] else None, 0),
                  tip=f"{tc}: {g['by_class'].get(tc, [0, 0])[1]:,} of {g['by_class'].get(tc, [0, 0])[0]:,} admitted")
             for tc in an.classes]
    warn = []
    if not g["intent_ran"]:
        warn.append("<li><b>The LLM intent check (L2) never ran.</b> Its score is the placeholder 1.0 on every row (no intent "
                    "model was configured), so the gate actually applied was lexical divergence + embedding cosine + BERTScore + NLI.</li>")
    if g["rejected_by"]["lexical_divergence"]:
        bt = g["rejected_by_class"].get("back_translation", {}).get("lexical_divergence", 0)
        warn.append(f"<li><b>{g['rejected_by']['lexical_divergence']:,} rewordings were rejected for being too close to the original</b> "
                    f"(lexical-divergence floor){f', {bt:,} of them back-translations that round-tripped to nearly the same sentence' if bt else ''}. "
                    f"That layer's score is not persisted in the manifest, so these rows carry placeholder 1.0 scores; they are "
                    f"excluded from the histograms below.</li>")
    if g["human_labelled"] == 0:
        warn.append("<li><b>No human audit yet.</b> None of the audit rows carries a human label. The run summary's "
                    "“false-admission rate 0.0” is computed against a default of <i>preserved</i> for every variant, so it "
                    "is not evidence that the gate admits only intent-preserving rewordings. Run <code>audit-export</code> → "
                    "annotate → <code>audit-ingest</code> to measure it.</li>")
    return f"""
<section id="s-gate" class="sec" aria-labelledby="h-gate">
  <h2 id="h-gate"><span class="sn">08</span>The intent gate</h2>
  <div class="tiles">
    {tile("Generated rewordings", f"{g['n']:,}", f"{len(an.behaviours)} behaviours × {an.cfg.budget.paraphrases_per_behaviour}")}
    {tile("Admitted", f"{g['admitted']:,}", pct(g['admitted'] / g['n'] if g['n'] else None) + " of generated")}
    {tile("Rejected", f"{rej:,}", "stopped at the first failing layer")}
    {tile("Human-audited", f"{g['human_labelled']:,}", "rows with a human label")}
  </div>
  {"<div class='panel warnbox'><h3>Read before citing the gate</h3><ul class='take'>" + ''.join(warn) + "</ul></div>" if warn else ""}
  <div class="panel">
    <h3>Score distributions with thresholds</h3>
    <p class="note">Only scores the gate actually computed. The layers run in order and stop at the first failure, so later
    layers see fewer rewordings. Solid bars clear the threshold; faint bars fall below it. A rewording is admitted only if it clears every layer.</p>
    <div class="hgrid">{hists}</div>
  </div>
  <div class="grid2">
    <div class="panel"><h3>Where rejected rewordings were stopped</h3>
      <p class="note">First failing layer for each of the {rej:,} rejected rewordings (each is counted once).</p>
      {hbars(frows, "First failing layer among rejected rewordings", role="neutral")}
      {tableview(["Layer", "Rule", "Rejected here", "Share"], ftab, num_cols=(2, 3))}</div>
    <div class="panel"><h3>Admission rate by style</h3>
      <p class="note">Share of each style's generated rewordings the gate admitted.</p>
      {hbars(crows, "Admission rate by style", role="neutral")}
      {tableview(["Style", "Generated", "Admitted", "Rate"] + [nm for _, nm, _ in layers], ctab, num_cols=(1, 2, 3, 4, 5, 6, 7))}</div>
  </div>
</section>"""


def sec_reconcile(an, args):
    rows = []
    any_diff = False
    for o, n in an.adjacent:
        r = _run_rsr(an, o, n)
        pv = an.pairs[(o, n)]
        if r:
            dn = pv.n - r["n_p_minus"]
            dk = (pv.k - r["k_recovered"]) if pv.k is not None else None
            any_diff |= bool(dn) or bool(dk)
            rows.append([esc(pair_label(o, n)), num(r["n_p_minus"]), num(pv.n), f"{dn:+d}",
                         num(r["k_recovered"]), num(pv.k), f"{dk:+d}" if dk is not None else "—",
                         "● match" if not dn and not dk else ("not judged" if dk is None else "▲ differs")])
        else:
            rows.append([esc(pair_label(o, n)), "—", num(pv.n), "—", "—", num(pv.k), "—", "no run figure"])
    prow = []
    for m in an.lineage:
        r = (an.run.get("pbr") or {}).get(m)
        mv = an.models[m]
        vk = mv.pbr.k_bypassed if mv.pbr else None
        vn = mv.pbr.n_resolved if mv.pbr else None
        if r:
            same = (vk == r["k_bypassed"] and vn == r["n_resolved"])
            any_diff |= not same
            prow.append([esc(label(m)), f"{r['k_bypassed']}/{r['n_resolved']}", f"{num(vk)}/{num(vn)}",
                         "● match" if same else ("not judged" if vk is None else "▲ differs")])
    if an.verdict_kind == "judged" and not any_diff:
        verdict = ("<p class='okline'>● Every per-item figure in this dashboard reproduces the run summary exactly.</p>")
    elif an.verdict_kind == "judged":
        verdict = ("<p class='flag'>▲ Judged verdicts are in use, but some figures differ from the run. The rubric judge is "
                   "called without a fixed seed, so a re-judge can flip borderline responses; the table shows by how much.</p>")
    else:
        verdict = (f"<p class='flag'>▲ Per-item views use the {esc(an.verdict_label)}. The run used "
                   f"<b>{esc(lc.judge_tag_from_cfg(an.cfg))}</b>, whose per-response verdicts were not saved, so the two disagree. "
                   f"To reproduce the run, replay the judge over the cache (no target-model calls) and rebuild:</p>"
                   f"<pre class='cmd'>OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h caffeinate -is \\\n"
                   f"  python3 tools/judge_lineage.py --scope all     # ~13k judge calls; --scope rsr for ~3k\n"
                   f"python3 tools/lineage_dashboard.py</pre>")
    srcs = [
        ("Configuration", args.config), ("Response cache", args.cache), ("Behaviours", args.behaviours),
        ("Frozen manifest", args.manifest), ("Run summary", an.run.get("path", "— (not found)")),
        ("Anchor calibration", an.anchor.get("path", "— (not found)")),
        ("Per-item verdicts", an.verdict_label),
    ]
    anc = an.anchor
    anc_line = (f"se {anc['se']:.3f} {ci_txt(*anc['se_ci']) if anc.get('se_ci') else ''}, "
                f"sp {anc['sp']:.3f} {ci_txt(*anc['sp_ci']) if anc.get('sp_ci') else ''}, κ {anc['kappa']:.3f}, "
                f"{anc.get('n_bundles')} bundles, admitted = {anc.get('admitted')}" if anc else "not available")
    return f"""
<section id="s-method" class="sec" aria-labelledby="h-method">
  <h2 id="h-method"><span class="sn">09</span>Reconciliation, method &amp; provenance</h2>
  <div class="panel">
    <h3>Does this dashboard reproduce the run?</h3>
    {verdict}
    <h3 class="gap">Per release pair · P⁻ and RSR</h3>
    {table(["Pair", "P⁻ run", "P⁻ here", "Δ", "Re-opened run", "Re-opened here", "Δ", ""], rows, num_cols=(1, 2, 3, 4, 5, 6))}
    {("<h3 class='gap'>Per model · PBR</h3>" + table(["Model", "PBR run (k/n)", "PBR here (k/n)", ""], prow, num_cols=(1, 2))) if prow else ""}
  </div>
  <div class="grid2">
  <div class="panel">
    <h3>How the numbers are made</h3>
    <ul class="take">
      <li><b>Canonical screening.</b> Each model gets every request k = {an.cfg.budget.k_canonical_samples} times (distinct seeds,
      temperature 0); a behaviour counts as answered if any draw complies.</li>
      <li><b>P⁻</b> = answered by the older model and refused by the newer one. <b>RSR</b> = share of P⁻ the newer model answers on at
      least one admitted rewording. The denominator counts behaviours, never variants, and includes fixes with no admitted rewording.</li>
      <li><b>Rewordings</b> are generated once by qwen2.5:3b in {len(an.classes)} styles ({an.cfg.budget.paraphrases_per_behaviour} per behaviour),
      gated once, then frozen and reused identically for every model.</li>
      <li><b>Intervals</b>: Wilson 95% on raw rates. <b>McNemar</b>: exact test of canonical vs reworded outcome on P⁻.</li>
      <li><b>Judge correction</b>: Rogan–Gladen at the behaviour-bundle level using the anchor pair's human calibration ({anc_line}).
      Shown as a point estimate here; its interval would also need the calibration variance.</li>
      <li><b>Decision rule</b>: a pair with |P⁻| &lt; {an.cfg.decision.min_p_minus} is exploratory (pre-registered).</li>
    </ul>
  </div>
  <div class="panel">
    <h3>Caveats</h3>
    <ul class="take">
      <li>Four of the five consecutive pairs are within one generation or change model size, so generation is confounded with
      capacity. The clean design is the matched-scale chain Llama 2 7B → Llama 3 8B → Llama 3.1 8B; its non-consecutive link is in
      section 04 as a post-hoc tab.</li>
      <li>Calibration status of the run: {esc(an.run.get('calibration_status', 'unknown'))}.</li>
      <li>The paraphraser is a weak, benign attacker, so RSR is a conservative lower bound on wording-dependence.</li>
      <li>Category names follow the SORRY-Bench 44-class taxonomy; ids are the dataset's own.</li>
    </ul>
    <h3>Sources</h3>
    {table(["Input", "Path / source"], [[esc(a), f"<code>{esc(b)}</code>"] for a, b in srcs])}
  </div>
  </div>
</section>"""


# =============================================================================
# Page shell (CSS / JS kept out of f-strings on purpose)
# =============================================================================
CSS = r"""
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap");
/* DFKI corporate design (dfki.de, 2024): ink #06171C, deep violet #201E5F, magenta #EC619F, Abisko green #6ABFA3,
   sky #7ABFE7, Guam blue #1D3A8F, moon grey #D7DBDD; typeface IBM Plex Sans / Plex Mono. "-ink" = text-safe shade. */
:root{
  color-scheme: light;
  --page:#f6f8f8; --surface:#ffffff; --card:#ffffff; --ink:#06171c; --ink2:#3d4c52; --muted:#76868c;
  --grid:#dde3e5; --axis:#b8c2c6; --ring:rgba(6,23,28,.10); --wash:#eef3f3;
  --brand:#201e5f; --brand2:#ec619f; --accent:#6abfa3; --sky:#7abfe7;
  --comply:#ec619f; --comply-ink:#b83a75; --refuse:#1d3a8f; --fix:#6abfa3; --fix-ink:#2f7d63; --none:#c9d1d4; --neutral:#4a5a60;
  --good:#6abfa3; --good-ink:#2f7d63; --warn:#f7a712; --warn-ink:#8a5a00; --crit:#d2386c;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --page:#061418; --surface:#0b1d22; --card:#0e2229; --ink:#f1f5f5; --ink2:#bccacd; --muted:#7f9298;
    --grid:#1b333a; --axis:#2b4750; --ring:rgba(255,255,255,.10); --wash:#12292f;
    --brand:#b9b6f2; --brand2:#f07fb0; --accent:#6abfa3; --sky:#7abfe7;
    --comply:#f07fb0; --comply-ink:#f59cc2; --refuse:#7abfe7; --fix:#6abfa3; --fix-ink:#8fd3bb; --none:#3e5258; --neutral:#bccacd;
    --good-ink:#8fd3bb; --warn-ink:#f7a712;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --page:#061418; --surface:#0b1d22; --card:#0e2229; --ink:#f1f5f5; --ink2:#bccacd; --muted:#7f9298;
  --grid:#1b333a; --axis:#2b4750; --ring:rgba(255,255,255,.10); --wash:#12292f;
  --brand:#b9b6f2; --brand2:#f07fb0; --accent:#6abfa3; --sky:#7abfe7;
  --comply:#f07fb0; --comply-ink:#f59cc2; --refuse:#7abfe7; --fix:#6abfa3; --fix-ink:#8fd3bb; --none:#3e5258; --neutral:#bccacd;
  --good-ink:#8fd3bb; --warn-ink:#f7a712;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:16px}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
a{color:inherit}
code,pre{font-family:var(--mono);font-size:.88em}
code{background:var(--wash);padding:1px 5px;border-radius:4px}
.layout{display:grid;grid-template-columns:220px minmax(0,1fr);min-height:100vh}
nav.side{position:sticky;top:0;height:100vh;overflow-y:auto;padding:28px 14px;border-right:1px solid var(--grid);background:var(--surface)}
nav.side .brand{display:flex;align-items:center;gap:11px;padding:0 6px 18px 10px;margin-bottom:10px;border-bottom:1px solid var(--grid)}
nav.side .brand .logo{flex:none;width:40px;height:34px;color:var(--brand);display:block}
nav.side .brand .logo svg{width:100%;height:100%;display:block}
nav.side .brand .bt{min-width:0}
nav.side .brand b{display:block;font-weight:700;font-size:15px;letter-spacing:-.01em;color:var(--brand)}
nav.side .brand .bt span{display:block;color:var(--muted);font-weight:400;font-size:11.5px;line-height:1.35}
nav.side .toc{position:relative;margin-left:4px;padding-left:12px}
.hero>.brand{display:flex;align-items:center;gap:11px;margin:0 0 14px}
.hero>.brand .logo{flex:none;width:40px;height:34px;color:var(--brand);display:block}.hero>.brand .logo svg{width:100%;height:100%;display:block}
.hero>.brand b{display:block;font-weight:700;font-size:15px;color:var(--brand)}.hero>.brand .bt span{display:block;color:var(--muted);font-size:11.5px}
nav.side .toc::before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;border-radius:2px;background:var(--grid)}
nav.side .toc-ind{position:absolute;left:0;top:0;width:2px;height:0;border-radius:2px;background:linear-gradient(var(--accent),var(--brand2));opacity:0;transition:transform .32s cubic-bezier(.2,.7,.2,1),height .32s cubic-bezier(.2,.7,.2,1),opacity .2s}
nav.side .toc-ind.on{opacity:1}
nav.side a{display:block;text-decoration:none;color:var(--ink2);font-size:13px;padding:6px 10px;border-radius:7px;transition:color .18s,background-color .18s}
nav.side a:hover{background:var(--wash);color:var(--ink)}
nav.side a.active{color:var(--brand);font-weight:600}
nav.side a:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
nav.side .toc-sub{display:block;margin:1px 0 6px 10px;border-left:1px dashed color-mix(in srgb,var(--grid) 80%,var(--muted))}
nav.side .toc-sub a{font-size:12px;line-height:1.3;padding:4px 10px 4px 12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
nav.side .toc-sub a.deep{padding-left:24px;font-size:11.5px}
nav.side .toc-sub a.tabref{color:var(--ink2);font-weight:500}
nav.side .toc-sub a.tabref::before{content:"▾";display:inline-block;width:12px;margin-left:-2px;color:var(--muted)}
nav.side .toc-sub a.tabref.on::before{color:var(--accent)}
nav.side .toc-sub a:hover{color:var(--ink)}
nav.side .toc-sub a.active{color:var(--brand);font-weight:600}
nav.side a.active-parent{color:var(--ink);font-weight:600}
nav.side .toc>a{display:flex;align-items:baseline;gap:0}
nav.side .toc>a .n{flex:none}
nav.side .toc>a .rqchip{margin-left:auto;padding-left:5px;flex:none;align-self:center}
.rqchip{display:inline-block;vertical-align:1px;margin-left:6px;padding:0 5px;border-radius:999px;font:600 9px/15px var(--mono);white-space:nowrap;letter-spacing:.04em;color:var(--brand);background:color-mix(in srgb,var(--brand) 10%,transparent);border:1px solid color-mix(in srgb,var(--brand) 26%,transparent)}
h2 .rqchip{font-size:11px;line-height:19px;vertical-align:4px;margin-left:10px}
nav.side a.active .rqchip{background:color-mix(in srgb,var(--accent) 22%,transparent)}
.sidelegend{margin:22px 6px 0 10px;padding:14px 0 0;border-top:1px solid var(--grid);font-size:11.5px;color:var(--ink2)}
.sidelegend .lt{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.sidelegend .lk{display:grid;grid-template-columns:14px 1fr;gap:3px 8px;align-items:center;margin:0 0 12px}
.sidelegend .sw{width:12px;height:12px;border-radius:3px;display:block}
.sidelegend .sub{font-size:10.5px;color:var(--muted);margin:2px 0 6px;line-height:1.35}
.sidelegend .bar{height:10px;border-radius:5px;margin:4px 0 3px}
.sidelegend .bar.div{background:linear-gradient(90deg,var(--comply),color-mix(in oklab,var(--comply) 40%,var(--surface)) 30%,var(--surface) 50%,color-mix(in oklab,var(--sky) 40%,var(--surface)) 70%,var(--sky));border:1px solid var(--grid)}
.sidelegend .bar.seq{background:linear-gradient(90deg,color-mix(in oklab,var(--comply) 6%,var(--surface)),color-mix(in oklab,var(--comply) 85%,var(--surface)));border:1px solid var(--grid)}
.sidelegend .ticks{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;color:var(--muted)}
.rqgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.rqcard{position:relative;padding:16px 18px 14px;border:1px solid var(--grid);border-radius:14px;background:var(--card);min-width:0}
.rqcard::before{content:"";position:absolute;left:0;top:14px;bottom:14px;width:3px;border-radius:0 3px 3px 0;background:var(--accent)}
.rqcard.primary::before{background:linear-gradient(var(--brand2),var(--brand))}
.rqcard .rqh{display:flex;align-items:baseline;gap:10px;margin:0 0 6px}
.rqcard .rqh b{font:700 13px var(--mono);color:var(--brand)}
.rqcard .rqh span{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.rqcard .q{font-size:15px;font-weight:600;line-height:1.35;margin:0 0 8px;color:var(--ink)}
.rqcard p{margin:0 0 8px;font-size:13px;line-height:1.5;color:var(--ink2)}
.rqcard .how{font-size:12px;color:var(--muted)}
.rqcard .ans{margin:10px 0 6px;padding:9px 11px;border-radius:9px;background:var(--wash);font-size:12.5px;line-height:1.5;color:var(--ink)}
.rqcard .ans b{color:var(--brand)}
.rqcard .where{font-size:12px;margin:6px 0 0}
.rqcard .where a{color:var(--brand);text-decoration:none;border-bottom:1px solid color-mix(in srgb,var(--brand) 30%,transparent);margin-right:10px;white-space:nowrap}
.rqframe{margin:0 0 14px;font-size:13.5px;line-height:1.55;color:var(--ink2);max-width:80ch}
.rqframe b{color:var(--ink)}
nav.side .grp{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);padding:14px 10px 4px}
#readbar{position:fixed;top:0;left:0;height:3px;width:100%;transform-origin:0 50%;transform:scaleX(0);z-index:60;background:linear-gradient(90deg,var(--accent),var(--sky),var(--brand2));pointer-events:none}
nav.side a .n{font-family:var(--mono);color:var(--muted);margin-right:8px;font-size:11px}
.themebtn{margin:18px 10px 0;font:inherit;font-size:12px;color:var(--ink2);background:var(--card);border:1px solid var(--grid);border-radius:999px;padding:6px 12px;cursor:pointer}
.themebtn:hover{color:var(--ink);border-color:var(--axis)}
main{padding:40px clamp(16px,4vw,56px) 80px;max-width:1240px;width:100%;min-width:0}
.layout>*{min-width:0}
h3.gap{margin-top:18px}
.tbl code{word-break:break-all}
.hero{position:relative;padding:26px clamp(18px,3vw,32px) 22px;border-radius:18px;border:1px solid var(--grid);background:radial-gradient(55% 90% at 100% 0%,color-mix(in srgb,var(--accent) 30%,transparent),transparent 70%),radial-gradient(45% 80% at 0% 100%,color-mix(in srgb,var(--sky) 24%,transparent),transparent 70%),radial-gradient(35% 60% at 72% 100%,color-mix(in srgb,var(--brand2) 14%,transparent),transparent 70%),var(--card)}
.hero h1{font-size:clamp(26px,3.4vw,38px);line-height:1.12;letter-spacing:-.015em;margin:6px 0 12px;font-weight:700;color:var(--brand)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0}
.lede{font-size:16px;color:var(--ink2);max-width:76ch;margin:0 0 14px}
.lede b{color:var(--ink)}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink2);margin:8px 0}
.meta b{color:var(--ink);font-weight:600}
.sec{margin-top:52px}
h2{font-size:20px;letter-spacing:-.01em;margin:0 0 14px;display:flex;align-items:baseline;gap:12px}
h2 .sn{font-family:var(--mono);font-size:12px;color:var(--muted);font-weight:500}
h3{font-size:15px;margin:0 0 6px;font-weight:650}
.panel{background:var(--card);border:1px solid var(--ring);border-radius:14px;padding:20px 22px;margin:14px 0}
.note{color:var(--ink2);font-size:13.5px;margin:0 0 12px;max-width:92ch}
.flag{background:color-mix(in oklab,var(--warn) 14%,var(--card));border:1px solid color-mix(in oklab,var(--warn) 45%,var(--card));color:var(--ink);border-radius:10px;padding:9px 13px;font-size:13.5px;margin:10px 0}
.okline{color:var(--good-ink);font-weight:600;margin:0 0 10px}
.warnbox{border-color:color-mix(in oklab,var(--warn) 55%,var(--card))}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:14px}
.grid2>.panel{margin:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr));gap:12px;margin:12px 0}
.tile{background:var(--card);border:1px solid var(--ring);border-radius:12px;padding:14px 16px}
.tile-emph{border-color:color-mix(in oklab,var(--fix) 60%,var(--card));box-shadow:inset 3px 0 0 var(--fix)}
.tile-l{font-size:12.5px;color:var(--ink2)}
.tile-v{font-size:30px;font-weight:650;letter-spacing:-.01em;line-height:1.15;margin:4px 0 2px}
.tile-s{font-size:12.5px;color:var(--ink2)}
.take{margin:0;padding-left:18px}
.take li{margin:6px 0;max-width:92ch}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:11.5px;font-weight:600;padding:2px 9px;border-radius:999px;border:1px solid var(--grid);color:var(--ink2);background:var(--card);white-space:nowrap}
.chip-ok{color:var(--good-ink);border-color:color-mix(in oklab,var(--good) 45%,var(--card))}
.chip-warn{color:var(--warn-ink);border-color:color-mix(in oklab,var(--warn) 60%,var(--card))}
.chip-matched-scale{color:var(--ink);border-color:var(--fix-ink)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--ink2);margin:6px 0 4px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.lg-dot{background:var(--comply);border-radius:50%!important}
.lg-dia{background:transparent;border:2px solid var(--ink);transform:rotate(45deg) scale(.8)}
.lg-mute{background:var(--none);border-radius:50%!important}
.lg-ramp{width:44px!important;background:linear-gradient(90deg,var(--surface),var(--comply))}
.lg-adj{background:transparent;border:2px solid var(--ink)}
.lg.r-comply{background:var(--comply)}.lg.r-refuse{background:var(--refuse)}.lg.r-fix{background:var(--fix)}.lg.r-none{background:var(--none)}
/* svg */
.chart{display:block;max-width:100%;height:auto;overflow:visible}
.chart text{fill:var(--ink2);font-size:12px;font-family:var(--sans)}
.chart .ax{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.chart .lbl{fill:var(--ink2)}
.chart .strong{fill:var(--ink);font-weight:600}
.chart .val{fill:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.chart .halo{paint-order:stroke;stroke:var(--card);stroke-width:4px;stroke-linejoin:round}
.chart .stage{fill:var(--muted);font-size:11px;letter-spacing:.08em;text-transform:uppercase}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .axis{stroke:var(--axis);stroke-width:1}
.chart .refline{stroke:var(--ink2);stroke-width:1.5}
.chart .whisk{stroke:var(--ink2);stroke-width:2;stroke-linecap:round}
.chart .pt{fill:var(--comply);stroke:var(--card);stroke-width:2}
.chart .corr{fill:none;stroke:var(--ink);stroke-width:1.8}
.chart .row.muted .pt{fill:var(--none)}
.chart .row.muted .whisk{stroke:var(--muted)}
.chart .hit{fill:transparent}
.chart .row:hover .hit,.chart .row:focus .hit{fill:var(--wash)}
.chart .row:focus{outline:none}
.chart .bar-comply{fill:var(--comply)}.chart .bar-neutral{fill:var(--neutral)}
.chart .bar-pass{fill:var(--neutral)}.chart .bar-fail{fill:var(--axis)}
.chart .node.r-ink{fill:var(--ink2)}
.chart .r-comply{fill:var(--comply)}.chart .r-refuse{fill:var(--refuse)}.chart .r-fix{fill:var(--fix)}.chart .r-none{fill:var(--none)}
.chart .band{opacity:.32;transition:opacity .15s}
.chart .band:hover{opacity:.7}
.chart .leader{stroke:var(--axis);stroke-width:1}
.chart .line-emph{fill:none;stroke:var(--comply);stroke-width:2.5;stroke-linejoin:round}
.chart .line-mute{fill:none;stroke:var(--none);stroke-width:1.6;stroke-linejoin:round}
.chart .dot-emph{fill:var(--comply);stroke:var(--card);stroke-width:2}
.chart .dot-mute{fill:var(--none);stroke:var(--card);stroke-width:2}
.chart .hitpt{fill:transparent}
.chart [tabindex]:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.scroll,.tscroll,.strip-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.scroll,.tscroll,.strip-scroll{max-width:100%}.panel,.tile,[class*="grid"]>*{min-width:0}
p a,p code,li a,li code,.note a,.note code,.lede a{overflow-wrap:anywhere;word-break:break-word}
.pos-wrap{overflow-x:auto;max-width:100%;-webkit-overflow-scrolling:touch}
.scroll>svg{min-width:640px}
.tscroll.tall{max-height:640px;overflow-y:auto}
/* lineage strip */
.strip{display:flex;align-items:stretch;gap:0;min-width:760px;padding:6px 2px}
.mnode{flex:1 1 0;border:1px solid var(--ring);border-radius:12px;padding:12px 8px;background:var(--surface);text-align:center;min-width:92px}
.mgen{font-size:12px;color:var(--ink2)}.mname{font-size:24px;font-weight:700;line-height:1.1}
.mrel{font-size:11.5px;color:var(--muted);font-family:var(--mono)}
.mstat{font-size:12.5px;margin-top:6px;font-variant-numeric:tabular-nums}.mstat span{color:var(--muted)}
.medge{flex:0 0 96px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:12px;padding:0 6px;text-align:center}
.earrow{font-size:20px;color:var(--axis);line-height:1}
.epm{font-weight:650}.ersr{color:var(--ink2)}
.ekind{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.e-matched-scale .earrow{color:var(--fix-ink)}.e-matched-scale .ekind{color:var(--ink)}
.mnode:focus-visible,.medge:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
/* tabs */
.tablist{display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--grid);padding-bottom:8px;margin:6px 0 10px}
.tab{font:inherit;display:flex;flex-direction:column;align-items:flex-start;gap:1px;padding:8px 14px;border:1px solid var(--grid);border-radius:10px;background:var(--card);color:var(--ink2);cursor:pointer;text-align:left}
.tab .tt{font-weight:600;font-size:13.5px;color:var(--ink)}.tab .ts{font-size:11.5px;color:var(--muted)}
.tab:hover{border-color:var(--axis)}
.tab[aria-selected="true"]{border-color:var(--ink);box-shadow:inset 0 -3px 0 var(--fix)}
.tab.posthoc{border-style:dashed}
.tab:focus-visible,.fbtn:focus-visible,.themebtn:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.pairhead{display:flex;flex-wrap:wrap;align-items:center;gap:8px 10px;margin:8px 0}
.pairhead h3{font-size:18px;margin:0}
/* 2x2 */
.m2{display:grid;grid-template-columns:minmax(90px,auto) 1fr 1fr;gap:8px;align-items:stretch}
.mh{font-size:12px;color:var(--ink2);font-weight:600;display:flex;align-items:flex-end;justify-content:center;text-align:center;padding:2px 4px}
.mh.mrow{justify-content:flex-end;align-items:center;text-align:right}
.mc{border-radius:10px;padding:14px;min-height:84px;display:flex;flex-direction:column;justify-content:center;border:1px solid var(--ring)}
.mc b{font-size:28px;line-height:1;font-variant-numeric:tabular-nums}
.mc span{font-size:12px;color:var(--ink2);margin-top:6px}
.mc.c-comply{background:color-mix(in oklab,var(--comply) 22%,var(--card))}
.mc.c-regress{background:color-mix(in oklab,var(--comply) 12%,var(--card));background-image:repeating-linear-gradient(45deg,transparent 0 7px,color-mix(in oklab,var(--comply) 22%,transparent) 7px 9px)}
.mc.c-refuse{background:color-mix(in oklab,var(--refuse) 14%,var(--card))}
.mc.c-fix{background:color-mix(in oklab,var(--fix) 26%,var(--card));border-color:var(--fix-ink)}
.mc:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
/* matrix */
.mx{display:grid;gap:4px;min-width:640px}
.mx-corner{font-size:11px;color:var(--muted);display:flex;align-items:flex-end;padding:4px}
.mx-h{font-size:12.5px;font-weight:600;display:flex;align-items:flex-end;justify-content:center;padding:4px}
.mx-h.mx-r{justify-content:flex-end;align-items:center}
.mx-c{border-radius:8px;min-height:58px;display:flex;flex-direction:column;align-items:center;justify-content:center;border:1px solid var(--ring);background:var(--surface);position:relative}
.mx-c b{font-size:16px;font-variant-numeric:tabular-nums}.mx-c span{font-size:11px;color:var(--ink2)}
.mx-na{background:transparent;border:none}
.mx-adj{border:2px solid var(--ink)}
.mx-ms::after{content:"◆";position:absolute;top:3px;right:6px;font-size:10px;color:var(--ink2)}
.mx-small b{color:var(--muted)}
.mx-c:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
/* tables */
.tbl{border-collapse:collapse;width:100%;font-size:13px}
.tbl th,.tbl td{padding:7px 10px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:middle}
.tbl thead th,.tbl tr:first-child th[scope=col]{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.05em;background:var(--card);position:sticky;top:0;z-index:1}
.tbl .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl th[scope=row]{font-weight:500}
.tbl tr.dom th,.tbl tr.dom td{font-weight:650;border-top:1px solid var(--axis);background:var(--wash)}
.tbl tr.dom th span{font-weight:400;color:var(--muted);font-size:12px}
.hmtbl td.hm{min-width:62px}
.cnum{font-family:var(--mono);font-size:11px;color:var(--muted);margin-right:4px}
.sub{color:var(--muted);font-size:11.5px;font-weight:400}
details.tv{margin-top:10px}
details.tv>summary{cursor:pointer;font-size:12.5px;color:var(--ink2);width:max-content;max-width:100%}
details.tv[open]>summary{margin-bottom:8px}
td.hm:focus-visible{outline:2px solid var(--ink);outline-offset:-2px}
/* gate */
.hgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px}
.hcell{border:1px solid var(--ring);border-radius:10px;padding:8px 10px;background:var(--surface)}
/* cases */
.filters{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px}
.fbtn{font:inherit;font-size:12.5px;padding:5px 12px;border-radius:999px;border:1px solid var(--grid);background:var(--card);color:var(--ink2);cursor:pointer}
.fbtn span{color:var(--muted);margin-left:4px}
.fbtn.active{background:var(--ink);color:var(--card);border-color:var(--ink)}.fbtn.active span{color:var(--card);opacity:.75}
details.case{border:1px solid var(--ring);border-radius:10px;margin:6px 0;background:var(--surface)}
details.case>summary{list-style:none;cursor:pointer;display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;padding:10px 14px;font-size:13px}
details.case>summary::-webkit-details-marker{display:none}
details.case>summary::before{content:"+";font-family:var(--mono);color:var(--muted);width:10px}
details.case[open]>summary::before{content:"–"}
.cid{font-family:var(--mono);font-size:12px;color:var(--muted)}
.ccat{font-weight:500}
.sens{margin-left:auto;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--warn-ink);border:1px solid color-mix(in oklab,var(--warn) 60%,var(--card));border-radius:999px;padding:1px 7px}
.pill{font-size:11.5px;font-weight:600;border-radius:999px;padding:2px 9px;border:1px solid var(--grid);color:var(--ink2)}
.pill-comply{border-color:var(--comply-ink)}.pill-refuse{border-color:var(--refuse)}.pill-none{border-style:dashed}
.cbody{padding:0 14px 14px}
.txt{font-family:var(--mono);font-size:12px;line-height:1.5;color:var(--ink);background:var(--wash);border-radius:8px;padding:9px 11px;margin:6px 0;white-space:pre-wrap;word-break:break-word;max-height:190px;overflow:auto}
.tl{display:block;font-family:var(--sans);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:3px}
.vlist{margin-top:12px;border-top:1px solid var(--grid);padding-top:8px}
.vrow{border-left:3px solid var(--grid);padding:4px 0 4px 12px;margin:8px 0}
.vrow.v-comply{border-left-color:var(--comply-ink)}.vrow.v-refuse{border-left-color:var(--refuse)}
.vhead,.vrow{font-size:12.5px}
.vtc{font-family:var(--mono);font-size:11.5px;margin-right:10px}
.vv{font-weight:600}
.v-comply .vv{color:var(--ink)}.v-refuse .vv{color:var(--ink2)}
pre.cmd{background:var(--wash);border-radius:8px;padding:10px 12px;overflow-x:auto;white-space:pre}
/* tooltip */
#tip{position:fixed;z-index:100;pointer-events:none;max-width:340px;background:var(--ink);color:var(--card);font-size:12.5px;line-height:1.45;padding:8px 10px;border-radius:8px;white-space:pre-line;box-shadow:0 4px 18px rgba(0,0,0,.18)}
footer{margin-top:48px;padding-top:14px;border-top:1px solid var(--grid);font-size:12px;color:var(--muted)}
@media (max-width:980px){
  .layout{display:block}
  nav.side{position:sticky;top:0;z-index:50;height:auto;border-right:none;border-bottom:1px solid var(--grid);padding:10px 12px;background:color-mix(in srgb,var(--surface) 92%,transparent);backdrop-filter:saturate(1.4) blur(8px)}
  nav.side .brand{padding:0 0 6px;margin-bottom:6px;border-bottom:none}
  nav.side .brand .logo{width:30px;height:26px}
  nav.side .brand .bt span{display:none}
  nav.side .toc{display:flex;gap:2px;overflow-x:auto;scrollbar-width:none;margin:0;padding:0}
  nav.side .toc::before,nav.side .toc-ind,nav.side .grp,nav.side .toc-sub,nav.side .toc-sub.open,.sidelegend,nav.side a .rqchip{display:none}
  nav.side a{white-space:nowrap;padding:5px 10px;border-radius:999px}
  nav.side a.active{background:color-mix(in srgb,var(--accent) 22%,transparent)}
  .themebtn{position:absolute;top:10px;right:12px;margin:0;padding:4px 10px;font-size:11px}
}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}.chart .band{transition:none}nav.side .toc-ind,nav.side a{transition:none}}
@media print{nav.side,.themebtn,.sidelegend,#tip,.filters,#readbar{display:none!important}.layout{display:block}.panel,.tile{break-inside:avoid}details.case[open]{break-inside:avoid}}
"""

JS = r"""
(function(){
  var root = document.documentElement, KEY = 'rb_lineage_theme';
  var btn = document.getElementById('themebtn');
  function applyTheme(t){
    if (t === 'light' || t === 'dark') root.setAttribute('data-theme', t); else root.removeAttribute('data-theme');
    if (btn) btn.textContent = 'Theme: ' + (t === 'light' ? 'light' : t === 'dark' ? 'dark' : 'system');
    if (btn) btn.setAttribute('data-t', t);
  }
  var saved = 'system';
  try { saved = localStorage.getItem(KEY) || 'system'; } catch (e) {}
  applyTheme(saved);
  if (btn) btn.addEventListener('click', function(){
    var cur = btn.getAttribute('data-t') || 'system';
    var nxt = cur === 'system' ? 'light' : cur === 'light' ? 'dark' : 'system';
    applyTheme(nxt);
    try { localStorage.setItem(KEY, nxt); } catch (e) {}
  });

  // tooltip: hover + keyboard focus, text only (textContent, never innerHTML)
  var tip = document.getElementById('tip');
  function place(x, y){
    var w = tip.offsetWidth, h = tip.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    var left = x + 14, top = y + 14;
    if (left + w > vw - 8) left = Math.max(8, x - w - 14);
    if (top + h > vh - 8) top = Math.max(8, y - h - 14);
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  function show(el, x, y){
    var t = el.getAttribute('data-tip'); if (!t) return;
    tip.textContent = t; tip.hidden = false; place(x, y);
  }
  function hide(){ tip.hidden = true; }
  document.addEventListener('pointermove', function(e){
    var el = e.target.closest ? e.target.closest('[data-tip]') : null;
    if (el) show(el, e.clientX, e.clientY); else hide();
  });
  document.addEventListener('pointerleave', hide);
  document.addEventListener('focusin', function(e){
    var el = e.target.closest ? e.target.closest('[data-tip]') : null;
    if (!el) { hide(); return; }
    var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.bottom);
  });
  document.addEventListener('focusout', hide);
  document.addEventListener('keydown', function(e){ if (e.key === 'Escape') hide(); });
  window.addEventListener('scroll', hide, {passive: true});

  // tabs (WAI-ARIA pattern) + deep link via #panel-<id>
  Array.prototype.forEach.call(document.querySelectorAll('[role="tablist"]'), function(list){
    var tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"]'));
    function select(i, focus){
      tabs.forEach(function(t, j){
        var on = (i === j);
        t.setAttribute('aria-selected', on ? 'true' : 'false');
        t.tabIndex = on ? 0 : -1;
        var p = document.getElementById(t.getAttribute('aria-controls'));
        if (p) p.hidden = !on;
      });
      if (focus) tabs[i].focus();
      document.dispatchEvent(new CustomEvent('rb:tabchange'));
    }
    tabs.forEach(function(t, i){
      t.addEventListener('click', function(){ select(i, false); });
      t.addEventListener('keydown', function(e){
        var k = e.key, n = tabs.length;
        if (k === 'ArrowRight') { e.preventDefault(); select((i + 1) % n, true); }
        else if (k === 'ArrowLeft') { e.preventDefault(); select((i - 1 + n) % n, true); }
        else if (k === 'Home') { e.preventDefault(); select(0, true); }
        else if (k === 'End') { e.preventDefault(); select(n - 1, true); }
      });
    });
    var h = (location.hash || '').slice(1);
    if (h) tabs.forEach(function(t, i){ if (t.getAttribute('aria-controls') === h) select(i, false); });
  });

  // case filters (scoped to their own list)
  Array.prototype.forEach.call(document.querySelectorAll('.cases'), function(box){
    var btns = box.querySelectorAll('.fbtn');
    Array.prototype.forEach.call(btns, function(b){
      b.addEventListener('click', function(){
        var f = b.getAttribute('data-filter');
        Array.prototype.forEach.call(btns, function(x){
          var on = (x === b); x.classList.toggle('active', on); x.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
        Array.prototype.forEach.call(box.querySelectorAll('details.case'), function(d){
          d.hidden = !(f === 'all' || d.getAttribute('data-outcome') === f);
        });
      });
    });
  });

  // table of contents: nested sections + scrollspy + sliding active indicator + smooth scrolling + reading progress
  (function(){
    var nav = document.querySelector('nav.side'); if (!nav) return;
    var toc = nav.querySelector('.toc') || nav;
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var ind = document.createElement('span'); ind.className = 'toc-ind'; ind.setAttribute('aria-hidden', 'true');
    if (toc !== nav) toc.appendChild(ind);
    var bar = document.createElement('div'); bar.id = 'readbar'; bar.setAttribute('aria-hidden', 'true');
    document.body.appendChild(bar);
    var tops = Array.prototype.slice.call(toc.children).filter(function(a){ return a.matches && a.matches('a[href^="#"]'); });
    function target(a){ return document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1))); }
    // research-question chips (window.RB_RQ = {sectionId: ["RQ1", ...]}) on the nav entry and the section heading
    var RQ = window.RB_RQ || {};
    tops.forEach(function(a){
      var id = a.getAttribute('href').slice(1), el = target(a);
      var qs = RQ[id] || []; if (!qs.length) return;
      var c = document.createElement('span'); c.className = 'rqchip';            // one compact chip in the nav: RQ1·2·3
      c.textContent = 'RQ' + qs.map(function(q){ return q.replace(/^RQ/, ''); }).join('·'); c.title = qs.join(', ');
      a.appendChild(c);
      var h = el && el.querySelector('h2');
      if (h) qs.forEach(function(q){
        if (h.querySelector('.rqchip[data-q="' + q + '"]')) return;
        var c2 = document.createElement('span'); c2.className = 'rqchip'; c2.textContent = q; c2.setAttribute('data-q', q); h.appendChild(c2);
      });
    });
    var items = [], current = -1, locked = false, idle = null, hardStop = null;
    function slug(t){ return t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'part'; }
    function build(){       // every section fully expanded: benchmark tabs, each with its panel's h3s, then the section's own h3s
      Array.prototype.forEach.call(toc.querySelectorAll('.toc-sub'), function(d){ d.remove(); });
      items = [];
      tops.forEach(function(a){
        var sec = target(a);
        items.push({a: a, el: sec, top: a});
        if (!sec) return;
        var seen = {}, subs = [], byPanel = {};
        Array.prototype.forEach.call(sec.querySelectorAll('[role="tab"], h3'), function(h){
          if (h.getAttribute('role') === 'tab') {
            var pid = h.getAttribute('aria-controls'), pan = pid && document.getElementById(pid);
            var lab = h.querySelector('.tt'), tt = ((lab || h).textContent || '').replace(/\s+/g, ' ').trim();
            if (pan && tt) subs.push({el: pan, t: tt, tab: h, on: h.getAttribute('aria-selected') === 'true'});
            return;
          }
          var pan = h.closest('[role="tabpanel"]');
          if (h.closest('details.case, details.mech, .tile, figure, .pairhead')) return;
          if (pan ? h.closest('details:not([open])') : (h.offsetParent === null || h.closest('[hidden]'))) return;
          var t = (h.textContent || '').replace(/\s+/g, ' ').trim(); if (!t) return;
          if (!h.id) { var b = (pan ? pan.id : sec.id) + '--' + slug(t), id = b, n = 2; while (document.getElementById(id)) id = b + '-' + n++; h.id = id; }
          var key = (pan ? pan.id + ':' : '') + t;
          if (seen[key]) return; seen[key] = 1;
          var x = {el: h, t: t, deep: !!pan};
          if (pan) (byPanel[pan.id] = byPanel[pan.id] || []).push(x); else subs.push(x);
        });
        var ordered = [];
        subs.forEach(function(x){
          ordered.push(x);
          if (x.tab) (byPanel[x.el.id] || []).forEach(function(d){ d.tab = x.tab; ordered.push(d); });
        });
        if (!ordered.length) return;
        var box = document.createElement('div'); box.className = 'toc-sub open';
        ordered.forEach(function(x){
          var s = document.createElement('a');
          s.href = '#' + x.el.id; s.textContent = x.t; s.title = x.t;
          s.className = x.deep ? 'deep' : x.tab ? 'tabref' + (x.on ? ' on' : '') : '';
          box.appendChild(s);
          var k = items.length;
          items.push({a: s, el: x.el, top: a, sub: true, tab: x.tab});
          if (x.tab) s.addEventListener('click', function(ev){     // a heading in another benchmark: switch tab first
            ev.preventDefault();
            if (x.tab.getAttribute('aria-selected') !== 'true') x.tab.click();   // fires rb:tabchange -> rebuild
            var j = items.findIndex(function(y){ return y.el === x.el; });
            if (j >= 0) go(j)(ev);
          });
          else s.addEventListener('click', go(k));
        });
        a.parentNode.insertBefore(box, a.nextSibling);
      });
    }
    function place(i){
      if (i === current) return;
      current = i;
      var it = items[i];
      items.forEach(function(x, k){
        x.a.classList.toggle('active', k === i);
        if (k === i) x.a.setAttribute('aria-current', 'location'); else x.a.removeAttribute('aria-current');
      });
      tops.forEach(function(a){
        var on = !!it && it.top === a;
        a.classList.toggle('active-parent', on && it.sub === true);
        var box = a.nextElementSibling;
      });
      if (!it) { ind.classList.remove('on'); return; }
      var a = it.a.offsetParent === null ? it.top : it.a;           // collapsed / mobile: point at the section
      ind.style.transform = 'translateY(' + a.offsetTop + 'px)';
      ind.style.height = a.offsetHeight + 'px';
      ind.classList.add('on');
      var sc = toc.scrollWidth > toc.clientWidth ? toc : nav;
      if (sc === toc) {
        var l = it.top.offsetLeft - toc.offsetLeft;
        if (l < toc.scrollLeft || l + it.top.offsetWidth > toc.scrollLeft + toc.clientWidth)
          toc.scrollTo({left: l - 24, behavior: reduce ? 'auto' : 'smooth'});
      } else if (nav.scrollHeight > nav.clientHeight) {
        var r = a.getBoundingClientRect(), n = nav.getBoundingClientRect();
        if (r.top < n.top + 40 || r.bottom > n.bottom - 120)
          nav.scrollTo({top: nav.scrollTop + (r.top - n.top) - n.height / 3, behavior: reduce ? 'auto' : 'smooth'});
      }
    }
    function topInset(){   // on narrow screens the TOC is a sticky top bar: keep headings below it
      var st = window.matchMedia('(max-width: 980px)').matches && getComputedStyle(nav).position === 'sticky';
      var inset = st ? Math.ceil(nav.getBoundingClientRect().height) + 12 : 16;
      document.documentElement.style.scrollPaddingTop = inset + 'px';
      return inset;
    }
    var inset = topInset();
    function unlock(){ locked = false; clearTimeout(idle); clearTimeout(hardStop); spy(); }
    function spy(){
      var docH = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = 'scaleX(' + (docH > 0 ? Math.min(1, window.scrollY / docH) : 0) + ')';
      if (locked) { clearTimeout(idle); idle = setTimeout(unlock, 140); return; }   // a click's smooth scroll is still running
      var probe = Math.max(window.innerHeight * 0.28, inset + 24), best = -1;
      for (var k = 0; k < items.length; k++) {
        var el = items[k].el;
        if (el && el.offsetParent !== null && el.getBoundingClientRect().top - probe <= 0) best = k;
      }
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {   // at the very bottom
        for (var j = items.length - 1; j >= 0; j--) if (items[j].el && !items[j].sub) { best = Math.max(best, j); break; }
      }
      place(best < 0 && items.length ? 0 : best);
    }
    function go(k){
      return function(ev){
        var el = items[k] && items[k].el; if (!el) return;
        ev.preventDefault();
        place(k);
        locked = true;                                        // the indicator jumps to the target, not through every section
        clearTimeout(idle); idle = setTimeout(unlock, 200); clearTimeout(hardStop); hardStop = setTimeout(unlock, 2500);
        el.scrollIntoView({behavior: reduce ? 'auto' : 'smooth', block: 'start'});
        if (history.replaceState) history.replaceState(null, '', '#' + el.id);
        el.setAttribute('tabindex', '-1'); el.focus({preventScroll: true});
      };
    }
    build();
    tops.forEach(function(a){ a.addEventListener('click', function(ev){ go(items.findIndex(function(x){ return x.a === a; }))(ev); }); });
    var ticking = false;
    function onScroll(){ if (!ticking) { ticking = true; requestAnimationFrame(function(){ ticking = false; spy(); }); } }
    window.addEventListener('scroll', onScroll, {passive: true});
    if ('onscrollend' in window) window.addEventListener('scrollend', function(){ if (locked) unlock(); });
    window.addEventListener('resize', function(){ inset = topInset(); var c = current; current = -1; place(c); onScroll(); });
    document.addEventListener('rb:tabchange', function(){ build(); current = -1; spy(); });
    document.addEventListener('toggle', function(e){ if (e.target && e.target.matches && e.target.matches('details') && !e.target.closest('details.case')) { build(); current = -1; spy(); } }, true);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function(){ var c = current; current = -1; place(c); });
    spy();
  })();
})();
"""

_LOGO = None


def brand_html(subtitle: str) -> str:
    """sidebar brand: the DFKI wordmark (tools/assets/dfki_logo_small.svg, single colour, inherits --brand) + name."""
    global _LOGO
    if _LOGO is None:
        f = Path(__file__).resolve().parent / "assets" / "dfki_logo_small.svg"
        _LOGO = f.read_text() if f.exists() else ""
    logo = f'<span class="logo" title="DFKI">{_LOGO}</span>' if _LOGO else ""
    return f'<div class="brand">{logo}<span class="bt"><b>RecycleBench</b><span>{esc(subtitle)}</span></span></div>'



def legend_html() -> str:
    """colour key at the foot of the sidebar: the same roles every chart on every page uses."""
    def row(var, lab):
        return f'<span class="sw" style="background:var({var})"></span><span>{lab}</span>'
    return ('<div class="sidelegend" aria-label="Colour key"><p class="lt">Colour key</p>'
            '<div class="lk">' + row("--comply", "answered · re-opened") + row("--refuse", "refused · held")
            + row("--fix", "fix (in P⁻) · admitted") + row("--none", "not tested · no data") + '</div>'
            '<p class="lt">Signed values</p><div class="bar div"></div>'
            '<div class="ticks"><span>Positive</span><span>Zero</span><span>Negative</span></div>'
            '<p class="sub">Neuron activations and other differences: magenta above zero, blue below, pale at zero; '
            'stronger tint = larger magnitude.</p>'
            '<p class="lt" style="margin-top:12px">Shares</p><div class="bar seq"></div>'
            '<div class="ticks"><span>0 %</span><span>50 %</span><span>100 %</span></div>'
            '<p class="sub">Heat cells (matrices, category tables): tint = share of behaviours. '
            'Whiskers are Wilson 95 % intervals; dashed lines mark a reference (π* = 0.15).</p></div>')


# research questions as registered in the exposé (§3) and the pre-registration
RQS = [
    ("RQ1", "primary · recycling", "Can the safety improvement from one Llama generation to the next be bypassed by meaning-preserving rewording?",
     "Take the behaviours the newer model newly refuses (P⁻: older complies, newer refuses) and reword each one through the intent gate. "
     "If the fix was learned at the level of intent, rewordings should stay refused; if it patched a wording, some should be answered again.",
     "Recycling success rate RSR = share of P⁻ re-opened by at least one admitted rewording, per pair and pooled, Wilson 95 % CI, "
     "McNemar exact test; judge error corrected at bundle level (Rogan–Gladen)."),
    ("RQ2", "interpretation", "Do re-opened fixes reflect memorised wording or intent-level generalisation, and where is it brittle?",
     "RSR broken down by harm category and benchmark, to locate where the newer model's refusals depend on surface form. "
     "Neither direction is claimed as mechanistic proof.",
     "RSR per harm category / SORRY-Bench domain with Wilson intervals."),
    ("RQ3", "drivers", "Which kinds of rewording re-open patched requests?",
     "Six transform classes (paraphrase, syntactic, lexical, back-translation, tense shift, register) are compared on the same fixes.",
     "Mixed-effects logistic regression of per-rewording recovery on transform class + surface distance, random intercept per behaviour, "
     "Holm-corrected."),
    ("RQ4", "backbone · instrument check", "How robust is each model's own refusal set to rewording?",
     "Paraphrase brittleness is always estimable, even when P⁻ is small: reword what a model already refuses and count how much opens up.",
     "Paraphrase brittleness rate PBR over the model's canonically-refused set C_M, against the pre-registered reference π* = 0.15."),
]
BEYOND = ("Beyond the RQs", "white-box · exploratory", "Does a re-opening rewording move the newer model off its refusal direction?",
          "Exposé §8 future strengthening: load the exact served weights, compute the refusal direction (Arditi et al., 2024 difference of "
          "means) and compare the re-opening rewording with one the model still refuses.",
          "Sign test over behaviours on the last-layer refusal-direction score at the final prompt token; correlational only.")


def rq_section(sid="s-rq", num_="00", answers=None, where=None, frame="", status="") -> str:
    """the research questions as cards; answers[q] = html of the current answer, where[q] = [(href, label)] links on this page."""
    answers, where = answers or {}, where or {}
    cards = []
    for i, (q, kind, text, why, how) in enumerate(RQS + [BEYOND]):
        ans = answers.get(q)
        links = where.get(q) or []
        cards.append(f'<article class="rqcard{" primary" if i == 0 else ""}"><div class="rqh"><b>{esc(q)}</b><span>{esc(kind)}</span></div>'
                     f'<p class="q">{esc(text)}</p><p>{esc(why)}</p><p class="how"><b>Estimand.</b> {esc(how)}</p>'
                     + (f'<div class="ans">{ans}</div>' if ans else "")
                     + (('<p class="where">On this page: ' + "".join(f'<a href="#{h}">{esc(l)}</a>' for h, l in links) + '</p>') if links else "")
                     + '</article>')
    fr = f'<p class="rqframe">{frame}</p>' if frame else ""
    st = f'<p class="note">{status}</p>' if status else ""
    return (f'<section id="{sid}" class="sec" aria-labelledby="h-{sid}"><h2 id="h-{sid}"><span class="sn">{num_}</span>Research questions</h2>'
            f'{fr}<div class="rqgrid">{"".join(cards)}</div>{st}</section>')


def rq_script(mapping: dict) -> str:
    """JS: section id -> RQ labels, read by the TOC script to put chips on nav entries and section headings."""
    return f"window.RB_RQ = {json.dumps(mapping)};\n"


RQ_FRAME = ("<b>Does safety alignment generalise across model generations?</b> When a newer Llama refuses a request its predecessor "
            "answered, either it learned what the request is <i>for</i> (intent) or it learned how the request is <i>worded</i>. "
            "RecycleBench rewords exactly those fixed requests, certifies every rewording still asks for the same thing, and checks "
            "whether the newer model answers again.")


NAV = [("s-rq", "00", "Research questions"), ("s-headline", "01", "Headline"), ("s-lineage", "02", "Lineage"), ("s-models", "03", "Each model"),
       ("s-pairs", "04", "Pair explorer"), ("s-matrix", "05", "Every ordered pair"), ("s-categories", "06", "Harm categories"),
       ("s-styles", "07", "Rewording styles"), ("s-gate", "08", "Intent gate"), ("s-method", "09", "Reconciliation & method")]


LINEAGE_RQ_MAP = {"s-headline": ["RQ1"], "s-lineage": ["RQ1"], "s-pairs": ["RQ1"], "s-matrix": ["RQ1"],
                  "s-categories": ["RQ2"], "s-styles": ["RQ3"], "s-models": ["RQ4"]}


def sec_rq_lineage():
    where = {"RQ1": [("s-headline", "Headline"), ("s-lineage", "Lineage"), ("s-pairs", "Pair explorer"), ("s-matrix", "Every ordered pair")],
             "RQ2": [("s-categories", "Harm categories")], "RQ3": [("s-styles", "Rewording styles")], "RQ4": [("s-models", "Each model")]}
    return rq_section(where=where, frame=RQ_FRAME,
                      answers={"Beyond the RQs": "Not on this page: the white-box probe is shown in the llama3-8b → llama3.1-8b pair dashboard."},
                      status="This page answers the questions on SORRY-Bench across the six-model lineage; only the matched-scale "
                             "llama3-8b → llama3.1-8b rung is firm, the other rungs are exploratory.")


def render(an: Analysis, args) -> str:
    today = _dt.date.today().isoformat()
    nav = "".join(f'<a href="#{i}"><span class="n">{n}</span>{t}</a>' for i, n, t in NAV)
    body = (sec_header(an, args, today) + sec_rq_lineage() + sec_headline(an) + sec_lineage(an) + sec_models(an)
            + sec_pairs(an, args.redact) + sec_matrix(an) + sec_categories(an) + sec_styles(an)
            + sec_gate(an) + sec_reconcile(an, args))
    title = "SORRY-Bench Lineage Recycling"
    foot = ("Generated by tools/lineage_dashboard.py · recyclebench. "
            + ("Prompt and response text is omitted in this build. " if args.redact else
               "Contains benchmark prompts and model responses (collapsed, truncated): a working file, not a released artefact — see ETHICS.md. ")
            + "All figures are raw judge output unless marked corrected.")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<title>{esc(title)}</title><style>{CSS}</style></head><body>'
            f'<div class="layout"><nav class="side" aria-label="Sections">'
            f'{brand_html("SORRY-Bench · Llama lineage")}<div class="toc">{nav}</div>{legend_html()}'
            f'<button type="button" class="themebtn" id="themebtn" data-t="system">Theme: system</button></nav>'
            f'<main id="main">{body}<footer>{esc(foot)}</footer></main></div>'
            f'<div id="tip" role="tooltip" hidden></div><script>{rq_script(LINEAGE_RQ_MAP)}{JS}</script></body></html>')


# =============================================================================
# CLI
# =============================================================================
def pick_verdicts(args, cfg, behaviours, manifest, cache):
    if args.offline:
        return lc.OfflineVerdicts(), "forced by --offline"
    vp = Path(args.verdicts) if args.verdicts else None
    if not vp or not vp.exists():
        return lc.OfflineVerdicts(), f"no verdict cache at {args.verdicts}"
    tag = lc.judge_tag_from_cfg(cfg)
    if not tag:
        return lc.OfflineVerdicts(), "config has no judges: block"
    sv = lc.StoredVerdicts(lc.VerdictStore(vp), tag)
    obs = lc.observe(cfg, behaviours, manifest, cache, sv, cfg.lineage)
    miss = sum(o.missing_verdicts["canonical"] for o in obs.values())
    if miss:
        tot = sum(len(o.draws) * cfg.budget.k_canonical_samples for o in obs.values())
        return lc.OfflineVerdicts(), (f"verdict cache incomplete: {miss} of {tot} canonical draws unjudged "
                                      f"(finish tools/judge_lineage.py)")
    return sv, f"{len(sv.store):,} stored verdicts"


def default_out(run_dir: Path, redact: bool) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    base = run_dir / f"dashboard_lineage{'_redacted' if redact else ''}_{stamp}.html"
    out, i = base, 2
    while out.exists():                         # never overwrite
        out = base.with_name(f"{base.stem}_{i}.html")
        i += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/real_llama_lineage_all.yaml")
    ap.add_argument("--cache", default="cache/llama_gen.jsonl")
    ap.add_argument("--behaviours", default="data/sorrybench.csv")
    ap.add_argument("--run-dir", default="results/runs/lineage_sorrybench_20261020")
    ap.add_argument("--manifest", default=None, help="default: <run-dir>/manifest.jsonl")
    ap.add_argument("--summary", default=None, help="default: <run-dir>/summary_real.json")
    ap.add_argument("--anchor", default="results/runs/recycle_3_31_sorrybench/corrected_sorrybench.json")
    ap.add_argument("--verdicts", default="cache/judge_lineage_sorrybench.jsonl")
    ap.add_argument("--offline", action="store_true", help="use the offline refusal rule even if verdicts exist")
    ap.add_argument("--redact", action="store_true", help="omit every prompt and response text")
    ap.add_argument("--out", default=None, help="default: <run-dir>/dashboard_lineage_<stamp>.html (never overwrites)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    args.manifest = args.manifest or str(run_dir / "manifest.jsonl")
    args.summary = args.summary or str(run_dir / "summary_real.json")
    cfg = Config.load(args.config)
    if not cfg.lineage or len(cfg.lineage) < 2:
        raise SystemExit("config needs a `lineage:` of at least two models")
    behaviours = load_jailbreakbench(args.behaviours)
    manifest = Manifest.load(args.manifest)
    cache = lc.load_response_cache(args.cache)
    verdicts, why = pick_verdicts(args, cfg, behaviours, manifest, cache)
    print(f"verdicts : {verdicts.label}  ({why})")
    if not Path(args.summary).exists():
        print(f"summary  : {args.summary} not found -> headline figures recomputed from the {verdicts.kind} verdicts")
    an = analyse(cfg, behaviours, manifest, cache, verdicts, args.summary, args.anchor)
    if an.verdict_note:
        print("note     :", an.verdict_note)
    out = Path(args.out) if args.out else default_out(run_dir, args.redact)
    if out.exists():
        raise SystemExit(f"{out} exists -- refusing to overwrite (omit --out for a fresh dated name)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(an, args))
    print(f"wrote    : {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    for o, n in an.adjacent:
        pv = an.pairs[(o, n)]
        r = _run_rsr(an, o, n)
        print(f"  {o:>11} -> {n:<11} P^- here {pv.n:>3}" + (f" (run {r['n_p_minus']})" if r else "")
              + f"  re-opened here {num(pv.k):>3}" + (f" (run {r['k_recovered']})" if r else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
