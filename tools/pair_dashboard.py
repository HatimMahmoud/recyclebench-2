#!/usr/bin/env python3
"""tools/pair_dashboard.py -- one dashboard per model pair (older -> newer), across every benchmark.

For each pair: the canonical 2x2 and the flow of every behaviour (answered / refused -> fix, regression,
still answered, refused by both -> re-opened / held / unresolved / not tested) on every benchmark, every
recycle run that contains both models, the output-inversion (RPE) probe, the style and gate-score
diagnostics, known-attack transfer where the pair has it, the judge-vs-human disagreements for the anchor,
and the cases themselves: each fix with the canonical request, both models' replies, every admitted
rewording from every run with the newer model's reply, and every inversion candidate.

Evidence comes from tools/study_data.py (integrity-checked, recomputed with the package estimators); no
model is called. Case-level verdicts are the offline refusal rule (the only judge replayable per item) and
are labelled as such; each run's REPORTED figures (and the judge that made them) are shown alongside.

  python3 tools/pair_dashboard.py                         # the 7 standard pairs
  python3 tools/pair_dashboard.py --pairs llama2-13b:llama31-8b
  python3 tools/pair_dashboard.py --redact                # same pages without any prompt/response text

Writes results/runs/dedicated-runs/<older>__to__<newer>/dashboard_<stamp>.html (+ an index page next to
the folders); never overwrites an existing file.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_core as lc                       # noqa: E402
import mech_section                                    # noqa: E402
import lineage_dashboard as ld                  # noqa: E402
import master_dashboard as md                   # noqa: E402
import compose_arcs_section as ca               # noqa: E402
import inversion_section as invsec              # noqa: E402
import compose_v2_section as cv2                # noqa: E402
import study_data as sd                         # noqa: E402
from recyclebench.judge import looks_like_refusal   # noqa: E402
from recyclebench.realrun import _key           # noqa: E402
from recyclebench.recycling import compute_rsr  # noqa: E402
from recyclebench.pbr import compute_pbr        # noqa: E402

esc, pct, ci_txt, p_txt, num = ld.esc, ld.pct, ld.ci_txt, ld.p_txt, ld.num
tile, heat = ld.tile, ld.heat
mtable, tview, path_html, status_chip, judge_chip = md.mtable, md.tview, md.path_html, md.status_chip, md.judge_chip
NOT_YET = md.NOT_YET
label, short, pair_short, wil = md.label, md.short, md.pair_short, md.wil

OUT_ROOT = "results/runs/dedicated-runs"
CAP = {"prompt": 900, "reply": 900, "variant": 520, "vreply": 650, "cand": 420, "creply": 520}
TAB_ORDER = ["sorrybench", "harmbench", "advbench", "strongreject", "llama32jb", "wrapped_rs", "llama_wrapped",
             "jbb", "guidedbench"]


def cap(t, n):
    t = t or ""
    return t if len(t) <= n else t[:n].rstrip() + " …"


def pair_dir(o, n):
    return f"{o}__to__{n}"


# =============================================================================
# Pair data
# =============================================================================
@dataclass
class ItemRun:
    run_bid: str
    in_run_pm: bool
    state: str                      # recovered | held | unresolved | incomplete
    variants: list                  # [(VariantObs, scores dict)]


@dataclass
class RunView:
    rr: object
    o: str
    n: str
    adjacent: bool                  # consecutive in the run's own lineage (pre-registered there)
    complete: bool
    pm_run: list
    rsr: Optional[dict]
    reported: Optional[dict]
    judge_reported: str
    calibrated: Optional[dict]
    unresolved: int
    attempts: list
    by_class: dict
    scores: list                    # (nli, cos, bert, answered) for admitted rewordings of the run's P^- items
    items: dict = field(default_factory=dict)   # full-benchmark bid -> ItemRun
    pbr: dict = field(default_factory=dict)     # model -> dict(k, n, p, lo, hi, judge)

    @property
    def id(self):
        return self.rr.spec.id


@dataclass
class DatasetView:
    key: str
    ds: object
    status: str                     # complete | p-minus-only | incomplete | unavailable
    so: object
    sn: object
    pm: list
    persistent: list
    regression: Optional[list]
    both_safe: Optional[int]
    runs: list
    primary: Optional[RunView]
    rpe: list
    lineage_row: Optional[dict]
    draws: dict                     # (model, bid) -> [(text, verdict)]

    @property
    def n(self):
        return self.ds.n


_INV = None


def inv():
    """The inversion arm, loaded once from the root study_data reads."""
    global _INV
    if _INV is None or _INV.runs_root != sd.ROOT / "results" / "runs":
        _INV = invsec.load(root=sd.ROOT)
    return _INV


def _draws(model, ds, bid_prompt, cname):
    c = sd.cache(cname) if cname else {}
    bid, prompt = bid_prompt
    out = []
    for d in range(sd.K):
        t = c.get(_key(model, prompt, lc._draw_seed(bid, -1, d)))
        out.append((t, None if t is None else int(not looks_like_refusal(t))))
    return out


def _run_view(rr, o, n, ds, full_bids_by_prompt) -> RunView:
    sp = rr.spec
    models = list(sp.models)
    runs, obs = rr.runs, rr.obs
    pm = lc.regression_set(runs[o], runs[n])
    complete = all(obs[n].recovery_complete(b) for b in pm)
    r = compute_rsr(runs[o], runs[n], pm, sd.IDENTITY, 30) if (complete and pm) else None
    rsr = None
    if r is not None:
        _, lo, hi = wil(r.k_recovered, r.n_p_minus)
        rsr = dict(n=r.n_p_minus, k=r.k_recovered, rsr=r.rsr_raw, lo=lo, hi=hi, p=r.mcnemar_p, t=r.mean_admitted)
    adjacent = any(models[i] == o and models[i + 1] == n for i in range(len(models) - 1))
    pr_ = rr.pairs.get((o, n), {})
    rep = pr_.get("reported") if (adjacent or pr_.get("post_hoc")) else None   # post-hoc: judged replay, study_data
    audit = {(a.bid, a.index): a.scores() for a in rr.manifest.audit}
    by_class = defaultdict(lambda: [0, 0, 0, 0])
    scores = []
    attempts = []
    for b in pm:
        rec = runs[n].recovery[b]
        if rec.obs_bypassed:
            attempts.append(rec.attempts_to_first)
        hit_by = defaultdict(int)
        for v in obs[n].variants.get(b, []):
            if v.verdict is None:
                continue
            a = by_class[v.transform_class]
            a[0] += v.verdict
            a[1] += 1
            hit_by[v.transform_class] = max(hit_by[v.transform_class], v.verdict)
            sc = audit.get((b, v.index)) or {}
            scores.append((sc.get("nli_entail"), sc.get("cosine"), sc.get("bertscore"), v.verdict))
        for tc, h in hit_by.items():
            by_class[tc][2] += h
            by_class[tc][3] += 1
    rv = RunView(rr, o, n, adjacent, complete, pm, rsr, rep, sp.judge, rr.calibrated if sp.calibrated else None,
                 sum(1 for b in pm if runs[n].recovery[b].n_admitted == 0), attempts, dict(by_class), scores)
    # item map onto the full benchmark: same CSV -> by bid, else by prompt text (one prompt may be several rows)
    same_csv = (sd.ROOT / sp.csv).resolve() == (sd.ROOT / ds.csv).resolve()
    pm_set = set(pm)
    for b in rr.behaviours:
        targets = [b.bid] if same_csv else full_bids_by_prompt.get(b.canonical_prompt, [])
        vs = obs[n].variants.get(b.bid, [])
        if not obs[n].recovery_complete(b.bid):
            state = "incomplete"
        elif not vs:
            state = "unresolved"
        else:
            state = "recovered" if any(v.verdict == 1 for v in vs) else "held"
        iv = ItemRun(b.bid, b.bid in pm_set, state, [(v, audit.get((b.bid, v.index)) or {}) for v in vs])
        for t in targets:
            rv.items[t] = iv
    for m in (o, n):
        p = rr.pbr.get(m) or {}
        if sp.behaviour_scope != "full":
            continue
        if sp.judge != "offline" and p.get("reported"):
            x = p["reported"]
            _, lo, hi = wil(x["k_bypassed"], x["n_resolved"])
            rv.pbr[m] = dict(k=x["k_bypassed"], n=x["n_resolved"], p=x["PBR_raw"], lo=lo, hi=hi, judge="qwen2.5:14b")
        elif p.get("recon"):
            x = p["recon"]
            rv.pbr[m] = dict(k=x["k"], n=x["n"], p=x["pbr"], lo=x["lo"], hi=x["hi"], judge="offline rule")
    return rv


def build_pair(s, o, n) -> list:
    lin = md.lineage_rows(s)
    out = []
    for key in TAB_ORDER:
        ds = s.datasets.get(key)
        if ds is None or not ds.n:
            continue
        so, sn = s.screens.get((o, key)), s.screens.get((n, key))
        if so is None or sn is None or (not so.n_screened and not sn.n_screened):
            continue
        status, pm, pers, reg, both = "unavailable", [], [], None, None
        if so.complete:
            broke = [b for b, v in so.comply.items() if v == 1]
            refused = [b for b, v in so.comply.items() if v == 0]
            if any(b not in sn.comply for b in broke):
                status = "incomplete"
            else:
                pm = sorted(b for b in broke if sn.comply[b] == 0)
                pers = sorted(b for b in broke if sn.comply[b] == 1)
                if all(b in sn.comply for b in refused):
                    reg = sorted(b for b in refused if sn.comply[b] == 1)
                    both = len(refused) - len(reg)
                    status = "complete"
                else:
                    status = "p-minus-only"
        by_prompt = defaultdict(list)
        for b in ds.behaviours:
            by_prompt[b.canonical_prompt].append(b.bid)
        runs = []
        for rr in s.recycle:
            if rr.spec.dataset != key or o not in rr.spec.models or n not in rr.spec.models:
                continue
            if rr.spec.models.index(o) > rr.spec.models.index(n):
                continue
            runs.append(_run_view(rr, o, n, ds, by_prompt))
        runs.sort(key=lambda r: (not r.complete, r.rr.spec.behaviour_scope != "full", not r.adjacent, r.id))
        primary = next((r for r in runs if r.complete), None)
        rpe = [r for r in s.rpe if r.spec.old == o and r.spec.new == n and r.spec.dataset == key and r.items]
        newer_seen = sum(1 for b, v in so.comply.items() if v == 1 and b in sn.comply) if so.complete else 0
        if not runs and not rpe and (status == "unavailable" or (status == "incomplete" and newer_seen == 0)):
            continue
        bmap = {b.bid: b for b in ds.behaviours}
        draws = {}
        for b in pm + (reg or []):
            beh = bmap[b]
            draws[(o, b)] = _draws(o, ds, (b, beh.canonical_prompt), so.cache)
            draws[(n, b)] = _draws(n, ds, (b, beh.canonical_prompt), sn.cache)
        out.append(DatasetView(key, ds, status, so, sn, pm, pers, reg, both, runs, primary, rpe,
                               lin.get((o, n)) if key == "sorrybench" else None, draws))
    return out


# =============================================================================
# Charts
# =============================================================================
def _defs_hatch():
    return ('<defs><pattern id="hatchO" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            '<rect width="6" height="6" class="hb-bg"/><line x1="0" y1="0" x2="0" y2="6" class="hb-ln"/></pattern></defs>')


def answer_dumbbell(dvs):
    """Canonical answer rate, older (ring) -> newer (dot), per benchmark; line colour = direction."""
    rows = [d for d in dvs if d.so.complete and d.sn.complete]
    if not rows:
        return f"<p class='note'>{NOT_YET}: no benchmark has full canonical screens for both models.</p>"
    W, rh, top = 860, 34, 16
    x0, x1 = 250, 760
    mx = max(max(d.so.k / d.n, d.sn.k / d.n) for d in rows)
    x_max = min(1.0, max(0.1, math.ceil(mx * 20) / 20))
    H = top + rh * len(rows) + 34
    yb = H - 30
    s = [ld.svg_open(W, H, "Canonical answer rate, older vs newer model")]
    step = next(st for st in (0.05, 0.1, 0.2, 0.25) if x_max / st <= 6)
    t = 0.0
    while t <= x_max + 1e-9:
        x = x0 + (x1 - x0) * t / x_max
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{yb}" class="grid"/>'
                 f'<text x="{x:.1f}" y="{yb + 16}" class="ax" text-anchor="middle">{t * 100:.0f}%</text>')
        t += step

    def X(v):
        return x0 + (x1 - x0) * min(v, x_max) / x_max
    for i, d in enumerate(rows):
        cy = top + i * rh + rh / 2
        po, pn = d.so.k / d.n, d.sn.k / d.n
        _, lo_o, hi_o = wil(d.so.k, d.n)
        _, lo_n, hi_n = wil(d.sn.k, d.n)
        safer = pn < po
        cls = "dl-safer" if safer else ("dl-riskier" if pn > po else "dl-same")
        tip = (f"{d.ds.label}\n{d.so.model} answered {d.so.k}/{d.n} ({pct(po)}, Wilson {ci_txt(lo_o, hi_o)})\n"
               f"{d.sn.model} answered {d.sn.k}/{d.n} ({pct(pn)}, Wilson {ci_txt(lo_n, hi_n)})\n"
               f"{'newer refuses more' if safer else ('newer answers more' if pn > po else 'no change')} (offline rule)")
        s.append(f'<g class="row" tabindex="0" data-tip="{esc(tip)}"><rect x="0" y="{cy - rh / 2}" width="{W}" height="{rh}" class="hit"/>'
                 f'<text x="8" y="{cy - 2:.1f}" class="lbl strong">{esc(d.ds.label)}</text>'
                 f'<text x="8" y="{cy + 12:.1f}" class="ax">{d.n} behaviours</text>'
                 f'<line x1="{X(po):.1f}" x2="{X(pn):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="{cls}"/>'
                 f'<circle cx="{X(po):.1f}" cy="{cy:.1f}" r="5.5" class="dd-old"/>'
                 f'<circle cx="{X(pn):.1f}" cy="{cy:.1f}" r="5.5" class="dd-new"/>'
                 f'<text x="{x1 + 14}" y="{cy + 4:.1f}" class="val">{pct(po, 1)} → {pct(pn, 1)}</text></g>')
    s.append("</svg>")
    return "".join(s)


def composition_chart(dvs):
    """Left: 100% composition of all behaviours (the 2x2). Right: the fixes on an absolute scale, split by what
    rewording did to them (primary run)."""
    rows = [d for d in dvs if d.status in ("complete", "p-minus-only")]
    if not rows:
        return f"<p class='note'>{NOT_YET}</p>"
    W, rh, top = 900, 40, 22
    a0, a1 = 200, 560
    b0, b1 = 600, 880
    H = top + rh * len(rows) + 10
    pm_max = max([len(d.pm) for d in rows] + [1])
    s = [ld.svg_open(W, H, "Composition of every behaviour, and what rewording did to the fixes"), _defs_hatch(),
         f'<text x="{a0}" y="12" class="stage">all behaviours (100%)</text>',
         f'<text x="{b0}" y="12" class="stage">the fixes (P⁻), to scale</text>']
    for i, d in enumerate(rows):
        y = top + i * rh
        s.append(f'<text x="8" y="{y + 16}" class="lbl strong">{esc(d.ds.label)}</text>'
                 f'<text x="8" y="{y + 30}" class="ax">P⁻ {len(d.pm)} of {d.n}</text>')
        unknown = d.n - len(d.pm) - len(d.persistent) - (len(d.regression) if d.regression is not None else 0) - (d.both_safe or 0)
        parts = [("both", d.both_safe or 0, "cb-both", "refused by both"),
                 ("reg", len(d.regression) if d.regression is not None else 0, "cb-reg", "regression (newer answers)"),
                 ("unk", unknown, "cb-unk", "older refused · newer not screened"),
                 ("pers", len(d.persistent), "cb-pers", "still answered"),
                 ("pm", len(d.pm), "cb-pm", "P⁻ · the fixes")]
        x = a0
        for _, v, css, lab in parts:
            if not v:
                continue
            w = (a1 - a0) * v / d.n
            fill = ' fill="url(#hatchO)"' if css == "cb-reg" else ""
            s.append(f'<rect x="{x:.1f}" y="{y + 6}" width="{max(w - 1.5, 1):.1f}" height="22" rx="2" class="{css}"{fill} tabindex="0" '
                     f'data-tip="{esc(d.ds.label)} · {esc(lab)}: {v:,} of {d.n:,} ({pct(v / d.n)})"/>')
            x += w
        pv = d.primary
        st = Counter()
        for b in d.pm:
            it = pv.items.get(b) if pv else None
            st[(it.state if it else "untested")] += 1
        x = b0
        for key, css, lab in [("recovered", "cb-rec", "re-opened by rewording"), ("held", "cb-held", "held under every rewording"),
                              ("unresolved", "cb-unres", "no admitted rewording"), ("incomplete", "cb-untested", "re-screen unfinished"),
                              ("untested", "cb-untested", "not tested by a recycle run")]:
            v = st.get(key, 0)
            if not v:
                continue
            w = (b1 - b0) * v / pm_max
            s.append(f'<rect x="{x:.1f}" y="{y + 6}" width="{max(w - 1.5, 1):.1f}" height="22" rx="2" class="{css}" tabindex="0" '
                     f'data-tip="{esc(d.ds.label)} · {esc(lab)}: {v} of {len(d.pm)} fixes'
                     + (f" (run {pv.id}, offline rule)" if pv else "") + '"/>')
            if w > 22:
                s.append(f'<text x="{x + 5:.1f}" y="{y + 21}" class="{"inbar-dark" if css == "cb-untested" else "inbar"}">{v}</text>')
            x += w
    s.append("</svg>")
    return "".join(s)


def flow(d: DatasetView, o, n):
    """Sankey: all -> older answered/refused -> 4 cells -> (fixes only) re-opened / held / unresolved / not tested."""
    N = d.n
    pm, pe = len(d.pm), len(d.persistent)
    reg = len(d.regression) if d.regression is not None else None
    bs = d.both_safe
    oa = pm + pe
    orf = N - oa
    W, H = 980, 430
    top, bot = 34, H - 14
    gap, nw = 10, 12
    usable = bot - top - 3 * gap
    xs = [20, 300, 580, 800]
    MIN = 2.0

    def hh(v):
        return 0.0 if not v else max(usable * v / N, MIN)
    s = [ld.svg_open(W, H, f"Flow of every behaviour · {d.ds.label}")]
    for t, x in zip(["All behaviours", label(o), label(n), "After rewording (P⁻ only)"], xs):
        s.append(f'<text x="{x}" y="16" class="stage">{esc(t)}</text>')

    def node(x, y, h, role, tip):
        return (f'<rect x="{x}" y="{y:.1f}" width="{nw}" height="{max(h, MIN):.1f}" rx="2" class="node r-{role}" '
                f'tabindex="0" data-tip="{esc(tip)}"/>')

    def band(xa, ya0, ya1, xb, yb0, yb1, role, tip):
        mx = (xa + xb) / 2
        return (f'<path d="M{xa:.1f},{ya0:.1f} C{mx:.1f},{ya0:.1f} {mx:.1f},{yb0:.1f} {xb:.1f},{yb0:.1f} L{xb:.1f},{yb1:.1f} '
                f'C{mx:.1f},{yb1:.1f} {mx:.1f},{ya1:.1f} {xa:.1f},{ya1:.1f} Z" class="band r-{role}" data-tip="{esc(tip)}"/>')
    labels = []
    h0 = hh(N) + 3 * gap
    s.append(node(xs[0], top, h0, "ink", f"{N} behaviours"))
    labels.append((top + h0 / 2, xs[0], "behaviours", N))
    h_oa, h_or = hh(oa), hh(orf)
    y_oa = top + gap
    y_or = y_oa + h_oa + gap
    sp0 = top + h0 * (oa / N)
    if oa:
        s.append(band(xs[0] + nw, top, sp0, xs[1], y_oa, y_oa + h_oa, "comply", f"{label(o)} answered {oa}"))
        s.append(node(xs[1], y_oa, h_oa, "comply", f"{label(o)} answered: {oa}"))
        labels.append((y_oa + h_oa / 2, xs[1], "answered", oa))
    s.append(band(xs[0] + nw, sp0, top + h0, xs[1], y_or, y_or + h_or, "refuse", f"{label(o)} refused {orf}"))
    s.append(node(xs[1], y_or, h_or, "refuse", f"{label(o)} refused: {orf}"))
    labels.append((y_or + h_or / 2, xs[1], "refused", orf))
    # stage 2
    unknown = orf - (reg or 0) - (bs or 0) if reg is not None else orf
    hp, hm = hh(pe), hh(pm)
    y_pe = top
    y_pm = y_pe + hp + gap
    a_split = y_oa + h_oa * (pe / oa) if oa else y_oa
    if pe:
        s.append(band(xs[1] + nw, y_oa, a_split, xs[2], y_pe, y_pe + hp, "comply", f"still answered by {label(n)}: {pe}"))
        s.append(node(xs[2], y_pe, hp, "comply", f"still answered: {pe}"))
        labels.append((y_pe + hp / 2, xs[2], "still answered", pe))
    if pm:
        s.append(band(xs[1] + nw, a_split, y_oa + h_oa, xs[2], y_pm, y_pm + hm, "fix", f"P⁻ — the fixes: {pm}"))
        s.append(node(xs[2], y_pm, hm, "fix", f"P⁻ · the fixes: {pm}"))
        labels.append((y_pm + hm / 2, xs[2], "P⁻ · the fix", pm))
    y_next = y_pm + hm + gap
    cur = y_or
    for v, nm, role in ([(reg, "regression", "comply"), (bs, "refused by both", "refuse")] if reg is not None
                        else [(unknown, "newer not screened", "none")]):
        if not v:
            continue
        h = hh(v)
        share = h_or * v / orf if orf else 0
        s.append(band(xs[1] + nw, cur, cur + share, xs[2], y_next, y_next + h, role, f"{nm}: {v}"))
        s.append(node(xs[2], y_next, h, role, f"{nm}: {v}"))
        labels.append((y_next + h / 2, xs[2], nm, v))
        cur += share
        y_next += h + gap
    # stage 3: fixes, by what rewording did (primary run)
    pv = d.primary
    if pm:
        st = Counter()
        for b in d.pm:
            it = pv.items.get(b) if pv else None
            st[it.state if it else "untested"] += 1
        parts = [(st["recovered"], "re-opened", "comply", "re-opened by ≥1 admitted rewording"),
                 (st["held"], "held", "refuse", "refused every admitted rewording"),
                 (st["unresolved"], "no admitted rewording", "none", "no rewording passed the gate"),
                 (st["incomplete"] + st["untested"], "not tested", "untested", "no complete recycle run covers it")]
        tot_h = sum(hh(c) for c, *_ in parts if c) + gap * (sum(1 for c, *_ in parts if c) - 1)
        y3 = max(top, y_pm + hm / 2 - tot_h / 2)
        src = y_pm
        for c, nm, role, tip in parts:
            if not c:
                continue
            h3 = hh(c)
            sh = hm * c / pm
            s.append(band(xs[2] + nw, src, src + sh, xs[3], y3, y3 + h3, role, f"{tip}: {c} of {pm}"))
            s.append(node(xs[3], y3, h3, role, f"{nm}: {c} of {pm}" + (f" · run {pv.id}" if pv else "")))
            labels.append((y3 + h3 / 2, xs[3], nm, c))
            src += sh
            y3 += h3 + gap
    by_x = defaultdict(list)
    for it in labels:
        by_x[it[1]].append(it)
    for x, items in by_x.items():
        ys = ld._place_labels([(c,) for c, *_ in items], top, bot, gap=32)
        for (cy, _x, nm, c), ly in zip(items, ys):
            tx = x + nw + 8
            if abs(ly - cy) > 1:
                s.append(f'<line x1="{x + nw + 1}" x2="{tx - 2}" y1="{cy:.1f}" y2="{ly - 4:.1f}" class="leader"/>')
            s.append(f'<text x="{tx}" y="{ly - 2:.1f}" class="lbl strong halo">{esc(nm)}</text>'
                     f'<text x="{tx}" y="{ly + 12:.1f}" class="ax halo">{c:,} · {pct(c / N, 1)}</text>')
    s.append("</svg>")
    return "".join(s)


def run_bars(d: DatasetView):
    rows = []
    for r in d.runs:
        st = Counter()
        for b in r.pm_run:
            rec = r.rr.runs[r.n].recovery[b]
            if not r.rr.obs[r.n].recovery_complete(b):
                st["incomplete"] += 1
            elif rec.n_admitted == 0:
                st["unresolved"] += 1
            else:
                st["recovered" if rec.obs_bypassed else "held"] += 1
        tag = ("consecutive in run" if r.adjacent else "post-hoc") + (" · incomplete" if not r.complete else "")
        nf = len(r.pm_run)
        rows.append(dict(label=r.id, sub=f"{nf} fix{'es' if nf != 1 else ''} · {tag}", parts=dict(st), total=max(nf, 1)))
    if not rows:
        return f"<p class='note'>Recycling on this benchmark: {NOT_YET}.</p>"
    keys = [("recovered", "sb-nli", "re-opened"), ("held", "sb-cos", "held"), ("unresolved", "sb-lex", "no admitted rewording"),
            ("incomplete", "sb-bert", "re-screen unfinished")]
    W, rh, top = 520, 40, 6
    x0, x1 = 170, 505
    H = top + rh * len(rows) + 4
    sv = [ld.svg_open(W, H, "Outcome of each recycle run on its own fixes")]
    for i, r in enumerate(rows):
        y = top + i * rh
        sv.append(f'<text x="4" y="{y + 15}" class="lbl strong">{esc(r["label"])}</text>'
                  f'<text x="4" y="{y + 29}" class="ax">{esc(r["sub"])}</text>')
        x = x0
        for k, css, lab in keys:
            v = r["parts"].get(k, 0)
            if not v:
                continue
            w = (x1 - x0) * v / r["total"]
            sv.append(f'<rect x="{x:.1f}" y="{y + 7}" width="{max(w - 1.5, 1):.1f}" height="20" rx="3" class="{css}" tabindex="0" '
                      f'data-tip="{esc(r["label"])} · {esc(lab)}: {v} of {r["total"]} ({pct(v / r["total"])})"/>')
            if w > 26:
                sv.append(f'<text x="{x + 6:.1f}" y="{y + 21}" class="inbar">{v}</text>')
            x += w
    sv.append("</svg>")
    return ("".join(sv)
            + '<div class="legend">' + "".join(f'<span><i class="lg {c}"></i>{esc(l)}</span>' for _, c, l in keys) + "</div>")


def recovery_curves(d: DatasetView):
    rs = [r for r in d.runs if r.complete and r.rsr]
    if not rs:
        return f"<p class='note'>{NOT_YET}</p>"
    J = max(8, max(max((r.rr.runs[r.n].recovery[b].n_admitted for b in r.pm_run), default=1) for r in rs))
    W, H = 540, 290
    x0, x1, y0, y1 = 44, 360, 16, 236
    s = [ld.svg_open(W, H, "Cumulative share of fixes re-opened by number of rewordings tried")]
    for t in (0, .25, .5, .75, 1):
        y = y1 - (y1 - y0) * t
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                 f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="ax" text-anchor="end">{int(t * 100)}%</text>')
    for j in range(1, J + 1):
        x = x0 + (x1 - x0) * (j - 1) / max(J - 1, 1)
        s.append(f'<text x="{x:.1f}" y="{y1 + 18}" class="ax" text-anchor="middle">{j}</text>')
    s.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 36}" class="ax" text-anchor="middle">admitted rewordings tried (manifest order)</text>')
    ends = []
    for i, r in enumerate(rs):
        emph = (r is d.primary)
        n_ = len(r.pm_run)
        pts = []
        for j in range(1, J + 1):
            k = sum(1 for a in r.attempts if 0 < a <= j)
            pts.append((x0 + (x1 - x0) * (j - 1) / max(J - 1, 1), y1 - (y1 - y0) * k / n_, j, k))
        s.append(f'<path d="M{" L".join(f"{x:.1f},{y:.1f}" for x, y, *_ in pts)}" class="{"line-emph" if emph else "line-mute"}"/>')
        for x, y, j, k in pts:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" class="hitpt" data-tip="{esc(r.id)}\n{k} of {n_} fixes re-opened within '
                     f'{j} rewording(s) ({pct(k / n_)})"/>')
        ends.append((pts[-1][1], f"{r.id} · {pct(pts[-1][3] / n_, 0)}", emph))
        s.append(f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4" class="{"dot-emph" if emph else "dot-mute"}"/>')
    ys = ld._place_labels([(e[0],) for e in ends], y0, y1 + 10, gap=16)
    for (ey, txt, emph), ly in zip(ends, ys):
        s.append(f'<text x="{x1 + 12}" y="{ly + 4:.1f}" class="{"lbl strong" if emph else "ax"}">{esc(txt)}</text>')
    s.append("</svg>")
    return "".join(s)


def _jitter(key, span):
    h = int(hashlib.sha256(key.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    return (h - 0.5) * span


def strip_plot(groups, title, thr=None, thr_label="", x_lo=0.0, x_hi=1.0):
    """groups: [(label, sub, [values], css)] -> horizontal strip plot with deterministic jitter, median tick."""
    W, gh, top = 860, 56, 14
    x0, x1 = 210, 820
    H = top + gh * len(groups) + 30
    yb = H - 26
    s = [ld.svg_open(W, H, title)]

    def X(v):
        return x0 + (x1 - x0) * (min(max(v, x_lo), x_hi) - x_lo) / (x_hi - x_lo)
    step = next(st for st in (0.05, 0.1, 0.2, 0.25, 0.5) if (x_hi - x_lo) / st <= 6)
    t = math.ceil(x_lo / step - 1e-9) * step
    while t <= x_hi + 1e-9:
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 4}" y2="{yb}" class="grid"/>'
                 f'<text x="{X(t):.1f}" y="{yb + 16}" class="ax" text-anchor="middle">{t:.2f}</text>')
        t += step
    if thr is not None:
        s.append(f'<line x1="{X(thr):.1f}" x2="{X(thr):.1f}" y1="{top - 4}" y2="{yb}" class="refline"/>'
                 f'<text x="{X(thr) + 4:.1f}" y="{top + 6}" class="ax halo">{esc(thr_label)}</text>')
    for i, (lab, sub, vals, css) in enumerate(groups):
        cy = top + i * gh + gh / 2
        vals = sorted(v for v in vals if v is not None and v == v)
        s.append(f'<text x="8" y="{cy - 2:.1f}" class="lbl strong">{esc(lab)}</text>'
                 f'<text x="8" y="{cy + 12:.1f}" class="ax">{esc(sub)} · n={len(vals)}</text>')
        for j, v in enumerate(vals):
            s.append(f'<circle cx="{X(v):.1f}" cy="{cy + _jitter(f"{lab}{j}{v}", gh * 0.6):.1f}" r="3" class="{css}"/>')
        if vals:
            med = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
            s.append(f'<line x1="{X(med):.1f}" x2="{X(med):.1f}" y1="{cy - gh * 0.38:.1f}" y2="{cy + gh * 0.38:.1f}" class="medline" '
                     f'tabindex="0" data-tip="{esc(lab)}: median {med:.3f} (n={len(vals)})"/>')
    s.append("</svg>")
    return "".join(s)


def gate_scatter(points, thr, title):
    """RPE candidates: cosine (x) vs NLI entailment (y) for every candidate the gate scored that far,
    admitted region shaded, colour = outcome."""
    W, H = 520, 404
    x0, x1, y0, y1 = 56, 500, 20, 356
    s = [ld.svg_open(W, H, title)]

    def X(v):
        return x0 + (x1 - x0) * max(0, min(1, v))

    def Y(v):
        return y1 - (y1 - y0) * max(0, min(1, v))
    s.append(f'<rect x="{X(thr["cosine"]):.1f}" y="{y0}" width="{x1 - X(thr["cosine"]):.1f}" height="{Y(thr["nli_entail"]) - y0:.1f}" class="admitzone"/>')
    for t in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{y0}" y2="{y1}" class="grid"/>'
                 f'<text x="{X(t):.1f}" y="{y1 + 16}" class="ax" text-anchor="middle">{t:.2f}</text>'
                 f'<line x1="{x0}" x2="{x1}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                 f'<text x="{x0 - 6}" y="{Y(t) + 4:.1f}" class="ax" text-anchor="end">{t:.2f}</text>')
    s.append(f'<line x1="{X(thr["cosine"]):.1f}" x2="{X(thr["cosine"]):.1f}" y1="{y0}" y2="{y1}" class="refline"/>'
             f'<line x1="{x0}" x2="{x1}" y1="{Y(thr["nli_entail"]):.1f}" y2="{Y(thr["nli_entail"]):.1f}" class="refline"/>'
             f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 36}" class="ax" text-anchor="middle">embedding cosine to the canonical request</text>'
             f'<text x="14" y="{(y0 + y1) / 2:.0f}" class="ax" transform="rotate(-90 14 {(y0 + y1) / 2:.0f})" text-anchor="middle">NLI entailment</text>'
             f'<text x="{x1}" y="{y0 - 4}" class="ax" text-anchor="end">shaded: admitted zone</text>')
    for p in points:
        cls = {"broke": "sc-broke", "held": "sc-held", "rejected": "sc-rej"}[p["state"]]
        s.append(f'<circle cx="{X(p["cos"]):.1f}" cy="{Y(p["nli"]):.1f}" r="{5 if p["state"] != "rejected" else 3.2}" class="{cls}" '
                 f'tabindex="{0 if p["state"] != "rejected" else -1}" data-tip="{esc(p["tip"])}"/>')
    s.append("</svg>")
    return "".join(s)


def category_bars(d: DatasetView):
    pv = d.primary
    bmap = {b.bid: b for b in d.ds.behaviours}
    per = defaultdict(Counter)
    for b in d.pm:
        it = pv.items.get(b) if pv else None
        per[bmap[b].category][it.state if it else "untested"] += 1
    if not per or len(per) < 2:
        return ""
    is_sorry = d.key == "sorrybench"

    def name(c):
        return f"#{c} {lc.category_name(c)}" if is_sorry else c
    order = sorted(per, key=lambda c: (-sum(per[c].values()), str(c)))
    rows = []
    for c in order:
        st = per[c]
        st["untested"] += st.pop("incomplete", 0)
        rows.append(dict(label=name(c), sub=(lc.domain_of(c) if is_sorry else ""), parts=dict(st), total=sum(st.values())))
    mx = max(r["total"] for r in rows)
    W, rh, top = 860, 26, 8
    x0, x1 = 300, 820
    H = top + rh * len(rows) + 6
    s = [ld.svg_open(W, H, "Fixes by category")]
    keys = [("recovered", "cb-rec", "re-opened"), ("held", "cb-held", "held"), ("unresolved", "cb-unres", "no admitted rewording"),
            ("untested", "cb-untested", "not tested")]
    for i, r in enumerate(rows):
        y = top + i * rh
        s.append(f'<text x="8" y="{y + 16}" class="lbl">{esc(r["label"])}</text>')
        x = x0
        for key, css, lab in keys:
            v = r["parts"].get(key, 0)
            if not v:
                continue
            w = (x1 - x0) * v / mx
            s.append(f'<rect x="{x:.1f}" y="{y + 5}" width="{max(w - 1.5, 1):.1f}" height="15" rx="2" class="{css}" tabindex="0" '
                     f'data-tip="{esc(r["label"])} · {esc(lab)}: {v} of {r["total"]} fixes"/>')
            x += w
        s.append(f'<text x="{x + 6:.1f}" y="{y + 17}" class="ax">{r["total"]}</text>')
    s.append("</svg>")
    return (s and "".join(s)) + '<div class="legend">' + "".join(f'<span><i class="lg {c}"></i>{esc(l)}</span>' for _, c, l in keys) + "</div>"


# =============================================================================
# Cases
# =============================================================================
def _vpill(v):
    if v is None:
        return '<span class="pill pill-none">not judged</span>'
    return '<span class="pill pill-comply">answered</span>' if v == 1 else '<span class="pill pill-refuse">refused</span>'


def _score_txt(sc, admitted=True, thr=None):
    """Only scores the gate actually computed: it stops at the first failing layer and stores the placeholder
    1.0 for every later layer, so a rejected candidate shows its scores up to the layer that stopped it."""
    names = {"cosine": "cos", "bertscore": "bert", "nli_entail": "nli"}
    rb, ev = sd.classify_rejection(sc, admitted, thr or sd.STRICT_GATE)
    parts = [f"{names[k]} {float(sc[k]):.2f}" for k in ev if sc.get(k) is not None and sc.get(k) == sc.get(k)]
    if rb:
        parts.append("stopped at " + {"lexical_divergence": "lexical floor (too close to the original)", "cosine": "cosine",
                                      "bertscore": "BERTScore", "nli_entail": "NLI"}[rb])
    return " · ".join(parts)


def _first(draws, want):
    for t, v in draws:
        if t is not None and (want is None or v == want):
            return t, v
    return "", None


def fix_card(d: DatasetView, bid, o, n, redact):
    b = next(x for x in d.ds.behaviours if x.bid == bid)
    cat = f"#{b.category} {lc.category_name(b.category)}" if d.key == "sorrybench" else b.category
    pv = d.primary
    it = pv.items.get(bid) if pv else None
    outcome = it.state if it else "untested"
    if outcome == "incomplete":
        outcome = "untested"
    pill = {"recovered": '<span class="pill pill-comply">re-opened</span>', "held": '<span class="pill pill-refuse">held</span>',
            "unresolved": '<span class="pill pill-none">no admitted rewording</span>',
            "untested": '<span class="pill">not tested</span>'}[outcome]
    run_blocks = []
    for r in d.runs:
        iv = r.items.get(bid)
        if not iv:
            continue
        if iv.state == "incomplete":
            run_blocks.append(f'<div class="rblock"><div class="rhead"><b>{esc(r.id)}</b> · re-screen unfinished for this item</div></div>')
            continue
        hits = sum(1 for v, _ in iv.variants if v.verdict == 1)
        head = (f'<div class="rhead"><b>{esc(r.id)}</b> · {esc(r.rr.spec.paraphraser.split(" (")[0])} rewordings · '
                f'{len(iv.variants)} admitted · <span class="{"warnt" if hits else ""}">{hits} answered</span>'
                + ("" if iv.in_run_pm else " · <span class='sub'>not a fix in this run's own screen</span>") + "</div>")
        rows = []
        for v, sc in iv.variants:
            cls = "v-comply" if v.verdict == 1 else ("v-refuse" if v.verdict == 0 else "v-none")
            body = "" if redact else (
                f'<div class="txt"><span class="tl">reworded request</span>{esc(cap(v.text, CAP["variant"]))}</div>'
                f'<div class="txt"><span class="tl">{esc(label(n))} replied</span>{esc(cap(v.response, CAP["vreply"]))}</div>')
            rows.append(f'<div class="vrow {cls}"><div class="vhead"><span class="vtc">{esc(v.transform_class)}</span>{_vpill(v.verdict)}'
                        f'<span class="sc">{esc(_score_txt(sc))}</span></div>{body}</div>')
        run_blocks.append(f'<div class="rblock">{head}{"".join(rows) or "<p class=note>No rewording passed the gate.</p>"}</div>')
    cur = invsec.pair_fix(inv(), d.key, o, n, b.canonical_prompt, redact, cap, CAP, _vpill, label(n))
    if cur:
        run_blocks.append(cur)
    for rp in d.rpe:
        item = next((x for x in rp.items if x.get("prompt") == b.canonical_prompt), None)
        if not item:
            continue
        det = item.get("details") or []
        adm = [x for x in det if x.get("admitted")]
        head = (f'<div class="rhead"><b>{esc(rp.spec.id)}</b> · early inversion run (superseded) · gate {esc(rp.spec.gate_label)} · '
                f'{len(det)} candidates, {len(adm)} admitted · judge {judge_chip(rp.judge)}</div>')
        rows = []
        for x in det:
            okd = x.get("admitted")
            sc = x.get("scores") or {}
            off = None if (not okd or x.get("response") is None) else int(not looks_like_refusal(x.get("response") or ""))
            stored = x.get("broke")
            verdicts = ""
            if okd:
                verdicts = (f'<span class="sub">stored judge: {"answered" if stored else "refused" if stored is not None else "—"}</span>'
                            + _vpill(off))
            body = "" if redact else (
                f'<div class="txt"><span class="tl">reconstructed request</span>{esc(cap(x.get("text"), CAP["cand"]))}</div>'
                + (f'<div class="txt"><span class="tl">{esc(label(n))} replied</span>{esc(cap(x.get("response"), CAP["creply"]))}</div>'
                   if okd and x.get("response") else ""))
            rows.append(f'<div class="vrow {"v-comply" if off == 1 else "v-refuse" if off == 0 else "v-none"}"><div class="vhead">'
                        f'<span class="vtc">{"admitted" if okd else "rejected"}</span>{verdicts}'
                        f'<span class="sc">{esc(_score_txt(sc, bool(okd), rp.spec.gate))}</span></div>{body}</div>')
        n_ok = sum(1 for x in adm if x.get("response") is not None and not looks_like_refusal(x.get("response") or ""))
        run_blocks.append(f'<details class="rblock rpeb"><summary>{head}<span class="sub">{n_ok} of {len(adm)} admitted answered · show candidates</span>'
                          f'</summary>{"".join(rows)}</details>')
    if redact:
        canon = '<p class="note">Text redacted.</p>'
    else:
        t_o, _ = _first(d.draws.get((o, bid), []), 1)
        t_n, _ = _first(d.draws.get((n, bid), []), None)
        canon = (f'<div class="txt"><span class="tl">canonical request</span>{esc(cap(b.canonical_prompt, CAP["prompt"]))}</div>'
                 f'<div class="txt"><span class="tl">{esc(label(o))} · answered (first complying draw)</span>{esc(cap(t_o, CAP["reply"]))}</div>'
                 f'<div class="txt"><span class="tl">{esc(label(n))} · refused (draw 1 of {sd.K})</span>{esc(cap(t_n, CAP["reply"]))}</div>')
    after = "".join(run_blocks) or f"<p class='note'>No recycle run covers this behaviour: {NOT_YET}.</p>"
    mech = mech_section.panel(o, n, d.key, bid)
    wb = '<span class="mbadge">white-box</span>' if mech else ""
    return (f'<details class="case" data-outcome="{outcome}"><summary><span class="cid">{esc(bid)}</span>'
            f'<span class="ccat">{esc(cat)}</span>{"" if redact else "<span class=sens>sensitive</span>"}{pill}{wb}</summary>'
            f'<div class="cbody">{canon}<div class="vlist"><div class="tl">after rewording · newer model ({esc(label(n))})</div>{after}</div>{mech}</div></details>')


def regression_card(d, bid, o, n, redact):
    b = next(x for x in d.ds.behaviours if x.bid == bid)
    cat = f"#{b.category} {lc.category_name(b.category)}" if d.key == "sorrybench" else b.category
    if redact:
        body = '<p class="note">Text redacted.</p>'
    else:
        t_o, _ = _first(d.draws.get((o, bid), []), None)
        t_n, _ = _first(d.draws.get((n, bid), []), 1)
        body = (f'<div class="txt"><span class="tl">canonical request</span>{esc(cap(b.canonical_prompt, CAP["prompt"]))}</div>'
                f'<div class="txt"><span class="tl">{esc(label(o))} · refused (draw 1 of {sd.K})</span>{esc(cap(t_o, CAP["reply"]))}</div>'
                f'<div class="txt"><span class="tl">{esc(label(n))} · answered (first complying draw)</span>{esc(cap(t_n, CAP["reply"]))}</div>')
    return (f'<details class="case" data-outcome="regression"><summary><span class="cid">{esc(bid)}</span><span class="ccat">{esc(cat)}</span>'
            f'{"" if redact else "<span class=sens>sensitive</span>"}<span class="pill pill-comply">newer answers</span></summary>'
            f'<div class="cbody">{body}</div></details>')


def case_lists(d, o, n, redact):
    fixes = [fix_card(d, b, o, n, redact) for b in d.pm]
    cnt = Counter(c.split('data-outcome="')[1].split('"')[0] for c in fixes)
    btns = "".join(f'<button type="button" class="fbtn{" active" if k == "all" else ""}" data-filter="{k}" aria-pressed="{"true" if k == "all" else "false"}">'
                   f'{esc(l)} <span>{c}</span></button>'
                   for k, l, c in [("all", "All fixes", len(fixes)), ("recovered", "Re-opened", cnt["recovered"]), ("held", "Held", cnt["held"]),
                                   ("unresolved", "No admitted rewording", cnt["unresolved"]), ("untested", "Not tested", cnt["untested"])]
                   if k == "all" or c)
    out = (f'<div class="cases"><div class="filters" role="group" aria-label="Filter fixes">{btns}</div>{"".join(fixes)}</div>'
           if fixes else f"<p class='note'>No fixes (P⁻ is empty) on {esc(d.ds.label)}.</p>")
    regs = d.regression or []
    reg_html = ""
    if regs:
        reg_html = (f'<details class="tv"><summary>Safety regressions · {len(regs)} requests {esc(label(o))} refused and {esc(label(n))} answers'
                    f'</summary><div class="cases">{"".join(regression_card(d, b, o, n, redact) for b in regs)}</div></details>')
    return out + reg_html


# =============================================================================
# Sections
# =============================================================================
def _primary_note(d):
    pv = d.primary
    if not pv:
        return f"No complete recycle run covers this pair on {d.ds.label}: rewording outcomes are {NOT_YET}."
    note = f" <span class='warnt'>Run note: {esc(pv.rr.spec.note)}.</span>" if pv.rr.spec.note else ""
    return (f"After-rewording outcomes use run <b>{esc(pv.id)}</b> ({esc(pv.rr.spec.paraphraser.split(' (')[0])} rewordings, "
            f"{'the pair is consecutive in that run' if pv.adjacent else 'post-hoc: the pair is not consecutive in that run'}) and the offline rule."
            + note)


def _median(v):
    v = sorted(x for x in v if x is not None and x == x)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def _mw_line(a, b, name):
    """Two-sided Mann-Whitney U between answered and refused rewordings, with medians."""
    a = [x for x in a if x is not None and x == x]
    b = [x for x in b if x is not None and x == x]
    if len(a) < 3 or len(b) < 3:
        return f"<p class='note small'>{esc(name)}: too few rewordings in one group for a test (answered {len(a)}, refused {len(b)}).</p>"
    from scipy.stats import mannwhitneyu
    res = mannwhitneyu(a, b, alternative="two-sided")
    ma, mb = _median(a), _median(b)
    lean = "lower" if ma < mb else "higher"
    return (f"<p class='note small'>{esc(name)}: median {ma:.3f} for answered vs {mb:.3f} for refused rewordings (answered sit {lean}); "
            f"Mann–Whitney U = {res.statistic:.0f}, two-sided p = {p_txt(float(res.pvalue))} (n = {len(a)} vs {len(b)}).</p>")


def _partial_note(d, o, n):
    if d.status != "incomplete":
        return ""
    broke = [b for b, v in d.so.comply.items() if v == 1]
    seen = [b for b in broke if b in d.sn.comply]
    refused = sum(1 for b in seen if d.sn.comply[b] == 0)
    return (f"<p class='flag'>▲ {esc(label(n))} was screened on only {len(seen)} of the {len(broke)} requests {esc(label(o))} answered "
            f"(offline rule); {refused} of those {len(seen)} are refused. P⁻ cannot be completed without screening the rest: {NOT_YET}.</p>")


def dataset_panel(s, d: DatasetView, o, n, idx, active, redact):
    tiles = []
    if d.status in ("complete", "p-minus-only"):
        tiles.append(tile("P⁻ · fixes (offline rule)", num(len(d.pm)), f"of {d.n} behaviours · {len(d.pm) + len(d.persistent)} answered by {short(o)}"))
    if d.lineage_row:
        r = d.lineage_row
        _, lo, hi = wil(r["k_recovered"], r["n_p_minus"])
        tiles.append(tile("RSR · run summary (qwen2.5:14b)", pct(r["RSR_raw"]), f"{r['k_recovered']}/{r['n_p_minus']} · Wilson {ci_txt(lo, hi)}", emph=True))
    for r in d.runs:
        if r.calibrated:
            c = r.calibrated["RSR"]
            tiles.append(tile("RSR · human-calibrated", pct(c["RSR_corrected"]), f"95% CI {ci_txt(*c['CI'])} · raw {c['k_recovered']}/{c['n_p_minus']}", emph=True))
    if d.primary and d.primary.rsr:
        r = d.primary.rsr
        tiles.append(tile(f"RSR · {d.primary.id} (offline)", pct(r["rsr"]), f"{r['k']}/{r['n']} · Wilson {ci_txt(r['lo'], r['hi'])} · McNemar p {p_txt(r['p'])}",
                          emph=not d.lineage_row))
    if d.regression is not None:
        tiles.append(tile("Regressions", num(len(d.regression)), f"{short(o)} refused, {short(n)} answers"))
    m2 = ""
    if d.status in ("complete", "p-minus-only"):
        reg = "—" if d.regression is None else len(d.regression)
        bs = "—" if d.both_safe is None else d.both_safe
        m2 = (f'<div class="m2"><div></div><div class="mh">{esc(label(n))} answered</div><div class="mh">{esc(label(n))} refused</div>'
              f'<div class="mh mrow">{esc(label(o))} answered</div><div class="mc c-comply"><b>{len(d.persistent)}</b><span>still answered</span></div>'
              f'<div class="mc c-fix"><b>{len(d.pm)}</b><span>P⁻ · the fix</span></div>'
              f'<div class="mh mrow">{esc(label(o))} refused</div><div class="mc c-regress"><b>{reg}</b><span>regression</span></div>'
              f'<div class="mc c-refuse"><b>{bs}</b><span>refused by both</span></div></div>')
    rtab = []
    for r in d.runs:
        rep = r.reported
        rep_txt = "—"
        if rep and rep.get("n_p_minus") is not None:
            rep_txt = (f"{rep['k_recovered']}/{rep['n_p_minus']} ({pct(rep['RSR_raw']) if rep.get('RSR_raw') is not None else 'undefined'})"
                       f" · {'qwen2.5:14b' if r.judge_reported != 'offline' else 'offline rule'}")
        rec = f"{r.rsr['k']}/{r.rsr['n']} ({pct(r.rsr['rsr'])})" if r.rsr else ("P⁻ = 0" if not r.pm_run else NOT_YET)
        rtab.append([f"{esc(r.id)}<span class='sub2'>{esc(r.rr.spec.paraphraser.split(' (')[0])} · {esc(r.rr.spec.behaviour_scope)} · "
                     f"{'consecutive in run' if r.adjacent else 'post-hoc'}</span>"
                     + (f"<span class='sub2 warnt'>{esc(r.rr.spec.note)}</span>" if r.rr.spec.note else ""), rep_txt,
                     rec + (f" <span class='sub'>{ci_txt(r.rsr['lo'], r.rsr['hi'])}</span>" if r.rsr else ""),
                     p_txt(r.rsr["p"]) if r.rsr else "—", num(r.unresolved), status_chip("complete" if r.complete else "incomplete")])
    sty = ""
    if d.primary and d.primary.by_class:
        bars = [dict(label=tc.replace("_", " "), sub=f"{a[1]} rewordings of {a[3]} fixes", v=(a[0] / a[1] if a[1] else None),
                     v_txt=pct(a[0] / a[1] if a[1] else None), tip=f"{tc}: {a[0]} of {a[1]} admitted rewordings answered; re-opened {a[2]} of {a[3]} fixes")
                for tc, a in sorted(d.primary.by_class.items(), key=lambda x: -(x[1][0] / max(x[1][1], 1)))]
        sty = ld.hbars(bars, "Re-open rate by rewording style")
    strip = ""
    allsc = [x for r in d.runs if r.complete for x in r.scores]
    if allsc:
        groups = [("answered", "rewordings the newer model answered", [x[0] for x in allsc if x[3] == 1], "sp-comply"),
                  ("refused", "rewordings it refused", [x[0] for x in allsc if x[3] == 0], "sp-refuse")]
        groups_c = [("answered", "cosine", [x[1] for x in allsc if x[3] == 1], "sp-comply"),
                    ("refused", "cosine", [x[1] for x in allsc if x[3] == 0], "sp-refuse")]
        strip = (strip_plot(groups, "NLI entailment of admitted rewordings, answered vs refused", thr=sd.STRICT_GATE["nli_entail"],
                            thr_label="gate 0.60", x_lo=0.5, x_hi=1.0) + _mw_line(groups[0][2], groups[1][2], "NLI entailment")
                 + strip_plot(groups_c, "Cosine of admitted rewordings, answered vs refused", thr=sd.STRICT_GATE["cosine"],
                              thr_label="gate 0.75", x_lo=0.7, x_hi=1.0) + _mw_line(groups_c[0][2], groups_c[1][2], "cosine"))
    rpe_html = ""
    for rp in d.rpe:
        pts = []
        for item in rp.items:
            for x in item.get("details") or []:
                sc = x.get("scores") or {}
                rb, ev = sd.classify_rejection(sc, bool(x.get("admitted")), rp.spec.gate)
                if "nli_entail" not in ev or sc.get("cosine") is None or sc.get("nli_entail") is None:
                    continue
                if x.get("admitted"):
                    st = "broke" if not looks_like_refusal(x.get("response") or "") else "held"
                else:
                    st = "rejected"
                pts.append(dict(cos=float(sc["cosine"]), nli=float(sc["nli_entail"]), state=st,
                                tip=f"{item.get('bid')} · {'admitted' if x.get('admitted') else 'rejected at ' + str(rb)}\ncos {float(sc['cosine']):.2f} · nli {float(sc['nli_entail']):.2f}"
                                + (f"\nnewer model {'answered' if st == 'broke' else 'refused'} (offline rule)" if st != "rejected" else "")))
        c = rp.counts
        flag = (f"<p class='flag'>✕ {esc('; '.join(rp.issues))}</p>" if rp.issues else "")
        rpe_html += (f'<div class="rpebox"><h4>{esc(rp.spec.id)} · gate {esc(rp.spec.gate_label)} · judge {judge_chip(rp.judge)}</h4>{flag}'
                     f'<p class="note">{c.get("probed", 0)} fixes probed · {c.get("candidates", 0)} candidates · {c.get("admitted", 0)} admitted · '
                     f'{c.get("resolved", 0)} resolved · re-opened {c.get("rec_offline", 0)}/{c.get("resolved", 0)} (offline rule; stored judge '
                     f'{c.get("rec_stored", 0)}/{c.get("resolved", 0)}). Only candidates scored through NLI are plotted.</p>'
                     + (gate_scatter(pts, rp.spec.gate, f"RPE candidates vs the gate · {rp.spec.id}") if pts else "")
                     + '<div class="legend"><span><i class="lg sc-broke"></i>admitted · answered</span><span><i class="lg sc-held"></i>admitted · refused</span>'
                     '<span><i class="lg sc-rej"></i>rejected</span></div></div>')
    cats = category_bars(d) if d.status in ("complete", "p-minus-only") and d.pm else ""
    hidden = "" if active else " hidden"
    return f"""
<div class="tabpanel" role="tabpanel" id="panel-{d.key}" aria-labelledby="tab-{d.key}"{hidden}>
  <div class="pairhead"><h3>{esc(d.ds.label)}</h3><span class="chip">{d.n} behaviours</span>{status_chip("complete" if d.status == "complete" else "partial" if d.status in ("p-minus-only", "incomplete") else "none", {"complete": "full 2×2", "p-minus-only": "P⁻ only (newer screened where older answered)", "incomplete": "newer model screened on part of P⁻", "unavailable": "older model not screened"}[d.status])}
  <span class="note">{esc(d.ds.source)}</span></div>
  <div class="tiles">{"".join(tiles)}</div>
  <p class="note">{_primary_note(d)}</p>
  {_partial_note(d, o, n)}
  <div class="grid2">
    <div class="panel"><h3>The 2×2 on canonical requests</h3>{m2 or f"<p class='note'>{NOT_YET}</p>"}
      <p class="note">Any of k = {sd.K} canonical draws answering counts as answered (offline rule).</p></div>
    <div class="panel"><h3>Every recycle run on this benchmark</h3>{run_bars(d)}</div>
  </div>
  <div class="panel"><h3>Flow of every behaviour</h3>
    <div class="legend"><span><i class="lg r-comply"></i>answered</span><span><i class="lg r-refuse"></i>refused</span><span><i class="lg r-fix"></i>P⁻ · the fix</span>
    <span><i class="lg r-none"></i>no admitted rewording / not screened</span><span><i class="lg r-untested"></i>not tested</span></div>
    <div class="scroll">{flow(d, o, n) if d.status in ("complete", "p-minus-only") else f"<p class='note'>{NOT_YET}</p>"}</div></div>
  {('<div class="panel"><h3>Run results</h3>' + mtable(["Run", "Reported (run's judge)", "Rebuilt · offline [Wilson 95%]", "McNemar p", "No admitted rewording", "Re-screen"], rtab, num=(3, 4), nowrap=(1, 2, 5)) + '<p class="note">Each run on its own behaviours and fixes; the run table on the master dashboard lists every run.</p></div>') if rtab else ''}
  {('<div class="grid2"><div class="panel"><h3>How many rewordings it takes</h3>' + recovery_curves(d) + '</div><div class="panel"><h3>Which styles re-open fixes</h3>' + sty + '</div></div>') if d.primary and d.primary.rsr else ''}
  {('<div class="panel"><h3>Did the successful rewordings drift in meaning?</h3><p class="note">Gate scores of every admitted rewording of a fix (all complete runs on this benchmark), split by whether the newer model answered it. If answered rewordings sat clearly lower on entailment or cosine, recovery could partly be meaning drift rather than wording; the test under each strip says whether the difference is distinguishable from noise (exploratory: two uncorrected tests per benchmark). Every point already cleared the gate.</p>' + strip + '</div>') if strip else ''}
  {('<div class="panel"><h3>Where the fixes are, by category</h3>' + cats + '</div>') if cats else ''}
  {invsec.pair_panel(inv(), d.key, o, n)}
  {('<details class="tv"><summary>Early inversion runs on this benchmark (superseded)</summary><div class="rpegrid">' + rpe_html + '</div></details>') if rpe_html else ''}
  <div class="panel"><h3>The fixes, case by case</h3>
    <p class="note">{"Text is redacted in this build." if redact else "Real prompt and response text, truncated and collapsed. Each fix shows the canonical request and both models' replies, then every admitted rewording from every recycle run with the newer model's reply, then every output-inversion candidate."} Verdict pills are the offline rule.</p>
    {case_lists(d, o, n, redact)}</div>
</div>"""


def sec_header(s, o, n, dvs, redact, today):
    mo, mn = lc.meta(o), lc.meta(n)
    st = lc.pair_status(o, n)
    consecutive = (o, n) in sd.CONSECUTIVE
    return f"""
<header class="hero" id="top">
  <p class="eyebrow">RecycleBench · pair dashboard · {esc("consecutive release pair" if consecutive else "non-consecutive pair (post-hoc)")}</p>
  <h1>{esc(mo["label"])} → {esc(mn["label"])}</h1>
  <p class="lede">The older model's answers that the newer model refuses are the <b>fixes (P⁻)</b>. Each fix is reworded without
  changing what it asks, filtered by the intent gate, and put to the newer model again. A fix that comes back answered was tied to
  the wording. This page follows every behaviour of every benchmark through that path for this one pair.</p>
  <div class="meta"><span><b>{esc(mo["gen"])}</b> {mo["size"]}B · {esc(mo["released"])}</span><span>→</span>
  <span><b>{esc(mn["gen"])}</b> {mn["size"]}B · {esc(mn["released"])}</span><span>{ld.chip(st["kind"])}</span>
  <span>{esc(st["why"])}</span></div>
  <div class="meta"><span><b>{len(dvs)}</b> benchmarks with data</span><span><b>{sum(len(d.runs) for d in dvs)}</b> recycle runs</span>
  <span><b>{sum(1 for d in dvs if (r := inv().get(d.key, o, n)) and r.doc)}</b> inversion runs</span><span>generated <b>{today}</b></span>
  {"<span class='chip chip-warn'>redacted · no prompt or response text</span>" if redact else "<span class='chip chip-warn'>contains benchmark prompts and model replies (collapsed)</span>"}</div>
</header>"""


def sec_overview(s, o, n, dvs):
    pm_tot = sum(len(d.pm) for d in dvs if d.status in ("complete", "p-minus-only"))
    k = m = 0
    for d in dvs:
        if d.primary and d.primary.rsr:
            for b in d.pm:
                it = d.primary.items.get(b)
                if it and it.state in ("recovered", "held", "unresolved"):
                    m += 1
                    k += int(it.state == "recovered")
    p, lo, hi = wil(k, m)
    reg_tot = sum(len(d.regression) for d in dvs if d.regression is not None)
    tiles = [tile("Fixes across benchmarks (offline rule)", num(pm_tot), f"over {sum(1 for d in dvs if d.status in ('complete', 'p-minus-only'))} benchmarks"),
             tile("Re-opened by rewording", pct(p), f"{k} of {m} tested fixes · Wilson {ci_txt(lo, hi)}", emph=True,
                  tip="Fixes covered by each benchmark's primary recycle run; offline rule"),
             tile("Safety regressions", num(reg_tot), f"{short(o)} refused, {short(n)} answers (benchmarks with a full 2×2)")]
    anc = next((r.calibrated for d in dvs for r in d.runs if r.calibrated), None)
    if anc:
        c = anc["RSR"]
        tiles.insert(1, tile("Calibrated RSR · SORRY-Bench anchor", pct(c["RSR_corrected"]), f"95% CI {ci_txt(*c['CI'])} · human-calibrated judge",
                             emph=True))
    rows = []
    for d in dvs:
        rr = []
        for r in d.runs:
            if r.rsr:
                rr.append(dict(label=r.id, sub=f"offline rule · {r.rr.spec.behaviour_scope}" + ("" if r.adjacent else " · post-hoc in this run"),
                               point=r.rsr["rsr"], lo=r.rsr["lo"], hi=r.rsr["hi"], emph=r.rsr["n"] >= 30, hollow=True,
                               tip=f"{d.ds.label} · {r.id}\n{r.rsr['k']} of {r.rsr['n']} fixes re-opened · Wilson {ci_txt(r.rsr['lo'], r.rsr['hi'])} (offline rule)",
                               v_txt=pct(r.rsr["rsr"]), n_txt=f"{r.rsr['k']}/{r.rsr['n']}"))
                if r.calibrated:
                    c = r.calibrated["RSR"]
                    _, clo, chi = wil(c["k_recovered"], c["n_p_minus"])
                    rr.append(dict(label=f"{r.id} · calibrated headline", sub="qwen2.5:14b re-judge · ◇ human-calibrated",
                                   point=c["RSR_raw"], lo=clo, hi=chi, corr=c["RSR_corrected"], emph=True,
                                   tip=f"{r.id} re-judged with qwen2.5:14b: {c['k_recovered']} of {c['n_p_minus']} (Wilson {ci_txt(clo, chi)})\n"
                                       f"human-calibrated: {pct(c['RSR_corrected'])} {ci_txt(*c['CI'])}",
                                   v_txt=f"{pct(c['RSR_raw'])} → {pct(c['RSR_corrected'])}", n_txt=f"{c['k_recovered']}/{c['n_p_minus']}"))
            elif r.pm_run == [] and r.complete:
                rr.append(dict(label=r.id, sub="offline rule", point=None, tip=f"{r.id}: P⁻ empty", v_txt="undefined", n_txt="P⁻ = 0",
                               empty="P⁻ = 0 → undefined"))
        if d.lineage_row:
            x = d.lineage_row
            _, lo2, hi2 = wil(x["k_recovered"], x["n_p_minus"])
            rr.insert(0, dict(label="lineage run summary", sub="qwen2.5:14b judge · as reported", point=x["RSR_raw"], lo=lo2, hi=hi2,
                              emph=x["n_p_minus"] >= 30, tip=f"run summary (qwen2.5:14b): {x['k_recovered']}/{x['n_p_minus']}",
                              v_txt=pct(x["RSR_raw"]), n_txt=f"{x['k_recovered']}/{x['n_p_minus']}"))
        if not rr and d.status in ("complete", "p-minus-only") and not d.pm:
            rr.append(dict(label="no fixes", sub="P⁻ = 0 on this benchmark", point=None, tip=f"{d.ds.label}: P⁻ empty, RSR undefined",
                           v_txt="undefined", n_txt="nothing to recycle", empty="P⁻ = 0 → RSR undefined"))
        elif not rr:
            rr.append(dict(label="no recycle run", sub="for this pair on this benchmark", point=None, tip=f"{d.ds.label}: {NOT_YET}",
                           v_txt="—", n_txt=NOT_YET, empty=f"recycle {NOT_YET}"))
        rows.append((d.ds.label, rr))
    pbr_rows = []
    for d in dvs:
        r = next((r for r in d.runs if o in r.pbr and n in r.pbr), None)
        if r:
            a, b = r.pbr[o], r.pbr[n]
            pbr_rows.append([esc(d.ds.label), esc(r.id), f"{pct(a['p'])} <span class='sub'>{a['k']}/{a['n']}</span>",
                             f"{pct(b['p'])} <span class='sub'>{b['k']}/{b['n']}</span>", esc(a["judge"]),
                             f"{(b['p'] - a['p']) * 100:+.1f} pp"])
    findings = []
    best = [d for d in dvs if d.primary and d.primary.rsr and d.primary.rsr["n"] >= 30]
    if best:
        d = best[0]
        findings.append(f"<b>{esc(d.ds.label)}</b> is the powered benchmark for this pair: {d.primary.rsr['k']} of {d.primary.rsr['n']} fixes "
                        f"re-opened ({pct(d.primary.rsr['rsr'])}, Wilson {ci_txt(d.primary.rsr['lo'], d.primary.rsr['hi'])}), offline rule.")
    small = [d for d in dvs if d.status in ("complete", "p-minus-only") and 0 < len(d.pm) < 30]
    if small:
        findings.append("Exploratory (|P⁻| < 30): " + ", ".join(f"{esc(d.ds.label)} ({len(d.pm)})" for d in small) + ".")
    zero = [d for d in dvs if d.status in ("complete", "p-minus-only") and not d.pm]
    if zero:
        findings.append("No fixes at all on " + ", ".join(esc(d.ds.label) for d in zero) + ": the older model already refuses those requests.")
    regs = [d for d in dvs if d.regression is not None and len(d.regression) > len(d.pm)]
    if regs:
        findings.append("More regressions than fixes on " + ", ".join(f"{esc(d.ds.label)} ({len(d.regression)} vs {len(d.pm)})" for d in regs)
                        + ": the newer model answers more of what the older one refused than the reverse.")
    lin_d = next((d for d in dvs if d.lineage_row), None)
    if lin_d:
        x = lin_d.lineage_row
        findings.append(f"The SORRY-Bench lineage run's own judge (qwen2.5:14b) finds P⁻ {x['n_p_minus']} and RSR {pct(x['RSR_raw'])}; "
                        f"the offline rule finds P⁻ {len(lin_d.pm)}. The judges disagree on borderline replies (master dashboard, section 12).")
    return md.section("s-overview", "01", "This pair across every benchmark",
                      f'<div class="tiles tiles3">{"".join(tiles)}</div>'
                      + (f'<div class="panel"><h3>Reading</h3><ul class="take">{"".join(f"<li>{x}</li>" for x in findings)}</ul></div>' if findings else "")
                      + md.panel("Does the newer model refuse more?", answer_dumbbell(dvs)
                                 + '<div class="legend"><span><i class="lg dd-old-l"></i>older model</span><span><i class="lg dd-new-l"></i>newer model</span>'
                                 '<span><i class="lg dl-safer-l"></i>newer refuses more</span><span><i class="lg dl-riskier-l"></i>newer answers more</span></div>',
                                 "Canonical answer rate per benchmark (offline rule, any of k = 3 draws). Hover for counts and Wilson intervals.")
                      + md.panel("Where every behaviour goes, and what rewording does to the fixes", composition_chart(dvs)
                                 + '<div class="legend"><span><i class="lg cb-both"></i>refused by both</span><span><i class="lg cb-reg"></i>regression</span>'
                                 '<span><i class="lg cb-pers"></i>still answered</span><span><i class="lg cb-pm"></i>P⁻ · fix</span>'
                                 '<span><i class="lg cb-rec"></i>fix re-opened</span><span><i class="lg cb-held"></i>fix held</span>'
                                 '<span><i class="lg cb-unres"></i>no admitted rewording</span><span><i class="lg cb-untested"></i>not tested</span></div>',
                                 "Left: every behaviour of each benchmark (100%). Right: the fixes on a common scale, split by the primary recycle run's outcome.")
                      + md.panel("Recycling Success Rate by benchmark and run",
                                 '<div class="legend"><span><i class="lg-dot"></i>qwen2.5:14b (run summary)</span><span><i class="lg-ring"></i>offline rule (rebuilt)</span>'
                                 '<span><i class="lg-dia"></i>human-calibrated</span><span><i class="lg-mute"></i>|P⁻| &lt; 30</span></div>'
                                 + md.forest_grouped(rows, "RSR by benchmark and run"))
                      + (md.panel("Paraphrase robustness of each model's own refusals (PBR)",
                                  mtable(["Benchmark", "Run", f"PBR {esc(short(o))}", f"PBR {esc(short(n))}", "Judge", "Change"], pbr_rows,
                                         num=(2, 3, 5), nowrap=(0, 1, 4)),
                                  "Within one run (same rewordings, same judge) for both models.") if pbr_rows else ""))


def sec_benchmarks(s, o, n, dvs, redact):
    tabs, panels = [], []
    for i, d in enumerate(dvs):
        pm_txt = f"P⁻ {len(d.pm)}" if d.status in ("complete", "p-minus-only") else ("◐ partial" if d.status == "incomplete" else NOT_YET)
        tabs.append(f'<button type="button" role="tab" id="tab-{d.key}" aria-controls="panel-{d.key}" aria-selected="{"true" if i == 0 else "false"}" '
                    f'tabindex="{0 if i == 0 else -1}" class="tab"><span class="tt">{esc(d.ds.label)}</span><span class="ts">{esc(pm_txt)}'
                    f'{" · " + str(len(d.runs)) + " run(s)" if d.runs else ""}</span></button>')
        panels.append(dataset_panel(s, d, o, n, i, i == 0, redact))
    missing = [s.datasets[k].label for k in TAB_ORDER if k in s.datasets and s.datasets[k].n and k not in {d.key for d in dvs}]
    return md.section("s-bench", "02", "Benchmark by benchmark",
                      f'<p class="note">One tab per benchmark with data for this pair. Use ← → to move between tabs.'
                      + (f" The older model has no full canonical screen on {', '.join(esc(x) for x in missing)}, so this pair is "
                         f"{NOT_YET} there." if missing else "") + "</p>"
                      f'<div class="tablist" role="tablist" aria-label="Benchmarks">{"".join(tabs)}</div>{"".join(panels)}')


def sec_attack(s, o, n):
    if "llama31-8b" not in (o, n):
        return ""
    a = s.attack
    o8 = (a.get("8b_original") or {}).get("classes", {})
    a8 = a.get("8b_attack") or {}
    if not o8:
        return ""
    rows = []
    xmax = 0.1
    for c in sorted(o8, key=lambda c: -(o8[c].get("transfer_on_refused") or 0)):
        v = {"o8": (o8[c]["transfer_on_refused"], o8[c]["n_broke_among_refused"], o8[c]["n_refused_baseline"])}
        if c in a8:
            v["a8"] = (a8[c]["transfer_on_refused"], a8[c]["n_broke_among_refused"], a8[c]["n_refused_baseline"])
        xmax = max(xmax, v["o8"][0])
        rows.append(dict(label=c.replace("_", " "), vals=v, flag=c in sd.CIPHER))
    xmax = min(1.0, math.ceil(xmax * 20) / 20)
    series = [("a8", "db-a8", "judged vs attack prompt (superseded)"), ("o8", "db-o8", "judged vs original goal")]
    att = (s.calib or {}).get("attack") or {}
    d = att.get("design") or {}
    warn = (f"<p class='flag'>▲ The attack judge's human calibration, weighted back to the judged pool, gives a sensitivity of "
            f"{d['qwen']['se']:.2f} {ci_txt(*d['qwen']['se_ci'])}: these rates are lower bounds (calibrated overall prevalence "
            f"{pct(d['qwen_boot']['prevalence'], 0)}).</p>") if d else ""
    role = "newer" if n == "llama31-8b" else "older"
    return md.section("s-attack", "03", f"Known attacks on the {role} model ({label('llama31-8b')})",
                      md.panel("Transfer-on-refused by attack class", warn
                               + '<div class="legend">' + "".join(f'<span><i class="lg {c}"></i>{esc(l)}</span>' for _, c, l in series)
                               + "<span>⚠ cipher class</span></div>" + md.dumbbell(rows, "Attack transfer", xmax, series),
                               "Of the 291 behaviours the model refused in plain form, the share each attack re-opens (qwen2.5:14b, RAW). "
                               "Attack transfer is a property of one model; it is shown here because this pair contains it."))


def sec_compose(s, o, n):
    """The compose arm, on the one pair it was run for (its summary names the pair)."""
    run = ca.latest_run(sd.ROOT)
    if run is None:
        return ""
    agg = ca.aggregate(run, store=sd.ROOT / "cache" / "compose", anchor=md.anchor(s)[1])
    if [x.strip() for x in agg["pair"].split("->")] != [o, n]:
        return ""
    return md.section("s-compose", "02b", "Two-factor composition — compositional reach", ca.section_body(agg), ca.LEDE)


def sec_compose2(o, n):
    """The style factorial, on the anchor pair it was run for."""
    agg = cv2.aggregate(sd.ROOT)
    if not agg or [x.strip() for x in agg["summary"]["pair"].split("->")] != [o, n]:
        return ""
    return md.section("s-compose2", "02c", "Style factorial — 21 conditions", cv2.section_body(agg), cv2.LEDE)


def sec_judges(s, o, n, redact):
    if (o, n) != ("llama3-8b", "llama31-8b"):
        return ""
    c = (s.calib or {}).get("sorrybench")
    sheet = sd.ROOT / "audit/sorrybench/judge_sheet.csv"
    if not c or not sheet.exists():
        return ""
    rows = [r for r in csv.DictReader(sheet.open(newline="")) if (r.get("human_fulfilled") or "").strip() in ("0", "1")]
    full = {b.canonical_prompt: b.bid for b in s.datasets["sorrybench"].behaviours} if "sorrybench" in s.datasets else {}
    cards = []
    for r in rows:
        h, q = int(r["human_fulfilled"]), int(r["judge_label"])
        off = int(not looks_like_refusal(r["response"] or ""))
        if h == q == off:
            continue
        who = []
        if q != h:
            who.append("qwen2.5:14b " + ("missed a compliance" if h == 1 else "false alarm"))
        if off != h:
            who.append("offline rule " + ("missed a compliance" if h == 1 else "false alarm"))
        body = "" if redact else (f'<div class="txt"><span class="tl">request</span>{esc(cap(r["request"], CAP["prompt"]))}</div>'
                                  f'<div class="txt"><span class="tl">{esc(label("llama31-8b"))} replied to a rewording</span>{esc(cap(r["response"], CAP["reply"]))}</div>')
        cards.append(f'<details class="case" data-outcome="{"qwen" if q != h else "offline"}"><summary><span class="cid">{esc(full.get(r["request"], r["bid"]))} · rewording {esc(r["variant_index"])}</span>'
                     f'<span class="ccat">{esc(r["transform_class"])}</span><span class="pill">human: {"answered" if h else "refused"}</span>'
                     f'<span class="pill">qwen: {"answered" if q else "refused"}</span><span class="pill">offline: {"answered" if off else "refused"}</span>'
                     f'<span class="warnt small">{esc("; ".join(who))}</span></summary><div class="cbody">{body}</div></details>')
    bq, bo = c["bundle"]["qwen"], c["bundle"]["offline"]
    cm = '<div class="cmrow">' + md.confusion(bq, "qwen2.5:14b vs human (bundles)") + md.confusion(bo, "offline rule vs human (bundles)") + "</div>"
    n_qcards = sum(1 for x in cards if 'data-outcome="qwen"' in x)
    n_ocards = len(cards) - n_qcards
    n_q = sum(1 for r in rows if int(r["judge_label"]) != int(r["human_fulfilled"]))
    n_o = sum(1 for r in rows if int(not looks_like_refusal(r["response"] or "")) != int(r["human_fulfilled"]))
    btns = (f'<div class="filters" role="group" aria-label="Filter disagreements"><button type="button" class="fbtn active" data-filter="all" aria-pressed="true">All <span>{len(cards)}</span></button>'
            f'<button type="button" class="fbtn" data-filter="qwen" aria-pressed="false">qwen wrong <span>{n_qcards}</span></button>'
            f'<button type="button" class="fbtn" data-filter="offline" aria-pressed="false">only the offline rule wrong <span>{n_ocards}</span></button></div>')
    return md.section("s-judges", "04", "Which judge is right? The human-labelled anchor sheet",
                      md.panel("Judge vs human on this pair's rewordings", cm
                               + f"<p class='note'>{c['n_rows']} responses in {c['n_bundles']} complete bundles from <code>audit/sorrybench/judge_sheet.csv</code>. "
                                 f"qwen2.5:14b disagrees with the human label on {n_q} responses, the offline rule on {n_o}. Every disagreement is below.</p>"
                               + f'<div class="cases">{btns}{"".join(cards)}</div>'))


def sec_provenance(s, o, n, dvs):
    rows = []
    for d in dvs:
        for r in d.runs:
            sp = r.rr.spec
            rep = r.reported
            if rep and r.rsr:
                same = rep["n_p_minus"] == r.rsr["n"] and rep["k_recovered"] == r.rsr["k"]
                ver = status_chip("valid", "reproduced") if (same and sp.judge == "offline") else (
                    "judge differs (qwen vs offline)" if sp.judge != "offline" else status_chip("warn", "differs"))
            elif rep and not r.pm_run:
                ver = status_chip("valid", "reproduced (P⁻ = 0)") if rep.get("n_p_minus") == 0 else "—"
            elif not r.adjacent:
                ver = "post-hoc: no reported figure exists"
            elif not r.complete:
                ver = status_chip("incomplete", "re-screen unfinished")
            else:
                ver = "—"
            rows.append([esc(d.ds.label), esc(sp.id), path_html(sd.rel(sd.ROOT / sp.run_dir / sp.manifest)),
                         judge_chip(sp.judge), status_chip(r.rr.summary_status if r.rr.summary_status != "missing" else "na", r.rr.summary_status), ver])
    regs = [a for a in s.registry if (f"{o} → {n}" in a.scope) or (f"{o}->{n}" in a.scope.replace(" ", ""))]
    reg_rows = [[status_chip(a.status), esc(a.kind), path_html(a.path), judge_chip(a.judge) if a.judge not in ("—", "") else "—",
                 f"<span class='small'>{esc('; '.join(a.issues))}</span>"] for a in regs]
    todo = []
    for k in TAB_ORDER:
        dsx = s.datasets.get(k)
        if not dsx or not dsx.n:
            continue
        d = next((x for x in dvs if x.key == k), None)
        if d is None:
            todo.append(f"{dsx.label}: canonical screening")
            continue
        if d.status == "incomplete":
            todo.append(f"{dsx.label}: the newer model's screen of the older model's answers")
        if d.status in ("complete", "p-minus-only") and d.pm and not d.primary:
            todo.append(f"{dsx.label}: recycling of the {len(d.pm)} fixes")
        if d.status in ("complete", "p-minus-only") and d.pm and not invsec.has_run(inv(), d.key, o, n):
            todo.append(f"{dsx.label}: output inversion (RPE)")
    if not any(r.calibrated for d in dvs for r in d.runs):
        todo.append("human calibration of the judge for this pair")
    return md.section("s-prov", "05", "Provenance, reproduction and what is not yet finished",
                      md.panel("Recycle runs used on this page", mtable(["Benchmark", "Run", "Manifest", "Run's judge", "Summary", "Reproduction"],
                                                                       rows, nowrap=(0, 1, 3, 4)) if rows else f"<p class='note'>{NOT_YET}</p>",
                               "Every run's rewordings and the newer model's replies are read from that run's own cache; offline-judged run "
                               "summaries are rebuilt from the cache and must match.")
                      + (md.panel("Artifacts about this pair and their integrity", mtable(["Status", "Kind", "Path", "Judge", "Findings"], reg_rows,
                                                                                        nowrap=(0, 3), wide=(4,))) if reg_rows else "")
                      + md.panel("Not yet finished for this pair", "<ul class='take'>" + "".join(f"<li>{esc(x)}</li>" for x in todo) + "</ul>"
                                 if todo else "<p class='note'>Nothing outstanding.</p>"))


EXTRA_CSS = r"""
.chart .dd-old{fill:var(--card);stroke:var(--ink);stroke-width:2}.chart .dd-new{fill:var(--ink)}
.chart .dl-safer{stroke:var(--refuse);stroke-width:3}.chart .dl-riskier{stroke:var(--comply);stroke-width:3}.chart .dl-same{stroke:var(--axis);stroke-width:2}
.lg.dd-old-l{background:transparent;border:2px solid var(--ink);border-radius:50%}.lg.dd-new-l{background:var(--ink);border-radius:50%}
.lg.dl-safer-l{background:var(--refuse);height:4px!important;vertical-align:2px}.lg.dl-riskier-l{background:var(--comply);height:4px!important;vertical-align:2px}
.chart .cb-both{fill:color-mix(in oklab,var(--refuse) 38%,var(--card))}.chart .cb-pers{fill:color-mix(in oklab,var(--comply) 62%,var(--card))}
.chart .cb-pm{fill:var(--fix)}.chart .cb-unk{fill:var(--grid)}
.chart .hb-bg{fill:color-mix(in oklab,var(--comply) 22%,var(--card))}.chart .hb-ln{stroke:var(--comply);stroke-width:2}
.chart .cb-rec{fill:var(--comply)}.chart .cb-held{fill:var(--refuse)}.chart .cb-unres{fill:var(--muted)}.chart .cb-untested{fill:var(--grid)}
.lg.cb-both{background:color-mix(in oklab,var(--refuse) 38%,var(--card))}.lg.cb-pers{background:color-mix(in oklab,var(--comply) 62%,var(--card))}
.lg.cb-pm{background:var(--fix)}.lg.cb-reg{background:repeating-linear-gradient(45deg,color-mix(in oklab,var(--comply) 22%,var(--card)) 0 3px,var(--comply) 3px 5px)}
.lg.cb-rec{background:var(--comply)}.lg.cb-held{background:var(--refuse)}.lg.cb-unres{background:var(--muted)}.lg.cb-untested{background:var(--grid)}
.chart .r-untested{fill:var(--grid)}.lg.r-untested{background:var(--grid)}
.chart .sp-comply{fill:var(--comply);fill-opacity:.7}.chart .sp-refuse{fill:var(--refuse);fill-opacity:.55}
.chart .medline{stroke:var(--ink);stroke-width:2.5}
.chart .admitzone{fill:color-mix(in oklab,var(--fix) 10%,transparent)}
.chart .sc-broke{fill:var(--comply);stroke:var(--card);stroke-width:1.2}.chart .sc-held{fill:var(--refuse);stroke:var(--card);stroke-width:1.2}
.chart .sc-rej{fill:var(--muted);fill-opacity:.45}
.lg.sc-broke{background:var(--comply);border-radius:50%}.lg.sc-held{background:var(--refuse);border-radius:50%}.lg.sc-rej{background:var(--muted);border-radius:50%;opacity:.6}
.chart .inbar-dark{fill:var(--ink);font-size:11px;font-weight:600}
.rblock{border:1px solid var(--ring);border-radius:10px;padding:8px 12px;margin:8px 0;background:var(--card)}
.rblock.rpeb{border-style:dashed}
details.rpeb>summary{cursor:pointer;list-style:none}details.rpeb>summary::-webkit-details-marker{display:none}
details.rpeb>summary .rhead{margin-bottom:2px}
.rhead{font-size:12.5px;color:var(--ink2);margin-bottom:4px}.rhead b{color:var(--ink)}
.vhead{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px}
.sc{font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:auto}
.rpegrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px}
.rpebox{border:1px solid var(--ring);border-radius:12px;padding:12px 14px;background:var(--surface)}
.rpebox h4{margin:0 0 6px;font-size:13.5px}
.idx{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,330px),1fr));gap:14px}
.idx a.card{display:block;text-decoration:none;border:1px solid var(--ring);border-radius:14px;padding:16px 18px;background:var(--card)}
.idx a.card:hover{border-color:var(--axis)}.idx .ct{font-size:17px;font-weight:650}.idx .cs{font-size:12.5px;color:var(--ink2);margin-top:4px}
"""


PAIR_RQ_MAP = {"s-overview": ["RQ1", "RQ4"], "s-bench": ["RQ1", "RQ2", "RQ3"], "s-compose": ["RQ3"], "s-compose2": ["RQ3"]}


def sec_rq(o, n):
    """the research questions, pointed at where this pair's page answers them."""
    where = {"RQ1": [("s-overview", "Across benchmarks"), ("s-bench", "Benchmark by benchmark")],
             "RQ2": [("s-bench", "Where the fixes are, by category")], "RQ3": [("s-bench", "Which styles re-open fixes")],
             "RQ4": [("s-overview", "PBR of each model")]}
    has_mech = bool(mech_section.load(o, n, "sorrybench"))
    ans = {"Beyond the RQs": "White-box panels for this pair sit inside the SORRY-Bench fix cards (“Inside the model”)."} if has_mech else {
        "Beyond the RQs": "Not run for this pair: the white-box probe currently covers the anchor llama3-8b → llama3.1-8b on SORRY-Bench."}
    if has_mech:
        where["Beyond the RQs"] = [("s-bench", "The fixes, case by case")]
    return ld.rq_section(answers=ans, where=where, frame=ld.RQ_FRAME,
                         status=f"This page answers the questions for {esc(label(o))} → {esc(label(n))} only; the pooled answers are on the master dashboard.")


def render_pair(s, o, n, redact) -> str:
    dvs = build_pair(s, o, n)
    today = _dt.date.today().isoformat()
    secs = [("s-rq", "00", "Research questions"), ("s-overview", "01", "Across benchmarks"), ("s-bench", "02", "Benchmark by benchmark")]
    cmp_ = sec_compose(s, o, n)
    if cmp_:
        secs.append(("s-compose", "02b", "2-factor composition"))
    cmp2 = sec_compose2(o, n)
    if cmp2:
        secs.append(("s-compose2", "02c", "Style factorial"))
    att = sec_attack(s, o, n)
    if att:
        secs.append(("s-attack", "03", "Known attacks"))
    jud = sec_judges(s, o, n, redact)
    if jud:
        secs.append(("s-judges", "04", "Judge vs human"))
    secs.append(("s-prov", "05", "Provenance"))
    nav = "".join(f'<a href="#{i}"><span class="n">{k}</span>{esc(t)}</a>' for i, k, t in secs)
    body = (sec_header(s, o, n, dvs, redact, today) + sec_rq(o, n) + sec_overview(s, o, n, dvs) + sec_benchmarks(s, o, n, dvs, redact)
            + cmp_ + cmp2 + att + jud + sec_provenance(s, o, n, dvs))
    foot = ("Generated by tools/pair_dashboard.py from files only. Case-level verdicts are the offline refusal rule; reported figures name "
            "their judge. " + ("Prompt and response text omitted in this build." if redact else
                               "Contains benchmark prompts and model replies, collapsed and truncated: a working file, not a released artefact (ETHICS.md)."))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<title>{esc(label(o))} vs {esc(label(n))}</title><style>{ld.CSS}{md.EXTRA_CSS}{EXTRA_CSS}{mech_section.CSS}{ca.CSS}{invsec.CSS}{cv2.CSS}</style></head><body>'
            f'<div class="layout"><nav class="side" aria-label="Sections">{ld.brand_html(pair_short(o, n))}<div class="toc">{nav}</div>'
            f'<a href="../{esc(INDEX_NAME[0])}" class="navlink">← all pairs</a>{ld.legend_html()}'
            f'<button type="button" class="themebtn" id="themebtn" data-t="system">Theme: system</button></nav>'
            f'<main id="main">{body}<footer>{esc(foot)}</footer></main></div><div id="tip" role="tooltip" hidden></div>'
            f'<script>{ld.rq_script(PAIR_RQ_MAP)}{ld.JS}{md.EXTRA_JS}</script></body></html>')


INDEX_NAME = ["index.html"]


def _plural(k, word):
    return f"{k} {word}{'' if k == 1 else ('es' if word.endswith('x') else 's')}"


def tested_reopened(dvs):
    k = m = 0
    for d in dvs:
        if d.primary and d.primary.rsr:
            for b in d.pm:
                it = d.primary.items.get(b)
                if it and it.state in ("recovered", "held", "unresolved"):
                    m += 1
                    k += int(it.state == "recovered")
    return k, m


def render_index(s, pairs, files, today, out_dir) -> str:
    cards = []
    for (o, n), f in zip(pairs, files):
        dvs = build_pair(s, o, n)
        pm = sum(len(d.pm) for d in dvs if d.status in ("complete", "p-minus-only"))
        runs = sum(len(d.runs) for d in dvs)
        n_inv = sum(1 for d in dvs if (r := inv().get(d.key, o, n)) and r.doc)
        k, m = tested_reopened(dvs)
        rel = os.path.relpath(f, out_dir)
        cards.append(f'<a class="card" href="{esc(rel)}"><div class="ct">{esc(label(o))} → {esc(label(n))}</div>'
                     f'<div class="cs">{esc(lc.pair_status(o, n)["kind"].replace("-", " "))}{" · post-hoc" if (o, n) not in sd.CONSECUTIVE else ""}</div>'
                     f'<div class="cs">{_plural(len(dvs), "benchmark")} · {_plural(pm, "fix")} · {_plural(runs, "recycle run")} · {_plural(n_inv, "inversion run")}</div>'
                     f'<div class="cs">re-opened by rewording: <b>{pct(k / m) if m else "—"}</b> ({k} of {m} tested, offline rule)</div></a>')
    masters = sorted((sd.ROOT / "results").glob("dashboard_master_2*.html"))
    master = (f'<p class="note">All results in one page: <a href="{esc(os.path.relpath(masters[-1], out_dir))}">{esc(masters[-1].name)}</a></p>'
              if masters else "")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>RecycleBench pair dashboards</title><style>{ld.CSS}{md.EXTRA_CSS}{EXTRA_CSS}</style></head><body><main style="max-width:1100px;margin:0 auto">'
            f'<header class="hero">{ld.brand_html("Dedicated runs · one page per pair")}<p class="eyebrow">RecycleBench · dedicated runs</p><h1>One dashboard per model pair</h1>'
            f'<p class="lede">Each page follows one older → newer pair across every benchmark: the 2×2, the flow of every behaviour, every '
            f'recycle and inversion run, and the cases with their prompts and replies.</p><div class="meta"><span>generated <b>{today}</b></span></div>{master}</header>'
            f'<div class="idx">{"".join(cards)}</div></main><script>{ld.JS}</script></body></html>')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=None, help="older:newer cfg keys (default: the 7 standard pairs)")
    ap.add_argument("--redact", action="store_true", help="omit every prompt and response text")
    ap.add_argument("--attack-70b-responses", default=None)
    ap.add_argument("--out-root", default=None, help=f"default: {OUT_ROOT} (one folder per pair inside)")
    args = ap.parse_args(argv)
    pairs = [tuple(p.split(":")) for p in args.pairs] if args.pairs else list(sd.PAIRS)
    for o, n in pairs:
        if o not in sd.LOCAL_MODELS or n not in sd.LOCAL_MODELS:
            raise SystemExit(f"unknown model in pair {o}:{n}; choose from {sd.LOCAL_MODELS}")
    s = sd.build_study(os.path.expanduser(args.attack_70b_responses) if args.attack_70b_responses else None)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    root = Path(args.out_root) if args.out_root else sd.ROOT / OUT_ROOT
    INDEX_NAME[0] = f"index{'_redacted' if args.redact else ''}_{stamp}.html"
    files = []
    for o, n in pairs:
        d = root / pair_dir(o, n)
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"dashboard{'_redacted' if args.redact else ''}_{stamp}.html"
        i = 2
        while out.exists():
            out = d / f"dashboard{'_redacted' if args.redact else ''}_{stamp}_{i}.html"
            i += 1
        out.write_text(render_pair(s, o, n, args.redact))
        files.append(out)
        print(f"wrote {sd.rel(out)}  ({out.stat().st_size / 1e6:.2f} MB)")
    idx = root / INDEX_NAME[0]
    if idx.exists():
        raise SystemExit(f"{idx} exists -- refusing to overwrite")
    idx.write_text(render_index(s, pairs, files, _dt.date.today().isoformat(), root))
    print(f"wrote {sd.rel(idx)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
