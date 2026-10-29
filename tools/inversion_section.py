#!/usr/bin/env python3
"""tools/inversion_section.py -- the inversion (RPE) arm on the master, pair and lineage dashboards.

One loader and three renderers over the current closed-loop runs (tools/rpe_lineage.py ->
tools/rpe_closedloop.py, gate decisions from tools/rpe_gate.py):

    load()                     every benchmark x pair run, its newest gate-scored artifact, the
                               convergent-validity report view (RSR beside PRR), the audit state
                               (tools/rpe_audit_score.status), the negative controls and the
                               reconciliation with the compose run
    master_body(inv, ...)      master section 09: tiles, one tab per benchmark, the strip chart
                               (coverage and re-opening rate under each admission), the table
    master_notes(inv, ...)     master section 09b: reconciliation, audit, controls, early runs
    pair_panel / pair_fix      a pair page's per-benchmark panel and per-fix candidate block
    lineage_table(inv)         the lineage page's section 09 table

Admission sets, as tools/rpe_closedloop.py defines them:

    A_gate  the production intent gate against the original request  (what RSR uses)
    A_loop  OLD(x_hat) complies and its reply ~= OLD's original answer (upper bound)
    A_both  both                                                       (primary)

Text-free except pair_fix, which embeds reconstructions and the newer model's replies (collapsed)
unless redact=True. Adds no <script>: the page's shared JS drives tooltips and tabs.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lineage_dashboard as ld                  # noqa: E402

esc, pct, num = ld.esc, ld.pct, ld.num

BENCHES = ("sorrybench", "harmbench", "advbench", "strongreject")
BENCH_LABEL = {"sorrybench": "SORRY-Bench", "harmbench": "HarmBench", "advbench": "AdvBench",
               "strongreject": "StrongREJECT"}
ROOTS = {"sorrybench": ("rpe_lineage", "rpe_lineage_posthoc")}
for _b in BENCHES[1:]:
    ROOTS[_b] = (f"rpe_bench/{_b}", f"rpe_bench_posthoc/{_b}")
PREFIX = {"sorrybench": "rpe_lin", **{b: f"rpe_{b}" for b in BENCHES[1:]}}
ANCHOR = ("sorrybench", "llama3-8b", "llama31-8b")
ADMISSIONS = (("gate", "intent gate"), ("both", "gate + closed loop"), ("loop", "closed loop"))
SHORT = {"llama2-7b": "2-7B", "llama2-13b": "2-13B", "llama3-8b": "3-8B", "llama31-8b": "3.1-8B",
         "llama32-1b": "3.2-1B", "llama32-3b": "3.2-3B"}
NOT_YET = "not yet finished"


# =============================================================================
# data
# =============================================================================
@dataclass
class InvRun:
    bench: str
    old: str
    new: str
    root: Path
    posthoc: bool
    p_minus: Optional[int] = None
    manifest: Optional[Path] = None
    closedloop: Optional[Path] = None
    doc: Optional[dict] = None
    view: Optional[dict] = None                  # the report row: RSR, PRR*, agreement*
    n_candidates: int = 0
    rejections: Counter = field(default_factory=Counter)

    @property
    def gated(self) -> bool:
        return bool(self.doc) and self.doc.get("gate_source", "manifest flag") != "manifest flag"

    def adm(self, a: str) -> Optional[dict]:
        return (self.doc or {}).get("admission", {}).get(a)

    @property
    def label(self) -> str:
        return f"{SHORT.get(self.old, self.old)} → {SHORT.get(self.new, self.new)}"


@dataclass
class Inv:
    runs: dict
    audit: dict
    controls: Optional[dict]
    reconcile: Optional[dict]
    runs_root: Path
    root: Path = ROOT

    def for_bench(self, bench):
        return [r for (b, o, n), r in self.runs.items() if b == bench]

    def get(self, bench, old, new) -> Optional[InvRun]:
        return self.runs.get((bench, old, new))


def _rel(p) -> str:
    if p is None:
        return ""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _newest(root: Path, pattern: str) -> Optional[Path]:
    xs = sorted(root.glob(pattern)) if root.exists() else []
    return xs[-1] if xs else None


def _report_views(root: Path) -> dict:
    p = _newest(root, "lineage_report_*.json") or (root / "lineage_report.json")
    if not p.exists():
        return {}
    return {tuple(v["pair"]): v for v in json.loads(p.read_text()).get("views", [])}


def load(root: Path = None, runs_root: Path = None, audit_dir: Path = None) -> Inv:
    """Every inversion run under `root` (default: the repo). Missing pieces load as empty, so a
    page built on a partial tree renders 'not yet finished' instead of failing."""
    root = Path(root) if root else ROOT
    runs_root = Path(runs_root) if runs_root else root / "results" / "runs"
    import rpe_audit_score as ras
    import rpe_gate as rg
    from recyclebench.config import Config
    cfg_p = root / "configs/real_llama_lineage_all.yaml"
    sig = rg.signature(Config.load(cfg_p if cfg_p.exists() else ROOT / "configs/real_llama_lineage_all.yaml"))
    store = rg.load_store(root / rg.STORE)
    runs = {}
    for bench in BENCHES:
        for i, rel in enumerate(ROOTS[bench]):
            base = runs_root / rel
            if not base.exists():
                continue
            views = _report_views(base)
            plan = base / "pairs.json"
            for row in json.loads(plan.read_text()) if plan.exists() else []:
                o, n = row["old"], row["new"]
                r = InvRun(bench, o, n, base, posthoc=bool(i), p_minus=row.get("p_minus"))
                man = base / f"rpe_manifest_{o}__to__{n}.jsonl"
                r.manifest = man if man.exists() and man.stat().st_size else None
                plain = base / f"closedloop_{o}__to__{n}.json"
                r.closedloop = _newest(base, f"closedloop_{o}__to__{n}_gated_*.json") or (plain if plain.exists() else None)
                r.doc = json.loads(r.closedloop.read_text()) if r.closedloop else None
                r.view = views.get((o, n))
                if r.manifest:
                    for line in r.manifest.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        m = json.loads(line)
                        for d in m.get("details") or []:
                            if not d.get("text"):
                                continue
                            r.n_candidates += 1
                            dec = rg.lookup(store, sig, m.get("prompt", ""), d["text"])
                            r.rejections["unscored" if dec is None else
                                         ("admitted" if dec["admitted"] else dec["rejected_by"])] += 1
                runs[(bench, o, n)] = r
    controls = _newest(runs_root / "rpe_controls", "control_report_*.json")
    rec = _newest(root / "results", "analysis_rpe_reconcile_*/reconcile.json")
    return Inv(runs, ras.status(Path(audit_dir) if audit_dir else root / "audit" / "rpe", root=root),
               json.loads(controls.read_text()) if controls else None,
               json.loads(rec.read_text()) if rec else None, runs_root, root)


def audited(inv: Inv, r: InvRun) -> Optional[dict]:
    """The audit row when every behaviour on the sheet is labelled (the bracket is closed), else None."""
    hb = harm_bracket(inv, r) if r else None
    return hb if hb and hb["pending"] == 0 else None


def brackets(inv: Inv) -> dict:
    """{(bench, old, new): audit row} for every run the newest harmful-ask sheet covers."""
    return {k: hb for k, r in (inv.runs.items() if inv else []) if (hb := harm_bracket(inv, r))}


def harm_bracket(inv: Inv, r: InvRun) -> Optional[dict]:
    """The harmful-ask audit row for this run, if the newest sheet covers it."""
    h = inv.audit.get("harm") if inv.audit else None
    if not h:
        return None
    for p in h["pairs"]:
        if p["benchmark"] == r.bench and p["pair"] == f"{r.old}->{r.new}":
            return p
    return None


# =============================================================================
# small formatting helpers
# =============================================================================
def _kn(x, empty="—"):
    if not x or not x.get("n"):
        return f'<span class="sub">{empty}</span>'
    return f'{pct(x["rate"], 0)} <span class="sub">{x["k"]}/{x["n"]}</span>'


def _res(x, p_minus):
    if not x or p_minus in (None, 0):
        return "—"
    return f'{x["n"]}/{p_minus}'


def _p(x):
    return "—" if x is None else ld.p_txt(x)


def status_line(inv: Inv, bench: str, old: str, new: str):
    """-> (css, text) for the master's 'What's finished' grid, or None when P^- is empty."""
    r = inv.get(bench, old, new) if inv else None
    if r is None:
        return None
    if not r.p_minus:
        return None
    lp = r.adm("loop")
    if not r.doc or not lp:
        return ("muted", f"RPE {NOT_YET}")
    g = r.adm("gate") if r.gated else None
    txt = f"PRR {lp['k']}/{lp['n']} closed loop"
    txt += f" · {g['k']}/{g['n']} gated" if g and g.get("n") else (" · gate not scored" if not r.gated else " · gate admits none")
    return ("ok" if r.gated else "warn", txt)


# =============================================================================
# charts
# =============================================================================
MARK = (("rsr", "RSR · recycle", "iv-rsr"), ("gate", "PRR · intent gate", "iv-gate"),
        ("both", "PRR · gate + loop", "iv-both"), ("loop", "PRR · closed loop", "iv-loop"))


LANE = {"rsr": -9, "gate": -3, "both": 3, "loop": 9}   # one lane per estimand: equal values never hide each other


def _marker(kind, x, cy, cls, r=5.5):
    if kind == "rsr":
        return f'<rect x="{x - r * .9:.1f}" y="{cy - r * .9:.1f}" width="{r * 1.8:.1f}" height="{r * 1.8:.1f}" rx="1.5" class="{cls}"/>'
    if kind == "both":
        d = r * 1.2
        return f'<path d="M{x:.1f},{cy - d:.1f} L{x + d:.1f},{cy:.1f} L{x:.1f},{cy + d:.1f} L{x - d:.1f},{cy:.1f} Z" class="{cls}"/>'
    return f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="{r}" class="{cls}"/>'


def strip(runs, title, brackets=None):
    """Two aligned panels per pair: share of P^- resolved, and share re-opened of the resolved,
    for RSR and the three PRR admissions. Coverage and rate must be read together: a high rate on
    a sliver of P^- is not the same evidence as a high rate on most of it."""
    rows = [r for r in runs if r.p_minus]
    if not rows:
        return ""
    W, rh, top = 880, 34, 34
    xa0, xa1, xb0, xb1 = 170, 470, 540, 840
    H = top + rh * len(rows) + 34
    y_bot = H - 30
    s = [ld.svg_open(W, H, title)]
    for (x0, x1, lab) in ((xa0, xa1, "resolved · share of P⁻"), (xb0, xb1, "re-opened · share of resolved")):
        s.append(f'<text x="{(x0 + x1) / 2:.0f}" y="16" class="ax strong" text-anchor="middle">{lab}</text>')
        s.append(ld.x_axis(x0, x1, top - 8, y_bot))
    for i, r in enumerate(rows):
        y = top + i * rh
        cy = y + rh / 2
        vals = {"rsr": (r.view or {}).get("RSR")}
        for a, _ in ADMISSIONS:
            vals[a] = r.adm(a) if (a == "loop" or r.gated) else None
        tip = [f"{BENCH_LABEL.get(r.bench, r.bench)} · {r.label} · P⁻ {r.p_minus}"
               + (" · post-hoc pair" if r.posthoc else "")]
        for kind, leg, _ in MARK:
            v = vals.get(kind)
            if not v or not v.get("n"):
                tip.append(f"{leg}: nothing resolved" if v is not None else f"{leg}: {NOT_YET}")
            else:
                tip.append(f"{leg}: resolved {v['n']}/{r.p_minus} · re-opened {v['k']}/{v['n']} ({pct(v['rate'])})")
        hb = (brackets or {}).get((r.bench, r.old, r.new))
        if hb and hb["verified"] + hb["refuted"]:
            tip.append(f"harmful-ask audit: verified {hb['verified']}, refuted {hb['refuted']}, pending {hb['pending']}"
                       f" → PRR in [{pct(hb['lower']['rate'])}, {pct(hb['upper']['rate'])}]")
        elif hb:
            tip.append(f"harmful-ask audit: {hb['pending']} re-opened behaviours on the sheet, none labelled yet")
        s.append(f'<g class="row" tabindex="0" data-tip="{esc(chr(10).join(tip))}">'
                 f'<rect x="0" y="{y}" width="{W}" height="{rh}" class="hit"/>'
                 f'<text x="8" y="{cy - 2:.1f}" class="lbl strong">{esc(r.label)}</text>'
                 f'<text x="8" y="{cy + 12:.1f}" class="ax">P⁻ {r.p_minus}{" · post-hoc" if r.posthoc else ""}</text>')
        if hb and hb["verified"] + hb["refuted"] and hb["lower"]["n"] and hb["upper"]["rate"] is not None:
            a, b = xb0 + (xb1 - xb0) * hb["lower"]["rate"], xb0 + (xb1 - xb0) * hb["upper"]["rate"]
            yl = cy + LANE["loop"]
            s.append(f'<line x1="{a:.1f}" x2="{b:.1f}" y1="{yl:.1f}" y2="{yl:.1f}" class="iv-brk"/>')
        for kind, _, cls in MARK:
            v = vals.get(kind)
            if not v or not v.get("n"):
                continue
            yk = cy + LANE[kind]
            s.append(_marker(kind, xa0 + (xa1 - xa0) * v["n"] / r.p_minus, yk, cls, 4.2))
            s.append(_marker(kind, xb0 + (xb1 - xb0) * v["rate"], yk, cls, 4.2))
        s.append("</g>")
    s.append("</svg>")
    return "".join(s)


def legend():
    items = "".join(f'<span><svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden="true">'
                    f'{_marker(k, 0, 0, c)}</svg>{esc(l)}</span>' for k, l, c in MARK)
    return (f'<div class="legend iv-legend">{items}<span><i class="lg iv-brk-lg"></i>harmful-ask audit bracket '
            f'(once labelled)</span></div>')


def funnel(runs, title):
    """Where the candidates go: first failing gate layer, per run (candidates, not behaviours)."""
    rows = [r for r in runs if r.n_candidates]
    if not rows:
        return ""
    keys = (("admitted", "iv-f-adm", "admitted by the gate"), ("lexical_divergence", "iv-f-lex", "too close (lexical)"),
            ("cosine", "iv-f-cos", "embedding cosine"), ("bertscore", "iv-f-bert", "BERTScore"),
            ("nli_entail", "iv-f-nli", "NLI entailment"), ("unscored", "iv-f-un", "not yet gated"))
    W, rh, top = 880, 30, 8
    x0, x1 = 190, 840
    H = top + rh * len(rows) + 8
    s = [ld.svg_open(W, H, title)]
    for i, r in enumerate(rows):
        y = top + i * rh
        tot = r.n_candidates
        s.append(f'<text x="8" y="{y + 14}" class="lbl strong">{esc(r.label)}</text>'
                 f'<text x="8" y="{y + 26}" class="ax">{tot} candidates</text>')
        x = x0
        for key, css, leg in keys:
            v = r.rejections.get(key, 0)
            if not v:
                continue
            w = (x1 - x0) * v / tot
            s.append(f'<rect x="{x:.1f}" y="{y + 5}" width="{max(w - 2, 1):.1f}" height="17" rx="3" class="{css}" '
                     f'tabindex="0" data-tip="{esc(r.label)} · {esc(leg)}: {v} of {tot} ({pct(v / tot)})"/>')
            if w > 30:
                s.append(f'<text x="{x + 5:.1f}" y="{y + 18}" class="inbar">{v}</text>')
            x += w
    s.append("</svg>")
    leg = "".join(f'<span><i class="lg {c}"></i>{esc(l)}</span>' for _, c, l in keys[:-1])
    return "".join(s) + f'<div class="legend">{leg}</div>'


# =============================================================================
# master
# =============================================================================
def _table(inv: Inv, runs):
    rows = []
    for r in runs:
        if not r.p_minus:
            rows.append([f'<span class="iv-nw">{esc(r.label)}</span>' + (' <span class="chip">post-hoc</span>' if r.posthoc else ""),
                         "0", '<span class="sub">P⁻ = 0 · nothing to invert</span>', "", "", "", "", "", ""])
            continue
        v = r.view or {}
        g, b, lp = (r.adm("gate") if r.gated else None), (r.adm("both") if r.gated else None), r.adm("loop")
        ag_l, ag_b = v.get("agreement"), v.get("agreement_both")
        hb = harm_bracket(inv, r)
        audit = "—"
        if hb and hb["verified"] + hb["refuted"]:
            audit = (f'[{pct(hb["lower"]["rate"], 0)}, {pct(hb["upper"]["rate"], 0)}] '
                     f'<span class="sub">{hb["verified"]} verified · {hb["refuted"]} refuted · {hb["pending"]} pending</span>')
        elif hb:
            audit = f'<span class="sub">not labelled yet · {hb["behaviours_on_sheet"]} behaviours on the sheet</span>'
        rows.append([f'<span class="iv-nw">{esc(r.label)}</span>' + (' <span class="chip">post-hoc</span>' if r.posthoc else ""),
                     num(r.p_minus), _kn(v.get("RSR")),
                     _kn(g, "gate not scored" if not r.gated else "none resolved") + f'<br><span class="sub">resolved {_res(g, r.p_minus)}</span>',
                     _kn(b, "—") + f'<br><span class="sub">resolved {_res(b, r.p_minus)}</span>',
                     _kn(lp) + f'<br><span class="sub">resolved {_res(lp, r.p_minus)}</span>',
                     _p(ag_l.get("mcnemar_p") if ag_l else None),
                     _p(ag_b.get("mcnemar_p") if ag_b else None), audit])
    head = ["pair", "P⁻", "RSR · recycle", "PRR · intent gate", "PRR · gate + loop", "PRR · closed loop",
            "McNemar RSR vs loop", "McNemar RSR vs both", "harmful-ask audit"]
    return ld.table(head, rows, num_cols=(1,))


def master_body(inv: Inv, uid="inv") -> str:
    if not inv or not inv.runs:
        return f"<p class='note'>No inversion run on disk: {NOT_YET}.</p>"
    br = brackets(inv)
    a = inv.get(*ANCHOR)
    tiles = []
    if a and a.doc:
        lp, g, b = a.adm("loop"), a.adm("gate"), a.adm("both")
        done = audited(inv, a)
        tiles.append(ld.tile("Anchor · closed loop" + (" · human-audited" if done else " (upper bound)"),
                             (f'{pct(done["lower"]["rate"], 0)} <span class="sub">{done["lower"]["k"]}/{done["lower"]["n"]}</span>'
                              if done else _kn(lp)),
                             f"resolved {lp['n']}/{a.p_minus} of P⁻ · Wilson {ld.ci_txt(*lp['ci'])}" if lp and lp.get("n") else "",
                             emph=True, tip=("A_loop, with every re-opened behaviour verified by the blind harmful-ask audit."
                                             if done else "A_loop: the older model's reply to the reconstruction is equivalent to its "
                                             "original answer. No intent check, so an upper bound until the harmful-ask audit.")))
        if a.gated:
            tiles.append(ld.tile("Anchor · intent gate (RSR's admission)", _kn(g, "none resolved"),
                                 f"resolved {g['n'] if g else 0}/{a.p_minus} of P⁻ · RSR {pct(((a.view or {}).get('RSR') or {}).get('rate'))}",
                                 tip="A_gate: the same production gate the paraphrase arm uses, against the original request."))
            tiles.append(ld.tile("Anchor · gate + loop (primary)", _kn(b, "none resolved"),
                                 f"resolved {b['n'] if b else 0}/{a.p_minus} of P⁻"))
    h = inv.audit.get("harm") if inv.audit else None
    tiles.append(ld.tile("Harmful-ask audit", f"{h['labelled']}/{h['rows']}" if h else "—",
                         (f"rows labelled · {sum(p['verified'] for p in h['pairs'])} re-opened behaviours verified"
                          if h and h["labelled"] == h["rows"] else "rows labelled · closes the bracket") if h else "no sheet exported yet",
                         tip="audit/rpe/rpe_harm_sheet_*.csv: does the reconstruction still ask for the same harmful thing?"))

    tabs, panels = [], []
    benches = [b for b in BENCHES if inv.for_bench(b)]
    for i, bench in enumerate(benches):
        on = i == 0
        runs = sorted(inv.for_bench(bench), key=lambda r: (r.posthoc, list(SHORT).index(r.old) if r.old in SHORT else 9))
        n_pm = sum(1 for r in runs if r.p_minus)
        tabs.append(f'<button type="button" role="tab" id="tab-{uid}-{bench}" aria-controls="panel-{uid}-{bench}" '
                    f'aria-selected="{"true" if on else "false"}" tabindex="{0 if on else -1}" class="tab">'
                    f'{esc(BENCH_LABEL[bench])} <span class="sub">{n_pm} pairs with P⁻</span></button>')
        gated = all(r.gated for r in runs if r.p_minus and r.doc)
        panels.append(
            f'<div role="tabpanel" id="panel-{uid}-{bench}" aria-labelledby="tab-{uid}-{bench}"{"" if on else " hidden"}>'
            + ('' if gated else '<p class="flag">▲ Some runs here were scored before the gate store existed: '
               'their intent-gate columns are empty because the gate never ran, not because it rejected everything.</p>')
            + f'<div class="panel"><h3>Coverage and re-opening, per admission rule</h3>{strip(runs, BENCH_LABEL[bench] + " inversion arm", br)}'
            + legend() + '</div>'
            + f'<div class="panel"><h3>Every pair</h3>{_table(inv, runs)}'
              '<p class="note">Per pair only: a behaviour sits in several pairs\' P⁻, so pooling would count it twice. '
              'McNemar compares RSR and PRR on the behaviours both arms resolved; κ is omitted because both arms '
              're-open most of what they resolve, which leaves no room for chance correction.</p></div>'
            + f'<div class="panel"><h3>Where the reconstructions go at the gate</h3>{funnel(runs, BENCH_LABEL[bench] + " gate outcomes")}'
              '<p class="note">First failing layer per candidate, with the same thresholds as the recycle arm. A reconstruction '
              'is a new request written from the answer, so most fail on surface similarity to the original request: the '
              '"entailment bound".</p></div></div>')
    tablist = f'<div class="tablist" role="tablist" aria-label="Benchmark">{"".join(tabs)}</div>' if len(tabs) > 1 else ""
    return f'<div class="tiles">{"".join(tiles)}</div>{tablist}{"".join(panels)}'


def _controls(inv: Inv) -> Optional[dict]:
    """-> {arm: (k, n)} candidates inside the equivalence class at the 0.75 floor, or None."""
    c = inv.controls if inv else None
    sw = (c or {}).get("threshold_sweep", {}).get("0.75")
    if not sw:
        return None
    return {k: (v["k"], v["n"]) for k, v in sw.items()}


def master_notes(inv: Inv, early_html: str = "") -> str:
    """Section 09b: why the numbers differ, what is audited, the controls, and the superseded runs."""
    out = []
    a = inv.get(*ANCHOR) if inv else None
    done = audited(inv, a) if a else None
    if done:
        take = [f"<li><b>On the anchor the closed loop is human-audited.</b> A blind harmful-ask audit verified all "
                f"{done['verified']} re-opened behaviours: each has at least one reconstruction that still asks for the same "
                f"harmful thing, so PRR {done['lower']['k']}/{done['lower']['n']} stands. Other pairs' closed-loop figures "
                "remain unaudited upper bounds.</li>"]
    else:
        take = ["<li><b>Read the closed-loop column as an upper bound.</b> It admits a reconstruction when the older "
                "model's reply to it matches its original answer; nothing checks that the reconstruction still asks "
                "for the same harmful thing.</li>"]
    if a and a.gated and a.adm("gate") and a.adm("loop"):
        g, lp, rs = a.adm("gate"), a.adm("loop"), (a.view or {}).get("RSR") or {}
        take.append(f"<li><b>Under the intent gate, inversion looks like paraphrase, on very few fixes.</b> On the anchor "
                    f"the gate resolves {g['n']} of {a.p_minus} fixes and re-opens {g['k']} ({pct(g['rate'])}), next to "
                    f"RSR {pct(rs.get('rate'))}; the closed loop resolves {lp['n']} and re-opens {lp['k']} "
                    f"({pct(lp['rate'])}). So the arm's lead over paraphrase on the anchor is carried by reconstructions "
                    + ("the gate rejects; the human audit finds those still ask for the same harmful thing, so the gate "
                       "under-admits inversions rather than filtering out drift.</li>" if done else
                       "the gate rejects, which is exactly the set the harmful-ask audit has to settle.</li>"))
    out.append(f"<div class='panel'><h3>How to read the inversion arm</h3><ul class='take'>{''.join(take)}</ul></div>")
    rec = inv.reconcile if inv else None
    if rec:
        rr = rec["runs"]
        rows = []
        for key, lab in (("compose_rpe_arm", "compose factorial · rpe arm"), ("dedicated_strict_2026_10_11", "first run · strict gate"),
                         ("dedicated_loose_2026_10_11", "first run · loose gate"), ("dedicated_lineage", "lineage run · 8 candidates"),
                         ("dedicated_lineage_first3_shared", "lineage run · first 3, shared behaviours")):
            x = rr.get(key)
            if not x:
                continue
            if "admission" in x and isinstance(x["admission"], dict):
                ad = x["admission"]
                for ak, al in (("gate", "intent gate"), ("loop", "closed loop")):
                    v = ad.get(ak) or {}
                    rows.append([esc(lab), esc(al), num(x["behaviours"]), num(v.get("admitted_candidates", 0)),
                                 f'{v.get("resolved", 0)}', f'{v.get("reopened", 0)}/{v.get("resolved", 0)}'])
            else:
                rows.append([esc(lab), esc(x.get("admission", "")), num(x["behaviours"]), num(x.get("admitted", 0)),
                             f'{x.get("resolved", 0)}', f'{x.get("reopened", "—")}/{x.get("resolved", 0)}'
                             if x.get("reopened") is not None else "— (circular judge)"])
        out.append("<div class='panel'><h3>38 of 49 against 7 of 61, reconciled</h3>"
                   + ld.table(["run", "admission", "behaviours", "admitted candidates", "resolved", "re-opened"], rows, num_cols=(2, 3, 4))
                   + "<ul class='take'>" + "".join(f"<li>{esc(f)}</li>" for f in rec.get("findings", [])) + "</ul>"
                   + f"<p class='note'>Recomputed from the stores by <code>tools/rpe_reconcile.py</code> "
                     f"({esc(rec.get('generated', ''))}).</p></div>")
    h = inv.audit.get("harm") if inv and inv.audit else None
    au = []
    if h:
        d = h["design"]
        au.append(f"<li><b>Harmful-ask sheet</b> <code>{esc(h['sheet'])}</code>: {h['labelled']} of {h['rows']} rows labelled. "
                  f"Up to {d.get('per_behaviour')} reconstructions per re-opened behaviour "
                  f"({d['population']['behaviours_reopened']} behaviours, {d['population']['reopening_candidates']} "
                  "re-opening reconstructions). A behaviour counts as re-opened only if one of its labelled "
                  "reconstructions still asks for the same harmful thing.</li>")
    for t in (inv.audit.get("topical") if inv and inv.audit else []) or []:
        w = t.get("design_weighted")
        au.append(f"<li><b>Earlier topical pass</b> <code>{esc(t['sheet'])}</code>: {t['rows']} rows, "
                  f"{'—' if w is None else pct(w)} judged to ask for the same thing. It covers the superseded 61-behaviour "
                  "loose-gate candidates, not the current runs, and the labels were read as a topical criterion, so it "
                  "neither confirms nor refutes the closed-loop figures above.</li>")
    if au:
        out.append(f"<div class='panel'><h3>Human audit</h3><ul class='take'>{''.join(au)}</ul></div>")
    cs = _controls(inv)
    if cs:
        c = inv.controls
        arms = c.get("arms", {})
        rows = [[esc(k), esc(arms.get(k, {}).get("shown", "")), f"{v[0]}/{v[1]}",
                 f'{arms.get(k, {}).get("old_complied", "—")}/{arms.get(k, {}).get("old_judged", "—")}',
                 f'{arms.get(k, {}).get("equiv_median", float("nan")):.2f}']
                for k, v in cs.items()]
        out.append("<div class='panel'><h3>Negative controls</h3>"
                   + ld.table(["inverter", "shown", "in the equivalence class (≥ 0.75)", "older model answered",
                               "median equivalence"], rows, num_cols=(2, 3, 4))
                   + "<p class='note'>Same inverter, judge and scoring, differing only in what the inverter sees. The older "
                     "model answers the control candidates about as readily as the real ones, but none lands in the original "
                     "answer's equivalence class, so closed-loop admission measures inversion, not topical guessing. The "
                     "controls ran on the first anchor run (61 behaviours, offline-rule P⁻), not on the lineage runs "
                     f"(<code>results/runs/rpe_controls/</code>, {esc(str(c.get('generated', '')))}).</p></div>")
    if early_html:
        out.append(f"<details class='tv'><summary>The first inversion runs, 2026-10 (superseded)</summary>{early_html}</details>")
    return "".join(out)


# =============================================================================
# pair pages
# =============================================================================
def pair_panel(inv: Inv, bench: str, old: str, new: str) -> str:
    r = inv.get(bench, old, new) if inv else None
    if r is None:
        return ""
    if not r.p_minus:
        return ("<div class='panel'><h3>Output inversion (RPE)</h3><p class='note'>Nothing to invert: under the qwen2.5:14b "
                "judge this pair has no fix here (judged P⁻ = 0), so the inversion arm has no answer to reconstruct from. "
                "Any fixes listed on this tab come from the offline refusal rule.</p></div>")
    hb = harm_bracket(inv, r)
    if not r.doc:
        return f"<div class='panel'><h3>Output inversion (RPE)</h3><p class='note'>{NOT_YET}: manifest built, closed loop not scored.</p></div>"
    tiles = []
    for a, lab in (("loop", "closed loop · " + ("human-audited" if audited(inv, r) else "upper bound")),
                   ("gate", "intent gate · RSR's admission"), ("both", "gate + loop · primary")):
        x = r.adm(a) if (a == "loop" or r.gated) else None
        tiles.append(ld.tile("PRR · " + lab, _kn(x, "none resolved" if r.gated or a == "loop" else "gate not scored"),
                             f"resolved {x['n']}/{r.p_minus} of P⁻" if x and x.get("n") is not None else ""))
    rs = (r.view or {}).get("RSR")
    tiles.append(ld.tile("RSR · recycle arm", _kn(rs), "same judged P⁻"))
    if hb:
        tiles.append(ld.tile("Harmful-ask audit", f'{hb["verified"]} verified' if hb["verified"] + hb["refuted"] else "not labelled",
                             f'{hb["refuted"]} refuted · {hb["pending"]} pending'))
    return (f"<div class='panel'><h3>Output inversion (RPE) · closed-loop run</h3><div class='tiles'>{''.join(tiles)}</div>"
            f"{strip([r], 'inversion arm on this pair', {(bench, old, new): hb} if hb else None)}{legend()}{funnel([r], 'gate outcomes on this pair')}"
            f"<p class='note'>{r.n_candidates} reconstructions of the older model's answers (qwen2.5:14b, 8 per fix), judged "
            "by qwen2.5:14b. Closed loop admits a reconstruction when the older model's reply to it matches its original "
            "answer (MiniLM cosine ≥ 0.75); the intent gate is the paraphrase arm's own gate against the original "
            f"request. Source: <code>{esc(_rel(r.closedloop))}</code>.</p></div>")


_pair_rows_cache: dict = {}


def _candidates(inv: Inv, r: InvRun):
    """-> {prompt: [row]} with text, gate decision, loop flag, verdicts and the NEW reply."""
    key = (str(inv.root), r.bench, r.old, r.new)
    if key in _pair_rows_cache:
        return _pair_rows_cache[key]
    import rpe_gate as rg
    from recyclebench.config import Config
    from recyclebench.realrun import ResponseCache
    from rpe_closedloop import seed_for
    cfg_p = inv.root / "configs/real_llama_lineage_all.yaml"
    sig = rg.signature(Config.load(cfg_p if cfg_p.exists() else ROOT / "configs/real_llama_lineage_all.yaml"))
    store = rg.load_store(inv.root / rg.STORE)
    cache = ResponseCache(inv.root / "cache" / f"{PREFIX[r.bench]}_{r.old}__{r.new}.jsonl")
    per = (r.doc or {}).get("per_behaviour", {})
    out = {}
    for line in r.manifest.read_text(encoding="utf-8").splitlines() if r.manifest else []:
        if not line.strip():
            continue
        m = json.loads(line)
        pb = per.get(m["bid"], {})
        loop = {x["ci"]: x for x in pb.get("loop") or []}
        gate = {x["ci"]: x for x in pb.get("gate") or []}
        rows = []
        for ci, d in enumerate(m.get("details") or []):
            if not d.get("text"):
                continue
            dec = rg.lookup(store, sig, m.get("prompt", ""), d["text"])
            row = loop.get(ci) or gate.get(ci) or {}
            rows.append({"ci": ci, "text": d["text"], "gate": dec, "loop": ci in loop,
                         "equiv": row.get("equiv"), "new_verdict": row.get("new_verdict"),
                         "reply": cache.get(r.new, d["text"], seed_for(m["bid"], ci)) if (ci in loop or ci in gate) else None})
        out[m.get("prompt", "")] = rows
    _pair_rows_cache[key] = out
    return out


def pair_fix(inv: Inv, bench: str, old: str, new: str, prompt: str, redact: bool, cap, caps, vpill, new_label) -> str:
    """One fix's inversion candidates for the case list (collapsed)."""
    r = inv.get(bench, old, new) if inv else None
    if r is None or not r.doc or not r.manifest:
        return ""
    rows = _candidates(inv, r).get(prompt)
    if not rows:
        return ""
    n_loop = sum(1 for x in rows if x["loop"])
    n_gate = sum(1 for x in rows if x["gate"] and x["gate"]["admitted"])
    hit = sum(1 for x in rows if (x["loop"] or (x["gate"] and x["gate"]["admitted"])) and x["new_verdict"] == 1)
    head = (f'<div class="rhead"><b>inversion arm</b> · {len(rows)} reconstructions · {n_loop} closed loop · '
            f'{n_gate} gate-admitted · judge qwen2.5:14b</div>')
    body = []
    for x in rows:
        g = x["gate"]
        gtxt = ("gate ✓" if g and g["admitted"] else f"gate ✕ {g['rejected_by']}" if g else "gate —")
        ltxt = f"loop {'✓' if x['loop'] else '✕'}" + (f" · eq {x['equiv']:.2f}" if x.get("equiv") is not None else "")
        scored = x["loop"] or (g and g["admitted"])
        cls = "v-comply" if x["new_verdict"] == 1 else ("v-refuse" if x["new_verdict"] == 0 else "v-none")
        txt = "" if redact else (
            f'<div class="txt"><span class="tl">reconstructed request</span>{esc(cap(x["text"], caps["cand"]))}</div>'
            + (f'<div class="txt"><span class="tl">{esc(new_label)} replied</span>{esc(cap(x["reply"], caps["creply"]))}</div>'
               if scored and x.get("reply") else ""))
        body.append(f'<div class="vrow {cls}"><div class="vhead"><span class="vtc">{esc(gtxt)}</span>'
                    f'<span class="vtc">{esc(ltxt)}</span>{vpill(x["new_verdict"]) if scored else ""}</div>{txt}</div>')
    return (f'<details class="rblock rpeb"><summary>{head}<span class="sub">{hit} answered by the newer model · '
            f'show reconstructions</span></summary>{"".join(body)}</details>')


def has_run(inv: Inv, bench: str, old: str, new: str) -> bool:
    r = inv.get(bench, old, new) if inv else None
    return bool(r and (r.doc or not r.p_minus))


# =============================================================================
# lineage page
# =============================================================================
def lineage_table(inv: Inv) -> str:
    runs = sorted(inv.for_bench("sorrybench"), key=lambda r: (r.posthoc, list(SHORT).index(r.old) if r.old in SHORT else 9)) if inv else []
    if not any(r.doc for r in runs):
        return ""
    return strip(runs, "SORRY-Bench inversion arm", brackets(inv)) + legend() + _table(inv, runs)


CSS = """
.chart .iv-rsr{fill:var(--brand)}
.chart .iv-gate{fill:var(--card);stroke:var(--ink2);stroke-width:2}
.chart .iv-both{fill:var(--fix-ink)}
.chart .iv-loop{fill:var(--comply)}
.chart .iv-brk{stroke:var(--warn);stroke-width:3;stroke-linecap:round}
.iv-legend svg{vertical-align:-2px;margin-right:5px}
.iv-legend svg .iv-rsr{fill:var(--brand)}.iv-legend svg .iv-gate{fill:var(--card);stroke:var(--ink2);stroke-width:2}
.iv-legend svg .iv-both{fill:var(--fix-ink)}.iv-legend svg .iv-loop{fill:var(--comply)}
.lg.iv-brk-lg{background:var(--warn);height:3px;border-radius:2px;vertical-align:middle}
.chart .iv-f-adm{fill:var(--fix)}.chart .iv-f-lex{fill:var(--none)}.chart .iv-f-cos{fill:var(--neutral);fill-opacity:.55}
.chart .iv-f-bert{fill:var(--sky)}.chart .iv-f-nli{fill:var(--refuse);fill-opacity:.7}.chart .iv-f-un{fill:var(--grid)}
.lg.iv-f-adm{background:var(--fix)}.lg.iv-f-lex{background:var(--none)}.lg.iv-f-cos{background:var(--neutral);opacity:.55}
.lg.iv-f-bert{background:var(--sky)}.lg.iv-f-nli{background:var(--refuse);opacity:.7}
.chart .ax.strong{font-weight:600;fill:var(--ink2)}
.iv-nw{white-space:nowrap}
"""
