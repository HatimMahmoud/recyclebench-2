#!/usr/bin/env python3
"""tools/method_section.py -- the method close-out (prereg Amendment 2) on the master dashboard, chapter 08b.

    aggregate(root)     newest results/analysis_power_* and results/analysis_c4_*, the anchor's stored corrected file and
                        its resolved-denominator variant (from the newest results/thesis_numbers_*.json), and the three
                        audits registered in Amendment 2 §B4 scored through their own tools (pending until labelled)
    section_body(agg)   four panels: the RSR denominator, naïve vs corrected (C4), power of the π* decision, the audits

Text-free. Adds no <script>; the page's shared JS drives the tooltips.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lineage_dashboard as ld                  # noqa: E402
import primary_calibration as pc                # noqa: E402

esc, pct, num, ci_txt = ld.esc, ld.pct, ld.num, ld.ci_txt
ANCHOR = "results/runs/recycle_3_31_sorrybench/corrected_sorrybench_recal_n60.json"
LEDE = ("How the estimates are read: which RSR denominator is registered, how far the naïve (string-matching) estimate sits "
        "from the corrected one, what the sample sizes can and cannot show, and the human audits still open. Specified in "
        "<code>prereg_amendment_2_DRAFT.md</code> (for signature).")
POWER_N = (9, 23, 49)
POWER_ROLE = {9: ("--sky", "confirmatory, n = 9"), 23: ("--fix", "2-7B → 3.1-8B, n = 23"), 49: ("--comply", "anchor, n = 49")}


def _newest(root, pattern):
    c = sorted(Path(root).glob(pattern))
    return c[-1] if c else None


def _read(p):
    try:
        return json.loads(Path(p).read_text()) if p else None
    except (OSError, ValueError):
        return None


def aggregate(root: Path = ROOT) -> dict:
    root = Path(root)
    import attack_audit as aa
    import gate_recheck as grc
    import gate_rejection_audit as gra
    import overrefusal_judge_audit as oja
    import csv
    nums = _read(_newest(root, "results/thesis_numbers_*.json"))
    rel = root / "audit/sorrybench/judge_sheet_blind_relabel_20261030.csv"
    rel_rows = list(csv.DictReader(rel.open())) if rel.exists() else []
    return dict(
        power=_read(_newest(root, "results/analysis_power_*/power.json")),
        c4=_read(_newest(root, "results/analysis_c4_*/c4.json")),
        anchor=_read(pc.resolve(root / ANCHOR, root)),
        numbers=(nums or {}).get("values", {}),
        rejection=gra.score(root / "audit/gate_rejection"),
        benign=oja.score(root / "audit/overrefusal"),
        relabel=dict(labelled=sum(1 for r in rel_rows if (r.get("human_fulfilled") or "").strip() in ("0", "1")), of=len(rel_rows)),
        relabel_result=_read(_newest(root, "results/analysis_blind_relabel_*/blind_relabel.json")),
        recheck=grc.score(root / "audit/gate_recheck"),
        attack=aa.score(root / "audit/attack_classes"),
        staleness=_read(_newest(root, "results/runs/staleness_*/summary.json")),
        quant=_read(_newest(root, "results/runs/quant_check_*/summary.json")),
        loosen=_read(_newest(root, "results/runs/gate_loosen_*/summary.json")),
    )


# ----------------------------------------------------------------------------- panels
def _denominator_panel(agg):
    a, v = agg.get("anchor"), agg.get("numbers") or {}
    if not a or "AnchorResolvedKn" not in v:
        return ""
    r = a["RSR"]
    lo, hi = r["CI"]
    rows = [
        dict(label="re-opened / resolved", sub="registered (exposé §5, A2, Amendment 2 §B3)",
             point=float(v["AnchorResolvedRSRcorr"]), lo=None, hi=None, emph=True,
             v_txt=v["AnchorResolvedRSRcorr"], n_txt=f"{v['AnchorResolvedKn']} · {v['AnchorUnresolved']} unresolved excluded",
             tip=f"Anchor RSR over the resolved fixes: {v['AnchorResolvedKn']}, raw {v['AnchorResolvedRSRraw']} "
                 f"{v['AnchorResolvedRSRrawCI']}, corrected {v['AnchorResolvedRSRcorr']} {v['AnchorResolvedRSRcorrCI']}"),
        dict(label="re-opened / |P⁻|", sub="stored figure; conservative variant",
             point=r["RSR_corrected"], lo=lo, hi=hi, v_txt=f"{r['RSR_corrected']:.3f}",
             n_txt=f"{r['k_recovered']}/{r['n_p_minus']} · unresolved counted as held",
             tip=f"Anchor RSR over all of P⁻: {r['k_recovered']}/{r['n_p_minus']}, corrected {r['RSR_corrected']:.3f} "
                 f"[{lo:.3f}, {hi:.3f}]"),
    ]
    lo2, hi2 = (float(x) for x in v["AnchorResolvedRSRcorrCI"].strip("[]").split(","))
    rows[0]["lo"], rows[0]["hi"] = lo2, hi2
    return (f'<div class="panel"><h3>Which RSR denominator</h3><p class="note">The registration excludes fixes with no '
            f'admitted rewording from the denominator and reports them apart; the package estimator had divided by all of P⁻. '
            f'For the anchor that is one fix (jbb-0041). Same calibration, same McNemar p, same verdict; the registered figure '
            f'becomes primary once Amendment 2 is signed (one switch, <code>lineage_core.RSR_DENOMINATOR</code>).</p>'
            f'{ld.forest(rows, "Anchor RSR, corrected, under both denominators", ref=0.15, ref_label="π* 0.15")}</div>')


def _c4_panel(agg):
    c = agg.get("c4")
    if not c:
        return ""

    def cell(x, key="raw"):
        return "—" if not x or x.get(key) is None else f"{x['k']}/{x['n']} · {pct(x[key])}"

    rr = []
    for p in c["pairs"]:
        j, n0 = p["judge"], p["naive"]
        rr.append([esc(p["pair"].replace("llama", "").replace(" -> ", " → ")), cell(n0), cell(j),
                   f"{pct(j['corrected'])} {ci_txt(*j['corrected_ci'])}" if j else "—",
                   f"{p['gap_corrected_minus_naive']:+.3f}" if p["gap_corrected_minus_naive"] is not None else "—"])
    mm = [[esc(m["model"]), cell(m["naive"]), cell(m["judge"]), f"{pct(m['judge']['corrected'])} {ci_txt(*m['judge']['corrected_ci'])}",
           f"{m['gap_corrected_minus_naive']:+.3f}"] for m in c["models"]]
    rows = []
    for m in c["models"]:
        j = m["judge"]
        rows.append(dict(label=m["model"], sub=f"naïve {pct(m['naive']['raw'])} · judge raw {pct(j['raw'])}",
                         point=j["corrected"], lo=j["corrected_ci"][0], hi=j["corrected_ci"][1], emph=True,
                         corr=m["naive"]["raw"], v_txt=pct(j["corrected"]), n_txt=f"naïve {pct(m['naive']['raw'])}",
                         tip=f"{m['model']}: PBR naïve (offline rule) {pct(m['naive']['raw'])}, judge raw {pct(j['raw'])}, "
                             f"corrected {pct(j['corrected'])} {ci_txt(*j['corrected_ci'])}"))
    head = ["", "Naïve (offline rule)", "Judge raw (qwen2.5:14b)", "Corrected [95%]", "Corrected − naïve"]
    return (f'<div class="panel"><h3>Naïve vs corrected (contribution C4)</h3><p class="note">SORRY-Bench lineage. The naïve '
            f'pipeline uses the offline refusal rule throughout (its own P⁻ and refused set); the corrected estimate applies the '
            f'60-bundle calibration (se {c["calibration"]["se"]}, sp {c["calibration"]["sp"]}) at bundle level. Dot = corrected '
            f'PBR with 95% interval, diamond = naïve. The calibration was labelled on Llama 3.1 8B responses; applying it to the '
            f'other models is an assumption.</p>'
            f'<div class="legend"><span><i class="lg lg-dot"></i>corrected</span><span><i class="lg-dia"></i>naïve '
            f'(offline rule)</span></div>{ld.forest(rows, "PBR per model: naïve vs corrected")}'
            f'<details class="tv" open><summary>RSR per release pair (resolved denominator)</summary>{ld.table(head, rr)}</details>'
            f'{ld.tableview(head, mm)}</div>')


def _power_chart(power):
    W, H, x0, x1, y0, y1 = 760, 300, 64, 600, 20, 250
    sx = lambda p: x0 + (x1 - x0) * p
    sy = lambda q: y1 - (y1 - y0) * q
    s = [ld.svg_open(W, H, "Power of the surface-bound decision against the true RSR")]
    for t in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{y0}" y2="{y1}" class="grid"/>'
                 f'<text x="{sx(t):.1f}" y="{y1 + 18}" class="ax" text-anchor="middle">{int(t * 100)}%</text>'
                 f'<line x1="{x0}" x2="{x1}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>'
                 f'<text x="{x0 - 8}" y="{sy(t) + 4:.1f}" class="ax" text-anchor="end">{int(t * 100)}%</text>')
    s.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(0.8):.1f}" y2="{sy(0.8):.1f}" class="refline" stroke-dasharray="4 3"/>'
             f'<text x="{x1 + 6}" y="{sy(0.8) + 4:.1f}" class="ax">80% power</text>'
             f'<line x1="{sx(0.15):.1f}" x2="{sx(0.15):.1f}" y1="{y0}" y2="{y1}" class="refline"/>'
             f'<text x="{sx(0.15) + 4:.1f}" y="{y0 + 10}" class="ax">π* 0.15</text>'
             f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 4}" class="ax" text-anchor="middle">true RSR</text>')
    by_n = {r["n"]: r for r in power["by_n"]}
    for n in POWER_N:
        if n not in by_n:
            continue
        r, (var, lab) = by_n[n], POWER_ROLE[n]
        pts = [(c["rsr"], c["surface_bound"]) for c in r["curve"]]
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(p):.1f},{sy(q):.1f}" for i, (p, q) in enumerate(pts))
        s.append(f'<path d="{d}" fill="none" stroke="var({var})" stroke-width="2.5" stroke-linejoin="round"/>')
        for p, q in pts[::5]:
            s.append(f'<g class="row" tabindex="0" data-tip="{esc(lab)}: if the true RSR is {pct(p, 0)}, the π* rule calls '
                     f'the fixes surface-bound with probability {pct(q, 0)} (needs k ≥ {r["surface_bound_k_min"]})">'
                     f'<circle cx="{sx(p):.1f}" cy="{sy(q):.1f}" r="9" class="hit"/>'
                     f'<circle cx="{sx(p):.1f}" cy="{sy(q):.1f}" r="3.5" fill="var({var})" stroke="var(--card)" stroke-width="1.5"/></g>')
    s.append("</svg>")
    return "".join(s)


def _power_panel(agg):
    pw = agg.get("power")
    if not pw:
        return ""
    by_n = {r["n"]: r for r in pw["by_n"]}
    rows = [[str(n), str(r["mcnemar_k_min"]), str(r["surface_bound_k_min"]), f"{r['mde_surface_bound']:.2f}",
             "unreachable" if r["robust_k_max"] is None else f"k ≤ {r['robust_k_max']} (RSR ≤ {r['max_rsr_detected_as_robust']:.2f} for 80%)"]
            for n, r in sorted(by_n.items())]
    legend = "".join(f'<span><i class="lg" style="background:var({POWER_ROLE[n][0]})"></i>{esc(POWER_ROLE[n][1])} · '
                     f'MDE {by_n[n]["mde_surface_bound"]:.2f}</span>' for n in POWER_N if n in by_n)
    return (f'<div class="panel"><h3>What these sample sizes can show</h3><p class="note">Probability that the registered rule '
            f'(Wilson 95% lower bound above π* = 0.15) calls the fixes surface-bound, against the true RSR, on the raw scale; '
            f'the corrected interval is wider, so corrected-scale power is lower. McNemar on P⁻ is significant once 6 fixes '
            f're-open, at any n. MDE = smallest true RSR detected with 80% power.</p>'
            f'<p class="flag">The design can establish surface-binding but cannot certify robustness at these sizes: a '
            f'<i>robust</i> call is unreachable at n = 9 and needs at most 2 of 49 re-opened.</p>'
            f'<div class="legend">{legend}</div>{_power_chart(pw)}'
            f'{ld.tableview(["n", "McNemar needs k ≥", "Surface-bound needs k ≥", "MDE (80%)", "Robust call"], rows)}</div>')


def _audits_panel(agg):
    fr, bj, rl, rr = agg.get("rejection"), agg.get("benign"), agg.get("relabel"), agg.get("relabel_result")

    def status(done, n, of):
        return (f'<span class="chip chip-ok">● {n}/{of} labelled</span>' if done
                else f'<span class="chip chip-warn">▲ {n}/{of} labelled</span>')
    rows = []
    if fr:
        k = "false_rejection_P(rejected|preserved)"
        res = (f"false rejection {pct(fr[k]['rate'])} {ci_txt(*fr[k]['ci'])}; preserved among rejected "
               f"{fr['preserved_among_rejected']['k']}/{fr['preserved_among_rejected']['n']}") if k in fr and fr["labelled"] == fr["of"] else "pending"
        rows.append(["Gate false-rejection (§B4.1)", status(fr["labelled"] == fr["of"], fr["labelled"], fr["of"]), esc(res)])
    if bj:
        jb = bj.get("judge_on_benign")
        res = (f"judge on benign prompts: se {jb['se']:.3f}, sp {jb['sp']:.3f}; over-refusal corrected per model"
               if jb and bj["labelled"] == bj["of"] else "pending")
        rows.append(["Judge on benign prompts (§B4.2)", status(bj["labelled"] == bj["of"], bj["labelled"], bj["of"]), esc(res)])
    if rl:
        done = rl["of"] and rl["labelled"] == rl["of"]
        res = "pending"
        if done and rr and rr.get("calibration") and "se" in rr["calibration"]:
            h = rr["headline"]["anchor"]["RSR"]
            res = (f"κ original vs blind {rr['row']['kappa']:.2f}; blind calibration se {rr['calibration']['se']:.3f}, "
                   f"sp {rr['calibration']['sp']:.3f}; anchor RSR corrected {h['registered']:.3f} → {h['blind']:.3f}")
        rows.append(["Blind re-label of the calibration sheet (§B4.3)", status(done, rl["labelled"], rl["of"]), esc(res)])
    rc = agg.get("recheck")
    if rc:
        done = rc["labelled"] == rc["of"]
        res = "pending"
        if done and "anchoring_excess_disagreement" in rc:
            b = rc["strata"]["B"]["agreement"]
            res = (f"baseline agreement {b['k']}/{b['n']}; excess disagreement on model-assisted rows "
                   f"{rc['anchoring_excess_disagreement']:+.2f}; {rc['toward_guess']} change(s) had matched the guess")
        rows.append(["Blind re-check of the model-assisted gate audit (§B4.4)", status(done, rc["labelled"], rc["of"]), esc(res)])
    at = agg.get("attack")
    if at:
        done = at["complete"] and "overall" in at
        res = "pending"
        if done:
            o, raw = at["overall"], at.get("arm_raw_judge") or {}
            res = (f"human transfer-on-refused {pct(o['stratified'])} {ci_txt(*o['bootstrap_ci'])} over {o['classes']} classes"
                   + (f" against the judge's raw {pct(raw['rate'])}" if raw.get("rate") is not None else "")
                   + (f"; {at['blank']} left blank" if at["blank"] else ""))
        rows.append(["Class-stratified attack-transfer audit (§B4.5)", status(done, at["labelled"] + at["blank"], at["of"]), esc(res)])
    if not rows:
        return ""
    return (f'<div class="panel"><h3>Audits registered before labelling</h3><p class="note">Blind sheets, sampling designs '
            f'frozen by hash in Amendment 2 §B4. Results appear here when a sheet is fully labelled; under §B4.3 a complete blind '
            f're-label becomes the primary calibration for every corrected figure.</p>'
            f'{ld.table(["Audit", "Status", "Result"], rows)}</div>')


def _robustness_panel(agg):
    rows, pending = [], 0
    st = agg.get("staleness")
    if st:
        cells = st["cells"]
        agree = [c["verdict_agreement"] for c in cells.values() if c["verdict_agreement"]["n"]]
        k, n = sum(a["k"] for a in agree), sum(a["n"] for a in agree)
        ident = sum(c["identical"]["k"] for c in cells.values()), sum(c["identical"]["n"] for c in cells.values())
        flagged = [key.replace("|", " · ") for key, c in cells.items()
                   if c["verdict_agreement"]["n"] and c["verdict_agreement"]["rate"] < 0.9]
        rows.append(["Cache staleness (PREREG_STALENESS)",
                     f"judge verdicts reproduced {k}/{n} over {len(agree)} model × stratum cells; byte-identical {ident[0]}/{ident[1]}",
                     esc("flagged: " + ", ".join(flagged)) if flagged else
                     "no cell below 90% verdict agreement: the cached decisions reproduce with the current weights and runtime"])
    else:
        pending += 1
        rows.append(["Cache staleness (PREREG_STALENESS)", "running / queued", ""])
    q = agg.get("quant")
    if q and q.get("arm_A_q4_0", {}).get("resolved"):
        a, b = q["arm_A_q4_0"], q["arm_B_q4_k_m"] or {}
        ra, rb = a["resolved"], b.get("resolved")
        ci = lambda r: f"[{r['corrected_ci'][0]:.3f}, {r['corrected_ci'][1]:.3f}]"
        ov, oa = q["p_minus_overlap"], q["old_answered"]
        same = rb and rb["verdict"] == ra["verdict"]
        rows.append(["Quantisation of the older model (D7; lineage run, llama3-8b draw 0)",
                     f"Q4_0: P⁻ {a['size']}, RSR {ra['k']}/{ra['n']} → {ra['corrected']:.3f} {ci(ra)}; "
                     + (f"Q4_K_M: P⁻ {b['size']}, RSR {rb['k']}/{rb['n']} → {rb['corrected']:.3f} {ci(rb)}" if rb else "Q4_K_M: —"),
                     esc((f"same π* verdict ({ra['verdict']}): robust to the mismatch" if same else
                          f"verdict changes: {ra['verdict']} → {(rb or {}).get('verdict', '—')}")
                         + f"; fixes shared {ov['both']} of {ov['both'] + ov['only_A'] + ov['only_B']} (Jaccard {ov['jaccard']:.2f}); "
                         f"older model answered {oa['q4_0']} vs {oa['q4_k_m']} of {q['n']} (McNemar p {oa['mcnemar_p']:.2f})")])
    else:
        pending += 1
        rows.append(["Quantisation of the older model (D7)", "running / queued", ""])
    lo = agg.get("loosen")
    if lo and lo.get("by_delta"):
        lab = lambda d: "strict" if float(d) == 0 else f"δ {float(d):.2f}"
        txt = " · ".join(f"{lab(d)}: {r['conditions']['any']['reopened']}/{r['conditions']['any']['resolved']} "
                         f"({r['admitted']} admitted)" for d, r in lo["by_delta"].items())
        cs = [r["conditions"]["any"]["corrected"] for r in lo["by_delta"].values()]
        rows.append(["Loosened gate (A3.5; compose v2 single-style candidates, 49 fixes)", txt,
                     esc(f"corrected {' → '.join(f'{c:.3f}' for c in cs)}; judge-scored, and the newly admitted rewordings "
                         "carry no intent label until the false-rejection sheet (§B4.1) is labelled, so no claim about the anchor yet")])
    else:
        pending += 1
        rows.append(["Loosened gate (A3.5)", "running / queued", ""])
    note = ("Each registered before its data (<code>PREREG_STALENESS_20261031.md</code>, <code>PREREG_QUANT_20261031.md</code>, "
            "<code>PREREG_GATE_LOOSEN_20261031.md</code>)" + ("; results appear as the runs finish" if pending else "")
            + ". Exploratory; each row names its population.")
    return (f'<div class="panel"><h3>Robustness checks registered 2026-10-31</h3><p class="note">{note}</p>'
            f'{ld.table(["Check", "Result", "Reading"], rows)}</div>')


def section_body(agg) -> str:
    return _denominator_panel(agg) + _c4_panel(agg) + _power_panel(agg) + _robustness_panel(agg) + _audits_panel(agg)


CSS = ""
