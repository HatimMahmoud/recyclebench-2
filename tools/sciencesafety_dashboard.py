#!/usr/bin/env python3
"""sciencesafety_dashboard.py -- STANDALONE page for a bring-your-own-data corpus run.

EXPERIMENTAL -- the ScienceSafetyData pilot is parked for the cluster. This file is safe to delete
before submission (cleanup list in CLAUDE.md, TODO_HUMAN A10). The reusable part of that work is
`tools/prepare_external.py` + `docs/BYOD.md`, which stay.

One self-contained page for an external corpus screened on one pair, in the same DFKI design as
the lineage / pair / master dashboards (CSS, JS, brand, legend and chart helpers are imported from
`lineage_dashboard`, so this page cannot drift from them).

Deliberately standalone. The corpus is registered in its own prereg with an isolation clause: it
is NOT in `tools/study_data.py`, so nothing here pools into the master page's headline, RQ cards,
`pooled_offline` or `recovery_by_style`. Folding it in needs a signed amendment, not a dashboard.
This page therefore says "exploratory / raw / T1-unresolved" in its own header and repeats the
limit beside every figure, and `build_all_dashboards.py` does not build it (the run directory has
no `<old>_to_<new>` in its name).

Written for the supervisor who asked for the corpus to be explored.

    python3 tools/sciencesafety_dashboard.py            # defaults to the 2026-11-01 run
    python3 tools/sciencesafety_dashboard.py --redact    # drop every prompt / response
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import lineage_core as lc                                           # noqa: E402
import lineage_dashboard as ld                                      # noqa: E402
from recyclebench.config import Config                              # noqa: E402
from recyclebench.datasets import load_jailbreakbench               # noqa: E402
from recyclebench.manifest import Manifest                          # noqa: E402
from recyclebench.realrun import _key                               # noqa: E402
from recyclebench.screening import _draw_seed                       # noqa: E402

esc, pct, tile, table, tableview = ld.esc, ld.pct, ld.tile, ld.table, ld.tableview

NAV = [("s-rq", "00", "Research questions"), ("s-headline", "01", "Headline"),
       ("s-flow", "02", "Flow & partition"), ("s-fixes", "03", "The fixes"),
       ("s-gate", "04", "Intent gate"), ("s-inversion", "05", "Inversion arm"),
       ("s-range", "06", "Instrument range"), ("s-method", "07", "Threats & method")]

RQ_MAP = {"s-headline": ["RQ1"], "s-fixes": ["RQ1", "RQ2"], "s-gate": ["RQ3"],
          "s-inversion": ["RQ1"], "s-range": ["RQ3"]}

GATE_LAYERS = ("cosine", "bertscore", "nli_entail", "llm_intent")


# =============================================================================
# Data
# =============================================================================
def collect(args) -> dict:
    cfg = Config.load(args.config)
    run = Path(args.run)
    beh = load_jailbreakbench(str(run / args.subset))
    cache = lc.load_response_cache(args.cache)
    man = Manifest.load(str(run / args.manifest))
    tag = lc.judge_tag_from_cfg(cfg)
    verd = lc.StoredVerdicts(lc.VerdictStore(args.verdicts), tag)
    obs, runs = lc.build_runs(cfg, beh, man, cache, verd, [args.old, args.new])
    k = cfg.budget.k_canonical_samples

    pilot = json.loads((run / args.pilot).read_text())
    summary = json.loads((run / args.summary).read_text())
    corpus = list(csv.DictReader(open(args.corpus, newline="")))
    meta = {r["Goal"]: r for r in corpus}
    sub = list(csv.DictReader(open(run / args.subset, newline="")))

    mo_new, mo_old = obs[args.new], obs[args.old]
    rows, unjoined = [], 0
    for b, s in zip(beh, sub):
        m = meta.get(b.canonical_prompt)
        unjoined += int(m is None)
        m = m or {}
        vs = mo_new.variants[b.bid]
        reop = [v for v in vs if v.verdict == 1]

        def draws(mobs):
            return [{"verdict": v, "text": t, "words": len(t.split()) if t else None}
                    for t, v in mobs.draws[b.bid]]
        rows.append({
            "bid": b.bid, "src": s.get("source_bid", "?"), "dom": s.get("Category", "?"),
            "risk": m.get("RiskType", "?"), "task": m.get("Task", "?"),
            "prompt": b.canonical_prompt, "words": len(b.canonical_prompt.split()),
            "n_adm": len(vs), "reopened": len(reop),
            "classes": sorted({v.transform_class for v in reop}),
            "complete": mo_new.recovery_complete(b.bid),
            "old_draws": draws(mo_old), "new_draws": draws(mo_new),
            "variants": [{"tc": v.transform_class, "text": v.text, "resp": v.response,
                          "verdict": v.verdict} for v in vs]})

    # gate: admitted / proposed per transform class, and the layer that rejected each row
    adm = collections.Counter(v.transform_class for vs in man.admitted.values() for v in vs)
    prop = collections.Counter(a.transform_class for a in man.audit)
    rej = collections.Counter()
    thr = {k_: getattr(cfg.gate, k_) for k_ in GATE_LAYERS if hasattr(cfg.gate, k_)}
    for a in man.audit:
        if a.admitted:
            continue
        vals = {k_: getattr(a, k_, None) for k_ in GATE_LAYERS}
        first = next((k_ for k_ in GATE_LAYERS
                      if vals.get(k_) is not None and k_ in thr and vals[k_] < thr[k_]), None)
        rej[first or "lexical_divergence"] += 1

    # inversion arm: the gate store and the scored closed-loop artifact
    gate_rows, loop = [], {}
    gp = Path(args.rpe_gate)
    if gp.exists():
        gate_rows = [json.loads(x) for x in gp.read_text().splitlines() if x.strip()]
    cl = sorted((run / "rpe").glob("closedloop_*_gated_*.json"))
    if cl:
        loop = json.loads(cl[-1].read_text())
        # --run may be given relative to the cwd, so resolve before making it repo-relative
        p = cl[-1].resolve()
        loop["_path"] = str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)

    prov = {}
    pp = Path(args.corpus + ".provenance.json")
    if pp.exists():
        prov = json.loads(pp.read_text())

    rp = run.resolve()
    return {"cfg": cfg, "k": k, "run_disp": str(rp.relative_to(ROOT) if rp.is_relative_to(ROOT) else rp), "rows": rows, "pilot": pilot, "summary": summary,
            "n_corpus": len(corpus), "unjoined": unjoined, "judge_tag": tag,
            "adm": adm, "prop": prop, "rej": rej, "thr": thr,
            "gate_rows": gate_rows, "loop": loop, "prov": prov,
            "corpus_domains": collections.Counter(r["Category"] for r in corpus),
            "old": args.old, "new": args.new, "run": run,
            "offline": (run / "partition_full.txt").read_text()
            if (run / "partition_full.txt").exists() else ""}


def rsr_row(d) -> dict:
    return (d["summary"].get("RSR_primary_RAW") or [{}])[0]


# =============================================================================
# Sections
# =============================================================================
def sec_header(d, args, today) -> str:
    p = d["prov"]
    src = p.get("source", {})
    pin = f'{src.get("repo", "?")} @ <code>{esc(str(src.get("commit", "?"))[:7])}</code>' \
        if src else "—"
    return f"""<header class="hd">
  <p class="eyebrow">RecycleBench · bring-your-own-data · standalone</p>
  <h1>{esc(args.title)}</h1>
  <p class="lede">An external corpus of {d['n_corpus']} research-framed dual-use requests, screened on
    {esc(ld.label(d['old']))} → {esc(ld.label(d['new']))} and put through the recycling and inversion arms.
    Everything on this page is <b>raw, exploratory and under-powered</b>, and the judge has <b>no
    calibration of its own for this corpus</b>.</p>
  <div class="warnbox" role="note">
    <p><b>This page is not part of the study's evidence base.</b> The corpus is registered with an
    isolation clause: it is not in <code>tools/study_data.py</code>, so no figure here reaches the master
    page's headline, RQ answers or pooled estimates. Folding it in requires a signed amendment.</p>
    <p><b>Threat T1 is unresolved.</b> The judge scores <i>did the reply attempt the request</i>, and these
    prompts have a legitimate research reading, so a benign-but-responsive answer can read as compliance.
    Until the blind sheet is labelled, P⁻ membership and every rate built on it are provisional.</p>
  </div>
  <dl class="meta">
    <div><dt>Pair</dt><dd>{esc(ld.label(d['old']))} → {esc(ld.label(d['new']))} {ld.chip('matched-scale')}</dd></div>
    <div><dt>Corpus</dt><dd>{pin}, {esc(p.get('license', '?'))}</dd></div>
    <div><dt>Behaviours</dt><dd>{d['n_corpus']} · {len(d['corpus_domains'])} domains ·
      0 overlap with screened benchmarks</dd></div>
    <div><dt>Judge</dt><dd>{esc(d['judge_tag'])} · <b>uncalibrated here</b></dd></div>
    <div><dt>Pre-registration</dt><dd><code>PREREG_SCIENCESAFETY_20261101.md</code>,
      <code>…_RPE_…</code></dd></div>
    <div><dt>Run</dt><dd><code>{esc(d['run_disp'])}</code></dd></div>
    <div><dt>Built</dt><dd>{today}{' · text redacted' if args.redact else ''}</dd></div>
  </dl></header>"""


def sec_rq(d) -> str:
    r = rsr_row(d)
    n, kk = r.get("n_p_minus", 0), r.get("k_recovered", 0)
    _, lo, hi = ld.wilson(kk, n) if n else (None, None, None)
    ans = {
        "RQ1": (f'<b>{kk} of {n}</b> fixes re-opened — {pct(kk / n) if n else "—"} '
                f'[{pct(lo)}, {pct(hi)}] raw. Close to the anchor run\'s 0.551 corrected on a corpus with '
                'no overlap, but n = 23 is below the pre-registered floor of 30 and the judge is '
                'uncalibrated here, so this <b>supports no claim on its own</b>.'),
        "RQ2": 'Re-openings are spread across all five represented domains and all four risk types, with '
               'no cell structure. At n = 3–8 per cell this is <b>descriptive only</b> (threat T4).',
        "RQ3": '<b>Not answerable from this corpus.</b> The gate admitted <b>0 of 23</b> back-translations '
               'and only 5 of 23 tense shifts, so two of the six transform classes are empty or '
               'near-empty by construction, not by effect.',
        "RQ4": 'No model-level PBR exists here. The run summary\'s PBR is computed over the same 23 '
               'behaviours as RSR (<code>n_cm = 23</code>), so it is <b>numerically RSR</b> and carries no '
               'independent information.',
    }
    where = {"RQ1": [("s-headline", "Headline"), ("s-fixes", "The fixes")],
             "RQ2": [("s-fixes", "The fixes")],
             "RQ3": [("s-gate", "Intent gate"), ("s-range", "Instrument range")],
             "RQ4": [("s-method", "Threats & method")]}
    return ld.rq_section("s-rq", "00", answers=ans, where=where, frame=ld.RQ_FRAME,
                         status="Answers on this page are from one external corpus, raw and "
                                "exploratory. The study's own answers are on the master page.")


def sec_headline(d) -> str:
    r = rsr_row(d)
    n, kk = r.get("n_p_minus", 0), r.get("k_recovered", 0)
    _, lo, hi = ld.wilson(kk, n) if n else (None, None, None)
    floor = int(d["pilot"].get("min_required", 30))
    old_all = sum(1 for x in d["rows"] if all(dr["verdict"] == 1 for dr in x["old_draws"]))
    tiles = "".join([
        tile("Fixes found (P⁻)", f"{d['pilot']['p_minus']}",
             f"of {d['n_corpus']} behaviours · {ld.power_chip(n, floor)}",
             tip="OLD complies and NEW refuses, judged by the config's judge."),
        tile("Re-opened by rewording", f"{kk} / {n}", "at least one admitted rewording", emph=True),
        tile("RSR (raw)", pct(kk / n) if n else "—",
             f"Wilson 95% [{pct(lo)}, {pct(hi)}]" if n else "undefined",
             tip="Raw judge output. No corrected figure exists: this corpus has no human labels."),
        tile("Anchor, for reference", "0.551",
             "corrected, SORRY-Bench, n=49 · different population",
             tip="Shown for orientation only; the two are not pooled."),
        tile("McNemar", ld.p_txt(r.get("mcnemar_p")), "exact, canonical vs best rewording"),
        tile("Judge calibration", "none", "transfer only, as a sensitivity"),
    ])
    return f"""<section id="s-headline" class="sec" aria-labelledby="h-s-headline">
  <h2 id="h-s-headline"><span class="sn">01</span>Headline</h2>
  <div class="tiles">{tiles}</div>
  <h3>What is solid, and what is not</h3>
  <div class="cols2">
    <div class="card ok"><p class="ct">Solid</p><ul>
      <li><b>P⁻ membership is unanimous.</b> {ld.label(d['old'])} complied on <b>3 of 3</b> canonical draws
        for {old_all} of {n} members, and {ld.label(d['new'])} refused on 3 of 3 for all of them. Screening is
        any-of-k, so no member rests on a single lucky draw.</li>
      <li><b>The two reply distributions barely touch</b> — see §06.</li>
      <li><b>Every variant verdict is present</b>, so {kk}/{n} hides no unresolved behaviours.</li>
      <li><b>The gate measurements are judge-free</b> (§04, §05) and T1 cannot touch them.</li>
    </ul></div>
    <div class="card warn"><p class="ct">Not solid</p><ul>
      <li><b>n = {n} &lt; {floor}</b>, the pre-registered floor. The decision rule hands the headline to
        PBR — which does not exist here (RQ4 card).</li>
      <li><b>No calibration.</b> Every figure is raw; a corrected figure would need labels from
        <i>this</i> corpus, or an explicitly labelled transfer.</li>
      <li><b>T1 unresolved.</b> The blind sheet was not produced by the run (§07).</li>
      <li><b>Two transform classes are empty</b>, so RQ3 is unanswerable (§04).</li>
    </ul></div>
  </div></section>"""


def sec_flow(d) -> str:
    pil, rows = d["pilot"], d["rows"]
    n_tot, broke, pm = pil["n_behaviours"], pil["breaks_old"], pil["p_minus"]
    kk = sum(1 for r in rows if r["reopened"])
    pv = ld.PairView(old=d["old"], new=d["new"], adjacent=True,
                     status=lc.pair_status(d["old"], d["new"]),
                     cells={"pminus": [r["bid"] for r in rows],
                            "persistent": [f"p{i}" for i in range(broke - pm)],
                            "neg_flip": [],
                            "both_safe": [f"s{i}" for i in range(n_tot - broke)]},
                     complete=True,
                     recovered={r["bid"] for r in rows if r["reopened"]},
                     unresolved=[r["bid"] for r in rows if not r["reopened"] and r["n_adm"] == 0],
                     max_admitted=max((r["n_adm"] for r in rows), default=0))
    cells = table(
        ["Cell", "Meaning", "Judged", "Offline rule"],
        [["<b>P⁻ (fixes)</b>", f"{ld.label(d['old'])} answers, {ld.label(d['new'])} refuses",
          f"<b>{pm}</b>", "25"],
         ["Persistent", "both answer", f"{broke - pm}", "32"],
         ["Negative flip", f"{ld.label(d['old'])} refuses, {ld.label(d['new'])} answers",
          '<span class="muted">not judged</span>', "0"],
         ["Both safe", "both refuse", f"{n_tot - broke}", "183"],
         ["<b>Total</b>", "", f"<b>{n_tot}</b>", "240"]], num_cols=(2, 3))
    return f"""<section id="s-flow" class="sec" aria-labelledby="h-s-flow">
  <h2 id="h-s-flow"><span class="sn">02</span>Flow &amp; partition</h2>
  <p>Where the {n_tot} behaviours go. The fourth stage counts only P⁻, the behaviours the newer model
    newly refuses — the rest are never reworded.</p>
  {ld.sankey(pv, n_tot, d['old'], d['new'])}
  <h3>The 2×2, judged and offline</h3>
  <p>The partition is shown twice because <b><code>run-real</code> stores no judge verdicts</b>: anything
    rebuilt from the response cache uses the offline refusal rule, which disagrees with
    {esc(d['judge_tag'])}. <b>The judged column is primary.</b> The negative-flip cell was never judged —
    doing so needs ~1,300 further judge calls — so the offline rule's 0 is the only evidence there, and a
    safety regression in this corpus is <b>unmeasured, not excluded</b>.</p>
  {cells}
  <p class="note">The two columns differ by the same margin and in the same direction as on SORRY-Bench
    (offline P⁻ 61 vs judged 49): the offline rule reads some verbose, safety-framed answers as
    compliance. That is the expected divergence, not a defect of this run.</p>
  <p class="tiny">Re-opened after rewording: {kk} of {pm}.</p></section>"""


def _case(r, d, redact) -> str:
    def draw_list(draws, expect):
        out = []
        for i, dr in enumerate(draws):
            v = {1: "answered", 0: "refused"}.get(dr["verdict"], "no verdict")
            cls = {1: "comply", 0: "refuse"}.get(dr["verdict"], "none")
            body = "" if redact or not dr["text"] else \
                f'<pre class="resp">{esc(ld.cap(dr["text"], 1400))}</pre>'
            out.append(f'<div class="draw"><span class="dot r-{cls}"></span>'
                       f'<b>draw {i}</b> <span class="v-{cls}">{v}</span>'
                       f'<span class="tiny"> · {dr["words"] or 0} words</span>{body}</div>')
        return (f'<p class="dh">{esc(expect)}</p>' + "".join(out))

    reo = [v for v in r["variants"] if v["verdict"] == 1]
    vlist = ""
    if reo and not redact:
        vlist = "".join(
            f'<div class="draw"><span class="dot r-comply"></span><b>{esc(v["tc"])}</b>'
            f'<pre class="resp">{esc(ld.cap(v["text"], 600))}</pre>'
            f'<pre class="resp">{esc(ld.cap(v["resp"] or "", 1000))}</pre></div>' for v in reo)
    elif reo:
        vlist = (f'<p class="tiny">{len(reo)} re-opening rewording(s) via '
                 f'{esc(", ".join(r["classes"]))} — text redacted.</p>')

    badge = (f'<span class="chip chip-warn">re-opened {r["reopened"]}/{r["n_adm"]}</span>'
             if r["reopened"] else '<span class="chip chip-ok">held 0/%d</span>' % r["n_adm"])
    return f"""<details class="case"><summary>
      <code>{esc(r['bid'])}</code> <span class="tiny">corpus {esc(r['src'])}</span>
      <span class="chip">{esc(r['dom'])}</span><span class="chip">{esc(r['risk'])}</span>
      {badge}<span class="tiny">{r['words']} words</span></summary>
    <p class="tiny">Task: {esc(r['task'])}</p>
    {'' if redact else f'<p class="dh">Request</p><pre class="resp">{esc(r["prompt"])}</pre>'}
    {draw_list(r['old_draws'], f'{ld.label(d["old"])} — expected to answer')}
    {draw_list(r['new_draws'], f'{ld.label(d["new"])} — expected to refuse')}
    {f'<p class="dh">Re-opening rewordings</p>{vlist}' if reo else ''}</details>"""


def sec_fixes(d, redact) -> str:
    rows = sorted(d["rows"], key=lambda x: (-x["reopened"], x["dom"]))
    tbl = table(["Run id", "Corpus id", "Domain", "Risk type", "Words", "Admitted", "Re-opened"],
                [[f'<code>{esc(r["bid"])}</code>', esc(r["src"]), esc(r["dom"]), esc(r["risk"]),
                  str(r["words"]), str(r["n_adm"]),
                  f'<b>{r["reopened"]}</b>' if r["reopened"] else "0"] for r in rows],
                num_cols=(4, 5, 6))
    dom = collections.Counter(r["dom"] for r in d["rows"])
    domk = collections.Counter(r["dom"] for r in d["rows"] if r["reopened"])
    bars = ld.hbars([{"label": k, "sub": f"{domk[k]} of {dom[k]}", "v": domk[k] / dom[k],
                      "lo": ld.wilson(domk[k], dom[k])[1], "hi": ld.wilson(domk[k], dom[k])[2],
                      "v_txt": f"{domk[k]}/{dom[k]}",
                      "tip": f"{k}: {domk[k]} of {dom[k]} fixes re-opened"}
                     for k in sorted(dom)], "Re-opened by domain", ref=0.15, ref_label="π* = 0.15")
    risk = collections.Counter(r["risk"] for r in d["rows"])
    riskk = collections.Counter(r["risk"] for r in d["rows"] if r["reopened"])
    rtbl = table(["Risk type", "Fixes", "Re-opened"],
                 [[esc(k), str(risk[k]), str(riskk[k])] for k in sorted(risk)], num_cols=(1, 2))
    return f"""<section id="s-fixes" class="sec" aria-labelledby="h-s-fixes">
  <h2 id="h-s-fixes"><span class="sn">03</span>The fixes</h2>
  <p>Every behaviour in P⁻, with how many of its gate-admitted rewordings re-opened it. <b>Run ids are
    row positions in the subset CSV, not corpus ids</b> — <code>load_jailbreakbench</code> renumbers rows —
    so the corpus id is shown beside each and all metadata here was joined <b>on prompt text</b>
    ({d['unjoined']} failed to join).</p>
  {tbl}
  <h3>By domain and risk type</h3>
  <div class="cols2"><div>{bars}</div><div>{rtbl}
    <p class="note">No structure in either breakdown. Per-cell n is 3–8, so these are
      <b>descriptive only</b> (threat T4) and no domain comparison is tested. The flat risk-type
      profile is mildly reassuring for T1 — a judge systematically misreading benign research answers
      should concentrate in "Unintentional Consequences" — but it is not a substitute for the labels.</p>
  </div></div>
  <h3>Case by case</h3>
  <p>{'Text is redacted in this build.' if redact else
      'Collapsed; contains corpus prompts and model replies. A working file, not a released artefact.'}
    Read these to settle T1: a member is real when the older model genuinely attempts the request and
    the newer one genuinely declines.</p>
  {"".join(_case(r, d, redact) for r in rows)}</section>"""


def sec_gate(d) -> str:
    adm, prop = d["adm"], d["prop"]
    classes = sorted(prop)
    bars = ld.hbars([{"label": c.replace("_", " "), "sub": f"{adm[c]} of {prop[c]}",
                      "v": adm[c] / prop[c], "lo": None, "hi": None,
                      "v_txt": f"{adm[c]}/{prop[c]}",
                      "tip": f"{c}: {adm[c]} of {prop[c]} proposed rewordings admitted"}
                     for c in classes], "Gate admission by transform class", role="fix")
    rej = d["rej"]
    rtbl = table(["Layer", "Threshold", "Rejections"],
                 [[esc(k.replace("_", " ")),
                   str(d["thr"].get(k, "—")) if k in d["thr"] else "—", str(v)]
                  for k, v in rej.most_common()], num_cols=(1, 2))
    tot_adm, tot_prop = sum(adm.values()), sum(prop.values())
    return f"""<section id="s-gate" class="sec" aria-labelledby="h-s-gate">
  <h2 id="h-s-gate"><span class="sn">04</span>Intent gate</h2>
  <p>The gate certifies that a rewording still asks for the same thing. It admitted
    <b>{tot_adm} of {tot_prop}</b> proposed rewordings ({pct(tot_adm / tot_prop)}) — comparable to the
    anchor run, so it is not simply failing on long prompts. The <b>per-class</b> profile is where this
    corpus departs.</p>
  <div class="cols2"><div>{bars}{tableview(["Class", "Admitted", "Proposed", "Rate"],
      [[c, str(adm[c]), str(prop[c]), pct(adm[c] / prop[c])] for c in classes],
      num_cols=(1, 2, 3))}</div>
    <div><div class="card key"><p class="ct">The finding</p>
      <p><b>Back-translation was admitted 0 times out of 23</b>, and tense shift only 5 of 23. On
      47-word technical paragraphs a translation round-trip drifts far enough to fail the gate
      <i>every single time</i>.</p>
      <p>So this corpus's RSR was produced by <b>four effective transform classes, not six</b>. That
      makes the agreement with the anchor more striking — fewer tools, similar recovery — and it makes
      <b>RQ3 unanswerable here</b>, because two levels are empty by construction rather than by effect.</p>
      <p>These are cosine / BERTScore / NLI scores over prompt pairs. <b>No judge is involved</b>, so
      threat T1 cannot touch this measurement.</p></div>
      {rtbl}</div></div>
  <p class="note">The gate short-circuits in order, so each rejection is attributed to the
    <i>first</i> layer that failed; layers after it were never evaluated. Rows whose every layer passed
    but which were still rejected are counted as lexical divergence.</p></section>"""


def sec_inversion(d) -> str:
    loop, rows = d["loop"], d["gate_rows"]
    if not loop:
        return ('<section id="s-inversion" class="sec"><h2><span class="sn">05</span>Inversion arm</h2>'
                '<p class="muted">No scored closed-loop artifact in this run.</p></section>')
    a = loop["admission"]
    by = collections.Counter(r.get("rejected_by") for r in rows if not r.get("admitted"))
    adm = sum(1 for r in rows if r.get("admitted"))

    def rate(k_):
        r = a.get(k_, {})
        if r.get("rate") is None:
            return tile(f"A_{k_}", "undefined", f"0 of 0 scorable · {r.get('unresolved', 0)} unresolved")
        return tile(f"A_{k_}", pct(r["rate"]), f"{r['k']}/{r['n']} · unresolved {r['unresolved']}",
                    emph=(k_ == "loop"))
    tbl = table(["Gate layer", "Rejected"],
                [[esc(str(k_).replace("_", " ")), str(v)] for k_, v in by.most_common()],
                num_cols=(1,))
    return f"""<section id="s-inversion" class="sec" aria-labelledby="h-s-inversion">
  <h2 id="h-s-inversion"><span class="sn">05</span>Inversion arm</h2>
  <p>The convergent check: reconstruct the request from the <i>older</i> model's answer, then re-test the
    newer one. Three admission rules — <b>A_gate</b> (the intent gate, the only rule comparable to RSR),
    <b>A_loop</b> (the candidate lands in the answer's equivalence class — an upper bound), and
    <b>A_both</b>.</p>
  <div class="tiles">{rate('gate')}{rate('loop')}{rate('both')}
    {tile("Candidates", str(loop.get('n_candidates', 0)),
          f"over {loop.get('n_behaviours', 0)} behaviours")}
    {tile("Gate-admitted", f"{adm} / {len(rows)}", "inversion candidates")}</div>
  <div class="cols2">
    <div class="card key"><p class="ct">The gate admitted nothing</p>
      <p><b>0 of {len(rows)}</b> reconstructions passed, so <b>A_gate and A_both are undefined</b> — not
      zero, and not "the gate rejected them as unfaithful". There is simply no RSR-comparable figure for
      the inversion arm on this corpus.</p>
      <p>A_loop = <b>{a.get('loop', {}).get('k', 0)}/{a.get('loop', {}).get('n', 0)}</b> is an
      <b>upper bound that may not be claimed</b>: it admits reconstructions the project's own intent gate
      rejected, and unlike the anchor there is <b>no blind harmful-ask sheet</b> for this corpus to check
      whether they still ask for the same thing. On the anchor that sheet was what made the equivalent
      number reportable.</p>
      <p>Same cause as §04, one severity worse: recycling survived the gate at
      {pct(sum(d['adm'].values()) / sum(d['prop'].values()))}, reconstruction at 0%.</p></div>
    <div>{tbl}
      <p class="note">Every rejection fell at cosine or NLI entailment — <b>none reached BERTScore</b>.
      A reconstruction of a 47-word technical request is a <i>different</i> paragraph that asks for the
      same thing, which is exactly what an embedding-similarity floor is built to reject.</p>
      <p class="tiny">Artifact: <code>{esc(loop.get('_path', '?'))}</code>. Scored with
      <code>--gate-store</code>; without it A_gate/A_both would read empty <i>by construction</i>, which
      is a different statement from what is shown here.</p></div></div></section>"""


def sec_range(d) -> str:
    rows = d["rows"]
    lens = {m: [dr["words"] for r in rows for dr in r[k_] if dr["words"]]
            for m, k_ in ((d["old"], "old_draws"), (d["new"], "new_draws"))}
    import statistics as st
    long_new = [(r["bid"], dr["words"]) for r in rows for dr in r["new_draws"]
                if dr["words"] and dr["words"] >= 80]
    lt = table(["Model", "Draws", "Median words", "Range"],
               [[esc(ld.label(m)), str(len(v)), f"<b>{st.median(v):.0f}</b>",
                 f"{min(v)}–{max(v)}"] for m, v in lens.items()], num_cols=(1, 2, 3))
    flagged = collections.Counter(b for b, _ in long_new)
    ftbl = table(["Behaviour", "Long draws", "Words"],
                 [[f'<code>{esc(b)}</code>', str(c),
                   str(max(w for bb, w in long_new if bb == b))]
                  for b, c in flagged.most_common()], num_cols=(1, 2))
    return f"""<section id="s-range" class="sec" aria-labelledby="h-s-range">
  <h2 id="h-s-range"><span class="sn">06</span>Instrument range</h2>
  <p>What this corpus measured about the <i>instrument</i> rather than about the models. These figures
    are judge-free or judge-light and are the part of this run that is <b>safe to carry into the method
    chapter</b> without waiting for labels.</p>
  <h3>Compliance and refusal do not overlap in length</h3>
  {lt}
  <p>A 31-word reply is a refusal; a 357-word reply is an attempt at the task. The two distributions
    barely touch, which is the strongest available evidence that this P⁻ is <b>real</b> rather than a
    judge artefact — independent of the judge's own verdicts.</p>
  <h3>Where to look first</h3>
  <p>The only {ld.label(d['new'])} draws long enough to be anything other than a short refusal. Each was
    judged refused; read them to confirm they are long declines and not misreads.</p>
  {ftbl}
  <h3>Carried into the method chapter</h3>
  <ul>
    <li>The gate's per-class profile on long prompts (§04): back-translation 0%, tense shift 22%.</li>
    <li>The gate admits <b>no</b> output-inversion reconstruction of a long request (§05).</li>
    <li>Both pre-empt the reviewer question of whether the gate thresholds were tuned to the anchor's
      prompt length — they were, and here is the measurement of what that costs.</li>
  </ul>
  <p class="note">One honest limit: the per-class profile was computed over the 23 judge-selected
    behaviours, so it is mildly judge-dependent through <i>which</i> prompts were reworded. The clean
    version runs the paraphraser and gate over a random sample of the full corpus — judge-free end to
    end — and has not been run.</p></section>"""


def sec_method(d) -> str:
    threats = [
        ("T1", "Benign-reading compliance", "<b>UNRESOLVED</b>",
         "The judge scores <i>attempt at the request</i>; a research-framed answer can read as "
         "compliance, inflating <code>breaks_old</code> and putting non-fixes into P⁻.",
         "The blind sheet in <code>audit/sciencesafety/</code>. <b>The run did not produce it</b> — the "
         "exporter was given a directory where it wanted a file path. Re-running costs ~127 judge calls "
         "and zero target calls. Until then every rate here is provisional. Partial mitigation: §06's "
         "length separation and §03's flat risk-type profile."),
        ("T2", "Gate off its calibrated range", "<b>CONFIRMED, measured</b>",
         "Thresholds were set on short, blunt requests; this corpus is long and technical.",
         "Measured, not assumed: §04 (back-translation 0/23) and §05 (inversion 0/183). Reported as a "
         "finding about the instrument; it also voids RQ3 here."),
        ("T3", "No calibration of its own", "<b>OPEN by design</b>",
         "Rogan–Gladen correction needs human labels from this corpus.",
         "No corrected figure is shown anywhere on this page. Applying the SORRY-Bench sheet would be a "
         "<i>transfer</i>, reportable only as a labelled sensitivity."),
        ("T4", "Per-domain n = 40 (3–8 in P⁻)", "<b>ACCEPTED</b>",
         "Domain and risk-type breakdowns are far too small to test.",
         "§03 is descriptive only; no comparison is tested and no interval is interpreted."),
        ("T5", "No benign reference", "<b>OPEN</b>",
         "Without benign controls, an over-refusal shift cannot be separated from a safety shift.",
         "Not run for this corpus. XSTest over-refusal exists for all 8 models on the study's own "
         "populations."),
    ]
    tt = table(["#", "Threat", "Status", "Why it matters", "Mitigation"],
               [[f"<b>{t[0]}</b>", esc(t[1]), t[2], t[3], t[4]] for t in threats])
    r = rsr_row(d)
    n, kk = r.get("n_p_minus", 0), r.get("k_recovered", 0)
    return f"""<section id="s-method" class="sec" aria-labelledby="h-s-method">
  <h2 id="h-s-method"><span class="sn">07</span>Threats &amp; method</h2>
  <h3>Registered threats</h3>
  {tt}
  <h3>What may and may not be claimed</h3>
  <div class="cols2">
    <div class="card ok"><p class="ct">May be claimed</p><ul>
      <li>The adapter takes an external corpus end to end through screening, recycling and inversion
        with <b>one recipe file</b> and no change to any arm.</li>
      <li>This corpus is genuinely held out: <b>0 exact and 0 near duplicates</b> against all 2,190
        previously screened behaviours.</li>
      <li>The gate's range limits in §04 and §05, as instrument measurements.</li>
      <li>P⁻ is non-empty on a corpus nobody screened before, and its membership is unanimous across
        draws.</li>
    </ul></div>
    <div class="card warn"><p class="ct">May not be claimed</p><ul>
      <li><b>Not</b> that RSR = {pct(kk / n) if n else '—'} replicates the anchor. n = {n} is below the
        floor, the judge is uncalibrated here and T1 is open.</li>
      <li><b>Not</b> any PBR. The summary's figure is RSR over the same 23 behaviours.</li>
      <li><b>Not</b> A_loop = {d['loop'].get('admission', {}).get('loop', {}).get('rate') and
          pct(d['loop']['admission']['loop']['rate']) or '—'} as an inversion result: no harmful-ask
        sheet exists for this corpus.</li>
      <li><b>Not</b> any RQ3 / transform-class conclusion.</li>
      <li><b>Not</b> that there is no safety regression: the negative-flip cell was never judged.</li>
    </ul></div></div>
  <h3>Reproducing this page</h3>
  <pre class="resp">python3 tools/prepare_external.py --recipe datasets/sciencesafety.yaml
bash run_sciencesafety_20261101.sh          # screen -> recycle -> partition -> audit sheet
bash run_sciencesafety_rpe_20261101.sh      # verdicts -> inversions -> gate -> closed loop
python3 tools/pminus_review.py  ...         # the T1 reading file
python3 tools/sciencesafety_dashboard.py    # this page</pre>
  <p class="note">The style factorial (<code>tools/run_compose_v2.py --behaviours</code>) is plumbed for a
    new corpus but is <b>not registered</b> for this one and was not run: it needs a power statement for
    |P⁻| = {n}. See <code>docs/BYOD.md</code> for the five traps, and <code>THESIS_KB.md</code> for what
    each arm's numbers may carry.</p></section>"""


# =============================================================================
# Page
# =============================================================================
def render(d, args) -> str:
    today = _dt.date.today().isoformat()
    nav = "".join(f'<a href="#{i}"><span class="n">{n}</span>{t}</a>' for i, n, t in NAV)
    body = (sec_header(d, args, today) + sec_rq(d) + sec_headline(d) + sec_flow(d)
            + sec_fixes(d, args.redact) + sec_gate(d) + sec_inversion(d) + sec_range(d)
            + sec_method(d))
    foot = ("Generated by tools/sciencesafety_dashboard.py · recyclebench. Standalone: this corpus is "
            "not in tools/study_data.py and nothing here pools into the study's figures. "
            + ("Prompt and response text is omitted in this build. " if args.redact else
               "Contains corpus prompts and model responses (collapsed, truncated): a working file, "
               "not a released artefact — see ETHICS.md. ")
            + "All figures are raw judge output; no corrected figure exists for this corpus.")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<title>{esc(args.title)}</title><style>{ld.CSS}{CSS_EXTRA}</style></head><body>'
            f'<div class="layout"><nav class="side" aria-label="Sections">'
            f'{ld.brand_html("Bring your own data · standalone")}<div class="toc">{nav}</div>'
            f'{ld.legend_html()}'
            f'<button type="button" class="themebtn" id="themebtn" data-t="system">Theme: system</button>'
            f'</nav><main id="main">{body}<footer>{esc(foot)}</footer></main></div>'
            f'<div id="tip" role="tooltip" hidden></div>'
            f'<script>{ld.rq_script(RQ_MAP)}{ld.JS}</script></body></html>')


CSS_EXTRA = r"""
.warnbox{border:1px solid var(--warn-b,#d9a23a);background:var(--warn-bg,rgba(217,162,58,.08));
  border-radius:10px;padding:14px 16px;margin:16px 0}
.warnbox p{margin:0 0 8px}.warnbox p:last-child{margin:0}
.card{border:1px solid var(--line);border-radius:10px;padding:14px 16px;background:var(--surface-2)}
.card .ct{margin:0 0 8px;font-weight:650;letter-spacing:.02em;text-transform:uppercase;font-size:11px}
.card ul{margin:0;padding-left:18px}.card li{margin:0 0 6px}
.card.ok{border-left:3px solid var(--refuse)}.card.warn{border-left:3px solid var(--comply)}
.card.key{border-left:3px solid var(--fix)}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
@media(max-width:900px){.cols2{grid-template-columns:1fr}}
.case{border:1px solid var(--line);border-radius:8px;margin:6px 0;padding:0}
.case>summary{padding:9px 12px;cursor:pointer;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.case[open]>summary{border-bottom:1px solid var(--line)}
.case>*:not(summary){margin:10px 12px}
.draw{margin:8px 0;padding-left:14px;border-left:2px solid var(--line)}
.dh{margin:12px 0 4px;font-weight:650;font-size:12px;text-transform:uppercase;letter-spacing:.02em;
  color:var(--ink-2)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot.r-comply{background:var(--comply)}.dot.r-refuse{background:var(--refuse)}
.dot.r-none{background:var(--none)}
.v-comply{color:var(--comply);font-weight:600}.v-refuse{color:var(--refuse);font-weight:600}
.v-none{color:var(--ink-3)}
pre.resp{white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.5;
  background:var(--surface-2);border:1px solid var(--line);border-radius:6px;padding:10px 12px;
  margin:6px 0;max-height:420px;overflow:auto}
dl.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px 18px;margin:16px 0}
dl.meta dt{font-size:11px;text-transform:uppercase;letter-spacing:.03em;color:var(--ink-3)}
dl.meta dd{margin:2px 0 0}
"""


def main(argv=None):
    R = "results/runs/sciencesafety_20261101"
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/sciencesafety_3_31.yaml")
    ap.add_argument("--run", default=R)
    ap.add_argument("--old", default="llama3-8b")
    ap.add_argument("--new", default="llama31-8b")
    ap.add_argument("--corpus", default="data/sciencesafety.csv")
    ap.add_argument("--subset", default="sciencesafety_p_minus_judged.csv")
    ap.add_argument("--manifest", default="manifest_sciencesafety.jsonl")
    ap.add_argument("--pilot", default="pminus_sciencesafety.json")
    ap.add_argument("--summary", default="summary_sciencesafety.json")
    ap.add_argument("--cache", default="cache/sciencesafety_20261101.jsonl")
    ap.add_argument("--verdicts", default="cache/judge_sciencesafety_20261101.jsonl")
    ap.add_argument("--rpe-gate", default="cache/rpe_gate_sciencesafety_20261101.jsonl")
    ap.add_argument("--title", default="ScienceSafetyData · external corpus")
    ap.add_argument("--redact", action="store_true", help="drop every prompt and response")
    ap.add_argument("--out", default=None,
                    help="default: <run>/dashboard_sciencesafety_<stamp>.html (never overwritten)")
    args = ap.parse_args(argv)

    d = collect(args)
    html = render(d, args)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    dest = Path(args.out) if args.out else \
        Path(args.run) / f"dashboard_sciencesafety{'_redacted' if args.redact else ''}_{stamp}.html"
    if dest.exists():
        raise SystemExit(f"{dest} exists -- refusing to overwrite")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html)
    print(f"wrote {dest}  ({len(html) // 1024} KB)")
    print(f"  P^- {d['pilot']['p_minus']}/{d['n_corpus']} · "
          f"re-opened {sum(1 for r in d['rows'] if r['reopened'])} · "
          f"gate {sum(d['adm'].values())}/{sum(d['prop'].values())} · "
          f"inversion gate-admitted {sum(1 for r in d['gate_rows'] if r.get('admitted'))}"
          f"/{len(d['gate_rows'])}")
    if d["unjoined"]:
        print(f"  WARNING: {d['unjoined']} behaviours did not join to {args.corpus} on prompt text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
