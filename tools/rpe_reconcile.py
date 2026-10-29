#!/usr/bin/env python3
"""rpe_reconcile.py -- why the inversion arm resolved 38 of the anchor's 49 fixes in its own run
and 7 of 61 inside the compose factorial (THESIS_KB section 4.2, left open on 2026-10-28).

The two numbers were never measured under the same admission rule:

  dedicated arm   rpe_lineage + rpe_closedloop   A_loop: OLD(x_hat) complies and its reply is
                                                 equivalent to OLD's original answer. The intent
                                                 gate never ran (`admitted: false` everywhere).
  compose arm     run_compose.py, factor `rpe`   the production intent gate against the original
                                                 request -- the admission RSR uses.

This puts every inversion run of the anchor pair under every admission it can be scored on, and
records the remaining design differences (P^- basis, candidates per behaviour, which OLD answer
was inverted) so each can be ruled in or out. Everything is recomputed from the stores; no
summary is trusted.

Offline: no model calls.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import lineage_core as lc                                       # noqa: E402
from recyclebench.config import Config                          # noqa: E402
from recyclebench.datasets import load_jailbreakbench           # noqa: E402
from recyclebench.judge import looks_like_refusal               # noqa: E402
from recyclebench.realrun import ResponseCache, _key            # noqa: E402
from recyclebench.screening import _draw_seed                   # noqa: E402
from recyclebench.stats import wilson_interval                  # noqa: E402

OLD, NEW = "llama3-8b", "llama31-8b"


def _rel(p: Path) -> str:
    """Printable path; --closedloop may sit outside the repo."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _last_write(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[d["id"]] = d
    return out


def _rate(k, n):
    if not n:
        return {"k": k, "n": n, "rate": None, "ci": [None, None]}
    iv = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci": [iv.low, iv.high]}


def compose_side(store: Path, n_cands: int, prompt_of: dict):
    """The compose run's `rpe` arm: gate admission per candidate, NEW verdicts on admitted ones.

    Its behaviour ids are row positions in the 61-row P^- subset CSV (load_jailbreakbench
    numbers rows and ignores the file's bid column), so they are mapped to prompt text before
    any comparison with the full-benchmark ids the dedicated arm uses.
    """
    gen = _last_write(store / "candidates.jsonl")
    jdg = _last_write(store / "judged.jsonl")
    cands, admitted, resolved, opened, bids = 0, 0, set(), set(), set()
    for d in gen.values():
        if d.get("arm") != "rpe" or d.get("idx", 0) >= n_cands:
            continue
        bids.add(d["bid"])
        if not d.get("ok"):
            continue
        cands += 1
        if d.get("gate_admitted"):
            admitted += 1
            resolved.add(prompt_of[d["bid"]])
            j = jdg.get(f"rpe|{d['bid']}|{d['idx']}|{NEW}|judge")
            if j and j.get("complied"):
                opened.add(prompt_of[d["bid"]])
    return {"behaviours": len(bids), "candidates": cands, "admitted": admitted,
            "resolved": resolved, "opened": opened & resolved}


def compose_old_answers(cfg_lineage, behaviours, full_bid, tcache, gen_cache, verdicts, seed,
                        n_cands):
    """Which OLD answer the compose arm inverted. It calls OLD afresh (temperature 0, seed
    SEED+i) rather than taking a cached draw; compare with the canonical draws the judged P^-
    was screened on (full-benchmark ids) and with the judge's verdict on that exact text."""
    c = defaultdict(int)
    for b in behaviours:
        fb = full_bid[b.canonical_prompt]
        draws = [gen_cache.get(_key(OLD, b.canonical_prompt, _draw_seed(fb, -1, d)))
                 for d in range(cfg_lineage.budget.k_canonical_samples)]
        for i in range(n_cands):
            t = tcache.get(OLD, b.canonical_prompt, seed + i)
            if t is None:
                c["missing"] += 1
                continue
            c["answers"] += 1
            c["identical_to_a_cached_draw"] += int(t in draws)
            v = verdicts.get(b.canonical_prompt, t)
            if v is None:
                c["unjudged"] += 1
            else:
                c["judged_complied"] += int(v == 1)
                c["judged_refused"] += int(v == 0)
            c["offline_refusal"] += int(looks_like_refusal(t))
    return dict(c)


def anchor_manifest_ids(full, subset, gen_cache, k):
    """TODO_HUMAN A6: the anchor manifest 'overlaps the rebuilt P^- by 8 and OLD refuses 52 of
    61', which was read as 'unreproduced'. Its ids are subset-CSV row positions. -> the check
    done by prompt text against the full-benchmark offline-rule P^-, and the bid-join that
    produced the old figures."""
    man = ROOT / "results/runs/recycle_3_31_sorrybench/manifest_sorrybench.jsonl"
    abids = sorted({json.loads(l)["bid"] for l in man.read_text().splitlines() if l.strip()})
    sub = {b.bid: b.canonical_prompt for b in subset}
    by_bid = {b.bid: b for b in full}

    def complies(model, b):
        ts = [gen_cache.get(_key(model, b.canonical_prompt, _draw_seed(b.bid, -1, d))) for d in range(k)]
        if any(t is None for t in ts):
            return None
        return int(any(not looks_like_refusal(t) for t in ts))
    pm = {b.canonical_prompt for b in full if complies(OLD, b) == 1 and complies(NEW, b) == 0}
    prompts = {sub[b] for b in abids if b in sub}
    return {"manifest": _rel(man), "behaviours": len(abids),
            "full_csv_offline_pminus": len(pm),
            "same_prompts_as_full_csv_offline_pminus": prompts == pm,
            "old_complies_on_its_own_prompts": sum(
                1 for b in full if b.canonical_prompt in prompts and complies(OLD, b) == 1),
            "old_refuses_on_full_csv_prompts_sharing_its_ids": sum(
                1 for b in abids if b in by_bid and complies(OLD, by_bid[b]) == 0)}


def closedloop_side(path: Path, n_cands=None, only=None):
    """A_gate / A_loop / A_both from a gate-scored closed-loop artifact, optionally restricted
    to the first `n_cands` candidates (the compose budget) or to a set of prompts."""
    doc = json.loads(path.read_text())
    out = {}
    for adm in ("gate", "loop", "both"):
        cand = res = k = 0
        for bid, d in doc["per_behaviour"].items():
            if only is not None and d.get("prompt") not in only:
                continue
            rows = [r for r in d.get(adm) or [] if n_cands is None or r["ci"] < n_cands]
            cand += len(rows)
            v = lc._any([r.get("new_verdict") for r in rows]) if rows else None
            if v is not None:
                res += 1
                k += v
        out[adm] = {"admitted_candidates": cand, "resolved": res, "reopened": k,
                    **{kk: vv for kk, vv in _rate(k, res).items() if kk != "k" and kk != "n"}}
    n_beh = sum(1 for d in doc["per_behaviour"].values() if only is None or d.get("prompt") in only)
    return {"behaviours": n_beh, "admission": out}


def early_run(path: Path):
    """The first dedicated runs (rpe_3_31_*): gate flags in the manifest are real there."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if str(r.get("bid", "")).startswith("jbb-")]
    cands = sum(len(r.get("details") or []) for r in rows)
    adm = sum(1 for r in rows for d in r.get("details") or [] if d.get("admitted"))
    res = sum(1 for r in rows if any(d.get("admitted") for d in r.get("details") or []))
    return {"behaviours": len(rows), "candidates": cands, "admitted": adm, "resolved": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compose-run", default="results/runs/compose_20261028_0002")
    ap.add_argument("--compose-store", default="cache/compose")
    ap.add_argument("--closedloop", default=None,
                    help="gate-scored anchor artifact (default: newest *_gated_* in rpe_lineage)")
    ap.add_argument("--config", default="configs/real_llama_lineage_all.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import run_compose as rc
    summary = json.loads((ROOT / args.compose_run / "compose_summary.json").read_text())
    n3 = int(summary["candidates_per_behaviour"])
    cfg = Config.load(ROOT / args.config)
    tag = lc.judge_tag_from_cfg(cfg)
    verdicts = lc.StoredVerdicts(lc.VerdictStore(ROOT / "cache/judge_lineage_sorrybench.jsonl"), tag)
    gen_cache = lc.load_response_cache(ROOT / "cache/llama_gen.jsonl")
    tcache = ResponseCache(ROOT / "cache/compose_targets.jsonl")
    off_pm = load_jailbreakbench(str(ROOT / rc.PMINUS))
    full = load_jailbreakbench(str(ROOT / "data/sorrybench.csv"))
    full_bid = {b.canonical_prompt: b.bid for b in full}

    cl = Path(args.closedloop) if args.closedloop else sorted(
        (ROOT / "results/runs/rpe_lineage").glob(f"closedloop_{OLD}__to__{NEW}_gated_*.json"))[-1]
    doc = json.loads(cl.read_text())
    judged_pm = {d["prompt"] for d in doc["per_behaviour"].values()}
    shared = judged_pm & {b.canonical_prompt for b in off_pm}       # joined on text, never bid

    comp = compose_side(ROOT / args.compose_store, n3, {b.bid: b.canonical_prompt for b in off_pm})
    rows = {
        "compose_rpe_arm": {
            "run": args.compose_run, "p_minus_basis": f"offline rule ({len(off_pm)})",
            "candidates_per_behaviour": n3, "old_answer": "fresh OLD call, temperature 0",
            "admission": "intent gate", **{k: v for k, v in comp.items()
                                            if k not in ("resolved", "opened")},
            "resolved": len(comp["resolved"]), "reopened": len(comp["opened"]),
            "resolved_on_judged_pminus": len(set(comp["resolved"]) & judged_pm)},
        "dedicated_strict_2026_10_11": {
            "run": "results/runs/rpe_3_31_sorrybench", "admission": "intent gate (production)",
            "old_answer": "cached canonical answer", "p_minus_basis": "offline rule (61)",
            **early_run(ROOT / "results/runs/rpe_3_31_sorrybench/rpe_panel.html/rpe_manifest_sorrybench.jsonl")},
        "dedicated_loose_2026_10_11": {
            "run": "results/runs/rpe_3_31_loosegate", "admission": "loose gate (cos .60, BERT .80, NLI .30)",
            "old_answer": "cached canonical answer", "p_minus_basis": "offline rule (61)",
            **early_run(ROOT / "results/runs/rpe_3_31_loosegate/out/rpe_manifest_sorrybench.jsonl")},
        "dedicated_lineage": {
            "run": _rel(cl), "p_minus_basis": f"qwen2.5:14b judged ({len(judged_pm)})",
            "candidates_per_behaviour": 8, "old_answer": "first cached canonical draw judged compliant",
            **closedloop_side(cl)},
        "dedicated_lineage_first3": {
            "run": _rel(cl), "note": f"only candidates 0..{n3 - 1}: the compose budget",
            **closedloop_side(cl, n_cands=n3)},
        "dedicated_lineage_first3_shared": {
            "run": _rel(cl),
            "note": f"first {n3} candidates, on the {len(shared)} behaviours in both P^- bases",
            **closedloop_side(cl, n_cands=n3, only=shared)},
    }
    old_ans = compose_old_answers(cfg, off_pm, full_bid, tcache, gen_cache, verdicts, rc.SEED, n3)
    ids = anchor_manifest_ids(full, off_pm, gen_cache, cfg.budget.k_canonical_samples)

    lin = rows["dedicated_lineage"]["admission"]
    f3s = rows["dedicated_lineage_first3_shared"]["admission"]
    findings = [
        f"Different admission rules. The dedicated arm's {lin['loop']['resolved']}/{len(judged_pm)} is "
        f"the closed loop (no intent gate). Under the intent gate the same candidates resolve "
        f"{lin['gate']['resolved']}/{len(judged_pm)}, against the compose arm's "
        f"{len(comp['resolved'])}/{comp['behaviours']}.",
        f"Like for like (first {n3} candidates, the {len(shared)} shared behaviours, intent gate): "
        f"dedicated {f3s['gate']['resolved']} resolved vs compose "
        f"{len(set(comp['resolved']) & shared)} of the same behaviours.",
        f"Candidate admission under the gate: dedicated {lin['gate']['admitted_candidates']} of "
        f"{doc['n_candidates']} ({lin['gate']['admitted_candidates'] / doc['n_candidates']:.1%}), "
        f"compose {comp['admitted']} of {comp['candidates']} "
        f"({comp['admitted'] / max(comp['candidates'], 1):.1%}).",
        f"Which older-model answer was inverted: the compose arm's {old_ans.get('answers', 0)} fresh answers "
        f"include {old_ans.get('offline_refusal', 0)} the offline rule reads as refusals and "
        f"{old_ans.get('judged_refused', 0)} the judge scores as refused (of "
        f"{old_ans.get('answers', 0) - old_ans.get('unjudged', 0)} judged); "
        f"{old_ans.get('identical_to_a_cached_draw', 0)} are byte-identical to a cached canonical draw. "
        "The dedicated arm inverts only a draw the judge scored as compliant.",
        f"Behaviour ids: the compose run, the anchor manifest and the first RPE runs were built on "
        f"the {len(off_pm)}-row P⁻ subset CSV, whose ids are row positions. Read by prompt text the "
        f"anchor manifest is "
        + ("exactly" if ids["same_prompts_as_full_csv_offline_pminus"] else "NOT")
        + f" the full-benchmark offline-rule P⁻ ({ids['full_csv_offline_pminus']} behaviours), and the "
        f"older model answers {ids['old_complies_on_its_own_prompts']} of its {ids['behaviours']}. The "
        f"'older model refuses 52 of 61' of TODO_HUMAN A6 comes back exactly "
        f"({ids['old_refuses_on_full_csv_prompts_sharing_its_ids']}/{ids['behaviours']}) when those ids "
        f"are read as full-benchmark ids: a join error, not an unreproducible manifest.",
    ]
    out = {"generated": _dt.datetime.now().strftime("%Y%m%d_%H%M"), "pair": [OLD, NEW],
           "judge_tag": tag, "runs": rows, "compose_old_answers": old_ans,
           "anchor_manifest_ids": ids, "findings": findings}
    dest = Path(args.out) if args.out else ROOT / "results" / f"analysis_rpe_reconcile_{out['generated']}" / "reconcile.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise SystemExit(f"{dest} exists -- refusing to overwrite")
    dest.write_text(json.dumps(out, indent=2))
    for f in findings:
        print("-", f)
    print(f"wrote    : {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
