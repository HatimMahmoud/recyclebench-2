#!/usr/bin/env python3
"""tools/llama4_suite.py -- the Llama 4 Scout experiment-completion matrix and result package.

Reads every artifact the Scout arms wrote (no model calls) and assigns each cell of
benchmark x older model x arm one status, with the evidence that decided it:

    COMPLETE — VERIFIED                         artifact present, coverage checked
    PENDING — EXECUTABLE                        prerequisites present, not (fully) run yet
    NOT APPLICABLE — SCIENTIFICALLY INELIGIBLE  the protocol excludes it (e.g. |P^-| below a floor)
    BLOCKED — MISSING PREREQUISITE              an input is absent here (named)
    FAILED — NEEDS REPAIR                       an artifact exists but is incomplete or inconsistent

Sources:
  canonical   results/runs/20261031_1816_llama4_scout_canonical/canonical.json (llama4_report.py)
  recycle     results/runs/llama4_scout/llama4_scout__<bench>__<old>/  (experiment driver)
  rpe         results/runs/llama4_scout/rpe/<bench>/                  (scout_rpe.sh)
  attack      results/runs/llama4_scout/attack_transfer/              (scout_attack.sh)

Every figure is RAW judge output: no human calibration exists for Scout, and the SORRY-Bench / HarmBench
calibrations are not transferred to it. Post hoc and scale-confounded: exploratory throughout.

  python3 tools/llama4_suite.py --out results/runs/llama4_scout/suite_<stamp>.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recyclebench.stats import wilson_interval              # noqa: E402

NEW = "llama4-scout"
OLDS = ("llama2-7b", "llama2-13b", "llama3-8b", "llama31-8b", "llama32-1b", "llama32-3b")
BENCHES = ("advbench", "harmbench", "strongreject")
CANONICAL = "results/runs/20261031_1816_llama4_scout_canonical/canonical.json"
SUITE = "results/runs/llama4_scout"
RECYCLE_FLOOR = 5            # ExperimentSpec.min_p_minus_for_recycle (the specs keep the default)

C, P, NA, B, F = ("COMPLETE — VERIFIED", "PENDING — EXECUTABLE",
                  "NOT APPLICABLE — SCIENTIFICALLY INELIGIBLE", "BLOCKED — MISSING PREREQUISITE",
                  "FAILED — NEEDS REPAIR")


def wilson(k, n):
    if not n:
        return None
    iv = wilson_interval(k, n)
    return [round(iv.low, 4), round(iv.high, 4)]


def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest(d: Path, pattern: str):
    xs = sorted(d.glob(pattern)) if d.exists() else []
    return xs[-1] if xs else None


def canonical(root: Path, floor: int) -> dict:
    rep = _json(root / CANONICAL) or {}
    out = {}
    for b in BENCHES:
        r = rep.get(b)
        if not r:
            continue
        models = {m: {**s, "rate": round(s["complied"] / s["judged"], 4) if s["judged"] else None,
                      "ci": wilson(s["complied"], s["judged"])}
                  for m, s in r["models"].items() if m != "llama4-maverick"}
        pairs = {}
        for o in OLDS:
            c = r["pairs"].get(f"{o}->{NEW}")
            if c:
                pairs[o] = {k: c[k] for k in ("pminus", "persistent", "neg_flip", "both_safe")}
        out[b] = {"n_behaviours": r["n_behaviours"], "models": models, "pairs": pairs,
                  "complete": all(models[m]["missing_responses"] == 0 and models[m]["missing_verdicts"] == 0
                                  for m in models)}
    out["_floor"] = floor
    return out


def manifest_stats(path: Path) -> dict:
    """Behaviours, variants and admitted variants in a frozen manifest (counts only)."""
    beh, var, adm, per = set(), 0, 0, {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("type") != "audit":
            continue
        var += 1
        beh.add(r.get("bid"))
        a = bool(r.get("admitted"))
        adm += a
        t = per.setdefault(r.get("transform_class") or "?", [0, 0])
        t[0] += a
        t[1] += 1
    return {"behaviours": len(beh), "variants": var, "admitted": adm,
            "by_class": {k: {"admitted": v[0], "n": v[1]} for k, v in sorted(per.items())}}


def recycle(root: Path, bench: str, old: str, p_stored: int) -> dict:
    name = f"llama4_scout__{bench}__{old}"
    d = root / SUITE / name
    cell = {"spec": f"experiments/llama4_scout/{bench}__{old}.yaml", "run_dir": f"{SUITE}/{name}",
            "p_minus_stored": p_stored}
    if p_stored < RECYCLE_FLOOR:
        cell.update(status=NA, why=f"|P^-| = {p_stored} < min_p_minus_for_recycle {RECYCLE_FLOOR}: "
                                   f"RSR would be undefined or unpowered; no spec")
        return cell
    if not d.exists():
        cell.update(status=P, why="spec present, run directory absent: not run yet")
        return cell
    exp = _json(d / "experiment.json") or {}
    cell["code_revision"] = (exp.get("fingerprint") or {}).get("code_revision")
    cell["slurm_job_id"] = (exp.get("runtime") or {}).get("slurm_job_id")
    cell["stages_done"] = sorted(p.stem for p in (d / "stages").glob("*.json")) if (d / "stages").exists() else []
    pilot = _json(d / "pminus_pilot.json")
    if pilot:
        cell["p_minus_rejudged"] = pilot.get("p_minus")
        cell["rejudge_matches_stored"] = pilot.get("p_minus") == p_stored
    summ = _json(d / "summary_real.json")
    man = _latest(d, "manifest_*.jsonl")
    if man:
        cell["manifest"] = {"path": str(man.relative_to(root)), **manifest_stats(man)}
    if summ:
        rsr = [r for r in summ.get("RSR_primary_RAW") or [] if r.get("pair") == f"{old} -> {NEW}"]
        if rsr:
            r = rsr[0]
            cell["rsr"] = {k: r.get(k) for k in ("n_p_minus", "k_recovered", "n_unresolved", "defined",
                                                 "RSR_raw", "mcnemar_p", "mean_admitted", "denominator")}
            if r.get("defined"):
                cell["rsr"]["wilson"] = wilson(r["k_recovered"], r["n_p_minus"])
        cell["pbr"] = {p.get("model") or p.get("name") or str(i): p
                       for i, p in enumerate(summ.get("PBR_backbone_RAW") or [])}
    tr = d / "trials" / f"trials_{NEW}.csv"
    if tr.exists():                        # variant level: descriptive only, never an RSR denominator
        import csv
        by = {}
        with tr.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                t = by.setdefault(r["transform_class"], [0, 0])
                t[0] += int(r["success"] or 0)
                t[1] += 1
        cell["scout_variants_by_class"] = {k: {"reopened": v[0], "admitted": v[1]} for k, v in sorted(by.items())}
    need = {"screen", "subset", "recycle", "partition", "audit"}
    if summ and man and need <= set(cell["stages_done"]):
        cell.update(status=C, why="all five stages recorded with artifacts; RSR raw, exploratory")
    elif pilot and pilot.get("p_minus", 0) < RECYCLE_FLOOR:
        cell.update(status=NA, why=f"re-judged |P^-| = {pilot.get('p_minus')} < {RECYCLE_FLOOR}: the driver "
                                   f"skips recycle (stored judged P^- = {p_stored})")
    elif cell["stages_done"]:
        cell.update(status=P, why=f"stages done {cell['stages_done']}; rerun the spec to resume")
    else:
        cell.update(status=F, why="run directory exists but no stage completed: read the job log")
    return cell


def rpe(root: Path, bench: str, old: str, p_stored: int) -> dict:
    d = root / SUITE / "rpe" / bench
    cell = {"p_minus_stored": p_stored}
    if p_stored == 0:
        cell.update(status=NA, why="|P^-| = 0: PRR undefined (never 0), nothing to invert")
        return cell
    man = d / f"rpe_manifest_{old}__to__{NEW}.jsonl"
    if not man.exists():
        cell.update(status=P, why="no inversion manifest yet (scout_rpe.sh step 1)")
        return cell
    rows = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
    cell["inverted_behaviours"] = len({r["bid"] for r in rows})
    cell["candidates"] = sum(len(r.get("details") or []) for r in rows)
    cl = _latest(d, f"closedloop_{old}__to__{NEW}_gated_*.json")
    if not cl:
        cell.update(status=P, why="manifest built; closed loop not scored yet")
        return cell
    j = _json(cl) or {}
    cell["closedloop"] = str(cl.relative_to(root))
    adm = j.get("admission") or {}
    cell["admission"] = {k: {"k": v["k"], "n": v["n"], "unresolved": v["unresolved"],
                             "wilson": wilson(v["k"], v["n"])} for k, v in adm.items()}
    covered = j.get("n_behaviours", 0)
    if covered < cell["inverted_behaviours"]:
        cell.update(status=F, why=f"scored {covered} of {cell['inverted_behaviours']} behaviours")
    else:
        cell.update(status=C, why="all P^- behaviours inverted, gated and scored (A_gate, A_loop, A_both)")
    return cell


def attack(root: Path) -> dict:
    d = root / SUITE / "attack_transfer"
    res = _latest(d, "llama4-scout_judged_original_*.json")
    classes = (d / "classes.txt").read_text().split() if (d / "classes.txt").exists() else []
    if not res:
        return {"status": P, "why": "scout_attack.sh has not written its result yet", "classes_listed": len(classes)}
    j = _json(res) or {}
    rows = {c: {"k": r["n_broke_among_refused"], "n": r["n_refused_baseline"],
                "transfer": r["transfer_on_refused"], "wilson": wilson(r["n_broke_among_refused"],
                                                                       r["n_refused_baseline"]),
                "bypass_k": r["n_broke"], "bypass_n": r["n"]}
            for c, r in (j.get("classes") or {}).items()}
    missing = sorted(set(classes) - set(rows))
    return {"status": C if rows and not missing else F, "path": str(res.relative_to(root)),
            "why": "every corpus class judged against the original goal" if not missing
                   else f"classes missing from the result: {missing}",
            "n_behaviours": j.get("n_behaviours"), "n_refused_baseline": j.get("n_refused_baseline"),
            "judge_against": j.get("judge_against"), "classes": rows}


STATIC = [  # arms decided by protocol or by inputs this machine does not hold
    ("canonical · sorrybench", B, "data/sorrybench.csv is absent on Pegasus and here (gated on Hugging Face: "
     "HF_TOKEN + tools/prepare_sorrybench.py, or copy the registered CSV from the Mac)"),
    ("canonical · heldout_jbb_guided", NA, "registered for the confirmatory llama3-8b → llama3.1-8b replication only "
     "(Amendment 1 §A2); the CSV is absent here and must never be rebuilt"),
    ("PBR on the frozen lineage manifests", B, "the HarmBench/AdvBench/StrongREJECT lineage manifests "
     "(results/runs/20260801_165721_llama32-1b_to_3b_recycle/manifest_*.jsonl) exist only on the Mac; with them "
     "Scout is re-screened on the identical variants (RSR for every pair + PBR)"),
    ("RPE inversion reuse (--reuse-from)", B, "lineage inversion manifests live only on the Mac; Scout's run "
     "inverted every OLD answer afresh (same inverter and defaults), so this costs reuse, not validity"),
    ("compose v2 (style factorial)", NA, "registered for the anchor pair's 49 fixes and an addendum pair; "
     "Scout's largest P^- is 7 and no addendum covers it"),
    ("mechanistic probe", NA, "tools/mech_probe.py implements dense Llama-3 blocks; Scout is a 16-expert MoE, "
     "so applying it would be a new method"),
    ("PTP", NA, "arm parked until best-of-N is registered prospectively"),
    ("human judge calibration for Scout", B, "no labelled sheet for Scout; the driver's audit stage writes "
     "audit/llama4_scout__*/judge_sheet.csv to label; until then every Scout figure is raw"),
]


def collect(root: Path) -> dict:
    can = canonical(root, RECYCLE_FLOOR)
    cells = []
    for b in BENCHES:
        if b not in can:
            continue
        for o in OLDS:
            p = can[b]["pairs"].get(o, {}).get("pminus", 0)
            cells.append({"bench": b, "old": o, "arm": "canonical", "status": C if can[b]["complete"] else F,
                          "why": "re-derived independently from cache + verdict store; 100% coverage"
                                 if can[b]["complete"] else "missing responses or verdicts",
                          **can[b]["pairs"].get(o, {})})
            cells.append({"bench": b, "old": o, "arm": "recycle", **recycle(root, b, o, p)})
            cells.append({"bench": b, "old": o, "arm": "rpe", **rpe(root, b, o, p)})
    arcs = sorted(str(p.relative_to(root)) for p in (root / SUITE).glob("style_arcs_*.html"))
    return {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "new": NEW,
            "designation": "post hoc, scale-confounded (109B MoE vs <=13B dense), exploratory; raw judge output",
            "recycle_floor": RECYCLE_FLOOR, "confirmatory_floor": 30,
            "canonical": can, "cells": cells, "attack": attack(root),
            "style_arcs": {"status": C if arcs else P, "files": arcs},
            "static": [{"arm": a, "status": s, "why": w} for a, s, w in STATIC]}


def markdown(s: dict) -> str:
    def md(x):                                   # a literal | (as in |P^-|) would split the table cell
        return str(x).replace("|", "\\|")
    L = [f"# Llama 4 Scout: experiment completion ({s['generated']})", "", f"_{s['designation']}_", "",
         "| benchmark | older model | arm | P⁻ | status | evidence |", "|---|---|---|---|---|---|"]
    for c in s["cells"]:
        p = c.get("pminus", c.get("p_minus_stored", ""))
        L.append(f"| {c['bench']} | {c['old']} | {c['arm']} | {p} | {c['status']} | {md(c['why'])} |")
    a = s["attack"]
    L += ["", "| arm | status | evidence |", "|---|---|---|",
          f"| attack transfer (Wei et al. corpus, its own 300 base behaviours) | {a['status']} | {md(a['why'])} |",
          f"| style arcs | {s['style_arcs']['status']} | {', '.join(s['style_arcs']['files']) or 'not drawn yet'} |"]
    L += [f"| {md(x['arm'])} | {x['status']} | {md(x['why'])} |" for x in s["static"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=None, help="JSON path; a .md beside it. Default: <suite>/suite_<stamp>.json")
    args = ap.parse_args(argv)
    root = Path(args.root)
    s = collect(root)
    out = Path(args.out) if args.out else root / SUITE / f"suite_{_dt.datetime.now():%Y%m%d_%H%M}.json"
    if out.exists():
        raise SystemExit(f"{out} exists -- refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(s), encoding="utf-8")
    print(markdown(s))
    print(f"wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
