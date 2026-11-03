#!/usr/bin/env python3
"""tools/external_suite.py -- GuidedBench, ScienceSafetyData and JBB-Behaviors on every model (Pegasus).

The design of the external-corpora run (PREREG_EXTERNAL_CLUSTER_20261103.md) lives here and nowhere else:
the three corpora with their registered fingerprints, the eight models, the 19 release pairs and the job
group each pair runs in. The pegasus/external_*.sh scripts and the committed configs and specs are generated
from it or read it, so a pair cannot be in one place and missing from another.

  configs   write the per-pair science configs from configs/llama4_pegasus.yaml (byte-identical except the
            header and `lineage:`); never overwrites. --check: every pair config matches the base body
  specs     write experiments/external/<corpus>__<old>__<new>.yaml (57 = 3 corpora x 19 pairs); --check
  data      check data/<corpus>.csv against the registered fingerprints (rows + ordered-request hash);
            --build runs the corpus's builder first when its CSV is absent
  report    canonical 2x2 of every pair on every corpus, from the response cache + verdict stores (no model calls)
  list      what a job runs: --pairs (rpe_lineage format), or --specs [--group G] [--eligible]
  matrix    the completion matrix: every corpus x pair x arm gets one status, with the evidence that decided it

RB_CORPORA (comma list, default all three) narrows what data / report / list and the pegasus/external_*.sh jobs
act on, so one corpus can run while another's CSV is still missing. The matrix always covers all three: a corpus
left out stays PENDING or BLOCKED there, never silently absent.

No harmful text is printed or written: every output is counts, ids and hashes.

  python3 tools/external_suite.py configs --check && python3 tools/external_suite.py specs --check
  python3 tools/external_suite.py data --build
  python3 tools/external_suite.py report --out results/runs/external/canonical_<stamp>.json
  python3 tools/external_suite.py list --specs --group dense --eligible
  python3 tools/external_suite.py matrix
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

PREREG = "PREREG_EXTERNAL_CLUSTER_20261103.md"
BASE_CONFIG = "configs/llama4_pegasus.yaml"
RUN_ROOT = "results/runs/external"
CACHE_DIR = "cache/external"
CANONICAL_CACHE = f"{CACHE_DIR}/canonical.jsonl"
RECYCLE_FLOOR = 5            # ExperimentSpec.min_p_minus_for_recycle, as in every earlier recycle spec
CONFIRMATORY_FLOOR = 30      # decision.min_p_minus

# Registered fingerprints, taken from the Mac's copies on 2026-11-03. Behaviour ids are row positions and
# canonical seeds derive from them, so the ORDERED request texts must match, not only the row count.
# goals_sha256 = sha256 of the requests joined by U+001E, as recyclebench.datasets.load_jailbreakbench reads them.
CORPORA = {
    "guidedbench": {
        "rows": 200, "categories": 20,
        "goals_sha256": "ff3fa755d8686036016246a866e0627bc847839244da07a361fe1191ac628f48",
        "file_sha256": "5af3394b949b6e385993f10dd3bee60e95a055e5f54a2bb8734dd5947679a53c",
        "build": None,
        "source": "GuidedBench (Huang et al., arXiv:2502.16903): the 200 harmful questions, columns Goal, Category",
    },
    "sciencesafety": {
        "rows": 240, "categories": 6,
        "goals_sha256": "b3b55bf39f27b8b4a0088cd20123c2165f450ce89c450b9a80d8327b0e124309",
        "file_sha256": "7c3cddbec048e676aa6cb3adc8f4e145bb789689a969093bf9c51ff751e452c6",
        "build": ["tools/prepare_external.py", "--recipe", "datasets/sciencesafety.yaml"],
        "source": "ulab-uiuc/SafeScientist @ 9db8659, data/ScienceSafetyData/Dataset (datasets/sciencesafety.yaml)",
    },
    "jbb": {
        "rows": 100, "categories": 10,
        "goals_sha256": "2911a91083756ee1e9ecb2167426307c14ad9e58716f3df6699e5f8ef1a07d28",
        "file_sha256": "d0c22d73c6a24adb610feb6bd8fa317820b1524381c286a3bbe83674e19fa61d",
        "build": ["tools/prepare_behaviours.py", "--benchmark", "jbb", "--out", "data/jbb.csv"],
        "source": "JailbreakBench/JBB-Behaviors (Hugging Face), harmful split",
    },
}


def _selected() -> tuple:
    names = tuple(c for c in os.environ.get("RB_CORPORA", "").replace(" ", ",").split(",") if c)
    bad = [c for c in names if c not in CORPORA]
    if bad:
        raise SystemExit(f"RB_CORPORA: unknown corpus {bad} (registered: {', '.join(CORPORA)})")
    return names or tuple(CORPORA)


SELECTED = _selected()        # what this invocation acts on (RB_CORPORA); CORPORA stays the registration

DENSE = ("llama2-7b", "llama2-13b", "llama3-8b", "llama31-8b", "llama32-1b", "llama32-3b")
MODELS = DENSE + ("llama4-scout", "llama4-maverick")

# group = the GPU shape a pair's inline-judging jobs need (docs/pegasus/LESSONS.md §4b.6):
#   dense     any 80 GB card (largest target 13B + the 14B judge + the gate models)
#   scout     Scout 67 GB + judge 9 GB: one card > 80 GB, or 2 x 80 GB
#   maverick  Maverick 245 GB: 2 x B200 / H200, or >= 3 x 80-96 GB
_SCOUT = "post hoc, scale-confounded (109B MoE, 17B active, vs dense <= 13B)"
_MAV = "post hoc, scale-confounded (400B MoE, 17B active, vs dense <= 13B)"
PAIRS = (
    ("llama2-7b", "llama2-13b", "dense", "within generation (Llama 2 7B -> 13B)"),
    ("llama2-13b", "llama3-8b", "dense", "adjacent rung, scale-confounded (13B -> 8B)"),
    ("llama3-8b", "llama31-8b", "dense", "matched scale (the anchor pair)"),
    ("llama31-8b", "llama32-1b", "dense", "adjacent rung, scale-confounded (8B -> 1B)"),
    ("llama32-1b", "llama32-3b", "dense", "within generation (Llama 3.2 1B -> 3B)"),
    ("llama2-7b", "llama3-8b", "dense", "matched scale, post hoc (not adjacent)"),
    ("llama2-7b", "llama31-8b", "dense", "matched scale, post hoc (not adjacent)"),
) + tuple((o, "llama4-scout", "scout", _SCOUT) for o in DENSE) \
  + tuple((o, "llama4-maverick", "maverick", _MAV) for o in DENSE)
GROUPS = ("dense", "scout", "maverick")
CONFIG_DIRS = {"dense": "configs/lineage_pairs", "scout": "configs/llama4_scout", "maverick": "configs/llama4_maverick"}

C, P, NA, B, F = ("COMPLETE — VERIFIED", "PENDING — EXECUTABLE",
                  "NOT APPLICABLE — SCIENTIFICALLY INELIGIBLE", "BLOCKED — MISSING PREREQUISITE",
                  "FAILED — NEEDS REPAIR")


def pair_config_path(old, new, group) -> str:
    return f"{CONFIG_DIRS[group]}/{old}__{new}.yaml"


def spec_name(corpus, old, new) -> str:
    return f"external__{corpus}__{old}__{new}"


def spec_path(corpus, old, new) -> str:
    return f"experiments/external/{corpus}__{old}__{new}.yaml"


def verdict_store(corpus) -> str:
    return f"{CACHE_DIR}/judge_qwen_{corpus}.jsonl"


# =============================================================================
# configs
# =============================================================================
_LINEAGE = re.compile(r"^lineage:[^\n]*\n(?:  - [^\n]*\n)+", re.M)


def _split_header(text: str):
    """(header, body): the header is the leading comment block closed by its second '# ====' rule."""
    lines = text.splitlines(keepends=True)
    rules = [i for i, ln in enumerate(lines) if ln.startswith("# ====")]
    if len(rules) < 2 or rules[0] != 0:
        raise SystemExit("config header must open and close with a '# ====' rule")
    return "".join(lines[:rules[1] + 1]), "".join(lines[rules[1] + 1:])


def pair_body(base_text: str, old: str, new: str) -> str:
    _, body = _split_header(base_text)
    m = _LINEAGE.search(body)
    if not m:
        raise SystemExit(f"no lineage: block in {BASE_CONFIG}")
    return body[:m.start()] + f"lineage:\n  - {old}\n  - {new}\n" + body[m.end():]


def pair_header(old, new, design) -> str:
    return (
        "# =====================================================================\n"
        f"# {old} -> {new} release pair, generated from {BASE_CONFIG}\n"
        f"# by tools/external_suite.py for the external-corpora run ({PREREG}).\n"
        "#\n"
        "# Byte-identical to that config except this header and `lineage:`. run-real screens every\n"
        "# lineage model and pairs ADJACENT rungs only, so the pair has to be the whole lineage. Seed,\n"
        "# judges, gate, budgets, transform classes, paraphraser and model endpoints are unchanged, so\n"
        "# cache keys and verdicts are shared with every other run of that config.\n"
        f"# Design: {design}. EXPLORATORY, raw, never pooled with the registered pairs.\n"
        "# =====================================================================\n")


def configs(root: Path, check: bool) -> list:
    """Write missing pair configs (never overwrite); return the problems a --check finds."""
    base = (root / BASE_CONFIG).read_text(encoding="utf-8")
    problems = []
    for old, new, group, design in PAIRS:
        p = root / pair_config_path(old, new, group)
        want = pair_body(base, old, new)
        if p.exists():
            _, have = _split_header(p.read_text(encoding="utf-8"))
            if have != want:
                problems.append(f"{p.relative_to(root)}: body differs from {BASE_CONFIG} with lineage [{old}, {new}]")
        elif check:
            problems.append(f"{p.relative_to(root)}: missing (run `external_suite.py configs`)")
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(pair_header(old, new, design) + want, encoding="utf-8")
            print(f"wrote {p.relative_to(root)}")
    return problems


# =============================================================================
# specs
# =============================================================================
def spec_text(corpus, old, new, group, design) -> str:
    return (
        f"# External-corpora run: {old} -> {new} on {corpus}. EXPLORATORY ({PREREG}).\n"
        "#\n"
        "# Generated by tools/external_suite.py; `external_suite.py specs --check` compares, so edit the\n"
        f"# generator, not this file. Design: {design}. Job group: {group} (pegasus/external_recycle.sh).\n"
        "# Every figure is raw: no human calibration exists for this corpus. pegasus/external_recycle.sh runs\n"
        f"# the spec only when the judged canonical |P^-| is at least {RECYCLE_FLOOR}; the driver applies the same floor\n"
        "# to its own re-judged P^-, so RSR is never reported for an unpowered pair.\n"
        f"name: {spec_name(corpus, old, new)}\n"
        f"science_config: {pair_config_path(old, new, group)}\n"
        f"old: {old}\n"
        f"new: {new}\n"
        f"behaviours: data/{corpus}.csv\n"
        "\n"
        "execution:\n"
        "  target: cluster\n"
        f"  run_root: {RUN_ROOT}\n"
        f"  cache_dir: {CACHE_DIR}        # this spec's own cache, seeded from {CANONICAL_CACHE}\n"
        "\n"
        "stages: [screen, subset, recycle, partition, audit]\n"
        "audit_bundles: 60\n"
        f"min_p_minus_for_recycle: {RECYCLE_FLOOR}\n")


def specs(root: Path, check: bool) -> list:
    problems = []
    for corpus in CORPORA:
        for old, new, group, design in PAIRS:
            p = root / spec_path(corpus, old, new)
            want = spec_text(corpus, old, new, group, design)
            if p.exists():
                if p.read_text(encoding="utf-8") != want:
                    problems.append(f"{p.relative_to(root)}: differs from what the generator writes")
            elif check:
                problems.append(f"{p.relative_to(root)}: missing (run `external_suite.py specs`)")
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(want, encoding="utf-8")
                print(f"wrote {p.relative_to(root)}")
    return problems


# =============================================================================
# data
# =============================================================================
def fingerprint(csv_path: Path) -> dict:
    from recyclebench.datasets import load_jailbreakbench
    bs = load_jailbreakbench(str(csv_path))
    return {"rows": len(bs), "categories": len({b.category for b in bs}),
            "goals_sha256": hashlib.sha256("\x1e".join(b.canonical_prompt for b in bs).encode()).hexdigest(),
            "file_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}


def data_check(root: Path, build: bool, corpora=None) -> dict:
    out = {}
    for corpus in (CORPORA if corpora is None else corpora):
        reg = CORPORA[corpus]
        p = root / "data" / f"{corpus}.csv"
        if not p.exists() and build and reg["build"]:
            print(f"building data/{corpus}.csv: {' '.join(reg['build'])}")
            subprocess.run([sys.executable, *reg["build"]], cwd=root, check=False)
        if not p.exists():
            how = " ".join(reg["build"]) if reg["build"] else f"place it there ({reg['source']})"
            out[corpus] = {"status": B, "why": f"data/{corpus}.csv absent: {how}"}
            continue
        fp = fingerprint(p)
        ok = fp["rows"] == reg["rows"] and fp["goals_sha256"] == reg["goals_sha256"]
        why = ("rows and ordered requests match the registration"
               + ("" if fp["file_sha256"] == reg["file_sha256"] else " (file bytes differ: encoding or quoting only)")
               if ok else f"registered {reg['rows']} rows / goals {reg['goals_sha256'][:12]}, found "
                          f"{fp['rows']} / {fp['goals_sha256'][:12]}: ids and seeds would not be the registered ones")
        out[corpus] = {"status": C if ok else F, "why": why, **fp}
    return out


# =============================================================================
# report (no model calls)
# =============================================================================
def report(root: Path, config: str, cache_path: str, corpora) -> dict:
    import lineage_core as lc
    from recyclebench.config import Config
    from recyclebench.datasets import load_jailbreakbench
    from recyclebench.manifest import Manifest

    cfg = Config.load(str(root / config))
    cache = lc.load_response_cache(str(root / cache_path))
    out = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "config": config,
           "cache": cache_path, "judge": lc.judge_tag_from_cfg(cfg), "corpora": {}}
    for corpus in corpora:
        csv = root / "data" / f"{corpus}.csv"
        if not csv.exists():
            out["corpora"][corpus] = {"missing": f"data/{corpus}.csv"}
            continue
        behaviours = load_jailbreakbench(str(csv))
        store = lc.VerdictStore(str(root / verdict_store(corpus)))
        verdicts = lc.StoredVerdicts(store, lc.judge_tag_from_cfg(cfg))
        obs, runs = lc.build_runs(cfg, behaviours, Manifest(), cache, verdicts, list(MODELS))
        store.close()
        rep = {"n_behaviours": len(behaviours), "fingerprint": fingerprint(csv), "models": {}, "pairs": {}}
        for m in MODELS:
            r = runs[m]
            rep["models"][m] = {"judged": len(r.canonical_obs), "complied": len(r.broken_set()),
                                "missing_responses": obs[m].missing_responses,
                                "missing_verdicts": obs[m].missing_verdicts["canonical"]}
        for old, new, group, _ in PAIRS:
            cells = lc.partition(runs[old], runs[new])
            complete = all(rep["models"][m]["missing_responses"] == 0 and rep["models"][m]["missing_verdicts"] == 0
                           for m in (old, new))
            rep["pairs"][f"{old}->{new}"] = {k: len(v) for k, v in cells.items()} | {
                "group": group, "complete": complete, "pminus_bids": sorted(cells["pminus"])}
        out["corpora"][corpus] = rep
    return out


def latest_report(root: Path):
    xs = sorted((root / RUN_ROOT).glob("canonical_*.json")) if (root / RUN_ROOT).exists() else []
    return xs[-1] if xs else None


def _pairs_of(rep_path, corpus) -> dict:
    rep = json.loads(Path(rep_path).read_text(encoding="utf-8"))
    return (rep.get("corpora", {}).get(corpus) or {}).get("pairs") or {}


def eligible_specs(root: Path, group=None, rep_path=None) -> list:
    """Specs of complete pairs whose judged canonical |P^-| reaches the floor (the Scout suite's rule)."""
    rep_path = rep_path or latest_report(root)
    if not rep_path:
        raise SystemExit(f"no {RUN_ROOT}/canonical_*.json: run the canonical job (external_suite.py report) first")
    out = []
    for corpus in SELECTED:
        pairs = _pairs_of(rep_path, corpus)
        for old, new, g, _ in PAIRS:
            c = pairs.get(f"{old}->{new}")
            if (group in (None, g)) and c and c["complete"] and c["pminus"] >= RECYCLE_FLOOR:
                out.append(spec_path(corpus, old, new))
    return out


# =============================================================================
# matrix (no model calls)
# =============================================================================
def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest(d: Path, pattern: str):
    xs = sorted(d.glob(pattern)) if d.exists() else []
    return xs[-1] if xs else None


def wilson(k, n):
    from recyclebench.stats import wilson_interval
    if not n:
        return None
    iv = wilson_interval(k, n)
    return [round(iv.low, 4), round(iv.high, 4)]


def _manifest_stats(path: Path) -> dict:
    beh, var, adm = set(), 0, 0
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r.get("type") == "audit":
                var += 1
                beh.add(r.get("bid"))
                adm += bool(r.get("admitted"))
    return {"behaviours": len(beh), "variants": var, "admitted": adm}


def recycle_cell(root: Path, corpus, old, new, p) -> dict:
    name = spec_name(corpus, old, new)
    d = root / RUN_ROOT / name
    cell = {"spec": spec_path(corpus, old, new), "run_dir": f"{RUN_ROOT}/{name}"}
    if p is None:
        return {**cell, "status": P, "why": "canonical screen not complete for this pair"}
    if p < RECYCLE_FLOOR:
        return {**cell, "status": NA, "why": f"judged |P^-| = {p} < floor {RECYCLE_FLOOR}: not run "
                                             f"(RSR would be undefined or unpowered)"}
    if not d.exists():
        return {**cell, "status": P, "why": "eligible, run directory absent: not run yet"}
    stages = sorted(x.stem for x in (d / "stages").glob("*.json")) if (d / "stages").exists() else []
    cell["stages_done"] = stages
    pilot, summ = _json(d / "pminus_pilot.json"), _json(d / "summary_real.json")
    man = _latest(d, "manifest_*.jsonl")
    if pilot:
        cell["p_minus_rejudged"] = pilot.get("p_minus")
    if man:
        cell["manifest"] = {"path": str(man.relative_to(root)), **_manifest_stats(man)}
    if summ:
        rsr = [r for r in summ.get("RSR_primary_RAW") or [] if r.get("pair") == f"{old} -> {new}"]
        if rsr:
            r = rsr[0]
            cell["rsr"] = {k: r.get(k) for k in ("n_p_minus", "k_recovered", "n_unresolved", "defined",
                                                 "RSR_raw", "mcnemar_p", "denominator")}
            if r.get("defined"):
                cell["rsr"]["wilson"] = wilson(r["k_recovered"], r["n_p_minus"])
    if summ and man and {"screen", "subset", "recycle", "partition", "audit"} <= set(stages):
        return {**cell, "status": C, "why": "all five stages recorded with artifacts; RSR raw, exploratory"}
    if pilot and (pilot.get("p_minus") or 0) < RECYCLE_FLOOR:
        return {**cell, "status": NA, "why": f"re-judged |P^-| = {pilot.get('p_minus')} < {RECYCLE_FLOOR}: the "
                                             f"driver skipped recycle (canonical judged P^- = {p})"}
    if stages:
        return {**cell, "status": P, "why": f"stages done {stages}; rerun the spec to resume"}
    return {**cell, "status": F, "why": "run directory exists but no stage completed: read the job log"}


def rpe_cell(root: Path, corpus, old, new, p) -> dict:
    d = root / RUN_ROOT / "rpe" / corpus
    if p is None:
        return {"status": P, "why": "canonical screen not complete for this pair"}
    if p == 0:
        return {"status": NA, "why": "|P^-| = 0: PRR undefined (never 0), nothing to invert"}
    man = d / f"rpe_manifest_{old}__to__{new}.jsonl"
    if not man.exists():
        return {"status": P, "why": "no inversion manifest yet (external_rpe.sh invert)"}
    rows = [json.loads(ln) for ln in man.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cell = {"inverted_behaviours": len({r["bid"] for r in rows}),
            "candidates": sum(len(r.get("details") or []) for r in rows)}
    cl = _latest(d, f"closedloop_{old}__to__{new}_gated_*.json")
    if not cl:
        return {**cell, "status": P, "why": "manifest built; closed loop not scored yet (external_rpe.sh loop)"}
    j = _json(cl) or {}
    cell["closedloop"] = str(cl.relative_to(root))
    cell["admission"] = {k: {"k": v["k"], "n": v["n"], "unresolved": v["unresolved"], "wilson": wilson(v["k"], v["n"])}
                         for k, v in (j.get("admission") or {}).items()}
    if j.get("n_behaviours", 0) < cell["inverted_behaviours"]:
        return {**cell, "status": F, "why": f"scored {j.get('n_behaviours', 0)} of {cell['inverted_behaviours']} behaviours"}
    return {**cell, "status": C, "why": "all P^- behaviours inverted, gated and scored (A_gate, A_loop, A_both)"}


STATIC = [  # arms this run does not execute, and why
    ("known-attack transfer", NA, "a separate estimand on its own corpus (liuyanchen1015, 300 base behaviours), "
     "independent of these three corpora; not part of this registration"),
    ("ScienceSafetyData Attack/ split", NA, "the same behaviours pre-wrapped in seven attack templates: an "
     "attack-transfer input that needs its own registration, never a recycle input (the gate must certify our "
     "paraphrase)"),
    ("compose v2 (style factorial)", NA, "registered for the anchor pair's SORRY-Bench fixes and one addendum pair; "
     "no registration covers these corpora"),
    ("mechanistic probe", NA, "tools/mech_probe.py implements the anchor pair's dense Llama-3 blocks only"),
    ("model-level PBR", NA, "each pair freezes a manifest over its own P^- only, so a PBR here would be RSR "
     "again; a model-level PBR needs a manifest over every refused behaviour"),
    ("human judge calibration", B, "the audit stage writes audit/external__*/judge_sheet.csv; until it is "
     "labelled every figure is raw"),
    ("70B models", NA, "out of scope by decision (2026-11-03)"),
]


def collect(root: Path, rep_path=None) -> dict:
    rep_path = rep_path or latest_report(root)
    rep = _json(rep_path) if rep_path else None
    data = data_check(root, build=False)
    cells = []
    for corpus in CORPORA:
        crep = ((rep or {}).get("corpora") or {}).get(corpus) or {}
        for old, new, group, design in PAIRS:
            c = (crep.get("pairs") or {}).get(f"{old}->{new}")
            p = c["pminus"] if c and c.get("complete") else None
            base = {"corpus": corpus, "old": old, "new": new, "group": group, "design": design}
            if c:
                base.update({k: c[k] for k in ("pminus", "persistent", "neg_flip", "both_safe")})
            if not rep:
                can = {"status": P if data[corpus]["status"] == C else B,
                       "why": "no canonical report yet (external_canonical.sh)" if data[corpus]["status"] == C
                       else data[corpus]["why"]}
            elif not c:
                can = {"status": B, "why": data[corpus]["why"] if data[corpus]["status"] != C
                       else "corpus absent from the canonical report"}
            elif c["complete"]:
                can = {"status": C, "why": "both models fully generated and judged; 2x2 from cache + verdict store"}
            else:
                can = {"status": P, "why": "missing responses or verdicts for this pair: rerun external_canonical.sh"}
            cells.append({**base, "arm": "canonical", **can})
            cells.append({**base, "arm": "recycle", **recycle_cell(root, corpus, old, new, p)})
            cells.append({**base, "arm": "rpe", **rpe_cell(root, corpus, old, new, p)})
    arcs = {}
    for corpus in CORPORA:
        files = sorted(str(x.relative_to(root)) for x in (root / RUN_ROOT).glob(f"style_arcs_{corpus}_*.html"))
        mans = list((root / RUN_ROOT).glob(f"external__{corpus}__*/manifest_*.jsonl"))
        arcs[corpus] = {"status": C if files else (P if mans else NA),
                        "why": "drawn" if files else ("frozen manifests exist; external_final.sh draws them" if mans
                                                       else "no frozen manifest (no eligible recycle run yet)"),
                        "files": files}
    return {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "prereg": PREREG,
            "designation": "exploratory throughout; raw judge output (qwen2.5:14b rubric); never pooled",
            "report": str(Path(rep_path).relative_to(root)) if rep_path else None,
            "recycle_floor": RECYCLE_FLOOR, "confirmatory_floor": CONFIRMATORY_FLOOR,
            "data": data, "canonical_models": {k: v.get("models") for k, v in ((rep or {}).get("corpora") or {}).items()},
            "cells": cells, "style_arcs": arcs,
            "static": [{"arm": a, "status": s, "why": w} for a, s, w in STATIC]}


def markdown(s: dict) -> str:
    def md(x):                                   # a literal | (as in |P^-|) would split a table cell
        return str(x).replace("|", "\\|")
    L = [f"# External corpora on every model: completion ({s['generated']})", "", f"_{s['designation']}_", "",
         "| corpus | data | evidence |", "|---|---|---|"]
    L += [f"| {c} | {d['status']} | {md(d['why'])} |" for c, d in s["data"].items()]
    L += ["", "| corpus | pair | group | arm | P⁻ | status | evidence |", "|---|---|---|---|---|---|---|"]
    for c in s["cells"]:
        L.append(f"| {c['corpus']} | {c['old']} → {c['new']} | {c['group']} | {c['arm']} | {c.get('pminus', '')} "
                 f"| {c['status']} | {md(c['why'])} |")
    L += ["", "| arm | status | evidence |", "|---|---|---|"]
    L += [f"| style arcs · {c} | {a['status']} | {md(', '.join(a['files']) or a['why'])} |" for c, a in s["style_arcs"].items()]
    L += [f"| {md(x['arm'])} | {x['status']} | {md(x['why'])} |" for x in s["static"]]
    return "\n".join(L) + "\n"


# =============================================================================
# CLI
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("configs", "specs"):
        sp = sub.add_parser(name)
        sp.add_argument("--check", action="store_true", help="write nothing; exit 1 on any difference")
    sd = sub.add_parser("data")
    sd.add_argument("--build", action="store_true", help="run the corpus's builder when its CSV is absent")
    sr = sub.add_parser("report")
    sr.add_argument("--config", default=BASE_CONFIG)
    sr.add_argument("--cache", default=CANONICAL_CACHE)
    sr.add_argument("--corpora", nargs="*", default=list(SELECTED))
    sr.add_argument("--out", default=None)
    sl = sub.add_parser("list")
    g = sl.add_mutually_exclusive_group(required=True)
    g.add_argument("--pairs", action="store_true", help="old:new,... for rpe_lineage.py")
    g.add_argument("--specs", action="store_true")
    sl.add_argument("--group", choices=GROUPS, default=None)
    sl.add_argument("--eligible", action="store_true", help="only specs whose judged canonical |P^-| >= floor")
    sl.add_argument("--corpus", choices=list(CORPORA), default=None, help="with --pairs --complete")
    sl.add_argument("--complete", action="store_true",
                    help="with --pairs: only pairs fully generated and judged on --corpus in the report")
    sl.add_argument("--report", default=None, help="canonical report (default: the newest)")
    sm = sub.add_parser("matrix")
    sm.add_argument("--report", default=None)
    sm.add_argument("--out", default=None, help="JSON path; a .md beside it. Default: <run root>/matrix_<stamp>.json")
    args = ap.parse_args(argv)
    root = Path(args.root)

    if args.cmd in ("configs", "specs"):
        problems = (configs if args.cmd == "configs" else specs)(root, args.check)
        for x in problems:
            print("DIFF", x)
        print(f"{args.cmd}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
        return 1 if problems else 0
    if args.cmd == "data":
        res = data_check(root, args.build, SELECTED)
        for c, d in res.items():
            print(f"{c:14s} {d['status']}: {d['why']}")
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = root / RUN_ROOT / f"data_{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out.relative_to(root)}")
        return 0 if all(d["status"] == C for d in res.values()) else 1
    if args.cmd == "report":
        rep = report(root, args.config, args.cache, args.corpora)
        for corpus, r in rep["corpora"].items():
            if "missing" in r:
                print(f"\n== {corpus}: {r['missing']} absent")
                continue
            print(f"\n== {corpus} ({r['n_behaviours']} behaviours) ==")
            for m, s in r["models"].items():
                rate = s["complied"] / s["judged"] if s["judged"] else float("nan")
                print(f"  {m:16s} judged {s['judged']:4d}  complied {s['complied']:4d} ({rate:.3f})"
                      f"  missing resp {s['missing_responses']}  missing verdicts {s['missing_verdicts']}")
            for pair, c in r["pairs"].items():
                print(f"  {pair:30s} P^- {c['pminus']:3d}  persistent {c['persistent']:3d}  neg_flip "
                      f"{c['neg_flip']:3d}  both_safe {c['both_safe']:3d}{'' if c['complete'] else '  (incomplete)'}")
        out = Path(args.out) if args.out else root / RUN_ROOT / f"canonical_{_dt.datetime.now():%Y%m%d_%H%M}.json"
        if out.exists():
            raise SystemExit(f"{out} exists -- refusing to overwrite")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"\nwrote {out}")
        return 0
    if args.cmd == "list":
        if args.pairs:
            keep = [(o, n) for o, n, g, _ in PAIRS if args.group in (None, g)]
            if args.complete:
                if not args.corpus:
                    raise SystemExit("--complete needs --corpus")
                rep = args.report or latest_report(root)
                if not rep:
                    raise SystemExit(f"no {RUN_ROOT}/canonical_*.json: run the canonical job first")
                done = _pairs_of(rep, args.corpus)
                keep = [(o, n) for o, n in keep if (done.get(f"{o}->{n}") or {}).get("complete")]
            print(",".join(f"{o}:{n}" for o, n in keep))
        elif args.eligible:
            print("\n".join(eligible_specs(root, args.group, args.report)))
        else:
            print("\n".join(spec_path(c, o, n) for c in SELECTED for o, n, g, _ in PAIRS if args.group in (None, g)))
        return 0
    if args.cmd == "matrix":
        s = collect(root, args.report)
        out = Path(args.out) if args.out else root / RUN_ROOT / f"matrix_{_dt.datetime.now():%Y%m%d_%H%M}.json"
        if out.exists():
            raise SystemExit(f"{out} exists -- refusing to overwrite")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
        out.with_suffix(".md").write_text(markdown(s), encoding="utf-8")
        print(markdown(s))
        print(f"wrote {out} and {out.with_suffix('.md')}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
