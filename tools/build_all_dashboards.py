#!/usr/bin/env python3
"""
tools/build_all_dashboards.py — batch-generate one dashboard per run.

Drives tools/make_dashboard.py (so every output has the same scroll/tabs toggle and the
01-07 section layout as the main dashboard). For each run under results/runs/* it:
  * infers the model pair from the directory name  (…_<old>_to_<new>_…)
  * finds the benchmarks from  manifest_<bench>.jsonl
  * pairs each with its  data/<bench>.csv
  * wires in any  corrected_*.json / rpe_manifest_*.jsonl  present in the run
  * adds the attack-transfer arm when the new model is llama31-8b
  * writes a DATED  dashboard_<YYYYmmdd_HHMM>.html  into the run folder (never overwrites)

Run from anywhere (it cd's to the repo root):
  python3 tools/build_all_dashboards.py                 # build all
  python3 tools/build_all_dashboards.py --dry-run       # print the plan only, build nothing
  python3 tools/build_all_dashboards.py --cache cache/llama_gen.jsonl --config configs/recycle_3_31.yaml

A run whose directory name has no parseable "<old>_to_<new>" is skipped with a note — add it
to PAIR_OVERRIDE below (e.g. the pminus-only run) and re-run.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import primary_calibration as pc                # noqa: E402

MAKE = "tools/make_dashboard.py"

# directory-name token -> cache/config model key
ALIAS = {
    "llama3-8b": "llama3-8b", "llama31-8b": "llama31-8b",
    "llama32-1b": "llama32-1b", "llama32-3b": "llama32-3b",
    "1b": "llama32-1b", "3b": "llama32-3b",
    "llama2-7b": "llama2-7b", "llama2-13b": "llama2-13b",
}
# runs whose folder name has no "<old>_to_<new>" — map them by hand here if you want them built
PAIR_OVERRIDE = {
    "recycle_3_31_sorrybench": ("llama3-8b", "llama31-8b"),   # the calibrated ANCHOR
    # "20260805_183352_RECYCLE_pminus_only": ("llama3-8b", "llama31-8b"),
}
# Runs whose manifest is a whole `lineage:` (plain manifest.jsonl, no manifest_<bench>.jsonl).
# These are DELEGATED to tools/lineage_dashboard.py, which renders the whole lineage as one native
# page; fanning them out through make_dashboard is the per-rung iframe approach that page replaced
# (handoff §14.1-§14.3). A plain manifest.jsonl is NOT itself the signal — confirm_heldout_3_31 and
# xfamily_mistral_* also have one and are ordinary two-model runs — so lineage runs are listed.
LINEAGE_RUNS = {
    "lineage_sorrybench_20261020": {
        "config": "configs/real_llama_lineage_all.yaml",
        "cache": "cache/llama_gen.jsonl",
        "behaviours": "data/sorrybench.csv",
        "verdicts": "cache/judge_lineage_sorrybench.jsonl",
    },
}
# Runs with no manifest_<bench>.jsonl that are NOT lineage runs: the current family for these is
# pair_dashboard.py / master_dashboard.py (make_dashboard's layout is superseded), so say so once
# instead of skipping in silence.
ELSEWHERE = {
    "confirm_heldout_3_31": "rendered by tools/pair_dashboard.py (and master §recycle); "
                            "behaviours data/heldout_jbb_guided.csv",
    "xfamily_mistral_20261031_0126": "rendered by tools/controls_section.py, on the master "
                                     "and lineage pages",
    "20260805_183352_RECYCLE_pminus_only": "a pminus-only run: no recycling manifest to render",
    # EXPERIMENTAL -- ScienceSafetyData pilot, parked for the cluster; drop this entry when the
    # pilot is deleted before submission (cleanup list in CLAUDE.md).
    "sciencesafety_20261101": "tools/sciencesafety_dashboard.py (standalone, deliberately isolated: "
                              "this corpus is not in study_data and pools into nothing)",
}
# runs whose RPE manifest lives OUTSIDE the run folder — point at it explicitly here
RPE_OVERRIDE = {
    "recycle_3_31_sorrybench":
        "results/runs/rpe_3_31_sorrybench/rpe_panel.html/rpe_manifest_sorrybench.jsonl",
}
# attack-transfer result files (added only when the NEW model matches)
ATTACK_8B = "results/runs/attack_transfer/llama31-8b_judged_original.json"
ATTACK_70B = "results/runs/attack_transfer/llama31-70b_judged.json"


def norm(tok: str) -> str:
    return ALIAS.get(tok.lower(), tok.lower())


def parse_pair(name: str):
    if name in PAIR_OVERRIDE:
        return PAIR_OVERRIDE[name]
    m = re.search(r"([a-z0-9.]+(?:-[a-z0-9.]+)*)_to_([a-z0-9.]+(?:-[a-z0-9.]+)*)", name.lower())
    return (norm(m.group(1)), norm(m.group(2))) if m else None


def benchmarks_in(run_dir: Path):
    """[(bench, manifest_path)] from manifest_<bench>.jsonl (ignores rpe_manifest_*)."""
    out = []
    for mf in sorted(run_dir.glob("manifest_*.jsonl")):
        out.append((mf.stem.replace("manifest_", ""), str(mf)))
    return out


def build_lineage(rd: Path, spec: dict, stamp: str, dry_run: bool) -> int:
    """Hand a whole-lineage run to tools/lineage_dashboard.py. Returns 1 if a page was written."""
    out = rd / f"dashboard_lineage_{stamp}.html"
    cmd = [sys.executable, "tools/lineage_dashboard.py",
           "--config", spec["config"], "--cache", spec["cache"],
           "--behaviours", spec["behaviours"], "--run-dir", str(rd), "--out", str(out)]
    if spec.get("verdicts"):
        cmd += ["--verdicts", spec["verdicts"]]
    print(f"{'PLAN ' if dry_run else 'BUILD'} {rd.name}: whole lineage -> lineage_dashboard.py")
    if dry_run:
        print("      " + " ".join(cmd))
        return 0
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"      -> {out}")
        return 1
    print(f"      FAILED (rc={r.returncode}):")
    print("      " + (r.stderr or r.stdout).strip().replace("\n", "\n      ")[-1200:])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/recycle_3_31.yaml")
    ap.add_argument("--cache", default="cache/llama_gen.jsonl")
    ap.add_argument("--runs-glob", default="results/runs/*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    if not Path(MAKE).exists():
        sys.exit(f"{MAKE} not found — run this from inside the repo (expected at {ROOT/MAKE}).")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dirs = [Path(p) for p in sorted(glob.glob(args.runs_glob))
                if Path(p).is_dir() and not Path(p).is_symlink()]

    built = skipped = planned = 0
    for rd in run_dirs:
        name = rd.name
        benches = benchmarks_in(rd)
        if not benches and name in LINEAGE_RUNS:
            built += build_lineage(rd, LINEAGE_RUNS[name], stamp, args.dry_run)
            planned += int(args.dry_run)
            continue
        if not benches:
            if name in ELSEWHERE:
                print(f"SKIP  {name}: {ELSEWHERE[name]}")
                skipped += 1
            elif (rd / "manifest.jsonl").exists():
                print(f"SKIP  {name}: has a bare manifest.jsonl — add it to LINEAGE_RUNS "
                      f"(whole lineage) or ELSEWHERE (rendered by another tool)")
                skipped += 1
            continue  # not a recycle run (attack_transfer/, rpe-only, etc.)

        pair = parse_pair(name)
        if not pair:
            # ELSEWHERE first: a run rendered by another tool should not invite a PAIR_OVERRIDE
            print(f"SKIP  {name}: " + (ELSEWHERE[name] if name in ELSEWHERE
                                       else "can't infer model pair — add it to PAIR_OVERRIDE"))
            skipped += 1
            continue
        old, new = pair

        merges, missing = [], []
        for bench, mf in benches:
            csv = Path("data") / f"{bench}.csv"
            (merges if csv.exists() else missing).append(
                (bench, mf, str(csv)) if csv.exists() else bench)
        if not merges:
            print(f"SKIP  {name}: no data/<bench>.csv for {benches and [b for b, _ in benches]}")
            skipped += 1
            continue

        out = rd / f"dashboard_{stamp}.html"
        cmd = [sys.executable, MAKE, "--config", args.config,
               "--old", old, "--new", new, "--cache", args.cache,
               "--behaviours", merges[0][2], "--out", str(out)]
        if len(merges) == 1:
            cmd += ["--manifest", merges[0][1]]
        else:
            cmd += ["--merge"] + [f"{mf}:{csv}" for _, mf, csv in merges]

        # best-effort extras found inside the run folder
        corr = sorted(c for c in glob.glob(str(rd / "corrected_*.json")) if "_blind_" not in c)
        if corr:      # newest name sorts last: *_recal_n60.json supersedes the pre-repair file; the primary
            cmd += ["--corrected", str(pc.resolve(corr[-1]))]      # calibration's twin once §B4.3 applies
        rpe = sorted(glob.glob(str(rd / "**/rpe_manifest_*.jsonl"), recursive=True))
        rpe_path = rpe[0] if rpe else None
        if name in RPE_OVERRIDE and Path(RPE_OVERRIDE[name]).exists():
            rpe_path = RPE_OVERRIDE[name]
        if rpe_path:
            cmd += ["--rpe-manifest", rpe_path]
        if new == "llama31-8b":
            if Path(ATTACK_8B).exists():
                cmd += ["--attack-transfer", ATTACK_8B]
            if Path(ATTACK_70B).exists():
                cmd += ["--attack-transfer-70b", ATTACK_70B]

        blist = "+".join(b for b, _, _ in merges)
        cmd += ["--meta", f"{name} · {old}→{new} · {blist}"]

        tag = "PLAN " if args.dry_run else "BUILD"
        note = f"  (no CSV: {missing})" if missing else ""
        print(f"{tag} {name}: {old}->{new}  [{blist}]{note}")
        if args.dry_run:
            print("      " + " ".join(cmd))
            planned += 1
            continue

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"      -> {out}")
            built += 1
        else:
            print(f"      FAILED (rc={r.returncode}):")
            print("      " + (r.stderr or r.stdout).strip().replace("\n", "\n      ")[-1200:])
            skipped += 1

    print(f"\n{'planned: ' + str(planned) if args.dry_run else 'built: ' + str(built)}   skipped: {skipped}")
    if not args.dry_run and built:
        print("each dashboard is dated, so re-running adds a new file rather than overwriting.")


if __name__ == "__main__":
    main()