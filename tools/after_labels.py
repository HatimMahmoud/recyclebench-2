#!/usr/bin/env python3
"""tools/after_labels.py -- everything that follows once the human audit sheets are labelled (no model calls).

  python3 tools/after_labels.py          # plan only: how far each sheet is, and what --run would do
  python3 tools/after_labels.py --run    # score, apply Amendment 2 §B4.3, recompute, rebuild -- every output new and dated

Steps, in order (a step whose input is missing is skipped and says so):
  1  progress of the five registered sheets (Amendment 2 §B4.1–B4.5)
  2  score each complete sheet (dated *_result_<stamp>.json beside it); the blind re-label (§B4.3) also when partial,
     as registered (partial agreement is reported)
  3  only if the blind re-label is complete: the blind calibration becomes primary (§B4.3) -- write its corrected twins
     (tools/primary_calibration.py apply) and re-score the two stored checks that carry a corrected value (quantisation,
     loosened gate) into new folders
  4  C4 table, exploratory analysis, numbers file
  5  dashboards: master, lineage, pair pages
Every tool it calls refuses to overwrite; a second --run in the same minute stops at the first collision. The prose in
Amendment 2 and THESIS_KB is written afterwards, by hand, from the printed results.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import primary_calibration as pc                                    # noqa: E402

PY = sys.executable
# (registration, sheet, label column, score command or None for the blind re-label)
SHEETS = [
    ("§B4.1 gate false rejection", "audit/gate_rejection/rejection_sheet.csv", "human_intent_preserved",
     ["tools/gate_rejection_audit.py", "score", "--dir", "audit/gate_rejection"]),
    ("§B4.2 judge on benign prompts", "audit/overrefusal/benign_sheet.csv", "human_fulfilled",
     ["tools/overrefusal_judge_audit.py", "score", "--dir", "audit/overrefusal"]),
    ("§B4.3 blind re-label of the calibration sheet", pc.BLIND, "human_fulfilled", ["tools/blind_relabel_score.py"]),
    ("§B4.4 blind re-check of the gate audit", "audit/gate_recheck/recheck_sheet.csv", "human_intent_preserved",
     ["tools/gate_recheck.py", "score", "--dir", "audit/gate_recheck"]),
    ("§B4.5 class-stratified attack audit", "audit/attack_classes/attack_sheet.csv", "human_fulfilled",
     ["tools/attack_audit.py", "score", "--dir", "audit/attack_classes"]),
]


def progress(sheet: str, col: str):
    """(rows done, rows); on the attack sheet a blank label with a note is a deliberate "cannot tell" (§B4.5)"""
    p = ROOT / sheet
    if not p.exists():
        return None
    with p.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    noted = sheet.endswith("attack_sheet.csv")
    return sum((r.get(col) or "").strip() in ("0", "1") or (noted and bool((r.get("note") or "").strip()))
               for r in rows), len(rows)


def plan() -> list:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    steps = []
    for name, sheet, col, cmd in SHEETS:
        pr = progress(sheet, col)
        if pr is None:
            print(f"  {name}: no sheet")
            continue
        k, n = pr
        partial_ok = sheet == pc.BLIND
        state = "complete" if k == n else f"{k}/{n} labelled"
        print(f"  {name}: {state}")
        if k == n or (partial_ok and k):
            steps.append((f"score {name}", cmd))
    st = pc.status()
    print(f"  primary calibration: {st['primary']} ({st['bundles_complete']}/{st['bundles']} bundles of the blind re-label)")
    if st["primary"] == "blind":
        twin = lambda rel: ROOT / Path(rel).with_name(Path(rel).name.replace(pc.REG_SUFFIX, pc.BLIND_SUFFIX))
        if not all(twin(rel).exists() for rel in pc.CORRECTED.values()):
            steps.append(("§B4.3: write the blind-calibration twins", ["tools/primary_calibration.py", "apply"]))
        steps += [("re-score the quantisation check under the primary calibration",
                   ["tools/quant_check.py", "--phase", "score", "--out", f"results/runs/quant_check_{stamp}_blindcal"]),
                  ("re-score the loosened gate under the primary calibration",
                   ["tools/gate_loosen.py", "--phase", "score", "--out", f"results/runs/gate_loosen_{stamp}_blindcal"])]
    steps += [("C4 table (naive vs corrected)", ["tools/naive_vs_corrected.py"]),
              ("exploratory analysis", ["tools/thesis_analysis.py"]),
              ("numbers file", ["tools/thesis_numbers.py"]),
              ("master dashboard", ["tools/master_dashboard.py"]
               + (["--attack-70b-responses", str(Path.home() / "Downloads/all_local_results_llama3_1_70b 1.jsonl")]
                  if (Path.home() / "Downloads/all_local_results_llama3_1_70b 1.jsonl").exists() else [])),
              ("lineage dashboard", ["tools/lineage_dashboard.py"]),
              ("pair dashboards", ["tools/pair_dashboard.py"])]
    return steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)
    print("sheets:")
    steps = plan()
    print("\nsteps:" if steps else "\nnothing to do")
    for i, (what, cmd) in enumerate(steps, 1):
        print(f"  {i}. {what}: python3 {' '.join(cmd)}")
    if not args.run:
        print("\nplan only; add --run to execute")
        return 0
    env = dict(os.environ, PYTHONPATH="src", TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_PROGRESS_BARS="1")
    for i, (what, cmd) in enumerate(steps, 1):
        print(f"\n=== {i}/{len(steps)} {what}", flush=True)
        r = subprocess.run([PY, *cmd], cwd=ROOT, env=env)
        if r.returncode != 0:
            print(f"stopped: step {i} exited {r.returncode}; nothing after it ran")
            return r.returncode
    print("\ndone. Next by hand: write the results into prereg_amendment_2_DRAFT.md (§B2/§B4) and THESIS_KB.md §7.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
