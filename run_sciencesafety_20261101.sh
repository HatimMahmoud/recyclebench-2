#!/usr/bin/env bash
# EXPERIMENTAL -- ScienceSafetyData pilot, parked for the cluster. Safe to delete before submission; see the cleanup list in CLAUDE.md and TODO_HUMAN A10.
# =====================================================================
# ScienceSafetyData on the anchor pair (llama3-8b -> llama31-8b), ISOLATED.
# Registered in PREREG_SCIENCESAFETY_20261101.md. Exploratory.
#
#   bash run_sciencesafety_20261101.sh            # run the queue
#   bash run_sciencesafety_20261101.sh --list     # show stage status, run nothing
#
# Sequential by construction: one writer on the cache at a time (two corrupt it).
# Every stage is resumable -- the response cache is append-only and keyed on
# (model, prompt, seed), so re-running the script continues where it stopped.
#
# ISOLATION: its own cache + run dir; the two fixed-path scratch files the CLI
# overwrites are backed up at stage 00 and restored at stage 90, so the existing
# results/pminus_pilot.json and results/summary_real.json are left as they were.
# Nothing here is registered in tools/study_data.py or on any dashboard.
# =====================================================================
set -u

export PYTHONPATH=src
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OLLAMA_MAX_LOADED_MODELS=3      # llama3:8b + llama3.1:8b + qwen2.5:14b ~= 18 GB
export OLLAMA_KEEP_ALIVE=1h            # or judge/target alternation reloads ~10 GB per call

CFG=configs/sciencesafety_3_31.yaml
BEH=data/sciencesafety.csv
BENCH=sciencesafety
CACHE=cache/sciencesafety_20261101.jsonl
RUN=results/runs/sciencesafety_20261101
OLD=llama3-8b
NEW=llama31-8b
STOP_AT=08:30                          # hard stop: no stage starts after this
MIN_PMINUS_FOR_RECYCLE=5               # below this, recycling is not worth the calls

LOG="$RUN/run.log"
mkdir -p "$RUN/cells" "$RUN/stages" "$(dirname "$CACHE")"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

past_stop() {
  local now; now=$(date +%s)
  local stop; stop=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) $STOP_AT" +%s 2>/dev/null)
  # the stop time is tomorrow if it has already passed by more than 12 h (we start near midnight)
  [ -n "$stop" ] && [ "$stop" -lt $((now - 43200)) ] && stop=$((stop + 86400))
  [ -n "$stop" ] && [ "$now" -ge "$stop" ]
}

done_marker() { echo "$RUN/stages/$1.done"; }
is_done()     { [ -f "$(done_marker "$1")" ]; }
mark_done()   { date "+%Y-%m-%d %H:%M:%S" > "$(done_marker "$1")"; }

if [ "${1:-}" = "--list" ]; then
  echo "run dir : $RUN"
  echo "cache   : $CACHE$([ -f "$CACHE" ] && echo "  ($(wc -l < "$CACHE" | tr -d ' ') lines)")"
  for s in 00_setup 10_screen 20_partition 30_recycle 40_audit_sheet 90_restore; do
    if is_done "$s"; then echo "  $s  done ($(cat "$(done_marker "$s")"))"
    else echo "  $s  pending"; fi
  done
  [ -f "$RUN/pminus_${BENCH}.json" ] && python3 -I -c "
import json;d=json.load(open('$RUN/pminus_${BENCH}.json'))
print('  |P^-| =', d['p_minus'], ' breaks_old =', d['breaks_old'], ' status =', d['status'])"
  exit 0
fi

# ---------------------------------------------------------------------
say "===== ScienceSafetyData queue starting (hard stop $STOP_AT) ====="
say "pair $OLD -> $NEW | config $CFG | cache $CACHE"

# --- 00 setup: back up the fixed-path scratch files -------------------
if ! is_done 00_setup; then
  say "--- 00 setup"
  for f in results/pminus_pilot.json results/summary_real.json; do
    [ -f "$f" ] && cp -p "$f" "$RUN/stages/$(basename "$f").preexisting" \
      && say "    backed up $f"
  done
  {
    echo "run        : ScienceSafetyData (BYOD) on the anchor pair, ISOLATED"
    echo "prereg     : PREREG_SCIENCESAFETY_20261101.md"
    echo "config     : $CFG"
    echo "behaviours : $BEH (240; see data/sciencesafety.csv.provenance.json)"
    echo "pair       : $OLD -> $NEW"
    echo "cache      : $CACHE (isolated)"
    echo "judge      : qwen2.5:14b rubric (the config's judges: block)"
    echo "started    : $(date)"
    echo ""
    echo "NOT registered in tools/study_data.py and on no dashboard, per the prereg's"
    echo "isolation clause. Fold in only if P^- is non-empty, by its own amendment."
    echo ""
    echo "Join any run built on the P^- subset back to the corpus ON PROMPT TEXT:"
    echo "load_jailbreakbench ignores a bid column and renumbers rows."
  } > "$RUN/META.txt"
  mark_done 00_setup
fi

# --- 10 canonical screen: the judged P^- ------------------------------
if is_done 10_screen; then say "--- 10 screen already done, skipping"
elif past_stop; then say "--- 10 screen SKIPPED (past $STOP_AT)"
else
  say "--- 10 canonical screen: 240 behaviours, k=3, judged by qwen2.5:14b"
  caffeinate -is python3 -m recyclebench pilot-pminus \
      --config "$CFG" --old "$OLD" --new "$NEW" \
      --behaviours "$BEH" --cache "$CACHE" 2>&1 | tee "$RUN/pminus_${BENCH}.txt"
  if [ -f results/pminus_pilot.json ]; then
    cp results/pminus_pilot.json "$RUN/pminus_${BENCH}.json"
    say "    snapshot -> $RUN/pminus_${BENCH}.json"
    mark_done 10_screen
  else
    say "    !! pilot produced no results/pminus_pilot.json -- stopping"
    exit 1
  fi
fi

# the judged P^- subset (NOT the offline-rule dump)
PM=0
if [ -f "$RUN/pminus_${BENCH}.json" ]; then
  python3 tools/pminus_subset.py --pilot "$RUN/pminus_${BENCH}.json" \
      --behaviours "$BEH" --out "$RUN/${BENCH}_p_minus_judged.csv" 2>&1 | tee -a "$LOG"
  PM=$(python3 -I -c "import json;print(json.load(open('$RUN/pminus_${BENCH}.json'))['p_minus'])")
  say "    judged |P^-| = $PM"
fi

# --- 30 recycle: manifest (paraphrase + gate), freeze, re-screen -------
# Deliberately BEFORE the 2x2 partition: RSR is the primary estimand, the negative-flip
# cell is a control. If the window runs out, the primary quantity is the one in hand.
# Resumable: the manifest is frozen to disk and every response is cached, so a kill
# mid-re-screen costs nothing -- re-running reloads the frozen manifest.
if is_done 30_recycle; then say "--- 30 recycle already done, skipping"
elif past_stop; then say "--- 30 recycle SKIPPED (past $STOP_AT)"
elif [ "$PM" -lt "$MIN_PMINUS_FOR_RECYCLE" ]; then
  say "--- 30 recycle SKIPPED: |P^-| = $PM < $MIN_PMINUS_FOR_RECYCLE"
  say "    RSR is undefined/underpowered here; the 2x2 and PBR carry this run."
else
  say "--- 30 recycle on $PM judged fixes (8 paraphrases x 6 classes, gated, then re-screened)"
  caffeinate -is python3 -m recyclebench run-real \
      --config "$CFG" \
      --behaviours "$RUN/${BENCH}_p_minus_judged.csv" \
      --manifest "$RUN/manifest_${BENCH}.jsonl" \
      --cache "$CACHE" 2>&1 | tee "$RUN/runreal_${BENCH}.txt"
  if [ -f results/summary_real.json ]; then
    cp results/summary_real.json "$RUN/summary_${BENCH}.json"
    say "    snapshot -> $RUN/summary_${BENCH}.json"
    mark_done 30_recycle
  else
    say "    !! run-real produced no results/summary_real.json"
  fi
fi

# --- 20 full 2x2 partition (+ the negative-flip cell) -----------------
# A control, so it runs after RSR. Its partition is rebuilt from the cache with the
# OFFLINE refusal rule (pminus_report has no verdict store), so it does NOT override
# stage 10's judged |P^-|; the two are reported side by side, as the standing
# convention requires.
if is_done 20_partition; then say "--- 20 partition already done, skipping"
elif past_stop; then say "--- 20 partition SKIPPED (past $STOP_AT)"
else
  say "--- 20 partition + inverse screen (fills the negative-flip cell)"
  caffeinate -is python3 tools/pminus_report.py \
      --config "$CFG" --old "$OLD" --new "$NEW" --cache "$CACHE" \
      "$BEH" --screen-inverse --dump "$RUN/cells" 2>&1 \
      | tee "$RUN/partition_full.txt"
  say "    NOTE: offline-rule partition; the judged P^- is stage 10's"
  mark_done 20_partition
fi

# --- 40 T1 audit sheet: blind human check of old-model compliance -----
# Zero target calls: replays the frozen manifest out of the response cache.
if is_done 40_audit_sheet; then say "--- 40 audit sheet already done, skipping"
elif [ ! -f "$RUN/manifest_${BENCH}.jsonl" ]; then
  say "--- 40 audit sheet SKIPPED (no manifest; stage 30 did not run)"
else
  say "--- 40 exporting the blind judge sheet (T1: benign-reading compliance)"
  mkdir -p "audit/sciencesafety"
  caffeinate -is python3 tools/judge_export_real.py \
      --config "$CFG" --new-model "$NEW" --behaviours "$RUN/${BENCH}_p_minus_judged.csv" \
      --manifest "$RUN/manifest_${BENCH}.jsonl" --cache "$CACHE" \
      --out audit/sciencesafety/judge_sheet.csv --n-bundles 60 2>&1 | tee "$RUN/audit_export.txt" \
    && mark_done 40_audit_sheet \
    || say "    audit export failed (non-fatal; see $RUN/audit_export.txt)"
fi

# --- 90 restore the fixed-path scratch files --------------------------
say "--- 90 restoring the pre-existing fixed-path scratch files"
for f in pminus_pilot.json summary_real.json; do
  if [ -f "$RUN/stages/$f.preexisting" ]; then
    cp -p "$RUN/stages/$f.preexisting" "results/$f" && say "    restored results/$f"
  fi
done
mark_done 90_restore

{
  echo ""
  echo "finished   : $(date)"
  echo "judged P^- : $PM of 240"
  echo "cache      : $CACHE ($(wc -l < "$CACHE" 2>/dev/null | tr -d ' ') lines)"
} >> "$RUN/META.txt"

say "===== queue finished. |P^-| = $PM. run dir: $RUN ====="
say "Next (NOT run here, each needs its own registration): RPE, the style factorial,"
say "and folding this corpus into study_data -- only if P^- is non-empty."
