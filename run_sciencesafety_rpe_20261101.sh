#!/usr/bin/env bash
# EXPERIMENTAL -- ScienceSafetyData pilot, parked for the cluster. Safe to delete before submission; see the cleanup list in CLAUDE.md and TODO_HUMAN A10.
# =====================================================================
# ScienceSafetyData, follow-on arms: verdict store -> inversions (RPE) -> closed loop.
# Registered in PREREG_SCIENCESAFETY_RPE_20261101.md. Exploratory.
#
#   bash run_sciencesafety_rpe_20261101.sh            # waits for the main queue, then runs
#   bash run_sciencesafety_rpe_20261101.sh --list     # stage status only
#
# A SEPARATE file from run_sciencesafety_20261101.sh on purpose: bash reads a script
# incrementally, so editing a running one can make it execute garbage. This waits for
# that queue to exit before touching the shared cache -- never two writers.
#
# The style factorial (compose v2) is deliberately NOT here: it sources its items from
# the anchor's SORRY-Bench P^- and cannot take another corpus without a code change,
# and it has not been power-checked for this one. See the prereg's last section.
# =====================================================================
set -u

export PYTHONPATH=src
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=1h

CFG=configs/sciencesafety_3_31.yaml
BENCH=sciencesafety
CACHE=cache/sciencesafety_20261101.jsonl
RUN=results/runs/sciencesafety_20261101
OLD=llama3-8b
NEW=llama31-8b
STOP_AT=08:30
MIN_PMINUS=5

PMCSV="$RUN/${BENCH}_p_minus_judged.csv"
MAN="$RUN/manifest_${BENCH}.jsonl"
VERD="cache/judge_${BENCH}_20261101.jsonl"
GSTORE="cache/rpe_gate_${BENCH}_20261101.jsonl"
RPEROOT="$RUN/rpe"
RPEMAN="$RPEROOT/rpe_manifest_${OLD}__to__${NEW}.jsonl"

LOG="$RUN/rpe.log"
mkdir -p "$RUN/stages" "$RPEROOT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

stop_epoch() {
  local s; s=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) $STOP_AT" +%s 2>/dev/null)
  [ -n "$s" ] && [ "$s" -lt $(( $(date +%s) - 43200 )) ] && s=$((s + 86400))
  echo "$s"
}
mins_left() { echo $(( ( $(stop_epoch) - $(date +%s) ) / 60 )); }
past_stop() { [ "$(mins_left)" -le 0 ]; }

done_marker() { echo "$RUN/stages/$1.done"; }
is_done()     { [ -f "$(done_marker "$1")" ]; }
mark_done()   { date "+%Y-%m-%d %H:%M:%S" > "$(done_marker "$1")"; }

if [ "${1:-}" = "--list" ]; then
  echo "follow-on arms for $RUN"
  for s in 45_verdicts 50_inversions 55_rpe_gate 60_closedloop; do
    if is_done "$s"; then echo "  $s  done ($(cat "$(done_marker "$s")"))"
    else echo "  $s  pending"; fi
  done
  echo "  minutes to hard stop: $(mins_left)"
  exit 0
fi

# --- wait for the main queue (one writer on the cache at a time) ------
say "===== follow-on arms: waiting for the main queue to finish ====="
while pgrep -f "run_sciencesafety_20261101.sh" >/dev/null 2>&1; do
  sleep 60
done
say "main queue has exited; $(mins_left) min to the $STOP_AT hard stop"

# --- preconditions ----------------------------------------------------
if [ ! -f "$MAN" ]; then
  say "ABORT: no recycle manifest at $MAN -- stage 30 did not run (|P^-| too small?)."
  say "       The inversion arm needs the frozen manifest and the judged P^- subset."
  exit 0
fi
PM=$(python3 -I -c "
import csv;print(sum(1 for _ in csv.DictReader(open('$PMCSV'))))" 2>/dev/null || echo 0)
say "judged |P^-| = $PM"
if [ "$PM" -lt "$MIN_PMINUS" ]; then
  say "ABORT: |P^-| = $PM < $MIN_PMINUS; every PRR would be undefined or unpowered."
  exit 0
fi

# --- 45 verdict store: replay the config's judge over cached responses -
# No target calls. Needed because run-real stores no verdicts and the offline
# refusal rule disagrees with the qwen judge.
if is_done 45_verdicts; then say "--- 45 verdicts already done, skipping"
elif past_stop; then say "--- 45 verdicts SKIPPED (past $STOP_AT)"
else
  say "--- 45 verdict store -> $VERD (judge replay, zero target calls)"
  python3 tools/judge_lineage.py --config "$CFG" --cache "$CACHE" \
      --behaviours "$PMCSV" --manifest "$MAN" --out "$VERD" --dry-run 2>&1 | tail -5 | tee -a "$LOG"
  caffeinate -is python3 tools/judge_lineage.py --config "$CFG" --cache "$CACHE" \
      --behaviours "$PMCSV" --manifest "$MAN" --out "$VERD" --scope rsr \
      --stop-at "$STOP_AT" 2>&1 | tail -20 | tee -a "$LOG"
  # sum EVERY phase's "remaining: N": a single phase reading 0 does not mean the store is
  # complete, and marking it done early would let stage 50 score a partial store.
  REMAIN=$(python3 tools/judge_lineage.py --config "$CFG" --cache "$CACHE" \
      --behaviours "$PMCSV" --manifest "$MAN" --out "$VERD" --scope rsr --dry-run 2>&1 \
      | grep -oE "remaining: [0-9]+" | grep -oE "[0-9]+" \
      | awk '{s+=$1} END {print (NR ? s : 1)}')
  if [ "${REMAIN:-1}" = "0" ]; then mark_done 45_verdicts; say "    verdict store complete"
  else say "    verdict store INCOMPLETE (${REMAIN:-?} remaining) -- downstream stages will not score"; fi
fi

# --- 50 inversions: reconstruct the request from the OLD model's answer -
if is_done 50_inversions; then say "--- 50 inversions already done, skipping"
elif ! is_done 45_verdicts; then say "--- 50 inversions SKIPPED (verdict store incomplete)"
elif past_stop; then say "--- 50 inversions SKIPPED (past $STOP_AT)"
else
  BUDGET=$(( $(mins_left) * 40 / 100 ))     # at most 40% of what is left
  say "--- 50 inversions (inverter qwen2.5:14b, budget ${BUDGET} min)"
  python3 tools/rpe_lineage.py --config "$CFG" --inverter-config "$CFG" \
      --inverter qwen25-14b --behaviours "$PMCSV" --manifest "$MAN" \
      --cache "$CACHE" --verdicts "$VERD" --pairs "${OLD}:${NEW}" \
      --benchmark "$BENCH" --out-root "$RPEROOT" --dry-run 2>&1 | tail -8 | tee -a "$LOG"
  if caffeinate -is python3 tools/rpe_lineage.py --config "$CFG" --inverter-config "$CFG" \
      --inverter qwen25-14b --behaviours "$PMCSV" --manifest "$MAN" \
      --cache "$CACHE" --verdicts "$VERD" --pairs "${OLD}:${NEW}" \
      --benchmark "$BENCH" --out-root "$RPEROOT" --minutes "$BUDGET" 2>&1 \
      | tail -25 | tee -a "$LOG"; then
    [ -f "$RPEMAN" ] && mark_done 50_inversions && say "    wrote $RPEMAN"
  fi
  [ -f "$RPEMAN" ] || say "    no inversion manifest produced"
fi

# --- 55 gate every inversion candidate -------------------------------
# Without this the closed-loop artifact has empty A_gate/A_both BY CONSTRUCTION
# (the inversion builder writes admitted: false and never runs the gate).
if is_done 55_rpe_gate; then say "--- 55 rpe gate already done, skipping"
elif [ ! -f "$RPEMAN" ]; then say "--- 55 rpe gate SKIPPED (no inversion manifest)"
elif past_stop; then say "--- 55 rpe gate SKIPPED (past $STOP_AT)"
else
  BUDGET=$(( $(mins_left) * 30 / 100 ))
  say "--- 55 gating every inversion candidate -> $GSTORE (budget ${BUDGET} min)"
  caffeinate -is python3 tools/rpe_gate.py --config "$CFG" --manifests "$RPEMAN" \
      --store "$GSTORE" --minutes "$BUDGET" 2>&1 | tail -15 | tee -a "$LOG"
  [ -s "$GSTORE" ] && mark_done 55_rpe_gate
fi

# --- 60 closed loop + scoring (A_gate / A_loop / A_both) --------------
if is_done 60_closedloop; then say "--- 60 closed loop already done, skipping"
elif [ ! -f "$RPEMAN" ]; then say "--- 60 closed loop SKIPPED (no inversion manifest)"
elif past_stop; then say "--- 60 closed loop SKIPPED (past $STOP_AT)"
else
  BUDGET=$(mins_left)
  say "--- 60 closed loop, all phases (budget ${BUDGET} min)"
  GATE_ARG=()
  if [ -s "$GSTORE" ]; then GATE_ARG=(--gate-store "$GSTORE")
  else say "    WARNING: no gate store -- A_gate/A_both will be empty BY CONSTRUCTION, not by rejection"; fi
  caffeinate -is python3 tools/rpe_closedloop.py --phase all --config "$CFG" \
      --rpe-manifest "$RPEMAN" --old "$OLD" --new "$NEW" \
      --cache "$CACHE" --verdicts "$VERD" "${GATE_ARG[@]+"${GATE_ARG[@]}"}" \
      --minutes "$BUDGET" --out "$RPEROOT/closedloop_${BENCH}_gated_$(date +%Y%m%d_%H%M).json" 2>&1 | tail -30 | tee -a "$LOG"
  ls -1 "$RPEROOT"/closedloop_*_gated_*.json >/dev/null 2>&1 && mark_done 60_closedloop
fi

{
  echo ""
  echo "follow-on arms finished: $(date)"
  for s in 45_verdicts 50_inversions 55_rpe_gate 60_closedloop; do
    echo "  $s: $(is_done "$s" && echo done || echo 'not completed')"
  done
} >> "$RUN/META.txt"

say "===== follow-on arms finished ====="
say "A_gate is the only rule comparable to RSR; A_loop is an upper bound and has NO"
say "harmful-ask sheet for this corpus (prereg T1). Join arms on prompt text, never bid."
