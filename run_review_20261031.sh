#!/usr/bin/env bash
# run_review_20261031.sh -- the model jobs that answer the external review of 2026-10-31, run one at a time from a queue.
#
#   bash run_review_20261031.sh              # preflight, then detaches (nohup + caffeinate); log: logs/review_20261031.log
#   bash run_review_20261031.sh --list       # queue state
#
# Jobs are files logs/queue_20261031/NN_name.sh, run in name order (lowest pending first) and never two at once; a job
# can be added while the runner is going. Each is marked .done (exit code inside) or .failed. The runner stops starting
# jobs at STOP_EPOCH (default launch + 5 h 45 min) and exits when nothing is pending and logs/queue_20261031/END exists,
# or at the deadline. Every job is resumable; each writes only its own store and dated run directory.
set -u
cd "$(dirname "$0")"
Q=logs/queue_20261031
LOG=logs/review_20261031.log
mkdir -p "$Q" logs
export PYTHONPATH=src TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h

if [ "${1:-}" = "--list" ]; then
  for j in "$Q"/*.sh; do
    [ -e "$j" ] || continue
    s=pending; [ -e "$j.running" ] && s=running; [ -e "$j.done" ] && s="done (exit $(cat "$j.done"))"; [ -e "$j.failed" ] && s="FAILED (exit $(cat "$j.failed"))"
    printf "%-40s %s\n" "$(basename "$j")" "$s"
  done
  [ -e "$Q/END" ] && echo "(END set: the runner exits when the queue is empty)"
  exit 0
fi

if [ -z "${REVIEW_DETACHED:-}" ]; then
  pmset -g batt | grep -q "AC Power" || echo "WARNING: not on AC power -- caffeinate cannot hold the Mac awake on battery"
  curl -sf http://localhost:11434/api/tags > /dev/null || { echo "Ollama is not answering on :11434"; exit 1; }
  STOP_EPOCH="${STOP_EPOCH:-$(( $(date +%s) + 345 * 60 ))}"
  echo "launching; no new job after $(date -r "$STOP_EPOCH" +%H:%M); log: $LOG"
  REVIEW_DETACHED=1 STOP_EPOCH="$STOP_EPOCH" nohup caffeinate -is bash "$0" >> "$LOG" 2>&1 &
  exit 0
fi

left() { echo $(( (STOP_EPOCH - $(date +%s)) / 60 )); }
export STOP_EPOCH
echo "=== $(date '+%F %T') runner up · $(left) min budget ==="
while :; do
  next=""
  for j in "$Q"/*.sh; do
    [ -e "$j" ] || continue
    [ -e "$j.done" ] || [ -e "$j.failed" ] || [ -e "$j.running" ] && continue
    next="$j"; break
  done
  if [ -z "$next" ]; then
    { [ -e "$Q/END" ] || [ "$(left)" -le 0 ]; } && { echo "=== $(date '+%F %T') queue empty, runner exits ==="; exit 0; }
    sleep 30; continue
  fi
  if [ "$(left)" -le 5 ]; then
    echo "=== $(date '+%F %T') deadline: not starting $(basename "$next") ==="; exit 0
  fi
  echo; echo "=== $(date '+%F %T') start $(basename "$next") · $(left) min left ==="
  touch "$next.running"
  MINUTES_LEFT="$(left)" bash "$next"
  rc=$?
  rm -f "$next.running"
  if [ $rc -eq 0 ]; then echo $rc > "$next.done"; else echo $rc > "$next.failed"; fi
  echo "=== $(date '+%F %T') end $(basename "$next") · exit $rc ==="
done
