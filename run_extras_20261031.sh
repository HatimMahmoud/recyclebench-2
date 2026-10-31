#!/usr/bin/env bash
# run_extras_20261031.sh -- the 8 h night of 2026-10-31: over-refusal, cross-family spot-check, contamination.
#
#   bash run_extras_20261031.sh            # preflight, then detaches (nohup + caffeinate); log below
#
# Registered in PREREG_OVERREFUSAL_20261031.md, PREREG_XFAMILY_MISTRAL_20261031.md, PREREG_CONTAMINATION_20261031.md.
# Order = thesis value; every step is resumable and stops starting new work at STOP_EPOCH (default launch + 8 h).
#   1  over-refusal, 7-8B chain (llama3-8b, llama31-8b, llama2-7b)          ~2.5 h
#   2  Mistral v0.2 -> v0.3 screen on SORRY-Bench (k = 1, judged)           ~1.5 h
#   3  Mistral recycle (run-real on the judged P^-, only if |P^-| >= 10)    ~1.5 h  + verdict replay
#   4  contamination probe (llama3-8b, llama31-8b, llama2-7b)               ~1.2 h
#   5  over-refusal, remaining models (13b, 1b, 3b, both Mistrals)          whatever time is left
# Own caches and run dirs only; nothing existing is written (results/summary_real.json is backed up first).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
STOP_EPOCH="${STOP_EPOCH:-$(( $(date +%s) + 8 * 3600 ))}"
LOG=logs/extras_${STAMP}.log
XDIR=results/runs/xfamily_mistral_${STAMP}

if [ -z "${EXTRAS_DETACHED:-}" ]; then
  mkdir -p logs
  pmset -g batt | grep -q "AC Power" || echo "WARNING: not on AC power -- plug in (caffeinate cannot hold the Mac awake on battery)"
  curl -sf http://localhost:11434/api/tags > /dev/null || { echo "Ollama is not answering on :11434"; exit 1; }
  for m in llama3:8b llama3.1:8b llama2:7b qwen2.5:14b qwen2.5:3b mistral:7b-instruct-v0.2-q4_K_M mistral:7b-instruct-v0.3-q4_K_M; do
    ollama list | awk '{print $1}' | grep -qx "$m" || { echo "missing model $m (ollama pull $m)"; exit 1; }
  done
  [ -f data/xstest.csv ] || { echo "missing data/xstest.csv"; exit 1; }
  echo "launching; stop starting new work at $(date -r "$STOP_EPOCH" +%H:%M); log: $LOG"
  EXTRAS_DETACHED=1 STAMP="$STAMP" STOP_EPOCH="$STOP_EPOCH" nohup caffeinate -is bash "$0" >> "$LOG" 2>&1 &
  exit 0
fi

left() { echo $(( (STOP_EPOCH - $(date +%s)) / 60 )); }
step() { echo; echo "=== $(date '+%F %T') · $1 · $(left) min left ==="; }
have_time() { [ "$(left)" -gt "${1:-5}" ] || { echo "out of time, skipping: $2"; return 1; }; }

step "1 over-refusal, 7-8B chain"
have_time 5 "1" && python3 tools/overrefusal.py --phase all --minutes "$(left)" \
    --extra-config configs/xfamily_mistral.yaml --models llama3-8b llama31-8b llama2-7b \
    --out results/runs/overrefusal_xstest_${STAMP}_chain

step "2 Mistral screen"
have_time 5 "2" && python3 tools/xfamily_screen.py --phase all --minutes "$(left)" --out "$XDIR"

step "3 Mistral recycle"
if [ -f "$XDIR/pminus_sorrybench.json" ]; then
  PM=$(python3 -c "import json; d=json.load(open('$XDIR/pminus_sorrybench.json')); print(d['p_minus'] if d['missing_verdicts'] == 0 else -1)")
  echo "judged |P^-| = $PM (-1 = screen incomplete)"
  if [ "$PM" -ge 10 ] && have_time 60 "3"; then
    [ -f results/summary_real.json ] && cp -p results/summary_real.json "results/summary_real.json.bak_before_xfamily_${STAMP}"
    # run-real has no wall-clock budget: give it until STOP_EPOCH, then stop it (resumable from cache + manifest)
    python3 -m recyclebench run-real --config configs/xfamily_mistral.yaml \
        --behaviours "$XDIR/sorrybench_p_minus.csv" --cache cache/xfamily_mistral.jsonl \
        --manifest "$XDIR/manifest.jsonl" &
    RR=$!
    while kill -0 $RR 2>/dev/null; do
      [ "$(left)" -le 0 ] && { echo "stop time: stopping run-real (resumable)"; kill -TERM $RR; sleep 30; break; }
      sleep 60
    done
    wait $RR && cp -p results/summary_real.json "$XDIR/summary_real.json"
    have_time 5 "3 verdict replay" && python3 tools/judge_lineage.py --config configs/xfamily_mistral.yaml \
        --cache cache/xfamily_mistral.jsonl --behaviours "$XDIR/sorrybench_p_minus.csv" \
        --manifest "$XDIR/manifest.jsonl" --out cache/judge_xfamily_mistral.jsonl --scope all \
        --stop-at "$(date -r "$STOP_EPOCH" +%H:%M)"
  elif [ "$PM" -ge 0 ] && [ "$PM" -lt 10 ]; then
    echo "|P^-| < 10: recycle not run (registered rule)"
  fi
fi

step "4 contamination probe"
have_time 5 "4" && python3 tools/contamination_probe.py --phase all --minutes "$(left)" \
    --models llama3-8b llama31-8b llama2-7b --out results/runs/contamination_${STAMP}

step "5 over-refusal, remaining models"
have_time 10 "5" && python3 tools/overrefusal.py --phase all --minutes "$(left)" \
    --extra-config configs/xfamily_mistral.yaml \
    --models llama2-13b llama32-1b llama32-3b mistral7b-v02 mistral7b-v03 \
    --out results/runs/overrefusal_xstest_${STAMP}_rest

step "final scoring (no model calls)"
python3 tools/overrefusal.py --phase score --extra-config configs/xfamily_mistral.yaml \
    --models llama2-7b llama2-13b llama3-8b llama31-8b llama32-1b llama32-3b mistral7b-v02 mistral7b-v03 \
    --out results/runs/overrefusal_xstest_${STAMP}_all
python3 tools/contamination_probe.py --phase score --models llama3-8b llama31-8b llama2-7b \
    --out results/runs/contamination_${STAMP}_final
step "done"
