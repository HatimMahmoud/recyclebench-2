#!/usr/bin/env bash
# =====================================================================
# Sequential cross-generational screen across ALL <70B models, every benchmark.
#
# Walks the lineage llama2-7b -> ... -> llama32-3b as CONSECUTIVE PAIRS. Ollama holds
# one model at a time; a SHARED cache means each model is screened exactly once across
# all benchmarks (its second appearance in the next pair is a pure cache hit), and every
# pair's 2x2 is then a free cache-diff. Resumable: if it stops, just run it again.
#
#   caffeinate -is  keeps the Mac awake   |   nohup  survives a closed terminal
#
# PREREQS (once):
#   ollama pull llama2:7b llama2:13b llama3:8b llama3.1:8b llama3.2:1b llama3.2:3b qwen2.5:3b
#   python3 tools/prepare_sorrybench.py --out data/sorrybench.csv     # (and/or donotanswer.csv)
#
# Usage (repo root, venv/PYTHONPATH set up):
#   bash run_all_models.sh
# Watch:  tail -f logs/all_models.log
# =====================================================================
set -u
export PYTHONPATH=src

CONFIG="configs/real_llama_lineage_all.yaml"
CACHE="cache/llama_gen.jsonl"                 # SAME shared cache as before -> reuse prior screens

# lineage order (edit to trim). Every model is screened once.
MODELS=(llama2-7b llama2-13b llama3-8b llama31-8b llama32-1b llama32-3b)

# benchmarks to screen each model on (sorrybench + donotanswer are the new ones)
BENCHMARKS=(sorrybench)

mkdir -p cache results logs

preflight() {
  [ -f "$CONFIG" ] || { echo "FATAL: $CONFIG missing"; exit 1; }
  curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 \
    || { echo "FATAL: Ollama not responding on :11434 (run 'ollama serve')"; exit 1; }
  local found=0
  for BM in "${BENCHMARKS[@]}"; do
    if [ -f "data/${BM}.csv" ]; then found=1; else echo "[warn] data/${BM}.csv not found — will skip"; fi
  done
  [ "$found" = 1 ] || { echo "FATAL: none of the benchmark CSVs exist under data/"; exit 1; }
  echo "preflight OK — models: ${MODELS[*]}"
  echo "            benchmarks: ${BENCHMARKS[*]}"
}

run_screen() {
  echo "=== all-models run started $(date) ==="
  echo "models: ${MODELS[*]}"
  echo "benchmarks: ${BENCHMARKS[*]}"
  local n=${#MODELS[@]}

  # screen every model via consecutive pairs (shared cache -> each model screened once)
  for ((i=0; i<n-1; i++)); do
    local OLD="${MODELS[$i]}" NEW="${MODELS[$((i+1))]}"
    echo "=== PAIR $OLD -> $NEW  $(date) ==="
    for BM in "${BENCHMARKS[@]}"; do
      local CSV="data/${BM}.csv"
      [ -f "$CSV" ] || { echo "[skip] $CSV not found"; continue; }
      echo "--- [$OLD -> $NEW | $BM] $(date) ---"
      python3 -m recyclebench pilot-pminus \
        --config "$CONFIG" --old "$OLD" --new "$NEW" \
        --behaviours "$CSV" --cache "$CACHE" \
        | tee "results/pminus_${OLD}_${NEW}_${BM}.txt"
      cp results/pminus_pilot.json "results/pminus_${OLD}_${NEW}_${BM}.json" 2>/dev/null || true
    done
  done

  # dump each consecutive pair's full 2x2 cells from the now-complete cache
  echo "=== dumping 2x2 cells per consecutive pair $(date) ==="
  local CSVS=""
  for BM in "${BENCHMARKS[@]}"; do [ -f "data/${BM}.csv" ] && CSVS="$CSVS data/${BM}.csv"; done
  for ((i=0; i<n-1; i++)); do
    local OLD="${MODELS[$i]}" NEW="${MODELS[$((i+1))]}"
    python3 tools/pminus_report.py --config "$CONFIG" --old "$OLD" --new "$NEW" \
      --cache "$CACHE" $CSVS --screen-inverse --dump "results/cells_${OLD}_${NEW}" \
      | tee "results/partition_${OLD}_${NEW}.txt"
  done

  echo "=== all-models run finished $(date) ==="
  echo "NOTE: every model is now cached; ANY pair's 2x2 (not just consecutive) is a free"
  echo "      cache-diff — re-run pminus_report.py with any --old/--new to get it."
}

# --- launch: preflight in the foreground, then re-exec THIS script detached -----------
# Re-executing the whole file (not just a function) is what makes the arrays survive into
# the background process — the earlier `declare -f` trick dropped them.
if [ "${_RB_DETACHED:-}" != "1" ]; then
  preflight
  _RB_DETACHED=1 nohup caffeinate -is bash "$0" > logs/all_models.log 2>&1 &
  echo "launched (PID $!). Watch:  tail -f logs/all_models.log"
  echo "safe to close this terminal — it keeps running and resumes from cache if interrupted."
  exit 0
fi

run_screen