#!/usr/bin/env bash
# Usage: bash run_pair.sh <old> <new> <cache_file> [benchmarks...]
set -u
export PYTHONPATH=src
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

OLD="$1"; NEW="$2"; CACHE="$3"; shift 3
BENCHMARKS=("$@")
[ ${#BENCHMARKS[@]} -eq 0 ] && BENCHMARKS=(harmbench advbench strongreject)

RUN="results/runs/$(date +%Y%m%d_%H%M%S)_${OLD}_to_${NEW}_full"
mkdir -p "$RUN/cells"
{
  echo "pair      : $OLD -> $NEW"
  echo "config    : configs/real_llama_lineage.yaml"
  echo "cache     : $CACHE"
  echo "benchmarks: ${BENCHMARKS[*]}"
  echo "includes  : P^-, persistent, negative flips (inverse), both-safe"
  echo "date      : $(date)"
} > "$RUN/META.txt"

for BM in "${BENCHMARKS[@]}"; do
  echo "===== pilot $BM $(date) ====="
  python3 -m recyclebench pilot-pminus \
    --config configs/real_llama_lineage.yaml \
    --old "$OLD" --new "$NEW" \
    --behaviours "data/${BM}.csv" --cache "$CACHE" \
    | tee "$RUN/pminus_${BM}.txt"
  cp results/pminus_pilot.json "$RUN/pminus_${BM}.json"
done

echo "===== partition + inverse $(date) ====="
CSVS=()
for BM in "${BENCHMARKS[@]}"; do CSVS+=("data/${BM}.csv"); done
python3 tools/pminus_report.py \
  --config configs/real_llama_lineage.yaml \
  --old "$OLD" --new "$NEW" --cache "$CACHE" \
  "${CSVS[@]}" --screen-inverse --dump "$RUN/cells" \
  | tee "$RUN/partition_full.txt"

echo "=== finished $(date) ===" >> "$RUN/META.txt"
echo "DONE -> $RUN"
