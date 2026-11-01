#!/usr/bin/env bash
# Canonical screen of ONE model over one or more benchmarks, run inside the image by run.sbatch:
#   1. k canonical draws of every behaviour   (tools/fill_responses.py: only what is missing)
#   2. the config's judge over those draws    (tools/judge_lineage.py, appended to a verdict store)
#   3. the canonical 2x2 of every earlier lineage model -> RB_MODEL  (tools/llama4_report.py)
# Every step skips what the caches already hold, so a job that hits its time limit resumes on
# resubmission. Generalised from pegasus/llama4_canonical.sh (docs/pegasus/LESSONS.md §6.3): the
# config is RB_CONFIG, so this is not Llama-4-specific.
#
#   sbatch -p A100-80GB --gpus=1 --mem=96G --time=10:00:00 pegasus/run.sbatch \
#          bash pegasus/canonical_screen.sh
#   RB_MODEL=llama4-maverick RB_CACHE=cache/llama4_maverick.jsonl RB_STEPS=generate \
#     sbatch --export=ALL -p B200,H200 --gpus=2 --mem=300G --time=08:00:00 ...
#   RB_BENCH=advbench RB_DATA=data/smoke ...            # smoke test on a 5-row subset CSV
#
# Two jobs must never append to one JSONL: a job running beside another generates into its own
# cache and skips judging (RB_CACHE=… RB_STEPS=generate), then the shards are merged with
# tools/cache_consolidate.py (which now merges verdict stores too) before RB_STEPS="judge report".
set -euo pipefail
CFG=${RB_CONFIG:-configs/llama4_pegasus.yaml}
BENCH=${RB_BENCH:-"advbench harmbench strongreject sorrybench"}
DATA=${RB_DATA:-data}
MODEL=${RB_MODEL:-llama4-scout}
CACHE=${RB_CACHE:-cache/llama_gen.jsonl}
STEPS=${RB_STEPS:-"generate judge report"}
[ -f "$CFG" ] || { echo "no config at $CFG (set RB_CONFIG)"; exit 1; }

# Per-benchmark verdict store. RB_STORE_DIR relocates them for a side run; the defaults are the
# stores the existing judged figures were built from, so a normal run keeps accumulating there.
SD=${RB_STORE_DIR:-cache}
declare -A STORE=([sorrybench]=$SD/judge_lineage_sorrybench.jsonl
                  [harmbench]=$SD/judge_qwen_harmbench.jsonl
                  [advbench]=$SD/judge_qwen_advbench.jsonl
                  [strongreject]=$SD/judge_qwen_strongreject.jsonl)
EMPTY=$(mktemp); trap 'rm -f "$EMPTY"' EXIT          # canonical scope needs no manifest
mkdir -p "$(dirname "$CACHE")"; touch "$CACHE"       # a side cache (RB_CACHE) starts empty

echo "config $CFG | model $MODEL | cache $CACHE | steps $STEPS | data $DATA"

if [[ " $STEPS " == *" generate "* ]]; then
  for B in $BENCH; do
    [ -f "$DATA/$B.csv" ] || { echo "skip $B: no $DATA/$B.csv"; continue; }
    echo "===== [$(date +%T)] generate $B ====="
    python3 tools/fill_responses.py --config "$CFG" --cache "$CACHE" --model "$MODEL" \
        --behaviours "$DATA/$B.csv" --canonical --label "$MODEL $B"
  done
fi

if [[ " $STEPS " == *" judge "* ]]; then
  for B in $BENCH; do
    [ -f "$DATA/$B.csv" ] || continue
    [ -n "${STORE[$B]:-}" ] || { echo "skip judge $B: no verdict store configured"; continue; }
    echo "===== [$(date +%T)] judge $B ====="
    python3 tools/judge_lineage.py --config "$CFG" --cache "$CACHE" --models "$MODEL" \
        --scope canonical --behaviours "$DATA/$B.csv" --manifest "$EMPTY" --out "${STORE[$B]}"
  done
fi

if [[ " $STEPS " == *" report "* && "$DATA" = data ]]; then
  RUN=results/runs/$(date +%Y%m%d_%H%M)_${MODEL//-/_}_canonical
  mkdir -p "$RUN"
  cp "${RB_LOGS:-/tmp}/provenance_${SLURM_JOB_ID:-local}.txt" "$RUN/" 2>/dev/null || true
  python3 tools/llama4_report.py --config "$CFG" --new "$MODEL" --out "$RUN/canonical.json" \
      | tee "$RUN/canonical.txt"
  echo "wrote $RUN"
elif [[ " $STEPS " == *" report "* ]]; then
  echo "skip report: RB_DATA=$DATA is a subset, so a 2x2 against the full benchmarks would mislead"
fi
