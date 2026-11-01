#!/usr/bin/env bash
# Llama 4 canonical screen, run inside the image by run.sbatch. RB_MODEL picks the config key:
# llama4-scout (default, one 80 GB+ GPU) or llama4-maverick (one node, >= 3 GPUs).
#   1. k=3 canonical draws of every behaviour (tools/fill_responses.py: only what is missing)
#   2. the qwen2.5:14b rubric judge over those draws, appended to each benchmark's verdict store
#   3. the canonical 2x2 of every earlier lineage model -> $RB_MODEL (tools/llama4_report.py)
# Every step skips what the caches already hold, so a timed-out job resumes on resubmission.
#   sbatch -p A100-80GB --gpus=1 --mem=96G --time=10:00:00 pegasus/run.sbatch bash pegasus/llama4_canonical.sh
#   RB_MODEL=llama4-maverick sbatch -p RTXB6000 --gpus=3 --mem=320G --time=1-00:00:00 ...
#   RB_BENCH="advbench" RB_DATA=data/smoke ...   # smoke test on a subset CSV directory
# Two jobs never share a cache file: a job running beside another one generates into its own cache
# and skips judging (RB_CACHE=cache/llama4_maverick.jsonl RB_STEPS=generate); merge that file into
# cache/llama_gen.jsonl afterwards (tools/cache_consolidate.py --sources ...), then run RB_STEPS="judge report".
set -euo pipefail
CFG=configs/llama4_pegasus.yaml
BENCH=${RB_BENCH:-"advbench harmbench strongreject sorrybench"}
DATA=${RB_DATA:-data}
MODEL=${RB_MODEL:-llama4-scout}
CACHE=${RB_CACHE:-cache/llama_gen.jsonl}
STEPS=${RB_STEPS:-"generate judge report"}
declare -A STORE=([sorrybench]=cache/judge_lineage_sorrybench.jsonl [harmbench]=cache/judge_qwen_harmbench.jsonl
                  [advbench]=cache/judge_qwen_advbench.jsonl [strongreject]=cache/judge_qwen_strongreject.jsonl)
EMPTY=$(mktemp); trap 'rm -f "$EMPTY"' EXIT          # canonical scope: no manifest needed
mkdir -p "$(dirname "$CACHE")"; touch "$CACHE"      # a side cache (RB_CACHE) starts empty

[[ " $STEPS " == *" generate "* ]] && for B in $BENCH; do
    [ -f "$DATA/$B.csv" ] || { echo "skip $B: no $DATA/$B.csv"; continue; }
    echo "===== [$(date +%T)] generate $B ====="
    python3 tools/fill_responses.py --config $CFG --cache "$CACHE" --model $MODEL \
        --behaviours "$DATA/$B.csv" --canonical --label "$MODEL $B"
done
[[ " $STEPS " == *" judge "* ]] && for B in $BENCH; do
    [ -f "$DATA/$B.csv" ] || continue
    echo "===== [$(date +%T)] judge $B ====="
    python3 tools/judge_lineage.py --config $CFG --cache "$CACHE" --models $MODEL --scope canonical \
        --behaviours "$DATA/$B.csv" --manifest "$EMPTY" --out "${STORE[$B]}"
done
if [[ " $STEPS " == *" report "* && "$DATA" = data ]]; then
    RUN=results/runs/$(date +%Y%m%d_%H%M)_${MODEL//-/_}_canonical
    mkdir -p "$RUN"
    cp "$RB_LOGS/provenance_${SLURM_JOB_ID:-local}.txt" "$RUN/" 2>/dev/null || true
    python3 tools/llama4_report.py --config $CFG --new $MODEL --out "$RUN/canonical.json" | tee "$RUN/canonical.txt"
fi
