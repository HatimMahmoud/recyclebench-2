#!/usr/bin/env bash
# External-corpora run, step 4 (CPU, no model calls): style arcs over each corpus's frozen manifests, the inversion
# lineage report per corpus, then the completion matrix (results/runs/external/matrix_<stamp>.{json,md}). Run by
# run.sbatch after every recycle and RPE job, whatever their exit state: the matrix is where a failure shows.
#
#   sbatch -p RTXA6000 --gpus=0 --mem=24G --time=01:00:00 pegasus/run.sbatch bash pegasus/external_final.sh
set -uo pipefail
C=cache/external
OUT=results/runs/external
CORPORA=$(python3 -c "import sys; sys.path.insert(0, 'tools'); import external_suite as e; print(' '.join(e.CORPORA))")
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
STAMP=$(date +%Y%m%d_%H%M)
EMPTY=$C/empty_manifest.jsonl; : > "$EMPTY"

for B in $CORPORA; do
  MANS=$(ls $OUT/external__${B}__*/manifest_*.jsonl 2>/dev/null || true)
  if [ -n "$MANS" ]; then
    for M in nli admission_rate; do
      python3 tools/style_arcs.py --manifests $MANS --metric $M --out "$OUT/style_arcs_${B}_${M}_$STAMP.html" \
        || echo "!! style arcs $B/$M exited $?"
    done
  else
    echo "-- $B: no frozen recycle manifest, no arcs"
  fi
  PAIRS=$(python3 tools/external_suite.py list --pairs --corpus "$B" --complete 2>/dev/null || true)
  if [ -n "$PAIRS" ] && ls $OUT/rpe/$B/rpe_manifest_*.jsonl > /dev/null 2>&1; then
    python3 tools/rpe_lineage_report.py --config configs/llama4_pegasus.yaml --behaviours "data/$B.csv" \
        --manifest "$EMPTY" --cache $C/canonical.jsonl --verdicts "$C/judge_qwen_$B.jsonl" --rpe-root "$OUT/rpe/$B" \
        --pairs "$PAIRS" --out "$OUT/rpe/$B/lineage_report_$STAMP.json" || echo "!! rpe report $B exited $?"
  fi
done
python3 tools/external_suite.py matrix
echo "done $(date -Is)"
