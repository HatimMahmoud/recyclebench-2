#!/usr/bin/env bash
# Inversion arm across every adjacent SORRY-Bench lineage rung, so its per-pair PRR
# is directly comparable to the recycle arm's per-pair RSR (P^- = 18, 3, 49, 12, 13).
#
# Each pair runs the whole closed-loop pipeline with its own OLD/NEW models and its
# own caches. Fully resumable; never writes to cache/llama_gen.jsonl.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

PAIRS="llama2-7b:llama2-13b llama2-13b:llama3-8b llama3-8b:llama31-8b llama31-8b:llama32-1b llama32-1b:llama32-3b"

echo "###############  lineage inversion arm  $(date '+%F %T')  ###############"
echo "--- build every pair's inversion manifest"
python3 tools/rpe_lineage.py --minutes 60 || echo "!! build exited $?"

for P in $PAIRS; do
  O="${P%%:*}"; N="${P##*:}"
  MAN="results/runs/rpe_lineage/rpe_manifest_${O}__to__${N}.jsonl"
  [ -s "$MAN" ] || { echo "== skip ${O}->${N}: no manifest"; continue; }
  echo ""
  echo "######## ${O} -> ${N}   $(date '+%F %T')"
  for ph in old judge-old equiv new judge-new score; do
    echo "--- ${O}->${N} / ${ph}  $(date '+%F %T')"
    python3 tools/rpe_closedloop.py \
        --config configs/real_llama_lineage_all.yaml \
        --rpe-manifest "$MAN" --old "$O" --new "$N" \
        --cache "cache/rpe_lin_${O}__${N}.jsonl" \
        --verdicts "cache/judge_rpe_lin_${O}__${N}.jsonl" \
        --equiv-out "cache/rpe_lin_${O}__${N}_equiv.json" \
        --out "$PWD/results/runs/rpe_lineage/closedloop_${O}__to__${N}.json" \
        --phase "$ph" --minutes 90 || echo "!! ${O}->${N}/${ph} exited $?"
  done
done

echo ""
echo "=== tests  $(date '+%F %T')"
python3 -m pytest tests/ -q 2>&1 | tail -3
echo "###############  finished  $(date '+%F %T')  ###############"
