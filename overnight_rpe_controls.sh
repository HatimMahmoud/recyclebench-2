#!/usr/bin/env bash
# Negative controls for the inversion arm: blind (category only) and mismatched
# (another behaviour's answer, same category). Each is generated with the SAME
# inverter spec as the real arm, then scored through the IDENTICAL closed-loop
# pipeline, so the only difference is what the inverter was shown.
#
# Fully resumable; own caches; never writes to cache/llama_gen.jsonl.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

run_control() {  # run_control <name> <gen-minutes> <phase-minutes>
  local C="$1" GM="$2" PM="$3"
  local MAN="results/runs/rpe_controls/rpe_manifest_${C}.jsonl"
  echo ""
  echo "######## control: ${C}   $(date '+%F %T')"
  python3 tools/rpe_controls.py --control "$C" --minutes "$GM" \
      || echo "!! generate ${C} exited $?"
  for ph in old judge-old equiv new judge-new score; do
    echo "--- ${C}/${ph}  $(date '+%F %T')"
    python3 tools/rpe_closedloop.py \
        --config configs/real_llama_lineage_all.yaml \
        --rpe-manifest "$MAN" \
        --cache "cache/rpe_${C}.jsonl" \
        --verdicts "cache/judge_rpe_${C}.jsonl" \
        --equiv-out "cache/rpe_${C}_equiv.json" \
        --out "results/runs/rpe_controls/closedloop_${C}.json" \
        --phase "$ph" --minutes "$PM" || echo "!! ${C}/${ph} exited $?"
  done
}

echo "###############  controls  $(date '+%F %T')  ###############"
run_control blind      20 150
run_control mismatched 40 150
echo ""
echo "###############  finished  $(date '+%F %T')  ###############"
