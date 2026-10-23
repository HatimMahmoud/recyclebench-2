#!/usr/bin/env bash
# Closed-loop (preimage) admission for the independent-probe arm -- 12 h budget.
#
# Phase-ordered so Ollama holds ONE target model at a time and never reloads ~10 GB
# per call. Every phase is resumable: re-running the script continues where it
# stopped. Writes only to its own caches (cache/rpe_closedloop*.jsonl); the shared
# study cache cache/llama_gen.jsonl is NOT touched.
#
#   bash overnight_rpe_closedloop.sh          # foreground
#   caffeinate -is nohup bash overnight_rpe_closedloop.sh &> logs/x.log &
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

CL="python3 tools/rpe_closedloop.py --config configs/real_llama_lineage_all.yaml"

step() {  # step <label> <minutes> <phase>
  echo ""
  echo "=== $1   (budget $2 min)   $(date '+%F %T')"
  $CL --phase "$3" --minutes "$2" || echo "!! $1 exited $? -- continuing"
  echo "--- $1 done $(date '+%F %T')"
}

echo "###############  closed-loop overnight  $(date '+%F %T')  ###############"
step "S1 OLD answers every candidate"      150 old
step "S2 judge OLD answers"                180 judge-old
step "S3 response equivalence (offline)"    30 equiv
step "S4 NEW answers A_loop candidates"    120 new
step "S5 judge NEW answers"                120 judge-new
step "S6 score the three admission sets"    10 score

echo ""
echo "=== S7 refresh the repaired agreement (offline)  $(date '+%F %T')"
python3 tools/rpe_repair.py || echo "!! rpe_repair exited $?"

echo ""
echo "=== S8 tests  $(date '+%F %T')"
python3 -m pytest tests/ -q 2>&1 | tail -3

echo ""
echo "###############  finished  $(date '+%F %T')  ###############"
ollama ps 2>/dev/null | head -3
