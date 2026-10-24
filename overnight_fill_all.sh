#!/usr/bin/env bash
# Close every remaining AUTOMATIC gap, then rebuild every dashboard.
#
# Waits for the lineage inversion run to finish first: both want Ollama, and three
# resident models (llama2:13b + llama3:8b + qwen2.5:14b) exceed OLLAMA_MAX_LOADED_MODELS=2
# and would thrash ~10 GB per call.
#
# Human-only and cluster-only work is NOT attempted here; see TODO_HUMAN.md.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

CFG=configs/real_llama_lineage_all.yaml
L3=results/runs/20260805_025658_llama3-8b_to_llama31-8b_RECYCLE
A70="$HOME/Downloads/all_local_results_llama3_1_70b 1.jsonl"
STOP_AT="${STOP_AT:-21:50}"          # model calls stop here; offline steps always run

step() { echo ""; echo "=== $1   $(date '+%F %T')"; shift; "$@" || echo "!! exited $? -- continuing"; }

echo "###############  fill-all  $(date '+%F %T')  (model calls stop $STOP_AT)  ###############"

# 0. wait for the lineage inversion run to release Ollama
while pgrep -f "rpe_lineage.sh|rpe_closedloop.py" >/dev/null; do
  echo "waiting for the lineage run ... $(date '+%T')"; sleep 120
done
echo "lineage run finished; Ollama is free  $(date '+%F %T')"

# 1. PBR: the only two 'not estimable' cells -- llama2-13b rewording responses
for b in advbench strongreject; do
  step "PBR fill: llama2-13b x $b" \
    python3 tools/fill_responses.py --config $CFG --model llama2-13b --behaviours data/$b.csv \
      --manifest $L3/manifest_$b.jsonl --stop-at "$STOP_AT" --label "pbr-fill-$b"
done

# 2. judge whatever that produced (resumable, skips what is already judged)
for b in advbench strongreject; do
  step "qwen judge $b, every variant (PBR)" \
    python3 tools/judge_lineage.py --config $CFG --behaviours data/$b.csv \
      --manifest $L3/manifest_$b.jsonl --models llama2-7b llama2-13b llama3-8b llama31-8b \
      --out cache/judge_qwen_$b.jsonl --scope all --stop-at "$STOP_AT"
done

# 3. offline: every analysis and report, then every dashboard, with fresh dated names
step "thesis analysis"        python3 tools/thesis_analysis.py
step "RPE repaired agreement" python3 tools/rpe_repair.py
step "RPE control report"     python3 tools/rpe_control_report.py
step "master dashboard"       python3 tools/master_dashboard.py --attack-70b-responses "$A70"
step "pair dashboards"        python3 tools/pair_dashboard.py --attack-70b-responses "$A70"
step "lineage dashboard"      python3 tools/lineage_dashboard.py
step "tests"                  python3 -m pytest tests/ -q

echo ""
echo "###############  finished  $(date '+%F %T')  ###############"
