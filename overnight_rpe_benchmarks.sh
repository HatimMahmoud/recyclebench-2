#!/usr/bin/env bash
# Extend the inversion arm beyond SORRY-Bench: HarmBench (99 P^- behaviours across the
# five rungs), AdvBench (9) and StrongREJECT (4). Tests whether the SORRY-Bench result
# -- inversion re-opening far more than paraphrase -- is a property of the estimand or
# of one benchmark.
#
# Waits for any running fill/judge to release Ollama. Fully resumable; own caches;
# never writes to cache/llama_gen.jsonl.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

CFG=configs/real_llama_lineage_all.yaml
L32=results/runs/20260801_165721_llama32-1b_to_3b_recycle
PAIRS="llama2-7b:llama2-13b llama2-13b:llama3-8b llama3-8b:llama31-8b llama31-8b:llama32-1b llama32-1b:llama32-3b"
STOP_MIN="${STOP_MIN:-110}"          # per-phase budget, minutes

while pgrep -f "fill_responses.py|judge_lineage.py|rpe_closedloop.py" >/dev/null; do
  echo "waiting for Ollama ... $(date '+%T')"; sleep 120
done
echo "###############  inversion arm across benchmarks  $(date '+%F %T')  ###############"

for B in harmbench advbench strongreject; do
  ROOT_B="results/runs/rpe_bench/$B"
  echo ""
  echo "######## build $B manifests  $(date '+%F %T')"
  python3 tools/rpe_lineage.py --config $CFG --behaviours "data/$B.csv" \
      --manifest "$L32/manifest_$B.jsonl" --verdicts "cache/judge_qwen_$B.jsonl" \
      --out-root "$ROOT_B" --minutes 45 || echo "!! build $B exited $?"

  for P in $PAIRS; do
    O="${P%%:*}"; N="${P##*:}"
    MAN="$ROOT_B/rpe_manifest_${O}__to__${N}.jsonl"
    [ -s "$MAN" ] || { echo "== skip $B ${O}->${N}: empty/no manifest"; continue; }
    echo "--- $B ${O}->${N}  $(date '+%F %T')"
    for ph in old judge-old equiv new judge-new score; do
      python3 tools/rpe_closedloop.py --config $CFG --rpe-manifest "$MAN" \
          --old "$O" --new "$N" \
          --cache "cache/rpe_${B}_${O}__${N}.jsonl" \
          --verdicts "cache/judge_rpe_${B}_${O}__${N}.jsonl" \
          --equiv-out "cache/rpe_${B}_${O}__${N}_equiv.json" \
          --out "$PWD/$ROOT_B/closedloop_${O}__to__${N}.json" \
          --phase "$ph" --minutes "$STOP_MIN" || echo "!! $B ${O}->${N}/${ph} exited $?"
    done
  done
done

echo ""
echo "=== reports  $(date '+%F %T')"
python3 tools/rpe_lineage_report.py || echo "!! sorrybench report exited $?"
for B in harmbench advbench strongreject; do
  python3 tools/rpe_lineage_report.py --behaviours "data/$B.csv" \
      --manifest "$L32/manifest_$B.jsonl" --verdicts "cache/judge_qwen_$B.jsonl" \
      --rpe-root "results/runs/rpe_bench/$B" \
      --out "$PWD/results/runs/rpe_bench/$B/lineage_report.json" || echo "!! $B report exited $?"
done
echo "###############  finished  $(date '+%F %T')  ###############"
