#!/usr/bin/env bash
# Complete the inversion (RPE) arm. Short (~1.5 h), so it runs now rather than in the
# overnight stack. Only the steps that need Ollama live here; scoring, reports and dashboards
# are offline and run afterwards.
#
#   A  the 48 gate-admitted candidates outside A_loop on the 15 existing pair x benchmark runs
#      have no NEW reply yet (the `new` phase used to cover A_loop only): reply + judge, so
#      A_gate -- the admission comparable to RSR -- is scored, not just A_loop.
#   B  the two matched-scale post-hoc pairs (llama2-7b -> llama3-8b, -> llama31-8b) have no
#      inversion run at all: build their manifests (copying the llama2-7b inversions already
#      made for llama2-7b -> llama2-13b wherever the OLD answer is identical), gate them
#      offline, then run the closed loop.
#
# Fully resumable; own caches only, never cache/llama_gen.jsonl. Re-run to continue.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

CFG=configs/real_llama_lineage_all.yaml
GATE=cache/rpe_gate.jsonl
L32=results/runs/20260801_165721_llama32-1b_to_3b_recycle
ADJ="llama2-7b:llama2-13b llama2-13b:llama3-8b llama3-8b:llama31-8b llama31-8b:llama32-1b llama32-1b:llama32-3b"
POST="llama2-7b:llama3-8b llama2-7b:llama31-8b"
MIN="${MIN:-60}"                     # per-phase budget, minutes

while pgrep -f "run-real|fill_responses.py|judge_lineage.py|rpe_closedloop.py|run_compose.py" >/dev/null; do
  echo "waiting for Ollama ... $(date '+%T')"; sleep 120
done
echo "###############  RPE completion  $(date '+%F %T')  ###############"

# loop <cache-prefix> <manifest> <old> <new> <phase> [extra args...]
loop() {
  local P="$1" MAN="$2" O="$3" N="$4" PH="$5"; shift 5
  python3 tools/rpe_closedloop.py --config "$CFG" --rpe-manifest "$MAN" --old "$O" --new "$N" \
      --cache "cache/${P}_${O}__${N}.jsonl" --verdicts "cache/judge_${P}_${O}__${N}.jsonl" \
      --equiv-out "cache/${P}_${O}__${N}_equiv.json" --gate-store "$GATE" \
      --phase "$PH" --minutes "$MIN" "$@" || echo "!! ${P} ${O}->${N}/${PH} exited $?"
}

echo ""
echo "======== A  NEW replies for gate-admitted candidates outside A_loop  $(date '+%F %T')"
for P in $ADJ; do
  O="${P%%:*}"; N="${P##*:}"
  M="results/runs/rpe_lineage/rpe_manifest_${O}__to__${N}.jsonl"
  [ -s "$M" ] && for ph in new judge-new; do loop rpe_lin "$M" "$O" "$N" "$ph"; done
  for B in harmbench advbench strongreject; do
    M="results/runs/rpe_bench/$B/rpe_manifest_${O}__to__${N}.jsonl"
    [ -s "$M" ] && for ph in new judge-new; do loop "rpe_$B" "$M" "$O" "$N" "$ph"; done
  done
done

echo ""
echo "======== B1  build the post-hoc manifests  $(date '+%F %T')"
python3 tools/rpe_lineage.py --pairs llama2-7b:llama3-8b,llama2-7b:llama31-8b \
    --out-root results/runs/rpe_lineage_posthoc --minutes "$MIN" \
    --reuse-from 'results/runs/rpe_lineage/rpe_manifest_llama2-7b__to__*.jsonl' \
    --reuse-from 'results/runs/rpe_lineage_posthoc/rpe_manifest_llama2-7b__to__*.jsonl' \
    || echo "!! build sorrybench exited $?"
python3 tools/rpe_lineage.py --pairs llama2-7b:llama3-8b,llama2-7b:llama31-8b \
    --behaviours data/harmbench.csv --manifest "$L32/manifest_harmbench.jsonl" \
    --verdicts cache/judge_qwen_harmbench.jsonl \
    --out-root results/runs/rpe_bench_posthoc/harmbench --minutes "$MIN" \
    --reuse-from 'results/runs/rpe_bench/harmbench/rpe_manifest_llama2-7b__to__*.jsonl' \
    --reuse-from 'results/runs/rpe_bench_posthoc/harmbench/rpe_manifest_llama2-7b__to__*.jsonl' \
    || echo "!! build harmbench exited $?"
for B in advbench strongreject; do       # judged P^- is 0 on both: records the plan, calls nothing
  python3 tools/rpe_lineage.py --pairs llama2-7b:llama3-8b,llama2-7b:llama31-8b \
      --behaviours "data/$B.csv" --manifest "$L32/manifest_$B.jsonl" --verdicts "cache/judge_qwen_$B.jsonl" \
      --out-root "results/runs/rpe_bench_posthoc/$B" || echo "!! build $B exited $?"
done

echo ""
echo "======== B2  gate the new candidates (offline)  $(date '+%F %T')"
python3 tools/rpe_gate.py || echo "!! gate exited $?"

echo ""
echo "======== B3  closed loop for the post-hoc pairs  $(date '+%F %T')"
for P in $POST; do
  O="${P%%:*}"; N="${P##*:}"
  M="results/runs/rpe_lineage_posthoc/rpe_manifest_${O}__to__${N}.jsonl"
  if [ -s "$M" ]; then
    for ph in old judge-old equiv new judge-new; do
      loop rpe_lin "$M" "$O" "$N" "$ph" \
          --reuse-cache "cache/rpe_lin_llama2-7b__llama2-13b.jsonl" \
          --reuse-cache "cache/rpe_lin_llama2-7b__llama3-8b.jsonl" \
          --reuse-verdicts "cache/judge_rpe_lin_llama2-7b__llama2-13b.jsonl" \
          --reuse-verdicts "cache/judge_rpe_lin_llama2-7b__llama3-8b.jsonl"
    done
  fi
  M="results/runs/rpe_bench_posthoc/harmbench/rpe_manifest_${O}__to__${N}.jsonl"
  if [ -s "$M" ]; then
    for ph in old judge-old equiv new judge-new; do
      loop rpe_harmbench "$M" "$O" "$N" "$ph" \
          --reuse-cache "cache/rpe_harmbench_llama2-7b__llama2-13b.jsonl" \
          --reuse-cache "cache/rpe_harmbench_llama2-7b__llama3-8b.jsonl" \
          --reuse-verdicts "cache/judge_rpe_harmbench_llama2-7b__llama2-13b.jsonl" \
          --reuse-verdicts "cache/judge_rpe_harmbench_llama2-7b__llama3-8b.jsonl"
    done
  fi
done
echo "###############  finished  $(date '+%F %T')  ###############"
