#!/usr/bin/env bash
# Llama 4 Scout inversion (RPE) arm: every earlier lineage model -> llama4-scout on AdvBench, HarmBench
# and StrongREJECT, with the established tools and their defaults (8 candidates, inverter
# real.qwen25-14b from configs/recycle_3_31.yaml, MiniLM equivalence 0.75, production gate). Run inside
# the gate image by run.sbatch:
#
#   RB_IMAGE=$RB_SCRATCH/images/recyclebench_gate_<rev>.sqsh \
#   sbatch --export=ALL -p H100,H100-PCI,A100-80GB,A100-RP,RTXB6000,H200,B200 --gpus=1 --cpus-per-task=8 \
#          --mem=64G --time=06:00:00 --job-name=rb-scout-rpe pegasus/run.sbatch bash pegasus/scout_rpe.sh
#
# Same pipeline the rpe_bench runs used (run_rpe_complete.sh): rpe_lineage.py -> rpe_gate.py ->
# rpe_closedloop.py old/judge-old/equiv/new/judge-new/score -> rpe_lineage_report.py. The P^- comes
# from the judged canonical screen (cache + verdict store); the empty manifest only satisfies
# observe(), which uses it for variants, not for P^-. A pair with empty P^- writes its plan and calls
# nothing (P^- undefined, never 0). Every phase loads one Ollama model at a time, so an 80 GB GPU is
# enough. Everything lands in cache/llama4_scout/ and results/runs/llama4_scout/rpe/, never in the
# shared rpe stores. Resumable: rerun the same command.
#
# Not reused: the lineage inversions (results/runs/rpe_*) that --reuse-from could copy live only on the
# Mac, so every OLD answer is inverted afresh here (same inverter, same answer, new samples).
set -euo pipefail
CFG=configs/llama4_pegasus.yaml
NEW=llama4-scout
OLDS=${RB_OLDS:-"llama2-7b llama2-13b llama3-8b llama31-8b llama32-1b llama32-3b"}
BENCH=${RB_BENCH:-"advbench harmbench strongreject"}
C=cache/llama4_scout
OUT=results/runs/llama4_scout/rpe
GATE=$C/rpe_gate.jsonl
MIN=${RB_MINUTES:-90}                                   # per-phase budget, minutes
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1                                  # gate models are prefetched into $HF_HOME
mkdir -p "$C" "$OUT"
EMPTY=$C/empty_manifest.jsonl; : > "$EMPTY"
PAIRS=$(for O in $OLDS; do printf "%s:%s," "$O" "$NEW"; done); PAIRS=${PAIRS%,}
python3 tools/env_check.py --strict
source pegasus/gpu_guard.sh                     # refuse a silent CPU fallback

echo "===== [$(date +%T)] 1. inversion manifests"
for B in $BENCH; do
  python3 tools/rpe_lineage.py --config "$CFG" --behaviours "data/$B.csv" --manifest "$EMPTY" \
      --cache cache/llama_gen.jsonl --verdicts "cache/judge_qwen_$B.jsonl" --pairs "$PAIRS" \
      --benchmark "$B" --out-root "$OUT/$B" --minutes "$MIN"
done

echo "===== [$(date +%T)] 2. gate every candidate (offline)"
MANS=$(ls "$OUT"/*/rpe_manifest_*.jsonl 2>/dev/null || true)
[ -n "$MANS" ] && python3 tools/rpe_gate.py --config "$CFG" --manifests $MANS --store "$GATE" --minutes "$MIN"

echo "===== [$(date +%T)] 3. closed loop"
STAMP=$(date +%Y%m%d_%H%M)
for B in $BENCH; do
  for O in $OLDS; do
    M=$OUT/$B/rpe_manifest_${O}__to__${NEW}.jsonl
    [ -s "$M" ] || { echo "-- $B $O->$NEW: empty P^- (nothing to invert)"; continue; }
    for PH in old judge-old equiv new judge-new score; do
      python3 tools/rpe_closedloop.py --config "$CFG" --rpe-manifest "$M" --old "$O" --new "$NEW" \
          --cache "$C/rpe_${B}_${O}__${NEW}.jsonl" --verdicts "$C/judge_rpe_${B}_${O}__${NEW}.jsonl" \
          --equiv-out "$C/rpe_${B}_${O}__${NEW}_equiv.json" --gate-store "$GATE" \
          --phase "$PH" --minutes "$MIN" \
          --out "$OUT/$B/closedloop_${O}__to__${NEW}_gated_$STAMP.json" \
        || echo "!! $B $O->$NEW/$PH exited $?"
    done
  done
  python3 tools/rpe_lineage_report.py --config "$CFG" --behaviours "data/$B.csv" --manifest "$EMPTY" \
      --cache cache/llama_gen.jsonl --verdicts "cache/judge_qwen_$B.jsonl" --rpe-root "$OUT/$B" \
      --pairs "$PAIRS" --out "$OUT/$B/lineage_report_$STAMP.json" || echo "!! report $B exited $?"
done
echo "done $(date -Is)"
