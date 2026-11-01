#!/usr/bin/env bash
# Llama 4 Scout recycle arm: the three eligible pairs (experiments/llama4_scout/*.yaml), run inside
# the gate image by run.sbatch, one spec after another in ONE job (no shard: nothing else writes
# these caches, and an unsharded cache is what a resubmission resumes from).
#
#   RB_IMAGE=$RB_SCRATCH/images/recyclebench_gate_<rev>.sqsh \
#   sbatch --export=ALL -p RTXB6000,H200,B200 --gpus=1 --cpus-per-task=8 --mem=64G --time=06:00:00 \
#          --job-name=rb-scout-recycle pegasus/run.sbatch bash pegasus/scout_recycle.sh
#
# Needs a GPU with more than 80 GB: run-real alternates Scout (67 GB) with the qwen2.5:14b judge
# (9 GB) per call, and the gate's torch models sit on the same device, so on an 80 GB card Ollama
# would evict and reload Scout (~8 min) on every judge call.
#
# Per spec:
#   1. seed its cache once from cache/llama_gen.jsonl (tools/cache_consolidate.py: verbatim, append-
#      only), so the screen re-judges the canonical draws already paid for instead of regenerating them;
#   2. `experiment run` until a pass runs nothing (the driver plans once per invocation, so subset and
#      recycle become runnable only on the pass after screen; every stage is resumable);
#   3. per-variant trials (RECYCLEBENCH_TRIALS_DIR) for the style breakdown.
# Then the style arcs over the three frozen manifests (offline, gate scores only).
set -euo pipefail
SPECS=${RB_SPECS:-"experiments/llama4_scout/harmbench__llama3-8b.yaml
                   experiments/llama4_scout/harmbench__llama32-3b.yaml
                   experiments/llama4_scout/advbench__llama3-8b.yaml"}
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
python3 tools/env_check.py --strict
source pegasus/gpu_guard.sh                     # refuse a silent CPU fallback

for S in $SPECS; do
  NAME=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$S")
  CACHE=cache/llama4_scout/$NAME.jsonl
  RUN=results/runs/llama4_scout/$NAME
  mkdir -p "$(dirname "$CACHE")" "$RUN"
  if [ ! -f "$CACHE.seeded" ]; then
    touch "$CACHE"
    python3 tools/cache_consolidate.py --target "$CACHE" --sources cache/llama_gen.jsonl --apply
    { echo "seeded_from cache/llama_gen.jsonl"; echo "source_lines $(wc -l < cache/llama_gen.jsonl)";
      echo "source_sha256 $(sha256sum cache/llama_gen.jsonl | cut -d' ' -f1)"; echo "when $(date -Is)";
      echo "job ${SLURM_JOB_ID:-local}"; } > "$CACHE.seeded"
  fi
  export RECYCLEBENCH_TRIALS_DIR=$RUN/trials
  for pass in 1 2 3 4; do
    echo "===== [$(date +%T)] $NAME pass $pass ====="
    # a failed stage records nothing; move on to the next spec and let a resubmission resume it
    out=$(python3 -m recyclebench experiment run --spec "$S" --target cluster | tee /dev/stderr) \
        || { echo "!! $NAME: a stage failed (rc=$?), continuing with the next spec"; break; }
    echo "$out" | grep -q "^0 stage(s) ran" && break
  done
  python3 -m recyclebench experiment plan --spec "$S" --target cluster
done

STAMP=$(date +%Y%m%d_%H%M)
MANS=$(ls results/runs/llama4_scout/llama4_scout__*/manifest_*.jsonl 2>/dev/null || true)
if [ -n "$MANS" ]; then
  for M in nli admission_rate; do
    python3 tools/style_arcs.py --manifests $MANS --metric $M \
        --out "results/runs/llama4_scout/style_arcs_${M}_$STAMP.html"
  done
fi
echo "done $(date -Is)"
