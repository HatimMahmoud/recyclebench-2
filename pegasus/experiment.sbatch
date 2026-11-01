#!/bin/bash
#SBATCH --job-name=rb-experiment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/netscratch/%u/recyclebench/logs/%x_%j.out
# Run ONE experiment spec on a compute node, with the same semantics as the local run.
#
#   sbatch -p A100-80GB --gpus=1 --mem=96G --time=08:00:00 \
#          pegasus/experiment.sbatch experiments/<spec>.yaml
#   RB_STAGE=screen sbatch ... pegasus/experiment.sbatch experiments/<spec>.yaml   # one stage
#   RB_PLAN=1       sbatch -p RTXA6000 --gpus=0 --mem=8G --time=00:10:00 \
#                   pegasus/experiment.sbatch experiments/<spec>.yaml              # plan only
#
# The experiment driver is the same code path as `python3 -m recyclebench experiment run` on a
# laptop: identical stage order, identical guards, identical artifacts. Only three things differ,
# and none of them is scientific:
#   * the command runs inside the image with a private Ollama (pegasus/run.sbatch, in_container.sh);
#   * --shard $SLURM_JOB_ID gives the job its own cache shard, so two jobs can never append to one
#     JSONL (docs/pegasus/LESSONS.md §6.2). Merge afterwards with tools/cache_consolidate.py;
#   * resources come from the sbatch line, never from a Python file.
#
# Defaults above are deliberately small: ask for what the job needs (docs/pegasus/kb/05-*.md), and
# check the queue with `sbatch --test-only <same options>` first. Every partition is a GPU
# partition, so CPU-only work passes --gpus=0.
set -euo pipefail
SPEC=${1:?usage: sbatch [resources] pegasus/experiment.sbatch <experiment spec yaml> }
cd "$SLURM_SUBMIT_DIR"
source pegasus/env.sh
[ -f "$SPEC" ] || { echo "no spec at $SPEC"; exit 1; }
[ -f "$RB_IMAGE" ] || { echo "no image at $RB_IMAGE: run pegasus/build_image.sh first"; exit 1; }

ACTION=run
[ -n "${RB_PLAN:-}" ] && ACTION=plan
ARGS=(experiment "$ACTION" --spec "$SPEC" --target cluster)
# A plan must stay read-only, so it gets no shard (a shard only names an output path, but keeping
# the plan free of job-specific state means two plans of one spec are identical).
[ "$ACTION" = run ] && ARGS+=(--shard "${SLURM_JOB_ID:-local}")
[ -n "${RB_STAGE:-}" ] && ARGS+=(--stage "$RB_STAGE")
[ -n "${RB_FORCE:-}" ] && ARGS+=(--force)

srun --container-image="$RB_IMAGE" \
     --container-mounts="/netscratch:/netscratch,/ds:/ds:ro,$RB_REPO:$RB_REPO" \
     --container-workdir="$RB_REPO" \
     bash pegasus/in_container.sh python3 -m recyclebench "${ARGS[@]}"
