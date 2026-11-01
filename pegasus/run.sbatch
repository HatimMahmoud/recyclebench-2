#!/bin/bash
#SBATCH --job-name=recyclebench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/netscratch/%u/recyclebench/logs/%x_%j.out
# Run any repo command inside the recyclebench image with a private Ollama server.
# Submit from the repo on the login node; partition, GPUs, memory and time go on the command line:
#   sbatch -p A100-80GB --gpus=1 --mem=96G --time=08:00:00 pegasus/run.sbatch python3 tools/...
#   sbatch -p RTXA6000 --gpus=0 --mem=24G --time=02:00:00 pegasus/run.sbatch ollama pull llama3.1:8b
# The defaults above are deliberately small; ask for what the job needs (docs/pegasus/kb/05-*.md).
# Every partition is a GPU partition: CPU-only jobs pass --gpus=0.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source pegasus/env.sh
[ -f "$RB_IMAGE" ] || { echo "no image at $RB_IMAGE: run pegasus/build_image.sh first"; exit 1; }

srun --container-image="$RB_IMAGE" \
     --container-mounts="/netscratch:/netscratch,/ds:/ds:ro,$RB_REPO:$RB_REPO" \
     --container-workdir="$RB_REPO" \
     bash pegasus/in_container.sh "$@"
