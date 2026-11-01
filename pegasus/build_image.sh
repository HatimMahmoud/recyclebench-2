#!/usr/bin/env bash
# Build docker/Dockerfile into an enroot image on a compute node (never the login node), the way the
# Pegasus docs prescribe (custom software, option 5): podman build -> enroot import. Submitted with
# sbatch, so it survives a dropped SSH session; watch logs/rb-build_<job>.out.
#   bash pegasus/build_image.sh                    # -> $RB_IMAGE
#   RB_OLLAMA_VERSION=0.40.13 bash pegasus/build_image.sh
#   WITH_GATE=1 RB_BUILD_MEM=96G RB_IMAGE=/netscratch/$USER/recyclebench/images/recyclebench_gate.sqsh \
#       bash pegasus/build_image.sh
# The gate layer (torch + CUDA wheels) is OOM-killed at 24G while podman commits it: give it 96G.
# The build context is $RB_REPO (the container workdir), so to build an exact commit point RB_REPO
# at a clean checkout of it.
set -euo pipefail
cd "$(dirname "$0")/.."
source pegasus/env.sh
mkdir -p "$(dirname "$RB_IMAGE")" "$RB_LOGS"
TAG=recyclebench:ollama-$RB_OLLAMA_VERSION

# CPU-only, but every partition is a GPU partition: pick one with idle cores (sinfo -o "%P %C %e")
sbatch -p "${RB_BUILD_PARTITION:-RTXA6000}" --gpus=0 --cpus-per-task=4 --mem="${RB_BUILD_MEM:-24G}" \
       --time=01:00:00 --job-name=rb-build --output="$RB_LOGS/%x_%j.out" \
       --container-image=/enroot/podman+enroot.sqsh \
       --container-mounts="/dev/fuse:/dev/fuse,$RB_SCRATCH:$RB_SCRATCH,$RB_REPO:$RB_REPO" \
       --container-workdir="$RB_REPO" \
       --wrap "set -euo pipefail
               podman build -f docker/Dockerfile -t $TAG \
                   --build-arg OLLAMA_VERSION=$RB_OLLAMA_VERSION --build-arg WITH_GATE=${WITH_GATE:-0} .
               rm -f '$RB_IMAGE'
               enroot import -o '$RB_IMAGE' podman://$TAG
               ls -la '$RB_IMAGE'"
