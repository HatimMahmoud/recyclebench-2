#!/usr/bin/env bash
# Runs inside the recyclebench image on a compute node (called by run.sbatch): serve Ollama on a
# per-job port, wait until it answers, optionally preload models, run "$@", then stop Ollama.
#   RB_PRELOAD="model_id ..."   load these before the command (a 67 GB model takes minutes)
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$OLLAMA_MODELS" "$RB_LOGS"
JOB=${SLURM_JOB_ID:-local$$}

# another job on the same node may hold 11434, so every job gets its own port
PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))
export OLLAMA_HOST=127.0.0.1:$PORT
export RECYCLEBENCH_OLLAMA_URL=http://127.0.0.1:$PORT    # read by backends/providers.py

ollama serve > "$RB_LOGS/ollama_$JOB.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true; wait $OLLAMA_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
    curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null && break
    kill -0 $OLLAMA_PID 2>/dev/null || { echo "ollama serve died, see $RB_LOGS/ollama_$JOB.log"; exit 1; }
    sleep 1
done
curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null || { echo "ollama did not answer on $PORT"; exit 1; }

for m in ${RB_PRELOAD:-}; do
    echo "[$(date +%T)] preloading $m"
    curl -sf "http://$OLLAMA_HOST/api/generate" -d "{\"model\": \"$m\", \"keep_alive\": \"1h\"}" > /dev/null
done

# provenance: the run folder should say which Ollama, GPU and model digests produced it
{
    echo "job      : $JOB  node $(hostname)  $(date -Is)"
    echo "ollama   : $(ollama --version 2>&1 | tail -1)"
    echo "gpu      : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | paste -sd';' || echo none)"
    echo "command  : $*"
    ollama list
} > "$RB_LOGS/provenance_$JOB.txt" 2>&1
cat "$RB_LOGS/provenance_$JOB.txt"

cd "$RB_REPO"
export PYTHONPATH=src TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
"$@"
