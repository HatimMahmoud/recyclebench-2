#!/usr/bin/env bash
# Source from a GPU job script inside the image: refuse to run unless Ollama really offloads to the GPU.
# An Ollama build without kernels for the node's GPU (e.g. a newer architecture) falls back to the CPU
# silently, and a 6-12 h allocation then crawls. Loads the judge model (every arm needs it anyway) and
# requires `ollama ps` to report 100% GPU for it.
#   source pegasus/gpu_guard.sh            # model: RB_GUARD_MODEL, default qwen2.5:14b
_gm=${RB_GUARD_MODEL:-qwen2.5:14b}
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null \
    || { echo "GPU GUARD: no nvidia-smi in this job"; exit 3; }
curl -sf "http://$OLLAMA_HOST/api/generate" -d "{\"model\": \"$_gm\", \"prompt\": \"ok\", \"stream\": false, \"options\": {\"num_predict\": 1}}" > /dev/null \
    || { echo "GPU GUARD: $_gm did not answer"; exit 3; }
ollama ps
ollama ps | grep -F "$_gm" | grep -q "100% GPU" \
    || { echo "GPU GUARD: $_gm is not fully on the GPU (CPU fallback?); stopping before the run"; exit 3; }
grep -m3 -E "inference compute|library=" "$RB_LOGS/ollama_${SLURM_JOB_ID:-local}.log" 2>/dev/null || true
echo "GPU GUARD: ok ($_gm 100% GPU)"
