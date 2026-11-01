#!/usr/bin/env bash
# Load one model once, then time a few benign prompts: load seconds and steady tokens/s on this GPU.
# Run it before a long screen, under a short time cap, to catch load timeouts and slow GPUs cheaply:
#   sbatch -p H100 --gpus=1 --mem=96G --time=00:30:00 pegasus/run.sbatch bash pegasus/speed_check.sh
#   RB_SPEED_MODEL=llama4:17b-maverick-128e-instruct-q4_K_M sbatch --export=ALL -p B200 --gpus=2 ...
# The first request after a load is slow (one-off warm-up); read the later rows.
set -euo pipefail
M=${RB_SPEED_MODEL:-llama4:17b-scout-16e-instruct-q4_K_M}
PROMPTS=("Say OK." "Explain how photosynthesis works." "Write a short story about a lighthouse keeper."
         "Describe the history of the printing press." "Explain how a refrigerator works.")
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
for i in "${!PROMPTS[@]}"; do
    curl -s "$RECYCLEBENCH_OLLAMA_URL/api/chat" -d "{\"model\":\"$M\",\"stream\":false,
        \"options\":{\"temperature\":0,\"num_predict\":300},
        \"messages\":[{\"role\":\"user\",\"content\":\"${PROMPTS[$i]}\"}]}" \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
s = lambda n, t: n / max(t, 1) * 1e9
print('run $i  load_s=%.1f  prompt_tok=%s (%.0f/s)  gen_tok=%s (%.1f/s)' % (
    d.get('load_duration', 0) / 1e9, d.get('prompt_eval_count'), s(d.get('prompt_eval_count', 0), d.get('prompt_eval_duration', 1)),
    d.get('eval_count'), s(d.get('eval_count', 0), d.get('eval_duration', 1))))"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
done
ollama ps
grep -a -E "offloaded|Load failed" "$RB_LOGS/ollama_${SLURM_JOB_ID:-local}.log" | tail -5 || true
