#!/usr/bin/env bash
# Llama 4 Scout known-attack transfer arm (Wei et al. classes, liuyanchen1015 HF corpus), run inside the
# gate image (it carries `datasets`) by run.sbatch:
#
#   RB_IMAGE=$RB_SCRATCH/images/recyclebench_gate_<rev>.sqsh \
#   sbatch --export=ALL -p RTXB6000,H200,B200 --gpus=1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
#          --job-name=rb-scout-attack pegasus/run.sbatch bash pegasus/scout_attack.sh
#
# Identical method to the valid 8B/70B artifacts (results/runs/attack_transfer/llama31-8b_judged_original.json,
# llama31-70b_judged.json): every attack class the corpus holds, all 300 base behaviours, the config's
# qwen2.5:14b judge scoring each reply against the ORIGINAL goal (never the encoded prompt), no intent
# gate. A separate estimand from RSR/PRR: never pooled. Raw: the judge's design-weighted sensitivity on
# attack output is ~0.07 (THESIS_KB §4.3), so transfer is under-counted.
#
# The judge runs inline beside Scout (67 GB + 9 GB), hence a >80 GB GPU. Responses are cached in
# cache/llama4_scout/attack_transfer.jsonl (never the shared cache/attack_transfer.jsonl); verdicts are
# not cached by this tool, so a resubmission after a timeout re-judges (minutes) but never re-generates.
set -euo pipefail
CFG=configs/llama4_pegasus.yaml
OUT=results/runs/llama4_scout/attack_transfer
CACHE=cache/llama4_scout/attack_transfer.jsonl
DS=liuyanchen1015/Llama-3.2-1B-Instruct_jailbreak_responses_with_judgment
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p "$OUT" "$(dirname "$CACHE")"; touch "$CACHE"
python3 tools/env_check.py --strict

# every class except the no-transform baseline, as the corpus defines them (recorded beside the result)
CLASSES=$(python3 - "$DS" << 'EOF'
import sys
from datasets import load_dataset
d = load_dataset(sys.argv[1])["train"]
print(" ".join(sorted({r["jailbreak_prompt_name"] for r in d} - {"original_prompt"})))
EOF
)
echo "$CLASSES" | tr ' ' '\n' > "$OUT/classes.txt"
echo "classes ($(wc -l < "$OUT/classes.txt")): $CLASSES"

python3 tools/run_attack_transfer.py --config "$CFG" --model llama4-scout --classes $CLASSES \
    --cache "$CACHE" --judge-against original \
    --out "$OUT/llama4-scout_judged_original_$(date +%Y%m%d_%H%M).json"
echo "done $(date -Is)"
