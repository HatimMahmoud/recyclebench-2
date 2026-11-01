#!/usr/bin/env bash
# One CPU job that downloads what the Scout arms need, so the GPU jobs never wait on the network and can
# run with HF_HUB_OFFLINE=1. Run inside the gate image by run.sbatch (its private Ollama does the pull):
#
#   RB_IMAGE=$RB_SCRATCH/images/recyclebench_gate_<rev>.sqsh \
#   sbatch --export=ALL -p RTXA6000 --gpus=0 --cpus-per-task=4 --mem=24G --time=01:00:00 \
#          --job-name=rb-prefetch pegasus/run.sbatch bash pegasus/scout_prefetch.sh
#
#   qwen2.5:3b        the configured paraphraser (~2 GB), the only model of the recycle arm not yet pulled
#   gate models       all-MiniLM-L6-v2, microsoft/deberta-large-mnli, roberta-large (BERTScore, lang=en)
#   attack corpus     liuyanchen1015/Llama-3.2-1B-Instruct_jailbreak_responses_with_judgment
# All into $OLLAMA_MODELS / $HF_HOME on /netscratch. Idempotent.
set -euo pipefail
ollama pull qwen2.5:3b
ollama list
python3 - << 'EOF'
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bert_score import BERTScorer
from datasets import load_dataset
SentenceTransformer("all-MiniLM-L6-v2")
for m in ("microsoft/deberta-large-mnli",):
    AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m)
s = BERTScorer(lang="en", rescale_with_baseline=False)
print("bertscore model:", s.model_type)
d = load_dataset("liuyanchen1015/Llama-3.2-1B-Instruct_jailbreak_responses_with_judgment")["train"]
print("attack corpus rows:", len(d), "classes:", len({r["jailbreak_prompt_name"] for r in d}),
      "behaviours:", len({r["prompt_name"] for r in d}))
EOF
du -sh "$HF_HOME" "$OLLAMA_MODELS"
