#!/usr/bin/env bash
# Build a reduced-scale PTP inverse model locally, end to end.
#
# FEASIBILITY TEST, not the deliverable. The paper probes every vocabulary token
# (~38M tokens/epoch, ~1 h on an A100); this probes a subset on an M1 Max. Appendix E
# shows the pre-training stage is what establishes the inverse mapping, so a subset
# narrows exactly what makes the method work -- hence Gate A, set before looking.
#
# Validated entirely on BENIGN data (Alpaca). The forward model qwen2.5:3b is also the
# project's paraphraser, so this build is NOT decorrelated and cannot back a reported
# convergent check; the cluster build uses a third family (TODO_HUMAN.md B1).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h
export PYTORCH_ENABLE_MPS_FALLBACK=1

B="python3 tools/ptp_build.py --forward-model qwen2.5:3b --size local"
step() { echo ""; echo "=== $1  (budget $2 min)  $(date '+%F %T')"; shift 2; "$@" || echo "!! exited $? -- continuing"; }

echo "###############  PTP local build  $(date '+%F %T')  ###############"
step "probe: vocabulary subset"       180 $B --phase probe    --n-seeds 3500 --minutes 180
step "ftdata: benign prompt/response"  40 $B --phase ftdata    --minutes 40
step "pretrain: reversed synthetic"   300 $B --phase pretrain --epochs 30 --minutes 300
step "finetune: reversed pairs"        40 $B --phase finetune --ft-epochs 12 --minutes 40
step "eval: GATE A"                    45 $B --phase eval      --minutes 45 --gate-token-f1 0.30
echo ""
echo "=== tests  $(date '+%F %T')"
python3 -m pytest tests/ -q 2>&1 | tail -3
echo "###############  finished  $(date '+%F %T')  ###############"
