#!/bin/bash
# Compositional 2-factor transform arm. Spec: PREREG_COMPOSE_20261027.md (exploratory).
#
# Nine arms over the 61 anchor-P^- behaviours, 3 candidates each, run WHOLE and in priority
# order so the 7h budget leaves complete arms rather than a partial slice of all of them:
#   para · rpe · para__rpe · rpe__para        <- intent-gated, the new question + its controls
#   attack · para__attack · attack__para · rpe__attack · attack__rpe   <- separate estimand
#
# Everything is append-only in cache/compose/ and resumable: rerun the same command to continue.
# Target responses go to cache/compose_targets.jsonl, NOT the shared study cache.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h
echo "=== start $(date '+%Y-%m-%d %H:%M:%S') ==="
python3 -u tools/run_compose.py --minutes 420 --candidates 3
echo "=== tests $(date '+%H:%M:%S') ==="
python3 -m pytest tests/ -q 2>&1 | tail -3
echo "=== done $(date '+%Y-%m-%d %H:%M:%S') ==="
