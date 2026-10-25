#!/usr/bin/env bash
# S3: the CONFIRMATORY replication on the held-out set, now that Amendment 1 is
# signed and stamped. Waits for the PTP build to release Ollama first -- both want the
# server, and run-real alternates target and judge models.
#
# This is a run-real: it writes to the shared cache. Nothing else may write while it
# runs (CLAUDE.md: never run two run-real at once).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false

while pgrep -f "ptp_build.py" >/dev/null; do
  echo "waiting for the PTP build ... $(date '+%T')"; sleep 180
done
echo "PTP finished; Ollama is free  $(date '+%F %T')"
echo "###############  S3 confirmatory replication  $(date '+%F %T')  ###############"
bash overnight_stack.sh S3 || echo "!! S3 exited $?"
echo "=== post-run analysis  $(date '+%F %T')"
python3 tools/thesis_analysis.py || echo "!! analysis $?"
echo "###############  finished  $(date '+%F %T')  ###############"
