#!/usr/bin/env bash
# External-corpora run, step 1: canonical screen of every model on GuidedBench, ScienceSafetyData and JBB-Behaviors
# (PREREG_EXTERNAL_CLUSTER_20261103.md). Run inside the gate image by run.sbatch; pegasus/external_submit.sh
# submits it. A thin loop over pegasus/canonical_screen.sh (k = 3 canonical draws, then the qwen2.5:14b judge)
# with the run's own cache and verdict stores under cache/external/, never the shared ones.
#
#   bash pegasus/external_canonical.sh <steps> <models>
#     steps   comma list of merge, generate, judge, report        (default generate,judge,report)
#     models  all | dense | scout | maverick                        (default all; tools/external_suite.py)
#
# One job can do everything (`generate,judge,report all`). To generate in parallel, each model set writes its own
# cache (cache/external/canonical__<models>.jsonl, so two jobs never append to one JSONL); a later
# `merge,judge,report all` folds those into cache/external/canonical.jsonl and judges every model there.
# Resumable: every step skips what the caches already hold.
set -euo pipefail
STEPS=",${1:-generate,judge,report},"
SET=${2:-all}
CFG=configs/llama4_pegasus.yaml
C=cache/external
MAIN=$C/canonical.jsonl
CORPORA=$(python3 -c "import sys; sys.path.insert(0, 'tools'); import external_suite as e; print(' '.join(e.CORPORA))")
case "$SET" in
  all)      MODELS=$(python3 -c "import sys; sys.path.insert(0, 'tools'); import external_suite as e; print(' '.join(e.MODELS))") ;;
  dense)    MODELS=$(python3 -c "import sys; sys.path.insert(0, 'tools'); import external_suite as e; print(' '.join(e.DENSE))") ;;
  scout)    MODELS=llama4-scout ;;
  maverick) MODELS=llama4-maverick ;;
  *) echo "models must be all, dense, scout or maverick (got $SET)"; exit 2 ;;
esac
GEN_CACHE=$MAIN; [ "$SET" = all ] || GEN_CACHE=$C/canonical__$SET.jsonl
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$C" results/runs/external
touch "$MAIN" "$GEN_CACHE"
echo "steps ${STEPS//,/ } | models $MODELS | corpora $CORPORA"
python3 tools/env_check.py --strict
python3 tools/external_suite.py data             # exit 1 on an absent or unregistered corpus: nothing runs

if [[ $STEPS == *,merge,* ]]; then
  SRC=$(ls $C/canonical__*.jsonl 2>/dev/null || true)
  if [ -n "$SRC" ]; then
    echo "===== [$(date +%T)] merge $SRC -> $MAIN"
    python3 tools/cache_consolidate.py --target "$MAIN" --sources $SRC --apply
  else
    echo "merge: no canonical__*.jsonl side caches"
  fi
fi

if [[ $STEPS == *,generate,* || $STEPS == *,judge,* ]]; then
  source pegasus/gpu_guard.sh                    # refuse a silent CPU fallback
fi

if [[ $STEPS == *,generate,* ]]; then
  for M in $MODELS; do
    RB_CONFIG=$CFG RB_BENCH="$CORPORA" RB_MODEL=$M RB_CACHE=$GEN_CACHE RB_STORE_DIR=$C RB_STEPS=generate \
        bash pegasus/canonical_screen.sh
  done
fi

if [[ $STEPS == *,judge,* ]]; then
  [ "$SET" = all ] || { echo "judge runs on the merged cache: use models = all"; exit 2; }
  for M in $MODELS; do
    RB_CONFIG=$CFG RB_BENCH="$CORPORA" RB_MODEL=$M RB_CACHE=$MAIN RB_STORE_DIR=$C RB_STEPS=judge \
        bash pegasus/canonical_screen.sh
  done
fi

if [[ $STEPS == *,report,* ]]; then
  # counts only; pegasus/external_recycle.sh and external_rpe.sh read the newest report for eligibility
  python3 tools/external_suite.py report --config "$CFG" --cache "$MAIN"
fi
echo "done $(date -Is)"
