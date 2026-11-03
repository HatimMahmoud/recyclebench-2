#!/usr/bin/env bash
# External-corpora run, step 3: the inversion arm (RPE) for every pair on every corpus, run inside the gate image by
# run.sbatch (pegasus/external_submit.sh submits it). Same pipeline and defaults as pegasus/scout_rpe.sh
# (rpe_lineage.py -> rpe_gate.py -> rpe_closedloop.py old/judge-old/equiv/new/judge-new/score; 8 candidates,
# inverter real.qwen25-14b, MiniLM equivalence 0.75, production gate), over the 19 pairs of tools/external_suite.py.
#
#   bash pegasus/external_rpe.sh <steps> <groups>
#     steps   comma list of invert, gate, loop                      (default invert,gate,loop)
#     groups  comma list of dense, scout, maverick for `loop`        (default dense,scout)
#
# invert and gate call only qwen2.5:14b and the gate models, and every OLD model is dense, so they fit an 80 GB card.
# The closed loop loads one model per phase: dense and scout pairs fit 80 GB, maverick pairs need 2 x B200 / H200,
# hence a separate loop job for them. Only pairs fully judged in the newest canonical report are inverted (a partial
# screen would give a partial P^-); a pair with empty P^- writes its plan and calls nothing (PRR undefined, never 0).
# Everything lands in cache/external/ and results/runs/external/rpe/. Resumable: rerun the same command.
set -euo pipefail
STEPS=",${1:-invert,gate,loop},"
LOOP_GROUPS=${2:-dense,scout}
CFG=configs/llama4_pegasus.yaml
C=cache/external
OUT=results/runs/external/rpe
GATE=$C/rpe_gate.jsonl
MIN=${RB_MINUTES:-300}                                  # per-phase budget, minutes (resumable past it)
CORPORA=$(python3 -c "import sys; sys.path.insert(0, 'tools'); import external_suite as e; print(' '.join(e.SELECTED))")
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1                                  # gate models are prefetched into $HF_HOME
mkdir -p "$C" "$OUT"
EMPTY=$C/empty_manifest.jsonl; : > "$EMPTY"
python3 tools/env_check.py --strict
source pegasus/gpu_guard.sh                     # refuse a silent CPU fallback

if [[ $STEPS == *,invert,* ]]; then
  echo "===== [$(date +%T)] 1. inversion manifests"
  for B in $CORPORA; do
    PAIRS=$(python3 tools/external_suite.py list --pairs --corpus "$B" --complete)
    [ -n "$PAIRS" ] || { echo "-- $B: no fully judged pair yet"; continue; }
    python3 tools/rpe_lineage.py --config "$CFG" --behaviours "data/$B.csv" --manifest "$EMPTY" \
        --cache $C/canonical.jsonl --verdicts "$C/judge_qwen_$B.jsonl" --pairs "$PAIRS" \
        --benchmark "$B" --out-root "$OUT/$B" --minutes "$MIN"
  done
fi

if [[ $STEPS == *,gate,* ]]; then
  echo "===== [$(date +%T)] 2. gate every candidate"
  MANS=$(ls "$OUT"/*/rpe_manifest_*.jsonl 2>/dev/null || true)
  [ -n "$MANS" ] && python3 tools/rpe_gate.py --config "$CFG" --manifests $MANS --store "$GATE" --minutes "$MIN"
fi

if [[ $STEPS == *,loop,* ]]; then
  echo "===== [$(date +%T)] 3. closed loop, groups $LOOP_GROUPS"
  STAMP=$(date +%Y%m%d_%H%M)
  for B in $CORPORA; do
    for G in ${LOOP_GROUPS//,/ }; do
      for PAIR in $(python3 tools/external_suite.py list --pairs --corpus "$B" --complete --group "$G" | tr ',' ' '); do
        O=${PAIR%%:*}; NEW=${PAIR##*:}
        M=$OUT/$B/rpe_manifest_${O}__to__${NEW}.jsonl
        [ -s "$M" ] || { echo "-- $B $O->$NEW: empty P^- (nothing to invert)"; continue; }
        for PH in old judge-old equiv new judge-new score; do
          python3 tools/rpe_closedloop.py --config "$CFG" --rpe-manifest "$M" --old "$O" --new "$NEW" \
              --cache "$C/rpe_${B}_${O}__${NEW}.jsonl" --verdicts "$C/judge_rpe_${B}_${O}__${NEW}.jsonl" \
              --equiv-out "$C/rpe_${B}_${O}__${NEW}_equiv.json" --gate-store "$GATE" \
              --phase "$PH" --minutes "$MIN" \
              --out "$OUT/$B/closedloop_${O}__to__${NEW}_gated_$STAMP.json" \
            || echo "!! $B $O->$NEW/$PH exited $?"
        done
      done
    done
  done
fi
echo "done $(date -Is)"
