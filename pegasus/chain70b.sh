#!/usr/bin/env bash
# The 70B chain on SORRY-Bench (PREREG_70B_CHAIN_20261105.md, signed), run inside the gate image by run.sbatch.
# Same pipeline and defaults as the external-corpora run (pegasus/external_{canonical,recycle,rpe}.sh).
#
#   bash pegasus/chain70b.sh judge            merge the blind-generated side caches, judge every draw, canonical report
#   bash pegasus/chain70b.sh recycle <o:n>    one pair's recycle run (skips unless its judged |P^-| >= 5)
#   bash pegasus/chain70b.sh rpe              inversion for every pair with |P^-| >= 1 (invert, gate, closed loop)
#
# Every step resumes. One job per output: recycle jobs write only their own spec cache and run directory.
set -euo pipefail
STEP=${1:?judge | recycle <old:new> | rpe}
CFG=configs/chain70b.yaml
C=cache/chain70b
MAIN=$C/canonical.jsonl
STORE=$C/judge_sorrybench.jsonl
OUT=results/runs/chain70b
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p "$C" "$OUT"
grep -m1 "Status:" PREREG_70B_CHAIN_20261105.md | grep -q SIGNED || { echo "registration not signed: nothing runs"; exit 1; }
python3 tools/env_check.py --strict
EMPTY=$C/empty_manifest.jsonl; : > "$EMPTY"

case "$STEP" in
judge)
  touch "$MAIN"
  python3 tools/cache_consolidate.py --target "$MAIN" --sources $C/canonical__*.jsonl --apply | tail -4
  source pegasus/gpu_guard.sh
  python3 tools/judge_lineage.py --config "$CFG" --cache "$MAIN" --models llama2-70b llama3-70b llama31-70b llama33-70b \
      --scope canonical --behaviours data/sorrybench.csv --manifest "$EMPTY" --out "$STORE"
  python3 tools/chain70b_report.py
  ;;
recycle)
  PAIR=${2:?old:new}; O=${PAIR%%:*}; N=${PAIR##*:}
  python3 tools/chain70b_report.py --eligible recycle | grep -qx "$PAIR" \
      || { echo "$PAIR: judged |P^-| < 5 (or incomplete): recycle not applicable"; exit 0; }
  S=experiments/chain70b/sorrybench__${O}__${N}.yaml
  NAME=chain70b__sorrybench__${O}__${N}
  CACHE=$C/$NAME.jsonl
  RUN=$OUT/$NAME
  mkdir -p "$RUN"
  source pegasus/gpu_guard.sh
  if [ ! -f "$CACHE.seeded" ]; then
    touch "$CACHE"
    python3 tools/cache_consolidate.py --target "$CACHE" --sources "$MAIN" --apply | tail -2
    { echo "seeded_from $MAIN"; echo "source_sha256 $(sha256sum "$MAIN" | cut -d' ' -f1)"; echo "when $(date -Is)";
      echo "job ${SLURM_JOB_ID:-local}"; } > "$CACHE.seeded"
  fi
  export RECYCLEBENCH_TRIALS_DIR=$RUN/trials
  for pass in 1 2 3 4; do
    echo "===== [$(date +%T)] $NAME pass $pass ====="
    out=$(python3 -m recyclebench experiment run --spec "$S" --target cluster | tee /dev/stderr) \
        || { echo "!! $NAME: a stage failed"; exit 1; }
    echo "$out" | grep -q "^0 stage(s) ran" && break
  done
  python3 -m recyclebench experiment plan --spec "$S" --target cluster
  ;;
rpe)
  R=$OUT/rpe/sorrybench
  GATE=$C/rpe_gate.jsonl
  mkdir -p "$R"
  PAIRS=$(python3 tools/chain70b_report.py --eligible rpe | paste -sd, -)
  [ -n "$PAIRS" ] || { echo "no pair with a fix: PRR undefined everywhere"; exit 0; }
  source pegasus/gpu_guard.sh
  export HF_HUB_OFFLINE=1
  python3 tools/rpe_lineage.py --config "$CFG" --behaviours data/sorrybench.csv --manifest "$EMPTY" --cache "$MAIN" \
      --verdicts "$STORE" --pairs "$PAIRS" --benchmark sorrybench --out-root "$R" --minutes 300
  MANS=$(ls "$R"/rpe_manifest_*.jsonl 2>/dev/null || true)
  [ -n "$MANS" ] && python3 tools/rpe_gate.py --config "$CFG" --manifests $MANS --store "$GATE" --minutes 300
  STAMP=$(date +%Y%m%d_%H%M)
  for P in ${PAIRS//,/ }; do
    O=${P%%:*}; N=${P##*:}; M=$R/rpe_manifest_${O}__to__${N}.jsonl
    [ -s "$M" ] || { echo "-- $O->$N: empty manifest"; continue; }
    for PH in old judge-old equiv new judge-new score; do
      python3 tools/rpe_closedloop.py --config "$CFG" --rpe-manifest "$M" --old "$O" --new "$N" \
          --cache "$C/rpe_${O}__${N}.jsonl" --verdicts "$C/judge_rpe_${O}__${N}.jsonl" \
          --equiv-out "$C/rpe_${O}__${N}_equiv.json" --gate-store "$GATE" --phase "$PH" --minutes 300 \
          --out "$R/closedloop_${O}__to__${N}_gated_$STAMP.json" || echo "!! $O->$N/$PH exited $?"
    done
  done
  ;;
*) echo "unknown step $STEP"; exit 2 ;;
esac
echo "done $(date -Is)"
