#!/usr/bin/env bash
# External-corpora run, step 2: the recycle arm (RSR) for every eligible pair of one job group, run inside the gate
# image by run.sbatch (pegasus/external_submit.sh submits one job per group).
#
#   bash pegasus/external_recycle.sh <group>      # dense | scout | maverick
#
# Eligible = fully judged in the newest results/runs/external/canonical_*.json with |P^-| >= 5 (the Scout suite's
# rule; tools/external_suite.py list --eligible). The other pairs are NOT APPLICABLE, which the completion matrix
# records. Per spec, as pegasus/scout_recycle.sh:
#   1. seed its own cache once from cache/external/canonical.jsonl (verbatim, append-only), so the screen re-judges
#      the canonical draws already paid for instead of regenerating them;
#   2. `experiment run` until a pass runs nothing (the driver plans once per invocation; every stage resumes);
#   3. per-variant trials (RECYCLEBENCH_TRIALS_DIR) for the style breakdown.
# Group shapes (docs/pegasus/LESSONS.md §4b.6): dense fits any 80 GB card; scout judges inline beside a 67 GB model,
# so it needs one card > 80 GB or 2 x 80 GB; maverick (245 GB) needs 2 x B200 / H200.
set -euo pipefail
GROUP=${1:?usage: external_recycle.sh dense|scout|maverick}
SEED_FROM=cache/external/canonical.jsonl
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python3 tools/env_check.py --strict
[ -s "$SEED_FROM" ] || { echo "no $SEED_FROM: run pegasus/external_canonical.sh first"; exit 1; }
SPECS=$(python3 tools/external_suite.py list --specs --group "$GROUP" --eligible)
if [ -z "$SPECS" ]; then
  echo "group $GROUP: no eligible pair (every judged |P^-| < 5 or incomplete); nothing to run"
  exit 0
fi
echo "group $GROUP, eligible specs:"; echo "$SPECS"
source pegasus/gpu_guard.sh                     # refuse a silent CPU fallback

for S in $SPECS; do
  NAME=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$S")
  CACHE=cache/external/$NAME.jsonl
  RUN=results/runs/external/$NAME
  mkdir -p "$RUN"
  if [ ! -f "$CACHE.seeded" ]; then
    touch "$CACHE"
    python3 tools/cache_consolidate.py --target "$CACHE" --sources "$SEED_FROM" --apply
    { echo "seeded_from $SEED_FROM"; echo "source_lines $(wc -l < "$SEED_FROM")";
      echo "source_sha256 $(sha256sum "$SEED_FROM" | cut -d' ' -f1)"; echo "when $(date -Is)";
      echo "job ${SLURM_JOB_ID:-local}"; } > "$CACHE.seeded"
  fi
  export RECYCLEBENCH_TRIALS_DIR=$RUN/trials
  for pass in 1 2 3 4; do
    echo "===== [$(date +%T)] $NAME pass $pass ====="
    # a failed stage records nothing; move on to the next spec and let a resubmission resume it
    out=$(python3 -m recyclebench experiment run --spec "$S" --target cluster | tee /dev/stderr) \
        || { echo "!! $NAME: a stage failed (rc=$?), continuing with the next spec"; break; }
    echo "$out" | grep -q "^0 stage(s) ran" && break
  done
  python3 -m recyclebench experiment plan --spec "$S" --target cluster
done
echo "done $(date -Is)"
