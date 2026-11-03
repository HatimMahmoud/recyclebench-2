#!/usr/bin/env bash
# The whole external-corpora run on Pegasus in one command (PREREG_EXTERNAL_CLUSTER_20261103.md,
# docs/EXTERNAL_CORPORA_CLUSTER.md): GuidedBench, ScienceSafetyData and JBB-Behaviors on all eight models.
# Prints the job chain (default) or submits it with explicit dependencies (this Slurm ignores SBATCH_DEPENDENCY
# and refuses a dependency on a job it no longer knows: docs/pegasus/LESSONS.md §4b.2). Run it on the login node
# from the repo: it only calls sbatch; every job runs in the gate image through pegasus/run.sbatch.
#
#   bash pegasus/external_submit.sh                          # print the plan, submit nothing
#   bash pegasus/external_submit.sh --submit                 # submit the full chain
#   bash pegasus/external_submit.sh --submit --only J,R-dense,F   # resubmit some jobs (after a timeout or a fix);
#                                                            # a dependency on a job not in --only is dropped
#
#   data  D        check the three CSVs against the registered fingerprints (builds JBB + ScienceSafety if absent)
#   canon G-dense  generate the six dense models        G-scout  Llama 4 Scout       G-mav  Llama 4 Maverick
#         J        merge the three caches, judge every model, write the canonical report (P^- of all 19 pairs)
#   arms  R-dense  R-scout  R-mav     recycle (RSR), eligible pairs of each group
#         P-prep   invert + gate every candidate     P-loop  closed loop, dense + scout     P-mav  closed loop, maverick
#   end   F        style arcs, inversion reports, completion matrix (runs even when an arm failed)
#
# RB_IMAGE defaults to the validated gate image of the Scout suite; RB_PARTS_* override the partition lists.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RB_SCRATCH:=/netscratch/$USER/recyclebench}"
export RB_SCRATCH
export RB_IMAGE=${RB_IMAGE:-$RB_SCRATCH/images/recyclebench_gate_8fdd17c.sqsh}
source pegasus/env.sh

MODE=print ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --submit) MODE=submit ;;
    --only) ONLY=",$2,"; shift ;;
    *) echo "unknown argument $1 (--submit, --only a,b,c)"; exit 2 ;;
  esac
  shift
done

P80=${RB_PARTS_80:-H100,H100-PCI,A100-80GB,A100-RP,RTXB6000,H200,B200}   # one 80 GB card is enough
PBIG=${RB_PARTS_BIG:-RTXB6000,H200,B200}                                  # one card > 80 GB (Scout + inline judge)
PMAV=${RB_PARTS_MAV:-B200,H200}                                           # two cards: Maverick is 245 GB
PCPU=${RB_PARTS_CPU:-RTXA6000}
CPU=(-p "$PCPU" --gpus=0 --cpus-per-task=4 --mem=24G)      # unpacking the 10.9 GB image needs ~16 GB (LESSONS §4b.4)
G80=(-p "$P80" --gpus=1 --cpus-per-task=8 --mem=64G)
GBIG=(-p "$PBIG" --gpus=1 --cpus-per-task=8 --mem=64G)
GMAV=(-p "$PMAV" --gpus=2 --cpus-per-task=8 --mem=300G)

if [ "$MODE" = submit ]; then
  [ -f "$RB_IMAGE" ] || { echo "no image at $RB_IMAGE (set RB_IMAGE to the gate image)"; exit 1; }
  git diff --quiet HEAD -- tools pegasus configs experiments src || { echo "uncommitted changes under code paths: commit first, so every job records a real revision"; exit 1; }
  mkdir -p "$RB_LOGS"
fi
LOG=$RB_LOGS/external_submit_$(date +%Y%m%d_%H%M%S).txt
declare_ids=""                                             # name=id pairs, bash 3 compatible

id_of() { echo "$declare_ids" | tr ' ' '\n' | sed -n "s/^$1=//p" | tail -1; }

# job NAME TIME DEPTYPE "DEPS" RESOURCES... -- COMMAND...
job() {
  local name=$1 time=$2 dtype=$3 deps=$4; shift 4
  local res=() cmd=()
  while [ "$1" != "--" ]; do res+=("$1"); shift; done; shift
  cmd=("$@")
  if [ -n "$ONLY" ] && [[ $ONLY != *",$name,"* ]]; then return 0; fi
  local ids=() d
  for d in ${deps//,/ }; do
    local x; x=$(id_of "$d")
    [ -n "$x" ] && ids+=("$x")
  done
  local args=(--export=ALL --job-name="rb-ext-$name" --time="$time" "${res[@]}")
  if [ ${#ids[@]} -gt 0 ]; then
    local joined; joined=$(IFS=:; echo "${ids[*]}")
    args+=(--dependency="$dtype:$joined")
  fi
  if [ "$MODE" = submit ]; then
    local jid; jid=$(sbatch --parsable "${args[@]}" pegasus/run.sbatch "${cmd[@]}")
    jid=${jid%%;*}
    declare_ids="$declare_ids $name=$jid"
    printf "%-8s %-10s %s\n" "$name" "$jid" "${cmd[*]}" | tee -a "$LOG"
  else
    declare_ids="$declare_ids $name=<$name>"
    echo "sbatch ${args[*]} pegasus/run.sbatch ${cmd[*]}"
  fi
}

job D       00:30:00 afterok ""                     "${CPU[@]}"  -- python3 tools/external_suite.py data --build
job G-dense 08:00:00 afterok D                      "${G80[@]}"  -- bash pegasus/external_canonical.sh generate dense
job G-scout 06:00:00 afterok D                      "${G80[@]}"  -- bash pegasus/external_canonical.sh generate scout
job G-mav   08:00:00 afterok D                      "${GMAV[@]}" -- bash pegasus/external_canonical.sh generate maverick
job J       05:00:00 afterok G-dense,G-scout,G-mav  "${G80[@]}"  -- bash pegasus/external_canonical.sh merge,judge,report all
job R-dense 08:00:00 afterok J                      "${G80[@]}"  -- bash pegasus/external_recycle.sh dense
job R-scout 08:00:00 afterok J                      "${GBIG[@]}" -- bash pegasus/external_recycle.sh scout
job R-mav   08:00:00 afterok J                      "${GMAV[@]}" -- bash pegasus/external_recycle.sh maverick
job P-prep  06:00:00 afterok J                      "${G80[@]}"  -- bash pegasus/external_rpe.sh invert,gate
job P-loop  08:00:00 afterok P-prep                 "${G80[@]}"  -- bash pegasus/external_rpe.sh loop dense,scout
job P-mav   06:00:00 afterok P-prep                 "${GMAV[@]}" -- bash pegasus/external_rpe.sh loop maverick
job F       01:00:00 afterany R-dense,R-scout,R-mav,P-loop,P-mav "${CPU[@]}" -- bash pegasus/external_final.sh

if [ "$MODE" = submit ]; then
  echo "submitted; ids in $LOG. Check: squeue -u $USER; the matrix appears as results/runs/external/matrix_<stamp>.md"
else
  echo; echo "(plan only: add --submit to submit)"
fi
