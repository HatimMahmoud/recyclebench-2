#!/usr/bin/env bash
# Offline refresh of the inversion (RPE) arm: no Ollama calls.
#
#   1  score every pair x benchmark closed-loop run with the gate store
#        -> closedloop_<old>__to__<new>_gated_<stamp>.json next to its manifest
#   2  per-benchmark convergent-validity reports (RSR vs PRR under A_loop / A_both / A_gate)
#        -> lineage_report_<stamp>.json in each run root
#   3  reconcile the dedicated arm with the compose run's rpe arm -> results/analysis_rpe_reconcile_<stamp>/
#   4  (--export) the harmful-ask audit sheet for the anchor -> audit/rpe/rpe_harm_sheet_<stamp>.csv
#
# Run after run_rpe_complete.sh. Never overwrites: every output carries the stamp.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1
STAMP="$(date +%Y%m%d_%H%M)"
CFG=configs/real_llama_lineage_all.yaml
GATE=cache/rpe_gate.jsonl
L32=results/runs/20260801_165721_llama32-1b_to_3b_recycle
ADJ="llama2-7b:llama2-13b llama2-13b:llama3-8b llama3-8b:llama31-8b llama31-8b:llama32-1b llama32-1b:llama32-3b"
POST="llama2-7b:llama3-8b llama2-7b:llama31-8b"
EXPORT=0; [ "${1:-}" = "--export" ] && EXPORT=1

score() {  # <cache-prefix> <run-root> <old> <new>
  local P="$1" R="$2" O="$3" N="$4" M="$2/rpe_manifest_${3}__to__${4}.jsonl"
  [ -s "$M" ] || return 0
  python3 tools/rpe_closedloop.py --config "$CFG" --rpe-manifest "$M" --old "$O" --new "$N" \
      --cache "cache/${P}_${O}__${N}.jsonl" --verdicts "cache/judge_${P}_${O}__${N}.jsonl" \
      --equiv-out "cache/${P}_${O}__${N}_equiv.json" --gate-store "$GATE" --phase score \
      --out "$PWD/$R/closedloop_${O}__to__${N}_gated_${STAMP}.json" 2>&1 | grep -E "A_(gate|loop|both)|wrote|exists|have no gate" \
      || echo "!! score ${R} ${O}->${N} exited"
}

echo "######## 1 score  ($STAMP)"
for P in $ADJ; do O="${P%%:*}"; N="${P##*:}"
  score rpe_lin results/runs/rpe_lineage "$O" "$N"
  for B in harmbench advbench strongreject; do score "rpe_$B" "results/runs/rpe_bench/$B" "$O" "$N"; done
done
for P in $POST; do O="${P%%:*}"; N="${P##*:}"
  score rpe_lin results/runs/rpe_lineage_posthoc "$O" "$N"
  score rpe_harmbench results/runs/rpe_bench_posthoc/harmbench "$O" "$N"
done

echo ""
echo "######## 2 reports"
python3 tools/rpe_lineage_report.py --out "$PWD/results/runs/rpe_lineage/lineage_report_${STAMP}.json"
[ -d results/runs/rpe_lineage_posthoc ] && python3 tools/rpe_lineage_report.py \
    --pairs llama2-7b:llama3-8b,llama2-7b:llama31-8b --rpe-root results/runs/rpe_lineage_posthoc \
    --out "$PWD/results/runs/rpe_lineage_posthoc/lineage_report_${STAMP}.json"
for B in harmbench advbench strongreject; do
  python3 tools/rpe_lineage_report.py --behaviours "data/$B.csv" --manifest "$L32/manifest_$B.jsonl" \
      --verdicts "cache/judge_qwen_$B.jsonl" --rpe-root "results/runs/rpe_bench/$B" \
      --out "$PWD/results/runs/rpe_bench/$B/lineage_report_${STAMP}.json"
done
[ -d results/runs/rpe_bench_posthoc/harmbench ] && python3 tools/rpe_lineage_report.py \
    --behaviours data/harmbench.csv --manifest "$L32/manifest_harmbench.jsonl" \
    --verdicts cache/judge_qwen_harmbench.jsonl --pairs llama2-7b:llama3-8b,llama2-7b:llama31-8b \
    --rpe-root results/runs/rpe_bench_posthoc/harmbench \
    --out "$PWD/results/runs/rpe_bench_posthoc/harmbench/lineage_report_${STAMP}.json"

echo ""
echo "######## 3 reconcile"
python3 tools/rpe_reconcile.py --out "$PWD/results/analysis_rpe_reconcile_${STAMP}/reconcile.json"

if [ "$EXPORT" = 1 ]; then
  echo ""
  echo "######## 4 harmful-ask audit sheet"
  python3 tools/rpe_audit_export.py
fi
echo "######## done  $(date '+%F %T')"
