#!/usr/bin/env bash
# Overnight 2026-10-25.
#
# S1 re-runs the PTP inverse model on the EXISTING corpus with the fixed init. The
#    first run's Gate A failure was a divergence bug (He init on residual-writing
#    projections, no warmup), not a scale result -- every weight was NaN. Probing is
#    already done, so this is cheap.
# S2 exports the judge-calibration sheet for the confirmatory run, which finished RAW.
# S3 extends the probe toward 10k seeds, S4 re-trains on it: two corpus sizes give a
#    scaling point instead of a single pass/fail, and size the cluster run.
# S5 rebuilds every analysis and dashboard.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h OLLAMA_NUM_PARALLEL=6
export PYTORCH_ENABLE_MPS_FALLBACK=1
A70="$HOME/Downloads/all_local_results_llama3_1_70b 1.jsonl"
B="python3 tools/ptp_build.py --forward-model qwen2.5:3b --size local"
step(){ echo ""; echo "=== $1  $(date '+%F %T')"; shift; "$@" || echo "!! exited $? -- continuing"; }

echo "###############  overnight  $(date '+%F %T')  ###############"

step "S1a PTP pretrain (fixed init, existing corpus)" $B --phase pretrain --epochs 30 --minutes 110
step "S1b PTP finetune"                               $B --phase finetune --ft-epochs 12 --minutes 30
step "S1c PTP eval -> GATE A @3.5k seeds"             $B --phase eval --minutes 40 --gate-token-f1 0.30
cp cache/ptp/report.json cache/ptp/report_3500seeds.json 2>/dev/null

step "S2 judge-export for the confirmatory run" \
  python3 -m recyclebench judge-export --config configs/confirm_heldout_3_31.yaml \
    --behaviours data/heldout_jbb_guided.csv --cache cache/llama_gen.jsonl --n-bundles 60

step "S3 probe to 10k seeds"  $B --phase probe --n-seeds 10000 --workers 6 --minutes 260
step "S4a PTP pretrain @10k"  $B --phase pretrain --epochs 24 --minutes 170
step "S4b PTP finetune @10k"  $B --phase finetune --ft-epochs 12 --minutes 30
step "S4c PTP eval -> GATE A @10k seeds" $B --phase eval --minutes 40 --gate-token-f1 0.30
cp cache/ptp/report.json cache/ptp/report_10000seeds.json 2>/dev/null

step "S5a analysis"          python3 tools/thesis_analysis.py
step "S5b rpe repair"        python3 tools/rpe_repair.py
step "S5c lineage report"    python3 tools/rpe_lineage_report.py
step "S5d master dashboard"  python3 tools/master_dashboard.py --attack-70b-responses "$A70"
step "S5e pair dashboards"   python3 tools/pair_dashboard.py --attack-70b-responses "$A70"
step "S5f lineage dashboard" python3 tools/lineage_dashboard.py
step "S5g tests"             python3 -m pytest tests/ -q

echo ""
echo "###############  finished  $(date '+%F %T')  ###############"
