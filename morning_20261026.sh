#!/usr/bin/env bash
# Morning 2026-10-26. Picks up when the overnight stack ends (~07:00) and must be
# finished by 13:00, so every phase is budgeted to land well before that.
#
# The night gives Gate A at two corpus sizes (3.5k and 10k seeds). Whatever the
# outcome, the useful next move is the same: add a third point at 15k. If it passes
# somewhere, the curve sizes the cluster run; if it fails everywhere with the loss
# converging, a three-point monotone curve is a far stronger negative than one pass/fail.
# The divergence guard means a broken run now aborts instead of faking a result.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h OLLAMA_NUM_PARALLEL=6
export PYTORCH_ENABLE_MPS_FALLBACK=1
A70="$HOME/Downloads/all_local_results_llama3_1_70b 1.jsonl"
B="python3 tools/ptp_build.py --forward-model qwen2.5:3b --size local"
step(){ echo ""; echo "=== $1  $(date '+%F %T')"; shift; "$@" || echo "!! exited $? -- continuing"; }

while pgrep -f "overnight_stack_20261025.sh" >/dev/null; do
  echo "waiting for the overnight stack ... $(date '+%T')"; sleep 300
done
echo "###############  morning  $(date '+%F %T')  ###############"
python3 - <<'PY'
import json, pathlib
for n in ("report_3500seeds", "report_10000seeds"):
    f = pathlib.Path(f"cache/ptp/{n}.json")
    if not f.exists():
        print(f"{n}: absent"); continue
    d = json.loads(f.read_text())
    e, pt = d.get("eval", {}), d.get("pretrain", {})
    print(f"{n}: gate={e.get('gate_passed')} token_f1={e.get('token_f1')} "
          f"cosine={e.get('cosine_mean')} pretrain_val_loss={pt.get('val_loss')}")
PY

step "M1 probe to 15k seeds"  $B --phase probe --n-seeds 15000 --workers 6 --minutes 150
step "M2 pretrain @15k"       $B --phase pretrain --epochs 20 --minutes 150
step "M3 finetune @15k"       $B --phase finetune --ft-epochs 12 --minutes 25
step "M4 eval -> GATE A @15k" $B --phase eval --minutes 35 --gate-token-f1 0.30
cp cache/ptp/report.json cache/ptp/report_15000seeds.json 2>/dev/null

step "M5a analysis"          python3 tools/thesis_analysis.py
step "M5b rpe repair"        python3 tools/rpe_repair.py
step "M5c lineage report"    python3 tools/rpe_lineage_report.py
step "M5d master dashboard"  python3 tools/master_dashboard.py --attack-70b-responses "$A70"
step "M5e pair dashboards"   python3 tools/pair_dashboard.py --attack-70b-responses "$A70"
step "M5f lineage dashboard" python3 tools/lineage_dashboard.py
step "M5g tests"             python3 -m pytest tests/ -q

echo ""
echo "=== PTP scaling curve  $(date '+%F %T')"
python3 - <<'PY'
import json, pathlib
print(f"{'seeds':>8} {'gate':>6} {'token_f1':>9} {'cosine':>8} {'val_loss':>9}")
for n, s_ in (("report_3500seeds", 3500), ("report_10000seeds", 10000), ("report_15000seeds", 15000)):
    f = pathlib.Path(f"cache/ptp/{n}.json")
    if not f.exists():
        continue
    d = json.loads(f.read_text()); e, pt = d.get("eval", {}), d.get("pretrain", {})
    g = {True: "PASS", False: "FAIL"}.get(e.get("gate_passed"), "—")
    print(f"{s_:>8} {g:>6} {e.get('token_f1', float('nan')):>9.3f} "
          f"{e.get('cosine_mean', float('nan')):>8.3f} {pt.get('val_loss') or float('nan'):>9.3f}")
PY
echo "###############  finished  $(date '+%F %T')  ###############"
