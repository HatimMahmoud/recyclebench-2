#!/usr/bin/env bash
# Capacity vs corpus size: the last axis separating a failing local PTP from the paper.
#
# Established so far, corpus held fixed and vocabulary restricted:
#   15k seeds, 27M non-emb   token F1 0.137  cosine 0.247  val 4.838
#   25k seeds, 27M non-emb   token F1 0.123  cosine 0.231  val 4.888   <- data plateaued
# Baselines on the same held-out set: fixed generic string 0.246 / echo response 0.236
# (token F1), 0.193 / 0.713 (cosine). The model is still below a constant string.
#
# C1 holds the 25k corpus fixed and raises the model 27M -> 92M non-embedding (3.4x).
#    If it moves, capacity is the binding constraint and the cluster run is sized by it.
#    If it does not, corpus coverage is, and more GPU will not help.
# C2 reruns the offline analyses and dashboards afterwards.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
A70="$HOME/Downloads/all_local_results_llama3_1_70b 1.jsonl"
V="--vocab-min-count 3"
step(){ echo ""; echo "=== $1  $(date '+%F %T')"; shift; "$@" || echo "!! exited $? -- continuing"; }

# the 27M checkpoints cannot be loaded into a 92M model
mkdir -p cache/ptp/local27m_25k
mv cache/ptp/inverse_pretrained.pt cache/ptp/inverse_finetuned.pt cache/ptp/local27m_25k/ 2>/dev/null

M="python3 tools/ptp_build.py --forward-model qwen2.5:3b --size mid"
echo "###############  PTP capacity test  $(date '+%F %T')  ###############"
step "C1a pretrain 92M @25k seeds"  $M $V --phase pretrain --epochs 14 --minutes 300
step "C1b finetune"                 $M $V --phase finetune --ft-epochs 12 --minutes 40
step "C1c eval -> GATE A"           $M $V --phase eval --minutes 45 --gate-token-f1 0.30
cp cache/ptp/report.json cache/ptp/report_25000seeds_mid92m.json 2>/dev/null

step "C2a analysis"          python3 tools/thesis_analysis.py
step "C2b rpe repair"        python3 tools/rpe_repair.py
step "C2c lineage report"    python3 tools/rpe_lineage_report.py
step "C2d master dashboard"  python3 tools/master_dashboard.py --attack-70b-responses "$A70"
step "C2e pair dashboards"   python3 tools/pair_dashboard.py --attack-70b-responses "$A70"
step "C2f lineage dashboard" python3 tools/lineage_dashboard.py
step "C2g tests"             python3 -m pytest tests/ -q

echo ""
echo "=== every PTP run  $(date '+%F %T')"
python3 - <<'PY'
import json, pathlib
hdr = f"{'run':>28} {'vocab':>8} {'nonemb':>8} {'gate':>5} {'tokF1':>7} {'cos':>6} {'val':>6}"
print(hdr); print('-'*len(hdr))
for n in ("report_3500seeds","report_10000seeds","report_15000seeds",
          "report_15000seeds_vocab58k","report_25000seeds_vocab","report_25000seeds_mid92m"):
    f = pathlib.Path(f"cache/ptp/{n}.json")
    if not f.exists(): continue
    d=json.loads(f.read_text()); e,pt,c=d.get("eval",{}),d.get("pretrain",{}),d.get("config",{})
    g={True:"PASS",False:"FAIL"}.get(e.get("gate_passed"),"-")
    ne=pt.get("params_non_embedding",0)/1e6
    print(f"{n.replace('report_',''):>28} {c.get('model_vocab',151665):>8,} {ne:>7.0f}M {g:>5} "
          f"{e.get('token_f1',float('nan')):>7.3f} {e.get('cosine_mean',float('nan')):>6.3f} "
          f"{(pt.get('val_loss') or float('nan')):>6.3f}")
print("\nbaselines: token F1  generic 0.246 / echo 0.236   cosine  generic 0.193 / echo 0.713")
PY
echo "###############  finished  $(date '+%F %T')  ###############"
