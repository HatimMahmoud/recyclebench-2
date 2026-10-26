#!/usr/bin/env bash
# Does restricting the vocabulary fix PTP at subset scale?
#
# The 15k-seed run scored token F1 0.078 -- below a fixed generic string (0.246) and
# below echoing the response (0.236). Diagnosis: probing a SUBSET leaves 40% of
# embedding rows without a single gradient, while the embedding is 74% of the model,
# so the softmax ranks 60k classes it was never trained to rank.
#
# E1 retrains on the SAME corpus with the vocabulary restricted to tokens seen >= 3
#    (58k ids, 98.7% token coverage). Same data, same seeds, one variable changed.
# E2 then extends the corpus to 25k seeds and repeats, so the scaling curve continues
#    under whichever vocabulary condition E1 shows is right.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h OLLAMA_NUM_PARALLEL=6
export PYTORCH_ENABLE_MPS_FALLBACK=1
B="python3 tools/ptp_build.py --forward-model qwen2.5:3b --size local"
V="--vocab-min-count 3"
step(){ echo ""; echo "=== $1  $(date '+%F %T')"; shift; "$@" || echo "!! exited $? -- continuing"; }

# the model shape changes with the vocabulary, so full-vocab weights cannot be loaded
mkdir -p cache/ptp/fullvocab_15k
mv cache/ptp/inverse_pretrained.pt cache/ptp/inverse_finetuned.pt cache/ptp/fullvocab_15k/ 2>/dev/null
rm -f cache/ptp/compact_vocab.json

echo "###############  PTP vocabulary experiment  $(date '+%F %T')  ###############"
step "E1a pretrain @15k seeds, 58k vocab"  $B $V --phase pretrain --epochs 20 --minutes 110
step "E1b finetune"                        $B $V --phase finetune --ft-epochs 12 --minutes 25
step "E1c eval -> GATE A"                  $B $V --phase eval --minutes 35 --gate-token-f1 0.30
cp cache/ptp/report.json cache/ptp/report_15000seeds_vocab58k.json 2>/dev/null

step "E2a probe to 25k seeds"              $B --phase probe --n-seeds 25000 --workers 6 --minutes 150
rm -f cache/ptp/compact_vocab.json cache/ptp/inverse_pretrained.pt cache/ptp/inverse_finetuned.pt
step "E2b pretrain @25k seeds, restricted" $B $V --phase pretrain --epochs 16 --minutes 120
step "E2c finetune"                        $B $V --phase finetune --ft-epochs 12 --minutes 25
step "E2d eval -> GATE A"                  $B $V --phase eval --minutes 35 --gate-token-f1 0.30
cp cache/ptp/report.json cache/ptp/report_25000seeds_vocab.json 2>/dev/null

echo ""
echo "=== PTP results so far  $(date '+%F %T')"
python3 - <<'PY'
import json, pathlib
hdr = f"{'run':>26} {'modelvocab':>11} {'gate':>6} {'token_f1':>9} {'cosine':>8} {'val_loss':>9}"
print(hdr); print('-'*len(hdr))
for n in ("report_3500seeds", "report_10000seeds", "report_15000seeds",
          "report_15000seeds_vocab58k", "report_25000seeds_vocab"):
    f = pathlib.Path(f"cache/ptp/{n}.json")
    if not f.exists(): continue
    d = json.loads(f.read_text()); e, pt, c = d.get("eval", {}), d.get("pretrain", {}), d.get("config", {})
    g = {True:"PASS", False:"FAIL"}.get(e.get("gate_passed"), "—")
    print(f"{n.replace('report_',''):>26} {c.get('model_vocab', 151665):>11,} {g:>6} "
          f"{e.get('token_f1', float('nan')):>9.3f} {e.get('cosine_mean', float('nan')):>8.3f} "
          f"{(pt.get('val_loss') or float('nan')):>9.3f}")
print("\nbaselines on the same held-out set: fixed generic string 0.246 · echo response 0.236")
PY
echo "###############  finished  $(date '+%F %T')  ###############"
