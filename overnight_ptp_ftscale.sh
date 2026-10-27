#!/bin/bash
# Does more fine-tuning data close the SEMANTIC gap?
#
# The boundary-token fix lifted lexical recovery sharply (best-of-32 token F1
# 0.217 -> 0.335) but left the number that matters flat: only 1 of 86 held-out items
# has a candidate the intent gate would admit at cosine >= 0.75. So PTP still
# contributes nothing usable, and the question is what would move the semantic side.
#
# `tools/ptp_diagnose.py` localised the break. Teacher-forced accuracy by depth into
# the prompt (measured backwards from the response boundary) is
#     d=1 0.326 · d=2 0.047 · d<=4 0.256 · d<=8 0.274 · d<=16 0.392
# The model continues a prompt backwards competently and fails at the HAND-OFF from
# response into prompt. That hand-off is exactly what the fine-tuning stage teaches,
# and it was taught with 347 pairs. This takes it to ~2,850.
#
# Held constant: pre-trained checkpoint, compact vocabulary, boundary token, held-out
# 86 (now frozen in ptp_build so adding pairs cannot move it), decode sweep.
# Changed: the number of fine-tuning pairs. The pre-training corpus is untouched, so
# if Appendix E is right that pre-training establishes the mapping, this should NOT
# rescue it -- which is itself the result worth having before the cluster run.
#
# Hits Ollama (qwen2.5:3b, the forward model being inverted). Writes only to
# cache/ptp/, never to the shared study cache.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1
STAMP=$(date +%Y%m%d_%H%M)
say() { echo; echo "=== $1  $(date '+%Y-%m-%d %H:%M:%S')"; }

say "F1 generate responses for the new prompts (Ollama, threaded)"
python3 -u tools/ptp_build.py --phase ftdata --vocab-min-count 3 --workers 6 \
    --minutes 75 || echo "ftdata exited $?"

say "F2 fine-tune from the pre-trained 27M, with the boundary token"
python3 -u tools/ptp_build.py --phase finetune --vocab-min-count 3 --boundary \
    --size local --ft-epochs 12 --minutes 40 || echo "finetune exited $?"

say "F3 score with the same sweep as every other run"
python3 -u tools/ptp_decode_sweep.py --checkpoint cache/ptp/inverse_finetuned.pt \
    --vocab cache/ptp/compact_vocab_eop.json --eos 69367 --n-samples 32 \
    --out cache/ptp/decode_sweep_27m_ftscale.json || echo "sweep exited $?"

say "F4 every PTP decode run side by side"
python3 - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("cache/ptp/decode_sweep_*.json")):
    d = json.load(open(f))
    b32 = d.get("best_of", {}).get("32")
    rows.append((os.path.basename(f)[13:-5], d.get("eos") is not None,
                 d["greedy"]["token_f1"], b32,
                 d.get("best_of_precision", {}).get("32"),
                 d.get("best_of_recall", {}).get("32"),
                 d.get("best_of_cosine", {}).get("32"),
                 d.get("hit_rate_f1_030", {}).get("32"),
                 d.get("hit_rate_cos_075", {}).get("32")))
hdr = ("run", "stop", "greedy", "bo32 F1", "prec", "recall", "cos", "hit F1", "hit cos")
w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
print("  ".join(str(h).rjust(w[i]) for i, h in enumerate(hdr)))
for r in rows:
    print("  ".join(fmt(v).rjust(w[i]) for i, v in enumerate(r)))
print("\nbaselines: token F1  generic 0.246 / echo 0.236   cosine  generic 0.193 / echo 0.713")
print("the gate admits a preimage at cosine >= 0.75; 'hit cos' is the share of the 86")
print("held-out items with at least one candidate over that line.")
PY

say "F5 tests"
python3 -m pytest tests/ -q 2>&1 | tail -3
echo; echo "###############  finished  $(date '+%Y-%m-%d %H:%M:%S')  ###############"
