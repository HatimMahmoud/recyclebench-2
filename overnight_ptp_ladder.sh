#!/bin/bash
# PTP fine-tune scaling ladder, rungs 4 and 5 (+ one capacity x data cell).
#
# Registered in PTP_PROTOCOL_REGISTRATION_20261027.md, written BEFORE this data exists.
# Read it first: it fixes best-of-32 decoding, the frozen 86, Gate B (>= 20/86 with a
# Wilson lower bound over 10%) and the STOP RULE (close the arm if 18,000 pairs do not
# reach 15/86).
#
# Why this and not more compute: the gate-admissible rate went 1/86 -> 4/86 -> 8/86 as
# fine-tuning pairs went 347 -> 2,846 -> 5,912, with no sign of flattening, while a
# compute-matched control (same 347 pairs, 8x steps) got WORSE. The axis that moves is
# data diversity, and that axis is Ollama generation, not GPU. Steps are therefore held
# at ~4,434 on every rung (3 epochs at 12k, 2 at 18k -- within 1%) so the comparison is
# data, not training length.
#
# Held constant: pre-trained checkpoint, compact vocabulary (LOADED, never rebuilt, so
# eos 69367 and the embedding stay aligned), boundary token, the frozen 86, decode sweep.
# New prompts are unused tatsu-lab/alpaca no-input instructions (seed 20261027); their
# vocabulary coverage is 98.68% against 98.76% for the existing pool, so the new rungs
# are not handicapped relative to the old ones.
#
# Hits Ollama (qwen2.5:3b, the forward model being inverted). Writes ONLY to cache/ptp/,
# never to the shared study cache. Nothing here touches results/ or any dashboard.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 OLLAMA_KEEP_ALIVE=1h
PRE27=cache/ptp/local27m_25k/inverse_pretrained.pt
PRE92=cache/ptp/inverse_pretrained.pt
say() { echo; echo "=== $1  $(date '+%Y-%m-%d %H:%M:%S')"; }

# Refuse to fine-tune a rung whose pair count is not the registered one: a short ftdata
# (budget hit, Ollama stall) would otherwise train on whatever it managed and report it
# as the 12,000 rung, with steps no longer matched and the ladder quietly broken.
need() {
  python3 - "$1" << 'PY'
import sys
n = sum(1 for l in open("cache/ptp/ftpairs_qwen2.5_3b.jsonl") if l.strip())
want = int(sys.argv[1])
print(f"[guard] {n} pairs stored, rung needs {want}")
sys.exit(0 if n >= want else 1)
PY
}

say "L1 generate benign pairs up to a stored total of 12,000 (Ollama)"
python3 -u tools/ptp_build.py --phase ftdata --vocab-min-count 3 --workers 6 \
    --max-pairs 12000 --minutes 160 || echo "ftdata exited $?"

if need 12000; then
  say "L2 fine-tune 27M on 11,914 pairs, 3 epochs = 4,470 steps"
  python3 -u tools/ptp_build.py --phase finetune --vocab-min-count 3 --boundary \
      --size local --pretrained "$PRE27" --ft-epochs 3 --minutes 45 \
      --finetuned cache/ptp/inverse_boundary_ft12k.pt || echo "finetune exited $?"
  say "L3 score rung 4 with the registered sweep"
  python3 -u tools/ptp_decode_sweep.py --checkpoint cache/ptp/inverse_boundary_ft12k.pt \
      --vocab cache/ptp/compact_vocab_eop.json --eos 69367 --n-samples 32 \
      --out cache/ptp/decode_sweep_27m_ft12k.json || echo "sweep exited $?"
else
  echo "[guard] SKIPPING rung 4 -- pair count short of 12,000"
fi

say "L4 generate benign pairs up to a stored total of 18,000 (Ollama)"
python3 -u tools/ptp_build.py --phase ftdata --vocab-min-count 3 --workers 6 \
    --max-pairs 18000 --minutes 160 || echo "ftdata exited $?"

if need 18000; then
  say "L5 fine-tune 27M on 17,914 pairs, 2 epochs = 4,480 steps"
  python3 -u tools/ptp_build.py --phase finetune --vocab-min-count 3 --boundary \
      --size local --pretrained "$PRE27" --ft-epochs 2 --minutes 45 \
      --finetuned cache/ptp/inverse_boundary_ft18k.pt || echo "finetune exited $?"
  say "L6 score rung 5 with the registered sweep"
  python3 -u tools/ptp_decode_sweep.py --checkpoint cache/ptp/inverse_boundary_ft18k.pt \
      --vocab cache/ptp/compact_vocab_eop.json --eos 69367 --n-samples 32 \
      --out cache/ptp/decode_sweep_27m_ft18k.json || echo "sweep exited $?"

  say "L7 capacity x data: the 92M on the same 18,000 pairs, 2 epochs"
  python3 -u tools/ptp_build.py --phase finetune --vocab-min-count 3 --boundary \
      --size mid --pretrained "$PRE92" --ft-epochs 2 --minutes 80 \
      --finetuned cache/ptp/inverse_boundary_92m_ft18k.pt || echo "finetune exited $?"
  say "L8 score the 92M cell"
  python3 -u tools/ptp_decode_sweep.py --checkpoint cache/ptp/inverse_boundary_92m_ft18k.pt \
      --vocab cache/ptp/compact_vocab_eop.json --eos 69367 --n-samples 32 --size mid \
      --out cache/ptp/decode_sweep_92m_ft18k.json || echo "sweep exited $?"
else
  echo "[guard] SKIPPING rung 5 and the 92M cell -- pair count short of 18,000"
fi

say "L9 every PTP decode run side by side, and the registered decision"
python3 - << 'PY'
import json, glob, os, sys
sys.path.insert(0, "src")
from recyclebench.stats import wilson_interval

rows = []
for f in sorted(glob.glob("cache/ptp/decode_sweep_*.json")):
    d = json.load(open(f))
    rows.append((os.path.basename(f)[13:-5], d["greedy"]["token_f1"],
                 d.get("best_of", {}).get("32"),
                 d.get("best_of_precision", {}).get("32"),
                 d.get("best_of_cosine", {}).get("32"),
                 d.get("hit_rate_cos_075", {}).get("32")))
hdr = ("run", "greedy", "bo32 F1", "prec", "cos", "hit cos")
w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
print("  ".join(str(h).rjust(w[i]) for i, h in enumerate(hdr)))
for r in rows:
    print("  ".join(fmt(v).rjust(w[i]) for i, v in enumerate(r)))

print("\n--- registered decision (PTP_PROTOCOL_REGISTRATION_20261027.md section 3) ---")
p = "cache/ptp/decode_sweep_27m_ft18k.json"
if not os.path.exists(p):
    print("rung 5 did not produce a sweep; no decision can be read.")
else:
    hit = json.load(open(p)).get("hit_rate_cos_075", {}).get("32")
    k = round(hit * 86) if hit is not None else None
    if k is None:
        print("no cosine hit rate in the rung-5 sweep; no decision can be read.")
    else:
        iv = wilson_interval(k, 86)
        print(f"gate-admissible at 18,000 pairs: {k}/86 = {k/86:.1%}  "
              f"Wilson 95% [{iv.low:.1%}, {iv.high:.1%}]")
        print(f"  Gate B  (continue: >= 20/86 AND Wilson low > 10%) : "
              f"{'MET' if (k >= 20 and iv.low > 0.10) else 'not met'}")
        print(f"  Stop rule (close the arm if < 15/86)              : "
              f"{'TRIGGERED -- close the arm' if k < 15 else 'not triggered'}")
        print("\nprior rungs: 1/86 @347 · 4/86 @2,846 · 8/86 @5,912")
print("\nbaselines: token F1  generic 0.246 / echo 0.236   cosine  generic 0.193 / echo 0.713")
PY

say "L10 tests"
python3 -m pytest tests/ -q 2>&1 | tail -3
echo; echo "###############  finished  $(date '+%Y-%m-%d %H:%M:%S')  ###############"
