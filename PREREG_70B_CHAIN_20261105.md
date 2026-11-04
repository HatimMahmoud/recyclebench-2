# Registration: the 70B matched-scale chain on SORRY-Bench (Pegasus)

**Status: DRAFT, unsigned.** Written 2026-11-05. Sign below and commit before any reply of this chain is judged.

**Blind generation before signing (disclosed).** To save cluster time, the canonical replies of the four 70B models
were generated from 2026-11-05 (night) before this registration was signed: `tools/fill_responses.py` only, into
side caches `cache/chain70b/canonical__<model>.jsonl`. No judge was run on them, no reply was read, and no figure,
count of compliance or P⁻ was computed until signature. Generation is fully specified by §2 (model, prompt as
written, seed-derived draws, temperature 0), so it cannot have been steered by results. The job IDs are in the run log.

**Designation: exploratory, post hoc.** `prereg.md` registered the 70B chain (Llama 2 70B → Llama 3 70B → Llama 3.1
70B → Llama 3.3 70B); signed deviation D2 (Amendment 1) moved it to future work because it did not fit the 32 GB
machine. Running it now, after the 7–8B results are known, makes it post hoc. Figures are raw judge output, are
never pooled with the anchor, the lineage or the held-out replication, and carry no verdict against π*.

## 1. Population

SORRY-Bench base, 440 behaviours (`data/sorrybench.csv`, rebuilt on Pegasus 2026-11-04 from
`sorry-bench/sorry-bench-202503`; identity verified: 1,320/1,320 lineage cache hits for each of the six dense models,
so ids, order and seeds are the anchor study's).

## 2. Models and instrument

| key | Ollama tag | Pegasus ID |
|---|---|---|
| llama2-70b | llama2:70b | e7f6c06ffef4 |
| llama3-70b | llama3:70b | 786f3184aec0 |
| llama31-70b | llama3.1:70b | 711a9e8463af |
| llama33-70b | llama3.3:70b | a6eb4748fd29 |

Ollama default quantisation (not matched across models; Amendment 2 D7). No Mac reference exists for these IDs.
Config `configs/chain70b.yaml`: identical to `configs/llama4_pegasus.yaml` in every science setting (seed 20260101,
k = 3, judge qwen2.5:14b rubric, paraphraser qwen2.5:3b, 8 rewordings over 6 styles, gate 0.75 / 0.85 / 0.60).

## 3. Pairs and arms

Three consecutive matched-scale pairs: 2-70B → 3-70B, 3-70B → 3.1-70B, 3.1-70B → 3.3-70B.

1. **Canonical screen** of all four models (k = 3, every draw judged); answer rates (Wilson) and each pair's 2×2.
2. **Recycle (RSR)** for every pair with judged |P⁻| ≥ 5 (driver floor), as in the cluster run: own frozen manifest,
   both models re-screened, raw RSR with Wilson and McNemar as descriptives; unresolved fixes never count as robust.
3. **Inversion (RPE)** for every pair with |P⁻| ≥ 1: A_gate, A_loop (upper bound only), A_both.
4. **Not run:** compose v2, the mechanistic probe (dense Llama-3 8B code), PBR at model level, human calibration
   (the audit stage exports blind sheets; until labelled, every figure is raw).

**Reading.** Each matched pair is read as in the anchor (high RSR ⇒ surface patching; low ⇒ consistent with
generalisation), descriptively. Agreement or disagreement with the 7–8B anchor (3-8B → 3.1-8B) is reported side by
side; no pooling, no test against π*.

## 4. Threats

- **T1, scale within the chain:** matched by design; quantisation differs by model (as served).
- **T2, judge range:** the judge was calibrated on 7–8B replies to SORRY-Bench; 70B replies may be longer.
- **T3, environment:** CUDA, Ollama 0.40.12; no Mac reference for the 70B weights.
- **T4, post hoc:** chosen after the 7–8B results; exploratory only.

## Sign-off

- Candidate: ______________________  date: __________
- Supervisor: _____________________  date: __________
