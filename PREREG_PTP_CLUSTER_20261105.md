# Registration: PTP proof of concept on Pegasus, as an alternative inverter for the RPE arm

**Status: SIGNED 2026-11-05** (candidate and supervisor). Written 2026-11-05. Sign below and commit before the held-out evaluation runs.

**Training before signing (disclosed).** To save cluster time, the build's corpus, fine-tuning pairs and model weights are
produced from 2026-11-05 before signature (`tools/ptp_build.py` phases probe, ftdata, pretrain, finetune). Nothing in
those phases reads the held-out pairs' inversion quality; the `eval` phase and `tools/ptp_decode_sweep.py` run only after
signature. Job IDs are in the run log.

**Designation: exploratory proof of concept.** It does not replace the local PTP arm, its registration
(`PTP_PROTOCOL_REGISTRATION_20261027.md`, the qwen2.5:3b ladder) or any figure from it, and it writes only under its own
directory, `cache/ptp_cluster_llama3_8b/`.

## 1. Why

The RPE arm reconstructs a request from the OLDER model's answer, with an LLM inverter (qwen2.5:14b, 8 candidates). Its
intent gate admitted 0 of 967 reconstructions on ScienceSafetyData and few elsewhere. PTP (Suhail et al., arXiv:2607.29378)
trains an inverse language model from scratch, by previous-token prediction, on the target model's own outputs. If it
reconstructs requests closer to the original, it could replace the LLM inverter in RPE. This build tests whether it does,
on benign data first, before it is ever pointed at the safety corpus.

## 2. Build

| | |
|---|---|
| target (forward) model | **llama3:8b**, the OLDER model of the anchor pair, whose answers RPE inverts |
| tokenizer | Qwen/Qwen2.5-0.5B (as the local arm; reversal works on any consistent tokenization) |
| inverse model | `--size paper` (Qwen3-0.6B-like decoder), trained from scratch, `--boundary` (end-of-prompt token) |
| synthetic corpus | 30,000 seed tokens × 256 tokens (the paper probes the whole vocabulary; reduced for cluster time) |
| fine-tuning pairs | 6,000 benign Alpaca no-input instructions (`tatsu-lab/alpaca`, seed 20261105) with llama3:8b's replies |
| held-out | frozen by the finetune phase at its first run (`heldout.json` in the build directory, hash in the run log) |
| serving | Ollama 0.40.12; `OLLAMA_NUM_PARALLEL=8` for this build only (synthetic data, never compared token for token) |

## 3. Evaluation (after signature)

Exactly the local arm's registered protocol (PTP_PROTOCOL_REGISTRATION_20261027 §2): best of **N = 32** sampled
candidates, end-of-prompt stopping, 40-token budget, greedy reported alongside.
- **Primary outcome: gate-admissible rate**, the share of held-out items with ≥ 1 of 32 candidates at cosine ≥ 0.75 (the
  study's intent-gate threshold), Wilson 95 %.
- Secondary: best-of-32 and greedy token F1, exact match.
- **Comparison:** the local arm's best point (8/86 at 5,912 pairs, qwen2.5:3b target), side by side, never pooled.
- **Decision (descriptive):** PTP is worth wiring into RPE as an alternative inverter only if the gate-admissible rate
  reaches the local arm's Gate B (≥ 20 % with a Wilson lower bound above 10 %). Wiring it in would need its own
  registration; this PoC never inverts a safety-corpus answer.

## 4. Threats

- **Reduced probing** (30k of ~150k seeds) narrows what pre-training covers (the paper's Appendix E).
- **Tokenizer mismatch** with the target model: the inverse model's token reversal is not the target's own.
- **Benign-only:** a good result on Alpaca does not show the model inverts refusal-adjacent answers.

## Sign-off

- Candidate: **Hatim Mahmoud**  date: **2026-11-05**
- Supervisor (acknowledged): _signed off_  date: **2026-11-05**

_Both sign-offs were given by the candidate in session on 2026-11-05, while the build jobs were still queued (no corpus, weights or evaluation existed)._
