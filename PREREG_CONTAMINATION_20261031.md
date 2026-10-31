# Pre-registration — benchmark contamination probe (exposé §8)

**Registered 2026-10-31 01:25, before any probe data exists** (3-prompt smoke test to a scratch store, not used).
Exploratory; it delivers the contamination audit the exposé promises (§8). Decided by the candidate.

* **Question:** do the probed models reproduce benchmark prompts verbatim, i.e. were the benchmarks in their
  training data? Contamination would make a "fix" on that benchmark possibly a memorised refusal of that wording.
* **Benchmarks:** AdvBench (Jul 2023), HarmBench (Feb 2024), StrongREJECT (Feb 2024), SORRY-Bench (Jun 2024) and the
  held-out set `data/heldout_jbb_guided.csv`; duplicate prompts counted once. **Controls:** SORRY-Bench and the
  held-out set, released after the pre-training cutoffs of every probed model (Llama 3 / 3.1 report data to end of
  2023; Llama 2 to Sep 2022). Note that post-training (instruction/safety) data may be later; that is part of what the
  probe can show.
* **Models:** llama3-8b, llama31-8b, llama2-7b (in this order).
* **Probe:** the first ⌊n/2⌋ words of the prompt as raw text (Ollama `/api/generate`, `raw: true`, no chat template),
  greedy, `num_predict` 32, Ollama's other defaults. Prompts with a suffix under 5 words are skipped.
* **Rule:** *reproduced* = the continuation repeats the next min(8, |suffix|) words exactly (lower-cased, punctuation
  dropped). Also reported: mean fraction of the suffix matched in order.
* **Reading:** a benchmark is flagged as contaminated for a model when its reproduction rate's Wilson 95% interval lies
  entirely above the higher of the two control rates' upper bounds. Descriptive, no multiplicity correction.
* Results: `results/runs/contamination_<stamp>/summary.{json,txt}` (numbers only; continuations stay in
  `cache/contamination/`).

*Edit log: 2026-10-31 01:27, after launch: the stated registration time was corrected from 01:30 to 01:25 (the time of writing). No other change.*
