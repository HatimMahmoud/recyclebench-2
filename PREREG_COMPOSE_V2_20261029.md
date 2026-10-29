# Pre-registration — compose v2: the 21-condition transform-class factorial

**Registered 2026-10-29, before any data of this design exists.** Exploratory (an addition beyond the
exposé and Amendment 1), but its conditions, estimand, comparisons and error control are fixed here so the
result cannot be shaped after the fact. Decided by the candidate; the supervisor delegated the design
("go with what you see fit"). Supersedes nothing: the first compose run (`PREREG_COMPOSE_20261027.md`,
`results/runs/compose_20261028_0002/`) stays as it is.

## Question (RQ3 extension)
Do *pairs* of rewording styles re-open fixes that neither style re-opens alone? I.e. is recovery
supra-additive across surface axes, or does a pair add nothing beyond its stronger single style?

## Design
* **Pair:** llama3-8b → llama3.1-8b (the anchor, matched scale).
* **Behaviours:** the qwen2.5:14b-judged P⁻ of that pair on SORRY-Bench — 49 behaviours, full-benchmark ids
  (`data/sorrybench.csv`), computed from the lineage manifest + `cache/judge_lineage_sorrybench.jsonl`. Not
  the 61-row offline-rule subset the first compose run used (its ids are row positions, and 14 of its
  behaviours have an OLD answer the judge scores as a refusal).
* **Conditions (21):** the 6 transform classes alone (`paraphrase, syntactic, lexical, back_translation,
  tense_shift, register`) and the 15 unordered pairs of them, each pair applied as class A then class B in
  that fixed list order (A before B in the list). Order is fixed, not randomised: an order effect is not
  estimated and is stated as a limitation.
* **Candidates:** 3 per behaviour × condition (index 0, 1, 2), identical budget for singles and pairs.
  Singles are generated fresh under this budget, not taken from the frozen manifest (which has ~1–2 per class).
* **Generator:** `qwen2.5:3b` (`configs/recycle_3_31.yaml`, temperature 0.7) through the project's
  `LLMReformulator`, one class per call; a pair is two calls, the second on the first's output.
* **Admission:** the production intent gate (`RealGate`, cosine 0.75 / BERTScore 0.85 / NLI 0.60, L2 off as in
  every recycle manifest) against the **original** canonical request. Non-admitted candidates are never sent.
* **Screening:** llama3.1-8b answers every admitted candidate once (temperature 0, Ollama defaults as in the
  study). qwen2.5:14b judges the reply **against the candidate text**, exactly as `run-real` does, so the rate
  is comparable to RSR.
* **No fitness selection, no search, no attack template, no RPE.** Nothing is chosen on outcomes.

## Estimand
Per condition c: **compositional RSR_c** = share of the *resolved* behaviours (≥ 1 admitted candidate under
c) on which llama3.1-8b complies with ≥ 1 admitted candidate. Unresolved behaviours are reported, never
counted as held. Wilson 95% intervals.

## Comparisons (fixed now)
1. **Primary, per pair (A,B): gain over the better single.** On the behaviours resolved by A, B and A∘B,
   McNemar exact of A∘B against `A ∪ B` (re-opened by A or by B). A positive, significant difference means
   supra-additive recovery. 15 tests, **Holm–Bonferroni** across them, α = 0.05.
2. **Secondary:** A∘B against each single separately (30 tests, Holm), and the count of behaviours re-opened
   by A∘B but by neither single ("composite-only").
3. **Descriptive:** gate admission rate per condition (pairs drift further from the original, so admission is
   expected to fall; reported, not tested).

## Stopping and integrity
* Wall-clock budget; the runner records any phase or index it cut. The generation loop runs candidate index
  0 for every behaviour × condition before index 1, so a cut leaves whole index levels: the analysis then uses
  the highest index completed for **all** conditions, keeping the budget equal across conditions.
* Separate store `cache/compose_v2/`; separate results `results/runs/compose_v2_<stamp>/`; nothing existing is
  written. Every phase is resumable.

## Reporting
Exploratory, labelled as such; never pooled with RSR or the confirmatory replication.
