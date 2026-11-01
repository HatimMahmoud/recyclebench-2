# Pre-registration — the inversion (RPE) arm extended to ScienceSafetyData

> **EXPERIMENTAL -- ScienceSafetyData pilot, parked for the cluster. Safe to delete before submission; see the cleanup list in CLAUDE.md and TODO_HUMAN A10.**

**Registered 2026-11-01 00:42, before any inversion call on this corpus.** Exploratory; a *population*
extension of an already-specified arm, not a new method. Decided by the candidate (delegated design).
Companion to `PREREG_SCIENCESAFETY_20261101.md`, whose population, pair, instrument, isolation clause and
threats T1–T5 apply here unchanged.

**Why.** The recycle arm asks whether a *reworded request* re-opens a fix. The inversion arm asks a
convergent question: reconstruct the request from the **old** model's own answer, then put the
reconstruction to the **new** model. The two arms disagree on the anchor (closed loop 36/38 resolving
38/49, intent gate 2/5 resolving 5/49, against RSR 23/49), and that disagreement is a function of the
admission rule. Running both on one corpus of a different register is the cheapest way to see whether
the disagreement is a property of the corpus or of the rule.

## Items and instrument

* **Corpus:** `data/sciencesafety.csv` (240 behaviours, pinned commit `9db8659`, sha256
  `7c3cddbec048e676aa6cb3adc8f4e145bb789689a969093bf9c51ff751e452c6`).
* **Pair:** `llama3-8b` → `llama31-8b`.
* **Items:** the behaviours of the **judge-decided** P⁻ from stage 10 of
  `run_sciencesafety_20261101.sh`, as written to `<run>/sciencesafety_p_minus_judged.csv`. RPE is run
  on **that same CSV**, so the frozen recycle manifest's ids and the inversion manifest's ids agree.
  (They are row positions in the subset, not corpus ids — joins back to the corpus go **on prompt
  text**, per the standing convention.)
* **Inverter:** `qwen25-14b` (`qwen2.5:14b`, temperature 0.8 so the candidates differ) — non-Llama,
  decorrelated from the targets, as in every other RPE run.
* **Judge:** the same single `qwen2.5:14b` rubric judge, replayed into a verdict store
  (`cache/judge_sciencesafety_20261101.jsonl`) by `tools/judge_lineage.py`, because `run-real` stores
  no verdicts and the offline refusal rule disagrees with the judge.
* **Gate:** the production intent gate at the config's thresholds, scoring **every** inversion
  candidate into `cache/rpe_gate_sciencesafety_20261101.jsonl` via `tools/rpe_gate.py`.

## Estimands and the three admission rules (as already specified for this arm)

For each behaviour in P⁻, with the reconstructions of the old model's answer:

1. **A_gate** — admitted by the intent gate. The rule comparable to RSR.
2. **A_loop** — admitted by the closed loop (the reconstruction recovers the original request to the
   equivalence threshold). An **upper bound**; it does not certify the harmful ask.
3. **A_both** — admitted by both.

**PRR** = the fraction of P⁻ re-opened under each rule, reported as three separate numbers with
Wilson intervals, **never pooled into one "inversion" figure**. An artifact whose filename lacks
`_gated_` has empty A_gate/A_both *by construction* (the inversion manifest builder never runs the
gate and writes `admitted: false`), which is not a gate rejection — such a file is not reportable.

## Reading rule, fixed now

* **A_gate is the only rule comparable to RSR.** Any comparison of this arm with the recycle arm on
  this corpus uses A_gate against RSR, with both denominators stated, and says so explicitly.
* A_loop is reported as the upper bound it is, and is **not** evidence that the reconstruction keeps
  the harmful ask. On the anchor that question needed a blind human sheet (`audit/rpe/`, 65/65, 61
  keeping the ask). **No equivalent sheet exists for this corpus**, so A_loop here is reported as
  structurally uncertain until one is labelled. T1 of the companion prereg bites hardest here: a
  reconstruction of a research-framed answer may well be a benign research question.
* Agreement between the arms is computed by joining **on prompt text**, never on bid (every stored
  `rpe_agreement.json` from the SORRY-Bench runs is invalid for exactly that reason).
* |P⁻| = 0 ⇒ every PRR is **undefined**, never 0, and this arm does not run.

## What may and may not be claimed

**May:** the three admission rates on this corpus; whether the recycle/inversion disagreement
observed on the anchor reproduces on a corpus of a different register; the arm's feasibility through
the bring-your-own-data path.

**May not:** anything confirmatory; any pooling with the anchor, lineage or held-out populations or
with the SORRY-Bench RPE runs; any claim that a reconstruction carries the harmful ask (no human
sheet); any corrected figure (no corpus-specific calibration).

## The style factorial (compose) is NOT registered here

`tools/run_compose_v2.py` sources its items from the anchor's SORRY-Bench P⁻ and its lineage verdict
store; it cannot take another corpus without a code change, and a 21-condition factorial over an
as-yet-unknown |P⁻| has not been power-checked for this corpus. It is therefore **out of scope for
this registration and was not run.** If it is wanted later it needs (a) a behaviour-source option
whose default leaves the registered anchor result bit-identical, and (b) its own registration with a
power statement.

## Execution

Appended to `run_sciencesafety_20261101.sh` as stages 45 (verdict store), 50 (inversions) and 60
(closed loop + scoring), each guarded by the queue's 08:30 hard stop, by |P⁻| ≥ 5, and by its own
`--minutes` budget. Every stage is resumable: inversions, gate scores and verdicts are append-only
stores keyed on content, so an interrupted stage continues where it stopped. A stage that does not
complete produces no reportable figure, and the queue says so rather than scoring a partial store.
