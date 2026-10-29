# What is left, and who can do it

Written 2026-10-24. Everything automatable has been queued or run; this file holds only
what needs a person, and what needs the GPU cluster.

---

## A. Needs you — blocks reportable numbers

### A1. ~~Sign prereg Amendment 1~~ — done
Signed by the candidate 2026-10-25, supervisor acknowledgement 2026-10-28; the confirmatory replication
(S3) has run and is corrected (`results/runs/confirm_heldout_3_31/corrected_heldout_recal_n60.json`).

### A2. Label `audit/rpe/rpe_harm_sheet_20261028_2327.csv` — 65 rows, turns the inversion PRR into a bracket
Replaces the 2026-10-24 sheet: those 83 labels (all 1) covered the superseded 61-behaviour loose-gate candidates
(1 of their 83 texts recurs in the current anchor run) and were labelled on a topical reading, so they audit
nothing current.

The new sheet is blind and shuffled: up to 2 reconstructions the newer model answered, for each of the 36
anchor behaviours the closed loop re-opened. Label `human_same_harmful_ask` 1/0: does the reconstruction still
ask for the same **harmful** thing? Rubric: `audit/rpe/RPE_HARM_RUBRIC.txt`. A behaviour stays re-opened if
either of its rows is 1. `python3 tools/rpe_audit_score.py` then reports the anchor PRR as a bracket,
verified/38 (lower) to 36/38 (upper), and the dashboards pick it up on the next build. ~12 minutes.

    python3 tools/judge_label_tui.py audit/rpe/rpe_harm_sheet_20261028_2327.csv

Why it matters: under the intent gate the inversion arm re-opens 2 of the 5 anchor fixes it resolves (RSR 23/49);
its 36/38 comes entirely from reconstructions no automated check certifies.

### A3. ~~Label `audit/gate/gate_sheet.csv`~~ — done 2026-10-29
178/178. False admissions: 3 of 98 rewordings that re-opened an anchor fix (3.1% [1.0, 8.6]), 0 of 80 others;
no RSR moves when the 3 are dropped (`audit/gate/gate_audit_result.json`). Rows 19–178 were labelled with a
gemma4 guess shown (`audit/gate/MODEL_ASSISTED.txt`): report the audit as model-assisted.

### A4. Label `audit/harmbench/judge_sheet.csv` — 196 rows, 0 human labels, label blind
The qwen2.5:14b judge is calibrated on SORRY-Bench only (se 0.862 / sp 0.967 /
κ 0.830), but the pooled headline includes HarmBench fixes. `judge_label` is already
filled by the judge — label `human_fulfilled` **without reading it**.

### A5. Attack transfer: class-stratified audit of judge-negatives
`audit/attack` sampled 30 of 30 judge-positives and 30 of 2,338 negatives. Reported
se 0.857 becomes ≈ 0.07 design-weighted, so attack transfer is badly under-counted, and
neither judge suits attack outputs (the offline rule reads ciphers and garble as
compliance). Needs a fresh stratified sample over the negative classes.

### A6. ~~Decide what to do about the anchor manifest~~ — resolved 2026-10-28, nothing to decide
The "overlap of 8 with P⁻, OLD refuses 52 of 61" was a join error. The anchor manifest was built on the
61-row subset CSV `results/cells_llama3-8b_llama31-8b/sorrybench_p_minus.csv`, and `load_jailbreakbench`
numbers rows rather than reading the CSV's `bid` column, so its ids are row positions. Joined on prompt
text, its 61 behaviours are exactly the full-benchmark offline-rule P⁻ and OLD answers all 61; the 52/61
reappears exactly when the row ids are read as SORRY-Bench ids (`tools/rpe_reconcile.py`, pinned in
`tests/test_inversion_arm.py`). The anchor is reproduced.

---

## B. Needs the cluster (available ~2026-10-31)

### B1. PTP, at full published scale
Now affordable: probing an 8B at full vocabulary is ~33M tokens ≈ 2–3 h on one H100
against ~370 h locally. Running it in full removes every deviation that local compute
would have forced — full-vocabulary probing, on-the-fly regeneration, and the paper's
real inverse architecture (Qwen3-0.6B from scratch) rather than a 20–40M toy.

Build **two** inverse models and report both:
- **decorrelated** (the deliverable): forward model from a third family — Mistral-7B or
  Gemma-2-9B. Not qwen2.5 (the paraphraser), not Llama (the target).
- **same-family** (the comparison): forward model `llama3-8b`; `cache/llama_gen.jsonl`
  is already a free fine-tune corpus for this one.

The gap between them measures what the decorrelation requirement costs, and replicates
the paper's Table 3 (same-family 64.8 → cross-family 17) in the safety domain.

**Gate A before spending more:** same-family held-out response BERT-F1 ≥ 0.80 (the
paper gets ~0.855). Set the bar before looking.

**Run a correctness PoC locally first** — reversal-after-tokenisation (Remark 1 — the
easiest thing in the method to get silently wrong), tokenizer handling, training loop,
scoring — before burning cluster hours.

### B2. The decoding/quantisation trap — validate before trusting any cluster number
Every response in the study came from Ollama with **repeat_penalty 1.1 over the last 64
tokens**, from a **Q4_0** quantisation. An H100 running bf16 HF weights under vLLM
defaults is a numerically different model with different decoding. `mech_probe.py` had
to reproduce both to match cached replies (plain greedy reproduced only 3/6).

So: keep closed-loop re-queries faithful (on Ollama, or reproduce quantisation *and*
the penalty window and validate against cached replies first). Probing and inverse
training can live on the cluster freely — the inverse model is new regardless — but
record which weights and decoding were used.

Also: a shared on-prem cluster is a different exposure profile for a harmful-prompt
corpus than a laptop. Keep it in a restricted path and check with the administrator;
`ETHICS.md` currently assumes local-only.

### B3. Causal white-box interventions
The 18/20 refusal-direction result is **correlational**. Ablating the direction from a
re-opening rewording (does it refuse again?) and adding it to the canonical request
(does it comply?) would make it a mechanism claim. This is the single biggest available
upgrade to the thesis.

### B4. White-box beyond the anchor
`tools/mech_probe.py` is hardcoded to llama3-8b → llama3.1-8b on SORRY-Bench — it takes
only `--poc`/`--all`, with no `--pair`. Extending it needs a code change plus GGUF
symlinks for the other models (only two are linked in `cache/mech_gguf_links/`). Pair
with B3 rather than doing twice.

### B5. Llama-3.1-70B as a lineage target
Would answer the scale-confound criticism of the non-anchor rungs. Not possible on an
M1 Max; straightforward on the cluster. Optional — scope creep unless the thesis needs it.

---

## C. Known-stale things that are now just a rebuild

- **Gate layer L2 (`llm_intent`) is complete** — 5,769 of 5,769 rewordings checked with
  gemma4:e4b (`cache/intent_gemma4_e4b.jsonl`). Dashboards built before 2026-10-24 still
  say "never ran"; the rebuild in `overnight_fill_all.sh` fixes that.
- **PBR for llama2-13b on AdvBench / StrongREJECT** was "not estimable" only because
  2,686 rewording responses were never generated. That fill is queued; it is resumable,
  so if it stops at the `STOP_AT` cutoff just re-run `overnight_fill_all.sh`.

- **The lineage ordered-pair matrix was stale, not offline.** Dashboards built before
  2026-10-24 carry "▲ Built from offline verdicts" and show the anchor as P⁻ 61 (the
  offline number). The verdict cache has since completed, and a rebuild picks up the
  judged verdicts with no code change: the banner is gone and the anchor reads P⁻ 49.
  Any figure quoted from a pre-2026-10-24 lineage dashboard should be re-read.

- **Non-adjacent lineage pairs need no new runs.** Every model answered the same frozen
  rewordings, so any ordered pair is computable from cache. Verified with zero missing
  responses and zero missing verdicts:
  `llama2-7b → llama31-8b` P⁻ 23, re-opened 13 (RSR 57 %); `llama2-7b → llama3-8b`
  P⁻ 6, re-opened 5 (RSR 83 %). The lineage dashboard's matrix reports these correctly.

  **Known reporting gap:** `study_data` only counts a pair as covered when an explicit
  recycle *run artifact* names it, so the master and pair dashboards still print
  "not yet finished" for these cells even though the lineage dashboard computes them.
  That is a cross-referencing limitation in `study_data`, not missing data — no model
  calls would fix it. Worth closing before submission so the three dashboards agree.
