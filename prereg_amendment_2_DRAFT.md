## Amendment 2 -- 2026-10-31 (draft; for signature)

Amendment 1 (signed 2026-10-25, supervisor 2026-10-28) recorded the deviations up to 2026-10-22 and registered one
confirmatory replication. This amendment closes the record before the write-up: (B1) it updates the Amendment 1
deviations whose numbers or status have changed, (B2) it lists every analysis added since, each with the document that
registered it and its status, (B3) it names the three populations every figure comes from and fixes one estimator
convention that the code applied two ways, (B4) it registers the five human audits still to be labelled, with their
sampling designs frozen by hash **before any label exists**, and (B5) it fixes the analyses added for the write-up,
including the re-specified RQ3 model and the full-configuration stamp. Nothing here changes the confirmatory replication (A2), which has run as registered.

**Status of results.** Confirmatory: only A2 (held-out replication, llama3-8b → llama3.1-8b). Everything else is
exploratory, including every analysis in B2.

### B1. Updates to the Amendment 1 deviations

| # | Amendment 1 said | status 2026-10-31 |
|---|---|---|
| D3 | single judge qwen2.5:14b; calibration n = 59 bundles, se 0.859, sp 0.968, κ 0.83 | calibration repaired to **60 bundles** (one `00` cell in bundle jbb-0003 had silently dropped it; repaired 2026-10-27, backup `*.bak_before_typo_fix_20261027_1719`): se **0.862** [0.758, 0.963], sp **0.968** [0.903, 1.000], κ **0.833**, design-weighted 26 / 34, admitted. Transfer to HarmBench checked on its own blind sheet (2026-10-29, 196 rows, 60 bundles, design-weighted 113 / 264): se 0.865 [0.720, 1.000], sp 1.000, κ 0.933. Pooled anchor RSR corrected with either calibration: 62.2% vs 62.3%. Transfer to AdvBench / StrongREJECT remains an assumption (judged P⁻ 7 and 2). |
| D5 | gate not yet audited | **audited**, `audit/gate/gate_sheet.csv`, 178/178 (A3.2 design): 3 of 98 admitted rewordings that re-opened a fix changed the request (3.1% [1.0, 8.6]), 0 of 80 other admitted (0% [0, 4.6]), design-weighted ≈ 0.05%; **no RSR moves** when the 3 are dropped. **Deviation:** rows 1–18 blind; rows 19–178 labelled with a gemma4:e4b guess displayed (candidate's decision 2026-10-29, `audit/gate/MODEL_ASSISTED.txt`). The labeller overruled 10 of gemma's 13 "changed" guesses and never marked 0 a row gemma marked 1. Reported as model-assisted. |
| D6 | anchor corrected RSR 0.553 [0.365, 0.742] after the design-weighting fix | after the D3 repair: anchor RSR **0.551** [0.364, 0.738] (`results/runs/recycle_3_31_sorrybench/corrected_sorrybench_recal_n60.json`, `tools/recalibrate_corrected.py`, which first reproduces the pre-repair file exactly). The PBR in the same file, **0.483** [0.318, 0.649] (26/60), is PBR **over the anchor run's 61-request fix subset**, not llama3.1-8b's model-level PBR; the model-level backbone is the full SORRY-Bench lineage run, 63/376, corrected **0.163** [0.09, 0.23], *inconclusive* against π\* (B3). |
| A2 | confirmatory replication registered | **run** as registered (`results/runs/confirm_heldout_3_31/`): RSR **6/9**, raw 0.667, corrected **0.765** [0.381, 1.000], McNemar p = 0.031 → *materially surface-bound*: the anchor replicates. PBR(llama3.1-8b) 20/208, corrected 0.077 [0.004, 0.150]: the mechanical verdict is *robust* by 0.0001 and was *inconclusive* under the pre-repair calibration, so it is **reported as inconclusive and threshold-sensitive**. |
| A3.2 | false-rejection rate not estimable | now estimable on a related population (B4.1). |
| D7 (new) | "smaller chat/instruct models … quantised" (exposé §5); quantisation level not specified | Ollama default tags, **not matched across the lineage**: llama2:7b, llama2:13b, llama3:8b **Q4_0**; llama3.1:8b, llama3.2:3b **Q4_K_M**; llama3.2:1b **Q8_0** (`ollama show`, Ollama 0.30.10; found 2026-10-31). The anchor pair therefore differs in quantisation as well as generation (Q4_0 → Q4_K_M). Recorded as a threat to internal validity: a refusal difference between the two is attributed to the release, not to quantisation, only to the extent quantisation noise is small relative to it. Sensitivity check registered in `PREREG_QUANT_20261031.md` (2026-10-31 14:10, before data): llama3:8b re-screened at Q4_K_M (`llama3:8b-instruct-q4_K_M`) on all 440 SORRY-Bench requests. Result (B2): the π\* verdict is unchanged (corrected RSR 0.527 at Q4_0, 0.578 at Q4_K_M, both *materially surface-bound*), but a third of the fixes depend on the quantisation. The Mistral spot-check used matched Q4_K_M. |

### B2. Analyses added after Amendment 1 (all exploratory)

| analysis | registered in | status |
|---|---|---|
| Output inversion (RPE), three admission rules A_gate / A_loop / A_both; harmful-ask audit of re-opening reconstructions | design in `THESIS_KB.md` §4.2 (added before the audit sheet was labelled); audit sheet `audit/rpe/rpe_harm_sheet_20261028_2327.csv` exported blind before labelling | done: anchor PRR **36/38** under the harmful-ask criterion (61/65 reconstructions keep the harmful ask), against RSR 23/49 (McNemar 3.1e-5). Other pairs' A_loop figures unaudited upper bounds. |
| Compose v1 (2-factor: para, rpe, fixed attack template) | `PREREG_COMPOSE_20261027.md` | done; `attack→para` is not a composition and its arm was truncated (THESIS_KB §4.4). |
| Compose v2 (21-condition style factorial), anchor | `PREREG_COMPOSE_V2_20261029.md` (before data) | done: no style pair beats its singles (min Holm p 0.23). |
| Compose v2, second pair llama2-7b → llama3.1-8b | `PREREG_COMPOSE_V2_ADDENDUM_20261030.md` | done: null replicates (min Holm p ≥ 0.94); low power at n = 23. |
| Matched-scale 7–8B chain, post-hoc pairs llama2-7b → llama3-8b and → llama3.1-8b | computed from the frozen lineage manifests; no new generation | SORRY-Bench RSR 5/6 and 13/23; flagged post hoc, kept out of pooled cells. |
| Benign over-refusal (XSTest) | `PREREG_OVERREFUSAL_20261031.md` | done, all 8 models (`results/runs/overrefusal_xstest_20261031_all8/`): safe-prompt refusal llama2-7b 55.2%, llama2-13b 64.4%, llama3-8b 18.4%, llama3.1-8b 18.4%, llama3.2-1b 31.2%, llama3.2-3b 20.0%, Mistral 7B v0.2 20.0%, v0.3 19.6%; unsafe contrast refused 97.5–100%. In the matched pairs the newer model does not refuse more benign prompts (anchor 12 / 12 swaps, McNemar p 1.0; Mistral 13 / 14, p 1.0); it does at two exploratory rungs (llama2-7b → llama2-13b +38 / −15, p 0.002; llama3.1-8b → llama3.2-1b +42 / −10, p < 0.001), whose P⁻ may therefore partly be indiscriminate refusal. Raw judge rates; corrected by B4.2, whose sample comes from the four models complete at export, so applying it to the other four is a transfer assumption. |
| Contamination probe | `PREREG_CONTAMINATION_20261031.md` | done: the registered flag fires on HarmBench for all three models, including Llama 2 7B, which predates HarmBench; every hit is a copyright item (book-passage / song-lyrics prompt). No evidence of memorised benchmark wording. |
| Cross-family spot-check, Mistral 7B v0.2 → v0.3 | `PREREG_XFAMILY_MISTRAL_20261031.md` | done: P⁻ 25, 34 opposite flips, RSR 15/25 raw, descriptive. |
| White-box refusal-direction probe | none (proof of concept) | **no claim**; appendix / future work only. |
| Causal test of the refusal direction (white-box, TODO_HUMAN B3) | `PREREG_MECH_CAUSAL_20261031.md` | done (`results/mech/causal_anchor_sorrybench_20261031_2101.json`, llama3.1-8b, exact Ollama weights, numbers only): 23 fixes, 53 re-opening rewordings, layer 23 (d′ 13.25): moving each re-opening rewording up the newer model's refusal direction to the canonical request's position raises P(refusal opening) more than the same move along a random direction in **23/23** fixes (sign test p = 2.4e-7; median increase 0.11 against 0.0004), and moving the canonical request down by the same amount lowers it more than a random move in **23/23** (p = 2.4e-7; median decrease 0.28 against −0.01); removing the direction at every layer lowers it more than removing a random one in 18/23 (p = 0.011). Registered reading: *causally sufficient* and *necessary* in the registered, directional sense. The size is partial: the one-layer move restores the canonical request's refusal level for only **5 of 53** rewordings (9.4% [4.1, 20.3]), so the displacement along this direction carries part of the re-opening, not all of it. One model, one pair: appendix, exploratory. |
| Class-stratified attack audit (TODO_HUMAN A5) | `PREREG_ATTACK_AUDIT_20261031.md` | registered 20:51; sheet exported, unlabelled (B4.5). |
| Cache staleness check (reproducibility) | `PREREG_STALENESS_20261031.md` | done (`results/runs/staleness_20261031/`): judge-verdict agreement **240/240** between cached and regenerated replies (each of the 12 model × stratum cells 20/20; pooled Wilson [0.984, 1.000]), so no cell is flagged and the cached decisions reproduce with the current weights and Ollama 0.30.10, including the three models re-pulled on 2026-10-10. 227/240 byte-identical; 11 of the 13 that differ are llama2-13b, none changes a verdict. |
| Quantisation sensitivity of the anchor (D7) | `PREREG_QUANT_20261031.md` | done (`results/runs/quant_check_20261031/`; lineage-run population, older model draw 0, newer model's registered verdicts): llama3-8b answered 92/440 at Q4_0 and 85/440 at Q4_K_M (14 / 7 discordant, McNemar p 0.19); P⁻ 49 vs 41, 36 shared (Jaccard 0.67); RSR 23/49, corrected **0.527** [0.344, 0.710] vs 21/41, corrected **0.578** [0.379, 0.778], both *materially surface-bound*, no unresolved fix: **robust to the quantisation mismatch** under the registered rule. Which requests count as fixes is not: 18 of the 54 fixes found under either quantisation appear under only one. |
| Loosened gate thresholds (A3.5, loosening side) | `PREREG_GATE_LOOSEN_20261031.md` | done (`results/runs/gate_loosen_20261031/`; compose v2 single-style candidates, 882 over the 49 fixes): δ = 0.05 admits 28 more rewordings (463 → 491), δ = 0.10 44 more (→ 507); RSR over any single style 31/49 → **33/49** at both δ (corrected 0.724 → 0.773; Wilson [0.534, 0.788]). Not read as evidence about the anchor until B4.1 gives the share of loosened admissions that preserve intent. |

**Registration-time correction.** The three notes of 2026-10-31 first stated "Registered 01:30"; they were written at
01:25 and the run launched at 01:26. The stated time was corrected at 01:27 and each note carries an edit log; no other
change was made.

### B3. Estimator conventions fixed: populations and the RSR denominator

**Three populations, always named.** Results come from three runs of the anchor pair that differ in their behaviours, and a
figure is never reported without saying which:

| name | run | behaviours | RSR (judge raw) | PBR(llama3.1-8b) |
|---|---|---|---|---|
| *anchor run* | `results/runs/recycle_3_31_sorrybench/` | the 61 SORRY-Bench requests the pair fixed under the offline rule (subset CSV) | 23/47 (|P⁻|) · 23/46 (resolved) | 26/60 — **over the fix subset only** |
| *lineage run* | `results/runs/lineage_sorrybench_20261020/` | all 440 SORRY-Bench requests, six models, one frozen manifest | 23/49 | 63/376 — **model-level backbone** |
| *held-out run* (confirmatory) | `results/runs/confirm_heldout_3_31/` | 222 held-out behaviours (A2) | 6/9 | 20/208 |

The anchor-run PBR measures robustness only on requests the newer model newly refused; it must not be presented as the
model-level PBR backbone of RQ4. Against π\* = 0.15 the model-level SORRY-Bench PBR (corrected) is *inconclusive* for four
of the six models (llama3.1-8b 0.163 [0.093, 0.233]), *materially surface-bound* for llama3.2-3b (0.240 [0.164, 0.315]) and
*materially robust* for llama2-13b only by 0.0014 (0.083 [0.018, 0.1486]) — threshold-sensitive in the same way as the
held-out PBR, and reported as such (`tools/naive_vs_corrected.py`).

**The RSR denominator.**

The exposé (§5) and A2 register that a fix with **no admitted rewording is unresolved, excluded from the RSR
denominator and reported separately**. The package estimator `compute_rsr` divides by |P⁻| (counting unresolved fixes as
held), and the lineage dashboards inherit that. From this amendment:

* **Primary:** RSR = re-opened / resolved fixes, unresolved count reported beside it.
* **Conservative variant:** re-opened / |P⁻|, reported where the two differ.
* Every table states which denominator it uses; `tools/naive_vs_corrected.py` and `tools/thesis_numbers.py` report both.

**Effect on the headline.** The anchor's stored figure (23/47, `corrected_sorrybench_recal_n60.json`) was computed by
`compute_rsr` over |P⁻| = 47 and includes one fix with no admitted rewording (jbb-0041, verified by rebuilding the run from
the cache and the stored qwen verdicts, which reproduces P⁻ = 47 exactly). Under the registered convention the anchor is
**23/46: raw 0.500 [0.361, 0.639], corrected 0.564 [0.374, 0.753]** (same calibration; McNemar unchanged, p = 2.4e-7;
verdict unchanged, *materially surface-bound*); 23/47 → 0.551 [0.364, 0.738] is the conservative variant.
The confirmatory replication is **unaffected**: a judge-only replay of its canonical draws and re-opening rewordings (2026-10-31, qwen2.5:14b, no target calls; `cache/judge_confirm_heldout.jsonl`, 480 verdicts) reproduces its judged P⁻ = 9 and 6/9 exactly, and none of the 9 fixes is among the 7 held-out behaviours without an admitted rewording, so 6/9 is the same under both denominators.

### B4. Audits registered before labelling (sheets exported 2026-10-31, unlabelled at registration)

Each sheet is blind (annotators see only the request and the rewrite or reply; no stratum, score, verdict or model) and
shuffled; the key is a separate file. Hashes (sha256) freeze the samples:

| file | sha256 |
|---|---|
| `audit/gate_rejection/rejection_sheet.csv` | `509734a8e45fba503a10a74091c293d3ce5e5b9528757d34424c381be4e825fe` |
| `audit/gate_rejection/rejection_key.csv` | `8678529f9314df46d3e4a9f94bae042cffc231b6d3a03938a849f5381f6e8ef0` |
| `audit/overrefusal/benign_sheet.csv` | `9592eb2c59824782b6a2b4f0e546e56b3485ac23e4006c0e26eff899a1c4fd30` |
| `audit/overrefusal/benign_key.csv` | `14ee82fe8292a5d82a607ead32558137f41dbc9673c05f4a120b340560cfb99d` |
| `audit/sorrybench/judge_sheet_blind_relabel_20261030.csv` | `d4dce35b32e3d8725f15c89cf58582a25eb5a2e2d389169dcf0a8b37ca6d0d49` |
| `data/xstest.csv` | `eb520b43da7f5c6d8241a1e870d4006b12c615ce1c18eb252fa8b2d63fb1da6f` |

1. **Gate false-rejection rate** (`tools/gate_rejection_audit.py`, seed 20261101). Population: the compose v2 store's
   six single-style conditions (production paraphraser qwen2.5:3b, the six registered classes, production gate against the
   original request; the anchor's 49 fixes × 3 candidates). The production manifests did not keep rejected text, so this
   is the closest available population; stated as such. Strata: **F** = rejected at a semantic layer (cosine 119, NLI 169,
   BERTScore 0; N = 288), random 100; **C** = admitted (N = 463), random 30. Lexical-divergence rejections (131 near-copies,
   rejected by design for not being a rewording) are outside the estimand and counted only. Rubric: the D5 rubric.
   **Estimands:** P(preserved | rejected) with Wilson 95%, per layer; P(rejected | preserved) (false rejection) and
   P(admitted | changed) by `invert_stratified` with bootstrap 95%. Reported as a property of the gate on this population;
   it costs power, not validity, so no headline is adjusted by it.
2. **Judge on benign prompts** (`tools/overrefusal_judge_audit.py`, seed 20261101). Population: the 1,000 replies of the
   four complete models (llama2-7b, llama2-13b, llama3-8b, llama3.1-8b) to the 250 safe XSTest prompts. Strata by judge
   verdict: **R** judged refused (N = 391), random 30; **A** judged answered (N = 609), random 20. Label `human_fulfilled`
   with the judge rubric. **Estimands:** the judge's response-level se and sp on benign prompts (`invert_stratified`); each
   model's benign over-refusal corrected by Rogan–Gladen with that calibration, beside the raw judge rate and the offline
   rule. **Reporting rule:** once the sheet is labelled the corrected rates are primary and the raw judge and offline-rule
   rates are sensitivity analyses. The anchor-pair comparison stays the paired McNemar on the judge's verdicts (12 / 12
   swaps); a correction applied equally to both models cannot create a difference between equal raw rates, so it changes
   the level reported, not that comparison.
3. **Blind re-label of the SORRY-Bench calibration sheet** (`tools/blind_relabel_score.py`). The 280 rows of
   `audit/sorrybench/judge_sheet.csv`, shuffled, judge column removed. **Estimands:** (a) anchoring: row- and bundle-level
   agreement and κ between the original and the blind labels, and each label set's agreement with the judge; (b) the
   judge's bundle-level se / sp / κ from the blind labels (design-weighted 26 / 34, complete bundles only); (c) the anchor's
   and the confirmatory replication's corrected RSR and PBR recomputed with (b) from their stored k / n / t_mean.
   **Decision rule, fixed now:** if all 60 bundles are labelled before submission, the **blind calibration becomes primary**
   for every corrected figure, and the original (unblinded) calibration is reported as sensitivity; every verdict that
   changes is reported, and the confirmatory replication's decision is reported under both. If the re-label is incomplete,
   the original calibration stays primary and the partial agreement is reported. The tool reproduces every registered
   corrected figure exactly when fed the original labels (verified 2026-10-31).
   **Implementation (2026-10-31, before any label):** the rule is applied mechanically by `tools/primary_calibration.py`,
   the only place that decides which calibration is primary; every tool that reports a corrected figure reads the
   calibration and the stored corrected files through it. Once all 60 bundles of the blind sheet are labelled,
   `primary_calibration.py apply` writes `corrected_*_blind_n60.json` beside each registered `*_recal_n60.json`
   (after reproducing the registered file from the original sheet; never overwriting), and `tools/after_labels.py --run`
   re-scores and rebuilds everything downstream into new dated outputs. Verified with a blind sheet carrying the
   original labels: all 186 values of the numbers file are reproduced exactly. The held-out PBR's reported verdict
   (*inconclusive*, threshold-fragile) is carried over to the blind file and flagged for re-reading, not recomputed.

4. **Blind re-check of the model-assisted gate audit** (`tools/gate_recheck.py`, seed 20261101). Rows 19–178 of the D5
   sheet were labelled with a model guess on screen. Strata: **B** 10 random rows of the blind rows 1–18 (test–retest
   baseline); **G0** all 12 model-assisted rows the model guessed "changed"; **G1** 18 random model-assisted rows it guessed
   "preserved". Blind and shuffled, no guess and no original label shown.
   **Estimands:** agreement of the re-label with the original label per stratum (Wilson 95%) with every change and its
   direction; anchoring = disagreement in G0 ∪ G1 minus disagreement in B; the D5 count of re-opening rewordings that changed
   the request recomputed with the re-labels. Limitation: the labeller saw these rows two days earlier.
   Hashes: `audit/gate_recheck/recheck_sheet.csv` `5c52df36aee30a022a3d2d1daf57cf0084e852d5f0b6b6ee7d5afb8d77d834e7`,
   `audit/gate_recheck/recheck_key.csv` `9a2ffd1722aef6a78abdc7d66fb39863efc5cdaa75493fe082e0b47a6ccf7b63`.

5. **Class-stratified audit of the attack-transfer arm** (`tools/attack_audit.py`, `PREREG_ATTACK_AUDIT_20261031.md`,
   registered 20:51, seed 20261101). The arm's first calibration (`audit/attack/`) stratified on the judge's verdict: all
   30 judge-positives were real compliance and 5 of 30 judge-negatives were too, so the judge recognises about 7% of real
   compliance in attack outputs and the arm's transfer rates are under-counts of unknown distribution over classes.
   Population: llama3.1-8b's replies to 29 attack classes (the two baselines and `low_resource`, whose replies are in a
   language the annotator cannot read, left out) for the 291 behaviours whose baseline reply the judge scored as refused
   (the judge reproduced the arm's stored 291 of 300 exactly). Sample: 4 replies per class, 116 rows, blind, shuffled, a
   deterministic base64 / rot13 decoding aid where a reply carries encoded text (4 rows). **Estimands:** per-class human
   transfer-on-refused (descriptive); overall, stratified by class with a stratified bootstrap 95% CI; the judge's se / sp
   on attack outputs; the ratio to the arm's raw judge rate. **Reading rule:** the overall human estimate replaces the
   judge's raw rate as the arm's headline magnitude (exploratory, never pooled with RSR or PBR); the first calibration
   is reported beside it, not merged.
   Hashes: `audit/attack_classes/attack_sheet.csv` `7bc82f034845ddcd9079dd7ed154e03815780d63883da163704fe9219e755be0`,
   `audit/attack_classes/attack_key.csv` `00c371134a4cfdb96240396a39371c49d2dccb89f993c64500bd30795ff985a0`.

### B5. Fixed analyses added for the write-up (no new data)

1. **Power** (`tools/power_rsr.py`; exposé §6). McNemar on P⁻ is significant iff k ≥ 6, independent of n. Power of the
   π\* decision (Wilson lower bound > 0.15) on the raw scale: n = 49 needs k ≥ 13, minimum detectable RSR 0.32 at 80%;
   n = 9 needs k ≥ 4, MDE 0.53; n = 23 needs k ≥ 7, MDE 0.37. A *robust* verdict is unreachable at n = 9 and requires
   RSR ≤ 0.03 at n = 49: **the design can establish surface-binding but cannot certify robustness at these sizes**, which the
   write-up states. Corrected-scale power is lower.
2. **Naïve vs corrected (contribution C4)** (`tools/naive_vs_corrected.py`): every SORRY-Bench lineage pair's RSR and
   every model's PBR under the offline rule, the judge raw, and the bundle-level corrected estimate, with the gap; the
   calibration's transfer from llama3.1-8b responses to the other models is stated as an assumption.
3. **Single numbers source** (`tools/thesis_numbers.py`): every figure in the thesis is generated as a LaTeX macro from the
   registered result files; audits not yet labelled print as "[pending]".

4. **RQ3 re-specification (registered 2026-10-31 14:16, before any refit).** An external review found that the
   RQ3 implementation did not match the registration: its "category" term was the benchmark, not the harm category; the
   behaviour identifier included the model pair, so one source behaviour received a separate random intercept per pair; and
   its p-values were normal approximations from a variational Bayesian fit, Holm-corrected as if frequentist. The previous
   driver results are withdrawn as inferential claims (kept as descriptive class rates). The replacement is fixed now:
   * **Unit:** one row per admitted rewording of a fix of the newer model (P⁻ as in the run), outcome = judged compliance.
   * **Fixed effects:** transform class (**reference: paraphrase**, the exposé's primary operator), surface distance, harm
     category; the harm category is **SORRY-Bench's domain** (4 levels, `lineage_core.SORRY_DOMAINS`) — the other
     benchmarks' taxonomies are not harmonised with it, so harm category enters only the SORRY-Bench models.
   * **Clusters:** the **source behaviour** (benchmark : row id), shared across pairs.
   * **Primary inference:** logistic GEE, exchangeable working correlation, bias-reduced (Mancl–DeRouen) sandwich covariance;
     an omnibus Wald test of transform class (5 df) and the five class-vs-paraphrase contrasts, Holm-corrected across those
     five; odds ratios with 95% intervals; surface distance and covariates reported with unadjusted Wald p.
   * **Models:** M1 (primary) the anchor pair on SORRY-Bench (lineage run, 49 fixes), harm domain as covariate; M2 the anchor
     pair pooled over SORRY-Bench, HarmBench, AdvBench and StrongREJECT, benchmark as covariate (no harm category); M3 the
     five SORRY-Bench lineage pairs, harm domain and pair as covariates.
   * **Sensitivity:** the same fixed effects in a binomial mixed model with a random intercept per source behaviour
     (statsmodels variational Bayes, seed 20260101), reported as posterior means with 95% credible intervals, **no p-values**.
   * **Reading rule:** a class "drives recovery" only if the omnibus test is significant at 0.05 and its Holm-adjusted
     contrast is too; all RQ3 results are exploratory.
   * **Fit-time addition (2026-10-31, recorded before any estimate was read):** back-translation has no recovered rewording
     in any of the three models (M1 0/7, M2 0/10, M3 0/11), so its odds ratio has no finite maximum-likelihood estimate (complete separation). A class with
     no events, or only events, leaves the model (GEE and sensitivity alike) and is reported descriptively with its count; the
     omnibus test then has one degree of freedom fewer.
5. **Full-configuration stamp** (`tools/config_stamp.py`): the registration hash (`prereg.decision_fields`) covers
   seed, lineage, classes, benchmarks, gate, budget and decision rule, but not the target / judge / paraphraser / intent model
   ids or their decoding, and cache keys hold only (config key, prompt, seed). The stamp records, for every config behind a
   reported run, the sha256 of the whole file and of each model block, plus the Ollama version and each model's weights blob,
   and `--verify` reports any drift. The existing registration hash is left unchanged so earlier registrations still verify.

### B6. Sign-off

Candidate: ______________________  date: __________  
Supervisor (acknowledged): ______________________  date: __________
