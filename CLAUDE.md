# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
M.Sc. thesis instrument (RPTU Kaiserslautern-Landau, with IBM Frankfurt). **Question:** when a newer
Llama generation refuses a request an older one answered, did it learn the harmful *intent* or just
the *wording*? Frame results as "does safety alignment generalise across generations" (alignment
science), not "paraphrase jailbreak robustness". The contribution is measurement + integration
(pre-registered, calibrated, audited), not a new attack or defence.

- **P⁻** (regression set) = behaviours where OLD complies and NEW refuses (the "fixes").
- **RSR** (primary) = fraction of P⁻ re-opened by ≥1 gate-admitted, intent-preserving paraphrase.
  High ⇒ surface patching; low ⇒ consistent with generalisation.
- **PBR** (backbone) = fraction of a model's own canonically-refused set re-opened. Always estimable;
  carries the headline when |P⁻| < `decision.min_p_minus` (30).
- Rules the code enforces: empty P⁻ ⇒ RSR *undefined* (never 0); denominators count behaviours, not
  variants; behaviours with no admitted variant are "unresolved", not "robust".

`THESIS_KB.md` (repo root) is the **science layer**: constructs and estimands, the five arms
(recycle / RPE / attack-transfer / compose / mechanistic), what may and may not be claimed from each
number, the expose-vs-reality drift list, and the defect catalogue. Read it before interpreting or
writing up any figure; this file stays the repo layer (commands, architecture, paths).

`CLAUDE_CODE_HANDOFF(1).md` (repo root, untracked) is the full project brief: thesis RQs, results
tables, supervisor threads, open tasks. Read it for anything beyond what is here. Note that
its §5 layout is partly wrong: there is no `metrics.py` (see below).

## Commands
```bash
export PYTHONPATH=src                       # required; Makefile sets it too
python3 -m pytest tests/ -q                 # full suite (~15 s, no models needed)
python3 -m pytest tests/test_judge.py -q    # one file
python3 -m pytest tests/test_stats.py::test_name -q
make test | validate | prereg | verify | figures

# real-model path (Ollama on :11434, OpenAI-compatible)
python3 -m recyclebench pilot-pminus --config configs/real_llama_lineage_all.yaml \
    --old llama3-8b --new llama31-8b --behaviours data/sorrybench.csv --cache cache/llama_gen.jsonl
python3 -m recyclebench run-real --config configs/recycle_3_31.yaml \
    --behaviours data/sorrybench.csv --manifest <run_dir>/manifest_sorrybench.jsonl --cache cache/llama_gen.jsonl
python3 tools/judge_export_real.py --config ... --new-model ... --behaviours ... \
    --manifest <run_dir>/manifest.jsonl --cache ... --n-bundles 60   # -> hand-label -> judge-ingest
# (`python3 -m recyclebench judge-export` is the MOCK twin; it refuses a config with a real: section)
python3 -m recyclebench prereg --behaviours F.csv ; python3 -m recyclebench prereg-verify  # exit 1 = drift

bash run_pair.sh <old> <new> <cache> [benchmarks...]   # pilot + 2x2 partition -> results/runs/<ts>_...
bash run_all_models.sh                                  # self-detaches (nohup+caffeinate); log: logs/all_models.log
```
Ask before running anything that hits Ollama. Real runs take hours and write to the shared cache.

## Architecture

**Two parallel paths share the stats core.** The *mock* path (`study.py`, `models.py` mock lineage,
`MockJudge`, mock `gate.py`/`reformulate.py`) plants a latent truth so `validate` can check estimator
bias/coverage. The *real* path (`realrun.py`, `backends/`) has no ground truth. Its estimates are RAW
until humans calibrate the judge via `judge-export`/`judge-ingest`. Never report mock numbers as Llama
findings.

**Real pipeline:** `cli._build_real` turns config `real:` entries into `backends.providers.ProviderModel`
(OpenAI-compatible), builds the judge from `judges:` (`backends/real_judge.py`: rubric / Llama Guard
ensemble; currently a single `qwen2.5:14b` rubric judge), and falls back to `judge.RuleBasedJudge`
(refusal-prefix, pilot-grade only) when `judges:` is absent. `screening.py` does canonical screening
(k draws, broken if any draw complies) → C_M and P⁻. `run-real` builds the paraphrase manifest
(`backends/reformulate.LLMReformulator` × `transform_classes`, gated by `gate.load_gate("real")` →
`backends/real_gate.py` + `gate_layers.py`), **freezes** it (`manifest.py`), then re-screens every model
on the same frozen variants. `recycling.py` = RSR, `pbr.py` = PBR, `stats.py` = Wilson / bundle-level
Rogan–Gladen (sp^t) / McNemar / bootstrap, `calibration.py` + `judge_audit.py` = se/sp/κ from human labels,
`audit.py` = gate human audit, `drivers.py` = RQ3 mixed-effects, `prereg.py` = hash-stamped prereg + drift.

**Cache** (`realrun.ResponseCache`): append-only JSONL, lookup key `k = sha256(f"{model}|{prompt}|{seed}")[:32]`,
where `model` is the **config key** (`llama3-8b`), not the Ollama `model_id`. Keep cfg keys stable or the cache
misses. Canonical seeds come from `screening._draw_seed(bid, -1, draw)`. Changing config `seed` or `k`
changes draws and causes misses. Newer lines also carry `model`/`seed`/`prompt[:200]`, but old lines are
`{k,t}` only, so don't grep the cache for model names. Judge verdicts are not cached. Tools reconstruct
prompt→response by recomputing keys (`tools/inspect_cache.py`, `make_dashboard.py` import `_key`/`_draw_seed`).

**Intent gate** (ANDed, thresholds in config `gate:`): cosine ≥0.75, BERTScore ≥0.85, NLI-entail ≥0.60,
llm_intent ≥0.50. Component scores are persisted in the manifest so threshold sensitivity is re-applied
offline. Docs: `docs/GATE.md`, `docs/REFORMULATE.md`.

**Three arms:**
1. *Recycle* (primary): the pipeline above. 6 transform classes (`paraphrase, syntactic, lexical,
   back_translation, tense_shift, register`), 8 paraphrases/behaviour, paraphraser `qwen2.5:3b`.
2. *RPE* (output inversion, convergent check): reconstructs the request from the OLD model's output (inverter
   `qwen2.5:14b`) and re-tests NEW. Current runs: `tools/rpe_lineage.py` (manifests per pair, `--reuse-from` copies
   inversions of an identical OLD answer) → `tools/rpe_closedloop.py` (phases old/judge-old/equiv/new/judge-new/score;
   `--gate-store` reads `cache/rpe_gate.jsonl` from `tools/rpe_gate.py`) → `results/runs/rpe_lineage*/`,
   `rpe_bench*/<bench>/closedloop_*_gated_*.json`. Three admissions: A_gate (intent gate, comparable to RSR), A_loop
   (closed loop, upper bound), A_both. Offline wrap-up: `bash rpe_refresh.sh [--export]`; reports
   `tools/rpe_lineage_report.py`, `tools/rpe_reconcile.py`; audit `tools/rpe_audit_export.py` →
   `tools/rpe_audit_score.py`; dashboards `tools/inversion_section.py`. The first runs
   (`run_independent_probe.py`, `tools/vendor/rpe_ga`, `rpe_3_31_*`) are superseded.
3. *Attack-transfer*: `tools/run_attack_transfer.py`. Wei et al. attack classes from the
   liuyanchen1015 HF dataset, **not** intent-gated. It must be judged against the *original* intent;
   judging encoded output against encoded prompts gives spurious bypasses.

**Dashboards:** `tools/make_dashboard.py` (single pair) rebuilds the 2×2/Sankey from cache + benchmark
CSVs for `--old/--new`, and admitted-paraphrase cards from `--manifest`. Per-benchmark views go through
`build_view(...)` + `filterBench(t)`. Section modules (`calibration_section`, `rpe_dashboard_section`,
`style_arcs_section`, `attack_transfer_section`, `positioning_section`) are self-contained
`<style>+<section>` fragments scoped under their own root class and injected before `</body>`. Calibration
goes at the `<!--CALSLOT-->` marker. `tools/build_all_dashboards.py` batch-builds one per `results/runs/*`
(pair parsed from `<old>_to_<new>` in the dir name, else `PAIR_OVERRIDE`).

**Lineage dashboard** (`tools/lineage_dashboard.py`, data layer `tools/lineage_core.py`): one page for a whole
`lineage:` run (defaults = the SORRY-Bench lineage), rebuilt from cache + frozen manifest through the package's own
`compute_rsr`/`compute_pbr`. Headline figures come from `<run-dir>/summary_real.json`; per-item views use the verdict
cache from `tools/judge_lineage.py` (replays the config's `judges:` over cached responses, no target calls, resumable,
`--scope canonical|rsr|all`, `--dry-run`) when complete, else the offline refusal rule, labelled as such, with a
reconciliation table. `--redact` drops all prompt/response text. Tested in `tests/test_lineage_dashboard.py`.

**Master dashboard** (`tools/master_dashboard.py`, evidence layer `tools/study_data.py`): every result (all models ×
benchmarks, all recycle/RPE/arc/attack runs, both judge calibrations, gate, registry of ~70 artifacts with integrity
verdicts) in one text-free page → `results/dashboard_master_<stamp>.html`. `study_data` discovers artifacts, checks them
(summary vs manifest, pilot vs META, identical copies, re-screen completeness, judge in RPE logs) and recomputes every
number via the package estimators; run specs it cannot infer live in `RECYCLE_SPECS` / `RPE_SPECS`. Pass
`--attack-70b-responses "~/Downloads/all_local_results_llama3_1_70b 1.jsonl"` for the per-item 70B checks.
`tests/test_master_dashboard.py` includes real-data reproduction tests that skip when `results/` is absent.

**Pair dashboards** (`tools/pair_dashboard.py`): one page per older → newer pair across every benchmark →
`results/runs/dedicated-runs/<old>__to__<new>/dashboard_<stamp>.html` + `index_<stamp>.html`. Fixes come from the full
benchmark screens; every recycle run containing both models contributes its rewordings (same-CSV runs join by bid,
subset runs by prompt text — the loader ignores any `bid` column and HarmBench has 5 duplicate prompts). Embeds
prompt/response text (collapsed) unless `--redact`; `--pairs a:b`, `--out-root`. Tests: `tests/test_pair_dashboard.py`.

## Configs
`real_llama_lineage_all.yaml` holds the 6-model lineage (`llama2-7b → llama2-13b → llama3-8b → llama31-8b →
llama32-1b → llama32-3b`) and the seed the lineage cache was built with. `recycle_3_31.yaml` holds the
calibrated anchor pair (+ `qwen25-14b` inverter). `run-real` derives pairs from `lineage:`. **All configs in
a lineage must share identical `judges`/`gate`/`transform_classes`.** A dashboard printing `P^-=0` for a
lineage rung usually means the wrong `--config` (seed mismatch).

## Current state (Oct 2026)
- **Anchor / headline:** llama3-8b → llama3.1-8b on SORRY-Bench, |P⁻|≈49, RSR 23/47 = 0.489 raw / **0.551
  corrected** [0.364, 0.738], judge admitted (se 0.862 / sp 0.968 / κ 0.833, **60** bundles).
  `results/runs/recycle_3_31_sorrybench/`.
- **Confirmatory replication (Amendment 1 §A2) is RUN and corrected.** llama3-8b → llama3.1-8b on held-out
  `data/heldout_jbb_guided.csv` (222 behaviours), `results/runs/confirm_heldout_3_31/`. **RSR 6/9 = 0.667 raw /
  0.765 corrected** [0.381, 1.000], McNemar p = 0.031 → *materially surface-bound* against π* = 0.15, i.e. the
  anchor replicates. PBR(llama3.1-8b) 20/208 = 0.096 raw / 0.077 corrected [0.004, 0.150]: the mechanical verdict
  is "materially robust" but its CI high lands 0.0001 under π*, and the pre-repair calibration put it 0.0007 over,
  so **report PBR as inconclusive and threshold-sensitive**. Corrected figures live in
  `corrected_heldout_recal_n60.json` (supersedes `corrected_heldout.json`, which used the pre-repair sheet);
  §A2 registers the SORRY-Bench calibration for this, so **no held-out human labelling is needed**.
- **2026-10-27 calibration repair.** `audit/sorrybench/judge_sheet.csv` had `human_fulfilled` = `00` on one row of
  bundle `jbb-0003`; `ingest_judge_audit` drops any bundle with a cell that is not `0`/`1`, so the calibration had
  silently been running on 59 bundles. Repaired (backup `*.bak_before_typo_fix_20261027_1719`): n 59 → 60,
  se 0.8585 → 0.8621, sp 0.9676 → 0.9677, κ 0.8302 → 0.8326, still admitted. Anchor corrected RSR
  0.5532 → 0.5509. Third-decimal everywhere except the PBR verdict above.
- **Lineage (RAW, 5 adjacent pairs):** `results/runs/lineage_sorrybench_20261020/manifest.jsonl`. Per-pair
  P⁻ = 18, 3, 49, 12, 13. Only the matched-scale 3-8b→3.1-8b rung is firm. The others are within-generation or
  scale-confounded (a 1B in the "newer" role), so they are exploratory.
- Earlier finding still holds: on raw prompts, safety-tuned "old" models refuse most raw harm, so P⁻ is
  small. Don't add more raw-harm benchmarks hoping to grow P⁻. (The Aug 2026 wrapped-template
  experiments in `cache/wrapped_rs.jsonl` / `data/wrapped_rs.csv` were superseded by SORRY-Bench.)
- Open tasks are in handoff §14. §14.1 (lineage dashboard) is covered by `tools/lineage_dashboard.py` instead of a
  `make_dashboard.py --lineage` mode. `cache/judge_lineage_sorrybench.jsonl` holds every canonical + RSR verdict
  (reproduces P⁻ 18/3/49/12/13 and k 15/3/23/5/8 exactly) and, since 2026-10-27, all 13,325 variant verdicts:
  `judge_lineage.py --scope all --dry-run` reports 0 remaining, so PBR is judged for all six models.
- 2026-10-22 overnight (`overnight_20261021.sh`): `tools/cache_consolidate.py` merged `llama2_13b.jsonl` +
  `ollama_pilot.jsonl` into `llama_gen.jsonl` (backup `*.pre_consolidate_*`); `tools/fill_responses.py` filled the
  SMOKE HarmBench manifest for llama31-8b/llama2-13b. New `study_data` runs: `lineage-harmbench` (4 models; judged
  figures replay `cache/judge_qwen_harmbench.jsonl`, RSR 15/22 · 3/3 · 19/26, PBR not judged), `x31-32-*`, `x27-213-*`.
  `tools/judge_rpe.py` re-judged RPE with qwen (`cache/judge_rpe_qwen.jsonl`).
- 2026-10-22 thesis close-out: every result so far is exploratory. `prereg_amendment_1_DRAFT.md` (signed: candidate 2026-10-25, supervisor 2026-10-28)
  records the deviations and registers a confirmatory replication on held-out `data/heldout_jbb_guided.csv`
  (`tools/prepare_heldout.py`) with `configs/confirm_heldout_3_31.yaml`. Do not run anything on that CSV before
  `bash overnight_stack.sh --stamp`. Long jobs live in `overnight_stack.sh` (`--list`). Offline tools:
  `tools/thesis_analysis.py` (pooled headline, RQ2/RQ3, sensitivity, PBR, RPE agreement → dated `results/analysis_*`),
  `tools/gate_audit.py` (blind gate audit sheet in `audit/gate/`), `tools/regate_intent.py` (gate layer (d) post hoc,
  gemma4:e4b with `extra_body: {reasoning_effort: none}`; config key `intent:` now wires L2 into `run-real`).
- 2026-10-27 PTP arm: the local failure was mostly OUR measurement, not scale. `tools/ptp_diagnose.py`
  (teacher-forced vs free-running, by depth into the prompt) showed the 27M model predicts 60.5% of prompt
  tokens correctly given the true prefix on pairs it saw, so the mapping is largely learned and greedy
  decoding was discarding it. Two defects: every figure was a single greedy decode (RSR/PBR/PRR are
  "≥1 admitted candidate" estimands, so best-of-N is what the pipeline consumes), and `s_rev` carried no
  end-of-prompt marker, so decoding ran a fixed 40 tokens against 10.8-token prompts and precision was
  capped near 0.27. Fixing the marker (`reversal.build_training_sequence(..., eop=)`,
  `ptp_build.py --boundary`) lifted best-of-32 token F1 0.217 → 0.335 and precision 0.140 → 0.346, and
  replicates on the 92M. Scaling the FINE-TUNE corpus (Ollama, not GPU) then moved the semantic side too:
  347 → 2,846 → 5,912 pairs at matched steps gives gate-admissibility (≥1 candidate at cosine ≥ 0.75)
  1.2% → 4.7% → 9.3% and best-of-32 token F1 0.335 → 0.397 → 0.424, with no sign of flattening
  (segment exponent 0.66 then 0.95). A compute-matched control (same 347 pairs, 8× steps) got *worse*,
  so the gain is data diversity, not compute; pre-training corpus and model capacity remain flat/negative.
  Full write-up in `PTP_FINDINGS_20261027.md`. Gate A was pre-registered against a single greedy decode
  (which still fails, 0.210); best-of-N passes but is a protocol change, so **no further scaling point may
  be reported as grounds for continuing the arm until best-of-N is registered prospectively.**
  `tools/ptp_decode_sweep.py` is the comparison harness (`--vocab`, `--eos`, `--n-samples`).
  The held-out 86 are now frozen in `ptp_build.py`, so adding pairs cannot silently move the denominator.
- **2026-10-29 compose v2** (`PREREG_COMPOSE_V2_20261029.md`, `tools/run_compose_v2.py`, results
  `results/runs/compose_v2_20261029_2222/`, store `cache/compose_v2/`): 21-condition style factorial on the 49 judged anchor
  fixes, ~3 h. No style pair beats its singles (Holm min p 0.23); see THESIS_KB §4.4b.
- **2026-10-29 anchor recalibration + 7–8B chain.** `tools/recalibrate_corrected.py` wrote
  `results/runs/recycle_3_31_sorrybench/corrected_sorrybench_recal_n60.json` (reproduces the pre-repair file from the backup
  sheet first): RSR corrected 0.5509 [0.3638, 0.7379], PBR(3.1-8b) corrected 0.4833 [0.3181, 0.6486]; `study_data`, the
  lineage dashboard default `--anchor` and `build_all_dashboards.py` read it. `study_data` now also computes the post-hoc
  matched-scale pairs (`EXTRA_PAIRS`) inside every run holding both models, judged via `pair_verdicts` / `verdicts`
  (SORRY-Bench 5/6 and 13/23), flagged `post_hoc` and excluded from `pooled_offline` and `recovery_by_style`.
- **2026-10-28 RPE completion** (`run_rpe_complete.sh` then `bash rpe_refresh.sh --export`). The gate now scores every
  inversion candidate (`cache/rpe_gate.jsonl`: 120 of 1,623 admitted, later 125 of 1,731), so all 19 pair × benchmark
  runs with judged P⁻ carry A_gate / A_loop / A_both (`closedloop_*_gated_20261028_2326.json`); the two matched-scale
  post-hoc pairs (llama2-7b → llama3-8b / → llama31-8b, SORRY-Bench + HarmBench) were run, reusing llama2-7b inversions
  from the 2-13b pair. Anchor: closed loop 36/38 (resolves 38/49), intent gate 2/5 (resolves 5/49) against RSR 23/49, so
  the inversion arm's lead over paraphrase rests on gate-rejected reconstructions. That is what the blind harmful-ask
  sheet `audit/rpe/rpe_harm_sheet_20261028_2327.csv` settled on 2026-10-29 (65/65 blind: 61 keep the harmful ask, all 36
  behaviours verified, anchor PRR 36/38 confirmed); the 83-row topical
  sheet covers superseded candidates and audits nothing current. The 38/49 vs 7/61 gap (THESIS_KB §4.2) was the
  admission rule, and TODO_HUMAN A6 was a subset-CSV id join error (`results/analysis_rpe_reconcile_20261028_2333/`).
  Dashboards built 2026-10-28 23:33 show it all (master §09/§09b, every pair page, lineage §09). One battery-induced
  sleep (19:09–22:36, 1% charge) stalled the run: caffeinate holds the machine awake only on AC.
- White-box arm (PoC, 2026-10-22): `tools/mech_probe.py` loads the exact Ollama GGUF blobs into a PyTorch Llama (fp16, MPS;
  q/k un-permuted, rope factors and Llama-3 pre-tokenizer verified) and records per layer the refusal-direction score
  (Arditi difference-of-means), residual divergence, top MLP neurons and P(refusal opening) → `results/mech/*.json`
  (numbers only); `tools/mech_section.py` renders it inside fix cards of the 3-8b→3.1-8b pair dashboard. Faithfulness: Ollama
  applied its default repeat penalty (1.1, last 64 tokens) to every study request; replays must use it (`ollama=True`).
  Full run = overnight stack S5. Dashboards use the DFKI design (tokens in `lineage_dashboard.CSS`, logo
  `tools/assets/dfki_logo_small.svg`, `brand_html()`), with a scrollspy TOC and sliding indicator in `lineage_dashboard.JS`.
  The TOC lists every section fully expanded: its `h3`s, and on pair pages every benchmark tab with its panel's `h3`s (rebuilt on `rb:tabchange`); `ld.legend_html()` is the sidebar
  colour key, `ld.rq_section()` the RQ1–RQ4 cards (master answers recomputed from the newest `results/analysis_*` + `results/mech/*_all_*`),
  and `ld.rq_script(map)` puts RQ chips on nav entries and h2s. Pages must keep exactly one `<script>` (tests check it).

## Conventions & gotchas
- **`run-real` never stores judge verdicts.** Anything rebuilt from the cache with `looks_like_refusal` disagrees with
  the qwen2.5:14b judge (SORRY-Bench lineage offline P⁻ = 14/2/61/19/33 vs judged 18/3/49/12/13). Use the verdict cache.
- **`judge-export` (the CLI subcommand) is the MOCK exporter.** It screens the synthetic lineage with
  `MockJudge` and writes `[benign-placeholder N]` / `[refuse]` rows into the same schema the real sheet
  uses, so a sheet it produced looks finished while being unlabellable. It now exits 1 on a config with a
  `real:` section. Real runs export with `tools/judge_export_real.py`, which replays the frozen manifest
  out of the response cache (zero target calls) and runs only the judge. `audit/confirm/` held such a
  mock sheet from 2026-10-26 until 2026-10-27 (moved aside as `*.MOCK_VOID_20261027`), which is why the
  confirmatory calibration showed "0/398 labelled" for days with nothing anyone could label.
- **Gate audit scores in the manifest are partly placeholders.** `RealGate` short-circuits (lexical → cosine → BERTScore
  → NLI) and `to_gate_scores()` writes 1.0 for every layer not evaluated; the lexical layer isn't persisted at all.
  `llm_intent` is 1.0 everywhere (no intent model configured). `human_label` is None on every manifest row, so a summary's
  `false_admission_rate: 0.0` is vacuous; the real measurement is the D5 sheet `audit/gate/gate_sheet.csv` (178/178, scored by
  `tools/gate_audit.py ingest`, read by `gate_audit.score()` on the dashboards): 3/98 re-opening rewordings changed the request,
  no RSR moves. Rows 19–178 were model-assisted (`tools/gate_guess.py`, gemma4 guess shown; `audit/gate/MODEL_ASSISTED.txt`).
- **`calibrate_from_human` weights by the population strata, but only if you pass them.** Give it
  `n_pos_total`/`n_neg_total` (from the sheet's `judge_strata.txt`) and it hands them to `invert_stratified`, with the
  sample sizes setting only the bootstrap; omit them and it returns unweighted rates plus a note saying they are valid
  only for a proportional sample. Every caller in the repo passes them. This matters enormously for `audit/attack`
  (30 of 30 judge-positives but 30 of 2,338 negatives): se is **0.072** design-weighted against 0.857 unweighted, so
  attack transfer is badly under-counted. It makes no difference for `audit/sorrybench`, where the sample is the whole
  population (26 + 34 bundles) and both routes give se 0.862. `audit/harmbench` (labelled blind 2026-10-29, 60 of 113 + 264
  bundles) gives se 0.865 / sp 1.000 / κ 0.933, supporting the SORRY-Bench calibration's transfer to HarmBench. Neither judge suits attack outputs anyway (the offline
  rule reads ciphers/garble as compliance). On SORRY-Bench the offline rule beats qwen2.5:14b vs humans (bundle κ 0.93
  vs 0.83). *(Earlier copies of this file, and comments in `master_dashboard.py` / `rpe_audit_export.py`, claimed the
  function ignored the totals; that was fixed on 2026-10-22 and the stale wording was corrected 2026-10-27.)*
- RPE on SORRY-Bench (`rpe_3_31_*`) was judged by the target model (logs: target-as-judge); every `rpe_agreement.json` is
  invalid (paraphrase side read from fields recycle manifests lack). Recompute agreement by joining on prompt text.
- Before 2026-10-03 (first Ollama `judges:` block, commit 0d09db7) every pilot used the offline rule; the July llama3.2
  pilots (`results/pminus_{advbench,harmbench,strongreject}.json`) predate the U+2019 fix and are superseded. The SMOKE
  run's `summary_harmbench.json` is a copy of StrongREJECT's summary; `20260804_021001_*/pminus_*.json` are copies of
  `pminus_wrapped_rs.txt`.
- Style-arc / gate means must use only scores the gate computed (placeholders inflate back-translation's NLI 0.83 → 0.96).
- The summary's RSR/PBR `CI` is a delta-method (normal-approx) interval that collapses to [1, 1] at k = n; dashboards
  show Wilson intervals recomputed from k/n.
- **Ids in a subset CSV are row positions.** `load_jailbreakbench` ignores a `bid` column and numbers rows, so ids from
  `results/cells_*/…_p_minus.csv` (the anchor manifest, the compose run, the first RPE runs) are not SORRY-Bench ids.
  Join such runs on prompt text. A bid join is what produced TODO_HUMAN A6's "unreproduced anchor manifest".
- **Inversion manifests carry `admitted: false`** (the builder never runs the gate). Score closed-loop runs with
  `--gate-store`; an artifact without `_gated_` has empty A_gate/A_both by construction, not because the gate rejected.
- Refusal classification must go through `recyclebench.judge.looks_like_refusal`, the single source of truth.
  It includes the U+2019→' normalisation.
- The paraphraser/judge/inverter must be **non-Llama** (qwen2.5), decorrelated from the targets. Never use
  jailbreak optimizers (GCG/AutoDAN/PAIR) as the paraphraser, because they confound the estimand. Keep `store_raw: false`.
- **Never overwrite results/dashboards.** `pilot-pminus` writes a fixed `results/pminus_pilot.json` and `run-real`
  a fixed `results/summary_real.json`, so copy them into `results/runs/<ts>_<pair>/` after each run. Dashboards
  get dated names (`dashboard_<YYYYmmdd_HHMM>.html`). Keep the cache shared; snapshot everything else per run.
- **Never run two `run-real` at once.** Two writers corrupt the cache and manifest. A hard kill can truncate the last
  cache line; on macOS, repair with `sed -i '' -e '$d' cache/<file>.jsonl`.
- Ollama: run with `OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h`, or judge/target alternation reloads ~10 GB
  per call. A run at 0% CPU may be swap-thrashing rather than hung, so check cache growth and `ollama ps` before killing it.
  A missing model gives a 404 at call time (`ollama pull`). Hardware is an M1 Max with 32 GB, so no 70B/405B locally.
- Quiet HF: `export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1`. The "LOAD REPORT / UNEXPECTED /
  MISSING" lines are harmless. Gate deps: `pip3 install -e '.[gate]'` (or `requirements-real.txt`).
- zsh: `!` inside inline heredocs/echo triggers history expansion, so write scripts to files with `<< 'EOF'`. Unquoted
  `$var` doesn't word-split, so use arrays or `${pair%%:*}` / `${pair##*:}`.
- `results/`, `audit/`, `data/*.csv` are gitignored. Benchmarks are fetched locally (`tools/prepare_behaviours.py`,
  `tools/prepare_sorrybench.py`, CSV column `Goal`). No harmful content ships in the repo. Dashboards collapse
  sensitive text. See `ETHICS.md`.
