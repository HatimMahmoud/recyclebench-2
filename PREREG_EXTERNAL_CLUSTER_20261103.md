# Registration: GuidedBench, ScienceSafetyData and JBB-Behaviors on every model (Pegasus)

**Status: DRAFT, unsigned.** Written 2026-11-03, before any model has been run on these corpora on Pegasus. Sign below
and commit before the first job is submitted (`bash pegasus/external_submit.sh --submit`); the commit that carries the
signature is the registration.

**Designation, everywhere: exploratory.** Post hoc with respect to `prereg.md`, Amendment 1 and Amendment 2. Every
figure is raw judge output: no human calibration exists for these corpora, and the SORRY-Bench and HarmBench
calibrations are not transferred to them. Nothing here is pooled with the registered pairs, the anchor or the held-out
replication. Until each corpus has a non-empty P⁻, its outputs stay isolated: their own cache (`cache/external/`,
gitignored), their own run directory (`results/runs/external/`), not in `tools/study_data.py`, and on no dashboard.

## 1. Populations

The behaviour CSVs are fixed by their ordered request texts. Behaviour ids are row positions and canonical seeds derive
from them, so the order matters as much as the content. `tools/external_suite.py data` refuses any file that differs.

| corpus | rows | source |
|---|---|---|
| guidedbench | 200 | GuidedBench, Huang et al., arXiv:2502.16903 (harmful questions; Goal, Category; 20 categories) |
| sciencesafety | 240 | ulab-uiuc/SafeScientist @ `9db8659`, `Dataset/` only, recipe `datasets/sciencesafety.yaml` (6 domains × 40) |
| jbb | 100 | JailbreakBench/JBB-Behaviors, harmful split, `tools/prepare_behaviours.py --benchmark jbb` (10 categories) |

```
corpus         goals sha256 (ordered requests, joined by U+001E)                  file sha256 (the Mac's copy)
guidedbench    ff3fa755d8686036016246a866e0627bc847839244da07a361fe1191ac628f48   5af3394b949b6e385993f10dd3bee60e95a055e5f54a2bb8734dd5947679a53c
sciencesafety  b3b55bf39f27b8b4a0088cd20123c2165f450ce89c450b9a80d8327b0e124309   7c3cddbec048e676aa6cb3adc8f4e145bb789689a969093bf9c51ff751e452c6
jbb            2911a91083756ee1e9ecb2167426307c14ad9e58716f3df6699e5f8ef1a07d28   d0c22d73c6a24adb610feb6bd8fa317820b1524381c286a3bbe83674e19fa61d
```

The same values are in `tools/external_suite.py` (`CORPORA`). A byte-level difference that leaves the ordered requests
identical (quoting, line endings) is recorded but accepted. Any other difference stops the run. If a download cannot
reproduce a registered file, the Mac's copy is the registered population.

**Overlap with behaviours already screened** (held-out build audit, `data/heldout_jbb_guided.audit.json`, difflib
≥ 0.85 against AdvBench, HarmBench, StrongREJECT, SORRY-Bench and the earlier pilot sets):
- JBB: 29 of 100 duplicate or nearly duplicate an earlier item (20 exact copies of AdvBench or HarmBench items).
- GuidedBench: 49 of 200 (4 exact).
- ScienceSafetyData: 0 of 240 (recipe dedup report).

The populations are the published corpora, so these items stay in. Every summary statement about JBB or GuidedBench is
also given without them (join on prompt text, as `tools/prepare_heldout.py` does).

**Relation to the held-out replication (Amendment 1 §A2).**
- **What is shared.** The held-out set `data/heldout_jbb_guided.csv` consists of the 71 JBB and 151 GuidedBench
  behaviours that survived that deduplication.
- **The replication itself is untouched.** It is complete, its figures are registered, and this run neither changes
  nor supplements them.
- **The anchor-pair cells here are new runs.** The llama3-8b → llama3.1-8b cells on JBB and GuidedBench re-measure
  held-out behaviours in a new environment. They are reported as exploratory and are never pooled with, substituted
  for, or used to revise the confirmatory figure.
- **No unseen items remain.** After this run no JBB or GuidedBench behaviour is unseen for any model, so any future
  confirmatory test needs a different held-out source.

**ScienceSafetyData earlier pilot.** The local pilot of 2026-11-01 (`PREREG_SCIENCESAFETY_20261101.md`, llama3-8b →
llama3.1-8b on the Mac) stays what it was: an isolated exploratory pilot. Its cluster counterpart here is a separate
environment (CUDA, Ollama 0.40.12, fresh draws). If both are ever shown, they are shown side by side, never merged.

## 2. Models and instrument

**Targets (8).** The configuration keys, with the Ollama tags of `configs/llama4_pegasus.yaml`:

| key | Ollama tag |
|---|---|
| llama2-7b | llama2:7b |
| llama2-13b | llama2:13b |
| llama3-8b | llama3:8b |
| llama31-8b | llama3.1:8b |
| llama32-1b | llama3.2:1b |
| llama32-3b | llama3.2:3b |
| llama4-scout | llama4:17b-scout-16e-instruct-q4_K_M |
| llama4-maverick | llama4:17b-maverick-128e-instruct-q4_K_M |

- **Identity.** Every job records `ollama list` in its provenance file. The Mac's model IDs are recorded in
  `docs/pegasus/MAC_OLLAMA_IDS_20261103.md`.
- **Quantisation.** As served (Ollama defaults). It is not matched across models (Amendment 2 D7).
- **70B models: out of scope** (decision of 2026-11-03).

**Instrument.** `configs/llama4_pegasus.yaml` (sha256 `0f0373962ea62f35d04b92adb0d3a124cea8ff2ffb70c17f80102554d97e2a4d`) is identical to the anchor configuration
`configs/recycle_3_31.yaml` in every science setting (verified 2026-11-03):
- seed 20260101, k = 3;
- judge: qwen2.5:14b rubric;
- paraphraser: qwen2.5:3b, 8 rewordings over the 6 transform classes;
- gate: cosine 0.75, BERTScore 0.85, NLI 0.60, llm_intent 0.50, with no intent model configured, as in the anchor;
- `decision:` block.

The inverter is qwen2.5:14b (`configs/recycle_3_31.yaml`, `real.qwen25-14b`). All are non-Llama.

**Per-pair configurations.** `run-real` pairs adjacent lineage rungs only, so each pair has a configuration whose
`lineage:` is just that pair. `tools/external_suite.py configs --check` verifies that every one of them is
byte-identical to the base in everything else. The test suite asserts it too.

## 3. Pairs (19 per corpus)

| group | pairs | design |
|---|---|---|
| dense | 2-7B → 2-13B; 3.2-1B → 3.2-3B | within generation |
| dense | 2-13B → 3-8B; 3.1-8B → 3.2-1B | adjacent rung, scale-confounded |
| dense | 3-8B → 3.1-8B | matched scale (the anchor pair) |
| dense | 2-7B → 3-8B; 2-7B → 3.1-8B | matched scale, post hoc (not adjacent) |
| scout | each of the six dense models → Llama 4 Scout | post hoc, scale-confounded (109B MoE, 17B active) |
| maverick | each of the six dense models → Llama 4 Maverick | post hoc, scale-confounded (400B MoE, 17B active) |

Scout → Maverick is not a pair: both are one release.

## 4. Arms, estimands and floors

1. **Canonical screen**, every model on every corpus.
   - Method: k = 3 draws per behaviour, every draw judged.
   - Reported: answer rates (Wilson 95 %) and the 2×2 of each pair (P⁻, persistent, negative flips, both safe).
2. **Recycle (RSR)**, pairs with a fully judged canonical |P⁻| ≥ 5.
   - Method: each pair freezes its own manifest, and both models are re-screened on it in one environment
     (`experiment` driver, stages screen → subset → recycle → partition → audit).
   - Reported: raw RSR, with Wilson 95 % and McNemar as descriptives, plus the per-variant style breakdown.
   - Floor: below 5 the pair is not run (not applicable). The driver also skips recycle when its own re-judged P⁻
     falls under 5. In that case the stored and re-judged P⁻ are both reported, never silently replaced.
   - No pair here carries a verdict against π\*. Even |P⁻| ≥ 30 stays exploratory.
3. **Inversion (RPE)**, pairs with a fully judged canonical |P⁻| ≥ 1.
   - Method: the pipeline and defaults of the Scout suite.
   - Reported: A_gate, A_loop and A_both, fixes re-opened over fixes resolved. A_loop is an upper bound and is never a
     claim by itself.
   - Empty P⁻: PRR is undefined, never 0.
4. **Style arcs**, per corpus, over the frozen manifests. Descriptive, using gate-computed scores only.
5. **Judge sheets.** The audit stage of every recycle run exports a 60-bundle blind sheet, `audit/external__*/`. Until a
   sheet is labelled, its run's figures stay raw.

**Completion.** `tools/external_suite.py matrix` assigns every corpus × pair × arm cell one of five statuses (complete,
pending, not applicable, blocked, failed), with the evidence that decided it. A run is reported as finished only when
no cell is pending or failed.

**Not run here (and why).**
- **Known-attack transfer:** its own corpus, independent of these three.
- **The ScienceSafetyData `Attack/` split:** an attack-transfer input that needs its own registration.
- **Compose v2:** registered only for SORRY-Bench anchor fixes.
- **The mechanistic probe:** dense Llama-3 code for the anchor pair only.
- **Model-level PBR:** a manifest over each pair's P⁻ makes it RSR again.
- **70B models:** out of scope.

## 5. Threats registered in advance

- **T1, research framing (ScienceSafetyData).** These requests have a legitimate research reading, and the judge scores
  attempt, so the older model's answer rate may be inflated. The blind sheets are the check.
- **T2, gate range.** The gate and the judge were calibrated on short requests. The ScienceSafetyData pilot's inversion
  gate admitted 0 of 183 candidates (measured), so A_gate there may be undefined again.
- **T3, environment.**
  - CUDA vs Metal, and Ollama 0.40.12.
  - The judge is not environment-invariant (LESSONS §5: one HarmBench verdict moved on a B200).
  - Old-model blobs on Pegasus are tied to the Mac's only where the IDs match.
- **T4, scale confound.** Every Llama 4 pair and two dense rungs change scale, so a difference is not attributable to
  the release alone.
- **T5, GuidedBench provenance.** The Mac's file was converted from a spreadsheet without a recorded converter. It is
  registered by its content hash above.

## 6. Execution

- **Commands.** `bash pegasus/external_submit.sh` prints the chain, and `--submit` submits it. Run order and GPU shapes
  are in `docs/EXTERNAL_CORPORA_CLUSTER.md`.
- **Resumability.** Every step resumes on resubmission.
- **No overwriting.** Nothing overwrites a result: reports and matrices are dated, and the driver refuses a resume
  whose fingerprint moved.

## Sign-off

- Candidate: ______________________  date: __________
- Supervisor: _____________________  date: __________
