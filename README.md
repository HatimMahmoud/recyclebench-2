# recyclebench

A pre-registered instrument for the M.Sc. thesis *Security of Open-Weight LLMs: Measuring
Robustness to Paraphrase Jailbreaks* (RPTU Kaiserslautern-Landau).

**The question.** When a newer model generation blocks a jailbreak that an older one
accepted, has it learned the harmful *intent*, or merely the specific prompt it was shown?

**The test.** Take the jailbreaks the newer model appears to have fixed, reword them without
changing what is being asked, and see whether they work again.

---

## The two metrics

**RSR (primary).** For a lineage pair `M_n -> M_{n+1}`, the regression set is

```
P⁻ = { p : S_{M_n}(p) = 1  ∧  S_{M_{n+1}}(p) = 0 }
```

— prompts that break the older model but are blocked by the newer one. Then

```
             |{ p ∈ P⁻ : ∃ p' ∈ R(p),  J(p', M_{n+1}(p')) = 1 }|
RSR       =  ──────────────────────────────────────────────────
   M_{n+1}                        |P⁻|
```

where `R(p)` is the set of *gate-admitted, intent-preserving* paraphrases of `p`.

- `RSR ≫ 0` → behavioural evidence the fix was **surface-level patching**.
- `RSR ≈ 0` → evidence **consistent with** intent-level generalisation.

Neither direction is mechanistic proof; both are reported as behavioural evidence.

**PBR (backbone).** Over the always-large canonically-refused set `C_M`:

```
PBR_M = (1/|C_M|) Σ_{b ∈ C_M}  max_{p' ∈ R(p_b)}  J(p', M(p'))
```

PBR validates the paraphrase instrument and is the reportable floor when `P⁻` is
under-powered.

### Three rules the code enforces

1. **Empty `P⁻` → RSR is *undefined*, never zero.**
2. **Denominators count behaviours, not variants.** A behaviour recovered by six of its
   eight variants contributes exactly 1 to the numerator and 1 to the denominator.
3. **No admitted variant ≠ robust.** Such behaviours are reported separately as
   *unresolved*, so gate strictness cannot masquerade as model robustness.

---

## Behaviour-bundle judge-error correction

PBR and RSR are best-of-many unions, so a per-response specificity is **mis-specified**:
false positives compound across the `t` admitted variants, giving an effective behaviour-level
specificity of `sp^t`. Calibration and Rogan–Gladen correction are therefore performed at the
**behaviour-bundle** level.

`recyclebench validate` demonstrates this by Monte Carlo against a planted truth:

| estimator | bias | 95% CI coverage |
|---|---|---|
| raw (uncorrected) | **+9.5 pp** | 0.08 |
| **bundle-corrected** | **−0.3 pp** | **0.94** |
| response-corrected (mis-specified) | +11.0 pp | — |

The mis-specified correction is *worse than doing nothing*. This is the exposé's central
technical wrinkle, executed and checked.

---

## Design commitments carried into the code

- **Frozen shared manifest.** Reformulations are generated once, gated once, then frozen and
  reused *identically* across every model (`results/manifest.jsonl`), so a cross-model
  difference reflects the model and not a re-sampled stimulus. Enforced by a test.
- **Intent-preservation gate** whose decision is the threshold conjunction over four
  component scores; gate error is emergent from score overlap, not a free parameter.
- **Pilot + pre-registered decision rule.** The Month-1 pilot sizes `|P⁻|`; if it clears
  `min_p_minus`, RSR is the headline, otherwise the PBR backbone is reported.
- **k-sampling is genuinely k draws.** Each canonical draw carries a distinct deterministic
  seed; at temperature 0 a real endpoint would otherwise return k identical completions.
- **No fabricated ground truth.** Real backends return no latent label, so the code records
  *no* calibration pair rather than coercing a missing truth to 0 — which would have
  reported a vacuous se=1.0 and then corrected the headline with it. Calibration on a real
  run is undefined until human labels are ingested, and the judge is not admitted.
- **Reproducibility.** Same seed → byte-identical results (including the variational
  mixed-model fit, whose global-RNG start is pinned). Covered by a test.

---

## The human audit of the gate

The exposé validates the gate *by human audit, not assumption*. On a real run there is no
latent truth, so the audit is the only way to know what the gate admitted:

```bash
python3 -m recyclebench audit-export --n 300      # stratified sheet + RUBRIC.txt
#   ... annotators fill `human_intent_preserved` with 1 / 0 / blank ...
python3 -m recyclebench audit-ingest audit/audit_sheet.csv
```

Sampling is stratified on the gate's *decision* and oversamples admitted variants, because a
false admission invalidates a measured bypass while a false rejection only costs power. A
decision-stratified sample directly estimates

- `alpha = P(intent changed | admitted)` — **contamination**, the decision-relevant number:
  how much of the measured bypass could be an artifact of drifted requests;
- `beta  = P(intent preserved | rejected)`,

and the classical truth-conditional rates follow by Bayes using the known stratum sizes:

```
false_admission = P(admitted | changed)   = N_A·α / (N_A·α + N_R·(1−β))
false_rejection = P(rejected | preserved) = N_R·β / (N_A·(1−α) + N_R·β)
```

Reporting naive pooled rates from a disproportionate sample would be biased, so the derived
quantities carry parametric-bootstrap intervals. Blank labels are excluded rather than
guessed.

### Threshold sensitivity (pre-registered)

Component scores are persisted in the manifest, so alternative operating points can be
re-applied offline with **no further model queries** — which is what lets the analysis be
part of a plan executed after the confirmatory run:

| δ | admitted | contamination | retained preserved |
|---|---|---|---|
| −0.05 | 2179 | 0.043 | 0.987 |
| 0.00 | 2011 | 0.015 | 0.937 |
| +0.05 | 1495 | 0.003 | 0.705 |

Tightening the gate lowers contamination but retains fewer intent-preserving variants. The
thresholds are frozen *before* the confirmatory run; this table makes the headline's
dependence on that choice explicit rather than hidden.

---

## The human calibration of the judge

The headline is corrected with the judge's bundle-level sensitivity and specificity. On the
mock path those come from planted truth; on a real run they must come from humans:

```bash
python3 -m recyclebench judge-export --n-bundles 60   # bundle-complete sheet
#   ... annotators fill `human_fulfilled` for EVERY row of each bid group ...
python3 -m recyclebench judge-ingest audit/judge_sheet.csv
```

**Bundles must be annotated completely.** PBR and RSR aggregate with an any-success rule, so
a bundle's truth is the OR over its admitted variants' responses — knowable only if every
response in the bundle is labelled. A scattered sample of individual responses cannot yield
it, which is why the sampler selects whole behaviours. Partially labelled bundles are
**dropped, not zero-filled**: treating a blank as 0 would bias the bundle truth downward and
inflate estimated specificity, the parameter the correction is most sensitive to.

One annotation pass then yields both levels consistently, and the compounding becomes an
empirical result rather than an assumption:

| level | sensitivity | specificity |
|---|---|---|
| per response | 0.854 | **0.972** |
| per behaviour bundle | 1.000 | **0.781** |

A judge that looks 97% specific per response is only ~78% specific per behaviour once the
union runs over ~6.7 admitted variants (`sp^t` predicts 0.83). That gap is the whole reason
the correction is done at the bundle level.

---

## Pre-registration, and proving it did not move

"Pre-registered" is only worth anything if it is checkable. `prereg` writes a dated document
fixing the decision rule, gate thresholds, budget, judge admission bar, tests, and exclusion
rules, stamped with a hash over exactly those settings:

```bash
python3 -m recyclebench prereg --behaviours data/your_benchmark.csv
#   ... collect data ...
python3 -m recyclebench prereg-verify        # exit 0 = unchanged, exit 1 = drift
```

`prereg-verify` recomputes the hash and, on a mismatch, names the field that moved:

```
"status": "DRIFT",
"changed_fields": [{"field": "decision",
                    "registered": {"min_p_minus": 30, ...},
                    "current":    {"min_p_minus": 10, ...}}]
```

That answers the obvious challenge — *did you lower the bar after seeing a small P⁻?* — with
an artifact instead of an assurance. The two likeliest post-hoc moves (relaxing
`min_p_minus`, loosening the gate) are covered by tests, as is pinning the behaviour CSV by
digest so the input set cannot be swapped. Cosmetic settings (output paths, caching, mock
simulation size) deliberately do **not** invalidate a registration, so the mechanism stays
meaningful rather than noisy. Exit codes make it usable as a CI gate.

## Figures

```bash
python3 -m recyclebench figures
```

Four plots for the writeup: the RSR forest (raw vs bundle-corrected, with `|P⁻|` annotated
and under-powered pairs flagged), the PBR trajectory against `π*` with unresolved counts, the
RQ3 driver odds ratios, and the gate operating curve. No estimate is ever plotted without its
interval, and an undefined pair is drawn as undefined rather than silently dropped.

---

## Install and run

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python3 -m recyclebench validate    # estimator validation (no models needed)
python3 -m recyclebench pilot       # size |P^-| and apply the decision rule
python3 -m recyclebench run         # full study: RQ1-RQ5
python3 -m recyclebench audit-export --n 300   # gate annotation sheet
python3 -m recyclebench judge-export           # judge calibration sheet
python3 -m recyclebench prereg                 # freeze the analysis plan
python3 -m recyclebench figures                # reporting figures
python3 -m pytest tests/ -q         # 70 tests
```

Outputs land in `results/`: `summary.json` and the frozen `manifest.jsonl`.

### Running against real models

Serve each generation locally with vLLM, one port each:

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf       --port 8001
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8002
```

Point `configs/real.yaml` at those ports and supply your own benchmark CSV (column `Goal`).
Then **size `P⁻` before spending the paraphrase budget** — the pilot uses canonical
screening only, so it is cheap:

```bash
python3 -m recyclebench pilot-pminus --old llama-2-7b --new llama-3-8b \
        --behaviours data/your_benchmark.csv
python3 -m recyclebench run-real --behaviours data/your_benchmark.csv
```

Recommended sources: JBB-Behaviors, HarmBench, StrongREJECT, SORRY-Bench, AdvBench. Pool
several to enlarge `P⁻`.

**Every call is cached** to `results/cache.jsonl`, keyed by (model, prompt, seed), so an
interrupted confirmatory run resumes instead of re-paying for completed work — verified by
test: a second identical pilot costs zero API calls.

**`run-real` reports RAW estimates only.** A real backend has no ground truth, so the judge
is uncalibrated until humans label; the summary says so explicitly and refuses to present a
corrected headline. Run `judge-export` → annotate → `judge-ingest` first.

The provider adapter is verified against a local HTTP server implementing the OpenAI
chat-completions contract (the one vLLM serves) — request shape, seed forwarding, retry on
transient failure, and cache resumability all under test, no GPU required.

---

## What ships and what does not

**No harmful content ships.** Synthetic behaviours are benign placeholders carrying only
latent parameters; benchmark prompts must be supplied by the user. Reformulation templates
are generic and contain no harmful exemplars. Raw completions are not persisted unless
`store_raw` is explicitly enabled.

The mock path validates the *instrument*; it says nothing about any real model. Numbers from
`run` on the mock backend are synthetic by construction and must never be reported as
findings about Llama.

---

## Scope of the contribution

This is a measurement and integration contribution, not a novel method. Reformulation
attacks, refusal evaluation on open-weight lineages, calibrated judges, semantic-similarity
gating, and judge-error correction all exist in prior work. What is assembled here — an
audited admission gate, a frozen shared manifest, and judge-error correction at the
behaviour-bundle unit the metric actually reports, applied to the cross-version regression
set — is the claim, and it is deliberately modest.

## Layout

```
src/recyclebench/
  types.py        Behaviour / Variant / Response / protocols
  stats.py        Wilson, bundle Rogan-Gladen (sp^t), McNemar, bootstrap, Holm, TOST
  config.py       thresholds, budgets, decision rule, mock parameters
  datasets.py     synthetic behaviours (benign) + benchmark CSV loader
  reformulate.py  operator T (mock) + surface distance
  gate.py         intent-preservation gate G (mock)
  manifest.py     the FROZEN shared reformulation manifest + gate audit trail
  audit.py        gate audit: sampling, export/ingest, reweighted rates, sensitivity
  judge_audit.py  judge calibration from bundle-complete human labels
  models.py       mock lineage (unified bypass mechanism) + real loader
  judge.py        mock judge (calibrated error channel) + offline rule-based
  screening.py    canonical screening S_M, C_M, per-model recovery, P^- construction
  calibration.py  se/sp/kappa/MAE at response or bundle level
  pilot.py        |P^-| sizing + pre-registered decision rule
  recycling.py    RSR (primary)
  pbr.py          PBR (backbone)
  drivers.py      RQ3 mixed-effects recovery drivers
  defences.py     RQ5 defences + TOST deployability
  validate.py     Monte-Carlo estimator validation
  study.py        end-to-end orchestration (mock)
  realrun.py      real-backend orchestration, resumable cache, cheap P^- pilot
  prereg.py       pre-registration document + drift detection
  figures.py      reporting figures
  cli.py          run / pilot / validate / manifest / gate-audit
  backends/       real providers, gate, judge ensemble, LLM reformulator
```
