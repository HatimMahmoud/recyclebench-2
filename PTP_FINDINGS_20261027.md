# PTP arm: the local failure was mostly a measurement defect

**Status: exploratory. Nothing here is confirmatory, and the Gate A question below is
not settled — see "What must be registered before this is reported".**

## What was believed yesterday

Five local PTP runs all failed Gate A (token F1 ≥ 0.30), the best reaching 0.137, and
a sixth run overnight tested model capacity and came back worse:

| run | vocab | non-emb | token F1 | cosine | val loss |
|---|--:|--:|--:|--:|--:|
| 3.5k seeds | 151,665 | 27M | 0.118 | 0.146 | 7.999 |
| 10k seeds | 151,665 | 27M | 0.100 | 0.109 | 7.191 |
| 15k seeds | 151,665 | 27M | 0.078 | 0.104 | 6.480 |
| 15k + compact vocab | 59,974 | 27M | 0.137 | 0.247 | 4.838 |
| 25k + compact vocab | 69,367 | 27M | 0.123 | 0.231 | 4.888 |
| 25k + compact vocab | 69,367 | 92M | 0.101 | 0.155 | 5.695 |

Trivial baselines on the same held-out set: a **fixed generic string** scores 0.246
token F1, and **echoing the response back** scores 0.713 cosine. Every model was below
both. The reading was "PTP needs the paper's scale", and the cluster run was to be
sized accordingly.

That reading was wrong, or at least premature.

## Attribution (`tools/ptp_diagnose.py`)

Before spending H100 hours, the failure was attributed by separating three outcomes:
the loop is broken, the mapping has not generalised, or decoding is throwing away a
mapping that was learned.

| | teacher-forced, seen | teacher-forced, unseen | free-running F1, seen | free-running F1, unseen |
|---|--:|--:|--:|--:|
| 27M | **0.605** | 0.286 | 0.170 | 0.123 |
| 92M | 0.330 | 0.211 | 0.103 | 0.101 |

Given the true prefix, the 27M model predicts 60.5% of prompt tokens correctly on
pairs it was fine-tuned on, against a chance rate of 1/69,367. It has learned a great
deal of the inverse mapping. Free-running decoding collapses that to 0.170.

Accuracy by depth into the prompt (unseen, 27M), measured backwards from the response
boundary:

| depth | 1 | 2 | ≤4 | ≤8 | ≤16 |
|---|--:|--:|--:|--:|--:|
| accuracy | 0.326 | **0.047** | 0.256 | 0.274 | 0.392 |

The model is good at continuing a prompt backwards and bad at the hand-off from
response into prompt. Depth 1 is usually punctuation; depth 2 is the first real
content token, inferred from the response alone, and it is where the chain breaks.

## Two defects in our own measurement

**1. Every PTP figure reported a single greedy decode.** `invert_ids` runs at
temperature 0. The paper's section 5 reconstructs each prompt several times, and more
importantly RSR/PRR are "at least one admitted candidate re-opens the behaviour"
estimands — downstream, every candidate goes through the intent gate and is
re-screened. Scoring one greedy decode measures something the pipeline never does.

**2. There was no stop condition.** In `s_rev = Reverse(prompt + response)` the
sequence ends at the prompt's first token, and nothing marked it. Decoding ran a fixed
40 tokens and sailed past the prompt's start into invented text. True prompts average
**10.8 tokens**, so precision was structurally capped near 0.27 however good the model
was.

Measured (27M, 86 held-out pairs, `tools/ptp_decode_sweep.py`):

| decode | token F1 | precision | recall | cosine |
|---|--:|--:|--:|--:|
| greedy | 0.123 | 0.079 | 0.296 | 0.231 |
| best-of-8 | 0.188 | 0.121 | 0.452 | 0.355 |
| best-of-32 | 0.217 | 0.140 | **0.522** | 0.408 |

Recall of 0.522 says over half the prompt's tokens were already being recovered;
precision of 0.140 says most of what was emitted was overrun.

## The fix (`tools/ptp_boundary.py`)

One end-of-prompt token appended to every fine-tuning sequence, decoded as the stop
token. Same pre-trained checkpoint, same 347 fine-tuning pairs, same hyper-parameters,
same held-out 86, same sweep — only the marker differs.

| best-of-32, 27M | no boundary | with boundary |
|---|--:|--:|
| token F1 | 0.217 | **0.335** |
| precision | 0.140 | **0.346** |
| recall | 0.522 | 0.472 |
| cosine | 0.408 | 0.402 |
| items with ≥1 candidate at F1 ≥ 0.30 | 8.1% | **60.5%** |
| items with ≥1 candidate at cosine ≥ 0.75 | 0.0% | 1.2% |

The same fix on the 92M model reproduces the effect (best-of-32 token F1 0.269,
precision 0.277, Gate A hit rate 39.5%), so it is not specific to one model size. The
27M still beats the 92M on every measure, consistent with the 92M being undertrained
rather than capacity-limited.

All three sweeps side by side (86 held-out pairs, best-of-32):

| run | stop token | greedy F1 | bo32 F1 | precision | recall | cosine | hit F1≥.30 | hit cos≥.75 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 27M | no | 0.123 | 0.217 | 0.140 | 0.522 | 0.408 | 0.081 | **0.000** |
| 27M | yes | 0.131 | **0.335** | 0.346 | 0.472 | 0.402 | **0.605** | 0.012 |
| 92M | yes | 0.105 | 0.269 | 0.277 | 0.437 | 0.346 | 0.395 | 0.012 |

Token F1 is now above both Gate A (0.30) and the generic-string baseline (0.246).
Recall dips slightly because the marker sometimes fires early — under greedy decoding
it fires almost immediately, which is why greedy barely moves (0.123 → 0.131) while
best-of-N moves a lot.

Folded into the package: `reversal.build_training_sequence(..., eop=)`,
`train.make_finetune_corpus(..., eop=)`, `ptp_build.py --boundary`, and
`tests/test_ptp.py::test_boundary_token_marks_where_the_prompt_starts` so a future run
cannot silently lose it.

## Does more fine-tuning data close the semantic gap? (yes, partly)

The boundary fix moved lexical recovery and left the semantic side flat. The depth
profile said why: the break is at the hand-off from response into prompt, and that is
what the fine-tuning stage teaches — with 347 pairs. Scaling the fine-tuning corpus to
2,846 pairs (Alpaca instructions, responses from the same forward model, same frozen
held-out 86) moved everything, including cosine.

Because more pairs also means more optimisation steps, a compute-matched control was
run: the original 347 pairs for 98 epochs = 4,251 steps, against the 2,846-pair run's
4,272.

| run | steps | bo32 F1 | precision | recall | cosine | hit F1≥.30 | hit cos≥.75 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 27M · 347 pairs · no stop | 528 | 0.217 | 0.140 | 0.522 | 0.408 | 8.1% | 0.0% |
| 27M · 347 pairs · stop | 528 | 0.335 | 0.346 | 0.472 | 0.402 | 60.5% | 1.2% |
| 92M · 347 pairs · stop | 528 | 0.269 | 0.277 | 0.437 | 0.346 | 39.5% | 1.2% |
| 27M · 347 pairs · stop **[control]** | 4251 | 0.314 | 0.329 | 0.450 | 0.383 | 48.8% | 1.2% |
| 27M · 2,846 pairs · stop | 4272 | **0.397** | 0.426 | 0.505 | **0.489** | 72.1% | **4.7%** |
| 27M · 5,912 pairs · stop | 4434 | **0.424** | 0.477 | 0.511 | **0.501** | 77.9% | **9.3%** |

Eight times the compute on the *same* data made best-of-32 slightly **worse** (0.335 →
0.314) and the fine-tune loss much lower (best 1.46 vs 2.34) — it memorised. At matched
compute, eight times the *data* gave +0.083 token F1, +0.106 cosine and a four-fold
gate-hit rate. The gain is data diversity, and the fine-tuning corpus is generated with
Ollama, so this axis is cheap and not GPU-bound.

A third point at 5,912 pairs (steps again held near 4,300) continued the trend without
any sign of flattening:

| fine-tune pairs | gate-admissible (cos ≥ 0.75, best-of-32) | Wilson 95% |
|--:|--:|---|
| 347 | 1/86 = 1.2% | [0.2%, 6.3%] |
| 2,846 | 4/86 = 4.7% | [1.8%, 11.4%] |
| 5,912 | 8/86 = 9.3% | [4.8%, 17.3%] |

The per-segment exponent is 0.66 then 0.95 — if anything the curve is steepening, not
saturating. A power law through the three points reaches 20% at ≈19,000 pairs and 30%
at ≈33,000. **The honest caveat is that these are counts of 1, 4 and 8 out of 86, and
the last two Wilson intervals overlap substantially.** Three thin points are a weak
basis for extrapolating 3x beyond the last measurement; the direction is consistent,
the slope is not well determined.

## What this does and does not change

**Does:** the local runs were not a clean test of PTP at small scale. They measured an
implementation without a stop condition under a decoding protocol the pipeline does
not use. Taking that to the cluster would have reproduced the defect at twenty times
the cost.

**Does not:** PTP still contributes nothing usable to RecycleBench *at present scale*.
The metric that governs whether the arm can produce an admissible preimage is the intent
gate's cosine ≥ 0.75, and that hit rate is **8 of 86 items** at best-of-32 with oracle
selection. Lexical overlap improved much faster than semantic adequacy — the paper's own
lexical/semantic split landing on the unfavourable side, fragments of wording recovered
without the request being recovered. The difference from this morning's reading is that
the semantic side is no longer flat: it is moving, on an axis that costs Ollama time.

So the open question for the cluster is better posed, and partly answered: the
fine-tuning corpus moves the semantic side and costs Ollama time, not GPU time, while
the pre-training corpus (3.5k → 25k seeds) and model capacity (27M → 92M) did not move
it at all. Full-vocabulary probing remains untested — 4.6M tokens reused across epochs
against the paper's ~38M per epoch regenerated on the fly, about 12% of one paper epoch.

**Is the arm worth scaling?** The stop rule set earlier today was: extend the gate-hit
curve to a third point, and if it does not extrapolate past ~20%, close the arm. **It
does extrapolate past 20%, so the rule does not fire, and the earlier recommendation to
close the arm is withdrawn.**

Two qualifications keep this from being an endorsement of a large commitment:

1. **Scale still does not mean GPU.** The axis that moves is the fine-tuning corpus,
   generated by Ollama on the laptop at ~1.1 s/pair. Reaching ~19,000 pairs is roughly
   four more hours of local generation plus a 20-minute fine-tune — an overnight job,
   not a cluster allocation. The two GPU-bound axes (pre-training corpus, model
   capacity) remain flat and negative. So the case for *scaling PTP* has strengthened
   while the case for *spending cluster hours on PTP* has not.
2. **Even a successful projection does not make PTP competitive.** 20% gate-admissibility
   would still be a quarter of the RPE arm's 80.0% PRR [49.0, 94.3], which an untrained
   `qwen2.5:14b` achieves with no training at all. PTP's claim was never to beat RPE on
   yield; it was to be data-free and to invert the forward model's own dynamics rather
   than lean on a general model's world knowledge. That validity threat is already
   evidenced away by the blind and mismatched controls (equivalence medians 0.752 vs
   0.172 / 0.175, zero control candidates reaching preimage at any threshold), so the
   data-free property is a nice-to-have, not a gap in the thesis.

The defensible plan: one more local scaling point (~12k–19k pairs, overnight, Ollama
only) to test whether the power law holds or breaks, and report PTP either as a
converging method with a measured scaling curve or as a bounded negative with the curve
as evidence. Either way the write-up is stronger than it was this morning, and cluster
time still belongs to the white-box arm, which speaks to RQ1 directly.

## What must be registered before this is reported

Gate A (token F1 ≥ 0.30) was pre-registered against a single greedy decode. Under
best-of-32 with a stop condition the best run now scores 0.424 and would pass; under
the registered greedy protocol the same run scores 0.210 and would fail. **The verdict
is decided entirely by a change of measurement protocol made after seeing the data, so
it must be registered as a deviation before it is reported as a pass.** It is not a
result that can be adopted silently. The defensible framing:

- the missing stop condition is an **implementation defect**, and fixing a defect does
  not need a protocol amendment;
- moving from greedy to best-of-N **is** a protocol change, and does need one, even
  though best-of-N is the metric the downstream estimand actually consumes.

This now matters more than it did at the time of the 4.7% result. The earlier draft
argued the Gate A verdict was moot because the gate-admissibility rate was ~0 under
either protocol; at 9.3% and rising that argument no longer holds, and the arm may
continue on the strength of a number produced by an unregistered protocol. **No further
PTP scaling point should be reported as evidence for continuing the arm until the
best-of-N protocol is registered prospectively.** Running the next point is fine —
measurement is not reporting — but the registration must precede the claim.

Suggested: record both numbers (greedy and best-of-N) for every run, register the
best-of-N protocol prospectively with the gate-admissibility rate as the arm's primary
criterion, and state Gate A's greedy verdict as a failed pre-registered check rather
than quietly superseding it.

## Reproduce

```bash
export PYTHONPATH=src
python3 -u tools/ptp_diagnose.py
python3 -u tools/ptp_decode_sweep.py --n-samples 32
python3 -u tools/ptp_boundary.py
python3 -u tools/ptp_decode_sweep.py --checkpoint cache/ptp/inverse_boundary.pt \
    --vocab cache/ptp/compact_vocab_eop.json --eos 69367 --n-samples 32

# the fine-tune scaling curve (Ollama for the pairs, then fine-tune + score)
python3 -u tools/ptp_build.py --phase ftdata --vocab-min-count 3 --workers 6 --minutes 70
python3 -u tools/ptp_boundary.py --pretrained cache/ptp/local27m_25k/inverse_pretrained.pt \
    --ft-epochs 6 --out-ckpt cache/ptp/inverse_boundary_ft6k.pt \
    --out-vocab cache/ptp/compact_vocab_eop.json
python3 -u tools/ptp_decode_sweep.py --checkpoint cache/ptp/inverse_boundary_ft6k.pt \
    --vocab cache/ptp/compact_vocab_eop.json --eos 69367 --n-samples 32 \
    --out cache/ptp/decode_sweep_27m_ft6k.json
```

The compute-matched control is the same `ptp_boundary.py` with `--n-pairs 347` and the
epoch count raised to hold total steps constant.

Artifacts: `cache/ptp/diagnose_*.json`, `cache/ptp/boundary_report_*.json`, and the
sweeps `decode_sweep_27m{,_boundary,_compmatched,_ftscale,_ft6k}.json` plus
`decode_sweep_92m_boundary.json`. Logs: `logs/ptp*_20261027_*.log`.
