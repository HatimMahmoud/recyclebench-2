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

## What this does and does not change

**Does:** the local runs were not a clean test of PTP at small scale. They measured an
implementation without a stop condition under a decoding protocol the pipeline does
not use. Taking that to the cluster would have reproduced the defect at twenty times
the cost.

**Does not:** PTP still contributes nothing usable to RecycleBench. The metric that
governs whether the arm can produce an admissible preimage is the intent gate's cosine
≥ 0.75, and that hit rate is **1 of 86 items** even at best-of-32 with oracle
selection. Lexical overlap improved sharply; semantic adequacy did not. That is the
paper's own lexical/semantic split landing on the unfavourable side — fragments of
wording recovered without the request being recovered.

So the open question for the cluster is unchanged in kind but much better posed: does
full-vocabulary probing close the *semantic* gap? The corpus here is 4.6M tokens reused
across epochs against the paper's ~38M tokens per epoch regenerated on the fly — about
12% of one paper epoch.

## What must be registered before this is reported

Gate A (token F1 ≥ 0.30) was pre-registered against a single greedy decode. Under
best-of-32 with a stop condition the 27M model scores 0.335 and would pass. **That is a
change of measurement protocol made after seeing the data, and it must be registered as
a deviation before it is reported as a pass.** It is not a result that can be adopted
silently. The defensible framing:

- the missing stop condition is an **implementation defect**, and fixing a defect does
  not need a protocol amendment;
- moving from greedy to best-of-N **is** a protocol change, and does need one, even
  though best-of-N is the metric the downstream estimand actually consumes;
- the cosine ≥ 0.75 hit rate is ~0 either way, so the arm's conclusion does not
  actually turn on the Gate A verdict.

Suggested: record both numbers, register the best-of-N protocol prospectively for the
cluster run, and let the gate-admissibility rate — not Gate A — carry the arm.

## Reproduce

```bash
export PYTHONPATH=src
python3 -u tools/ptp_diagnose.py
python3 -u tools/ptp_decode_sweep.py --n-samples 32
python3 -u tools/ptp_boundary.py
python3 -u tools/ptp_decode_sweep.py --checkpoint cache/ptp/inverse_boundary.pt \
    --vocab cache/ptp/compact_vocab_eop.json --eos 69367 --n-samples 32
```

Artifacts: `cache/ptp/diagnose_*.json`, `cache/ptp/decode_sweep_27m.json`,
`cache/ptp/decode_sweep_27m_boundary.json`, `cache/ptp/boundary_report_*.json`.
