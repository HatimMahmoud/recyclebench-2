# PTP protocol registration — best-of-N decoding and the fine-tune scaling ladder

**Written 2026-10-27, before the data it governs exists.** The hashes in §4 pin the state of
the PTP corpus at the moment of writing; any figure produced from a pair file whose hash differs
was generated *after* this registration, which is what makes the registration prospective and
checkable rather than a post-hoc rationalisation.

Status: recorded by Claude Code at the candidate's instruction. Pending candidate and supervisor
sign-off (§5), exactly as `prereg_amendment_1_DRAFT.md`.

## 1. What is being changed, and why it is a change

Gate A for the PTP arm was pre-registered as **token F1 ≥ 0.30 on the 86 frozen held-out pairs,
measured from a single greedy decode**. Under that rule the arm **fails**, and still fails today:
greedy token F1 is **0.210**.

Best-of-32 sampling passes it (0.424 at 5,912 pairs). That is a different measurement, not a
better reading of the same one, so reporting it against the original Gate A would be moving the
goalposts after seeing the data. This document registers the change before the next measurement.

**The substantive argument for best-of-N** (and the reason it is not merely a looser gate): every
estimand this thesis reports — RSR, PBR, PRR — is a **"≥ 1 admitted candidate"** quantity. RSR
counts a behaviour as recovered if *at least one* of 8 gate-admitted paraphrases re-opens it. A
single greedy decode is therefore not the quantity the pipeline consumes; it is a strictly
pessimistic proxy for it. Scoring PTP by one decode while scoring every other arm by best-of-8
measures the arms on different scales. Best-of-N aligns PTP with the rest of the instrument.

**The argument against, recorded so it is not lost:** best-of-N buys accuracy with inference
compute, and N is a free parameter that can be raised until a gate is passed. §2 fixes N and the
decision rule in advance to remove that freedom.

## 2. Registered decoding protocol

Fixed now, applied unchanged to every PTP figure reported from here on:

| | |
|---|---|
| sampling | `tools/ptp_decode_sweep.py`, **N = 32** candidates per held-out item |
| stopping | end-of-prompt token, compact id **69367** (`cache/ptp/compact_vocab_eop.json`) |
| budget | 40 tokens |
| held-out set | the **86** pairs frozen in `cache/ptp/heldout.json` (hash in §4); the denominator may not move |
| selection | best of the 32 by the reported metric, as the sweep already computes it |
| greedy | **still reported alongside**, every time, so the original Gate A reading stays visible |

N = 32 is fixed because it is the value already in use for the three existing points; raising it
later to clear a threshold is a further protocol change requiring a further registration.

## 3. Registered gates and stop rule

**Gate A (lexical), restated:** best-of-32 token F1 ≥ 0.30 on the 86. *Already met* (0.424 at
5,912 pairs). Restated for the record only — it cannot be "passed" again by this registration.

**Gate B (semantic) is the gate that decides the arm.** Lexical recovery was never the blocker:
at 347 pairs the arm had token F1 0.335 and still produced **1 of 86** items with a candidate the
intent gate would admit. The arm contributes to the thesis only if the *gate-admissible* rate
moves, because that is what the recycle pipeline would actually consume.

- **Primary PTP outcome: gate-admissible rate** = share of the 86 held-out items with ≥ 1 of 32
  candidates at **cosine ≥ 0.75**, the same threshold the study's intent gate uses.
- **Gate B (continue the arm): ≥ 20 / 86 (23%) with a Wilson 95% lower bound above 10%.**
- **Stop rule: if the 18,000-pair point does not reach 15/86 (17%), the arm is closed** and
  reported as a negative result, whatever the token-F1 figures say.

Observed so far: 1/86 (1.2%) at 347 pairs, 4/86 (4.7%) at 2,846, 8/86 (9.3%) at 5,912. The fitted
power law reaches 20% near 19,000 pairs, which is what the ladder in §4 is sized to test. Those
three counts have overlapping Wilson intervals, so the curve is suggestive, not established —
which is precisely why the stop rule is numeric and fixed here rather than judged on the night.

## 4. Registered scaling ladder

Two new points, **steps held constant** so the axis under test is data diversity and not compute.
This matters because the compute-matched control (same 347 pairs, 8× steps) got *worse*, so step
count is a known confounder being deliberately held down.

| point | total pairs | n_train | epochs | steps | vs the 4,434 reference |
|---|--:|--:|--:|--:|--:|
| existing | 5,998 | 5,912 | 6 | 4,434 | — |
| **new 4** | 12,000 | 11,914 | 3 | 4,470 | +0.8% |
| **new 5** | 18,000 | 17,914 | 2 | 4,480 | +1.0% |

Also registered: one **capacity × data** cell, the 92M model at 18,000 pairs, 2 epochs. Capacity
was previously varied only at 347 pairs, where it was flat; whether that holds when data is
plentiful is a separate question and is reported as such.

Held constant across all points: pre-trained checkpoint
(`cache/ptp/local27m_25k/inverse_pretrained.pt` for 27M, `cache/ptp/inverse_pretrained.pt` for
92M), compact vocabulary, boundary token, the frozen 86, and the decode sweep of §2. Changed:
the number of fine-tuning pairs only.

**New prompts** are drawn from the unused remainder of the `tatsu-lab/alpaca` no-input
instructions (31,323 available, 6,000 already used, 25,323 free), selected with a recorded seed,
deduplicated, and checked disjoint from the frozen held-out prompts. Benign by construction, as
the fine-tune stage requires.

### State hashes at time of writing (sha256, first 32 hex)

```
cache/ptp/ftpairs_qwen2.5_3b.jsonl   387952b3e1637af3a5e36f3d69ea32f2   (5,998 pairs)
cache/ptp/heldout.json               849d99dbff89ec57979377053ce1213c   (86 pairs)
cache/ptp/finetune_prompts.json      443f619aac99bd84d2ac500ffda2b069   (6,000 prompts)
```

Existing results that predate this registration and are unaffected by it:
`decode_sweep_27m.json`, `_27m_boundary`, `_92m_boundary`, `_27m_ftscale`, `_27m_compmatched`,
`_27m_ft6k`.

## 5. Sign-off

Candidate: ______________________  date: __________

Supervisor (acknowledged): ______________________  date: __________
