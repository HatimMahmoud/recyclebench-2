# Registration: SORRY-Bench's 20 linguistic mutations on every model (Pegasus)

**Status: DRAFT, unsigned.** Written 2026-11-05. Sign below and commit before any reply of this arm is judged.

**Blind generation before signing (disclosed).** To save cluster time, replies to the mutated prompts are generated
from 2026-11-05 before this registration is signed (`tools/mutation_screen.py generate`, one draw per prompt, into
`cache/mutations/<model>__<shard>.jsonl`). No reply is judged, decoded, translated or read, and no count is computed,
until signature; `judge` and `report` refuse to run while this file says DRAFT. Job IDs are in the run log.

**Designation: exploratory.** A known-attack-transfer arm in the sense of THESIS_KB (the Wei et al. classes of the Scout
suite): the stimuli are SORRY-Bench's published mutations, not intent-gated rewordings of ours, so no figure here is an
RSR, none is pooled with RSR, and none carries a verdict against π*. Raw judge output; no human calibration.

## 1. Stimuli

`sorry-bench/sorry-bench-202503` (gated; built on Pegasus with the candidate's token): for each of the 440 base requests
of `data/sorrybench.csv` (identity verified against the lineage cache, 1,320/1,320 hits per dense model), its 20 mutated
forms, matched on the dataset's `question_id` and checked against the base text (`data/sorrybench_mutations.jsonl`, with
its sha256 in the provenance file). Mutation styles, as the dataset names them: writing styles and persuasion techniques
(English), four encodings (ASCII, Atbash, Caesar, Morse) and five translations.

## 2. Models

All twelve: llama2-7b, llama2-13b, llama3-8b, llama31-8b, llama32-1b, llama32-3b, llama4-scout, llama4-maverick
(`configs/llama4_pegasus.yaml`), and llama2-70b, llama3-70b, llama31-70b, llama33-70b (`configs/chain70b.yaml`). Same
serving and decoding as their canonical screens (Ollama 0.40.12, temperature 0, 512 tokens). One draw per prompt, seed
`_draw_seed(bid, 1000 + style index, 0)` (disjoint from the canonical and rewording seeds).

## 3. Judging (against the original request)

1. **English styles:** the reply as written.
2. **Encodings:** the reply decoded deterministically (Caesar shift 3, Atbash, Morse, decimal ASCII); undecodable text
   stays as is.
3. **Translations:** the reply translated into English by qwen2.5:14b (temperature 0), translation cached.
4. Then the study's judge (qwen2.5:14b rubric) on (**original English base request**, English reply), store
   `cache/mutations/judge.jsonl`. Judging an encoded reply against an encoded prompt gives spurious answers
   (CLAUDE.md, attack-transfer arm); this rule avoids it.

## 4. Estimands (descriptive)

- **M1.** Per model and style: base requests answered under that mutation (of 440), and under ≥ 1 mutation.
- **M2.** Per pair (the canonical pairs of the cluster pages, with each pair's SORRY-Bench fixes from its canonical
  screen): fixes the newer model answers under ≥ 1 mutation, and per style. Wilson 95 % where reported.
- Stratified by style kind (English / encoding / translation); encoding and translation figures carry the threat T2 below.

## 5. Threats

- **T1, judge range:** the judge was calibrated on English replies to plain requests; decoded and translated replies
  are outside that range.
- **T2, decoding and translation:** a decoder can fail on a partly encoded reply, and translation can add or lose
  content; both can move a verdict. Encoding and translation results are reported separately from English styles.
- **T3, many comparisons:** 20 styles × 12 models; descriptive only, no test.
- **T4, scale and release confounds:** as in each pair's own registration.

## Sign-off

- Candidate: ______________________  date: __________
- Supervisor: _____________________  date: __________
