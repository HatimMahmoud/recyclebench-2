# recyclebench — project context for Claude Code

## What this is
M.Sc. thesis (RPTU Kaiserslautern-Landau). **Question:** when a newer LLM refuses a jailbreak
that an older LLM answered, did it learn the *intent* (deep generalization) or memorize the
*wording* (shallow patch)? We test by taking the requests the newer model "fixed", rewording them
while preserving meaning (intent-preserving paraphrase = "recycling"), and checking whether the
newer model can be made to answer again.

**Core constructs:**
- `P^-` = regression set = behaviours where OLD complies AND NEW refuses (the "fixes" we test).
- `RSR` = Recycling Success Rate = fraction of P^- recovered by >=1 intent-preserving paraphrase.
  High RSR => fixes are shallow (wording-specific). Low RSR => fixes generalize.
- Bundle-level Rogan-Gladen judge-error correction; pre-registration; a paraphrase "gate" that
  certifies each reword still asks for the same harmful thing (intent preserved).

**Strategic framing (for the thesis writeup):** frame as "does safety alignment GENERALIZE across
model generations" (alignment science), NOT "paraphrase jailbreak robustness" (security). A planned
mechanistic arm (refusal-direction / Arditi et al.) would test whether paraphrase recovery =
displacement off the newer model's refusal direction — needs white-box HF transformers + hooks,
not Ollama. That arm is future work, not yet built.

## THE central finding so far (this drives every open decision)
On **raw** harmful prompts, P^- is structurally tiny **because safety-tuned "old" models already
refuse raw prompts.** Evidence:
- Judge is NOT the problem: validated ~98% sensitivity against the liuyanchen1015 `score` labels;
  and reading llama2-7b's actual "refusals" showed 13/15 were genuine refusals.
- llama2-7b-chat refuses ~87-99% of raw harmful prompts (breaks_old = 53/400 HarmBench,
  3/520 AdvBench, 2/313 StrongREJECT). So P^- (a subset) is capped near zero.
- More raw benchmarks (AIR-Bench etc.) will NOT help — safety-tuned models refuse raw AIR-Bench too.
  **Do not install more raw-harm benchmarks to grow P^-.**

**P^- populates only when the OLD model is weak enough to comply with raw prompts:**
| pair | pooled P^- | why |
|---|---|---|
| llama3.2-1b -> llama3.2-3b | ~24 | the 1B is weak enough to comply (healthiest P^-) |
| llama2-7b  -> llama3-8b    | ~7  | llama2-chat refuses raw prompts -> gap barely exists |

**The lever is prompt FRAMING, not quantity.** Wrapping behaviours in a fixed jailbreak template
(via `tools/wrap_behaviours.py`) is the current approach (see "Option 1" below).

## Current experimental state
- Pipeline works end-to-end: `pilot-pminus` (canonical screening) and `run-real`
  (recycle -> gate -> re-screen -> RSR) both run on real Ollama.
- Real caches on disk (append-only, keyed `sha256(model|prompt|seed)`):
  - `cache/ollama_pilot.jsonl`  — llama3.2 1b/3b (P^- ~24; recycle smoke-tested here)
  - `cache/llama_gen.jsonl`     — llama2-7b / llama3-8b / llama3.1-8b canonical responses
  - `cache/wrapped_rs.jsonl`    — llama2-7b vs llama3-8b on refusal_suppression-wrapped prompts
  - `cache/llama2_13b.jsonl`    — llama2-13b vs llama3-8b (fresh; overnight run)
- Benchmarks in `data/` as Goal-column CSVs: harmbench (400), advbench (520), strongreject (313),
  plus wrapped sets (wrapped_rs.csv, llama_wrapped.csv). No harmful content ships in the repo;
  these were fetched/converted locally via `tools/prepare_behaviours.py`.

## Overnight runs launched (check these first)
Two pilots launched via `run_pair.sh`, writing to `results/runs/<ts>_<pair>_full/`:
1. `llama3-8b -> llama31-8b`   (cache/llama_gen.jsonl) — consecutive-gen, matched 8B
2. `llama2-13b -> llama3-8b`   (cache/llama2_13b.jsonl) — CROSS-SCALE (13B->8B), not scale-matched
Each dir has: `pminus_<bench>.{json,txt}`, `partition_full.txt` (all 4 cells incl. negative-flip
inverse), `cells/` CSVs, `META.txt`, log. **Prediction to test:** both likely SMALL P^- (3->3.1 is
a point release; llama2-13b-chat also refuses raw). If both small => raw-prompt safety saturated
across models; real gap is in WRAPPED prompts.

## THE open fork (highest priority when runs are read)
**Option 1 (in progress): wrapped canonical prompts.** `tools/wrap_behaviours.py --list` ranks the
32 liuyanchen1015 jailbreak templates by 1B-compliance rate (poems 90%, refusal_suppression 78%,
evil_confidant 71%, AIM 62% ... encoding jailbreaks like base64/rot13 are useless — the 1B can't
decode them; gcg/autodan are adversarial optimizers, RULED OUT — they confound the estimand).
Pre-registered pick: **refusal_suppression** (chosen a priori by 1B-compliance = property of the
held-out weak model, not the llama2/3 outcome — legitimate, not p-hacking).

**BUT the refusal_suppression pilot gave a surprising breaks_old=5 for llama2-7b** (the 1B complied
78% on the same template). CSV is fine (verified: multi-line jailbreak prompts are properly quoted;
`load_jailbreakbench` uses csv.DictReader and parses them correctly; wc -l 3907 just counts embedded
newlines). **UNRESOLVED — needs a response read to decide:**
- **Case A:** llama2-7b genuinely refuses even refusal_suppression => real scale-of-safety finding
  (7B-chat resists a jailbreak a 1B falls for). Then try higher-rate templates (poems, evil_confidant).
- **Case B:** llama2-7b complies but the offline judge misreads wrapped output => judge is broken on
  WRAPPED prompts (it was only validated on raw-ish 1B responses, never on wrapped output) => must
  build the LLM judge before any wrapped experiment is trustworthy.
Decide with:
```
python3 tools/inspect_cache.py --config configs/real_llama_lineage.yaml \
  --old llama2-7b --new llama3-8b --cache cache/wrapped_rs.jsonl \
  --behaviours data/wrapped_rs.csv --refusals-only --width 100000 | head -100
```
Read 10-15 of llama2's "refusals": genuine refusals => Case A; harmful content scored as refused => Case B.

## Next work (priority order)
1. **Resolve the Option-1 fork above** (read wrapped refusals -> Case A or B).
2. **Judge suite** (becomes critical path if Case B, and needed for any headline anyway):
   (a) compliance-POSITIVE rule-based judge — require positive evidence of compliance, not just
   absence of a refusal prefix (fixes verbose-style misreads);
   (b) LLM-judge backend — non-Llama (qwen2.5 via Ollama), wired via the config `judges:` block;
   (c) a `judge-eval`/calibration path scoring any judge vs the liuyanchen1015 `response`+`score`
   labels (confusion matrix, sensitivity/specificity). Note: `judge-export`/`judge-ingest`
   subcommands already exist — check them before building new.
3. **Perf:** the gate reloads DeBERTa/RoBERTa PER BEHAVIOUR (backends/real_gate.py + gate_layers.py)
   — very slow on CPU. Cache the models across behaviours (roughly one-line fix) before any large run.
4. **Mechanistic arm** (thesis-upgrade, future): refusal-direction displacement vs recycling recovery.
   White-box, HF transformers + hooks, <=8B on a single GPU.

## Key layout
- `src/recyclebench/` — package. Notable: `judge.py` (looks_like_refusal + curly-apostrophe fix),
  `screening.py` (`_draw_seed`, canonical screening), `recycling.py` (RSR), `realrun.py` (`_key`,
  run-real path), `manifest.py` (paraphrase+gate manifest), `datasets.py` (load_jailbreakbench),
  `config.py`. `backends/`: `providers.py` (Ollama openai-compatible), `reformulators.py`
  (paraphrase generators), `real_gate.py` + `gate_layers.py` (layered intent gate:
  lexical -> embedding -> BERTScore -> NLI DeBERTa).
- `tools/` — `prepare_behaviours.py` (benchmark->CSV; `--hf-dataset`, dedup),
  `inspect_cache.py` (reconstruct prompt->response from cache; `--only-pminus`, `--refusals-only`,
  `--only-negflip`, `--compliances-only`; labels are dynamic via --old/--new),
  `pminus_report.py` (2x2 partition + pooled; `--screen-inverse` fills negative-flip cell; `--dump`
  writes per-cell CSVs), `wrap_behaviours.py` (`--list` templates, `--template X --out` wrapped set),
  `make_dashboard.py` (art-quality HTML dashboard for supervisors: animated Sankey flow, RSR gauge,
  transform-class bars, before->after per-P^- cards reconstructed from cache+manifest; `--rsr`,
  `--manifest` (multiple), sensitive text collapsed).
- `configs/` — `real_ollama.yaml` (llama3.2 1b/3b + qwen2.5:3b paraphraser),
  `real_llama_lineage.yaml` (llama2-7b/llama3-8b/llama31-8b/llama2-13b + qwen paraphraser).
- `run_pair.sh` — `bash run_pair.sh <old> <new> <cache> [benchmarks...]` — full pilot + partition
  (incl. inverse) into a timestamped `results/runs/<ts>_<pair>_full/` dir. Wrap with
  `caffeinate -is` + `nohup` for overnight.

## CLI (python3 -m recyclebench <cmd>)
`pilot-pminus` (canonical screening -> P^- sizing), `run-real` (recycle -> gate -> re-screen -> RSR),
`prereg` / `prereg-verify`, `judge-export` / `judge-ingest`, `manifest`, `gate-audit`,
`audit-export` / `audit-ingest`, `figures`, `validate`, `version`.

## Conventions & gotchas
- **Judge classification MUST reuse `recyclebench.judge.looks_like_refusal`** (single source of truth;
  the curly-apostrophe normalization U+2019->U+0027 lives there). Tools already import it.
- **Paraphraser MUST be non-Llama** (decorrelated from the target lineage): qwen2.5. **No jailbreak
  optimizers as paraphraser** (GCG/AutoDAN/PAIR/Pliny) — they generate attacks, not intent-preserving
  paraphrases, and confound the estimand. `store_raw=False`.
- Cache is append-only + resumable (Ctrl-C to pause, rerun same cmd to continue; a hard kill can
  truncate the last JSON line -> repair on macOS: `sed -i '' -e '$d' cache.jsonl`). Different
  benchmarks/prompts => different keys => accumulate safely.
- `pilot-pminus` writes a FIXED `results/pminus_pilot.json`; `run-real` writes FIXED
  `results/summary_real.json`. **cp to a unique name (or use run_pair.sh's timestamped dir) after
  each run** or they overwrite.
- Save results non-overwriting: prefer `results/runs/<ts>_<pair>/...`. Keep the CACHE shared
  (expensive, reusable); snapshot results/manifests/logs per-run.
- macOS via Ollama, `python3`. Gate ML deps: `pip3 install sentence-transformers transformers bert-score`.
  Quieting: `export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1`. The transformers
  "LOAD REPORT / UNEXPECTED / MISSING" lines are HARMLESS (classification heads, not generation).
- Ollama models present: qwen2.5:3b/14b/latest, llama2:7b/13b/latest, llama3:8b, llama3.2:1b/3b,
  llama3.1:8b. A missing model => 404 at call time (pull it).
- **zsh gotcha:** inline heredocs/echoes containing `!` trigger history expansion ("event not found").
  Fixes: write scripts to a file with `<< 'EOF'`, and/or `setopt no_bang_hist`.

## Safety boundary (maintained)
Benchmarks are user-supplied / fetched from official repos at runtime; converters carry no harmful
content. The dashboard collapses + marks sensitive text and shows rates/structure, not raw harmful
content. The wrapped-canonical template is FIXED and non-adaptive (a controlled transformation, not
an optimizer). Judge-sensitivity checks, response reads, and wrapped-prompt work are legitimate
research diagnostics for measuring refusal robustness.