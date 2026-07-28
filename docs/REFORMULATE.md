# Reformulation — how to use the paraphrase tools

The reformulator produces `p' = τ(p)`, the meaning-preserving variants that the gate then
admits or rejects. This is the **generation** side; correctness here means the variants are
intent-preserving and *target-blind* (never optimized against the model under test), so the
manifest is a metamorphic stimulus rather than an attack search.

Everything below drops into the existing pipeline: a reformulator just needs a
`.variants(behaviour) -> list[Variant]` method, and `CompositeReformulator` provides it.

## The one rule that protects the estimand

Do **not** use GCG / AutoDAN / PAIR / TAP / GPTFuzzer / PAP as the reformulator. They optimize
for bypass using target-model feedback, which confounds "does an intent-preserving rewrite
recover a fixed prompt?" with "can an optimizer find *a* jailbreak?". The tools below rewrite
blind to the target. Keep the paraphraser's model family disjoint from the target, the response
judge, and the gate's L2 classifier — `CompositeReformulator.families()` lets a run assert it.

## Install

```bash
pip install -e ".[reformulate]"          # transformers + sentencepiece + sacremoses + torch
pip install -e ".[reformulate-parrot]"   # optional: Parrot (sentence-level paraphraser)
```

The module imports and the mock path runs with none of these; DIPPER/Parrot raise a clear
error only when actually invoked, and back-translation surfaces a missing MT stack rather than
silently producing zero variants.

## The tools, and when to use each

| Tool | Best for | Notes |
|------|----------|-------|
| **LLM-prompted** (a decorrelated instruct model, e.g. Qwen/Mistral) | paraphrase, lexical, syntactic, register, **tense_shift** | highest quality, clean per-class control; one call per variant (no batching drift) |
| **Back-translation** (Opus-MT) | `back_translation` | faithful round-trip; pivots rotate (de = most semantics-preserving, zh = stronger change) |
| **DIPPER** | a graded `surface_distance` ladder | paragraph-trained → keep diversity moderate on short prompts; the knob *is* the RQ3 predictor |
| **Parrot** | short single-sentence prompts | sentence-level with built-in adequacy/fluency filtering; better than DIPPER for one-liners |

## Usage 1 — the recommended default (LLM classes + back-translation)

An open, decorrelated instruct model handles the linguistic classes; Opus-MT handles
back-translation. Reproducibility comes from **freezing the manifest**, so a closed or
non-deterministic generator is fine — the frozen variants are what ship.

```python
from recyclebench.config import Config
from recyclebench.backends.providers import ProviderConfig, ProviderModel
from recyclebench.backends.reformulators import build_reformulator
from recyclebench.manifest import Manifest
from recyclebench.gate import load_gate
from recyclebench.config import make_rng

cfg = Config.load("configs/real.yaml")

# a paraphraser from a family you are NOT testing (not Llama, not the judge, not the gate LLM)
paraphraser = ProviderModel("qwen-2.5-7b", ProviderConfig(
    base_url="http://localhost:8010/v1", model_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7, max_tokens=256, api_key_env="RECYCLEBENCH_KEY"))

reform = build_reformulator(cfg, paraphrase_model=paraphraser, use_back_translation=True)

# decorrelation check before spending the budget
assert "nli:llama-guard" not in reform.families()

gate = load_gate("real", make_rng(cfg.seed + 12), cfg.gate, cfg.mock, cfg, inventory=behaviours)
manifest = Manifest.build(behaviours, reform, gate)   # generate once → freeze
manifest.save(cfg.manifest_path)
```

## Usage 2 — add a DIPPER surface-distance ladder

Attach DIPPER as several rungs at increasing lexical diversity; `CompositeReformulator` cycles
them across the budget, so `surface_distance` sweeps from near-copy to aggressive rewrite —
exactly the RQ3 predictor. (DIPPER control codes are *similarity*; the wrapper sends `100 − diversity` for you.)

```python
reform = build_reformulator(cfg, paraphrase_model=paraphraser,
                            use_dipper=True, dipper_ladder=(20, 40, 60))
```

## Usage 3 — short prompts with Parrot

```python
from recyclebench.backends.reformulators import CompositeReformulator, ParrotGenerator, \
    LLMTransformGenerator, BackTranslationGenerator

respond = lambda s: (paraphraser.respond(s).text or "")
gens = {
    "paraphrase":       ParrotGenerator(use_gpu=True),
    "tense_shift":      LLMTransformGenerator("tense_shift", respond),
    "register":         LLMTransformGenerator("register", respond),
    "back_translation": BackTranslationGenerator(pivots=["de", "zh"]),
}
reform = CompositeReformulator(gens, cfg.transform_classes,
                               cfg.budget.paraphrases_per_behaviour, cfg.seed)
```

## Standalone tool calls (to sanity-check before a full run)

DIPPER:
```python
from recyclebench.backends.reformulators import DipperGenerator
print(DipperGenerator(lex_diversity=40, order_diversity=20).generate(
      "explain how to pick a lock", index=0, seed=0))
```

Back-translation:
```python
from recyclebench.backends.reformulators import BackTranslationGenerator
print(BackTranslationGenerator(pivots=["de"]).generate("explain how to pick a lock", 0, 0))
```

Parrot:
```python
from recyclebench.backends.reformulators import ParrotGenerator
print(ParrotGenerator().generate("explain how to pick a lock", index=0, seed=0))
```

## Gotchas the research surfaced (encoded in the wrappers)

- **No batching.** Each variant is a separate call; asking one model for N paraphrases at once
  makes the later ones drift. `CompositeReformulator` never batches.
- **No chaining.** Don't paraphrase a paraphrase — successive paraphrasing drifts toward
  attractor states. Every variant is generated from the canonical prompt.
- **Identity/empty rejected.** A rewrite equal to the source, or an empty string, is dropped
  rather than counted as a variant.
- **Reproducibility via the manifest.** Freeze once, ship the variants. Record the paraphraser
  snapshot, templates, temperature, and seed in the pre-registration.
- **DIPPER on short prompts.** Paragraph-trained; high lexical diversity can change proper
  nouns (which may alter the request). Keep it moderate and let the gate catch drift.

## Every variant still passes the gate

Generation does not trust itself. The gate (`docs/GATE.md`) admits or rejects each variant on
intent preservation, and the human audit measures the gate's false-admission rate. The
reformulator sets `intent_preserved=True` only as a placeholder; the *truth* is decided
downstream.
