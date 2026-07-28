# The intent-preservation gate G(p, p')

The gate is the construct-validity core: it certifies that a paraphrase `p'` still asks for the
*same harmful thing* as the canonical prompt `p`, so a measured recycling bypass cannot be
dismissed as a diluted or altered request. It is a **conjunction of layers**, each a necessary
condition, evaluated cheapest-first with short-circuit; its own error is *measured* by the
human audit rather than assumed.

## Layers

| layer | what it checks | backend | threshold (frozen) |
|------|----------------|---------|--------------------|
| L0a lexical divergence | `p'` is genuinely reworded, not a near-copy | pure Python (token edit distance) | `min_lexical_divergence` (default 0.15) |
| L0b embedding cosine | surface similarity floor | sentence-transformers MiniLM (or TextAttack USE) | `cfg.gate.cosine` |
| L0c BERTScore F1 | token-level similarity floor | bert-score (or TextAttack) | `cfg.gate.bertscore` |
| L1 bidirectional NLI | `p ⊨ p'` **and** `p' ⊨ p` | DeBERTa-MNLI cross-encoder | `cfg.gate.nli_entail` |
| L2 discriminative intent | `p'` still *uniquely identifies* its behaviour against distractors | injected LLM, **decorrelated from the response judge** | `cfg.gate.llm_intent` |

A variant is admitted iff **every enabled layer** admits. The first layer to fail is recorded
as `rejected_by`, so a rejection is explainable and the audit can stratify on it.

## Why layered (not a single similarity score)

The adversarial-NLP literature documents that USE and BERTScore fail to detect meaning changes
in adversarial text — a similarity-only gate admits softened requests. L1 adds a logical
equivalence relation; L2 adds a *discriminative* intent check (does the paraphrase still pick
out its own behaviour, or has it drifted to another?), which is the component no released tool
ships and where this gate's novelty over ParaGuard lives.

## Independence (why the model choices matter)

The bundle-level judge-error correction assumes gate errors and response-judge errors are
uncorrelated. Keep three model families disjoint:

* **paraphraser** — the LLM that generates `p'`
* **gate** — DeBERTa encoder (L1) + a non-judge LLM (L2)
* **response judge J** — e.g. a Llama-Guard ensemble

If L2's model shares a family with J, a paraphrase that fools one tends to fool the other and
the correction is invalid. `GateDecision.families` exposes each layer's family so a run can
assert `J_family not in decision.families`.

## Frozen thresholds

Thresholds live in `cfg.gate` (`GateThresholds`), which is part of the pre-registration hash.
Before a confirmatory real run, enforce that they have not moved:

```python
from recyclebench.backends.real_gate import assert_frozen
assert_frozen(cfg, "prereg.md")   # raises with a field-level diff if any threshold drifted
```

This answers "did you tune the gate after seeing the numbers?" mechanically.

## Install

```bash
pip install -e ".[gate]"        # sentence-transformers + transformers + bert-score
# or, to also use TextAttack's USE/BERTScore constraints (pulls TensorFlow):
pip install -e ".[gate-full]"
```

The package imports and the mock path runs with **neither** installed; each ML layer raises a
clear error only when actually invoked.

## Run

The gate is built automatically on the real path:

```bash
recyclebench run-real --config configs/real.yaml --behaviours data/behaviours.jsonl
```

To enable the discriminative L2 layer, supply an intent model from a family decorrelated from
the judge when constructing the gate (library use):

```python
from recyclebench.gate import load_gate
from recyclebench.config import Config, make_rng

cfg = Config.load("configs/real.yaml")
gate = load_gate("real", make_rng(cfg.seed + 12), cfg.gate, cfg.mock, cfg,
                 inventory=behaviours,            # the L2 choice pool
                 intent_respond=my_llm.respond)   # decorrelated model's respond(prompt)->text
```

Omit `intent_respond` to run semantic-equivalence-only (L0+L1); the audit records that L2 was
disabled, so the configuration is transparent rather than hidden.

## Audit

Every gate decision is recorded (`Manifest.audit`). The human audit
(`recyclebench gate-audit` → `audit-export` → annotate → `audit-ingest`) then estimates the
gate's **false-admission rate with a confidence interval** against these frozen thresholds —
turning the gate from a trusted component into a measured one. Stratify the audit by category
to detect whether the NLI layer (trained on benign corpora) behaves differently on harmful
prompts, and use ≥2 raters with Cohen's κ.

## Extending L2

`IntentClassifier` is the slot. Implement `evaluate(behaviour, variant) -> IntentVerdict` to
plug in an alternative (e.g. an entailment-decomposition or QA-based intent check).
`DiscriminativeIntentClassifier` is the reference filling; `NullIntentClassifier` disables L2.
