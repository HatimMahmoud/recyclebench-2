"""Layered intent-preservation gate G(p, p') -- the construct-validity core (exposé Sec. 5).

Why this exists
---------------
A measured "recycling" bypass is only trustworthy if the paraphrase p' still asks for the
*same harmful thing* as the canonical prompt p. The gate is what certifies that. A gate that
merely trusts an embedding cosine will admit softened or altered requests -- the exact failure
the adversarial-NLP literature documents for USE/BERTScore -- so the gate is built as a
*conjunction of layers*, each a necessary condition, and its own error is measured by the
human audit rather than assumed.

Design
------
Every layer is a :class:`GatePredicate` that maps (p, p') to a score in ``[0, 1]`` and admits
iff ``score >= threshold``. The gate admits iff **all enabled layers** admit (short-circuit on
first rejection, so the "reason" a variant was rejected is well defined and auditable).

The scoring functions are *injected*. In production the default factory wires real models
(TextAttack constraints, a DeBERTa NLI cross-encoder, a decorrelated LLM intent classifier),
lazily, so the package installs and the mock path runs without any ML dependency. In tests the
same predicates are driven by deterministic stubs -- so the entire control flow, thresholding,
short-circuit, audit record, and frozen-threshold check are exercised on every machine with
nothing heavier than numpy.

Layers
------
* **L0a lexical divergence** (pure Python, always available): p' must be *sufficiently
  different* from p, or "recycling" is trivial. This is a floor on normalized token edit
  distance; it has no ML dependency and therefore always runs.
* **L0b embedding cosine** (TextAttack USE / sentence-transformers): a blunt similarity floor.
  Necessary, never sufficient.
* **L0c BERTScore** (TextAttack / bert-score): token-level similarity floor.
* **L1  bidirectional NLI entailment** (DeBERTa-MNLI cross-encoder): p |= p' and p' |= p.
  Decorrelated from the Llama lineage by construction (a separate encoder family).
* **L2  discriminative intent** (pluggable): does p' still *uniquely identify* the same
  behaviour against distractor behaviours? This is the layer no released tool ships; it is a
  clean slot (:class:`IntentClassifier`) with a null default and a multiple-choice
  implementation in :mod:`recyclebench.backends.intent_layer`.

Independence
------------
The gate records which model family each layer uses (:attr:`GateDecision.families`) so a run
can *assert* that no gate layer shares a model with the response judge J -- correlated errors
between the gate and J would violate the independence the bundle-level correction assumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..types import Behaviour, GateScores, Variant


# ---------------------------------------------------------------------------
# per-layer records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LayerResult:
    """One layer's score, threshold, and admit bit -- an audit primitive."""
    name: str
    score: float
    threshold: float
    admitted: bool
    family: str = "n/a"          # model family, for the decorrelation assertion
    note: str = ""


@dataclass(frozen=True)
class GateDecision:
    """The full, auditable outcome of the gate for one (behaviour, variant).

    ``admitted`` is the conjunction over layers. ``rejected_by`` names the first layer that
    failed (empty when admitted). ``layers`` holds every layer that was *evaluated*
    (short-circuit means later layers may be absent, which is itself informative for the
    audit: a rejection at L0a costs no ML compute).
    """
    admitted: bool
    rejected_by: str
    layers: tuple[LayerResult, ...]

    @property
    def families(self) -> set[str]:
        return {lr.family for lr in self.layers if lr.family != "n/a"}

    def score(self, name: str, default: float = float("nan")) -> float:
        for lr in self.layers:
            if lr.name == name:
                return lr.score
        return default

    def to_gate_scores(self) -> GateScores:
        """Project onto the legacy 4-field record the rest of the package consumes.

        The layered gate has more than four components; the legacy consumers
        (``manifest.py``) expect exactly ``cosine, bertscore, nli_entail, llm_intent``. We map
        the corresponding layers and default any layer that short-circuited (was never
        evaluated) to its passing value, so a legacy audit row never mistakes "not evaluated"
        for "scored low". The authoritative record is :class:`GateDecision`; this projection
        exists only for backward compatibility.
        """
        return GateScores(
            cosine=self.score("cosine", 1.0),
            bertscore=self.score("bertscore", 1.0),
            nli_entail=self.score("nli_entail", 1.0),
            llm_intent=self.score("intent", 1.0),
            admitted=self.admitted,
        )


# ---------------------------------------------------------------------------
# a layer
# ---------------------------------------------------------------------------
# A scorer maps (p, p') -> score in [0, 1]. Higher = more intent-preserving.
Scorer = Callable[[str, str], float]


@dataclass
class GatePredicate:
    """One gate layer: a name, an injected scorer, a threshold, and a model-family tag.

    ``higher_is_better`` is True for similarity/entailment layers. ``enabled=False`` removes
    the layer from the conjunction entirely (and it is then never scored).
    """
    name: str
    scorer: Scorer
    threshold: float
    family: str = "n/a"
    enabled: bool = True
    higher_is_better: bool = True

    def evaluate(self, p: str, pp: str) -> LayerResult:
        s = float(self.scorer(p, pp))
        admit = (s >= self.threshold) if self.higher_is_better else (s <= self.threshold)
        return LayerResult(self.name, s, self.threshold, bool(admit), self.family)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
class LayeredGate:
    """Conjunction of gate layers with short-circuit evaluation and a full audit record.

    Parameters
    ----------
    predicates:
        Ordered layers. Order matters only for *which layer is named as the rejecter* and for
        compute economy (cheap pure-Python layers first). The admit decision is
        order-invariant (a conjunction).
    intent:
        The pluggable L2 classifier (see :class:`recyclebench.backends.intent_layer`
        .IntentClassifier). ``None`` disables L2 (semantic-equivalence-only gating), which is a
        valid, weaker configuration that must be declared in the pre-registration.
    """

    def __init__(self, predicates: Sequence[GatePredicate], intent: "IntentClassifier | None" = None):
        self.predicates = list(predicates)
        self.intent = intent

    # -- the two entry points --------------------------------------------------
    def decide(self, behaviour: Behaviour, variant: Variant) -> GateDecision:
        p, pp = behaviour.canonical_prompt, variant.text
        results: list[LayerResult] = []
        for pred in self.predicates:
            if not pred.enabled:
                continue
            lr = pred.evaluate(p, pp)
            results.append(lr)
            if not lr.admitted:
                return GateDecision(False, lr.name, tuple(results))

        # L2 last: it is the most expensive and the most discriminating.
        if self.intent is not None:
            iv = self.intent.evaluate(behaviour, variant)
            lr = LayerResult("intent", iv.score, iv.threshold, iv.admitted,
                             family=iv.family, note=iv.note)
            results.append(lr)
            if not lr.admitted:
                return GateDecision(False, "intent", tuple(results))

        return GateDecision(True, "", tuple(results))

    def check(self, behaviour: Behaviour, variant: Variant) -> GateScores:
        """Legacy entry point used by ``manifest.py`` -- returns the 4-field record."""
        return self.decide(behaviour, variant).to_gate_scores()


# ---------------------------------------------------------------------------
# L0a: lexical divergence -- pure Python, no dependency, always available
# ---------------------------------------------------------------------------
def _norm_tokens(s: str) -> list[str]:
    """Lowercase word tokens with surrounding punctuation stripped, so that trailing commas,
    periods, and case differences do not register as divergence (only genuine rewording does).
    Empty tokens (produced by stripping a standalone punctuation mark) are dropped."""
    import string
    toks = (t.strip(string.punctuation) for t in s.lower().split())
    return [t for t in toks if t]


def token_edit_divergence(p: str, pp: str) -> float:
    """1 - (token-level Levenshtein similarity). 0 = identical tokens, 1 = fully disjoint.

    A word-level edit distance (not character-level) so that genuine rewording registers as
    divergence while punctuation/case noise does not. Pure Python; used as the L0a floor to
    forbid trivial near-copies from counting as "recycling".
    """
    a, b = _norm_tokens(p), _norm_tokens(pp)
    if not a and not b:
        return 0.0
    # classic DP edit distance over tokens
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[n]
    return dist / max(m, n, 1)


def lexical_divergence_scorer() -> Scorer:
    """L0a scorer: higher = more divergent. Paired with ``higher_is_better=True`` and a
    *minimum* divergence threshold, it forbids near-identical paraphrases."""
    return token_edit_divergence


# ---------------------------------------------------------------------------
# forward reference: IntentClassifier lives in intent_layer.py to keep this module
# import-light. We only need its shape here.
# ---------------------------------------------------------------------------
try:                                   # pragma: no cover - import convenience
    from .intent_layer import IntentClassifier, IntentVerdict  # noqa: F401
except Exception:                      # pragma: no cover
    IntentClassifier = object          # type: ignore
    IntentVerdict = object             # type: ignore
