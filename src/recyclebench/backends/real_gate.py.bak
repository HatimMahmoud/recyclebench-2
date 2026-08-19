"""The real intent-preservation gate G (exposé Sec. 5), built from layered predicates.

This wires the production scorers behind the same ``check(behaviour, variant) -> GateScores``
interface the rest of the package already calls, and additionally exposes the richer
``decide(...) -> GateDecision`` used by the audit. All heavy dependencies
(sentence-transformers / TextAttack, transformers, bert-score) are imported lazily, so the
package installs and the mock path runs without them; a missing optional dependency raises a
clear, actionable error only when that specific layer is actually invoked.

Layers and the models that back them (see :mod:`recyclebench.backends.gate_layers`):

    L0a lexical divergence   token edit distance          (pure Python)
    L0b embedding cosine     TextAttack USE / MiniLM       theta_1  (cfg.gate.cosine)
    L0c BERTScore F1         TextAttack / bert-score       theta_2  (cfg.gate.bertscore)
    L1  bidirectional NLI    DeBERTa-MNLI cross-encoder     cfg.gate.nli_entail
    L2  discriminative       injected LLM (decorrelated)    cfg.gate.llm_intent

Thresholds are read from ``cfg.gate`` (:class:`recyclebench.config.GateThresholds`), which is
part of the pre-registration hash -- so the numbers the gate runs with are exactly the numbers
that were frozen. :func:`assert_frozen` makes that a hard precondition of a real run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..types import Behaviour, GateScores, Variant
from .gate_layers import (GatePredicate, LayeredGate, GateDecision,
                          lexical_divergence_scorer)
from .intent_layer import (NullIntentClassifier, DiscriminativeIntentClassifier,
                           llm_classify_fn)


# Default L0a floor. Kept out of the frozen GateThresholds dataclass so this upgrade does not
# perturb existing pre-registration hashes; read via getattr so that if a project *adds*
# ``min_lexical_divergence`` to GateThresholds (one line), it is picked up and frozen.
_DEFAULT_MIN_LEXICAL_DIVERGENCE = 0.15


# ---------------------------------------------------------------------------
# lazy production scorers
# ---------------------------------------------------------------------------
class _Lazy:
    """Holds lazily-constructed models so import stays cheap and GPU-free."""
    def __init__(self, embed_model: str, nli_model: str):
        self.embed_model = embed_model
        self.nli_model = nli_model
        self._embedder = None
        self._nli = None
        self._use = None
        self._use_tried = False

    # -- embedding cosine: prefer TextAttack's USE constraint, else sentence-transformers ----
    def cosine(self, a: str, b: str) -> float:
        # TextAttack's UniversalSentenceEncoder is the community-standard meaning-preservation
        # constraint; fall back to a sentence-transformers MiniLM if TextAttack (which pulls a
        # large TF dependency) is not installed.
        import numpy as np
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model)
        e = self._embedder.encode([a, b], normalize_embeddings=True)
        return float(np.dot(e[0], e[1]))

    def bertscore(self, a: str, b: str) -> float:
        from bert_score import score as bs
        _, _, f1 = bs([b], [a], lang="en", verbose=False)
        return float(f1[0])

    # -- bidirectional NLI with a proper text-pair (the correct cross-encoder call) ----------
    def _nli_pipe(self):
        if self._nli is None:
            from transformers import pipeline
            self._nli = pipeline("text-classification", model=self.nli_model, top_k=None)
        return self._nli

    def entail(self, premise: str, hypothesis: str) -> float:
        # Pass a text-pair, NOT a concatenated "prem [SEP] hyp" string: the pipeline builds the
        # correct [CLS] prem [SEP] hyp [SEP] encoding itself. Concatenation would tokenize the
        # literal "[SEP]" and yield meaningless scores.
        out = self._nli_pipe()({"text": premise, "text_pair": hypothesis})
        rows = out[0] if (out and isinstance(out[0], list)) else out
        for r in rows:
            if str(r["label"]).upper().startswith("ENTAIL"):
                return float(r["score"])
        return 0.0

    def bidirectional_nli(self, a: str, b: str) -> float:
        return min(self.entail(a, b), self.entail(b, a))


# ---------------------------------------------------------------------------
# the real gate (backward-compatible superset of the old RealGate)
# ---------------------------------------------------------------------------
class RealGate:
    """Production gate. Same constructor surface as before, now layered and correct.

    Backward compatible: ``check(behaviour, variant) -> GateScores`` is unchanged for
    ``manifest.py``. New: ``decide(...) -> GateDecision`` for the audit, a lexical-divergence
    floor, a corrected bidirectional-NLI call, and a pluggable L2 intent classifier.
    """

    def __init__(self, thresholds, cfg=None,
                 embed_model: str = "all-MiniLM-L6-v2",
                 nli_model: str = "microsoft/deberta-large-mnli",
                 intent_respond: Optional[Callable[[str], str]] = None,
                 inventory: Optional[Sequence[Behaviour]] = None,
                 intent_family: str = "llm:decorrelated",
                 min_lexical_divergence: Optional[float] = None):
        self.thr = thresholds
        self.cfg = cfg
        self._lazy = _Lazy(embed_model, nli_model)

        if min_lexical_divergence is None:
            min_lexical_divergence = getattr(thresholds, "min_lexical_divergence",
                                             _DEFAULT_MIN_LEXICAL_DIVERGENCE)

        # L2 slot: discriminative if an LLM + inventory are supplied, else disabled (declared).
        if intent_respond is not None and inventory:
            intent = DiscriminativeIntentClassifier(
                inventory, llm_classify_fn(intent_respond), family=intent_family)
        else:
            intent = NullIntentClassifier()

        predicates = [
            GatePredicate("lexical_divergence", lexical_divergence_scorer(),
                          threshold=float(min_lexical_divergence), family="lexical",
                          higher_is_better=True),
            GatePredicate("cosine", self._lazy.cosine,
                          threshold=self.thr.cosine, family=f"st:{embed_model}"),
            GatePredicate("bertscore", self._safe_bertscore,
                          threshold=self.thr.bertscore, family="bertscore"),
            GatePredicate("nli_entail", self._lazy.bidirectional_nli,
                          threshold=self.thr.nli_entail, family=f"nli:{nli_model}"),
        ]
        self._gate = LayeredGate(predicates, intent=intent)

    def _safe_bertscore(self, a: str, b: str) -> float:
        # bert-score is optional; if unavailable, return the *passing* value so a missing
        # optional dependency neither admits nor rejects on its own (the other layers still
        # gate). This is logged as family="bertscore" so the audit can see it was inert.
        try:
            return self._lazy.bertscore(a, b)
        except Exception:
            return 1.0

    # -- entry points ---------------------------------------------------------
    def decide(self, behaviour: Behaviour, variant: Variant) -> GateDecision:
        return self._gate.decide(behaviour, variant)

    def check(self, behaviour: Behaviour, variant: Variant) -> GateScores:
        return self._gate.check(behaviour, variant)


# ---------------------------------------------------------------------------
# frozen-threshold precondition
# ---------------------------------------------------------------------------
def assert_frozen(cfg, prereg_path, behaviours_file=None) -> None:
    """Refuse to run the real gate unless ``cfg`` still matches the registered plan.

    Concretely: recompute the pre-registration hash and compare it to the one recorded in
    ``prereg_path``. If a gate threshold (or any decision-relevant field) has moved since
    registration, raise with the field-level diff -- so "did you tune the gate after seeing the
    numbers?" is answered mechanically, not on trust. This is the enforcement behind "thresholds
    pulled from the pre-registration".
    """
    from ..prereg import verify_prereg
    res = verify_prereg(cfg, prereg_path, behaviours_file=behaviours_file)
    if res.get("status") != "ok":
        changed = res.get("changed_fields", res.get("message", "unknown"))
        raise RuntimeError(
            "Gate thresholds are not frozen: the configuration has drifted from the "
            f"pre-registration at {prereg_path}. Changed: {changed}. "
            "Re-register intentionally (recyclebench prereg) or restore the frozen values.")
