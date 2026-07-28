"""Intent-preservation gate G(p, p') in {0,1} (exposé Sec. 5, "construct control").

The real gate requires all of: cosine >= theta1, BERTScore >= theta2, bidirectional NLI
entailment, and an LLM intent-equivalence check. The mock gate models the *decision* and,
crucially, its error against the latent ``intent_preserved`` truth so the human audit
(false-admission / false-rejection rates) is exercised:

  * an intent-preserving variant is admitted w.p. ``gate_true_admit``;
  * an intent-changed variant is (wrongly) admitted w.p. ``gate_false_admit``.

Since a false admission invalidates a measured bypass while a false rejection only costs
power, the audit is oriented to bound the false-admission rate.
"""
from __future__ import annotations

import numpy as np

from .types import Behaviour, GateScores, Variant
from .config import GateThresholds, MockParams


class MockGate:
    def __init__(self, rng: np.random.Generator, thr: GateThresholds, params: MockParams):
        self.rng = rng
        self.thr = thr
        self.p = params

    # Score distributions are conditioned on the latent truth and centred so that each
    # component clears its threshold with probability q_preserved / q_changed; the gate then
    # admits iff ALL components clear. Gate error is therefore *emergent from score overlap*,
    # exactly as in the real gate, rather than a free parameter -- which is what makes the
    # threshold sensitivity analysis meaningful on the mock path too.
    _SIGMA = {"cosine": 0.06, "bertscore": 0.05, "nli_entail": 0.12, "llm_intent": 0.15}

    def _score(self, field: str, thr: float, preserved: bool) -> float:
        sd = self._SIGMA[field]
        # z such that P(X >= thr) equals the per-component clearance probability
        z = self.p.z_preserved if preserved else self.p.z_changed
        return float(np.clip(self.rng.normal(thr + z * sd, sd), 0.0, 1.0))

    def check(self, behaviour: Behaviour, variant: Variant) -> GateScores:
        preserved = variant.intent_preserved
        t = self.thr
        cos = self._score("cosine", t.cosine, preserved)
        bert = self._score("bertscore", t.bertscore, preserved)
        nli = self._score("nli_entail", t.nli_entail, preserved)
        llm = self._score("llm_intent", t.llm_intent, preserved)
        admitted = bool(cos >= t.cosine and bert >= t.bertscore
                        and nli >= t.nli_entail and llm >= t.llm_intent)
        return GateScores(cosine=cos, bertscore=bert, nli_entail=nli, llm_intent=llm,
                          admitted=admitted)


def load_gate(backend: str, rng: np.random.Generator, thr: GateThresholds,
              params: MockParams, cfg=None, *, inventory=None, intent_respond=None):
    """Return the intent-preservation gate for ``backend``.

    ``mock``  -> :class:`MockGate` (models the decision and its error against latent truth).
    ``real``  -> :class:`recyclebench.backends.real_gate.RealGate`, the layered production gate.
                 ``inventory`` (the behaviour pool) and ``intent_respond`` (an LLM ``respond``
                 from a family decorrelated from the response judge) enable the discriminative
                 L2 intent layer; omit them to run semantic-equivalence-only.
    """
    if backend == "mock":
        return MockGate(rng, thr, params)
    from .backends.real_gate import RealGate
    return RealGate(thr, cfg, inventory=inventory, intent_respond=intent_respond)
