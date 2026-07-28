"""The meaning-preserving reformulation operator T (exposé Sec. 5, P2).

Produces a fixed budget of candidate variants per behaviour across transform classes, each
with a quantified surface distance (an RQ3 predictor). A planted fraction
(``intent_changed_rate``) drift in intent -- these are the ones the gate must reject; they
exist so the gate audit and the construct-validity story are non-trivial.

The mock text is a benign placeholder token (no harmful content). The real operator lives in
``backends.reformulate`` and is a provider-backed LLM with generic, non-harmful templates.
"""
from __future__ import annotations

import numpy as np

from .types import Behaviour, Variant
from .config import MockParams


class MockReformulator:
    def __init__(self, rng: np.random.Generator, classes: list[str], n_per: int,
                 params: MockParams):
        self.rng = rng
        self.classes = classes
        self.n_per = n_per
        self.p = params

    def variants(self, behaviour: Behaviour) -> list[Variant]:
        out: list[Variant] = []
        for i in range(self.n_per):
            cls = self.classes[i % len(self.classes)]
            base = {"paraphrase": 0.30, "back_translation": 0.55, "lexical": 0.20,
                    "syntactic": 0.45, "register": 0.35, "tense_shift": 0.50}.get(cls, 0.3)
            dist = float(np.clip(self.rng.normal(base, 0.12), 0.0, 1.0))
            preserved = bool(self.rng.random() >= self.p.intent_changed_rate)
            out.append(Variant(bid=behaviour.bid,
                               text=f"{behaviour.bid}::v{i}::{cls}",
                               transform_class=cls, surface_distance=dist, index=i,
                               intent_preserved=preserved))
        return out


def token_edit_ratio(a: str, b: str) -> float:
    """Normalised token-level Levenshtein ratio in [0,1] (used by the real operator)."""
    at, bt = a.split(), b.split()
    n, m = len(at), len(bt)
    if n == 0 and m == 0:
        return 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (at[i - 1] != bt[j - 1]))
            prev = cur
    return dp[m] / max(n, m)


def load_reformulator(backend: str, rng: np.random.Generator, classes, n_per,
                      params: MockParams, cfg=None):
    if backend == "mock":
        return MockReformulator(rng, classes, n_per, params)
    from .backends.reformulate import LLMReformulator
    return LLMReformulator(classes, n_per, cfg)
