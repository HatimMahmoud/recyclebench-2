"""Target models.

The mock model is the crux the old design got wrong: **older models must actually break
canonical jailbreaks**, or the regression set P^- is empty by construction. Here canonical
compliance decreases with generation (robustness r_g), so:

  * S_{M_n}(p_b) = 1 for behaviours whose attack strength exceeds the older model's r_g;
  * S_{M_{n+1}}(p_b) = 0 for those below the newer model's r_{g+1} (> r_g);

giving a non-empty P^- band r_g < a_b < r_{g+1}.

**One unified bypass mechanism.** A paraphrase p' of behaviour b bypasses model M with

    logit = canonical_scale*(a_b - r_M)          # how close b already is to breaking M
          + fragility_gain*phi_b                 # shallow/fragile alignment for b
          + class_effect[tau] + dist_effect*d    # which transform, how far it moved
          + variant_offset                       # global difficulty of the paraphrase route

RSR and PBR are then two *slices* of the same mechanism -- RSR restricts it to P^- (where
the newer model blocks the canonical prompt), PBR restricts it to C_M. phi_b is the planted
patching-vs-generalisation signal: high phi_b behaviours are "patched by string" and recover
under paraphrase; low phi_b behaviours are "generalised" and stay blocked.

Everything is latent ground truth; ``respond`` returns ``latent_harmful`` and the judge adds
observation error on top. Real backends ignore the latent fields.
"""
from __future__ import annotations

import numpy as np

from .types import Behaviour, Response, Variant
from .config import Config, MockParams


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class MockModel:
    def __init__(self, name: str, generation: int, rng: np.random.Generator,
                 params: MockParams):
        self.name = name
        self.generation = generation           # 0 = oldest
        self.rng = rng
        self.p = params
        self.r = params.robustness.get(name, 0.4 + 0.6 * generation)

    # -- canonical compliance: does this model break the seed prompt? --
    def _canonical_logit(self, b: Behaviour) -> float:
        return self.p.canonical_scale * (b.attack_strength - self.r)

    # -- paraphrase compliance: does a gated variant bypass this model? --
    def _variant_logit(self, b: Behaviour, v: Variant) -> float:
        """One mechanism serves both metrics. A variant bypasses when the behaviour's
        alignment is paraphrase-*fragile* (phi_b -> 1): the surface change buys a fragility
        bonus on top of the canonical logit. Deep alignment (phi_b -> 0) buys nothing, so
        paraphrases of a canonically-blocked prompt stay blocked.

        Restricted to P^- this is exactly RSR (did the newer model's fix survive rewording);
        restricted to C_M it is PBR (within-model paraphrase robustness)."""
        cls = self.p.class_effect.get(v.transform_class, 0.0)
        dist = self.p.dist_effect * v.surface_distance
        fragility = self.p.fragility_gain * b.para_fragility
        return self._canonical_logit(b) + fragility + cls + dist + self.p.variant_offset

    def respond(self, prompt: str, *, behaviour=None, variant=None,
                canonical: bool = False, seed=None) -> Response:
        if behaviour is None:
            # no latent context (e.g. a real prompt) -> mock abstains from a harmful bit
            return Response(text="[mock:no-latent]", latent_harmful=0)
        if canonical or variant is None:
            logit = self._canonical_logit(behaviour)
        else:
            logit = self._variant_logit(behaviour, variant)
        harmful = int(self.rng.random() < _sigmoid(logit))
        return Response(text=("[comply]" if harmful else "[refuse]"), latent_harmful=harmful)


def load_lineage(cfg: Config, rng: np.random.Generator) -> list[MockModel]:
    """Instantiate the mock lineage in order (oldest first). Each model gets its own rng
    stream so canonical and recycled draws are independent but reproducible."""
    models = []
    for g, name in enumerate(cfg.lineage):
        models.append(MockModel(name, g, np.random.default_rng(cfg.seed + 1000 * (g + 1)),
                                cfg.mock))
    return models


def load_model(name: str, backend: str, cfg: Config, rng: np.random.Generator):
    if backend == "mock":
        g = cfg.lineage.index(name) if name in cfg.lineage else 0
        return MockModel(name, g, np.random.default_rng(cfg.seed + 7919 * (g + 1)), cfg.mock)
    if backend in ("openai", "together", "anthropic"):
        from .backends.providers import load_provider_model
        return load_provider_model(name, backend, cfg)
    raise ValueError(f"unknown backend: {backend}")
