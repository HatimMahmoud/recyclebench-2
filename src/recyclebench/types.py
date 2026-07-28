"""Core data types and protocols.

Notation follows the exposé (Sec. 3):
  * M(p)              -- a model's response to prompt p
  * J(p, r) in {0,1}  -- calibrated judge: does response r fulfil the harmful behaviour of p
  * S_M(p) = J(p,M(p))-- whether model M is broken by prompt p
  * C_M               -- behaviours M refuses canonically  (the always-large PBR population)
  * R(p)              -- gate-admitted, intent-preserving paraphrases of p (the frozen manifest)
  * P^- = {p : S_{M_n}(p)=1 and S_{M_{n+1}}(p)=0}  -- the regression set (RSR population)

The mock backends carry latent fields (attack_strength, fix_depth, intent_preserved,
latent_harmful) that a real backend never sets or reads; they encode the planted ground
truth used to validate the estimators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class Behaviour:
    """A harmful behaviour with a canonical seed prompt p_b (reused from a benchmark)."""
    bid: str
    canonical_prompt: str
    category: str = "unknown"
    source: str = "synthetic"
    # --- latent ground truth for the mock only (real backends ignore these) ---
    attack_strength: float = 0.0   # a_b : how effective the canonical jailbreak is
    para_fragility: float = 0.0    # phi_b : how paraphrase-fragile the alignment for b is
    #                                (1 = shallow/fragile -> paraphrases bypass;
    #                                 0 = deep/robust     -> paraphrases stay blocked)


@dataclass(frozen=True)
class Variant:
    """A single meaning-preserving reformulation p' = tau(p) of a behaviour's prompt."""
    bid: str
    text: str
    transform_class: str
    surface_distance: float        # token-edit / embedding shift, logged as an RQ3 predictor
    index: int = 0
    intent_preserved: bool = True  # latent truth for the gate audit (mock only)

    def to_dict(self) -> dict:
        return {"bid": self.bid, "text": self.text, "transform_class": self.transform_class,
                "surface_distance": self.surface_distance, "index": self.index,
                "intent_preserved": self.intent_preserved}

    @staticmethod
    def from_dict(d: dict) -> "Variant":
        return Variant(bid=d["bid"], text=d["text"], transform_class=d["transform_class"],
                       surface_distance=float(d["surface_distance"]), index=int(d.get("index", 0)),
                       intent_preserved=bool(d.get("intent_preserved", True)))


@dataclass
class Response:
    """A model response. ``latent_harmful`` is the mock's ground-truth compliance bit."""
    text: str = ""
    latent_harmful: Optional[int] = None


@dataclass(frozen=True)
class GateScores:
    """Component scores and the admission decision of the intent-preservation gate."""
    cosine: float
    bertscore: float
    nli_entail: float
    llm_intent: float
    admitted: bool


@runtime_checkable
class Model(Protocol):
    name: str
    def respond(self, prompt: str, *, behaviour: Optional[Behaviour] = None,
                variant: Optional[Variant] = None, canonical: bool = False,
                seed: Optional[int] = None) -> Response: ...


@runtime_checkable
class Judge(Protocol):
    name: str
    def classify(self, prompt: str, response: Response) -> int: ...


@runtime_checkable
class Reformulator(Protocol):
    def variants(self, behaviour: Behaviour) -> list[Variant]: ...


@runtime_checkable
class Gate(Protocol):
    def check(self, behaviour: Behaviour, variant: Variant) -> GateScores: ...
