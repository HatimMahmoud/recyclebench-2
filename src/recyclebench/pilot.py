"""The Month-1 pilot and the pre-registered decision rule (exposé Sec. 5-7).

The pilot sizes the binding power constraint |P^-| for each adjacent lineage pair (and
pooled). The pre-registered rule then selects the headline:

  * if any pair (or the pooled set) has |P^-| >= min_p_minus  -> RSR is the headline;
  * otherwise                                                  -> the PBR backbone is reported.

Empty P^- yields an undefined RSR; a small-but-nonempty P^- yields an exploratory RSR while
the always-large PBR carries the empirical weight.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .screening import ModelRun, regression_set


@dataclass
class PilotResult:
    pair_sizes: dict[str, int] = field(default_factory=dict)   # "old -> new" : |P^-|
    pooled: int = 0
    min_p_minus: int = 30
    headline: str = "PBR"                                       # "RSR" or "PBR"
    estimable_pairs: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"pair_sizes": self.pair_sizes, "pooled_p_minus": self.pooled,
                "min_p_minus": self.min_p_minus, "headline": self.headline,
                "estimable_pairs": self.estimable_pairs,
                "decision": (
                    "RSR is the primary headline" if self.headline == "RSR"
                    else "P^- under-powered -> PBR backbone is the reported headline")}


def run_pilot(runs: list[ModelRun], min_p_minus: int) -> PilotResult:
    res = PilotResult(min_p_minus=min_p_minus)
    pooled_ids: set[str] = set()
    for old, new in zip(runs, runs[1:]):
        pm = regression_set(old, new)
        key = f"{old.name} -> {new.name}"
        res.pair_sizes[key] = len(pm)
        pooled_ids.update(f"{key}::{b}" for b in pm)   # pooled across pairs (disjoint keys)
        if len(pm) >= min_p_minus:
            res.estimable_pairs.append(key)
    res.pooled = len(pooled_ids)
    res.headline = "RSR" if (res.estimable_pairs or res.pooled >= min_p_minus) else "PBR"
    return res
