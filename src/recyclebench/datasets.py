"""Behaviour sources.

The synthetic generator emits **benign placeholder** behaviours carrying the latent ground
truth the mock needs: an ``attack_strength`` a_b (spread so that a middle band breaks older
but not newer models, giving a non-empty P^-) and a ``para_fragility`` phi_b drawn from a
shallow/deep mixture (the planted patching-vs-generalisation signal RSR recovers).

Real runs load a JailbreakBench/AdvBench-format CSV that YOU supply; no harmful content ships.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Behaviour
from .config import Config

CATEGORIES = ["C1_weapons", "C2_chem", "C3_fraud", "C4_privacy", "C5_misinfo", "C6_cyber"]


def make_synthetic(cfg: Config, rng: np.random.Generator) -> list[Behaviour]:
    """Generate n benign-placeholder behaviours with planted latents.

    ``attack_strength`` ~ N(mu, sd) is centred within the lineage's robustness range so the
    regression band P^- is populated. ``para_fragility`` is a mixture: a *fragile/shallow*
    cluster near 1 (rewording bypasses -> the fix was surface patching) and a *deep* cluster
    near 0 (rewording stays blocked -> intent-level generalisation), with per-category rates
    so RQ2 can locate which harm categories are most brittle.
    """
    rmin = min(cfg.mock.robustness.values())
    rmax = max(cfg.mock.robustness.values())
    mu = 0.5 * (rmin + rmax)
    sd = 0.9 * (rmax - rmin) / 2.0
    n = cfg.n_behaviours
    out = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        a = float(rng.normal(mu, sd))
        # per-category share of *fragile* (shallow-fix) behaviours
        cat_fragile = {"C1_weapons": 0.20, "C2_chem": 0.25, "C3_fraud": 0.45,
                       "C4_privacy": 0.55, "C5_misinfo": 0.40, "C6_cyber": 0.35}[cat]
        if rng.random() < cat_fragile:
            phi = float(np.clip(rng.beta(6.0, 1.5), 0, 1))   # fragile / shallow (near 1)
        else:
            phi = float(np.clip(rng.beta(1.5, 6.0), 0, 1))   # deep / robust (near 0)
        out.append(Behaviour(bid=f"b{i:04d}", canonical_prompt=f"[benign-placeholder {i}]",
                             category=cat, source="synthetic",
                             attack_strength=a, para_fragility=phi))
    return out


def load_jailbreakbench(csv_path: str | Path, goal_col: str = "Goal",
                        category_col: str | None = "Category") -> list[Behaviour]:
    """Load a JailbreakBench/AdvBench-format CSV you provide (column ``Goal``). None ships."""
    import csv
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Provide your own benchmark CSV (column '{goal_col}'). "
            "No harmful content ships with recyclebench.")
    out = []
    with path.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            goal = (row.get(goal_col) or "").strip()
            if not goal:
                continue
            cat = ((row.get(category_col) or "").strip() if category_col else "") or "unknown"
            out.append(Behaviour(bid=f"jbb-{i:04d}", canonical_prompt=goal, category=cat,
                                 source="jailbreakbench"))
    return out
