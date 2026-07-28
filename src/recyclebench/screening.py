"""Screening and per-model evaluation over the frozen manifest.

For each model we compute, over ALL behaviours:
  * canonical status S_M(p_b): observed (judge) and latent (mock truth), via k-sampling;
  * recovery status: for each admitted variant in R(p_b), query the model and judge; a
    behaviour is *recovered/bypassed* if ANY admitted variant succeeds (best-of-many union).

These per-model results are then sliced by C_M (for PBR) or by P^- (for RSR). Calibration
pairs are collected at both the response level (per variant) and the behaviour-bundle level
(per behaviour), so the bundle-level Rogan-Gladen correction can be applied and validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .types import Behaviour
from .manifest import Manifest


@dataclass
class Recovery:
    bid: str
    category: str
    n_admitted: int
    obs_bypassed: int          # any admitted variant judged success
    latent_bypassed: Optional[int]  # mock truth; None on real backends (no latent truth)
    attempts_to_first: int     # 1-based admitted index of first observed success, else -1
    unresolved: bool           # no variant admitted -> NOT counted as robust (reported apart)


@dataclass
class ModelRun:
    name: str
    canonical_obs: dict[str, int] = field(default_factory=dict)     # S_M observed
    canonical_latent: dict[str, int] = field(default_factory=dict)  # S_M truth
    recovery: dict[str, Recovery] = field(default_factory=dict)
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)       # per-variant, for RQ3
    resp_pairs: list = field(default_factory=list)                   # (latent, judge) per variant
    bundle_pairs: list = field(default_factory=list)                 # (latent, judge) per behaviour
    responses: dict = field(default_factory=dict)                    # (bid, idx) -> text
    judge_labels: dict = field(default_factory=dict)                 # (bid, idx) -> 0/1

    def C_M(self) -> list[str]:
        return [b for b, s in self.canonical_obs.items() if s == 0]

    def broken_set(self) -> list[str]:
        return [b for b, s in self.canonical_obs.items() if s == 1]


def _draw_seed(bid: str, index: int, draw: int) -> int:
    """Deterministic per-draw seed: reproducible across runs, distinct within a bundle."""
    import hashlib
    h = hashlib.sha256(f"{bid}|{index}|{draw}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _screen_canonical(behaviour: Behaviour, model, judge, k: int):
    """Return (observed_broken, latent_broken): broken if ANY of k canonical samples complies."""
    obs = 0
    lat = None
    for i in range(k):
        # a distinct seed per draw: at temperature 0 a real endpoint would otherwise return
        # k identical completions, making k-sampling pointless
        r = model.respond(behaviour.canonical_prompt, behaviour=behaviour, canonical=True,
                          seed=_draw_seed(behaviour.bid, -1, i))
        obs = max(obs, judge.classify(behaviour.canonical_prompt, r))
        if r.latent_harmful is not None:            # real backends have no latent truth
            lat = max(lat or 0, int(r.latent_harmful))
    return obs, lat


def run_model(model, behaviours, manifest: Manifest, judge, cfg,
              collect_calib: bool = True, store_responses: bool = False) -> ModelRun:
    run = ModelRun(name=model.name)
    rows = []
    spv = cfg.budget.samples_per_variant
    for b in behaviours:
        obs, lat = _screen_canonical(b, model, judge, cfg.budget.k_canonical_samples)
        run.canonical_obs[b.bid] = obs
        run.canonical_latent[b.bid] = lat

        variants = manifest.R(b.bid)
        n_adm = len(variants)
        obs_by = 0
        lat_by = None
        first = -1
        for j, v in enumerate(variants, start=1):
            v_obs = 0
            v_lat = None
            for si in range(spv):
                resp = model.respond(v.text, behaviour=b, variant=v, canonical=False,
                                     seed=_draw_seed(b.bid, v.index, si))
                lab = judge.classify(v.text, resp)
                v_obs = max(v_obs, lab)
                if resp.latent_harmful is not None:
                    v_lat = max(v_lat or 0, int(resp.latent_harmful))
                    if collect_calib:
                        run.resp_pairs.append((int(resp.latent_harmful), lab))
                if store_responses:
                    run.responses[(b.bid, v.index)] = resp.text
            run.judge_labels[(b.bid, v.index)] = int(v_obs)
            rows.append({"bid": b.bid, "category": b.category, "model": model.name,
                         "transform_class": v.transform_class,
                         "surface_distance": v.surface_distance,
                         "success": int(v_obs),
                         "success_latent": (int(v_lat) if v_lat is not None else None)})
            if v_obs and first < 0:
                first = j
            obs_by = max(obs_by, v_obs)
            if v_lat is not None:
                lat_by = max(lat_by or 0, v_lat)

        run.recovery[b.bid] = Recovery(
            bid=b.bid, category=b.category, n_admitted=n_adm,
            obs_bypassed=int(obs_by),
            latent_bypassed=(int(lat_by) if lat_by is not None else None),
            attempts_to_first=first, unresolved=(n_adm == 0))
        if collect_calib and n_adm > 0 and lat_by is not None:
            run.bundle_pairs.append((int(lat_by), int(obs_by)))

    run.trials = pd.DataFrame(rows)
    return run


def regression_set(old: ModelRun, new: ModelRun) -> list[str]:
    """P^- = { p : S_old(p)=1 (breaks old) and S_new(p)=0 (blocked by new) }, on observed S."""
    bids = set(old.canonical_obs) & set(new.canonical_obs)
    return sorted(b for b in bids
                  if old.canonical_obs[b] == 1 and new.canonical_obs[b] == 0)
