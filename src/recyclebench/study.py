"""End-to-end study orchestration (mock backend).

Order of operations mirrors the exposé pipeline (Fig. 2 / Sec. 5):
  1. draw behaviours;
  2. build the FROZEN reformulation manifest once (reformulate -> gate -> admit);
  3. run every model in the lineage against the same manifest;
  4. calibrate the judge on the human-audited (bundle) subset;
  5. pilot -> pre-registered decision (RSR headline vs PBR floor);
  6. RSR per adjacent pair (primary); PBR per model (backbone); gate audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config, make_rng
from .datasets import make_synthetic
from .reformulate import load_reformulator
from .gate import load_gate
from .judge import MockJudge
from .manifest import Manifest
from .models import load_lineage
from .screening import run_model, regression_set
from .calibration import calibrate
from .pilot import run_pilot
from .recycling import compute_rsr
from .pbr import compute_pbr
from .drivers import fit_drivers
from .defences import simulate_defences

import numpy as np
import pandas as pd


@dataclass
class StudyResult:
    cfg: Config
    manifest: Manifest
    runs: list = field(default_factory=list)
    pilot: object = None
    rsr: list = field(default_factory=list)      # RSRResult per adjacent pair
    pbr: list = field(default_factory=list)      # PBRResult per model
    calib_bundle: object = None
    calib_response: object = None
    gate_audit: dict = field(default_factory=dict)
    drivers: object = None
    defences: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "SYNTHETIC": ("mock backend: every figure here is simulated with planted "
                          "ground truth and is NOT a result about any real model"),
            "backend": "mock",
            "lineage": self.cfg.lineage,
            "n_behaviours": self.cfg.n_behaviours,
            "gate_audit": self.gate_audit,
            "judge_calibration": {
                "bundle_level": self.calib_bundle.summary() if self.calib_bundle else None,
                "response_level": self.calib_response.summary() if self.calib_response else None,
            },
            "pilot": self.pilot.summary() if self.pilot else None,
            "RSR_primary": [r.summary() for r in self.rsr],
            "PBR_backbone": [p.summary() for p in self.pbr],
            "RQ3_drivers": self.drivers.summary() if self.drivers else None,
            "RQ5_defences": [d.summary() for d in self.defences],
        }


def run_study(cfg: Config, backend: str = "mock") -> StudyResult:
    rng = make_rng(cfg.seed)

    # 1. behaviours
    behaviours = make_synthetic(cfg, rng)

    # 2. FROZEN manifest (built once, reused across all models)
    reform = load_reformulator("mock", make_rng(cfg.seed + 11), cfg.transform_classes,
                               cfg.budget.paraphrases_per_behaviour, cfg.mock)
    gate = load_gate("mock", make_rng(cfg.seed + 12), cfg.gate, cfg.mock)
    manifest = Manifest.build(behaviours, reform, gate)

    # 3. per-model runs against the same manifest
    models = load_lineage(cfg, rng)
    # Each model gets its OWN judge stream so results are independent of evaluation order
    # and reproducible across invocations (the judge is stateful via its RNG).
    runs = []
    for i, m in enumerate(models):
        judge_i = MockJudge(make_rng(cfg.seed + 13 + 101 * i),
                            se=cfg.mock.judge_se, sp=cfg.mock.judge_sp)
        runs.append(run_model(m, behaviours, manifest, judge_i, cfg))

    # 4. judge calibration on the human-audited subset (pool bundle/response pairs)
    resp_pairs = [p for r in runs for p in r.resp_pairs]
    bundle_pairs = [p for r in runs for p in r.bundle_pairs]
    calib_response = calibrate(resp_pairs, cfg.decision.kappa_floor, response_level=True)
    calib_bundle = calibrate(bundle_pairs, cfg.decision.kappa_floor, response_level=False)
    jc = calib_bundle.chars

    # 5. pilot + decision
    pilot = run_pilot(runs, cfg.decision.min_p_minus)

    # 6. RSR per adjacent pair (primary)
    rsr = []
    p_minus_sets = {}
    for old, new in zip(runs, runs[1:]):
        pm = regression_set(old, new)
        p_minus_sets[(old.name, new.name)] = pm
        rsr.append(compute_rsr(old, new, pm, jc, cfg.decision.min_p_minus))

    # backbone PBR per model
    pbr = [compute_pbr(r, jc, cfg.decision.pi_star) for r in runs]

    # RQ3 drivers: pooled across ALL lineage pairs (exposé: "pooled across pairs").
    # Variant-level data is used ONLY here; headline denominators stay behaviour counts.
    drivers = None
    pooled_frames = []
    for (o_name, n_name), pm in p_minus_sets.items():
        if not pm:
            continue
        nr = next(r for r in runs if r.name == n_name)
        sub = nr.trials[nr.trials["bid"].isin(set(pm))].copy()
        if sub.empty:
            continue
        sub["pair"] = f"{o_name}->{n_name}"
        # namespace behaviour ids per pair so the random effect is per (pair, behaviour)
        sub["bid"] = sub["pair"] + "::" + sub["bid"]
        pooled_frames.append(sub)
    if pooled_frames:
        pooled = pd.concat(pooled_frames, ignore_index=True)
        drivers = fit_drivers(pooled, list(pooled["bid"].unique()))

    # RQ5 optional defences, on the same P^- slice
    defences = []
    if p_minus_sets:
        (o_name, n_name), pm = max(p_minus_sets.items(), key=lambda kv: len(kv[1]))
        new_run = next(r for r in runs if r.name == n_name)
        base_vec = np.array([new_run.recovery[b].obs_bypassed for b in pm], int)
        if base_vec.size and base_vec.sum() > 0:
            defences = simulate_defences(base_vec, make_rng(cfg.seed + 21),
                                         cfg.decision.delta_util)

    return StudyResult(cfg=cfg, manifest=manifest, runs=runs, pilot=pilot, rsr=rsr, pbr=pbr,
                       calib_bundle=calib_bundle, calib_response=calib_response,
                       gate_audit=manifest.audit_rates(), drivers=drivers, defences=defences)
