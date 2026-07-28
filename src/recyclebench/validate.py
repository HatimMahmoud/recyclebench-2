"""Instrument validation (exposé Sec. 7, L0).

Monte-Carlo check that the estimators are correct *before* any real model is measured:

1. **Coverage** -- across replicates with a known planted truth, the bundle-level corrected
   95% interval should cover the true rate ~95% of the time, while the raw (uncorrected)
   interval should under-cover.
2. **Bias** -- the raw point estimate should be biased upward (judge false positives
   compound across the best-of-many union); the corrected estimate should be ~unbiased.
3. **Bundle-vs-response correction** -- applying a *response-level* specificity to the
   union (the mis-specification the exposé warns about) should be visibly worse than the
   bundle-level correction.

This is what licenses trust in the real number's machinery; it says nothing about Llama.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import JudgeChars, corrected_interval, wilson_interval


@dataclass
class ValidationReport:
    n_rep: int
    truth: float
    raw_coverage: float
    corrected_coverage: float
    raw_bias: float
    corrected_bias: float
    naive_response_bias: float
    passed: bool

    def summary(self) -> dict:
        return {"n_replicates": self.n_rep, "true_rate": round(self.truth, 4),
                "raw_coverage": round(self.raw_coverage, 4),
                "corrected_coverage": round(self.corrected_coverage, 4),
                "raw_bias": round(self.raw_bias, 4),
                "corrected_bias": round(self.corrected_bias, 4),
                "naive_response_level_bias": round(self.naive_response_bias, 4),
                "PASS": self.passed}


def simulate(n_behaviours: int = 300, t: int = 6, truth: float = 0.30,
             se: float = 0.92, sp: float = 0.97, n_rep: int = 400,
             seed: int = 7, n_calib: int = 800) -> ValidationReport:
    """Simulate behaviour bundles with a known true bypass rate and judge error.

    Each behaviour is truly bypassed w.p. ``truth``. A truly-bypassed behaviour has one
    genuinely harmful variant response (detected w.p. se); every one of its ``t`` admitted
    variants can also produce an independent false positive w.p. (1-sp). The union makes the
    observed bundle rate biased upward.
    """
    rng = np.random.default_rng(seed)
    raw_cov = corr_cov = 0
    raw_b = corr_b = naive_b = 0.0
    sp_bundle_true = sp ** t

    for _ in range(n_rep):
        truth_vec = rng.random(n_behaviours) < truth
        obs = np.empty(n_behaviours, dtype=int)
        for i, is_true in enumerate(truth_vec):
            fp = np.any(rng.random(t) > sp)              # any variant false-positives
            tp = is_true and (rng.random() < se)         # true positive detected
            obs[i] = int(tp or fp)
        k = int(obs.sum())

        # calibration sample at the BUNDLE level (what the protocol prescribes)
        cal_truth = rng.random(n_calib) < truth
        cal_obs = np.array([
            int(((t_ and rng.random() < se)) or np.any(rng.random(t) > sp))
            for t_ in cal_truth], int)
        npos = int(cal_truth.sum()); nneg = n_calib - npos
        se_hat = cal_obs[cal_truth].mean() if npos else 1.0
        sp_hat = (1 - cal_obs[~cal_truth]).mean() if nneg else 1.0
        jc_bundle = JudgeChars(se=se_hat, sp=sp_hat,
                               var_se=se_hat * (1 - se_hat) / max(npos, 1),
                               var_sp=sp_hat * (1 - sp_hat) / max(nneg, 1),
                               n_pos=npos, n_neg=nneg, response_level=False)

        raw = wilson_interval(k, n_behaviours)
        corr = corrected_interval(k, n_behaviours, jc_bundle)
        # the MIS-SPECIFIED alternative: use per-response sp without compounding
        jc_naive = JudgeChars(se=se, sp=sp, response_level=False)
        naive = corrected_interval(k, n_behaviours, jc_naive)

        raw_cov += int(raw.low <= truth <= raw.high)
        corr_cov += int(corr.low <= truth <= corr.high)
        raw_b += raw.point - truth
        corr_b += corr.point - truth
        naive_b += naive.point - truth

    raw_coverage = raw_cov / n_rep
    corrected_coverage = corr_cov / n_rep
    raw_bias = raw_b / n_rep
    corrected_bias = corr_b / n_rep
    naive_bias = naive_b / n_rep
    passed = bool(corrected_coverage >= 0.90 and abs(corrected_bias) < 0.03
                  and raw_bias > 0.02 and corrected_coverage > raw_coverage)
    return ValidationReport(n_rep, truth, raw_coverage, corrected_coverage,
                            raw_bias, corrected_bias, naive_bias, passed)
