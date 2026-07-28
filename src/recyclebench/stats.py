"""Statistical core (exposé Sec. 6).

Everything here is estimator-level and unit-tested in isolation:

* Wilson score interval for a proportion.
* **Behaviour-bundle** Rogan-Gladen correction. PBR/RSR are best-of-many unions, so a
  per-response specificity ``sp`` compounds across the ``t`` admitted variants of a
  behaviour: the effective behaviour-level specificity is ``sp**t``. We therefore correct
  at the bundle level, using either an explicitly calibrated bundle-level (se, sp) or a
  per-response sp compounded to the mean admitted-variant count.
* McNemar's exact test (paired canonical-vs-recycled; defence with/without).
* Bootstrap CI for a difference of paired rates.
* Holm-Bonferroni family-wise correction.
* TOST non-inferiority for benign over-refusal (defence deployability).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Interval container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def as_tuple(self):
        return (self.point, self.low, self.high)


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------
def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Interval:
    """Wilson score interval for k successes in n trials (95% by default)."""
    if n <= 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Judge characteristics + bundle-level Rogan-Gladen correction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JudgeChars:
    """Calibrated judge sensitivity/specificity with sampling variances.

    ``response_level`` marks whether (se, sp) were measured per response (and must be
    compounded to a bundle) or already at the behaviour-bundle level.
    """
    se: float
    sp: float
    var_se: float = 0.0
    var_sp: float = 0.0
    n_pos: int = 0
    n_neg: int = 0
    response_level: bool = False

    @property
    def youden(self) -> float:
        return self.se + self.sp - 1.0

    def bundle_specificity(self, t_mean: float) -> float:
        """Effective behaviour-level specificity. A behaviour registers a (false) positive
        if ANY of its t admitted variants does, so bundle sp = sp_response ** t_mean."""
        if not self.response_level or t_mean <= 1.0:
            return self.sp
        return self.sp ** t_mean


def rogan_gladen(p_obs: float, se: float, sp: float) -> float:
    """Prevalence corrected for misclassification: (p_obs + sp - 1) / (se + sp - 1)."""
    j = se + sp - 1.0
    if abs(j) < 1e-9:
        return float("nan")
    return min(1.0, max(0.0, (p_obs + sp - 1.0) / j))


def corrected_interval(k: int, n: int, jc: JudgeChars, t_mean: float = 1.0,
                       z: float = 1.959963984540054) -> Interval:
    """Bundle-level judge-error-corrected interval for k/n (behaviours), delta method.

    The observed rate's variance (Wilson-free binomial) is propagated together with the
    calibration uncertainty in (se, sp) through the Rogan-Gladen map. Specificity is taken
    at the bundle level (sp**t_mean) when the calibration is response-level.
    """
    if n <= 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    sp_eff = jc.bundle_specificity(t_mean)
    se = jc.se
    j = se + sp_eff - 1.0
    p_obs = k / n
    point = rogan_gladen(p_obs, se, sp_eff)
    if not math.isfinite(point) or abs(j) < 1e-9:
        return Interval(point, float("nan"), float("nan"))
    # variance of observed proportion
    var_obs = p_obs * (1 - p_obs) / n
    # partial derivatives of RG w.r.t. p_obs, se, sp
    d_p = 1.0 / j
    d_se = -(p_obs + sp_eff - 1.0) / (j * j)
    d_sp = (se - 1.0 - (p_obs - 1.0)) / (j * j)  # d/dsp of (p+sp-1)/(se+sp-1)
    # if sp is response-level, chain through sp_eff = sp**t
    if jc.response_level and t_mean > 1.0 and jc.sp > 0:
        d_sp = d_sp * (t_mean * jc.sp ** (t_mean - 1.0))
    var = (d_p ** 2) * var_obs + (d_se ** 2) * jc.var_se + (d_sp ** 2) * jc.var_sp
    half = z * math.sqrt(max(var, 0.0))
    return Interval(point, max(0.0, point - half), min(1.0, point + half))


# ---------------------------------------------------------------------------
# McNemar exact test (paired binary)
# ---------------------------------------------------------------------------
def mcnemar_exact(before: Sequence[int], after: Sequence[int]) -> dict:
    """Exact (binomial) McNemar test on paired 0/1 vectors.

    b = before=1,after=0 (fixed) ; c = before=0,after=1 (recovered/regressed).
    Two-sided exact p from Binomial(b+c, 0.5).
    """
    a = np.asarray(before, int); d = np.asarray(after, int)
    assert a.shape == d.shape, "paired vectors must align"
    b = int(np.sum((a == 1) & (d == 0)))
    c = int(np.sum((a == 0) & (d == 1)))
    nd = b + c
    if nd == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = min(1.0, 2.0 * sps.binom.cdf(k, nd, 0.5))
    return {"b": b, "c": c, "n_discordant": nd, "p_value": float(p)}


# ---------------------------------------------------------------------------
# Bootstrap CI for a difference of paired rates
# ---------------------------------------------------------------------------
def bootstrap_diff(x: Sequence[int], y: Sequence[int], n_boot: int = 5000,
                   seed: int = 0, z_ignored=None) -> Interval:
    """Percentile bootstrap 95% CI for mean(y) - mean(x) over paired observations."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    diffs = np.empty(n_boot)
    idx = np.arange(n)
    for i in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        diffs[i] = y[s].mean() - x[s].mean()
    return Interval(float(y.mean() - x.mean()),
                    float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------
def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Return per-hypothesis reject flags under Holm-Bonferroni at level alpha."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        thresh = alpha / (m - rank)
        if pvals[i] <= thresh:
            reject[i] = True
        else:
            break
    return reject


# ---------------------------------------------------------------------------
# TOST non-inferiority for benign over-refusal (defence deployability)
# ---------------------------------------------------------------------------
def tost_noninferiority(k_base: int, n_base: int, k_def: int, n_def: int,
                        margin: float, z: float = 1.6448536269514722) -> dict:
    """One-sided non-inferiority: is the defence's benign over-refusal increase below margin?

    Tests H1: (p_def - p_base) < margin. Returns the increase, its one-sided upper 95%
    confidence bound (UCB), and a deployable flag (UCB < margin).
    """
    pb = k_base / n_base if n_base else float("nan")
    pd = k_def / n_def if n_def else float("nan")
    diff = pd - pb
    se = math.sqrt(pb * (1 - pb) / max(n_base, 1) + pd * (1 - pd) / max(n_def, 1))
    ucb = diff + z * se
    return {"over_refusal": diff, "ucb": ucb, "margin": margin, "deployable": bool(ucb < margin)}
