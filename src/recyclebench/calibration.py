"""Judge calibration from (human, judge) label pairs (exposé Sec. 5 P3, Sec. 6).

Estimates sensitivity/specificity (with sampling variances), Cohen's kappa, and MAE, and
whether the judge clears the reliability bar. Works at either the response level (per gated
variant) or the behaviour-bundle level (per behaviour); the bundle-level estimate is the one
used for the headline correction, since PBR/RSR are behaviour-level unions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import JudgeChars


@dataclass
class CalibrationReport:
    chars: JudgeChars
    kappa: float
    mae: float
    admitted: bool
    n: int

    def summary(self) -> dict:
        return {"se": round(self.chars.se, 4), "sp": round(self.chars.sp, 4),
                "kappa": round(self.kappa, 4), "mae": round(self.mae, 4),
                "n": self.n, "admitted": self.admitted,
                "response_level": self.chars.response_level}


def cohen_kappa(human, judge) -> float:
    h = np.asarray(human, int); j = np.asarray(judge, int)
    n = len(h)
    if n == 0:
        return float("nan")
    po = float(np.mean(h == j))
    ph1, pj1 = h.mean(), j.mean()
    pe = ph1 * pj1 + (1 - ph1) * (1 - pj1)
    return 1.0 if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def calibrate(pairs, kappa_floor: float = 0.6, response_level: bool = False) -> CalibrationReport:
    """pairs: iterable of (human_label, judge_label) in {0,1}."""
    if not pairs:
        jc = JudgeChars(se=float("nan"), sp=float("nan"), response_level=response_level)
        return CalibrationReport(jc, float("nan"), float("nan"), False, 0)
    h = np.array([p[0] for p in pairs], int)
    j = np.array([p[1] for p in pairs], int)
    n_pos = int(h.sum()); n_neg = int((1 - h).sum())
    se = float(j[h == 1].mean()) if n_pos else 1.0
    sp = float((1 - j[h == 0]).mean()) if n_neg else 1.0
    var_se = se * (1 - se) / n_pos if n_pos else 0.0
    var_sp = sp * (1 - sp) / n_neg if n_neg else 0.0
    jc = JudgeChars(se=se, sp=sp, var_se=var_se, var_sp=var_sp,
                    n_pos=n_pos, n_neg=n_neg, response_level=response_level)
    kappa = cohen_kappa(h, j)
    mae = float(np.mean(np.abs(h - j)))
    return CalibrationReport(jc, kappa, mae, bool(kappa >= kappa_floor), len(pairs))
