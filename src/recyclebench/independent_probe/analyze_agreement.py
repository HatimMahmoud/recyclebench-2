"""
analyze_agreement.py -- the convergent-validity result for Possibility 1.

Given ProbeItems that carry BOTH arms' per-behaviour verdicts
(para_recovered from the recycle run, rpe_recovered from run_probe), compute:

    - RSR_paraphrase and RSR_rpe (each = recovered / resolved)
    - agreement rate on the behaviours BOTH arms resolved
    - Cohen's kappa (chance-corrected agreement)
    - divergence lists: RPE-only recoveries and paraphrase-only recoveries

The headline claim this supports: "the shallowness of cross-generational safety
fixes is robust to how you probe it -- an independent method (RPE) reproduces the
paraphrase result (kappa=...); where they diverge, inversion recovers additional
preimage members, tightening the lower bound."

No external deps -- kappa and Wilson CI computed in-module.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class AgreementReport:
    n_total: int
    n_both_resolved: int
    rsr_para: Optional[float]
    rsr_rpe: Optional[float]
    rsr_para_ci: tuple
    rsr_rpe_ci: tuple
    agreement_rate: Optional[float]
    cohen_kappa: Optional[float]
    mcnemar_p: Optional[float]
    both_recovered: list
    both_held: list
    rpe_only: list          # RPE recovered, paraphrase held  (inversion finds more)
    para_only: list         # paraphrase recovered, RPE held
    para_unresolved: int
    rpe_unresolved: int

    def summary(self) -> str:
        def pct(x): return "n/a" if x is None else f"{x*100:.1f}%"
        def ci(c): return "n/a" if c[0] is None else f"[{c[0]*100:.1f}, {c[1]*100:.1f}]"
        lines = [
            "=== Independent-probe convergent validity (paraphrase vs RPE) ===",
            f"behaviours (P-)                 : {self.n_total}",
            f"resolved by BOTH arms           : {self.n_both_resolved}",
            f"RSR_paraphrase                  : {pct(self.rsr_para)}  CI {ci(self.rsr_para_ci)}",
            f"RSR_rpe (MVP)                   : {pct(self.rsr_rpe)}  CI {ci(self.rsr_rpe_ci)}",
            f"agreement rate (both resolved)  : {pct(self.agreement_rate)}",
            f"Cohen's kappa                   : {'n/a' if self.cohen_kappa is None else f'{self.cohen_kappa:.3f}'}",
            f"McNemar exact p (para vs rpe)   : {'n/a' if self.mcnemar_p is None else f'{self.mcnemar_p:.4g}'}",
            f"both recovered                  : {len(self.both_recovered)}",
            f"both held                       : {len(self.both_held)}",
            f"RPE-only recoveries             : {len(self.rpe_only)}   (inversion finds more)",
            f"paraphrase-only recoveries      : {len(self.para_only)}",
            f"unresolved (para / rpe)         : {self.para_unresolved} / {self.rpe_unresolved}",
        ]
        return "\n".join(lines)


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return (max(0.0, centre-half), min(1.0, centre+half))


def _rsr(verdicts: list[Optional[int]]):
    resolved = [v for v in verdicts if v is not None]
    if not resolved:
        return None, (None, None), 0
    k = sum(resolved)
    n = len(resolved)
    return k/n, _wilson(k, n), n


def _cohen_kappa(pairs: list[tuple[int, int]]):
    """pairs of (para, rpe) verdicts, each in {0,1}, both resolved."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1*pb1 + (1-pa1)*(1-pb1)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def analyze(items) -> AgreementReport:
    para = [it.para_recovered for it in items]
    rpe = [it.rpe_recovered for it in items]

    rsr_para, ci_para, _ = _rsr(para)
    rsr_rpe, ci_rpe, _ = _rsr(rpe)

    both_recovered, both_held, rpe_only, para_only = [], [], [], []
    kappa_pairs = []
    for it in items:
        a, b = it.para_recovered, it.rpe_recovered
        if a is None or b is None:
            continue
        kappa_pairs.append((a, b))
        if a == 1 and b == 1: both_recovered.append(it.bid)
        elif a == 0 and b == 0: both_held.append(it.bid)
        elif a == 0 and b == 1: rpe_only.append(it.bid)
        elif a == 1 and b == 0: para_only.append(it.bid)

    n_both = len(kappa_pairs)
    agreement = (sum(1 for a, b in kappa_pairs if a == b) / n_both) if n_both else None
    kappa = _cohen_kappa(kappa_pairs)
    # kappa measures concordance; McNemar asks whether the two arms re-open at
    # different *rates* on the same behaviours (the discordant cells only).
    mcn = None
    if n_both:
        from recyclebench.stats import mcnemar_exact      # single source of truth
        mcn = mcnemar_exact([a for a, _ in kappa_pairs], [b for _, b in kappa_pairs]).get("p_value")

    return AgreementReport(
        n_total=len(items),
        n_both_resolved=n_both,
        rsr_para=rsr_para, rsr_rpe=rsr_rpe,
        rsr_para_ci=ci_para, rsr_rpe_ci=ci_rpe,
        agreement_rate=agreement, cohen_kappa=kappa, mcnemar_p=mcn,
        both_recovered=both_recovered, both_held=both_held,
        rpe_only=rpe_only, para_only=para_only,
        para_unresolved=sum(1 for v in para if v is None),
        rpe_unresolved=sum(1 for v in rpe if v is None),
    )
