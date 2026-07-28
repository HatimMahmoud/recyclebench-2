"""RQ3 -- which paraphrase classes recover patched jailbreaks (exposé Sec. 4, 6).

Mixed-effects logistic regression of *per-variant recovery*, restricted to P^-, on
transformation class, surface distance, and harm category, with a random intercept per
source behaviour (variant-level data is used ONLY here -- the headline denominators remain
numbers of behaviours). Family-wise error is controlled with Holm-Bonferroni.

Falls back to a cluster-robust logit if the mixed model fails to converge, and reports which
estimator was used so the choice is transparent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import holm_bonferroni


@dataclass
class DriverResult:
    method: str
    table: pd.DataFrame
    n_obs: int
    n_groups: int

    def summary(self) -> dict:
        return {"method": self.method, "n_obs": self.n_obs, "n_groups": self.n_groups,
                "table": self.table.to_dict(orient="records")}


def fit_drivers(trials: pd.DataFrame, p_minus: list[str],
                alpha: float = 0.05, vb_seed: int = 20260101) -> DriverResult | None:
    """trials: per-variant rows for the NEWER model; restricted here to P^-."""
    df = trials[trials["bid"].isin(set(p_minus))].copy()
    if df.empty or df["success"].nunique() < 2:
        return None
    df["surface_distance"] = df["surface_distance"].astype(float)

    formula = "success ~ C(transform_class) + surface_distance + C(category)"
    method = "mixedlm_binomial"
    try:
        import statsmodels.formula.api as smf
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        # ``fit_vb`` initialises its variational parameters with an unseeded draw from the
        # LEGACY global numpy random state, which makes an otherwise identical fit vary in
        # the last decimals. The instrument is pre-registered and must be byte-reproducible,
        # so we pin that global state around the fit and restore it afterwards.
        _state = np.random.get_state()
        np.random.seed(vb_seed)
        try:
            md = BinomialBayesMixedGLM.from_formula(
                formula, {"bid": "0 + C(bid)"}, df)
            res = md.fit_vb(verbose=False)
        finally:
            np.random.set_state(_state)
        names = list(res.model.exog_names)
        params = np.asarray(res.fe_mean)
        ses = np.asarray(res.fe_sd)
        z = params / np.where(ses > 0, ses, np.nan)
        from scipy import stats as sps
        pvals = 2 * (1 - sps.norm.cdf(np.abs(z)))
    except Exception:
        # fallback: cluster-robust logit
        method = "cluster_robust_logit"
        import statsmodels.formula.api as smf
        res = smf.logit(formula, data=df).fit(disp=False,
                                              cov_type="cluster",
                                              cov_kwds={"groups": df["bid"]})
        names = list(res.params.index)
        params = res.params.values
        ses = res.bse.values
        pvals = res.pvalues.values

    keep = [i for i, n in enumerate(names) if n.lower() != "intercept"]
    # The variational optimiser converges to ~1e-6; round the reported estimates so that a
    # re-run with the same seed is byte-identical (the instrument is pre-registered).
    p_keep = [round(float(pvals[i]), 6) for i in keep]
    flags = holm_bonferroni(p_keep, alpha=alpha)
    table = pd.DataFrame({
        "term": [names[i] for i in keep],
        "coef": [round(float(params[i]), 5) for i in keep],
        "odds_ratio": [round(float(np.exp(params[i])), 5) for i in keep],
        "se": [round(float(ses[i]), 5) for i in keep],
        "p_value": p_keep,
        "driver_after_holm": flags,
    }).sort_values("p_value").reset_index(drop=True)
    return DriverResult(method, table, len(df), df["bid"].nunique())
