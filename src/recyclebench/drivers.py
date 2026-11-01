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


# ---------------------------------------------------------------------------
# Registered RQ3 estimator for the real data (prereg Amendment 2 §B5.4)
# ---------------------------------------------------------------------------
def _short(term: str) -> str:
    """'C(transform_class, Treatment(reference='paraphrase'))[T.register]' -> 'transform_class:register'"""
    if "[T." not in term:
        return term
    head, lvl = term.split("[T.", 1)
    var = head[2:].split(",")[0].rstrip(")") if head.startswith("C(") else head
    return f"{var}:{lvl.rstrip(']')}"


def separated_classes(df: pd.DataFrame, class_var: str = "transform_class") -> dict:
    """classes whose rewordings all failed or all succeeded: no finite ML estimate (complete separation)"""
    g = df.groupby(class_var)["success"].agg(["sum", "count"])
    return {c: f"{int(r['sum'])}/{int(r['count'])}" for c, r in g.iterrows() if r["sum"] in (0, r["count"])}


def fit_drivers_gee(df: pd.DataFrame, formula: str, cluster: str = "cluster", class_var: str = "transform_class",
                    alpha: float = 0.05) -> dict | None:
    """Logistic GEE, exchangeable working correlation within source behaviour, bias-reduced (Mancl-DeRouen) sandwich
    covariance. Returns odds ratios with 95% intervals, an omnibus Wald test of the transform-class terms, and Holm across
    the class contrasts only (covariates keep unadjusted p). A class with no events (or only events) has no finite estimate:
    its rows leave the model and it is reported apart in ``separated``."""
    if df.empty or df["success"].nunique() < 2:
        return None
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    sep = separated_classes(df, class_var)
    df = df[~df[class_var].isin(sep)].sort_values(cluster).reset_index(drop=True)
    model = smf.gee(formula, groups=cluster, data=df, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit(cov_type="bias_reduced", maxiter=200)
    names = list(res.params.index)
    ci = res.conf_int()
    cls = [i for i, n in enumerate(names) if n.startswith(f"C({class_var}")]
    R = np.zeros((len(cls), len(names)))
    for r, i in enumerate(cls):
        R[r, i] = 1.0
    wt = res.wald_test(R, use_f=False, scalar=True) if cls else None
    p_cls = [float(res.pvalues.iloc[i]) for i in cls]
    flags = holm_bonferroni(p_cls, alpha=alpha) if cls else []
    # Holm-adjusted p (step-down) for reporting
    order = np.argsort(p_cls)
    adj, run = [None] * len(p_cls), 0.0
    for rank, j in enumerate(order):
        run = max(run, min(1.0, (len(p_cls) - rank) * p_cls[j]))
        adj[j] = run
    omni_p = float(np.asarray(wt.pvalue).ravel()[0]) if wt is not None else None
    rows = []
    for i, n in enumerate(names):
        if n.lower() == "intercept":
            continue
        row = dict(term=_short(n), coef=round(float(res.params.iloc[i]), 5), odds_ratio=round(float(np.exp(res.params.iloc[i])), 4),
                   or_ci=[round(float(np.exp(ci.iloc[i, 0])), 4), round(float(np.exp(ci.iloc[i, 1])), 4)],
                   se=round(float(res.bse.iloc[i]), 5), p_value=round(float(res.pvalues.iloc[i]), 6))
        if i in cls:
            k = cls.index(i)
            row.update(p_holm=round(float(adj[k]), 6),
                       driver=bool(flags[k]) and omni_p is not None and omni_p < alpha)
        rows.append(row)
    return dict(method="gee_logit_exchangeable_bias_reduced", formula=formula, separated=sep, n_obs=int(len(df)),
                n_clusters=int(df[cluster].nunique()),
                omnibus_class=dict(chi2=round(float(np.asarray(wt.statistic).ravel()[0]), 4), df=len(cls),
                                   p_value=round(omni_p, 6)) if wt is not None else None,
                working_correlation=round(float(res.model.cov_struct.dep_params), 4), table=rows)


def fit_drivers_vb(df: pd.DataFrame, formula: str, cluster: str = "cluster", vb_seed: int = 20260101) -> dict | None:
    """Sensitivity: binomial mixed model, random intercept per source behaviour, variational Bayes. Posterior means with
    95% credible intervals (normal VB posterior); deliberately no p-values."""
    if df.empty or df["success"].nunique() < 2:
        return None
    sep = separated_classes(df)
    df = df[~df["transform_class"].isin(sep)].reset_index(drop=True)      # same rows as the GEE
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    _state = np.random.get_state()
    np.random.seed(vb_seed)
    try:
        res = BinomialBayesMixedGLM.from_formula(formula, {cluster: f"0 + C({cluster})"}, df).fit_vb(verbose=False)
    finally:
        np.random.set_state(_state)
    rows = []
    for n, m, s in zip(res.model.exog_names, np.asarray(res.fe_mean), np.asarray(res.fe_sd)):
        if n.lower() == "intercept":
            continue
        rows.append(dict(term=_short(n), coef=round(float(m), 5), odds_ratio=round(float(np.exp(m)), 4),
                         or_cri=[round(float(np.exp(m - 1.96 * s)), 4), round(float(np.exp(m + 1.96 * s)), 4)]))
    return dict(method="bayes_mixed_glm_vb_random_intercept", formula=formula, separated=sep, n_obs=int(len(df)),
                n_clusters=int(df[cluster].nunique()), table=rows,
                note="posterior means and 95% credible intervals; no p-values (sensitivity analysis)")
