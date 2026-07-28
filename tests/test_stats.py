import math
import numpy as np
from recyclebench.stats import (
    wilson_interval, JudgeChars, rogan_gladen, corrected_interval,
    mcnemar_exact, bootstrap_diff, holm_bonferroni, tost_noninferiority,
)


def test_wilson_basic():
    iv = wilson_interval(5, 10)
    assert iv.point == 0.5
    assert 0.0 <= iv.low < 0.5 < iv.high <= 1.0
    assert math.isnan(wilson_interval(0, 0).point)


def test_wilson_edges():
    assert wilson_interval(0, 20).low == 0.0
    assert wilson_interval(20, 20).high == 1.0


def test_rogan_gladen_identity_when_perfect():
    # perfect judge -> no correction
    assert abs(rogan_gladen(0.3, 1.0, 1.0) - 0.3) < 1e-9


def test_rogan_gladen_direction():
    # low specificity inflates observed -> corrected below observed
    assert rogan_gladen(0.30, 0.9, 0.9) < 0.30
    # perfect sp, imperfect se -> corrected above observed
    assert rogan_gladen(0.30, 0.9, 1.0) > 0.30


def test_bundle_specificity_compounds():
    jc = JudgeChars(se=0.9, sp=0.95, response_level=True)
    assert jc.bundle_specificity(1.0) == 0.95
    # more admitted variants -> lower effective specificity (FPs compound)
    assert jc.bundle_specificity(6.0) < jc.bundle_specificity(2.0) < 0.95
    assert abs(jc.bundle_specificity(6.0) - 0.95 ** 6) < 1e-12


def test_corrected_interval_contains_point_and_orders():
    jc = JudgeChars(se=0.92, sp=0.96, var_se=1e-4, var_sp=1e-4, response_level=True)
    iv = corrected_interval(20, 100, jc, t_mean=5.0)
    assert 0.0 <= iv.low <= iv.point <= iv.high <= 1.0


def test_mcnemar_pure_reduction_significant():
    # 30 fixed (1->0), 0 regressed -> strongly significant
    before = [1] * 40 + [0] * 60
    after = [0] * 40 + [0] * 60
    r = mcnemar_exact(before, after)
    assert r["b"] == 40 and r["c"] == 0
    assert r["p_value"] < 1e-6


def test_mcnemar_balanced_not_significant():
    before = [1, 0, 1, 0]
    after = [0, 1, 0, 1]
    r = mcnemar_exact(before, after)
    assert r["p_value"] == 1.0


def test_bootstrap_diff_sign():
    x = [0] * 100
    y = [1] * 30 + [0] * 70
    iv = bootstrap_diff(x, y, n_boot=2000, seed=1)
    assert iv.point > 0 and iv.low > 0  # y clearly higher


def test_holm_monotone():
    flags = holm_bonferroni([0.001, 0.02, 0.2, 0.9])
    assert flags[0] is True
    assert flags[-1] is False


def test_tost_deployable_vs_not():
    dep = tost_noninferiority(3, 300, 4, 300, margin=0.05)
    assert dep["deployable"] is True
    notdep = tost_noninferiority(3, 300, 40, 300, margin=0.05)
    assert notdep["deployable"] is False
