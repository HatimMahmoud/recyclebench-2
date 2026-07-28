"""Tests for human calibration of the judge (bundle-complete annotation)."""
import csv

import pytest

from recyclebench.config import Config, make_rng
from recyclebench.datasets import make_synthetic
from recyclebench.reformulate import load_reformulator
from recyclebench.gate import load_gate
from recyclebench.manifest import Manifest
from recyclebench.models import load_lineage
from recyclebench.judge import MockJudge
from recyclebench.screening import run_model
from recyclebench.judge_audit import (sample_bundles, export_judge_audit,
                                      ingest_judge_audit, calibrate_from_human,
                                      IngestedLabels)


def _run(n=120):
    cfg = Config(n_behaviours=n)
    b = make_synthetic(cfg, make_rng(cfg.seed))
    r = load_reformulator("mock", make_rng(cfg.seed + 11), cfg.transform_classes,
                          cfg.budget.paraphrases_per_behaviour, cfg.mock)
    g = load_gate("mock", make_rng(cfg.seed + 12), cfg.gate, cfg.mock)
    m = Manifest.build(b, r, g)
    model = load_lineage(cfg, make_rng(cfg.seed))[-1]
    j = MockJudge(make_rng(cfg.seed + 13), se=cfg.mock.judge_se, sp=cfg.mock.judge_sp)
    run = run_model(model, b, m, j, cfg, store_responses=True)
    return cfg, b, m, run


# ---------------------------------------------------------------- sampling
def test_sampler_selects_whole_bundles_not_scattered_responses():
    """Bundle-level calibration needs the OR over a bundle, so every admitted variant of a
    selected behaviour must appear."""
    cfg, b, m, run = _run()
    s = sample_bundles(run, b, m, run.responses, n_bundles=20, seed=3)
    bids = {r["bid"] for r in s.rows}
    assert s.n_bundles == 20
    for bid in bids:
        rows_for = [r for r in s.rows if r["bid"] == bid]
        assert len(rows_for) == m.n_admitted(bid)      # bundle-complete


def test_sampler_covers_both_judge_verdicts():
    cfg, b, m, run = _run()
    s = sample_bundles(run, b, m, run.responses, n_bundles=30, positive_share=0.5, seed=3)
    verdicts = {run.recovery[bid].obs_bypassed for bid in {r["bid"] for r in s.rows}}
    assert verdicts == {0, 1}      # need positives for se and negatives for sp


# ---------------------------------------------------------------- ingest / OR rule
def _sheet(tmp_path, rows):
    p = tmp_path / "judge.csv"
    fields = ["bid", "variant_index", "transform_class", "judge_label",
              "request", "response", "human_fulfilled"]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in fields}, **r})
    return p


def test_bundle_truth_is_the_or_over_responses(tmp_path):
    p = _sheet(tmp_path, [
        {"bid": "b1", "variant_index": 0, "judge_label": "0", "human_fulfilled": "0"},
        {"bid": "b1", "variant_index": 1, "judge_label": "0", "human_fulfilled": "1"},
        {"bid": "b2", "variant_index": 0, "judge_label": "1", "human_fulfilled": "0"},
        {"bid": "b2", "variant_index": 1, "judge_label": "0", "human_fulfilled": "0"},
    ])
    lab = ingest_judge_audit(p)
    assert lab.n_bundles_complete == 2
    assert sorted(lab.bundle_pairs) == [(0, 1), (1, 0)]
    # b1: human OR = 1, judge OR = 0 ; b2: human OR = 0, judge OR = 1
    assert len(lab.response_pairs) == 4


def test_incomplete_bundle_is_dropped_not_treated_as_zero(tmp_path):
    """Treating a blank as 0 would bias the bundle truth downward and inflate specificity --
    exactly the parameter the correction is most sensitive to."""
    p = _sheet(tmp_path, [
        {"bid": "b1", "variant_index": 0, "judge_label": "1", "human_fulfilled": "1"},
        {"bid": "b1", "variant_index": 1, "judge_label": "0", "human_fulfilled": ""},
        {"bid": "b2", "variant_index": 0, "judge_label": "1", "human_fulfilled": "1"},
    ])
    lab = ingest_judge_audit(p)
    assert lab.n_bundles_complete == 1
    assert lab.n_bundles_dropped == 1
    assert lab.bundle_pairs == [(1, 1)]        # only b2 survives


# ---------------------------------------------------------------- calibration
def test_calibration_recovers_known_se_sp_on_a_census():
    """Hand-built labels with known confusion structure; a proportional (census) audit must
    reproduce se/sp exactly."""
    lab = IngestedLabels()
    # truth=1: 90 judged 1, 10 judged 0  -> se = 0.90
    lab.bundle_pairs += [(1, 1)] * 90 + [(1, 0)] * 10
    # truth=0: 80 judged 0, 20 judged 1  -> sp = 0.80
    lab.bundle_pairs += [(0, 0)] * 80 + [(0, 1)] * 20
    lab.n_bundles_complete = len(lab.bundle_pairs)
    n_pos = sum(1 for _, j in lab.bundle_pairs if j == 1)
    n_neg = sum(1 for _, j in lab.bundle_pairs if j == 0)
    cal = calibrate_from_human(lab, n_pos, n_neg, kappa_floor=0.5, n_boot=200)
    assert cal.bundle.chars.se == pytest.approx(0.90, abs=1e-9)
    assert cal.bundle.chars.sp == pytest.approx(0.80, abs=1e-9)


def test_no_labels_means_judge_cannot_be_admitted():
    cal = calibrate_from_human(IngestedLabels(), 10, 10)
    assert cal.bundle.admitted is False
    assert "cannot be admitted" in cal.note


def test_bundle_specificity_is_lower_than_response_specificity():
    """The empirical statement of the thesis's central wrinkle: false positives compound
    across a best-of-many union, so the same judge is measurably *less* specific at the
    behaviour-bundle level than per response."""
    cfg, b, m, run = _run(n=250)
    from recyclebench.calibration import calibrate
    resp = calibrate(run.resp_pairs, 0.5, response_level=True)
    bund = calibrate(run.bundle_pairs, 0.5, response_level=False)
    assert bund.chars.sp < resp.chars.sp
    # and the drop is in the ballpark predicted by sp**t
    t = m.mean_admitted()
    assert bund.chars.sp < resp.chars.sp ** (0.5 * t)   # clearly compounding, not a blip
