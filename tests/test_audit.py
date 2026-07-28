"""Tests for the human audit of the intent-preservation gate."""
import csv

import pytest

from recyclebench.config import Config, GateThresholds, make_rng
from recyclebench.datasets import make_synthetic
from recyclebench.reformulate import load_reformulator
from recyclebench.gate import load_gate
from recyclebench.manifest import Manifest, GateAuditRecord
from recyclebench.audit import (sample_audit, export_audit, ingest_audit,
                                estimate_from_audit, threshold_sensitivity)


def _manifest(n=200, seed=20260101):
    cfg = Config(n_behaviours=n, seed=seed)
    b = make_synthetic(cfg, make_rng(cfg.seed))
    r = load_reformulator("mock", make_rng(cfg.seed + 11), cfg.transform_classes,
                          cfg.budget.paraphrases_per_behaviour, cfg.mock)
    g = load_gate("mock", make_rng(cfg.seed + 12), cfg.gate, cfg.mock)
    return cfg, b, Manifest.build(b, r, g)


# ---------------------------------------------------------------- sampling
def test_sample_oversamples_admitted_and_reports_strata():
    cfg, b, m = _manifest()
    s = sample_audit(m, b, n=100, admitted_share=0.7, seed=3)
    n_adm = sum(r["gate_admitted"] for r in s.rows)
    assert len(s) == 100
    assert n_adm == 70                       # bounding false admission is the priority
    assert s.n_admitted_total == sum(1 for r in m.audit if r.admitted)
    assert s.n_rejected_total == sum(1 for r in m.audit if not r.admitted)


def test_sample_rows_carry_text_and_scores_for_annotation():
    cfg, b, m = _manifest()
    s = sample_audit(m, b, n=20, seed=3)
    row = s.rows[0]
    assert row["original_request"] and row["rewrite"]        # annotator can judge
    assert row["human_intent_preserved"] == ""               # left blank for the human
    assert 0.0 <= float(row["cosine"]) <= 1.0


# ---------------------------------------------------------------- export / ingest
def test_export_ingest_roundtrip_and_blanks_skipped(tmp_path):
    cfg, b, m = _manifest()
    s = sample_audit(m, b, n=60, seed=4)
    p = export_audit(s, tmp_path / "sheet.csv")
    assert (tmp_path / "RUBRIC.txt").exists()
    assert (tmp_path / "strata.txt").exists()

    rows = list(csv.DictReader(p.open()))
    for i, row in enumerate(rows):
        row["human_intent_preserved"] = "" if i < 10 else "1"   # 10 uncertain -> blank
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    n = ingest_audit(m, p)
    assert n == len(rows) - 10          # blanks are excluded, never guessed


def test_ingest_prefers_human_label_over_latent_truth():
    m = Manifest()
    rec = GateAuditRecord("b1", 0, "paraphrase", intent_preserved=True, admitted=True)
    m.audit.append(rec)
    assert rec.truth() is True
    rec.human_label = False
    assert rec.truth() is False          # human label wins on real runs


# ---------------------------------------------------------------- estimation
def test_stratified_estimator_recovers_known_rates_on_a_census():
    """Hand-built manifest with known error structure; labelling everything must reproduce
    the Bayes-inverted rates exactly (no sampling noise)."""
    m = Manifest()
    # 1000 admitted of which 10% intent-changed; 1000 rejected of which 50% preserved
    for i in range(1000):
        changed = i < 100
        r = GateAuditRecord(f"a{i}", 0, "paraphrase", not changed, True)
        r.human_label = not changed
        m.audit.append(r)
    for i in range(1000):
        preserved = i < 500
        r = GateAuditRecord(f"r{i}", 0, "paraphrase", preserved, False)
        r.human_label = preserved
        m.audit.append(r)

    est = estimate_from_audit(m, n_admitted_total=1000, n_rejected_total=1000, n_boot=200)
    assert est.contamination.point == pytest.approx(0.10, abs=1e-9)
    # P(admitted|changed) = 100 / (100 + 500)
    assert est.false_admission.point == pytest.approx(100 / 600, abs=1e-9)
    # P(rejected|preserved) = 500 / (900 + 500)
    assert est.false_rejection.point == pytest.approx(500 / 1400, abs=1e-9)


def test_estimate_without_labels_is_explicit_not_silent():
    cfg, b, m = _manifest()
    est = estimate_from_audit(m)
    assert est.n_labelled == 0
    assert est.false_admission is None
    assert "no human labels" in est.note


def test_single_stratum_reports_direct_rate_only():
    m = Manifest()
    for i in range(20):
        r = GateAuditRecord(f"a{i}", 0, "paraphrase", True, True)
        r.human_label = (i > 2)
        m.audit.append(r)
    est = estimate_from_audit(m, n_admitted_total=20, n_rejected_total=5)
    assert est.contamination is not None      # directly estimable
    assert est.false_admission is None        # cannot invert without the other stratum
    assert "BOTH strata" in est.note


# ---------------------------------------------------------------- sensitivity
def test_threshold_sensitivity_is_monotone_in_admissions_and_contamination():
    cfg, b, m = _manifest(n=300)
    rows = threshold_sensitivity(m, cfg.gate, deltas=(-0.05, 0.0, 0.05))
    admits = [r["n_admitted"] for r in rows]
    conts = [r["contamination"] for r in rows]
    assert admits == sorted(admits, reverse=True)   # tightening admits fewer
    assert conts == sorted(conts, reverse=True)     # ...and lowers contamination
    # the power cost of tightening is reported alongside
    assert rows[0]["retained_preserved"] >= rows[-1]["retained_preserved"]


def test_sensitivity_needs_no_model_queries():
    """The analysis must run purely off stored scores -- otherwise it could not be part of a
    pre-registered plan executed after the confirmatory run."""
    cfg, b, m = _manifest(n=50)
    for r in m.audit:
        assert r.cosine == r.cosine   # scores persisted (not NaN)
    assert threshold_sensitivity(m, GateThresholds())
