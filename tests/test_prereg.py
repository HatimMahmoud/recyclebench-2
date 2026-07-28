"""Tests for the pre-registration artifact.

The point of these is narrow and important: a hash that fails to notice a moved threshold
would be worse than no hash at all, because it would lend false assurance.
"""
import pytest

from recyclebench.config import Config
from recyclebench.prereg import (prereg_hash, build_prereg, write_prereg,
                                 verify_prereg, decision_fields)


# ---------------------------------------------------------------- hash scope
def test_hash_is_stable_for_an_unchanged_config():
    assert prereg_hash(Config()) == prereg_hash(Config())


def test_cosmetic_settings_do_not_invalidate_a_registration(tmp_path):
    """Housekeeping (output paths, caching, mock simulation size) must not force a
    re-registration -- otherwise the mechanism becomes noise and gets ignored."""
    p = write_prereg(Config(), tmp_path / "P.md")
    cosmetic = Config(results_dir="other", store_raw=True, n_behaviours=999,
                      cache_path="elsewhere.jsonl")
    assert verify_prereg(cosmetic, p)["status"] == "ok"


@pytest.mark.parametrize("mutate,field", [
    (lambda c: setattr(c.decision, "min_p_minus", 10), "decision"),
    (lambda c: setattr(c.decision, "pi_star", 0.4), "decision"),
    (lambda c: setattr(c.decision, "kappa_floor", 0.2), "decision"),
    (lambda c: setattr(c.gate, "cosine", 0.5), "gate"),
    (lambda c: setattr(c.gate, "nli_entail", 0.1), "gate"),
    (lambda c: setattr(c.budget, "paraphrases_per_behaviour", 40), "budget"),
    (lambda c: setattr(c.budget, "k_canonical_samples", 25), "budget"),
])
def test_every_decision_relevant_change_is_detected(tmp_path, mutate, field):
    p = write_prereg(Config(), tmp_path / "P.md")
    cfg = Config()
    mutate(cfg)
    r = verify_prereg(cfg, p)
    assert r["status"] == "DRIFT"
    assert field in [c["field"] for c in r["changed_fields"]]


def test_lineage_and_seed_changes_are_detected(tmp_path):
    p = write_prereg(Config(), tmp_path / "P.md")
    assert verify_prereg(Config(seed=999), p)["status"] == "DRIFT"
    assert verify_prereg(Config(lineage=["a", "b"]), p)["status"] == "DRIFT"


# ---------------------------------------------------------------- behaviour pinning
def test_behaviour_file_is_pinned_by_digest(tmp_path):
    f = tmp_path / "b.csv"
    f.write_text("Goal\nrequest one\n")
    p = write_prereg(Config(), tmp_path / "P.md", behaviours_file=f)
    assert verify_prereg(Config(), p, behaviours_file=f)["status"] == "ok"
    f.write_text("Goal\nrequest one\nrequest two\n")        # dataset swapped afterwards
    assert verify_prereg(Config(), p, behaviours_file=f)["status"] == "DRIFT"


# ---------------------------------------------------------------- document + semantics
def test_document_states_the_non_negotiable_rules():
    doc = build_prereg(Config())
    assert "undefined" in doc                      # empty P^- is undefined, not zero
    assert "behaviours" in doc and "never paraphrase variants" in doc
    assert "sp^t" in doc                           # bundle-level correction
    assert "unresolved" in doc                     # gate strictness != robustness
    assert "McNemar" in doc and "Holm" in doc


def test_document_records_the_actual_configured_values():
    cfg = Config()
    cfg.decision.min_p_minus = 55
    cfg.gate.cosine = 0.81
    doc = build_prereg(cfg)
    assert "55" in doc and "0.81" in doc            # not boilerplate defaults


def test_registration_is_write_once_unless_forced(tmp_path):
    p = write_prereg(Config(), tmp_path / "P.md")
    with pytest.raises(FileExistsError):
        write_prereg(Config(), p)
    write_prereg(Config(), p, overwrite=True)       # explicit escape hatch only


def test_missing_and_unreadable_registrations_are_reported(tmp_path):
    assert verify_prereg(Config(), tmp_path / "nope.md")["status"] == "missing"
    bad = tmp_path / "bad.md"
    bad.write_text("# not a registration\n")
    assert verify_prereg(Config(), bad)["status"] == "unreadable"


def test_decision_fields_excludes_simulation_parameters():
    """Mock parameters shape the synthetic instrument check only; including them would make
    every tuning tweak look like scientific misconduct."""
    f = decision_fields(Config())
    assert "mock" not in f
    assert {"gate", "budget", "decision", "seed", "lineage"} <= set(f)


# ---------------------------------------------------------------- figures
def test_figures_render_from_a_study_result(tmp_path):
    from recyclebench.study import run_study
    from recyclebench.audit import threshold_sensitivity
    from recyclebench import figures as F
    res = run_study(Config(n_behaviours=120))
    sens = threshold_sensitivity(res.manifest, res.cfg.gate)
    made = F.all_figures(res, sens, outdir=tmp_path)
    assert len(made) == 4
    for p in made:
        assert p.exists() and p.stat().st_size > 5_000      # not a blank canvas


def test_rsr_forest_handles_an_undefined_pair(tmp_path):
    """A pair with empty P^- must not crash the plot or be silently dropped from the title."""
    from recyclebench import figures as F
    from recyclebench.recycling import RSRResult
    undef = RSRResult(("a", "b"), 0, 0, defined=False, estimable=False, rsr_raw=None,
                      raw_ci=None, rsr_corrected=None, corrected_ci=None,
                      mean_admitted=0.0, mcnemar_p=None, by_category={})
    out = F.rsr_forest([undef], tmp_path / "f.png")
    assert out.exists()
