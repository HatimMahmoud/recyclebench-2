"""The configuration-driven experiment layer (src/recyclebench/experiment.py).

Covers the acceptance criteria that do not need a model: spec validation, plan purity (no writes),
stage ordering and guards, isolation between two experiments, resume safety under a changed
fingerprint, and the rule that a stage counts as done only once its artifacts exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from recyclebench.experiment import (                                   # noqa: E402
    STAGES, ExperimentSpec, SpecError, StageStore, build_plan, check_resume, validate,
)

BENIGN = ["Explain how a bicycle derailleur shifts gears.",
          "Summarise the water cycle for a school class.",
          "Describe how sourdough starter is maintained."]


def _csv(tmp_path, name="beh.csv", rows=BENIGN):
    p = tmp_path / name
    p.write_text("Goal,Category\n" + "".join(f'"{g}",demo\n' for g in rows))
    return p


def _spec(tmp_path, **over):
    d = {"name": "demo", "science_config": "configs/recycle_3_31.yaml",
         "old": "llama3-8b", "new": "llama31-8b",
         "behaviours": str(_csv(tmp_path)),
         "execution": {"run_root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}}
    d.update(over)
    return ExperimentSpec.from_dict(d, path="<test>")


# ---------------------------------------------------------------------------
# spec validation: fail clearly, never with a traceback
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("patch,msg", [
    ({"name": None}, "missing required key 'name'"),
    ({"behaviours": None}, "either behaviours:"),
    ({"recipe": "r.yaml"}, "not both"),
    ({"stages": ["screen", "nope"]}, "unknown stages"),
    ({"execution": {"target": "moon"}}, "local or cluster"),
    ({"wat": 1}, "unknown spec keys"),
])
def test_a_bad_spec_fails_with_an_actionable_message(tmp_path, patch, msg):
    base = {"name": "demo", "science_config": "configs/recycle_3_31.yaml",
            "old": "a", "new": "b", "behaviours": str(_csv(tmp_path))}
    base.update(patch)
    with pytest.raises(SpecError, match=msg):
        ExperimentSpec.from_dict({k: v for k, v in base.items() if v is not None})


def test_a_missing_spec_file_and_bad_yaml_fail_clearly(tmp_path):
    with pytest.raises(SpecError, match="no experiment spec"):
        ExperimentSpec.load(tmp_path / "absent.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(SpecError, match="not valid YAML"):
        ExperimentSpec.load(bad)
    notmap = tmp_path / "list.yaml"
    notmap.write_text("- a\n- b\n")
    with pytest.raises(SpecError, match="must contain a mapping"):
        ExperimentSpec.load(notmap)


def test_validate_reports_an_unknown_model_and_an_unparseable_csv(tmp_path):
    s = _spec(tmp_path, old="no-such-model")
    assert any("no-such-model" in p for p in validate(s))
    empty = tmp_path / "empty.csv"
    empty.write_text("NotGoal,Category\nx,y\n")
    s2 = _spec(tmp_path, behaviours=str(empty))
    probs = validate(s2)
    assert any("0 behaviours" in p or "does not parse" in p for p in probs), probs


def test_validate_insists_the_judge_is_configured(tmp_path):
    """A config with no judges: silently falls back to the refusal-prefix rule, which disagrees
    with the LLM judge. That has to be a stated choice, not a discovery after the run."""
    cfg = yaml.safe_load((ROOT / "configs/recycle_3_31.yaml").read_text())
    cfg.pop("judges", None)
    p = tmp_path / "nojudge.yaml"
    p.write_text(yaml.safe_dump(cfg))
    probs = validate(_spec(tmp_path, science_config=str(p)))
    assert any("judges" in x and "refusal-prefix" in x for x in probs), probs


# ---------------------------------------------------------------------------
# plan purity and ordering
# ---------------------------------------------------------------------------
def test_plan_writes_nothing_at_all(tmp_path):
    s = _spec(tmp_path)
    plan = build_plan(s)
    assert plan.writes == []
    assert not (tmp_path / "runs").exists() and not (tmp_path / "cache").exists()
    plan.render()
    assert not s.run_dir.exists(), "rendering a plan created the run directory"


def test_the_recycling_set_comes_from_the_judged_screen_not_the_offline_partition(tmp_path):
    """The invariant from docs/BYOD.md trap 1: `recycle` consumes the judged P^- subset that
    `subset` writes out of the pilot JSON, never the cells `pminus_report --dump` produces."""
    s = _spec(tmp_path)
    plan = build_plan(s)
    order = [st.stage for st in plan.steps]
    assert order.index("screen") < order.index("subset") < order.index("recycle")
    recycle = next(st for st in plan.steps if st.stage == "recycle")
    assert str(s.pminus_csv) in " ".join(recycle.argv)
    assert "cells" not in " ".join(recycle.argv)
    # and the control runs after the primary estimand, so a lost budget keeps RSR
    assert order.index("recycle") < order.index("partition")


def test_recycle_is_skipped_when_p_minus_is_below_the_floor(tmp_path):
    s = _spec(tmp_path)
    s.run_dir.mkdir(parents=True)
    (s.run_dir / "pminus_pilot.json").write_text(json.dumps({"p_minus": 2, "p_minus_bids": []}))
    step = next(st for st in build_plan(s).steps if st.stage == "recycle")
    assert step.why_skipped and "min_p_minus_for_recycle" in step.why_skipped
    assert not step.runnable


def test_stages_can_be_restricted_and_unlisted_stages_never_appear(tmp_path):
    s = _spec(tmp_path, stages=["screen", "subset"])
    assert [st.stage for st in build_plan(s).steps] == ["screen", "subset"]


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------
def test_two_experiments_share_no_artifact(tmp_path):
    a = _spec(tmp_path, name="exp_a")
    b = _spec(tmp_path, name="exp_b")
    for attr in ("run_dir", "cache_path", "verdict_path", "manifest_path",
                 "pminus_csv", "audit_dir"):
        assert getattr(a, attr) != getattr(b, attr), attr
    assert a.run_dir.name == "exp_a" and b.run_dir.name == "exp_b"


def test_a_shard_gets_its_own_cache_so_concurrent_jobs_cannot_collide(tmp_path):
    plain = _spec(tmp_path, name="x")
    sharded = _spec(tmp_path, name="x", execution={
        "run_root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache"),
        "shard": "task-07"})
    assert sharded.cache_path != plain.cache_path
    assert sharded.verdict_path != plain.verdict_path
    assert "task-07" in sharded.cache_path.name and sharded.cache_path.parent.name == "shards"


# ---------------------------------------------------------------------------
# stage completion and resume
# ---------------------------------------------------------------------------
def test_a_stage_is_not_recorded_done_without_its_artifact(tmp_path):
    """A marker written before validation turns a timed-out stage into a completed one."""
    store = StageStore(tmp_path / "run")
    with pytest.raises(RuntimeError, match="not recording it as done"):
        store.complete("screen", [tmp_path / "run" / "absent.json"], {"k": 1})
    assert not store.path("screen").exists()


def test_a_recorded_stage_whose_artifact_vanished_is_rerun(tmp_path):
    s = _spec(tmp_path)
    s.run_dir.mkdir(parents=True)
    art = s.run_dir / "pminus_pilot.json"
    art.write_text(json.dumps({"p_minus": 9, "p_minus_bids": []}))
    store = StageStore(s.run_dir)
    store.complete("screen", [art], s.fingerprint())
    assert next(st for st in build_plan(s).steps if st.stage == "screen").state.valid
    art.unlink()
    step = next(st for st in build_plan(s).steps if st.stage == "screen")
    assert step.state.done and step.state.missing and step.runnable


def test_a_truncated_stage_marker_counts_as_not_done(tmp_path):
    store = StageStore(tmp_path / "run")
    store.dir.mkdir(parents=True)
    store.path("screen").write_text('{"stage": "screen",')          # killed mid-write
    assert store.read("screen") is None


def test_resume_is_refused_when_the_dataset_changed(tmp_path):
    s = _spec(tmp_path)
    s.run_dir.mkdir(parents=True)
    art = s.run_dir / "pminus_pilot.json"
    art.write_text("{}")
    store = StageStore(s.run_dir)
    store.complete("screen", [art], s.fingerprint())
    assert check_resume(s, store) == []
    Path(s.behaviours).write_text("Goal,Category\n\"A different question entirely.\",demo\n")
    problems = check_resume(s, store)
    assert any("behaviours_sha256" in p for p in problems), problems


def test_resume_is_refused_when_the_seed_or_judge_changed(tmp_path):
    """Cache keys derive from the seed, so appending under a new seed mixes two experiments."""
    cfg = yaml.safe_load((ROOT / "configs/recycle_3_31.yaml").read_text())
    p = tmp_path / "sci.yaml"
    p.write_text(yaml.safe_dump(cfg))
    s = _spec(tmp_path, science_config=str(p))
    s.run_dir.mkdir(parents=True)
    art = s.run_dir / "pminus_pilot.json"
    art.write_text("{}")
    store = StageStore(s.run_dir)
    store.complete("screen", [art], s.fingerprint())
    cfg["seed"] = int(cfg.get("seed", 0)) + 1
    p.write_text(yaml.safe_dump(cfg))
    problems = check_resume(s, store)
    assert any("seed" in x for x in problems), problems
    assert any("science_config_sha256" in x for x in problems), problems


def test_editing_a_tool_does_not_block_a_resume(tmp_path):
    """code_revision moving is normal between stages; the dataset and seed moving is not."""
    s = _spec(tmp_path)
    s.run_dir.mkdir(parents=True)
    art = s.run_dir / "pminus_pilot.json"
    art.write_text("{}")
    store = StageStore(s.run_dir)
    fp = s.fingerprint()
    fp["code_revision"] = "0" * 40
    store.complete("screen", [art], fp)
    assert check_resume(s, store) == []


# ---------------------------------------------------------------------------
# the shipped benign example
# ---------------------------------------------------------------------------
def test_the_shipped_example_spec_and_recipe_are_valid_and_need_no_download():
    s = ExperimentSpec.load(ROOT / "experiments/example_local.yaml")
    assert s.recipe and not s.behaviours
    r = yaml.safe_load((ROOT / s.recipe).read_text())
    assert r["source"]["kind"] == "local", "the example must not need the network"
    assert (ROOT / "datasets/example_local/requests.json").exists()
    assert validate(s) == [] or all("example_local.csv" in p for p in validate(s))
    plan = build_plan(s)
    assert [st.stage for st in plan.steps] == ["prepare", "screen", "subset"]
    assert plan.writes == []


def test_stop_at_is_portable(monkeypatch):
    """The shell queues used BSD `date -j -f`, which does not exist on a Linux compute node."""
    from recyclebench.experiment import _past_stop
    assert _past_stop(None) is False
    assert _past_stop("23:59") is False
    assert _past_stop("00:00") is True
    with pytest.raises(SpecError, match="HH:MM"):
        _past_stop("half past two")


def test_every_stage_in_the_canonical_order_is_reachable(tmp_path):
    s = _spec(tmp_path, stages=list(STAGES))
    assert [st.stage for st in build_plan(s).steps] == list(STAGES)
