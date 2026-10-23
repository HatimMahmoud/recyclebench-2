"""Tests for the independent (RPE) probe arm.

The bug these pin: a `run-real` manifest carries frozen variants and gate audits
but no recovery verdict, and the parser used to default every behaviour to 0.
That silently turned the paraphrase side of the convergent-validity comparison
into a constant, so every kappa/agreement figure was computed against all-zeros.
Missing verdicts must read as *unresolved*, never as "held".
"""
from __future__ import annotations

import json
import pytest

from recyclebench.independent_probe.analyze_agreement import analyze
from recyclebench.independent_probe.contract import ProbeItem, parse_recycle_manifest


def _write(tmp_path, name, records):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


def test_run_real_manifest_yields_no_verdicts_and_warns(tmp_path):
    """The real schema: type=admitted (frozen variants) + type=audit (gate scores)."""
    man = _write(tmp_path, "manifest_sorrybench.jsonl", [
        {"type": "admitted", "bid": "jbb-0000", "variants": [
            {"bid": "jbb-0000", "text": "reworded one", "transform_class": "syntactic",
             "surface_distance": 0.6, "index": 1, "intent_preserved": True}]},
        {"type": "audit", "bid": "jbb-0000", "index": 1, "transform_class": "syntactic",
         "intent_preserved": True, "admitted": True, "cosine": 0.9, "bertscore": 0.9,
         "nli_entail": 0.8, "llm_intent": 1.0, "human_label": None},
    ])
    with pytest.warns(RuntimeWarning, match="no recovery verdict field"):
        got = parse_recycle_manifest(man)
    assert got == {}, "a verdict-free manifest must not fabricate verdicts"


def test_missing_verdict_is_unresolved_never_held(tmp_path):
    """A row without a verdict field must not be scored as 0 (held)."""
    man = _write(tmp_path, "m.jsonl", [
        {"bid": "b1", "recovered": 1},          # real verdict present
        {"bid": "b2", "note": "no verdict here"},
    ])
    got = parse_recycle_manifest(man)
    assert got["b1"] == 1
    assert got["b2"] is None, "unknown must stay unresolved, not become a held behaviour"


def test_verdicts_or_reduce_across_rows(tmp_path):
    man = _write(tmp_path, "m.jsonl", [
        {"bid": "b1", "bypassed": 0},
        {"bid": "b1", "bypassed": 1},           # any variant bypassing => recovered
        {"bid": "b2", "bypassed": 0},
        {"bid": "b2", "bypassed": 0},
    ])
    got = parse_recycle_manifest(man)
    assert got == {"b1": 1, "b2": 0}


def test_missing_file_is_empty(tmp_path):
    assert parse_recycle_manifest(tmp_path / "nope.jsonl") == {}


def _item(bid, para, rpe):
    it = ProbeItem(bid=bid, benchmark="sorrybench", prompt="p", old_response="r")
    it.para_recovered, it.rpe_recovered = para, rpe
    return it


def test_analyze_cells_and_statistics():
    items = [
        _item("a", 1, 1), _item("b", 1, 1),     # both recovered
        _item("c", 0, 0),                        # both held
        _item("d", 0, 1), _item("e", 0, 1),     # rpe only
        _item("f", 1, 0),                        # para only
        _item("g", 1, None),                     # unresolved on the rpe side
    ]
    rep = analyze(items)
    assert rep.n_total == 7
    assert rep.n_both_resolved == 6
    assert sorted(rep.both_recovered) == ["a", "b"]
    assert rep.both_held == ["c"]
    assert sorted(rep.rpe_only) == ["d", "e"]
    assert rep.para_only == ["f"]
    assert rep.rpe_unresolved == 1
    assert rep.agreement_rate == pytest.approx(3 / 6)
    assert rep.mcnemar_p is not None and 0.0 <= rep.mcnemar_p <= 1.0
    assert "McNemar" in rep.summary()


def test_analyze_all_unresolved_is_undefined_not_zero():
    """Empty resolved set => rates undefined. Never report 0."""
    rep = analyze([_item("a", None, None), _item("b", None, None)])
    assert rep.rsr_para is None and rep.rsr_rpe is None
    assert rep.n_both_resolved == 0
    assert rep.mcnemar_p is None
