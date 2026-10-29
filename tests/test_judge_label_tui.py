"""tools/judge_label_tui.py: which guess column is shown, the word-level diff, and the keymap."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import judge_label_tui as t                                  # noqa: E402


def test_guess_column_judge_first_then_model():
    assert t.guess_col(["request", "judge_label", "human_fulfilled"]) == "judge_label"
    assert t.guess_col(["row", "model_guess", "human_intent_preserved"]) == "model_guess"
    assert t.guess_col(["row", "original_request", "rewrite", "human_intent_preserved"]) is None


def test_diff_ignores_case_and_punctuation():
    a, b = t.diff_marks("Write a letter to my neighbour, threatening him.",
                        "write a letter to my NEIGHBOUR threatening him!")
    assert not any(c for _, c in a) and not any(c for _, c in b)
    a, b = t.diff_marks("Explain how to pick a lock", "Describe how someone picks a lock")
    assert [w for w, c in a if c] == ["Explain", "to", "pick"]
    assert [w for w, c in b if c] == ["Describe", "someone", "picks"]


def test_wrap_counts_visible_width_not_colour_codes():
    _, b = t.diff_marks("a b c", "x b y")
    for line, vis in t.wrap_marked(b, 3, t.YELLOW):
        assert vis <= 3 and len(line) > vis              # codes present, not counted


def test_flip_and_accept():
    assert t.decide("f", "1") == ("set", "0") and t.decide("f", "0") == ("set", "1")
    assert t.decide("\r", "1") == ("set", "1") and t.decide("q", "1") == ("quit", None)


def test_gate_audit_score_on_a_synthetic_sheet(tmp_path):
    import json
    import gate_audit as ga
    (tmp_path / "gate_strata.json").write_text(json.dumps({
        "N_R": 2, "N_A": 8, "per_bench": {"b": {"n_pminus": 3}},
        "rec_variants": {"b": {"x1": [0, 1], "x2": [0]}}}))
    (tmp_path / "gate_key.csv").write_text("row,stratum,bench,manifest,bid,index\n1,R,b,m,x1,0\n2,R,b,m,x2,0\n3,A,b,m,y,0\n")
    (tmp_path / "gate_sheet.csv").write_text("row,original_request,rewrite,human_intent_preserved,note\n"
                                             "1,o,r,0,\n2,o,r,0,\n3,o,r,1,\n")
    g = ga.score(tmp_path)
    assert g["contamination_R"]["k"] == 2 and g["contamination_A"]["k"] == 0
    # x1 still re-opens through its unaudited index 1; x2's only re-opening rewording changed the request
    assert g["rsr_after_audit"] == [dict(bench="b", n_pminus=3, reopened=2, reopened_after_audit=1)]
    assert g["model_assisted"] is None
