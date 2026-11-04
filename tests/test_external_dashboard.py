"""tools/external_dashboard.py: the external-corpus page renders from a synthetic matrix + canonical report, keeps one
script, carries no text, and states its own limits (raw, undefined A_gate, Maverick dropped)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import external_dashboard as ed                                 # noqa: E402
import external_suite as es                                     # noqa: E402

MODELS = [m for m in es.MODELS if m != "llama4-maverick"]


def _tree(tmp_path):
    run = tmp_path / es.RUN_ROOT
    run.mkdir(parents=True)
    models = {m: {"judged": 240, "complied": 40 + i, "missing_responses": 0, "missing_verdicts": 0}
              for i, m in enumerate(MODELS)}
    models["llama4-maverick"] = {"judged": 0, "complied": 0, "missing_responses": 720, "missing_verdicts": 0}
    rep = {"config": es.BASE_CONFIG, "cache": es.CANONICAL_CACHE, "judge": "rubric=qwen2.5:14b",
           "corpora": {"sciencesafety": {"n_behaviours": 240, "models": models,
                                         "fingerprint": {"rows": 240, "categories": 6, "goals_sha256": "b3" * 32,
                                                         "file_sha256": "7c" * 32}}}}
    (run / "canonical_x.json").write_text(json.dumps(rep), encoding="utf-8")
    cells = []
    for old, new, group, design in es.PAIRS:
        base = {"corpus": "sciencesafety", "old": old, "new": new, "group": group, "design": design,
                "pminus": 6, "persistent": 40, "neg_flip": 10, "both_safe": 184}
        cells.append({**base, "arm": "canonical", "status": es.C, "why": "ok"})
        cells.append({**base, "arm": "recycle", "status": es.C, "why": "ok", "p_minus_rejudged": 6,
                      "manifest": {"path": "missing.jsonl", "behaviours": 6, "variants": 48, "admitted": 30},
                      "rsr": {"n_p_minus": 5, "k_recovered": 4, "n_unresolved": 0, "defined": True, "RSR_raw": 0.8,
                              "mcnemar_p": 0.06, "denominator": "p_minus"}})
        cells.append({**base, "arm": "rpe", "status": es.C, "why": "ok", "candidates": 48,
                      "admission": {"gate": {"k": 0, "n": 0, "unresolved": 6}, "loop": {"k": 5, "n": 6, "unresolved": 0},
                                    "both": {"k": 0, "n": 0, "unresolved": 6}}})
    matrix = {"generated": "now", "report": f"{es.RUN_ROOT}/canonical_x.json", "recycle_floor": 5,
              "confirmatory_floor": 30, "data": {"sciencesafety": {"status": es.C, "why": "ok"}}, "cells": cells,
              "style_arcs": {"sciencesafety": {"files": []}}, "static": []}
    mp = run / "matrix_x.json"
    mp.write_text(json.dumps(matrix), encoding="utf-8")
    return mp


def test_page_renders_counts_only_and_states_its_limits(tmp_path, monkeypatch):
    mp = _tree(tmp_path)
    monkeypatch.setattr(ed, "ROOT", tmp_path)
    out = tmp_path / "page.html"
    assert ed.main(["--matrix", str(mp), "--no-category", "--revision", "abc1234", "--out", str(out)]) == 0
    h = out.read_text(encoding="utf-8")
    assert h.count("<script") == 1
    assert "Maverick" in h and "DROPPED BY DECISION" in h
    assert "undefined (n = 0)" in h                     # A_gate with no admitted reconstruction is never 0
    assert "exploratory" in h and "corrected" not in h.lower()
    assert "llama4-maverick" not in h                   # its pairs are not shown as estimated


def test_llama_guard_section_appears_with_its_report(tmp_path, monkeypatch):
    mp = _tree(tmp_path)
    monkeypatch.setattr(ed, "ROOT", tmp_path)
    m = MODELS[0]
    g = {"prereg": "PREREG_SCIENCESAFETY_LLAMAGUARD_20261104.md", "judge_primary": "rubric=qwen2.5:14b",
         "judge_second": "llamaguard=llama-guard3:8b", "missing": 0, "unparsed": 0, "hazards": {"S2": 7},
         "E1": {m: {"attempt_unsafe": 30, "attempt_safe": 10, "refusal_unsafe": 2, "refusal_safe": 100, "kappa": 0.75}},
         "E2": {m: {"primary": 40, "llamaguard": 33, "both": 30, "n": 240}},
         "E3": {f"{m}->llama2-13b": {"p_minus": 6, "confirmed": 4, "attempt_only": 2}},
         "E4": {f"{m}->llama2-13b": {"n_confirmed_with_rewordings": 4, "reopened_primary": 3, "reopened_both": 2}}}
    gp = mp.parent / "llamaguard_sciencesafety_x.json"
    gp.write_text(json.dumps(g), encoding="utf-8")
    out = tmp_path / "page.html"
    assert ed.main(["--matrix", str(mp), "--no-category", "--llamaguard", str(gp), "--out", str(out)]) == 0
    h = out.read_text(encoding="utf-8")
    assert 'id="s-guard"' in h and 'href="#s-guard"' in h and "Harm-confirmed" in h and h.count("<script") == 1


def test_refuses_to_overwrite(tmp_path, monkeypatch):
    mp = _tree(tmp_path)
    monkeypatch.setattr(ed, "ROOT", tmp_path)
    out = tmp_path / "page.html"
    out.write_text("x")
    try:
        ed.main(["--matrix", str(mp), "--no-category", "--out", str(out)])
    except SystemExit as e:
        assert "refusing to overwrite" in str(e)
    else:
        raise AssertionError("overwrote an existing page")


def test_text_section_shows_fixes_regressions_and_the_rewording_that_reopened():
    from types import SimpleNamespace as NS
    beh = [NS(bid="b0", category="bio", canonical_prompt="REQ-FIX"), NS(bid="b1", category="chem", canonical_prompt="REQ-REG"),
           NS(bid="b2", category="phy", canonical_prompt="REQ-HELD")]
    old = NS(canonical={"b0": 1, "b1": 0, "b2": 1},
             draws={"b0": [("OLD-ANS", 1)], "b1": [("OLD-REF", 0)], "b2": [("OLD-ANS2", 1)]})
    new = NS(canonical={"b0": 0, "b1": 1, "b2": 0},
             draws={"b0": [("NEW-REF", 0)], "b1": [("NEW-ANS", 1)], "b2": [("NEW-REF2", 0)]})
    cell = {"corpus": "sciencesafety", "old": "llama3-8b", "new": "llama31-8b", "group": "dense", "design": "x",
            "pminus": 2, "persistent": 0, "neg_flip": 1, "both_safe": 0}
    d = {"canonical": {"obs": {"llama3-8b": old, "llama31-8b": new}, "behaviours": beh},
         "matrix": {"recycle_floor": 1},
         "cells": [{**cell, "arm": "canonical"}, {**cell, "arm": "recycle", "rsr": {}}],
         "reworded": {("llama3-8b", "llama31-8b"): {
             "REQ-FIX": [{"style": "lexical", "text": "REWORD-OK", "success": 1, "reply": "NEW-ANS-REWORD"}],
             "REQ-HELD": [{"style": "syntactic", "text": "REWORD-NO", "success": 0, "reply": "NEW-REF-REWORD"}]}}}
    h = ed.sec_text(d)
    for s in ("REQ-FIX", "OLD-ANS", "NEW-REF", "REWORD-OK", "NEW-ANS-REWORD", "REQ-REG", "NEW-ANS", "REWORD-NO"):
        assert s in h
    assert 'data-outcome="reopened"' in h and 'data-outcome="regression"' in h and 'data-outcome="fix"' in h
    assert "re-opened · 1/1" in h and "held · 0/1" in h
