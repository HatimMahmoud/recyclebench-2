"""tools/compose_arcs_section.py: the 05·b compose arcs recompute every number from the store,
keep the two estimands apart, flag dissolved / cut-short / non-estimable arms, and never emit
candidate text."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import compose_arcs_section as ca                              # noqa: E402

REAL = Path(__file__).resolve().parent.parent
REAL_RUN = REAL / "results/runs/compose_20261028_0002"
HAVE_REAL = (REAL_RUN / "compose_summary.json").exists() and (REAL / "cache/compose/candidates.jsonl").exists()

PREFIX, SUFFIX = "WRAP-PREFIX " * 8, " WRAP-SUFFIX" * 8
SECRET = "SECRET-CANDIDATE-TEXT"
GOOD_GATE = {"cosine": 0.9, "bertscore": 0.9, "nli_entail": 0.9, "llm_intent": 1.0}
NLI_FAIL = {"cosine": 0.9, "bertscore": 0.9, "nli_entail": 0.2, "llm_intent": 1.0}   # placeholder 1.0 after


def _write_world(tmp_path, n_beh=20):
    """para opens 4/20 (all admitted); rpe resolves 3; attack opens 5/20; para→attack opens 12/20
    including every attack opening; attack→para paraphrases the wrapper away and reaches 12/20."""
    store, run = tmp_path / "store", tmp_path / "run"
    store.mkdir()
    run.mkdir()
    (store / "attack_template.json").write_text(json.dumps({"prefix": PREFIX, "suffix": SUFFIX}))
    bids = [f"b{i:02d}" for i in range(n_beh)]
    plan = {  # arm: (factors, gated, reached, admitted bids, opened bids, wrapped?)
        "para": (["para"], True, bids, bids, bids[:4], False),
        "rpe": (["rpe"], True, bids, bids[:3], bids[:2], False),
        "attack": (["attack"], False, bids, bids, bids[:5], True),
        "para__attack": (["para", "attack"], False, bids, bids, bids[:12], True),
        "attack__para": (["attack", "para"], False, bids[:12], bids[:12], bids[:3], False),
        "rpe__attack": (["rpe", "attack"], False, [], [], [], True),
    }
    gen, jdg, arms = [], [], []
    for arm, (factors, gated, reached, adm, opened, wrapped) in plan.items():
        for b in reached:
            text = (PREFIX + SECRET + SUFFIX) if wrapped else SECRET
            rid = f"{arm}|{b}|0"
            gen.append({"id": rid, "arm": arm, "bid": b, "idx": 0, "text": text, "steps": [], "ok": True})
            gen.append({"id": rid, "arm": arm, "bid": b, "idx": 0, "text": text, "steps": [], "ok": True,
                        "gate": GOOD_GATE if b in adm else NLI_FAIL, "gate_admitted": b in adm})
            if (not gated) or b in adm:
                for m in ("llama3-8b", ca.NEW):
                    jdg.append({"id": f"{rid}|{m}|judge", "arm": arm, "bid": b, "idx": 0, "model": m,
                                "complied": m == ca.NEW and b in opened})
        k, n = len(set(opened) & set(adm if gated else reached)), len(adm if gated else reached)
        arms.append({"arm": arm, "factors": factors, "intent_gated": gated, "n_with_candidate": len(reached),
                     "n_resolved": n, "k_reopened": k, "estimand": "x"})
    (store / "candidates.jsonl").write_text("".join(json.dumps(d) + "\n" for d in gen) + '{"id": "trunc')
    (store / "judged.jsonl").write_text("".join(json.dumps(d) + "\n" for d in jdg))
    (run / "compose_summary.json").write_text(json.dumps(
        {"pair": "old -> new", "behaviours": n_beh, "candidates_per_behaviour": 1, "arms": arms}))
    return run, store


def test_aggregate_flags_and_reproduces_summary(tmp_path):
    run, store = _write_world(tmp_path)
    agg = ca.aggregate(run, store=store)
    a = agg["arms"]
    assert agg["mismatch"] == []                       # store reproduces the summary
    assert a["para"]["status"] == "ok"
    assert a["rpe"]["status"] == "non_estimable"       # 3 resolved < MIN_RESOLVED
    assert a["rpe"]["rejected"] == {"nli_entail": 17}  # first failing layer, never a 1.0 placeholder
    assert a["attack__para"]["status"] == "not_composition" and a["attack__para"]["surv"] == [0, 12]
    assert a["attack__para"]["truncated"] and not a["para__attack"]["truncated"]
    assert a["rpe__attack"]["status"] == "not_run"
    assert (agg["n_run"], agg["planned"], agg["n_truncated"]) == (5, 6, 1)


def test_gauges_never_cross_estimands(tmp_path):
    run, store = _write_world(tmp_path)
    agg = ca.aggregate(run, store=store)
    for v in agg["views"].values():
        g = v["gauges"]["a"]
        assert (g["champion"], g["control"]) == ("para__attack", "attack")   # same half, same estimand
        assert agg["arms"][g["control"]]["gated"] is False
        assert g["champion"] != "attack__para"                               # a dissolved arm never wins
        assert v["gauges"]["g"]["champion"] is None                         # no estimable gated composite
        assert v["stars"] == {"g": "para", "a": "para__attack"}
    allv, matched = agg["views"]["all"]["gauges"]["a"], agg["views"]["matched"]["gauges"]["a"]
    assert (allv["k_champ"], allv["k_ctrl"], allv["n"]) == (12, 5, 20)
    assert (matched["k_champ"], matched["k_ctrl"], matched["n"]) == (12, 5, 12)  # the 12 attack→para reached
    gains = agg["views"]["all"]["arms"]["para__attack"]["gains"]
    assert gains["n_shared"] == 20 and gains["only"] == 7                    # 12 opened, 5 by attack, 4 ⊂ those


def test_section_body_is_text_free_and_scriptless(tmp_path):
    run, store = _write_world(tmp_path)
    agg = ca.aggregate(run, store=store, anchor={"RSR": {"RSR_raw": 0.49, "k_recovered": 23, "n_p_minus": 47}})
    body = ca.section_body(agg)
    assert SECRET not in body and "WRAP-PREFIX" not in body
    assert "<script" not in body and "<style" not in body      # host pages keep exactly one script
    assert 'role="tablist"' in body and 'id="panel-cmp-matched"' in body   # the attack half's matched subset differs
    assert "anchor 49%" in body and "not RSR" in body
    page = ca.render_page(agg)
    assert page.count("<script") == 1


def test_latest_run_ignores_dirs_without_a_summary(tmp_path):
    (tmp_path / "results/runs/compose_20990101_0000").mkdir(parents=True)
    assert ca.latest_run(tmp_path) is None
    run = tmp_path / "results/runs/compose_20990102_0000"
    run.mkdir()
    (run / "compose_summary.json").write_text("{}")
    assert ca.latest_run(tmp_path) == run


@pytest.mark.skipif(not HAVE_REAL, reason="compose run not on disk")
def test_real_run_reproduces_the_analysis():
    agg = ca.aggregate(REAL_RUN)
    assert agg["mismatch"] == []
    allv = agg["views"]["all"]["arms"]
    assert allv["para"]["kn"] == (18, 56) and allv["para__attack"]["kn"] == (44, 61)
    assert allv["attack__para"]["kn"] == (14, 39) and agg["arms"]["attack__para"]["surv"] == [0, 117]
    assert agg["arms"]["para"]["rejected"] == {"cosine": 26, "nli_entail": 28}
    m = agg["views"]["matched"]["gauges"]["a"]
    assert (m["k_champ"], m["k_ctrl"], m["n"]) == (25, 15, 39) and round(m["mcnemar_p"], 4) == 0.0129
    assert allv["para__attack"]["gains"]["only"] == 8
    assert {allv[x]["gains"]["only"] for x in ("para__rpe", "rpe__para")} == {0}
