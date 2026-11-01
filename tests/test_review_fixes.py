"""Fixes from the external review of 2026-10-31: RQ3 estimator, confirmatory separation, config stamp, gate re-check,
loosened-gate rule."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from recyclebench.drivers import fit_drivers_gee, fit_drivers_vb, separated_classes   # noqa: E402

FORMULA = "success ~ C(transform_class, Treatment(reference='paraphrase')) + surface_distance"


def _trials(seed=1, n_clusters=60, boost="register"):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        u = rng.normal(0, 0.5)                                     # behaviour-level heterogeneity
        for cls in ("paraphrase", "register", "syntactic", "lexical"):
            for _ in range(3):
                lin = -0.8 + u + (1.6 if cls == boost else 0.0)
                rows.append(dict(cluster=f"b{c}", transform_class=cls, surface_distance=rng.uniform(0.2, 0.9),
                                 success=int(rng.random() < 1 / (1 + np.exp(-lin)))))
    return pd.DataFrame(rows)


def test_gee_finds_a_planted_driver_and_reports_holm_within_classes():
    g = fit_drivers_gee(_trials(), FORMULA)
    assert g["method"].startswith("gee") and g["n_clusters"] == 60
    assert g["omnibus_class"]["df"] == 3 and g["omnibus_class"]["p_value"] < 0.05
    t = {r["term"]: r for r in g["table"]}
    assert t["transform_class:register"]["driver"] is True
    assert t["transform_class:register"]["or_ci"][0] > 1
    assert "p_holm" not in t["surface_distance"]                  # covariates are outside the Holm family


def test_gee_without_effect_declares_no_driver():
    g = fit_drivers_gee(_trials(seed=3, boost="none"), FORMULA)
    assert not any(r.get("driver") for r in g["table"])


def test_separated_class_leaves_the_model_and_is_reported():
    df = _trials()
    extra = pd.DataFrame([dict(cluster=f"b{i}", transform_class="back_translation", surface_distance=0.5, success=0)
                          for i in range(8)])
    df = pd.concat([df, extra], ignore_index=True)
    assert separated_classes(df) == {"back_translation": "0/8"}
    g = fit_drivers_gee(df, FORMULA)
    assert g["separated"] == {"back_translation": "0/8"}
    assert not any("back_translation" in r["term"] for r in g["table"])
    vb = fit_drivers_vb(df, FORMULA)
    assert vb["separated"] == {"back_translation": "0/8"} and all("p_value" not in r for r in vb["table"])
    assert all(r["or_cri"][0] <= r["odds_ratio"] <= r["or_cri"][1] for r in vb["table"])


def test_master_never_pools_the_confirmatory_run():
    import master_dashboard as md
    pr = dict(complete=True, recon=dict(k=5, n=10))
    run = lambda i, conf, ds: SimpleNamespace(spec=SimpleNamespace(id=i, confirmatory=conf, behaviour_scope="full", dataset=ds),
                                             pairs={("a", "b"): dict(pr)})
    s = SimpleNamespace(recycle=[run("x", False, "sorrybench"), run("held", True, "heldout")])
    cells = md.recycle_cells(s)
    assert ("a", "b", "sorrybench") in cells and ("a", "b", "heldout") not in cells
    assert md.pooled_offline(s)[:2] == (5, 10)


def test_config_stamp_detects_a_cache_key_collision(tmp_path, monkeypatch):
    import config_stamp as cs
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    block = lambda mt: {"provider": "openai_compatible", "model_id": "llama3:8b", "temperature": 0.0, "max_tokens": mt}
    for name, mt in (("a.yaml", 512), ("b.yaml", 512), ("c.yaml", 256)):
        (cfgs / name).write_text(json.dumps({"seed": 1, "lineage": ["llama3-8b"], "real": {"llama3-8b": block(mt)}}))
    monkeypatch.setattr(cs, "ROOT", tmp_path)
    monkeypatch.setattr(cs, "ollama_facts", lambda mids: {"ollama_version": "x", "models": {m: {"weights_blob": "w"} for m in mids}})
    st = cs.stamp()
    assert set(st["cache_key_collisions"]) == {"llama3-8b"}                     # c.yaml differs in max_tokens
    assert len(set(st["cache_key_collisions"]["llama3-8b"].values())) == 2
    assert cs.verify(st, st) == []
    changed = json.loads(json.dumps(st))
    changed["ollama"]["models"]["llama3:8b"]["weights_blob"] = "v"
    assert any("weights" in d for d in cs.verify(changed, st))


def test_gate_recheck_scores_anchoring_against_the_blind_baseline(tmp_path):
    import gate_recheck as gr
    (tmp_path / "recheck_strata.json").write_text(json.dumps(dict(n_B=2, n_G0=2, n_G1=2)))
    key = [dict(row=str(i), stratum=st, original_row=str(100 + i), original_label=ol, model_guess=mg, audit_stratum=a)
           for i, (st, ol, mg, a) in enumerate([("B", "1", "", "A"), ("B", "0", "", "R"), ("G0", "0", "0", "R"),
                                               ("G0", "1", "0", "A"), ("G1", "1", "1", "R"), ("G1", "1", "1", "A")], 1)]
    with (tmp_path / "recheck_key.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=gr.KEY_COLS)
        w.writeheader()
        w.writerows(key)
    new = ["1", "0", "1", "1", "0", "1"]                          # baseline agrees fully; two model-assisted rows flip
    with (tmp_path / "recheck_sheet.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=gr.SHEET_COLS)
        w.writeheader()
        w.writerows([dict(row=str(i), original_request="o", rewrite="r", human_intent_preserved=v, note="") for i, v in enumerate(new, 1)])
    r = gr.score(tmp_path)
    assert r["strata"]["B"]["agreement"]["k"] == 2
    assert r["anchoring_excess_disagreement"] == pytest.approx(0.5)            # 2 of 4 changed vs 0 of 2
    assert r["toward_guess"] == 2                                               # both originals had matched the guess
    assert r["reopening_rows_relabelled"] == dict(n=3, changed_request_now=2, changed_request_before=2)


def test_loosening_only_adds_admissions_and_keeps_lexical_rejections():
    import gate_loosen as gl
    thr = SimpleNamespace(cosine=0.75, bertscore=0.85, nli_entail=0.60)
    lay = {"x": dict(layers=dict(cosine=0.72, bertscore=0.9, nli_entail=0.7)),
           "y": dict(layers=dict(cosine=0.62))}
    adm = dict(id="a", admitted=True, rejected_by=None)
    x = dict(id="x", admitted=False, rejected_by="cosine")
    y = dict(id="y", admitted=False, rejected_by="cosine")
    lex = dict(id="l", admitted=False, rejected_by="lexical_divergence")
    assert gl.admitted_at(adm, lay, thr, 0.0) and gl.admitted_at(adm, lay, thr, 0.10)
    assert not gl.admitted_at(x, lay, thr, 0.0)
    assert gl.admitted_at(x, lay, thr, 0.05) and gl.admitted_at(x, lay, thr, 0.10)
    assert not gl.admitted_at(y, lay, thr, 0.10)                               # short-circuited below the loosest cut
    assert not gl.admitted_at(lex, lay, thr, 0.10)
    assert gl.admitted_at(dict(id="z", admitted=False, rejected_by="nli_entail"), lay, thr, 0.10) is None


def test_robustness_finding_headline_follows_the_checks(monkeypatch):
    import master_dashboard as md
    cell = lambda k: {"verdict_agreement": {"k": k, "n": 20, "rate": k / 20}}
    arm = lambda v, c: {"resolved": {"corrected": c, "verdict": v}}
    agg = dict(staleness={"cells": {"a|S": cell(20), "b|O": cell(20)}},
               quant={"arm_A_q4_0": arm("materially surface-bound", 0.53), "arm_B_q4_k_m": arm("materially surface-bound", 0.58),
                      "p_minus_overlap": {"both": 36, "only_A": 13, "only_B": 5}}, loosen=None)
    monkeypatch.setattr(md.mth, "aggregate", lambda root: agg)
    assert "survives" in md._robustness_finding() and "36 of 54 fixes" in md._robustness_finding()
    agg["quant"]["arm_B_q4_k_m"]["resolved"]["verdict"] = "inconclusive"
    assert "qualify" in md._robustness_finding()
    agg["quant"]["arm_B_q4_k_m"]["resolved"]["verdict"] = "materially surface-bound"
    agg["staleness"]["cells"]["b|O"] = cell(17)                                  # 85% < the registered 90%
    t = md._robustness_finding()
    assert "qualify" in t and "below the registered 90%" in t
    monkeypatch.setattr(md.mth, "aggregate", lambda root: dict(staleness=None, quant=None, loosen=None))
    assert md._robustness_finding() == ""


def test_rq4_reading_places_each_model_by_interval_overlap():
    import controls_section as cs
    t = cs._rq4_reading({"llama32-1b": "below", "llama3-8b": "at", "llama2-7b": "below"})
    assert "at the model's own over-refusal rate for Llama 3 8B" in t
    assert "below it for Llama 2 7B, Llama 3.2 1B" in t                         # listed in lineage order
    assert cs._rq4_reading({}) == ""


def test_overrefusal_pair_table_shows_release_pairs_and_flags_a_rise():
    import controls_section as cs
    rate = lambda k, n: dict(k=k, n=n, rate=k / n, ci=[max(0.0, k / n - 0.05), min(1.0, k / n + 0.05)])
    models = {m: dict(safe_refusal_judged=rate(50, 250), safe_refusal_offline=dict(rate=0.1), unsafe_refusal_judged=rate(196, 200),
                      unsafe_refusal_offline=dict(rate=0.9), safe_by_type={}, safe_missing=0, unsafe_missing=0)
              for m in ("llama31-8b", "llama32-1b", "llama32-3b", "mistral7b-v02", "mistral7b-v03")}
    pr = lambda o, n, b, c: dict(older=o, newer=n, n=250, older_rate=0.2, newer_rate=0.2 + (b - c) / 250, newly_refused=b,
                                 newly_answered=c, mcnemar_p=0.0)
    d = dict(models=models, pairs=[pr("llama32-3b", "mistral7b-v02", 16, 16), pr("mistral7b-v02", "mistral7b-v03", 13, 14),
                                   pr("llama31-8b", "llama32-1b", 42, 10)])
    h = cs._or_panel(dict(overrefusal=dict(data=d, complete=[m for m in cs.ORDER if m in models], partial=[])))
    assert "Mistral 7B v0.2 → Mistral 7B v0.3" in h and "Llama 3.2 3B → Mistral" not in h
    assert "does refuse more benign prompts in Llama 3.1 8B → Llama 3.2 1B" in h and "Mistral 7B v0.2 → Mistral 7B v0.3 (" not in h


def test_robustness_panel_reads_the_three_checks():
    import method_section as ms
    r = lambda k, n, c, v: dict(k=k, n=n, corrected=c, corrected_ci=[c - 0.2, c + 0.2], verdict=v)
    agg = dict(staleness={"cells": {"a|S": {"verdict_agreement": {"k": 20, "n": 20, "rate": 1.0}, "identical": {"k": 19, "n": 20}}}},
               quant=dict(n=440, arm_A_q4_0=dict(size=49, resolved=r(23, 49, 0.527, "materially surface-bound")),
                          arm_B_q4_k_m=dict(size=41, resolved=r(21, 41, 0.578, "materially surface-bound")),
                          p_minus_overlap=dict(both=36, only_A=13, only_B=5, jaccard=0.6667),
                          old_answered=dict(q4_0=92, q4_k_m=85, mcnemar_p=0.189)),
               loosen=dict(by_delta={"0.0": dict(admitted=463, conditions=dict(any=dict(reopened=31, resolved=49, corrected=0.72))),
                                     "0.1": dict(admitted=507, conditions=dict(any=dict(reopened=33, resolved=49, corrected=0.77)))}))
    h = ms._robustness_panel(agg)
    assert "appear as the runs finish" not in h
    assert "robust to the mismatch" in h and "Jaccard 0.67" in h and "strict: 31/49" in h and "δ 0.10: 33/49" in h
    assert "appear as the runs finish" in ms._robustness_panel(dict(staleness=None, quant=None, loosen=None))
