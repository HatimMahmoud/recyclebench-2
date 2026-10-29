"""tools/study_data.py + tools/master_dashboard.py: integrity checks, the gate short-circuit, the
design-weighted judge calibration, the RPE re-join, and graceful 'not yet finished' rendering."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import master_dashboard as md                                  # noqa: E402
import study_data as sd                                        # noqa: E402
from recyclebench.audit import invert_stratified               # noqa: E402
from recyclebench.manifest import GateAuditRecord, Manifest    # noqa: E402


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """An empty repository root: every loader must cope with missing files."""
    monkeypatch.setattr(sd, "ROOT", tmp_path)
    sd._CACHE_MEMO.clear()
    yield tmp_path
    sd._CACHE_MEMO.clear()


def test_empty_world_renders_not_yet_finished(world):
    study = sd.build_study(None)
    html = md.render(study)
    assert md.NOT_YET in html
    assert "<script>" in html and html.count("<script>") == 1
    assert "Traceback" not in html


def test_classify_rejection_reconstructs_short_circuit():
    thr = sd.STRICT_GATE
    assert sd.classify_rejection({"cosine": .9, "bertscore": .9, "nli_entail": .8}, True, thr) == (None, sd.GATE_ORDER)
    # cosine fails: later layers are placeholders and must not count as scores
    assert sd.classify_rejection({"cosine": .5, "bertscore": 1.0, "nli_entail": 1.0}, False, thr) == ("cosine", ["cosine"])
    assert sd.classify_rejection({"cosine": .9, "bertscore": .9, "nli_entail": .2}, False, thr) == \
        ("nli_entail", ["cosine", "bertscore", "nli_entail"])
    # rejected with every persisted score passing -> the unpersisted lexical-divergence floor
    assert sd.classify_rejection({"cosine": 1.0, "bertscore": 1.0, "nli_entail": 1.0}, False, thr) == ("lexical_divergence", [])


def test_gate_stats_excludes_placeholders_per_style():
    m = Manifest()
    m.audit = [
        GateAuditRecord("b", 0, "back_translation", True, False, cosine=1.0, bertscore=1.0, nli_entail=1.0, llm_intent=1.0),
        GateAuditRecord("b", 1, "back_translation", True, False, cosine=1.0, bertscore=1.0, nli_entail=1.0, llm_intent=1.0),
        GateAuditRecord("b", 2, "back_translation", True, True, cosine=.8, bertscore=.9, nli_entail=.7, llm_intent=1.0),
        GateAuditRecord("b", 3, "lexical", True, False, cosine=.5, bertscore=1.0, nli_entail=1.0, llm_intent=1.0),
    ]
    g = sd.gate_stats(m, sd.STRICT_GATE)
    bt = g["per_class"]["back_translation"]
    assert bt["nli_naive"] == [1.0, 1.0, 0.7]          # what the old arcs averaged
    assert bt["nli_eval"] == [0.7]                      # what the gate actually scored
    assert bt["rej"]["lexical_divergence"] == 2
    assert g["per_class"]["lexical"]["cos_eval"] == [0.5] and g["per_class"]["lexical"]["nli_eval"] == []
    assert not g["intent_ran"]


def test_design_weighted_calibration_matches_invert_stratified():
    # 30 judge-positives (all truly positive), 30 sampled judge-negatives with 5 true positives,
    # drawn from a pool of 30 / 2,338 -- the shape of audit/attack
    rows = [{"bid": f"p{i}", "h": 1, "q": 1, "o": 1} for i in range(30)]
    rows += [{"bid": f"n{i}", "h": int(i < 5), "q": 0, "o": int(i < 9)} for i in range(30)]
    w = sd._weighted(rows, "q", 30, 2338, n_boot=200)
    se_iv, sp_iv = invert_stratified(30, 2338, 1.0, 25 / 30, 30, 30, n_boot=200, seed=11)
    assert w["se"] == pytest.approx(se_iv.point, rel=1e-9)
    assert w["sp"] == pytest.approx(sp_iv.point, rel=1e-9)
    assert w["se"] == pytest.approx(30 / (30 + 2338 * 5 / 30), rel=1e-9)
    assert w["prevalence"] == pytest.approx((30 + 2338 * 5 / 30) / 2368, rel=1e-9)
    o = sd._weighted(rows, "o", 30, 2338, n_boot=50)
    assert o["se"] == pytest.approx(1.0) and o["sp"] == pytest.approx(21 / 25)


def _write_csv(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Goal,Category\n" + "".join(f"Request number {i},1\n" for i in range(n)))


def test_pilot_integrity_checks(world):
    for key, _, csv_path, *_ in sd.DATASET_SPECS:
        if key in ("advbench", "harmbench"):
            _write_csv(world / csv_path, {"advbench": 5, "harmbench": 4}[key])
    run = world / "results/runs/20260804_000000_llama2-13b_to_llama3-8b_full"
    run.mkdir(parents=True)
    (run / "META.txt").write_text("pair      : llama2-13b -> llama3-8b\n")
    stale = {"pair": "llama2-7b -> llama3-8b", "n_behaviours": 3, "breaks_old": 1, "blocked_by_new_among_those": 0, "p_minus": 0}
    (run / "pminus_advbench.json").write_text(json.dumps(stale))
    good = {"pair": "llama2-7b -> llama3-8b", "n_behaviours": 4, "breaks_old": 0, "blocked_by_new_among_those": 0, "p_minus": 0}
    (world / "results" / "pminus_llama2_llama3_harmbench.json").write_text(json.dumps(good))
    ds = sd.load_datasets()
    screens = sd.compute_screens(ds)
    cells = sd.compute_pair_cells(screens, ds)
    pil = {p.path: p for p in sd.collect_pilots(ds, cells, screens)}
    bad = pil["results/runs/20260804_000000_llama2-13b_to_llama3-8b_full/pminus_advbench.json"]
    assert bad.status == "stale-copy"
    assert any("contradicts the run's META" in i for i in bad.issues)
    assert any("n_behaviours=3" in i for i in bad.issues)
    ok = pil["results/pminus_llama2_llama3_harmbench.json"]
    assert ok.status == "valid" and ok.dataset == "harmbench"


def test_identical_pilots_keep_the_consistent_one(world):
    _write_csv(world / "data/wrapped_rs.csv", 3)
    _write_csv(world / "data/advbench.csv", 5)
    blob = {"pair": "llama2-7b -> llama3-8b", "n_behaviours": 3, "breaks_old": 1, "blocked_by_new_among_those": 0, "p_minus": 0}
    (world / "results").mkdir(parents=True, exist_ok=True)
    (world / "results/pminus_wrapped_rs.txt").write_text("NOTE: no 'judges:' configured\n" + json.dumps(blob, indent=2))
    run = world / "results/runs/20260804_000001_llama2-13b_to_llama3-8b_full"
    run.mkdir(parents=True)
    (run / "META.txt").write_text("pair      : llama2-13b -> llama3-8b\n")
    (run / "pminus_advbench.json").write_text(json.dumps(blob))
    ds = sd.load_datasets()
    screens = sd.compute_screens(ds)
    pil = {p.path: p for p in sd.collect_pilots(ds, sd.compute_pair_cells(screens, ds), screens)}
    assert pil["results/pminus_wrapped_rs.txt"].status == "valid"
    copy = pil["results/runs/20260804_000001_llama2-13b_to_llama3-8b_full/pminus_advbench.json"]
    assert copy.status == "stale-copy" and any("pminus_wrapped_rs.txt (the genuine result)" in i for i in copy.issues)


def test_rpe_judge_detection_from_logs(world):
    log = world / "logs/rpe.log"
    log.parent.mkdir(parents=True)
    log.write_text("  [judge] WARNING: no cfg.judges block -> falling back to target-as-judge (circular)\n")
    sp = sd.RPESpec("x", "m.jsonl", "sorrybench", "llama3-8b", "llama31-8b", sd.STRICT_GATE, "strict", log="logs/rpe.log")
    assert sd._rpe_judge(sp).startswith("target-as-judge")
    log.write_text("  [judge] using cfg.judges: ['rubric']\n")
    assert sd._rpe_judge(sp) == "llm (qwen2.5:14b)"


def test_rpe_placeholder_manifest_is_invalid(world):
    p = world / "results/runs/r/rpe/rpe_manifest_advbench.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"bid": "pair", "benchmark": "advbench", "admitted": [], "n_candidates": 1}) + "\n")
    spec = sd.RPESpec("rpe-x", "results/runs/r/rpe/rpe_manifest_advbench.jsonl", "advbench", "llama3-8b", "llama31-8b",
                      sd.STRICT_GATE, "strict")
    old = sd.RPE_SPECS
    try:
        sd.RPE_SPECS = [spec]
        runs = sd.build_rpe_runs({})
    finally:
        sd.RPE_SPECS = old
    assert runs[0].status == "invalid" and not runs[0].items


def test_path_html_is_wrap_safe_and_escaped():
    h = md.path_html("results/runs/a_b/<x>.json")
    assert "&lt;x&gt;.json" in h and "<wbr>" in h and 'title="results/runs/a_b/&lt;x&gt;.json"' in h


REAL = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(not (REAL / "results/runs/lineage_sorrybench_20261020/summary_real.json").exists()
                    or not (REAL / "audit/sorrybench/judge_sheet.csv").exists(),
                    reason="local RecycleBench results not present")
def test_real_results_reproduce():
    sd._CACHE_MEMO.clear()
    study = sd.build_study(None)
    checked = 0
    for rr in study.recycle:
        if rr.spec.judge != "offline" or rr.summary_status not in ("valid", "from-log"):
            continue
        for pair, pr in rr.pairs.items():
            if pr["reported"] is not None:
                assert pr["match"] is True, (rr.spec.id, pair)
                checked += 1
        for m, v in rr.pbr.items():
            if v["reported"] is not None and v["recon"] is not None:
                assert v["match"] is True, (rr.spec.id, m)
                checked += 1
    assert checked >= 20
    # the calibrated anchor is recomputed exactly from the human sheet
    anchor = next(r for r in study.recycle if r.spec.id == "anchor-sorrybench")
    cal = anchor.calibrated["RSR"]
    c = sd.corrected(cal["k_recovered"], cal["n_p_minus"], study.anchor_jc, cal["mean_admitted"])
    # Three recorded values, in order. (a) 0.5503 [0.3629, 0.7377] is the original headline, which
    # weighted the inversion by the SAMPLE strata. (b) 0.5532 [0.3646, 0.7417] is the 2026-10-22
    # design-weighted fix (population strata 26/34; prereg Amendment 1, D6) -- but it still ran on 59
    # bundles, because bundle jbb-0003 carried a mistyped '00' in human_fulfilled and
    # ingest_judge_audit silently drops any bundle with a cell that is not '0' or '1'. (c) the typo was
    # repaired on 2026-10-27 (backup: audit/sorrybench/judge_sheet.csv.bak_before_typo_fix_20261027_1719),
    # recovering that bundle, so the calibration is now over all 60 and se moves 0.8585 -> 0.8621.
    # Hence 0.5509. The judge stays admitted (kappa 0.8326 > the 0.60 floor) and the shift is third-decimal.
    assert c["point"] == pytest.approx(0.5509, abs=5e-4)
    assert (c["lo"], c["hi"]) == pytest.approx((0.3638, 0.7379), abs=5e-4)
    assert abs(c["point"] - cal["RSR_corrected"]) < 5e-3
    pkg = study.calib["sorrybench"]["package"]
    assert (pkg["se"], pkg["sp"], pkg["kappa"]) == pytest.approx((0.8621, 0.9677, 0.8326), abs=1e-4)
    assert pkg["n"] == 60, "the repaired sheet must calibrate on all 60 bundles, none dropped"
    # the stale SMOKE summary is caught, the pre-fix pilots are superseded, the circular RPE is flagged
    smoke = next(r for r in study.recycle if r.spec.id == "l3-harmbench-smoke")
    assert smoke.summary_status == "stale-copy"
    assert {r.spec.id for r in study.rpe if r.status == "circular-judge"} == {"rpe-sorrybench-strict", "rpe-sorrybench-loose"}


@pytest.mark.skipif(not Path(os.path.expanduser("~/Downloads/all_local_results_llama3_1_70b 1.jsonl")).exists()
                    or not (REAL / "results/runs/attack_transfer/llama31-70b_judged.json").exists(),
                    reason="70B responses file not present")
def test_70b_item_verdicts_match_stored_counts():
    sd._CACHE_MEMO.clear()
    items = sd.attack_70b_items(os.path.expanduser("~/Downloads/all_local_results_llama3_1_70b 1.jsonl"))
    stored = json.loads((REAL / "results/runs/attack_transfer/llama31-70b_judged.json").read_text())["classes"]
    for cls, a in items["classes"].items():
        assert a["q_missing"] == 0
        assert (a["n"], a["q"]) == (stored[cls]["n"], stored[cls]["n_broke"]), cls
