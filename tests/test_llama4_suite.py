"""tools/llama4_suite.py: every cell gets the status its artifacts and the protocol decide.

Synthetic run tree, no data and no models: the canonical report, one complete recycle run, one partial run, one
RPE pair, and nothing for attack transfer.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import llama4_suite as s                                        # noqa: E402

OLDS = s.OLDS


def _canonical(root, pm):
    models = {m: {"judged": 400, "complied": 10, "missing_responses": 0, "missing_verdicts": 0}
              for m in OLDS + (s.NEW,)}
    pairs = {f"{o}->{s.NEW}": {"pminus": pm.get(o, 0), "persistent": 1, "neg_flip": 2, "both_safe": 3,
                               "pminus_bids": []} for o in OLDS}
    p = root / s.CANONICAL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"harmbench": {"n_behaviours": 400, "models": models, "pairs": pairs}}))


def _recycle(root, old, stages, pilot_pm=7, rsr=True):
    d = root / s.SUITE / f"llama4_scout__harmbench__{old}"
    (d / "stages").mkdir(parents=True)
    for st in stages:
        (d / "stages" / f"{st}.json").write_text("{}")
    (d / "pminus_pilot.json").write_text(json.dumps({"p_minus": pilot_pm}))
    if rsr:
        (d / f"manifest_x.jsonl").write_text(
            json.dumps({"type": "audit", "bid": "jbb-0000", "transform_class": "paraphrase", "admitted": True}) + "\n")
        (d / "summary_real.json").write_text(json.dumps({"RSR_primary_RAW": [
            {"pair": f"{old} -> {s.NEW}", "n_p_minus": 4, "k_recovered": 2, "n_unresolved": 0, "defined": True,
             "RSR_raw": 0.5}], "PBR_backbone_RAW": []}))


def _cells(res, arm):
    return {c["old"]: c for c in res["cells"] if c["arm"] == arm}


def test_statuses_follow_artifacts_and_floors(tmp_path):
    _canonical(tmp_path, {"llama3-8b": 7, "llama32-3b": 5, "llama31-8b": 4})
    _recycle(tmp_path, "llama3-8b", ["screen", "subset", "recycle", "partition", "audit"])
    _recycle(tmp_path, "llama32-3b", ["screen", "subset"], pilot_pm=5, rsr=False)
    res = s.collect(tmp_path)
    rec = _cells(res, "recycle")
    assert rec["llama3-8b"]["status"] == s.C
    assert rec["llama3-8b"]["rsr"]["wilson"] is not None
    assert rec["llama32-3b"]["status"] == s.P                     # partial: resumable, not complete
    assert rec["llama31-8b"]["status"] == s.NA                    # |P^-| 4 < floor 5
    assert rec["llama2-7b"]["status"] == s.NA
    rpe = _cells(res, "rpe")
    assert rpe["llama2-7b"]["status"] == s.NA                     # empty P^-: undefined, never 0
    assert rpe["llama3-8b"]["status"] == s.P                      # fixes exist, nothing run yet
    assert res["attack"]["status"] == s.P
    assert all(c["status"] == s.C for c in res["cells"] if c["arm"] == "canonical")


def test_rejudged_floor_miss_is_ineligible_not_failed(tmp_path):
    _canonical(tmp_path, {"llama3-8b": 5})
    _recycle(tmp_path, "llama3-8b", ["screen"], pilot_pm=4, rsr=False)
    rec = _cells(s.collect(tmp_path), "recycle")["llama3-8b"]
    assert rec["status"] == s.NA and rec["rejudge_matches_stored"] is False


def test_rpe_scored_on_fewer_behaviours_than_inverted_needs_repair(tmp_path):
    _canonical(tmp_path, {"llama3-8b": 2})
    d = tmp_path / s.SUITE / "rpe" / "harmbench"
    d.mkdir(parents=True)
    (d / f"rpe_manifest_llama3-8b__to__{s.NEW}.jsonl").write_text(
        "\n".join(json.dumps({"bid": b, "details": [{"text": "x"}]}) for b in ("jbb-0001", "jbb-0002")))
    adm = {k: {"k": 0, "n": 0, "unresolved": 1} for k in ("gate", "loop", "both")}
    (d / f"closedloop_llama3-8b__to__{s.NEW}_gated_20261102_0000.json").write_text(
        json.dumps({"n_behaviours": 1, "admission": adm}))
    assert _cells(s.collect(tmp_path), "rpe")["llama3-8b"]["status"] == s.F


def test_markdown_keeps_table_cells_intact(tmp_path):
    _canonical(tmp_path, {})
    md = s.markdown(s.collect(tmp_path))
    rows = [l for l in md.splitlines() if l.startswith("| harmbench")]
    assert rows and all(l.replace("\\|", "").count("|") == 7 for l in rows)
