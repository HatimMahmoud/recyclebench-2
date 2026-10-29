"""The anchor's corrected figures on the repaired 60-bundle calibration, and the matched-scale
post-hoc pairs (llama2-7b -> llama3-8b / -> llama31-8b) in study_data. Real-data checks: they skip
when the local results are absent."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

ANCHOR = ROOT / "results/runs/recycle_3_31_sorrybench"
SHEET = ROOT / "audit/sorrybench/judge_sheet.csv"
BACKUP = ROOT / "audit/sorrybench/judge_sheet.csv.bak_before_typo_fix_20261027_1719"


@pytest.mark.skipif(not (ANCHOR / "corrected_sorrybench.json").exists() or not BACKUP.exists(),
                    reason="anchor results / calibration sheets not on disk")
def test_recalibration_reproduces_the_old_file_then_lands_on_the_pinned_values(tmp_path):
    import recalibrate_corrected as rc
    out = tmp_path / "recal.json"
    assert rc.main(["--corrected", str(ANCHOR / "corrected_sorrybench.json"), "--sheet", str(SHEET),
                    "--reference-sheet", str(BACKUP), "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["judge"]["n_bundles"] == 60
    assert (d["RSR"]["RSR_corrected"], *d["RSR"]["CI"]) == pytest.approx((0.5509, 0.3638, 0.7379), abs=1e-4)
    assert d["PBR_new"]["PBR_corrected"] == pytest.approx(0.4833, abs=1e-4)
    assert d["PBR_new"]["verdict"] == "materially surface-bound"
    on_disk = ANCHOR / "corrected_sorrybench_recal_n60.json"
    if on_disk.exists():                       # the committed file is exactly what the tool writes
        e = json.loads(on_disk.read_text())
        assert (e["RSR"], e["PBR_new"], e["judge"]) == (d["RSR"], d["PBR_new"], d["judge"])


@pytest.mark.skipif(not (ROOT / "results/runs/lineage_sorrybench_20261020/manifest.jsonl").exists()
                    or not (ROOT / "cache/judge_lineage_sorrybench.jsonl").exists(),
                    reason="lineage run / verdict cache not on disk")
def test_matched_scale_posthoc_pairs_are_judged():
    import study_data as sd
    sd._CACHE_MEMO.clear()
    s = sd.build_study(None)
    lin = next(r for r in s.recycle if r.spec.id == "lineage-sorrybench")
    a, b = lin.pairs[("llama2-7b", "llama3-8b")], lin.pairs[("llama2-7b", "llama31-8b")]
    assert a["post_hoc"] and b["post_hoc"] and a["complete"] and b["complete"]
    # the package estimator divides by |P^-| (unresolved fixes count as not re-opened)
    assert (a["reported"]["k_recovered"], a["reported"]["n_p_minus"]) == (5, 6)
    assert (b["reported"]["k_recovered"], b["reported"]["n_p_minus"]) == (13, 23)
    # consecutive pairs are untouched and stay the only ones in the style aggregate
    assert ("llama3-8b", "llama31-8b") in lin.pairs and not lin.pairs[("llama3-8b", "llama31-8b")].get("post_hoc")
    assert all(p not in sd.recovery_by_style(lin) for p in sd.EXTRA_PAIRS)
