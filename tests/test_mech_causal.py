"""The registered causal white-box test (PREREG_MECH_CAUSAL_20261031.md): its summary statistics and how pages read it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import mech_causal as mc                                              # noqa: E402
import mech_section as ms                                             # noqa: E402


def _item(bid, p_can, rows, p_c=0.2, p_c0=0.8):
    return dict(bid=bid, category="x", P_canonical=p_can, P_C=p_c, P_C0=p_c0,
                reopen=[dict(index=i, cls="paraphrase", P_rewording=a, P_A=b, P_A0=c, P_B=d, P_B0=e)
                        for i, (a, b, c, d, e) in enumerate(rows)])


def test_summary_counts_fixes_not_rewordings_and_reads_by_the_registered_rule():
    up = [_item(f"b{i}", 0.9, [(0.05, 0.5, 0.05, 0.3, 0.9), (0.0, 0.95, 0.0, 0.2, 0.9)]) for i in range(6)]
    s = mc.summarise(up)
    assert s["n_fixes"] == 6 and s["n_rewordings"] == 12
    assert s["A_vs_A0"]["sign_test"]["positive"] == 6 and s["A_vs_A0"]["sign_test"]["p_value"] < 0.05
    assert s["A_vs_A0"]["restored"]["k"] == 6                     # 0.95 >= 0.9 in one rewording per fix
    assert s["reading"] == dict(sufficient="causally sufficient", necessary="necessary")
    flat = [_item(f"b{i}", 0.9, [(0.05, 0.05, 0.06, 0.9, 0.9)]) for i in range(6)]
    assert mc.summarise(flat)["reading"] == dict(sufficient="not shown", necessary="not shown")


def test_pages_read_the_newest_full_run_and_never_a_smoke_test(tmp_path, monkeypatch):
    d = tmp_path / "results" / "mech"
    d.mkdir(parents=True)
    full = dict(prereg="P", limited=False, direction=dict(layer=23),
                summary=mc.summarise([_item("b1", 0.9, [(0.05, 0.5, 0.05, 0.3, 0.9)])]), items=[_item("b1", 0.9, [(0.05, 0.5, 0.05, 0.3, 0.9)])])
    (d / "causal_anchor_sorrybench_20261031_2101.json").write_text(json.dumps(full))
    (d / "causal_anchor_sorrybench_20261031_2200.json").write_text(json.dumps(dict(full, limited=True, items=[])))
    monkeypatch.setattr(ms, "ROOT", tmp_path)
    monkeypatch.setattr(ms, "_CACHE", {})
    c = ms.load_causal("sorrybench")
    assert c is not None and "b1" in c["items"]
    assert "layer 23" in ms.causal_sentence("sorrybench")
    assert "Causal test" in ms._causal_fig("sorrybench", "b1") and ms._causal_fig("sorrybench", "zz") == ""
