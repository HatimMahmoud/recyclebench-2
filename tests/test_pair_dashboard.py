"""tools/pair_dashboard.py: every number on a pair page is consistent with the evidence layer, case texts
are the right replies, placeholders are never shown as scores, and missing data renders as not yet finished."""
from __future__ import annotations

import glob
import html
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import pair_dashboard as pdb                                   # noqa: E402
import study_data as sd                                        # noqa: E402
from recyclebench.judge import looks_like_refusal              # noqa: E402

REAL = Path(__file__).resolve().parent.parent
HAVE_REAL = (REAL / "results/runs/lineage_sorrybench_20261020/manifest.jsonl").exists() and (REAL / "data/sorrybench.csv").exists()


def test_empty_world_pair_page(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "ROOT", tmp_path)
    sd._CACHE_MEMO.clear()
    try:
        study = sd.build_study(None)
        page = pdb.render_pair(study, "llama3-8b", "llama31-8b", False)
    finally:
        sd._CACHE_MEMO.clear()
    assert page.count("<script>") == 1
    assert "not yet finished" in page


def test_rejected_candidate_never_shows_placeholder_scores():
    sc = {"cosine": 0.62, "bertscore": 1.0, "nli_entail": 1.0, "llm_intent": 1.0}
    txt = pdb._score_txt(sc, admitted=False, thr=sd.STRICT_GATE)
    assert "cos 0.62" in txt and "bert" not in txt and "nli" not in txt and "stopped at cosine" in txt
    assert "lexical floor" in pdb._score_txt({"cosine": 1.0, "bertscore": 1.0, "nli_entail": 1.0}, admitted=False)
    assert pdb._score_txt({"cosine": .8, "bertscore": .9, "nli_entail": .7}, admitted=True) == "cos 0.80 · bert 0.90 · nli 0.70"


def test_drift_test_line():
    line = pdb._mw_line([0.70, 0.72, 0.75, 0.71], [0.95, 0.97, 0.99, 0.96], "nli")
    assert "answered sit lower" in line and "Mann–Whitney" in line
    assert "too few" in pdb._mw_line([0.9], [0.8, 0.7, 0.6], "nli")


@pytest.fixture(scope="module")
def study():
    if not HAVE_REAL:
        pytest.skip("local RecycleBench results not present")
    sd._CACHE_MEMO.clear()
    return sd.build_study(None)


@pytest.mark.parametrize("pair", sd.PAIRS)
def test_real_pair_page_is_consistent(study, pair):
    o, n = pair
    dvs = pdb.build_pair(study, o, n)
    page = pdb.render_pair(study, o, n, False)
    assert page.count("<script>") == 1
    for d in dvs:
        if d.status == "complete":
            assert len(d.pm) + len(d.persistent) + len(d.regression) + d.both_safe == d.n
        c = study.cells.get((o, n, d.key))
        if c and c.status in ("complete", "p-minus-only"):
            assert sorted(c.p_minus) == sorted(d.pm)
        pv = d.primary
        if pv and pv.rsr and (sd.ROOT / pv.rr.spec.csv).resolve() == (sd.ROOT / d.ds.csv).resolve():
            assert sorted(pv.pm_run) == sorted(d.pm)
            assert sum(1 for b in d.pm if pv.items[b].state == "recovered") == pv.rsr["k"]
        for b in d.pm:
            t_o, v_o = pdb._first(d.draws[(o, b)], 1)
            t_n, v_n = pdb._first(d.draws[(n, b)], None)
            assert v_o == 1 and not looks_like_refusal(t_o) and v_n == 0 and looks_like_refusal(t_n)
        for r in d.runs:
            for it in r.items.values():
                for v, _ in it.variants:
                    if v.response is not None:
                        assert v.verdict == int(not looks_like_refusal(v.response))
        m = re.search(rf'<div class="tabpanel" role="tabpanel" id="panel-{d.key}".*?(?=<div class="tabpanel"|<section)', page, re.S)
        assert m, d.key
        outs = Counter(re.findall(r'<details class="case" data-outcome="([a-z]+)"', m.group(0)))
        assert sum(v for k, v in outs.items() if k != "regression") == len(d.pm)
        assert outs.get("regression", 0) == len(d.regression or [])


def test_redacted_page_has_no_text(study):
    dvs = pdb.build_pair(study, "llama3-8b", "llama31-8b")
    page = html.unescape(pdb.render_pair(study, "llama3-8b", "llama31-8b", True))
    for d in dvs:
        bmap = {b.bid: b for b in d.ds.behaviours}
        for b in d.pm:
            assert bmap[b].canonical_prompt[:40] not in page
            t, _ = pdb._first(d.draws[("llama31-8b", b)], None)
            if t and len(t) > 40:
                assert t[:40] not in page


def test_main_writes_one_folder_per_pair(study, tmp_path):
    assert pdb.main(["--pairs", "llama32-1b:llama32-3b", "--out-root", str(tmp_path)]) == 0
    files = glob.glob(str(tmp_path / "llama32-1b__to__llama32-3b" / "dashboard_*.html"))
    assert len(files) == 1
    idx = glob.glob(str(tmp_path / "index_*.html"))
    assert len(idx) == 1 and "llama32-1b__to__llama32-3b/dashboard_" in Path(idx[0]).read_text()
