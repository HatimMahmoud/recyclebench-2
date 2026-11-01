"""EXPERIMENTAL -- delete with tools/sciencesafety_dashboard.py (see CLAUDE.md cleanup list).

tools/sciencesafety_dashboard.py: the standalone BYOD page states its own limits, keeps the page
contract the other dashboards keep, never shows a corrected figure for an uncalibrated corpus, and
honours --redact. Real-data checks skip when the run is absent."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import sciencesafety_dashboard as ssd                              # noqa: E402

RUN = ROOT / "results/runs/sciencesafety_20261101"
HAVE = (RUN / "summary_sciencesafety.json").exists() and (ROOT / "data/sciencesafety.csv").exists()


class Args:
    config = "configs/sciencesafety_3_31.yaml"
    run = str(RUN)
    old, new = "llama3-8b", "llama31-8b"
    corpus = "data/sciencesafety.csv"
    subset = "sciencesafety_p_minus_judged.csv"
    manifest = "manifest_sciencesafety.jsonl"
    pilot = "pminus_sciencesafety.json"
    summary = "summary_sciencesafety.json"
    cache = "cache/sciencesafety_20261101.jsonl"
    verdicts = "cache/judge_sciencesafety_20261101.jsonl"
    rpe_gate = "cache/rpe_gate_sciencesafety_20261101.jsonl"
    title = "t"
    redact = False
    out = None


@pytest.fixture(scope="module")
def built():
    if not HAVE:
        pytest.skip("sciencesafety run absent")
    d = ssd.collect(Args())
    return d, ssd.render(d, Args())


def test_page_contract_matches_the_other_dashboards(built):
    """One <script> and one <style>, every nav target present: the repo's page invariants."""
    _, page = built
    assert page.count("<script>") == 1 and page.count("<style>") == 1
    for sid, _n, _t in ssd.NAV:
        assert f'id="{sid}"' in page, f"nav points at a missing section: {sid}"
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")


def test_the_page_says_it_is_isolated_exploratory_and_t1_unresolved(built):
    """Without these the page would read as a study result, which is exactly what it is not."""
    _, page = built
    for claim in ("not part of the study's evidence base", "study_data.py",
                  "amendment", "T1", "exploratory", "uncalibrated"):
        assert claim in page, f"missing the {claim!r} disclosure"


def test_no_corrected_figure_is_shown_for_an_uncalibrated_corpus(built):
    """A corrected (Rogan-Gladen) rate needs human labels from this corpus; none exist."""
    d, page = built
    assert "no corrected figure exists" in page.lower()
    body = re.sub(r"<pre class=\"resp\">.*?</pre>", "", page, flags=re.S)
    # The RQ cards are the shared estimand boilerplate (identical on every dashboard) and describe
    # how correction works in general; they are not figures for this corpus.
    body = re.sub(r'<section id="s-rq".*?</section>', "", body, flags=re.S)
    # Describing the estimand is fine; attaching "corrected" to a RATE for this corpus is not.
    # Only the anchor's 0.551, shown for orientation, may be a corrected number.
    for m in re.findall(r".{60}corrected.{60}", body, flags=re.I | re.S):
        if re.search(r"\d+(\.\d+)?\s*%|0\.\d{3}", m):
            assert "0.551" in m, f"a corrected rate is shown for this corpus: {m!r}"
    # and the raw figure must never be relabelled: the summary stores corrected == raw here
    r = json.loads((RUN / Args.summary).read_text())["RSR_primary_RAW"][0]
    assert r["RSR_corrected"] == r["RSR_raw"], "summary gained a real correction; page must change"


def test_numbers_on_the_page_come_from_the_artifacts(built):
    d, page = built
    pilot = json.loads((RUN / Args.pilot).read_text())
    summ = json.loads((RUN / Args.summary).read_text())
    r = summ["RSR_primary_RAW"][0]
    assert len(d["rows"]) == pilot["p_minus"] == r["n_p_minus"]
    assert sum(1 for x in d["rows"] if x["reopened"]) == r["k_recovered"]
    assert f"{r['k_recovered']} / {r['n_p_minus']}" in page
    assert f'{pilot["p_minus"]}' in page and str(pilot["n_behaviours"]) in page
    # the 2x2 must add up in the judged column
    assert pilot["breaks_old"] >= pilot["p_minus"]


def test_the_judged_and_offline_partitions_are_both_shown(built):
    """run-real stores no verdicts, so a cache-rebuilt partition disagrees; hiding one is misleading."""
    _, page = built
    assert "offline" in page.lower() and "not judged" in page
    assert "negative-flip cell was never judged" in page or "never judged" in page


def test_gate_class_profile_is_reported_and_flags_the_empty_classes(built):
    d, page = built
    assert d["adm"]["back_translation"] == 0, "the finding this section reports has changed"
    assert d["prop"]["back_translation"] > 0
    assert "RQ3" in page and "unanswerable" in page.lower()


def test_inversion_arm_distinguishes_undefined_from_rejected(built):
    """A_gate with an empty gate store is undefined, NOT 'the gate rejected them'."""
    d, page = built
    if not d["loop"]:
        pytest.skip("no scored closed-loop artifact")
    assert d["loop"]["admission"]["gate"]["rate"] is None
    assert "undefined" in page
    assert "may not be claimed" in page


def test_redact_removes_every_prompt_and_response(built):
    d, _ = built
    page = ssd.render(d, type("A", (Args,), {"redact": True})())
    assert "redact" in page.lower()
    for row in d["rows"][:5]:
        assert row["prompt"][:60] not in page
        for dr in row["old_draws"]:
            if dr["text"]:
                assert dr["text"][:60] not in page


def test_subset_ids_are_labelled_as_row_positions(built):
    """The join trap: ids in a subset CSV are not corpus ids, so the page must show both."""
    d, page = built
    assert d["unjoined"] == 0, "metadata join on prompt text failed"
    assert "row positions" in page and "on prompt text" in page
    assert any(x["bid"] != x["src"] for x in d["rows"])
