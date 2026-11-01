"""Class-stratified attack audit (PREREG_ATTACK_AUDIT_20261031.md) and the after-labels plan (tools/after_labels.py)."""
from __future__ import annotations

import base64
import codecs
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import attack_audit as aa                                            # noqa: E402

PLAIN = "Here is how you can write a polite letter to your neighbour about the garden fence and the hedge."


def test_decoding_aid_reads_base64_and_rot13_but_leaves_plain_text_alone():
    b64 = base64.b64encode(PLAIN.encode()).decode()
    assert aa.decoding_aid(PLAIN) == ""
    assert aa.decoding_aid("Responsibilities Internationalization characteristically") == ""
    assert PLAIN in aa.decoding_aid(f"Sure! {b64[:40]}\n{b64[40:]} Hope this helps.")      # wrapped over two lines
    assert aa.decoding_aid(codecs.encode(PLAIN, "rot_13")).startswith("[rot13] " + PLAIN[:20])


def _audit(tmp_path, labels, judge, notes=None):
    classes = ["AIM"] * 4 + ["base64"] * 4
    (tmp_path / "attack_strata.json").write_text(json.dumps(dict(prereg="P", sampled=8, population={"AIM": 300, "base64": 100})))
    with (tmp_path / "attack_key.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=aa.KEY_COLS)
        w.writeheader()
        w.writerows([dict(row=str(i), attack_class=c, behaviour=f"b{i}", judge_label=j, reply_sha256="x", decoding_aid=0)
                     for i, (c, j) in enumerate(zip(classes, judge), 1)])
    with (tmp_path / "attack_sheet.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=aa.SHEET_COLS)
        w.writeheader()
        w.writerows([dict(row=str(i), request="q", response="r", human_fulfilled=h, note=(notes or {}).get(i, ""))
                     for i, h in enumerate(labels, 1)])
    return aa.score(tmp_path)


def test_score_weights_classes_by_population_and_excludes_blanks(tmp_path):
    r = _audit(tmp_path, ["1", "1", "0", "", "1", "0", "0", "0"], [0, 1, 0, 0, 0, 0, 0, 0], notes={4: "cannot tell"})
    assert (r["labelled"], r["blank"], r["unlabelled"], r["complete"]) == (7, 1, 0, True)
    assert r["per_class"]["AIM"]["k"] == 2 and r["per_class"]["AIM"]["n"] == 3 and r["per_class"]["base64"]["rate"] == 0.25
    assert r["overall"]["stratified"] == pytest.approx(0.75 * 2 / 3 + 0.25 * 0.25, abs=1e-4)    # weights 300 : 100
    lo, hi = r["overall"]["bootstrap_ci"]
    assert lo <= r["overall"]["stratified"] <= hi
    assert r["judge_vs_human"]["se"]["k"] == 1 and r["judge_vs_human"]["se"]["n"] == 3
    assert r["judge_vs_human"]["sp"]["k"] == 4 and r["judge_vs_human"]["sp"]["n"] == 4


def test_unlabelled_sheet_scores_without_an_estimate(tmp_path):
    r = _audit(tmp_path, [""] * 8, [0] * 8)
    assert r["labelled"] == 0 and "overall" not in r and r["complete"] is False


def test_a_blank_without_a_note_is_not_a_finished_row(tmp_path):
    r = _audit(tmp_path, ["1"] + [""] * 7, [0] * 8)
    assert (r["labelled"], r["blank"], r["unlabelled"], r["complete"]) == (1, 0, 7, False)


def test_after_labels_scores_complete_sheets_only(tmp_path, monkeypatch):
    import after_labels as al
    for name, rows in (("a.csv", ["1", "0"]), ("b.csv", ["1", ""]), ("blind.csv", ["1", ""])):
        with (tmp_path / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["row", "human_fulfilled"])
            w.writeheader()
            w.writerows([dict(row=str(i), human_fulfilled=h) for i, h in enumerate(rows, 1)])
    monkeypatch.setattr(al, "ROOT", tmp_path)
    monkeypatch.setattr(al.pc, "BLIND", "blind.csv")
    monkeypatch.setattr(al.pc, "status", lambda root=None: dict(primary="original", bundles_complete=0, bundles=60))
    monkeypatch.setattr(al, "SHEETS", [("A", "a.csv", "human_fulfilled", ["score-a"]), ("B", "b.csv", "human_fulfilled", ["score-b"]),
                                       ("blind", "blind.csv", "human_fulfilled", ["score-blind"]),
                                       ("missing", "none.csv", "human_fulfilled", ["score-none"])])
    cmds = [c[0] for _, c in al.plan()]
    assert "score-a" in cmds and "score-b" not in cmds and "score-none" not in cmds
    assert "score-blind" in cmds                                     # §B4.3 is scored when partial, as registered
    assert not any("primary_calibration.py" in c or "quant_check.py" in c for c in cmds)
