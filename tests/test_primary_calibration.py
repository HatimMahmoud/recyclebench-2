"""prereg Amendment 2 §B4.3: the blind re-label becomes the primary calibration only once every bundle is labelled, and
then every corrected figure follows it (tools/primary_calibration.py)."""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import primary_calibration as pc                                    # noqa: E402

COLS = ["bid", "variant_index", "transform_class", "judge_label", "request", "response", "human_fulfilled"]
BLIND_COLS = [c for c in COLS if c != "judge_label"]


def _write(path: Path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r[c] for c in cols} for r in rows])


def _root(tmp_path, labels):
    rows = [dict(bid=f"b{b}", variant_index=str(v), transform_class="paraphrase", judge_label=str((b + v) % 2),
                 request=f"q{b}", response=f"r{b}{v}", human_fulfilled=str(b % 2)) for b in range(3) for v in range(2)]
    _write(tmp_path / pc.ORIGINAL, COLS, rows)
    (tmp_path / pc.ORIGINAL).with_name("judge_strata.txt").write_text("n_pos_total=2\nn_neg_total=1\n")
    shuffled = rows[::-1]
    _write(tmp_path / pc.BLIND, BLIND_COLS, [dict(r, human_fulfilled=labels(r)) for r in shuffled])
    return rows


def test_blind_becomes_primary_only_when_every_bundle_is_labelled(tmp_path):
    _root(tmp_path, lambda r: "")
    assert pc.status(tmp_path)["primary"] == "original"
    _root(tmp_path, lambda r: "" if r["bid"] == "b2" and r["variant_index"] == "1" else "1")
    st = pc.status(tmp_path)
    assert st["primary"] == "original" and (st["bundles_complete"], st["bundles"]) == (2, 3)
    _root(tmp_path, lambda r: "1")
    assert pc.status(tmp_path)["primary"] == "blind"


def test_sheet_substitutes_the_blind_labels_and_keeps_the_strata(tmp_path):
    rows = _root(tmp_path, lambda r: "")
    with pc.sheet(tmp_path) as sh:
        assert sh == tmp_path / pc.ORIGINAL
    _root(tmp_path, lambda r: "0" if r["bid"] == "b1" else "1")
    with pc.sheet(tmp_path) as sh:
        got = {(r["bid"], r["variant_index"]): (r["human_fulfilled"], r["judge_label"]) for r in csv.DictReader(sh.open())}
        assert sh.with_name("judge_strata.txt").read_text().startswith("n_pos_total=2")
    assert got == {(r["bid"], r["variant_index"]): ("0" if r["bid"] == "b1" else "1", r["judge_label"]) for r in rows}
    assert not sh.exists()                                          # the substituted sheet lives only inside the block


def test_resolve_fails_loudly_instead_of_falling_back(tmp_path):
    reg = Path(pc.CORRECTED["anchor"])
    _root(tmp_path, lambda r: "")
    assert pc.resolve(reg, tmp_path) == reg
    _root(tmp_path, lambda r: "1")
    with pytest.raises(RuntimeError, match="apply"):
        pc.resolve(reg, tmp_path)
    twin = tmp_path / reg.with_name(reg.name.replace(pc.REG_SUFFIX, pc.BLIND_SUFFIX))
    twin.parent.mkdir(parents=True, exist_ok=True)
    twin.write_text("{}")
    assert pc.resolve(reg, tmp_path) == reg.with_name(twin.name)
    assert pc.resolve("results/x/corrected_heldout.json", tmp_path) == Path("results/x/corrected_heldout.json")


def test_blind_sheet_must_cover_the_calibration_rows(tmp_path):
    _root(tmp_path, lambda r: "1")
    rows = list(csv.DictReader((tmp_path / pc.BLIND).open()))
    _write(tmp_path / pc.BLIND, BLIND_COLS, rows[:-1])
    with pytest.raises(ValueError, match="row for row"):
        pc.status(tmp_path)


@pytest.mark.skipif(not (ROOT / pc.ORIGINAL).exists() or not all((ROOT / p).exists() for p in pc.CORRECTED.values()),
                    reason="needs the labelled SORRY-Bench sheet and the registered corrected files")
def test_identical_blind_labels_reproduce_every_registered_corrected_figure(tmp_path):
    for rel in [pc.ORIGINAL, str(Path(pc.ORIGINAL).with_name("judge_strata.txt")), *pc.CORRECTED.values()]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    orig = list(csv.DictReader((ROOT / pc.ORIGINAL).open(newline="")))
    random.Random(7).shuffle(orig)
    _write(tmp_path / pc.BLIND, BLIND_COLS, orig)                  # the blind sheet carrying the original labels
    written = pc.apply(tmp_path)
    assert len(written) == len(pc.CORRECTED)
    for rel in pc.CORRECTED.values():
        a = json.loads((tmp_path / rel).read_text())
        b = json.loads((tmp_path / pc.resolve(rel, tmp_path)).read_text())
        for blk in ("RSR", "PBR_new"):
            assert {k: v for k, v in a[blk].items() if k != "verdict_warning"} == \
                   {k: v for k, v in b[blk].items() if k != "verdict_warning"}
        assert (b["judge"]["se"], b["judge"]["sp"], b["judge"]["n_bundles"]) == (a["judge"]["se"], a["judge"]["sp"], 60)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pc.apply(tmp_path)
