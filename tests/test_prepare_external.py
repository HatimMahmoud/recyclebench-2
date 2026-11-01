"""The bring-your-own-data adapter (tools/prepare_external.py) and the judged-P^- subset (tools/pminus_subset.py)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import prepare_external as pe                                        # noqa: E402

BENIGN = ["Explain how a bicycle derailleur shifts gears.",
          "Summarise the water cycle for a school class.",
          "Describe how sourdough starter is maintained."]


def _recipe(tmp_path, **over):
    """A local-source recipe over two JSON files, with `out` inside tmp_path."""
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "alpha.json").write_text(json.dumps(
        [{"Prompt": BENIGN[0], "Task": "gears", "Risk Type": "none"},
         {"Prompt": BENIGN[1], "Task": "water", "Risk Type": "none"}]))
    (src / "beta.json").write_text(json.dumps(
        [{"Prompt": BENIGN[2], "Task": "bread", "Risk Type": "none"},
         {"Prompt": "   ", "Task": "empty", "Risk Type": "none"},          # dropped: empty goal
         {"Prompt": BENIGN[0], "Task": "dup", "Risk Type": "none"}]))      # dropped: in-set dup
    rec = {"name": "t", "license": "MIT",
           "source": {"kind": "local", "root": str(src), "paths": ["alpha.json", "beta.json"]},
           "format": "json",
           "mapping": {"goal": "Prompt", "category": "@stem", "extra": {"Task": "Task"}},
           "out": str(tmp_path / "out.csv")}
    rec.update(over)
    p = tmp_path / "recipe.yaml"
    import yaml
    p.write_text(yaml.safe_dump(rec))
    return p


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_maps_fields_drops_empty_and_duplicates_and_keeps_recipe_order(tmp_path):
    prov = pe.build(_recipe(tmp_path), fetch=False)
    rows = _rows(tmp_path / "out.csv")
    assert [r["Goal"] for r in rows] == [BENIGN[0], BENIGN[1], BENIGN[2]]   # recipe, then file order
    assert [r["Category"] for r in rows] == ["alpha", "alpha", "beta"]      # @stem
    assert rows[0]["Task"] == "gears"                                       # extra column
    c = prov["counts"]
    assert (c["records_read"], c["dropped_empty_goal"],
            c["dropped_duplicate_within_corpus"], c["written"]) == (5, 1, 1, 3)
    assert prov["columns"] == ["Goal", "Category", "Task"]


def test_is_deterministic_and_records_provenance(tmp_path):
    r = _recipe(tmp_path)
    a = pe.build(r, fetch=False)
    first = (tmp_path / "out.csv").read_bytes()
    b = pe.build(r, fetch=False)
    assert (tmp_path / "out.csv").read_bytes() == first                     # byte-identical
    assert a["out_sha256"] == b["out_sha256"] and a["recipe_sha256"] == b["recipe_sha256"]
    assert len(a["file_sha256"]) == 2 and a["license"] == "MIT"
    side = json.loads((tmp_path / "out.csv.provenance.json").read_text())
    assert side["mapping"]["goal"] == "Prompt" and side["per_file"]["alpha.json"]["kept"] == 2


def test_reports_overlap_with_screened_benchmarks_and_drops_only_when_asked(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with (data / "screened.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Goal", "Category"])
        w.writeheader()
        w.writerow({"Goal": BENIGN[1].upper(), "Category": "x"})            # same after norm()
    r = _recipe(tmp_path, dedup={"against": ["screened", "absent_bm"], "cutoff": 0.85})

    import prepare_external as mod
    orig, mod.ROOT = mod.ROOT, tmp_path                                     # data/ lives in tmp
    try:
        rep = pe.build(r, fetch=False, out=tmp_path / "out.csv")["overlap"]
        assert rep["n_exact"] == 1 and rep["absent"] == ["absent_bm"]
        assert len(_rows(tmp_path / "out.csv")) == 3                        # reported, not dropped
        pe.build(r, fetch=False, out=tmp_path / "out.csv", drop_overlap=True)
        assert [x["Goal"] for x in _rows(tmp_path / "out.csv")] == [BENIGN[0], BENIGN[2]]
    finally:
        mod.ROOT = orig


def test_category_specs_and_formats(tmp_path):
    src = tmp_path / "s"
    src.mkdir()
    (src / "f.jsonl").write_text("\n".join(json.dumps({"q": g, "d": "dom"}) for g in BENIGN))
    import yaml
    r = tmp_path / "r.yaml"
    r.write_text(yaml.safe_dump({
        "name": "t", "source": {"kind": "local", "root": str(src), "paths": ["f.jsonl"]},
        "format": "jsonl", "mapping": {"goal": "q", "category": "d"},
        "out": str(tmp_path / "o.csv")}))
    pe.build(r, fetch=False)
    assert [x["Category"] for x in _rows(tmp_path / "o.csv")] == ["dom"] * 3

    r2 = tmp_path / "r2.yaml"
    r2.write_text(yaml.safe_dump({
        "name": "t", "source": {"kind": "local", "root": str(src), "paths": ["f.jsonl"]},
        "format": "jsonl", "mapping": {"goal": "q", "category": "@literal:fixed"},
        "out": str(tmp_path / "o2.csv")}))
    pe.build(r2, fetch=False)
    assert [x["Category"] for x in _rows(tmp_path / "o2.csv")] == ["fixed"] * 3


def test_a_github_source_must_be_pinned_to_a_full_commit(tmp_path):
    import yaml
    r = tmp_path / "r.yaml"
    r.write_text(yaml.safe_dump({
        "name": "t", "source": {"kind": "github", "repo": "o/n", "commit": "main",
                                "paths": ["a.json"]},
        "format": "json", "mapping": {"goal": "Prompt"}, "out": str(tmp_path / "o.csv")}))
    with pytest.raises(SystemExit, match="full 40-char sha"):
        pe.build(r, fetch=False)


def test_a_wrong_goal_field_fails_loudly_instead_of_writing_an_empty_corpus(tmp_path):
    with pytest.raises(SystemExit, match="mapping.goal"):
        pe.build(_recipe(tmp_path, mapping={"goal": "NotAField"}), fetch=False)


def test_the_shipped_sciencesafety_recipe_is_a_valid_pinned_recipe():
    import yaml
    rec = yaml.safe_load((ROOT / "datasets" / "sciencesafety.yaml").read_text())
    assert rec["source"]["kind"] == "github" and len(rec["source"]["commit"]) == 40
    assert rec["mapping"]["goal"] == "Prompt" and rec["mapping"]["category"] == "@stem"
    assert len(rec["source"]["paths"]) == 6 and rec["license"] == "MIT"
    assert "Dataset" in rec["source"]["root"] and "Attack" not in rec["source"]["root"]


# ---------------------------------------------------------------------------
# the judged-P^- subset
# ---------------------------------------------------------------------------
def test_pminus_subset_takes_the_judged_bids_and_refuses_the_wrong_csv(tmp_path, capsys):
    import pminus_subset as ps
    beh = tmp_path / "b.csv"
    with beh.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Goal", "Category"])
        w.writeheader()
        for g in BENIGN:
            w.writerow({"Goal": g, "Category": "c"})
    pilot = tmp_path / "p.json"
    pilot.write_text(json.dumps({"pair": "a -> b", "n_behaviours": 3, "breaks_old": 2,
                                 "status": "exploratory", "p_minus": 2,
                                 "p_minus_bids": ["jbb-0000", "jbb-0002"]}))
    out = tmp_path / "pm.csv"
    ps.main(["--pilot", str(pilot), "--behaviours", str(beh), "--out", str(out)])
    rows = _rows(out)
    assert [r["Goal"] for r in rows] == [BENIGN[0], BENIGN[2]]
    assert [r["source_bid"] for r in rows] == ["jbb-0000", "jbb-0002"]

    pilot.write_text(json.dumps({"p_minus_bids": ["jbb-0099"]}))
    with pytest.raises(SystemExit, match="absent from"):
        ps.main(["--pilot", str(pilot), "--behaviours", str(beh), "--out", str(out)])


def test_pminus_subset_says_rsr_is_undefined_when_p_minus_is_empty(tmp_path, capsys):
    import pminus_subset as ps
    beh = tmp_path / "b.csv"
    with beh.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Goal", "Category"])
        w.writeheader()
        w.writerow({"Goal": BENIGN[0], "Category": "c"})
    pilot = tmp_path / "p.json"
    pilot.write_text(json.dumps({"p_minus": 0, "p_minus_bids": []}))
    ps.main(["--pilot", str(pilot), "--behaviours", str(beh), "--out", str(tmp_path / "e.csv")])
    assert "undefined" in capsys.readouterr().out
    assert _rows(tmp_path / "e.csv") == []


# ---------------------------------------------------------------------------
# the style factorial's bring-your-own-data path (tools/run_compose_v2.py --behaviours)
# ---------------------------------------------------------------------------
def test_compose_v2_behaviours_flag_uses_every_row_as_a_fix(tmp_path):
    """--behaviours takes an already-decided P^- subset; no screening, no verdict store."""
    import run_compose_v2 as cv
    csv_path = tmp_path / "pm.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bid", "source_bid", "Category", "Goal"])
        w.writeheader()
        for i, g in enumerate(BENIGN):
            w.writerow({"bid": f"x-{i}", "source_bid": f"jbb-{i:04d}",
                        "Category": "c", "Goal": g})
    items, n_pm = cv.pminus(str(csv_path))
    assert n_pm == 3 and [p for _, p in items] == BENIGN
    # ids are row positions (load_jailbreakbench renumbers), which is why joins go on text
    assert [b for b, _ in items] == ["jbb-0000", "jbb-0001", "jbb-0002"]


def test_compose_v2_default_still_resolves_the_registered_anchor_p_minus():
    """The default path must stay bit-identical: the registered compose-v2 result depends on it."""
    import run_compose_v2 as cv
    needed = [ROOT / "data/sorrybench.csv",
              ROOT / "results/runs/lineage_sorrybench_20261020/manifest.jsonl",
              ROOT / "cache/llama_gen.jsonl",
              ROOT / "cache/judge_lineage_sorrybench.jsonl"]
    if not all(p.exists() for p in needed):
        pytest.skip("anchor artifacts absent (results/ or cache/ not present)")
    items, n_pm = cv.pminus()                      # no argument == the anchor, as before
    assert n_pm == 49, f"anchor |P^-| moved: {n_pm}"
    assert len(items) == 49 and all(b.startswith("jbb-") for b, _ in items)


# ---------------------------------------------------------------------------
# cluster-readiness additions: streaming errors, the preprocess hook, --data-root
# ---------------------------------------------------------------------------
def test_a_malformed_record_names_its_file_and_line(tmp_path):
    """A bare JSONDecodeError on a 200k-line corpus tells you nothing useful."""
    src = tmp_path / "s"
    src.mkdir()
    (src / "f.jsonl").write_text(
        json.dumps({"q": BENIGN[0]}) + "\n" + "{not json}\n")
    import yaml
    r = tmp_path / "r.yaml"
    r.write_text(yaml.safe_dump({
        "name": "t", "source": {"kind": "local", "root": str(src), "paths": ["f.jsonl"]},
        "format": "jsonl", "mapping": {"goal": "q"}, "out": str(tmp_path / "o.csv")}))
    with pytest.raises(SystemExit, match=r"f\.jsonl:2: invalid JSON line"):
        pe.build(r, fetch=False)


def test_a_headerless_csv_and_a_non_list_json_fail_clearly(tmp_path):
    src = tmp_path / "s"
    src.mkdir()
    (src / "empty.csv").write_text("")
    (src / "scalar.json").write_text('"just a string"')
    import yaml
    for f, msg in (("empty.csv", "no header row"), ("scalar.json", "expected a JSON list")):
        r = tmp_path / f"r_{f}.yaml"
        r.write_text(yaml.safe_dump({
            "name": "t", "source": {"kind": "local", "root": str(src), "paths": [f]},
            "format": "csv" if f.endswith(".csv") else "json",
            "mapping": {"goal": "q"}, "out": str(tmp_path / "o.csv")}))
        with pytest.raises(SystemExit, match=msg):
            pe.build(r, fetch=False)


def test_jsonl_and_csv_stream_without_materialising_the_file(tmp_path):
    """iter_records must be a generator, or a large corpus loads whole into memory."""
    import inspect
    assert inspect.isgeneratorfunction(pe.iter_records)
    src = tmp_path / "s"
    src.mkdir()
    (src / "f.jsonl").write_text("\n".join(json.dumps({"q": g}) for g in BENIGN))
    it = pe.iter_records(src / "f.jsonl", "jsonl")
    assert next(it)["q"] == BENIGN[0]            # first record available before the file is read
    assert len(list(it)) == 2


def test_the_preprocess_hook_can_rewrite_and_drop_records(tmp_path):
    mod = tmp_path / "myprep.py"
    mod.write_text(
        "def fix(rec):\n"
        "    if rec.get('skip'):\n"
        "        return None\n"
        "    rec['q'] = rec['q'].replace('PREFIX: ', '')\n"
        "    return rec\n")
    import sys as _sys
    _sys.path.insert(0, str(tmp_path))
    src = tmp_path / "s"
    src.mkdir()
    (src / "f.jsonl").write_text("\n".join([
        json.dumps({"q": "PREFIX: " + BENIGN[0]}),
        json.dumps({"q": BENIGN[1], "skip": True}),
        json.dumps({"q": BENIGN[2]})]))
    import yaml
    r = tmp_path / "r.yaml"
    r.write_text(yaml.safe_dump({
        "name": "t", "source": {"kind": "local", "root": str(src), "paths": ["f.jsonl"]},
        "format": "jsonl", "mapping": {"goal": "q"}, "preprocess": "myprep:fix",
        "out": str(tmp_path / "o.csv")}))
    prov = pe.build(r, fetch=False)
    goals = [x["Goal"] for x in _rows(tmp_path / "o.csv")]
    assert goals == [BENIGN[0], BENIGN[2]]                     # prefix stripped, skip dropped
    assert prov["per_file"]["f.jsonl"]["dropped_by_preprocess"] == 1


@pytest.mark.parametrize("spec,msg", [
    ("myprep", "must be 'module:function'"),
    ("nosuchmodule:fix", "cannot import"),
    ("myprep:absent", "is not callable"),
])
def test_a_bad_preprocess_spec_fails_clearly(spec, msg):
    with pytest.raises(SystemExit, match=msg):
        pe.load_preprocess(spec)


def test_a_recipe_without_preprocess_loads_no_code():
    assert pe.load_preprocess(None) is None
    assert pe.load_preprocess("") is None


def test_data_root_relocates_the_output_and_the_overlap_reference(tmp_path):
    """On a cluster the benchmark CSVs live on shared scratch, not in the checkout."""
    alt = tmp_path / "scratch_data"
    alt.mkdir()
    with (alt / "screened.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Goal", "Category"])
        w.writeheader()
        w.writerow({"Goal": BENIGN[1], "Category": "x"})
    src = tmp_path / "s"
    src.mkdir()
    (src / "f.jsonl").write_text("\n".join(json.dumps({"q": g}) for g in BENIGN))
    import yaml
    r = tmp_path / "r.yaml"
    r.write_text(yaml.safe_dump({
        "name": "t", "source": {"kind": "local", "root": str(src), "paths": ["f.jsonl"]},
        "format": "jsonl", "mapping": {"goal": "q"},
        "dedup": {"against": ["screened"], "cutoff": 0.85},
        "out": "data/t.csv"}))
    prov = pe.build(r, fetch=False, data_root=alt)
    assert (alt / "t.csv").exists(), "output did not follow --data-root"
    assert prov["overlap"]["n_exact"] == 1, "overlap reference did not follow --data-root"
