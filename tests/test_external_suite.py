"""tools/external_suite.py: the external-corpora run's design, generated files, eligibility and completion matrix.

No data and no models: the design checks read the committed configs and specs; the status checks build a synthetic
run tree (a canonical report, recycle run directories, an inversion manifest) under tmp_path.
"""
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import external_suite as s                                      # noqa: E402
from recyclebench.experiment import ExperimentSpec              # noqa: E402


# ---------------- design ----------------
def test_pairs_follow_release_order_and_exclude_70b():
    lineage = yaml.safe_load((ROOT / s.BASE_CONFIG).read_text())["lineage"]
    assert len(s.PAIRS) == 19 and len({(o, n) for o, n, _, _ in s.PAIRS}) == 19
    for old, new, group, _ in s.PAIRS:
        assert lineage.index(old) < lineage.index(new), (old, new)
        assert group in s.GROUPS and old in s.DENSE                # every older model is dense
        assert "70b" not in old + new
    assert ("llama4-scout", "llama4-maverick") not in {(o, n) for o, n, _, _ in s.PAIRS}   # one release, not a transition
    assert {n for _, n, g, _ in s.PAIRS if g == "scout"} == {"llama4-scout"}
    assert {n for _, n, g, _ in s.PAIRS if g == "maverick"} == {"llama4-maverick"}
    assert set(s.MODELS) <= set(lineage)


def test_committed_configs_and_specs_are_what_the_generator_writes():
    assert s.configs(ROOT, check=True) == []
    assert s.specs(ROOT, check=True) == []


def test_pair_configs_keep_every_science_setting_of_the_base():
    base = yaml.safe_load((ROOT / s.BASE_CONFIG).read_text())
    for old, new, group, _ in s.PAIRS:
        cfg = yaml.safe_load((ROOT / s.pair_config_path(old, new, group)).read_text())
        assert cfg["lineage"] == [old, new]
        assert {k: v for k, v in cfg.items() if k != "lineage"} == {k: v for k, v in base.items() if k != "lineage"}


def test_every_spec_loads_through_the_driver_and_points_at_its_pair():
    n = 0
    for corpus in s.CORPORA:
        for old, new, group, _ in s.PAIRS:
            spec = ExperimentSpec.load(ROOT / s.spec_path(corpus, old, new))
            assert (spec.old, spec.new, spec.name) == (old, new, s.spec_name(corpus, old, new))
            assert spec.science_config == s.pair_config_path(old, new, group)
            assert spec.behaviours == f"data/{corpus}.csv" and spec.min_p_minus_for_recycle == s.RECYCLE_FLOOR
            assert spec.execution.cache_dir == s.CACHE_DIR and spec.execution.run_root == s.RUN_ROOT
            assert spec.cache_path.name == f"{spec.name}.jsonl"   # its own cache: no two specs share a JSONL
            n += 1
    assert n == 57


def test_config_body_check_catches_a_changed_science_setting(tmp_path):
    (tmp_path / "configs").mkdir()
    base = (ROOT / s.BASE_CONFIG).read_text()
    (tmp_path / s.BASE_CONFIG).write_text(base)
    old, new, group, _ = s.PAIRS[2]
    assert s.configs(tmp_path, check=False) == []                 # writes all 19
    p = tmp_path / s.pair_config_path(old, new, group)
    p.write_text(p.read_text().replace("cosine: 0.75", "cosine: 0.70"))
    assert any(str(s.pair_config_path(old, new, group)) in x for x in s.configs(tmp_path, check=True))


# ---------------- data ----------------
def _csv(path, goals):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Goal", "Category"])
        for i, g in enumerate(goals):
            w.writerow([g, f"c{i % 2}"])


def test_fingerprint_depends_on_request_order(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _csv(a, ["first request", "second request"])
    _csv(b, ["second request", "first request"])
    fa, fb = s.fingerprint(a), s.fingerprint(b)
    assert fa["rows"] == fb["rows"] == 2 and fa["goals_sha256"] != fb["goals_sha256"]


def test_data_check_blocks_absent_and_fails_unregistered(tmp_path):
    _csv(tmp_path / "data" / "jbb.csv", ["only one request"])
    res = s.data_check(tmp_path, build=False)
    assert res["guidedbench"]["status"] == s.B and res["sciencesafety"]["status"] == s.B
    assert res["jbb"]["status"] == s.F                            # wrong rows: ids and seeds would differ


# ---------------- eligibility + matrix ----------------
def _report(root, pm, incomplete=()):
    corpora = {}
    for corpus in s.CORPORA:
        pairs = {f"{o}->{n}": {"pminus": pm.get((corpus, o, n), 0), "persistent": 1, "neg_flip": 0, "both_safe": 3,
                               "group": g, "complete": (corpus, o, n) not in incomplete, "pminus_bids": []}
                 for o, n, g, _ in s.PAIRS}
        corpora[corpus] = {"n_behaviours": 10, "models": {}, "pairs": pairs}
    p = root / s.RUN_ROOT / "canonical_20261104_0000.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"corpora": corpora}))
    return p


def _recycle_run(root, corpus, old, new, stages, pilot_pm=7, rsr=True):
    d = root / s.RUN_ROOT / s.spec_name(corpus, old, new)
    (d / "stages").mkdir(parents=True)
    for st in stages:
        (d / "stages" / f"{st}.json").write_text("{}")
    (d / "pminus_pilot.json").write_text(json.dumps({"p_minus": pilot_pm}))
    if rsr:
        (d / "manifest_x.jsonl").write_text(json.dumps({"type": "audit", "bid": "jbb-0000", "admitted": True}) + "\n")
        (d / "summary_real.json").write_text(json.dumps({"RSR_primary_RAW": [
            {"pair": f"{old} -> {new}", "n_p_minus": 6, "k_recovered": 3, "n_unresolved": 0, "defined": True,
             "RSR_raw": 0.5}]}))


def _cell(res, corpus, old, new, arm):
    return next(c for c in res["cells"] if (c["corpus"], c["old"], c["new"], c["arm"]) == (corpus, old, new, arm))


def test_eligible_specs_need_a_complete_pair_at_the_floor(tmp_path):
    rep = _report(tmp_path, {("jbb", "llama3-8b", "llama31-8b"): 5, ("jbb", "llama2-7b", "llama2-13b"): 4,
                             ("guidedbench", "llama3-8b", "llama4-scout"): 9},
                  incomplete={("guidedbench", "llama3-8b", "llama4-scout")})
    assert s.eligible_specs(tmp_path, "dense", rep) == [s.spec_path("jbb", "llama3-8b", "llama31-8b")]
    assert s.eligible_specs(tmp_path, "scout", rep) == []         # incomplete screen: never eligible


def test_matrix_statuses_follow_artifacts_and_floors(tmp_path):
    rep = _report(tmp_path, {("jbb", "llama3-8b", "llama31-8b"): 6, ("jbb", "llama2-7b", "llama3-8b"): 5,
                             ("jbb", "llama2-7b", "llama2-13b"): 2})
    _recycle_run(tmp_path, "jbb", "llama3-8b", "llama31-8b", ["screen", "subset", "recycle", "partition", "audit"])
    _recycle_run(tmp_path, "jbb", "llama2-7b", "llama3-8b", ["screen"], pilot_pm=4, rsr=False)
    res = s.collect(tmp_path, rep)
    done = _cell(res, "jbb", "llama3-8b", "llama31-8b", "recycle")
    assert done["status"] == s.C and done["rsr"]["wilson"] is not None
    assert _cell(res, "jbb", "llama2-7b", "llama3-8b", "recycle")["status"] == s.NA   # re-judged below the floor
    assert _cell(res, "jbb", "llama2-7b", "llama2-13b", "recycle")["status"] == s.NA  # 2 < 5: not run
    assert _cell(res, "jbb", "llama2-7b", "llama2-13b", "rpe")["status"] == s.P       # fixes exist, not inverted
    assert _cell(res, "jbb", "llama2-13b", "llama3-8b", "rpe")["status"] == s.NA      # empty P^-: undefined
    assert res["style_arcs"]["jbb"]["status"] == s.P                                  # a manifest, no arcs yet
    assert len(res["cells"]) == 3 * 19 * 3


def test_matrix_without_report_and_markdown_cells(tmp_path):
    res = s.collect(tmp_path)
    assert all(c["status"] == s.B for c in res["cells"] if c["arm"] == "canonical")   # no data here
    rows = [ln for ln in s.markdown(res).splitlines() if ln.startswith("| jbb |") and "→" in ln]
    assert len(rows) == 19 * 3 and all(ln.replace("\\|", "").count("|") == 8 for ln in rows)
