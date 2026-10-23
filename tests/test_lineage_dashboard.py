"""tools/lineage_core + judge_lineage + lineage_dashboard: the cache-rebuilt lineage view must
reproduce `run-real` exactly when given the run's judge, and must never leak or mangle text."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import judge_lineage as jl                                     # noqa: E402
import lineage_core as lc                                      # noqa: E402
import lineage_dashboard as ld                                 # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.datasets import load_jailbreakbench          # noqa: E402
from recyclebench.manifest import GateAuditRecord, Manifest    # noqa: E402
from recyclebench.realrun import run_real_study                # noqa: E402
from recyclebench.types import Response, Variant               # noqa: E402

MODELS = ["llama2-7b", "llama3-8b", "llama31-8b"]
NASTY = "<script>alert('x')</script> & </details>"


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class FakeModel:
    """Deterministic target: text depends on (model, prompt, seed); some draws collide on purpose
    so the (prompt, response) de-duplication is exercised."""

    def __init__(self, name):
        self.name = name

    def respond(self, prompt, *, behaviour=None, variant=None, canonical=False, seed=None):
        return Response(text=f"{self.name} says {_h(self.name, prompt, (seed or 0) % 2) % 97} {NASTY}",
                        latent_harmful=None)


class FakeJudge:
    """Deterministic, prompt-aware judge (complies ~40% of the time)."""
    calls = 0

    def classify(self, prompt, response):
        FakeJudge.calls += 1
        return int(_h(prompt, response.text) % 5 < 2)


@pytest.fixture()
def world(tmp_path):
    csv = tmp_path / "beh.csv"
    rows = ["Goal,Category"] + [f'"Request {i} {NASTY if i == 0 else ""}",{1 + i % 44}' for i in range(60)]
    csv.write_text("\n".join(rows) + "\n")
    behaviours = load_jailbreakbench(str(csv))
    cfg = Config()
    cfg.lineage = list(MODELS)
    cfg.judges = {"rubric": {"model_id": "fake-judge"}}
    cfg.budget.k_canonical_samples = 3
    cfg.budget.samples_per_variant = 1
    classes = ["paraphrase", "syntactic", "lexical"]
    man = Manifest()
    for b in behaviours:
        vs = []
        for j in range(_h(b.bid, "n") % 5):                     # 0..4 admitted -> some unresolved
            vs.append(Variant(b.bid, f"variant {j} of {b.canonical_prompt}", classes[j % 3], 0.5, index=j))
        if vs:
            man.admitted[b.bid] = vs
        for j in range(5):
            adm = j < len(vs)
            man.audit.append(GateAuditRecord(b.bid, j, classes[j % 3], True, adm,
                                             cosine=0.9 if adm or j % 2 else 0.5, bertscore=1.0 if not adm else 0.9,
                                             nli_entail=1.0, llm_intent=1.0))
    cache_path = tmp_path / "cache.jsonl"
    res = run_real_study(cfg, behaviours, [FakeModel(m) for m in MODELS], FakeJudge(), man,
                         cache_path=cache_path)
    summary_path = tmp_path / "summary_real.json"
    summary_path.write_text(json.dumps(res.summary(), default=str))
    return dict(cfg=cfg, behaviours=behaviours, manifest=man, cache=lc.load_response_cache(cache_path),
                summary=summary_path, tmp=tmp_path, csv=csv)


def _judge_all(w, scope="all"):
    store = lc.VerdictStore(w["tmp"] / "verdicts.jsonl")
    tag = lc.judge_tag_from_cfg(w["cfg"])
    judge = FakeJudge()
    for phase in ["canonical", scope]:
        for k, prompt, resp in jl.todo_items(w["cfg"], w["behaviours"], w["manifest"], w["cache"],
                                             MODELS, store, tag, phase):
            store.put(k, judge.classify(prompt, Response(text=resp)))
    store.close()
    return lc.StoredVerdicts(lc.VerdictStore(w["tmp"] / "verdicts.jsonl"), tag)


def _args(w, redact=False):
    return argparse.Namespace(redact=redact, config="cfg.yaml", cache="cache.jsonl",
                              behaviours=str(w["csv"]), manifest="manifest.jsonl")


def test_replayed_judge_reproduces_run_real_exactly(world):
    verd = _judge_all(world)
    an = ld.analyse(world["cfg"], world["behaviours"], world["manifest"], world["cache"], verd,
                    world["summary"], None)
    run = json.loads(world["summary"].read_text())
    for row in run["RSR_primary_RAW"]:
        o, n = (s.strip() for s in row["pair"].split("->"))
        pv = an.pairs[(o, n)]
        assert pv.complete
        assert pv.n == row["n_p_minus"]
        assert pv.k == row["k_recovered"]
        assert round(pv.rsr.rsr_raw, 4) == row["RSR_raw"]
    for row in run["PBR_backbone_RAW"]:
        x = an.models[row["model"]].pbr
        assert (x.k_bypassed, x.n_resolved, x.n_unresolved) == (row["k_bypassed"], row["n_resolved"], row["n_unresolved"])
    assert sum(an.pairs[p].n for p in an.adjacent) > 0, "fixture should produce a non-empty P^-"
    html = ld.render(an, _args(world))
    assert "reproduces the run summary exactly" in html
    assert "per-item views reproduce the run exactly" in html
    assert ld.reconcile(an)[1] == 0


def test_rsr_scope_judges_only_what_rsr_needs(world):
    verd = _judge_all(world, scope="rsr")
    an = ld.analyse(world["cfg"], world["behaviours"], world["manifest"], world["cache"], verd,
                    world["summary"], None)
    run = json.loads(world["summary"].read_text())
    for row in run["RSR_primary_RAW"]:
        o, n = (s.strip() for s in row["pair"].split("->"))
        assert an.pairs[(o, n)].k == row["k_recovered"]
    # PBR needs every variant, so with rsr scope at least one model must report "not judged"
    assert any(m.pbr is None for m in an.models.values())
    ld.render(an, _args(world))                       # incomplete PBR must still render


def test_dedup_judges_each_prompt_response_once(world):
    store = lc.VerdictStore(world["tmp"] / "v.jsonl")
    tag = lc.judge_tag_from_cfg(world["cfg"])
    todo = jl.todo_items(world["cfg"], world["behaviours"], world["manifest"], world["cache"],
                         MODELS, store, tag, "all")
    keys = [k for k, *_ in todo]
    assert len(keys) == len(set(keys))
    n_items = sum(1 for it in lc.iter_items(world["cfg"], world["behaviours"], world["manifest"],
                                           world["cache"], MODELS))
    assert len(keys) < n_items                          # FakeModel collides seeds 0 and 2


def test_offline_mode_flags_itself_and_differs_cleanly(world):
    an = ld.analyse(world["cfg"], world["behaviours"], world["manifest"], world["cache"],
                    lc.OfflineVerdicts(), world["summary"], None)
    html = ld.render(an, _args(world))
    assert "offline reconstruction" in html
    assert "judge_lineage.py" in html                   # tells the reader how to reconcile


def test_text_is_escaped_and_redaction_removes_it(world):
    verd = _judge_all(world)
    an = ld.analyse(world["cfg"], world["behaviours"], world["manifest"], world["cache"], verd,
                    world["summary"], None)
    html = ld.render(an, _args(world))
    assert "<script>alert(" not in html                 # escaped, never live markup
    assert html.count("<script>") == 1                  # only the dashboard's own script
    red = ld.render(an, _args(world, redact=True))
    assert "Request 1" not in red and "says" not in red and "alert(" not in red
    if any(an.pairs[p].n for p in an.adjacent):
        assert "Text redacted" in red


def test_gate_stats_respects_short_circuit():
    th = {"cosine": .75, "bertscore": .85, "nli_entail": .6, "llm_intent": .5}
    m = Manifest()
    m.audit = [
        GateAuditRecord("b", 0, "paraphrase", True, True, cosine=.9, bertscore=.9, nli_entail=.8, llm_intent=1.0),
        GateAuditRecord("b", 1, "paraphrase", True, False, cosine=.5, bertscore=1.0, nli_entail=1.0, llm_intent=1.0),
        GateAuditRecord("b", 2, "lexical", True, False, cosine=.9, bertscore=.9, nli_entail=.3, llm_intent=1.0),
        GateAuditRecord("b", 3, "back_translation", True, False, cosine=1, bertscore=1, nli_entail=1, llm_intent=1),
    ]
    g = ld._gate_stats(m, th)
    assert g["rejected_by"] == {"cosine": 1, "nli_entail": 1, "lexical_divergence": 1}
    assert len(g["scores"]["cosine"]) == 3              # lexical-rejected row was never scored
    assert len(g["scores"]["bertscore"]) == 2           # cosine-rejected row stopped before BERTScore
    assert g["scores"]["nli_entail"] == [.8, .3]
    assert not g["intent_ran"] and g["scores"]["llm_intent"] == []


def test_verdict_store_survives_truncated_line(tmp_path):
    p = tmp_path / "v.jsonl"
    s = lc.VerdictStore(p)
    s.put("a" * 32, 1)
    s.close()
    with p.open("a") as fh:
        fh.write('{"k": "bbbb')                         # hard kill mid-line
    s = lc.VerdictStore(p)
    assert s.bad_lines == 1 and s.get("a" * 32) == 1
    s.put("c" * 32, 0)
    s.close()
    s = lc.VerdictStore(p)
    assert s.get("c" * 32) == 0 and s.bad_lines == 1


def test_verdict_key_depends_on_judge_identity():
    assert lc.verdict_key("rubric=a", "p", "r") != lc.verdict_key("rubric=b", "p", "r")
    assert lc.verdict_key("t", "p", "r") == lc.verdict_key("t", "p", "r")


@pytest.mark.parametrize("old,new,kind", [
    ("llama2-7b", "llama2-13b", "within-generation"),
    ("llama3-8b", "llama31-8b", "matched-scale"),
    ("llama2-7b", "llama3-8b", "matched-scale"),
    ("llama31-8b", "llama32-1b", "scale-confounded"),
    ("llama2-13b", "llama3-8b", "scale-confounded"),
])
def test_pair_status(old, new, kind):
    assert lc.pair_status(old, new)["kind"] == kind


def test_label_placement_never_overlaps():
    ys = ld._place_labels([(100,), (101,), (102,), (390,), (395,)], 0, 400, gap=30)
    s = sorted(ys)
    assert all(b - a >= 30 - 1e-9 for a, b in zip(s, s[1:]))
    assert min(ys) >= 8 and max(ys) <= 392


def test_sorry_taxonomy_is_complete():
    assert sorted(lc.SORRY_CATEGORIES) == list(range(1, 45))
    covered = [c for _, lo, hi in lc.SORRY_DOMAINS for c in range(lo, hi + 1)]
    assert covered == list(range(1, 45))


def test_incomplete_verdict_cache_falls_back_to_offline(world):
    store = lc.VerdictStore(world["tmp"] / "partial.jsonl")
    tag = lc.judge_tag_from_cfg(world["cfg"])
    todo = jl.todo_items(world["cfg"], world["behaviours"], world["manifest"], world["cache"],
                         MODELS, store, tag, "canonical")
    for k, prompt, resp in todo[: len(todo) // 2]:
        store.put(k, 1)
    store.close()
    args = argparse.Namespace(offline=False, verdicts=str(world["tmp"] / "partial.jsonl"))
    v, why = ld.pick_verdicts(args, world["cfg"], world["behaviours"], world["manifest"], world["cache"])
    assert v.kind == "offline" and "incomplete" in why
    _judge_all(world)
    args.verdicts = str(world["tmp"] / "verdicts.jsonl")
    v, _ = ld.pick_verdicts(args, world["cfg"], world["behaviours"], world["manifest"], world["cache"])
    assert v.kind == "judged"
    args.offline = True
    assert ld.pick_verdicts(args, world["cfg"], world["behaviours"], world["manifest"], world["cache"])[0].kind == "offline"


def test_summary_from_another_lineage_is_ignored(world):
    d = json.loads(world["summary"].read_text())
    d["lineage"] = ["someone-else"]
    other = world["tmp"] / "other.json"
    other.write_text(json.dumps(d))
    an = ld.analyse(world["cfg"], world["behaviours"], world["manifest"], world["cache"],
                    lc.OfflineVerdicts(), other, None)
    assert an.run == {}
    ld.render(an, _args(world))                       # renders from the reconstruction alone
