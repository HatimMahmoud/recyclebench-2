"""The inversion (RPE) arm's offline layer: the gate store, gate-aware closed-loop scoring,
reuse across pairs, the harmful-ask audit, the dashboards' loader/renderer, and the
reconciliation of the 38/49 vs 7/61 gap. No model is called."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import rpe_gate as rg                                       # noqa: E402
import rpe_closedloop as cl                                 # noqa: E402
import rpe_lineage as rl                                    # noqa: E402
import rpe_audit_score as ras                               # noqa: E402
import inversion_section as invsec                          # noqa: E402
import lineage_core as lc                                   # noqa: E402
from recyclebench.config import Config                      # noqa: E402
from recyclebench.realrun import ResponseCache              # noqa: E402

CFG = Config.load(ROOT / "configs/real_llama_lineage_all.yaml")
SIG = rg.signature(CFG)
SECRET = "SECRET-RECONSTRUCTION"
PROMPTS = {"jbb-0001": "original request one", "jbb-0002": "original request two", "jbb-0003": "original request three"}


def _manifest(path, pair=("old-m", "new-m")):
    rows = []
    for bid, pr in PROMPTS.items():
        rows.append({"bid": bid, "benchmark": "sorrybench", "pair": list(pair), "prompt": pr,
                     "old_response": f"answer to {pr}",
                     "details": [{"text": f"{SECRET} {bid} #{ci}", "admitted": False} for ci in range(2)]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_gate_store_roundtrip_and_truncation(tmp_path):
    store = tmp_path / "gate.jsonl"
    rec = {"k": rg.key(SIG, "p", "t"), "sig": SIG, "admitted": True, "rejected_by": None, "layers": {"cosine": 0.9}}
    store.write_text(json.dumps(rec) + "\n" + '{"k": "trunc')          # a hard kill mid-write
    st = rg.load_store(store)
    assert rg.lookup(st, SIG, "p", "t")["admitted"] is True
    assert rg.lookup(st, SIG, "p", "other") is None
    assert rg.key(SIG, "p", "t") != rg.key(SIG.replace("cos>=0.75", "cos>=0.7"), "p", "t")   # a moved threshold re-keys


def test_manifest_candidates_dedupe(tmp_path):
    m = tmp_path / "m.jsonl"
    _manifest(m)
    cands = rg.manifest_candidates([m, m])
    assert len(cands) == 6 and len(set(cands)) == 6


def test_apply_gate_store_overrides_the_manifest_flag(tmp_path):
    m = tmp_path / "rpe_manifest_old-m__to__new-m.jsonl"
    _manifest(m)
    store = tmp_path / "gate.jsonl"
    store.write_text(json.dumps({"k": rg.key(SIG, PROMPTS["jbb-0001"], f"{SECRET} jbb-0001 #0"), "admitted": True,
                                 "rejected_by": None, "layers": {}}) + "\n"
                     + json.dumps({"k": rg.key(SIG, PROMPTS["jbb-0002"], f"{SECRET} jbb-0002 #1"), "admitted": False,
                                   "rejected_by": "nli_entail", "layers": {}}) + "\n")
    cands, why, missing = cl.apply_gate_store(cl.candidates(m), store, CFG)
    flags = {(c[0], c[1]): c[3] for c in cands}
    assert flags[("jbb-0001", 0)] is True and flags[("jbb-0002", 1)] is False
    assert why[("jbb-0002", 1)] == "nli_entail" and missing == 4


class _Model:
    name = "old-m"

    def __init__(self):
        self.calls = 0

    def respond(self, text, seed=None):
        self.calls += 1
        return type("R", (), {"text": "fresh"})()


def test_reuse_cache_copies_instead_of_calling(tmp_path):
    m = tmp_path / "rpe_manifest_old-m__to__new-m.jsonl"
    _manifest(m)
    cands = cl.candidates(m)
    src = ResponseCache(tmp_path / "src.jsonl")
    for bid, ci, text, *_ in cands[:4]:
        src.put("old-m", text, cl.seed_for(bid, ci), f"cached {bid}#{ci}")
    dst, model = ResponseCache(tmp_path / "dst.jsonl"), _Model()
    cl.phase_generate(cands, model, dst, "old", reuse=[ResponseCache(tmp_path / "src.jsonl")])
    assert model.calls == 2                                     # only the two the source lacked
    bid, ci, text, *_ = cands[0]
    assert dst.get("old-m", text, cl.seed_for(bid, ci)) == f"cached {bid}#{ci}"


def test_reusable_inversions_match_on_the_old_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "ROOT", tmp_path)
    d = tmp_path / "runs"
    d.mkdir()
    _manifest(d / "rpe_manifest_old-m__to__x.jsonl")
    _manifest(d / "rpe_manifest_other__to__x.jsonl")              # a different OLD model never matches
    reuse = rl.reusable_inversions(["runs/rpe_manifest_*.jsonl"], "old-m")
    k = ("jbb-0001", PROMPTS["jbb-0001"], f"answer to {PROMPTS['jbb-0001']}")
    assert k in reuse and reuse[k][0] == [f"{SECRET} jbb-0001 #0", f"{SECRET} jbb-0001 #1"]
    assert all(src.endswith("rpe_manifest_old-m__to__x.jsonl") for _, src in reuse.values())


def _world(tmp_path):
    """A one-pair tree: manifest, gate store, gated closed-loop artifact, report, harmful-ask sheet."""
    base = tmp_path / "results/runs/rpe_lineage"
    base.mkdir(parents=True)
    (base / "pairs.json").write_text(json.dumps([{"old": "llama3-8b", "new": "llama31-8b", "p_minus": 3},
                                                 {"old": "llama2-13b", "new": "llama3-8b", "p_minus": 0}]))
    _manifest(base / "rpe_manifest_llama3-8b__to__llama31-8b.jsonl")
    gate = tmp_path / "cache/rpe_gate.jsonl"
    gate.parent.mkdir(parents=True)
    lines = []
    for bid, pr in PROMPTS.items():
        for ci in range(2):
            adm = bid == "jbb-0001" and ci == 0
            lines.append(json.dumps({"k": rg.key(SIG, pr, f"{SECRET} {bid} #{ci}"), "admitted": adm,
                                     "rejected_by": None if adm else "cosine", "layers": {}}))
    gate.write_text("\n".join(lines) + "\n")
    row = lambda bid, ci, gate_, loop, v: {"ci": ci, "text": f"{SECRET} {bid} #{ci}", "gate": gate_, "loop": loop,
                                           "equiv": 0.8, "new_verdict": v, "old_verdict": 1}
    per = {"jbb-0001": {"prompt": PROMPTS["jbb-0001"], "gate": [row("jbb-0001", 0, True, True, 1)],
                        "loop": [row("jbb-0001", 0, True, True, 1), row("jbb-0001", 1, False, True, 1)],
                        "both": [row("jbb-0001", 0, True, True, 1)]},
           "jbb-0002": {"prompt": PROMPTS["jbb-0002"], "gate": [], "loop": [row("jbb-0002", 0, False, True, 1)], "both": []},
           "jbb-0003": {"prompt": PROMPTS["jbb-0003"], "gate": [], "loop": [row("jbb-0003", 1, False, True, 0)], "both": []}}
    adm = {"gate": {"k": 1, "n": 1, "unresolved": 2, "rate": 1.0, "ci": [0.2, 1.0]},
           "loop": {"k": 2, "n": 3, "unresolved": 0, "rate": 2 / 3, "ci": [0.2, 0.94]},
           "both": {"k": 1, "n": 1, "unresolved": 2, "rate": 1.0, "ci": [0.2, 1.0]}}
    clp = base / "closedloop_llama3-8b__to__llama31-8b_gated_20990101_0000.json"
    clp.write_text(json.dumps({"generated": "20990101_0000", "pair": ["llama3-8b", "llama31-8b"],
                               "gate_source": "cache/rpe_gate.jsonl", "n_candidates": 6, "admission": adm,
                               "per_behaviour": per}))
    (base / "lineage_report_20990101_0000.json").write_text(json.dumps({"views": [
        {"pair": ["llama3-8b", "llama31-8b"], "p_minus": 3,
         "RSR": {"k": 1, "n": 3, "rate": 1 / 3, "ci": [0.06, 0.79]},
         "agreement": {"mcnemar_p": 0.5}, "agreement_both": {"mcnemar_p": 1.0}}]}))
    audit = tmp_path / "audit/rpe"
    audit.mkdir(parents=True)
    rel = str(clp.relative_to(tmp_path))
    with (audit / "rpe_harm_sheet_20990101_0000.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["row", "original_request", "reconstruction", ras.HARM_COL, "note"])
        w.writeheader()
        for i, lab in ((1, "0"), (2, "1"), (3, "0"), (4, "")):
            w.writerow({"row": i, "original_request": "o", "reconstruction": SECRET, ras.HARM_COL: lab, "note": ""})
    with (audit / "rpe_harm_key_20990101_0000.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["row", "benchmark", "pair", "bid", "candidate_index", "gate_admitted",
                                           "equiv_cosine", "closedloop"])
        w.writeheader()
        for i, bid in ((1, "jbb-0001"), (2, "jbb-0001"), (3, "jbb-0002"), (4, "jbb-0002")):
            w.writerow({"row": i, "benchmark": "sorrybench", "pair": "llama3-8b->llama31-8b", "bid": bid,
                        "candidate_index": 0, "gate_admitted": False, "equiv_cosine": 0.8, "closedloop": rel})
    (audit / "rpe_harm_design_20990101_0000.json").write_text(json.dumps(
        {"design": "reopened", "per_behaviour": 2, "population": {"behaviours_reopened": 2, "reopening_candidates": 3}}))
    return tmp_path


def test_harm_bracket_lower_and_upper(tmp_path):
    root = _world(tmp_path)
    h = ras.status(root / "audit/rpe", root=root)["harm"]
    p = h["pairs"][0]
    # jbb-0001: labels 0 and 1 -> verified; jbb-0002: one 0, one blank -> pending, never refuted
    assert (p["verified"], p["refuted"], p["pending"]) == (1, 0, 1)
    assert (p["lower"]["k"], p["lower"]["n"]) == (1, 3) and (p["upper"]["k"], p["upper"]["n"]) == (2, 3)
    assert h["labelled"] == 3 and h["candidate_preserved"]["k"] == 1


def test_loader_and_master_body_are_text_free(tmp_path):
    root = _world(tmp_path)
    inv = invsec.load(root=root)
    r = inv.get("sorrybench", "llama3-8b", "llama31-8b")
    assert r.gated and r.n_candidates == 6 and r.rejections == {"admitted": 1, "cosine": 5}
    assert inv.get("sorrybench", "llama2-13b", "llama3-8b").p_minus == 0
    body = invsec.master_body(inv) + invsec.master_notes(inv) + invsec.lineage_table(inv)
    assert SECRET not in body and "<script" not in body and "<style" not in body
    assert "2/3" in body and 'role="tablist"' not in body           # one benchmark: no tabs
    assert invsec.status_line(inv, "sorrybench", "llama3-8b", "llama31-8b") == ("ok", "PRR 2/3 closed loop · 1/1 gated")
    assert invsec.status_line(inv, "sorrybench", "llama2-13b", "llama3-8b") is None       # P^- = 0: nothing to say
    assert invsec.has_run(inv, "sorrybench", "llama2-13b", "llama3-8b")                  # P^- = 0 is finished


def test_pair_fix_respects_redact(tmp_path):
    root = _world(tmp_path)
    inv = invsec.load(root=root)
    cap = lambda t, n: (t or "")[:n]
    caps = {"cand": 200, "creply": 200}
    pill = lambda v: f"<i>{v}</i>"
    shown = invsec.pair_fix(inv, "sorrybench", "llama3-8b", "llama31-8b", PROMPTS["jbb-0001"], False, cap, caps, pill, "new")
    hidden = invsec.pair_fix(inv, "sorrybench", "llama3-8b", "llama31-8b", PROMPTS["jbb-0001"], True, cap, caps, pill, "new")
    assert SECRET in shown and SECRET not in hidden and "gate ✓" in hidden and "loop ✓" in hidden


REAL = (ROOT / "results/runs/recycle_3_31_sorrybench/manifest_sorrybench.jsonl").exists() and \
    (ROOT / "results/cells_llama3-8b_llama31-8b/sorrybench_p_minus.csv").exists() and \
    (ROOT / "cache/llama_gen.jsonl").exists()


@pytest.mark.skipif(not REAL, reason="anchor manifest / cache not on disk")
def test_anchor_manifest_is_the_offline_pminus_by_prompt_text():
    import rpe_reconcile as rr
    from recyclebench.datasets import load_jailbreakbench
    full = load_jailbreakbench(str(ROOT / "data/sorrybench.csv"))
    sub = load_jailbreakbench(str(ROOT / "results/cells_llama3-8b_llama31-8b/sorrybench_p_minus.csv"))
    ids = rr.anchor_manifest_ids(full, sub, lc.load_response_cache(ROOT / "cache/llama_gen.jsonl"), 3)
    assert ids["same_prompts_as_full_csv_offline_pminus"] is True
    assert (ids["behaviours"], ids["old_complies_on_its_own_prompts"]) == (61, 61)
    assert ids["old_refuses_on_full_csv_prompts_sharing_its_ids"] == 52     # the old 'unreproduced' figure


def test_reconcile_joins_compose_ids_on_prompt_text(tmp_path):
    """The compose run's ids are subset-CSV row positions: its resolved set must be compared with the
    dedicated arm's by prompt text, never by bid."""
    import rpe_reconcile as rr
    store = tmp_path / "compose"
    store.mkdir()
    gen = [{"id": "rpe|jbb-0000|0", "arm": "rpe", "bid": "jbb-0000", "idx": 0, "ok": True, "gate_admitted": True},
           {"id": "rpe|jbb-0001|0", "arm": "rpe", "bid": "jbb-0001", "idx": 0, "ok": True, "gate_admitted": False},
           {"id": "rpe|jbb-0000|5", "arm": "rpe", "bid": "jbb-0000", "idx": 5, "ok": True, "gate_admitted": True}]
    (store / "candidates.jsonl").write_text("".join(json.dumps(d) + "\n" for d in gen))
    (store / "judged.jsonl").write_text(json.dumps({"id": "rpe|jbb-0000|0|llama31-8b|judge", "complied": True}) + "\n")
    # row 0 of the subset is SORRY-Bench's jbb-0001: same prompt, different id
    comp = rr.compose_side(store, 3, {"jbb-0000": PROMPTS["jbb-0001"], "jbb-0001": PROMPTS["jbb-0002"]})
    assert comp["resolved"] == {PROMPTS["jbb-0001"]} and comp["opened"] == {PROMPTS["jbb-0001"]}
    assert (comp["candidates"], comp["admitted"]) == (2, 1)           # idx 5 is outside the 3-candidate budget
    root = _world(tmp_path / "w")
    cl_path = next((root / "results/runs/rpe_lineage").glob("closedloop_*_gated_*.json"))
    side = rr.closedloop_side(cl_path, n_cands=1, only={PROMPTS["jbb-0001"]})
    assert side["behaviours"] == 1
    assert (side["admission"]["gate"]["resolved"], side["admission"]["loop"]["resolved"]) == (1, 1)
