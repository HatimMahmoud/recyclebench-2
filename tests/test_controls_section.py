"""tools/controls_section.py: which run is picked, the contamination flag, the registry checks, text-free output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import controls_section as ctl                               # noqa: E402

R = lambda k, n: dict(k=k, n=n, rate=k / n if n else None, ci=[max(0, k / n - .05), min(1, k / n + .05)] if n else None)


def _model(k_safe, n_safe=250, missing=0):
    return dict(safe_refusal_judged=R(k_safe, n_safe), safe_refusal_offline=R(3, 250), safe_missing=missing,
                unsafe_refusal_judged=R(190, 200), unsafe_refusal_offline=R(150, 200), unsafe_missing=0,
                safe_by_type={"homonyms": R(1, 25)})


def _world(tmp, registered="01:25"):
    runs = tmp / "results" / "runs"
    a = runs / "overrefusal_xstest_20261031_0126_chain"
    b = runs / "overrefusal_xstest_20261031_0126_all"
    for d in (a, b):
        d.mkdir(parents=True)
    (a / "summary.json").write_text(json.dumps(dict(models={"llama3-8b": _model(46)}, pairs=[])))
    (b / "summary.json").write_text(json.dumps(dict(
        models={"llama3-8b": _model(46), "llama31-8b": _model(46), "llama32-1b": _model(0, 0, missing=250)},
        pairs=[dict(older="llama3-8b", newer="llama31-8b", n=250, older_rate=.184, newer_rate=.184,
                    newly_refused=12, newly_answered=12, mcnemar_p=1.0)])))
    c = runs / "contamination_20261031_0126"
    c.mkdir()
    cell = lambda k, n, lo, hi: dict(n=n, of=n, reproduced=k, rate=k / n, ci=[lo, hi], mean_suffix_matched=.1)
    (c / "summary.json").write_text(json.dumps(dict(controls=["sorrybench", "heldout_jbb_guided"], models={"llama3-8b": {
        "advbench": cell(1, 480, 0, .012), "harmbench": cell(18, 360, .032, .078), "strongreject": cell(0, 306, 0, .012),
        "sorrybench": cell(0, 426, 0, .009), "heldout_jbb_guided": cell(0, 211, 0, .018)}})))
    x = runs / "xfamily_mistral_20261031_0126"
    x.mkdir()
    (x / "pminus_sorrybench.json").write_text(json.dumps(dict(p_minus=25, negative_flips=34, old_complied=149,
                                                              new_complied=158, n=440, missing_verdicts=0)))
    (x / "summary_real.json").write_text(json.dumps(dict(pilot=dict(pooled_p_minus=25), gate_audit=dict(n_admitted=97, n_variants=200),
                                                         RSR_primary_RAW=[dict(k_recovered=15, n_p_minus=25)])))
    lin = tmp / ctl.LINEAGE_SUMMARY
    lin.parent.mkdir(parents=True)
    lin.write_text(json.dumps(dict(PBR_backbone_RAW=[dict(model="llama31-8b", k_bypassed=63, n_resolved=376)])))
    an = tmp / ctl.ANCHOR_CAL
    an.parent.mkdir(parents=True)
    an.write_text(json.dumps(dict(RSR=dict(RSR_raw=.489, RSR_corrected=.551, CI=[.364, .738], k_recovered=23, n_p_minus=47))))
    for f in ctl.PREREGS.values():
        (tmp / f).write_text(f"**Registered 2026-10-31 {registered}, before any data**\n")
    (tmp / "logs").mkdir()
    (tmp / "logs" / "extras_20261031_0126.log").write_text("\n=== 2026-10-31 01:26:50 · 1 over-refusal ===\n")
    return tmp


def test_picks_the_run_with_most_complete_models(tmp_path):
    agg = ctl.aggregate(_world(tmp_path))
    assert agg["overrefusal"]["path"].parent.name.endswith("_all")
    assert agg["overrefusal"]["complete"] == ["llama3-8b", "llama31-8b"]
    assert agg["overrefusal"]["partial"] == ["llama32-1b"]


def test_section_is_text_free_and_flags_only_above_both_controls(tmp_path):
    agg = ctl.aggregate(_world(tmp_path))
    h = ctl.section_body(agg) + ctl.xfamily_panel(agg) + ctl.models_panel(agg)
    assert "<script" not in h
    assert h.count("ctl-flag") == 1                     # HarmBench only: its lower bound 0.032 > the control high 0.018
    assert "15/25" in h and "97/200" in h               # Mistral RSR and gate admission
    assert "McNemar p 1" in ctl.pair_strip(agg, "llama3-8b", "llama31-8b")
    assert ctl.pair_strip(agg, "llama31-8b", "llama32-1b") == ""     # 1B not judged -> no strip


def test_exact_mcnemar_from_counts():
    assert ctl.mc_p(dict(newly_refused=12, newly_answered=12)) == 1.0
    assert ctl.mc_p(dict(newly_refused=3, newly_answered=95)) < 1e-20


def test_registry_flags_a_registration_after_launch(tmp_path):
    ok = {a[1]: a for a in ctl.artifacts(ctl.aggregate(_world(tmp_path / "a")))}
    assert all("AFTER launch" not in " ".join(a[5]) for a in ok.values())
    late = ctl.artifacts(ctl.aggregate(_world(tmp_path / "b", registered="01:40")))
    assert all("AFTER launch" in " ".join(a[5]) for a in late)


def test_empty_world_renders_placeholders(tmp_path):
    agg = ctl.aggregate(tmp_path)
    assert agg["overrefusal"] is None and ctl.xfamily_panel(agg) == "" and ctl.summary_tile(agg) == ""
    assert "Not run yet" in ctl.models_panel(agg) and "not run yet" in ctl.section_body(agg)
