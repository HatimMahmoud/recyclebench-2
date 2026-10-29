#!/usr/bin/env python3
"""rpe_lineage_report.py -- convergent validity across the lineage.

Two independent routes to the same question -- "is the newer model's fix tied to the
wording?" -- scored on the SAME judged P^- of the SAME pair:

  recycle arm    perturb the REQUEST (paraphrase) and re-test            -> RSR
  inversion arm  reconstruct a request from the OLD model's ANSWER       -> PRR
                 and re-test, admitting by preimage membership

The arms share no generator: the recycle arm never sees an answer, the inversion arm
never sees the original request. Agreement across pairs is therefore evidence that the
estimand is real rather than an artifact of the paraphraser. Disagreement localises
which arm is measuring something else.

Per pair this reports RSR, PRR, their overlap on behaviours both arms resolved, Cohen's
kappa and McNemar exact. Everything is recomputed from the frozen manifest, the response
cache and the judge verdict store through the package estimators -- no stored summary is
trusted.

Offline: no model calls.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import lineage_core as lc                                        # noqa: E402
from recyclebench.config import Config                           # noqa: E402
from recyclebench.datasets import load_jailbreakbench            # noqa: E402
from recyclebench.independent_probe.analyze_agreement import analyze   # noqa: E402
from recyclebench.independent_probe.contract import ProbeItem    # noqa: E402
from recyclebench.manifest import Manifest                       # noqa: E402
from recyclebench.stats import wilson_interval                   # noqa: E402

LABEL = {"llama2-7b": "2-7B", "llama2-13b": "2-13B", "llama3-8b": "3-8B",
         "llama31-8b": "3.1-8B", "llama32-1b": "3.2-1B", "llama32-3b": "3.2-3B"}


def recycle_side(cfg, behaviours, manifest, cache, verdicts, old, new):
    """-> (p_minus, {bid: 0/1/None}) exactly as the recycle arm scores it."""
    obs = lc.observe(cfg, behaviours, manifest, cache, verdicts, [old, new])
    runs = {m: lc.to_model_run(obs[m], behaviours) for m in (old, new)}
    part = lc.partition(runs[old], runs[new])
    pm = list(part["pminus"])
    rec = {}
    for bid in pm:
        vs = obs[new].variants.get(bid, [])
        rec[bid] = lc._any([v.verdict for v in vs]) if vs else None
    return pm, rec


ADMISSIONS = ("loop", "both", "gate")


def closedloop_path(root: Path, old: str, new: str):
    """Newest gate-scored artifact for the pair, else the original (A_loop-only) one.

    `closedloop_<o>__to__<n>_gated_<stamp>.json` is written by `rpe_closedloop.py --gate-store`;
    the plain name predates it and carries an empty A_gate (the gate never ran).
    """
    gated = sorted(root.glob(f"closedloop_{old}__to__{new}_gated_*.json"))
    if gated:
        return gated[-1]
    p = root / f"closedloop_{old}__to__{new}.json"
    return p if p.exists() else None


def inversion_side(path, admission: str = "loop"):
    """-> {bid: 0/1/None} from a closed-loop artifact under one admission set.

    loop = OLD(x_hat) complies and lands in the answer's equivalence class (upper bound);
    gate = passed the intent gate against x (the admission RSR uses);
    both = the intersection (primary).
    """
    if path is None or not Path(path).exists():
        return None
    doc = json.loads(Path(path).read_text())
    out = {}
    for bid, d in doc.get("per_behaviour", {}).items():
        rows = d.get(admission) or []
        out[bid] = lc._any([r.get("new_verdict") for r in rows]) if rows else None
    return out


def agreement(pm, rec, inv):
    items = []
    for b in pm:
        it = ProbeItem(bid=b, benchmark="", prompt="", old_response="")
        it.para_recovered, it.rpe_recovered = rec.get(b), inv.get(b)
        items.append(it)
    rep = analyze(items)
    return {"n_both_resolved": rep.n_both_resolved, "rate": rep.agreement_rate,
            "cohen_kappa": rep.cohen_kappa, "mcnemar_p": rep.mcnemar_p,
            "both_recovered": rep.both_recovered, "both_held": rep.both_held,
            "rpe_only": rep.rpe_only, "para_only": rep.para_only}


def rate(d, keys):
    vals = [d.get(b) for b in keys]
    res = [v for v in vals if v is not None]
    if not res:
        return {"k": 0, "n": 0, "rate": None, "ci": [None, None], "unresolved": len(vals)}
    k, n = sum(res), len(res)
    iv = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci": [iv.low, iv.high],
            "unresolved": len(vals) - n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/real_llama_lineage_all.yaml")
    ap.add_argument("--behaviours", default="data/sorrybench.csv")
    ap.add_argument("--manifest", default="results/runs/lineage_sorrybench_20261020/manifest.jsonl")
    ap.add_argument("--cache", default="cache/llama_gen.jsonl")
    ap.add_argument("--verdicts", default="cache/judge_lineage_sorrybench.jsonl")
    ap.add_argument("--rpe-root", default="results/runs/rpe_lineage")
    ap.add_argument("--pairs", default=None,
                    help="comma list old:new; default = adjacent rungs of cfg.lineage")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    tag = lc.judge_tag_from_cfg(cfg)
    behaviours = load_jailbreakbench(args.behaviours)
    manifest = Manifest.load(args.manifest)
    cache = lc.load_response_cache(args.cache)
    verdicts = lc.StoredVerdicts(lc.VerdictStore(args.verdicts), tag)
    if args.pairs:
        pairs = [tuple(p.split(":", 1)) for p in args.pairs.split(",")]
    else:
        L = cfg.lineage or []
        pairs = list(zip(L, L[1:]))

    print(f"judge    : {tag}")
    print(f"{'pair':<18}{'P^-':>5}{'RSR recycle':>22}{'PRR inversion':>22}"
          f"{'both':>6}{'kappa':>8}{'McNemar':>10}")
    rows, views = [], []
    for old, new in pairs:
        pm, rec = recycle_side(cfg, behaviours, manifest, cache, verdicts, old, new)
        path = closedloop_path(Path(args.rpe_root), old, new)
        lbl = f"{LABEL.get(old, old)} -> {LABEL.get(new, new)}"
        r = rate(rec, pm)
        if path is None:
            print(f"{lbl:<18}{len(pm):>5}{_fmt(r):>22}{'not scored yet':>22}")
            views.append({"pair": [old, new], "p_minus": len(pm), "RSR": r,
                          "PRR": None, "agreement": None, "closedloop": None})
            continue
        doc = json.loads(path.read_text())
        view = {"pair": [old, new], "p_minus": len(pm), "RSR": r,
                "closedloop": _rel(path), "gate_source": doc.get("gate_source", "manifest flag")}
        for adm in ADMISSIONS:
            inv = inversion_side(path, adm)
            sfx = "" if adm == "loop" else f"_{adm}"
            view["PRR" + sfx] = rate(inv, pm)
            view["agreement" + sfx] = agreement(pm, rec, inv)
        q, ag = view["PRR"], view["agreement"]
        kap = "n/a" if ag["cohen_kappa"] is None else f"{ag['cohen_kappa']:.2f}"
        mcn = "n/a" if ag["mcnemar_p"] is None else f"{ag['mcnemar_p']:.3g}"
        print(f"{lbl:<18}{len(pm):>5}{_fmt(r):>22}{_fmt(q):>22}"
              f"{ag['n_both_resolved']:>6}{kap:>8}{mcn:>10}"
              f"   both {_fmt(view['PRR_both'])}   gate {_fmt(view['PRR_gate'])}")
        views.append(view)

    out = {"generated": _dt.datetime.now().strftime("%Y%m%d_%H%M"), "judge_tag": tag,
           "note": "per-pair only: a behaviour appears in several pairs' P^-, so pooling "
                   "across pairs would count it more than once",
           "admissions": {"PRR": "A_loop: OLD(x_hat) complies and ~= y (upper bound until the "
                                 "harmful-ask audit)",
                          "PRR_both": "A_both: A_loop and the intent gate (primary)",
                          "PRR_gate": "A_gate: the intent gate against x (the admission RSR uses)"},
           "views": views}
    dest = Path(args.out) if args.out else \
        ROOT / "results" / "runs" / "rpe_lineage" / f"lineage_report_{out['generated']}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise SystemExit(f"{dest} exists -- refusing to overwrite")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote    : {dest}")
    return 0


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _fmt(r):
    if r["rate"] is None:
        return f"undefined ({r['unresolved']} unres)"
    return f"{r['rate']*100:.1f}% {r['k']}/{r['n']}"


if __name__ == "__main__":
    sys.exit(main())
