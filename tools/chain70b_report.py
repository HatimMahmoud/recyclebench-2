#!/usr/bin/env python3
"""tools/chain70b_report.py -- canonical screen of the 70B chain on SORRY-Bench (PREREG_70B_CHAIN_20261105.md).

Answer rate per model (Wilson) and the 2x2 of each matched pair, from the chain's merged response cache and its own
verdict store. Counts and ids only; no model call. pegasus/chain70b.sh reads the newest report for eligibility.

  python3 tools/chain70b_report.py [--out results/runs/chain70b/canonical_<stamp>.json]
  python3 tools/chain70b_report.py --eligible recycle|rpe     # print the eligible old:new pairs, one per line
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import lineage_core as lc                                      # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.datasets import load_jailbreakbench          # noqa: E402
from recyclebench.manifest import Manifest                     # noqa: E402

CONFIG = "configs/chain70b.yaml"
CACHE = "cache/chain70b/canonical.jsonl"
STORE = "cache/chain70b/judge_sorrybench.jsonl"
DATA = "data/sorrybench.csv"
RUN_ROOT = "results/runs/chain70b"
PAIRS = (("llama2-70b", "llama3-70b"), ("llama3-70b", "llama31-70b"), ("llama31-70b", "llama33-70b"))
FLOORS = {"recycle": 5, "rpe": 1}


def report():
    cfg = Config.load(str(ROOT / CONFIG))
    beh = load_jailbreakbench(str(ROOT / DATA))
    store = lc.VerdictStore(str(ROOT / STORE))
    obs, runs = lc.build_runs(cfg, beh, Manifest(), lc.load_response_cache(str(ROOT / CACHE)),
                              lc.StoredVerdicts(store, lc.judge_tag_from_cfg(cfg)), list(cfg.lineage))
    store.close()
    out = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "prereg": "PREREG_70B_CHAIN_20261105.md",
           "designation": "exploratory, post hoc; raw judge output", "config": CONFIG, "cache": CACHE, "store": STORE,
           "judge": lc.judge_tag_from_cfg(cfg), "n_behaviours": len(beh), "models": {}, "pairs": {}}
    for m in cfg.lineage:
        r = runs[m]
        out["models"][m] = {"judged": len(r.canonical_obs), "complied": len(r.broken_set()),
                            "missing_responses": obs[m].missing_responses,
                            "missing_verdicts": obs[m].missing_verdicts["canonical"]}
    for o, n in PAIRS:
        c = lc.partition(runs[o], runs[n])
        complete = all(out["models"][m]["missing_responses"] == 0 and out["models"][m]["missing_verdicts"] == 0
                       for m in (o, n))
        out["pairs"][f"{o}->{n}"] = {k: len(v) for k, v in c.items()} | {"complete": complete,
                                                                         "pminus_bids": sorted(c["pminus"])}
    return out


def latest():
    xs = sorted((ROOT / RUN_ROOT).glob("canonical_*.json"))
    return xs[-1] if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eligible", choices=list(FLOORS), default=None)
    a = ap.parse_args(argv)
    if a.eligible:
        p = latest()
        if not p:
            raise SystemExit("no results/runs/chain70b/canonical_*.json yet")
        rep = json.loads(p.read_text(encoding="utf-8"))
        for k, c in rep["pairs"].items():
            if c["complete"] and c["pminus"] >= FLOORS[a.eligible]:
                print(k.replace("->", ":"))
        return 0
    rep = report()
    for m, s in rep["models"].items():
        print(f"  {m:12s} judged {s['judged']:4d}  complied {s['complied']:4d}  missing resp {s['missing_responses']}"
              f"  missing verdicts {s['missing_verdicts']}")
    for k, c in rep["pairs"].items():
        print(f"  {k:26s} P^- {c['pminus']:3d}  persistent {c['persistent']:3d}  neg_flip {c['neg_flip']:3d}"
              f"  both_safe {c['both_safe']:3d}{'' if c['complete'] else '  (incomplete)'}")
    out = Path(a.out) if a.out else ROOT / RUN_ROOT / f"canonical_{_dt.datetime.now():%Y%m%d_%H%M}.json"
    if out.exists():
        raise SystemExit(f"{out} exists -- refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
