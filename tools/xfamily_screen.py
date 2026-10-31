#!/usr/bin/env python3
"""tools/xfamily_screen.py -- canonical screen of the cross-family pair, judged, with stored verdicts.

Registered in PREREG_XFAMILY_MISTRAL_20261031.md. Both models of configs/xfamily_mistral.yaml answer every
SORRY-Bench request (k = 1 draw, the study's seed `_draw_seed(bid, -1, 0)`, temperature 0, max_tokens 512);
the config's judge grades every reply against the request; verdicts are stored (unlike `run-real`). Writes the
judged P⁻ as a subset CSV (bid,Category,Goal -- the input `run-real` takes, as for the anchor) and a summary.

  python3 tools/xfamily_screen.py --phase all --out results/runs/xfamily_mistral_<stamp>
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import lineage_core as lc                                             # noqa: E402
from overrefusal import log, phase_gen, phase_judge                   # noqa: E402
from recyclebench.config import Config                                # noqa: E402
from recyclebench.datasets import load_jailbreakbench                 # noqa: E402
from recyclebench.realrun import ResponseCache                        # noqa: E402
from recyclebench.screening import _draw_seed                         # noqa: E402

CFG = "configs/xfamily_mistral.yaml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=("gen", "score", "all"))
    ap.add_argument("--config", default=CFG)
    ap.add_argument("--behaviours", default="data/sorrybench.csv")
    ap.add_argument("--cache", default="cache/xfamily_mistral.jsonl")
    ap.add_argument("--verdicts", default="cache/judge_xfamily_mistral.jsonl")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N behaviours (smoke test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = Config.load(ROOT / args.config)
    assert cfg.budget.k_canonical_samples == 1, "this screen draws once; the config must say k = 1"
    old, new = cfg.lineage
    behaviours = load_jailbreakbench(ROOT / args.behaviours)[:args.limit] if args.limit else load_jailbreakbench(ROOT / args.behaviours)
    items = [(b.bid, b.canonical_prompt, b.category, "") for b in behaviours]
    cache = ResponseCache(ROOT / args.cache)
    store = lc.VerdictStore(ROOT / args.verdicts)
    tag = lc.judge_tag_from_cfg(cfg)
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    log(f"x-family screen · {old} -> {new} · {len(items)} behaviours · judge {tag}")
    try:
        if args.phase in ("gen", "all"):
            for m in (old, new):
                if not phase_gen(m, cfg.real[m], items, cache, deadline, max_tokens=cfg.real[m]["max_tokens"]):
                    return 0
                if not phase_judge(cfg, m, items, cache, store, tag, deadline):
                    return 0
        if args.phase in ("score", "all"):
            v = {}
            for m in (old, new):
                for b in behaviours:
                    t = cache.get(m, b.canonical_prompt, _draw_seed(b.bid, -1, 0))
                    v[m, b.bid] = None if t is None else store.get(lc.verdict_key(tag, b.canonical_prompt, t))
            missing = sum(x is None for x in v.values())
            pm = [b for b in behaviours if v[old, b.bid] == 1 and v[new, b.bid] == 0]
            neg = [b for b in behaviours if v[old, b.bid] == 0 and v[new, b.bid] == 1]
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            with (out / "sorrybench_p_minus.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["bid", "Category", "Goal"])
                for b in pm:
                    w.writerow([b.bid, b.category, b.canonical_prompt])
            s = dict(created=_dt.datetime.now().isoformat(timespec="seconds"), old=old, new=new, judge=tag, k=1,
                     n=len(behaviours), missing_verdicts=missing,
                     old_complied=sum(v[old, b.bid] == 1 for b in behaviours),
                     new_complied=sum(v[new, b.bid] == 1 for b in behaviours),
                     p_minus=len(pm), negative_flips=len(neg), p_minus_bids=[b.bid for b in pm],
                     prereg="PREREG_XFAMILY_MISTRAL_20261031.md")
            (out / "pminus_sorrybench.json").write_text(json.dumps(s, indent=1))
            print(json.dumps({k: x for k, x in s.items() if k != "p_minus_bids"}, indent=1))
    finally:
        store.close()
        cache.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
