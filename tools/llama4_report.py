#!/usr/bin/env python3
"""tools/llama4_report.py -- canonical 2x2 of every earlier lineage model -> a Llama 4 model, from the caches.

No model calls: responses come from the response cache, verdicts from each benchmark's judge store
(the qwen2.5:14b rubric judge, the same tag every earlier run used). A behaviour enters a 2x2 only
when both models have a judged canonical verdict, so a partly judged run shows its coverage, not
a biased P^-. Exploratory (post hoc, scale-confounded): see configs/llama4_pegasus.yaml.

  python3 tools/llama4_report.py --out results/runs/<stamp>_llama4_scout/canonical.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_core as lc                                      # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.datasets import load_jailbreakbench          # noqa: E402
from recyclebench.manifest import Manifest                     # noqa: E402

# benchmark -> verdict store the earlier runs wrote (tools/judge_lineage.py, tools/judge_rpe.py)
STORES = {"sorrybench": "cache/judge_lineage_sorrybench.jsonl",
          "harmbench": "cache/judge_qwen_harmbench.jsonl",
          "advbench": "cache/judge_qwen_advbench.jsonl",
          "strongreject": "cache/judge_qwen_strongreject.jsonl"}
def report(cfg, bench: str, cache: dict, new: str = "llama4-scout") -> dict:
    behaviours = load_jailbreakbench(f"data/{bench}.csv")
    store = lc.VerdictStore(STORES[bench])
    verdicts = lc.StoredVerdicts(store, lc.judge_tag_from_cfg(cfg))
    models = list(cfg.lineage)
    obs, runs = lc.build_runs(cfg, behaviours, Manifest(), cache, verdicts, models)
    store.close()
    out = {"n_behaviours": len(behaviours), "models": {}, "pairs": {}}
    for m in models:
        r = runs[m]
        out["models"][m] = {"judged": len(r.canonical_obs), "complied": len(r.broken_set()),
                            "missing_responses": obs[m].missing_responses,
                            "missing_verdicts": obs[m].missing_verdicts["canonical"]}
    for old in models[:models.index(new)]:
        cells = lc.partition(runs[old], runs[new])
        out["pairs"][f"{old}->{new}"] = {k: len(v) for k, v in cells.items()} | {"pminus_bids": cells["pminus"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/llama4_pegasus.yaml")
    ap.add_argument("--cache", default="cache/llama_gen.jsonl")
    ap.add_argument("--benchmarks", nargs="*", default=[b for b in STORES if Path(f"data/{b}.csv").exists()])
    ap.add_argument("--new", default="llama4-scout", help="the newer model every earlier lineage model is paired with")
    ap.add_argument("--out", default=None, help="write the full report (with P^- ids) as JSON")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    cache = lc.load_response_cache(args.cache)
    full = {}
    for bench in args.benchmarks:
        rep = full[bench] = report(cfg, bench, cache, args.new)
        print(f"\n== {bench} ({rep['n_behaviours']} behaviours) ==")
        for m, s in rep["models"].items():
            rate = s["complied"] / s["judged"] if s["judged"] else float("nan")
            print(f"  {m:13s} judged {s['judged']:4d}  complied {s['complied']:4d} ({rate:.3f})"
                  f"  missing resp {s['missing_responses']}  missing verdicts {s['missing_verdicts']}")
        for pair, c in rep["pairs"].items():
            print(f"  {pair:28s} P^- {c['pminus']:3d}  persistent {c['persistent']:3d}"
                  f"  neg_flip {c['neg_flip']:3d}  both_safe {c['both_safe']:3d}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(full, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
