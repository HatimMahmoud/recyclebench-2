#!/usr/bin/env python3
"""tools/judge_lineage.py -- re-grade a cached lineage run with the config's judge, persistently.

`run-real` judges every response but never stores the verdicts, so a dashboard rebuilt from
the cache can only fall back to the offline refusal-prefix rule -- which disagrees with the
qwen2.5:14b rubric judge (on the SORRY-Bench lineage it gives P^- 14/2/61/19/33 instead of
the run's 18/3/49/12/13). This tool replays the SAME judge (the config's `judges:` block,
built through recyclebench.backends.real_judge.build_ensemble) over the cached responses and
writes every verdict to an append-only cache, so lineage_dashboard.py can reproduce the run.

NO target-model calls: responses come from the response cache. Only the judge runs.
Deduplicated on (prompt, response) -- identical draws are judged once. Resumable: Ctrl-C and
re-run the same command. Never run two copies against the same --out at once.

  # how much work is left? (no model calls)
  python3 tools/judge_lineage.py --config configs/real_llama_lineage_all.yaml \\
      --cache cache/llama_gen.jsonl --behaviours data/sorrybench.csv \\
      --manifest results/runs/lineage_sorrybench_20261020/manifest.jsonl \\
      --out cache/judge_lineage_sorrybench.jsonl --dry-run

  # judge what RSR needs first (all canonical draws + new-model variants on each pair's P^-),
  # then everything else (all variants, needed for PBR)
  OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h caffeinate -is \\
  python3 tools/judge_lineage.py ...same args... --scope rsr
  python3 tools/judge_lineage.py ...same args... --scope all

Verdicts are keyed on the judge identity (role=model_id) + prompt + response, so a different
judge never reuses another judge's verdicts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_core as lc                                      # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.datasets import load_jailbreakbench          # noqa: E402
from recyclebench.manifest import Manifest                     # noqa: E402
from recyclebench.types import Response                        # noqa: E402


def build_judge(cfg):
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    from recyclebench.backends.real_judge import build_ensemble
    specs = cfg.judges or {}
    if not specs:
        raise SystemExit("config has no `judges:` block -- nothing to replay. Use the config the "
                         "lineage run used (configs/real_llama_lineage_all.yaml).")
    jm = {role: ProviderModel(f"judge:{role}", ProviderConfig(**spec)) for role, spec in specs.items()}
    judge = build_ensemble(jm)
    if not judge.members:
        raise SystemExit(f"`judges:` roles {sorted(specs)} map to no ensemble member "
                         "(expected rubric / guard / harmbench)")
    return judge


def todo_items(cfg, behaviours, manifest, cache, models, store, tag, scope):
    """Unique (prompt, response) pairs still lacking a verdict, in priority order."""
    seen, out = set(), []

    def want(prompt, resp):
        if resp is None:
            return
        k = lc.verdict_key(tag, prompt, resp)
        if k in seen or store.get(k) is not None:
            return
        seen.add(k)
        out.append((k, prompt, resp))

    items = list(lc.iter_items(cfg, behaviours, manifest, cache, models))
    for it in items:                                     # 1. every canonical draw
        if it.kind == "canonical":
            want(it.prompt, it.response)
    if scope == "canonical":
        return out
    if scope == "rsr":
        # 2. variants of the NEW model on each consecutive pair's P^-, under the verdicts we have
        verd = lc.StoredVerdicts(store, tag)
        obs = lc.observe(cfg, behaviours, manifest, cache, verd, models)
        runs = {m: lc.to_model_run(obs[m], behaviours) for m in models}
        need = set()
        for o, n in zip(models, models[1:]):
            for bid in lc.regression_set(runs[o], runs[n]):
                need.add((n, bid))
        for it in items:
            if it.kind == "variant" and (it.model, it.bid) in need:
                want(it.prompt, it.response)
        return out
    for it in items:                                     # 3. every variant (PBR needs all)
        if it.kind == "variant":
            want(it.prompt, it.response)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/real_llama_lineage_all.yaml")
    ap.add_argument("--cache", default="cache/llama_gen.jsonl")
    ap.add_argument("--behaviours", default="data/sorrybench.csv")
    ap.add_argument("--manifest", default="results/runs/lineage_sorrybench_20261020/manifest.jsonl")
    ap.add_argument("--out", default="cache/judge_lineage_sorrybench.jsonl")
    ap.add_argument("--models", nargs="*", default=None, help="subset of cfg lineage (default: all)")
    ap.add_argument("--scope", choices=["canonical", "rsr", "all"], default="all",
                    help="canonical = every canonical draw (P^- membership); rsr = canonical, then "
                         "the new model's variants on each pair's P^- (RSR); all = canonical, then "
                         "every variant of every model (RSR + PBR). Phases run in that order.")
    ap.add_argument("--limit", type=int, default=None, help="judge at most N items this run")
    ap.add_argument("--stop-at", default=None, metavar="HH:MM", help="stop gracefully at this local time")
    ap.add_argument("--dry-run", action="store_true", help="count remaining work; no judge calls")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    models = args.models or list(cfg.lineage)
    unknown = [m for m in models if m not in (cfg.real or {})]
    if unknown:
        raise SystemExit(f"models not in config `real:`: {unknown}")
    tag = lc.judge_tag_from_cfg(cfg)
    if not tag:
        raise SystemExit("config has no `judges:` block")
    behaviours = load_jailbreakbench(args.behaviours)
    manifest = Manifest.load(args.manifest)
    cache = lc.load_response_cache(args.cache)
    store = lc.VerdictStore(args.out)
    if store.bad_lines:
        print(f"  [verdicts] ignored {store.bad_lines} malformed line(s) in {args.out}")

    print(f"judge      : {tag}")
    print(f"models     : {' '.join(models)}")
    print(f"scope      : {args.scope}")
    print(f"stored     : {len(store)} verdicts in {args.out}")

    # P^- membership depends on the canonical verdicts, so canonical is always phase 1 and the
    # rsr/all work list is recomputed after it -- one invocation is enough.
    phases = ["canonical"] + ([args.scope] if args.scope != "canonical" else [])
    budget = args.limit
    deadline = None
    if args.stop_at:
        hh, mm = (int(x) for x in args.stop_at.split(":"))
        deadline = _dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= _dt.datetime.now():
            deadline += _dt.timedelta(days=1)
    judge = None
    done = ones = 0
    t0 = time.time()
    try:
        for phase in phases:
            todo = todo_items(cfg, behaviours, manifest, cache, models, store, tag, phase)
            note = ""
            if args.dry_run and phase != "canonical":
                note = "  (includes the canonical items above"
                note += "; P^- is provisional until canonical is judged)" if phase == "rsr" else ")"
            print(f"[{phase}] remaining: {len(todo)} unique (prompt, response) pairs{note}")
            if args.dry_run or not todo:
                continue
            if budget is not None:
                todo = todo[:max(budget, 0)]
                budget -= len(todo)
                if not todo:
                    break
            if judge is None:
                judge = build_judge(cfg)
            for i, (k, prompt, resp) in enumerate(todo, start=1):
                if deadline and _dt.datetime.now() >= deadline:
                    raise TimeoutError
                v = int(judge.classify(prompt, Response(text=resp, latent_harmful=None)))
                store.put(k, v)
                done += 1
                ones += v
                if i % 25 == 0 or i == len(todo):
                    el = time.time() - t0
                    rate = el / done
                    print(f"  [{phase}] {i}/{len(todo)}  complied={ones}  {rate:.1f}s/item  "
                          f"eta {rate * (len(todo) - i) / 60:.0f} min", flush=True)
    except KeyboardInterrupt:
        print(f"\ninterrupted after {done} -- re-run the same command to resume.")
    except TimeoutError:
        print(f"\nreached --stop-at {args.stop_at} after {done} -- re-run the same command to resume.")
    finally:
        store.close()
    if not args.dry_run:
        print(f"done: {done} judged this run, {len(store)} stored -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
