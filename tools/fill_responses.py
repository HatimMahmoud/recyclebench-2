#!/usr/bin/env python3
"""tools/fill_responses.py -- generate exactly the missing target-model responses for one job, resumably.

`run-real` re-screens every lineage model on every behaviour and overwrites results/summary_real.json; this
fills only what a given analysis is missing, through the same code path, so the cache ends up exactly as if
run-real had produced it:

  * the model is a ProviderModel built from the config's `real:` entry (same temperature, max_tokens, seed);
  * calls go through realrun.CachedModel, so the key is sha256(cfg-key|prompt|seed) as always;
  * canonical draw d of behaviour b uses seed _draw_seed(b.bid, -1, d); sample si of variant v uses
    _draw_seed(b.bid, v.index, si) -- the same seeds screening.run_model uses;
  * behaviour ids come from the same loader on the same CSV, so they match the frozen manifest.

A job is one model on one benchmark: --canonical (k draws for every behaviour) and/or --manifest M (every
admitted rewording), optionally restricted with --only-pminus OLD:NEW to the fixes of that pair (P^- from the
cached canonical screens, offline rule). Already-cached items are skipped, so a re-run resumes.

A lock file next to the cache prevents two writers (run-real, another fill) from appending at once.

  python3 tools/fill_responses.py --model llama31-8b --behaviours data/harmbench.csv --canonical --dry-run
  python3 tools/fill_responses.py --model llama2-13b --behaviours data/harmbench.csv \\
      --manifest <run>/manifest_harmbench.jsonl --only-pminus llama2-7b:llama2-13b --stop-at 07:45
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_core as lc                                          # noqa: E402
from recyclebench.config import Config                             # noqa: E402
from recyclebench.datasets import load_jailbreakbench              # noqa: E402
from recyclebench.judge import looks_like_refusal                  # noqa: E402
from recyclebench.manifest import Manifest                         # noqa: E402
from recyclebench.realrun import CachedModel, ResponseCache, _key  # noqa: E402
from recyclebench.screening import _draw_seed                      # noqa: E402


def _deadline(stop_at: str | None):
    if not stop_at:
        return None
    now = _dt.datetime.now()
    hh, mm = (int(x) for x in stop_at.split(":"))
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= now:
        t += _dt.timedelta(days=1)
    return t


class Lock:
    def __init__(self, cache_path: Path):
        self.path = cache_path.with_name(cache_path.name + ".lock")

    def __enter__(self):
        if self.path.exists():
            try:
                pid = int(self.path.read_text().split()[0])
                os.kill(pid, 0)
                raise SystemExit(f"{self.path} held by running pid {pid}: another writer is active")
            except (ProcessLookupError, ValueError):
                print(f"  [lock] removing stale lock {self.path}")
                self.path.unlink()
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} fill_responses\n".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def plan(cfg, behaviours, manifest, cache: dict, model, canonical, only_bids):
    """-> list of (prompt, seed, behaviour, variant-or-None), in order, that are not cached yet."""
    k, spv = cfg.budget.k_canonical_samples, cfg.budget.samples_per_variant
    todo = []
    for b in behaviours:
        if canonical:
            for d in range(k):
                s = _draw_seed(b.bid, -1, d)
                if _key(model, b.canonical_prompt, s) not in cache:
                    todo.append((b.canonical_prompt, s, b, None))
        if manifest is not None and (only_bids is None or b.bid in only_bids):
            for v in manifest.R(b.bid):
                for si in range(spv):
                    s = _draw_seed(b.bid, v.index, si)
                    if _key(model, v.text, s) not in cache:
                        todo.append((v.text, s, b, v))
    return todo


def pminus_bids(cfg, behaviours, cache, old, new):
    """P^- of OLD -> NEW on these behaviours from the cached canonical screens (offline rule, any of k)."""
    k = cfg.budget.k_canonical_samples
    out, missing = set(), 0

    def screen(m, b):
        ts = [cache.get(_key(m, b.canonical_prompt, _draw_seed(b.bid, -1, d))) for d in range(k)]
        if any(t is None for t in ts):
            return None
        return int(any(not looks_like_refusal(t) for t in ts))
    for b in behaviours:
        so = screen(old, b)
        if so is None:
            missing += 1
            continue
        if so == 1:
            sn = screen(new, b)
            if sn is None:
                missing += 1
            elif sn == 0:
                out.add(b.bid)
    return out, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/real_llama_lineage_all.yaml")
    ap.add_argument("--cache", default="cache/llama_gen.jsonl")
    ap.add_argument("--model", required=True, help="cfg.real key (cache keys use it)")
    ap.add_argument("--behaviours", required=True)
    ap.add_argument("--canonical", action="store_true", help="the k canonical draws of every behaviour")
    ap.add_argument("--manifest", default=None, help="frozen manifest: every admitted rewording")
    ap.add_argument("--only-pminus", default=None, metavar="OLD:NEW", help="restrict rewordings to that pair's fixes")
    ap.add_argument("--stop-at", default=None, metavar="HH:MM", help="stop gracefully at this local time")
    ap.add_argument("--label", default="", help="name printed in the log")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.canonical and not args.manifest:
        raise SystemExit("nothing to do: pass --canonical and/or --manifest")

    cfg = Config.load(args.config)
    if args.model not in (cfg.real or {}):
        raise SystemExit(f"{args.model} has no real: entry in {args.config}")
    behaviours = load_jailbreakbench(args.behaviours)
    manifest = Manifest.load(args.manifest) if args.manifest else None
    if manifest is not None:
        ids = {b.bid for b in behaviours}
        if not set(manifest.admitted) <= ids:
            raise SystemExit("manifest behaviour ids are not this CSV's: wrong --behaviours for this manifest")
    cache_path = Path(args.cache)
    cache = lc.load_response_cache(cache_path)
    only = None
    tag = args.label or f"{args.model} · {Path(args.behaviours).stem}"
    if args.only_pminus:
        old, new = args.only_pminus.split(":")
        only, miss = pminus_bids(cfg, behaviours, cache, old, new)
        print(f"[{tag}] P⁻({old} → {new}) = {len(only)} behaviours (offline rule)"
              + (f"; {miss} behaviours lack a complete canonical screen and are excluded" if miss else ""))
    todo = plan(cfg, behaviours, manifest, cache, args.model, args.canonical, only)
    n_can = sum(1 for t in todo if t[3] is None)
    print(f"[{tag}] missing: {len(todo):,} calls ({n_can:,} canonical, {len(todo) - n_can:,} rewordings) "
          f"· model {cfg.real[args.model]['model_id']} · cache {cache_path}")
    if args.dry_run or not todo:
        return 0

    deadline = _deadline(args.stop_at)
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__("flag", True))
    with Lock(cache_path):
        rc = ResponseCache(cache_path)
        model = CachedModel(ProviderModel(args.model, ProviderConfig(**cfg.real[args.model])), rc)
        t0, done = time.time(), 0
        try:
            for prompt, seed, b, v in todo:
                if stop["flag"] or (deadline and _dt.datetime.now() >= deadline):
                    print(f"[{tag}] stopping ({'signal' if stop['flag'] else 'deadline'}) after {done} calls; re-run to resume")
                    break
                model.respond(prompt, behaviour=b, variant=v, canonical=(v is None), seed=seed)
                done += 1
                if done % 25 == 0 or done == len(todo):
                    rate = (time.time() - t0) / done
                    print(f"[{tag}] {done}/{len(todo)}  {rate:.2f} s/call  eta {rate * (len(todo) - done) / 60:.0f} min "
                          f"({_dt.datetime.now():%H:%M})", flush=True)
        except KeyboardInterrupt:
            print(f"[{tag}] interrupted after {done} calls; re-run to resume")
        except RuntimeError as e:                  # provider failed after its own retries
            print(f"[{tag}] PROVIDER FAILURE after {done} calls: {e}")
            rc.close()
            return 2
        finally:
            rc.close()
    print(f"[{tag}] done: {done} new responses ({model.calls} calls, {model.hits} cache hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
