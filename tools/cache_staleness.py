#!/usr/bin/env python3
"""tools/cache_staleness.py -- can today's Ollama reproduce the cached replies the study's results rest on?

Registered in PREREG_STALENESS_20261031.md. The response cache is keyed on (config key, prompt, seed) only: the model
weights, the Ollama runtime and the decoding settings are not part of the key, so a re-pulled model or a new runtime would
be served stale replies silently. llama2:13b and llama3.2:1b/3b were re-pulled on 2026-10-10, after their July/August
caches were written; SORRY-Bench was screened after that date.

For each of the six lineage models, a seeded random sample of cached canonical draw-0 replies is regenerated with the
study's exact request (config spec, temperature 0, same seed), in two strata:
  S  SORRY-Bench (screened 2026-10-19/20)
  O  older benchmarks (HarmBench / AdvBench / StrongREJECT, screened July-August)
Both the cached and the regenerated reply are judged by the study's judge (qwen2.5:14b). Reported per model x stratum:
byte-identical rate, whitespace-normalised identical rate, judge-verdict agreement (Wilson 95%). Nothing in the main cache
is replaced. Own store cache/staleness_20261031/; numbers-only output results/runs/staleness_<stamp>/.

  python3 tools/cache_staleness.py --phase all
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import lineage_core as lc                                             # noqa: E402
from recyclebench.config import Config                                # noqa: E402
from recyclebench.datasets import load_jailbreakbench                 # noqa: E402
from recyclebench.realrun import ResponseCache, _key                  # noqa: E402
from recyclebench.screening import _draw_seed                         # noqa: E402
from recyclebench.stats import wilson_interval                        # noqa: E402
from recyclebench.types import Response                               # noqa: E402

CFG = "configs/real_llama_lineage_all.yaml"
STORE = ROOT / "cache" / "staleness_20261031"
STRATA = {"S": ["sorrybench"], "O": ["harmbench", "advbench", "strongreject"]}
SEED = 20261101


def log(msg):
    print(f"{_dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def sample(cfg, cache, per_stratum, seed=SEED):
    """[(model, stratum, bench, bid, prompt, seed, cached_text)] -- only items whose draw-0 reply is cached"""
    rng = random.Random(seed)
    out = []
    benches = {b: load_jailbreakbench(ROOT / "data" / f"{b}.csv") for s in STRATA.values() for b in s}
    for m in cfg.lineage:
        for st, names in STRATA.items():
            pool = []
            for b in names:
                for be in benches[b]:
                    sd = _draw_seed(be.bid, -1, 0)
                    t = cache.get(_key(m, be.canonical_prompt, sd))
                    if t is not None:
                        pool.append((m, st, b, be.bid, be.canonical_prompt, sd, t))
            out += rng.sample(pool, min(per_stratum, len(pool)))
    return out


def ollama_facts(cfg):
    facts = {"ollama_version": subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()}
    for m in cfg.lineage:
        mid = cfg.real[m]["model_id"]
        mf = subprocess.run(["ollama", "show", mid, "--modelfile"], capture_output=True, text=True).stdout
        blob = next((l.split("sha256-")[-1][:16] for l in mf.splitlines() if l.startswith("FROM") and "sha256-" in l), None)
        facts[m] = dict(model_id=mid, weights_blob=blob)
    return facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=("gen", "judge", "score", "all"))
    ap.add_argument("--per-stratum", type=int, default=20)
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--minutes", type=float, default=None)
    args = ap.parse_args(argv)
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    cfg = Config.load(ROOT / CFG)
    cache = lc.load_response_cache(ROOT / "cache/llama_gen.jsonl")
    items = sample(cfg, cache, args.per_stratum)
    store = Path(args.store)
    regen = ResponseCache(store / "replies.jsonl")
    verd = lc.VerdictStore(store / "verdicts.jsonl")
    tag = lc.judge_tag_from_cfg(cfg)
    log(f"staleness check · {len(items)} cached replies sampled · store {store}")
    try:
        if args.phase in ("gen", "all"):
            from recyclebench.backends.providers import load_provider_model
            by_model = {}
            for it in items:
                by_model.setdefault(it[0], []).append(it)
            for m, its in by_model.items():
                pm = load_provider_model(m, "openai_compatible", cfg)
                todo = [it for it in its if regen.get(m, it[4], it[5]) is None]
                log(f"[gen {m}] {len(todo)} to regenerate")
                for k, (m_, st, b, bid, prompt, sd, _) in enumerate(todo, 1):
                    if deadline and time.time() > deadline:
                        log("budget reached")
                        return 0
                    regen.put(m, prompt, sd, pm.respond(prompt, seed=sd).text or "")
        if args.phase in ("judge", "all"):
            from recyclebench.backends.providers import ProviderConfig, ProviderModel
            from recyclebench.backends.real_judge import build_ensemble
            todo = []
            for m, st, b, bid, prompt, sd, t in items:
                r = regen.get(m, prompt, sd)
                for text in (t, r):
                    if text is not None and verd.get(lc.verdict_key(tag, prompt, text)) is None:
                        todo.append((prompt, text))
            todo = list(dict.fromkeys(todo))
            log(f"[judge] {len(todo)} replies to judge")
            judge = build_ensemble({role: ProviderModel(f"judge:{role}", ProviderConfig(**s)) for role, s in cfg.judges.items()}) if todo else None
            for k, (prompt, text) in enumerate(todo, 1):
                if deadline and time.time() > deadline:
                    log("budget reached")
                    return 0
                verd.put(lc.verdict_key(tag, prompt, text), int(judge.classify(prompt, Response(text=text))))
                if k % 50 == 0:
                    log(f"[judge] {k}/{len(todo)}")
        if args.phase in ("score", "all"):
            rows = {}
            for m, st, b, bid, prompt, sd, t in items:
                r = regen.get(m, prompt, sd)
                if r is None:
                    continue
                vo, vr = verd.get(lc.verdict_key(tag, prompt, t)), verd.get(lc.verdict_key(tag, prompt, r))
                d = rows.setdefault((m, st), dict(n=0, identical=0, normalised=0, judged=0, verdict_agree=0, flips=[]))
                d["n"] += 1
                d["identical"] += int(r == t)
                d["normalised"] += int(" ".join(r.split()) == " ".join(t.split()))
                if vo is not None and vr is not None:
                    d["judged"] += 1
                    d["verdict_agree"] += int(vo == vr)
                    if vo != vr:
                        d["flips"].append(f"{b}:{bid} cached {vo} -> now {vr}")

            def w(k, n):
                iv = wilson_interval(k, n)
                return dict(k=k, n=n, rate=round(iv.point, 4) if n else None, ci=[round(iv.low, 4), round(iv.high, 4)] if n else None)
            res = dict(created=_dt.datetime.now().isoformat(timespec="seconds"), prereg="PREREG_STALENESS_20261031.md",
                       judge=tag, sample_seed=SEED, per_stratum=args.per_stratum, facts=ollama_facts(cfg),
                       strata={"S": "SORRY-Bench (screened 2026-10-19/20)", "O": "HarmBench / AdvBench / StrongREJECT (July-August)"},
                       cells={f"{m}|{st}": dict(identical=w(d["identical"], d["n"]), normalised=w(d["normalised"], d["n"]),
                                                verdict_agreement=w(d["verdict_agree"], d["judged"]), verdict_flips=d["flips"])
                              for (m, st), d in sorted(rows.items())})
            lines = [f"cache staleness · {res['facts']['ollama_version']} · {res['created']}", "",
                     f"{'model':<14}{'stratum':<9}{'identical':>14}{'normalised':>14}{'verdict agree':>16}"]
            for key, c in res["cells"].items():
                m, st = key.split("|")
                f = lambda x: f"{x['k']}/{x['n']}"
                lines.append(f"{m:<14}{st:<9}{f(c['identical']):>14}{f(c['normalised']):>14}{f(c['verdict_agreement']):>16}"
                             + (f"   flips: {'; '.join(c['verdict_flips'])}" if c["verdict_flips"] else ""))
            out = Path(args.out) if args.out else ROOT / "results" / "runs" / f"staleness_{_dt.datetime.now():%Y%m%d_%H%M}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(json.dumps(res, indent=1))
            (out / "summary.txt").write_text("\n".join(lines) + "\n")
            print("\n".join(lines))
    finally:
        verd.close()
        regen.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
