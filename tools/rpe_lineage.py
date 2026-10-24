#!/usr/bin/env python3
"""rpe_lineage.py -- build inversion-probe manifests for every adjacent lineage pair.

The recycle arm reports RSR for all five adjacent SORRY-Bench rungs (P^- = 18, 3, 49,
12, 13). The inversion arm exists for exactly one pair, on 61 behaviours selected by a
superseded offline rule. That asymmetry is why the convergent-validity claim rests on
ten behaviours.

This rebuilds the inversion arm's inputs from first principles, for any pair:

  1. recompute P^- for the pair from the frozen manifest + response cache + judge
     verdict store (lineage_core.observe -> to_model_run -> partition), so the arm is
     anchored on the SAME judged P^- the recycle arm reports, not a stale set;
  2. take the OLD model's own compliant answer for each P^- behaviour from the cache
     (no new target calls -- these are the answers the study already paid for);
  3. invert each answer with the real arm's inverter spec (real.qwen25-14b) into n
     candidate requests;
  4. write a manifest in the probe manifest's schema.

Each manifest then goes through `rpe_closedloop.py` unchanged, so every pair is scored
by the identical pipeline and the per-pair PRR is directly comparable to the per-pair
RSR.

`admitted` is false on every row: the intent gate is not run here. These manifests are
for the closed-loop (preimage) criterion, which needs no gate; running DeBERTa over
every candidate would cost hours for a number this comparison does not use.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import lineage_core as lc                                   # noqa: E402
from recyclebench.config import Config                      # noqa: E402
from recyclebench.datasets import load_jailbreakbench       # noqa: E402
from recyclebench.independent_probe.rpe_engine import RPEEngine   # noqa: E402
from recyclebench.manifest import Manifest                  # noqa: E402
from recyclebench.realrun import _key                       # noqa: E402
from recyclebench.screening import _draw_seed               # noqa: E402


def pminus_with_answers(cfg, behaviours, manifest, cache, verdicts, old, new):
    """-> [(bid, canonical_prompt, old_answer)] for the judged P^- of this pair.

    The answer taken is the first canonical draw the judge scored as compliant: the
    inversion arm must invert an answer the OLD model actually gave *and complied
    with*, not an arbitrary draw.
    """
    obs = lc.observe(cfg, behaviours, manifest, cache, verdicts, [old, new])
    runs = {m: lc.to_model_run(obs[m], behaviours) for m in (old, new)}
    part = lc.partition(runs[old], runs[new])
    pm = set(part["pminus"])
    by_bid = {b.bid: b for b in behaviours}

    out, no_answer = [], 0
    for bid in sorted(pm):
        b = by_bid.get(bid)
        if b is None:
            continue
        answer = None
        for d in range(cfg.budget.k_canonical_samples):
            t = cache.get(_key(old, b.canonical_prompt, _draw_seed(bid, -1, d)))
            if t is None:
                continue
            if verdicts.get(b.canonical_prompt, t) == 1:    # complied on this draw
                answer = t
                break
        if answer is None:
            no_answer += 1
            continue
        out.append((bid, b.canonical_prompt, answer))
    return out, len(pm), no_answer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/real_llama_lineage_all.yaml")
    ap.add_argument("--inverter-config", default="configs/recycle_3_31.yaml",
                    help="config holding the real arm's real.<inverter> spec")
    ap.add_argument("--inverter", default="qwen25-14b")
    ap.add_argument("--behaviours", default="data/sorrybench.csv")
    ap.add_argument("--manifest", default="results/runs/lineage_sorrybench_20261020/manifest.jsonl")
    ap.add_argument("--cache", default="cache/llama_gen.jsonl")
    ap.add_argument("--verdicts", default="cache/judge_lineage_sorrybench.jsonl")
    ap.add_argument("--pairs", default=None,
                    help="comma list old:new; default = adjacent rungs of cfg.lineage")
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--out-root", default="results/runs/rpe_lineage")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    deadline = time.time() + args.minutes * 60 if args.minutes else None
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
    print(f"pairs    : {', '.join(f'{o}->{n}' for o, n in pairs)}\n")

    inv_cfg = Config.load(args.inverter_config)
    spec = (inv_cfg.real or {}).get(args.inverter)
    if not spec:
        raise SystemExit(f"{args.inverter_config} has no real.{args.inverter}")
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    pc = ProviderConfig(**spec)
    if pc.max_tokens < 64:
        raise SystemExit(f"real.{args.inverter} max_tokens={pc.max_tokens}: that is a judge spec")
    inverter = ProviderModel("inverter", pc)
    engine = RPEEngine(provider_respond=lambda p: (inverter.respond(p).text or ""))
    print(f"inverter : {pc.model_id}  temp {pc.temperature}  max_tokens {pc.max_tokens}\n")

    out_root = ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for old, new in pairs:
        items, n_pm, no_ans = pminus_with_answers(cfg, behaviours, manifest, cache,
                                                  verdicts, old, new)
        dest = out_root / f"rpe_manifest_{old}__to__{new}.jsonl"
        done = set()
        if dest.exists():
            for l in dest.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    done.add(json.loads(l)["bid"])
        todo = [x for x in items if x[0] not in done]
        print(f"{old} -> {new}:  P^- {n_pm}   with a compliant cached answer {len(items)}"
              + (f"   (no compliant draw: {no_ans})" if no_ans else "")
              + f"   already built {len(done)}   to build {len(todo)}")
        summary.append({"old": old, "new": new, "p_minus": n_pm,
                        "invertible": len(items), "no_compliant_answer": no_ans,
                        "manifest": str(dest.relative_to(ROOT))})
        if args.dry_run:
            continue
        with dest.open("a", encoding="utf-8") as fh:
            for i, (bid, prompt, answer) in enumerate(todo, 1):
                if deadline and time.time() > deadline:
                    print(f"  deadline reached, stopping cleanly at {i - 1}/{len(todo)}",
                          flush=True)
                    break
                try:
                    cands = engine.invert(answer, n_candidates=args.n_candidates)
                except Exception as e:
                    print(f"  {bid} FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                fh.write(json.dumps({
                    "bid": bid, "benchmark": "sorrybench", "pair": [old, new],
                    "rpe_recovered": None, "para_recovered": None,
                    "n_candidates": len(cands), "n_admitted": 0, "admitted": [],
                    "prompt": prompt, "old_response": answer,
                    "details": [{"text": c, "admitted": False, "scores": None,
                                 "response": None, "broke": None} for c in cands],
                }, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"    {i}/{len(todo)}", flush=True)

    (out_root / "pairs.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote    : {(out_root / 'pairs.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
