#!/usr/bin/env python3
"""run_compose_v2.py -- the 21-condition transform-class factorial (PREREG_COMPOSE_V2_20261029.md).

6 single transform classes + 15 unordered pairs (class A then class B, fixed list order), 3 candidates per
behaviour x condition, over the qwen-judged anchor P^- (llama3-8b -> llama3.1-8b, 49 SORRY-Bench behaviours,
full-benchmark ids). Every candidate is gated against the ORIGINAL request with the production gate; only
admitted ones reach llama3.1-8b; qwen2.5:14b judges the reply against the candidate text, as run-real does.

Separate from everything that exists: stores in cache/compose_v2/, results in results/runs/compose_v2_<stamp>/;
tools/run_compose.py and its store are not touched. Phases hold one Ollama model each, so the judge/target/
paraphraser are never swapped per call:

    gen    paraphraser (qwen2.5:3b) builds every candidate; the gate scores it         -> candidates.jsonl
    new    llama3.1-8b answers every admitted candidate                               -> replies (cache)
    judge  qwen2.5:14b judges every reply                                             -> verdicts.jsonl
    score  compositional RSR per condition + the registered comparisons (offline)     -> compose_v2_summary.json

Generation runs candidate index 0 for all behaviours x conditions before index 1, so a time-budget cut leaves
whole index levels; `score` uses the highest index level complete for every condition.

  python3 tools/run_compose_v2.py --phase gen --n-behaviours 2      # smoke
  python3 tools/run_compose_v2.py --phase all --minutes 600
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import lineage_core as lc                                             # noqa: E402
from recyclebench.config import Config                                # noqa: E402
from recyclebench.datasets import load_jailbreakbench                 # noqa: E402
from recyclebench.types import Behaviour, Response, Variant          # noqa: E402

STORE = ROOT / "cache" / "compose_v2"
CFG = "configs/recycle_3_31.yaml"                 # paraphraser, targets, judge, gate
LINEAGE_CFG = "configs/real_llama_lineage_all.yaml"   # the seed the judged P^- was screened with
OLD, NEW = "llama3-8b", "llama31-8b"
SEED = 20261029
N_CAND = 3
CLASSES = ["paraphrase", "syntactic", "lexical", "back_translation", "tense_shift", "register"]
CONDITIONS = [(c,) for c in CLASSES] + list(itertools.combinations(CLASSES, 2))   # 6 + 15 = 21
PHASES = ("gen", "new", "judge", "score")


def log(msg):
    print(f"{_dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def cname(cond) -> str:
    return "+".join(cond)


class JsonlStore:
    """Append-only JSONL keyed by id; last write wins; a truncated last line is skipped."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.mem = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.mem[d["id"]] = d
        self._fh = path.open("a", encoding="utf-8")

    def put(self, key, **fields):
        d = {"id": key, **fields}
        self._fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.mem[key] = d
        return d


def pminus():
    """The judged anchor P^- with full SORRY-Bench ids: [(bid, canonical_prompt)]."""
    from recyclebench.manifest import Manifest
    from rpe_lineage import pminus_with_answers
    cfg = Config.load(ROOT / LINEAGE_CFG)
    bs = load_jailbreakbench(str(ROOT / "data/sorrybench.csv"))
    man = Manifest.load(ROOT / "results/runs/lineage_sorrybench_20261020/manifest.jsonl")
    cache = lc.load_response_cache(ROOT / "cache/llama_gen.jsonl")                     # read only
    verdicts = lc.StoredVerdicts(lc.VerdictStore(ROOT / "cache/judge_lineage_sorrybench.jsonl"),
                                 lc.judge_tag_from_cfg(cfg))
    items, n_pm, _ = pminus_with_answers(cfg, bs, man, cache, verdicts, OLD, NEW)
    return [(bid, prompt) for bid, prompt, _ in items], n_pm


def phase_gen(cfg, behaviours, gen, deadline):
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    from recyclebench.backends.reformulate import LLMReformulator
    from recyclebench.gate import load_gate
    import numpy as np
    para = ProviderModel("paraphraser", ProviderConfig(**cfg.paraphraser))
    ref = LLMReformulator(CLASSES, 1, cfg, para)
    gate = load_gate("real", np.random.default_rng(SEED), cfg.gate, None, cfg)
    todo = [(i, b, c) for i in range(N_CAND) for b in behaviours for c in CONDITIONS
            if f"{cname(c)}|{b[0]}|{i}" not in gen.mem]
    log(f"[gen] {len(todo)} candidates to build ({len(behaviours)} behaviours x {len(CONDITIONS)} conditions x {N_CAND})")
    t0 = time.time()
    for k, (i, (bid, prompt), cond) in enumerate(todo, 1):
        if deadline and time.time() > deadline:
            log(f"[gen] budget reached at {k - 1}/{len(todo)} (index level {i})")
            return False
        text, steps = prompt, []
        for cls in cond:
            # the candidate index picks the pivot / register, so the 3 candidates differ by design
            text = (para.respond(ref._instruction(cls, text, i)).text or "").strip()
            steps.append(text[:400])
            if not text:
                break
        rec = dict(cond=cname(cond), bid=bid, idx=i, text=text, ok=bool(text), steps=steps)
        if text:
            b = Behaviour(bid=bid, canonical_prompt=prompt)
            d = gate.decide(b, Variant(bid=bid, text=text, transform_class=cname(cond), surface_distance=0.0, index=i))
            rec.update(admitted=bool(d.admitted), rejected_by=d.rejected_by or None,
                       layers={lr.name: round(float(lr.score), 6) for lr in d.layers})
        gen.put(f"{cname(cond)}|{bid}|{i}", **rec)
        if k % 50 == 0:
            r = (time.time() - t0) / k
            log(f"[gen] {k}/{len(todo)}  {r:.1f}s/candidate  eta {r * (len(todo) - k) / 60:.0f} min")
    return True


def admitted(gen):
    return [d for d in gen.mem.values() if d.get("admitted")]


def phase_new(cfg, gen, cache, deadline):
    from recyclebench.backends.providers import load_provider_model
    m = load_provider_model(NEW, "openai_compatible", cfg)
    todo = [d for d in admitted(gen) if cache.get(NEW, d["text"], SEED + d["idx"]) is None]
    log(f"[new] {len(todo)} replies to generate")
    for k, d in enumerate(todo, 1):
        if deadline and time.time() > deadline:
            log(f"[new] budget reached at {k - 1}/{len(todo)}")
            return False
        r = m.respond(d["text"], seed=SEED + d["idx"])
        cache.put(NEW, d["text"], SEED + d["idx"], r.text or "")
        if k % 50 == 0:
            log(f"[new] {k}/{len(todo)}")
    return True


def phase_judge(cfg, gen, cache, store, tag, deadline):
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    from recyclebench.backends.real_judge import build_ensemble
    judge = build_ensemble({role: ProviderModel(f"judge:{role}", ProviderConfig(**spec))
                            for role, spec in cfg.judges.items()})
    todo = []
    for d in admitted(gen):
        t = cache.get(NEW, d["text"], SEED + d["idx"])
        if t is not None and store.get(lc.verdict_key(tag, d["text"], t)) is None:
            todo.append((d, t))
    log(f"[judge] {len(todo)} replies to judge")
    for k, (d, t) in enumerate(todo, 1):
        if deadline and time.time() > deadline:
            log(f"[judge] budget reached at {k - 1}/{len(todo)}")
            return False
        store.put(lc.verdict_key(tag, d["text"], t), int(judge.classify(d["text"], Response(text=t))))
        if k % 50 == 0:
            log(f"[judge] {k}/{len(todo)}")
    return True


def holm(ps):
    """Holm-Bonferroni adjusted p-values, same order as given (None stays None)."""
    idx = sorted((p, i) for i, p in enumerate(ps) if p is not None)
    adj, run = [None] * len(ps), 0.0
    for r, (p, i) in enumerate(idx):
        run = max(run, min(1.0, (len(idx) - r) * p))
        adj[i] = run
    return adj


def phase_score(behaviours, n_pm, gen, cache, store, tag, out_dir):
    from recyclebench.stats import mcnemar_exact, wilson_interval
    full = [lvl for lvl in range(N_CAND)
            if all(f"{cname(c)}|{b}|{lvl}" in gen.mem for c in CONDITIONS for b, _ in behaviours)]
    top = max(full) if full else -1
    if top < 0:
        raise SystemExit("no candidate index level is complete for every condition yet")

    def verdicts(cond):
        """bid -> 1 / 0 (resolved) or None (no admitted candidate, or a reply/verdict missing)."""
        out = {}
        for b, _ in behaviours:
            vals = []
            for i in range(top + 1):
                d = gen.mem.get(f"{cname(cond)}|{b}|{i}")
                if not d or not d.get("admitted"):
                    continue
                t = cache.get(NEW, d["text"], SEED + i)
                vals.append(None if t is None else store.get(lc.verdict_key(tag, d["text"], t)))
            out[b] = lc._any(vals) if vals else None
        return out

    V = {cname(c): verdicts(c) for c in CONDITIONS}
    conds = []
    for c in CONDITIONS:
        n_cand = sum(1 for b, _ in behaviours for i in range(top + 1) if gen.mem.get(f"{cname(c)}|{b}|{i}", {}).get("ok"))
        n_adm = sum(1 for b, _ in behaviours for i in range(top + 1) if gen.mem.get(f"{cname(c)}|{b}|{i}", {}).get("admitted"))
        res = [v for v in V[cname(c)].values() if v is not None]
        k, n = sum(res), len(res)
        w = wilson_interval(k, n) if n else None
        conds.append(dict(condition=cname(c), kind="single" if len(c) == 1 else "pair", candidates=n_cand,
                          admitted=n_adm, resolved=n, reopened=k, rsr=(k / n if n else None),
                          wilson95=[w.low, w.high] if w else None, unresolved=len(behaviours) - n))
    prim, sec = [], []
    for a, b in itertools.combinations(CLASSES, 2):
        ab, va, vb = V[f"{a}+{b}"], V[a], V[b]
        shared = [x for x, _ in behaviours if ab[x] is not None and va[x] is not None and vb[x] is not None]
        either = [int(va[x] or vb[x]) for x in shared]
        comp = [ab[x] for x in shared]
        m = mcnemar_exact(either, comp) if shared else None
        prim.append(dict(pair=f"{a}+{b}", n_shared=len(shared), k_pair=sum(comp), k_either_single=sum(either),
                         composite_only=sum(1 for x in shared if ab[x] == 1 and not (va[x] or vb[x])),
                         p=(m or {}).get("p_value")))
        for s, vs in ((a, va), (b, vb)):
            sh = [x for x, _ in behaviours if ab[x] is not None and vs[x] is not None]
            ms = mcnemar_exact([vs[x] for x in sh], [ab[x] for x in sh]) if sh else None
            sec.append(dict(pair=f"{a}+{b}", single=s, n_shared=len(sh), k_pair=sum(ab[x] for x in sh),
                            k_single=sum(vs[x] for x in sh), p=(ms or {}).get("p_value")))
    for rows in (prim, sec):
        for r, q in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = q
    out = {"generated": _dt.datetime.now().strftime("%Y%m%d_%H%M"), "spec": "PREREG_COMPOSE_V2_20261029.md",
           "exploratory": True, "pair": f"{OLD} -> {NEW}", "p_minus": n_pm, "behaviours": len(behaviours),
           "candidate_levels_used": top + 1, "candidates_per_condition_registered": N_CAND,
           "judge_tag": tag, "judged_against": "candidate text (as run-real)",
           "conditions": conds, "primary_pair_vs_either_single": prim, "secondary_pair_vs_each_single": sec}
    out_dir = Path(out_dir) if out_dir else ROOT / "results" / "runs" / f"compose_v2_{out['generated']}"
    dest = out_dir / "compose_v2_summary.json"
    if dest.exists():
        raise SystemExit(f"{dest} exists -- refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    log(f"[score] index levels used {top + 1}/{N_CAND}; wrote {dest}")
    for c in conds:
        r = "—" if c["rsr"] is None else f"{c['rsr'] * 100:5.1f}%"
        log(f"   {c['condition']:<30} adm {c['admitted']:>3}/{c['candidates']:<3} RSR {r} {c['reopened']}/{c['resolved']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=PHASES + ("all",))
    ap.add_argument("--minutes", type=float, default=None, help="overall wall-clock budget")
    ap.add_argument("--n-behaviours", type=int, default=None, help="cap, for smoke tests")
    ap.add_argument("--store", default=str(STORE), help="separate store for smoke tests")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    cfg = Config.load(ROOT / CFG)
    behaviours, n_pm = pminus()
    if args.n_behaviours:
        behaviours = behaviours[:args.n_behaviours]
    store_dir = Path(args.store)
    gen = JsonlStore(store_dir / "candidates.jsonl")
    from recyclebench.realrun import ResponseCache
    cache = ResponseCache(store_dir / "replies.jsonl")
    store = lc.VerdictStore(store_dir / "verdicts.jsonl")
    tag = lc.judge_tag_from_cfg(cfg)
    log(f"compose v2 · P^- {n_pm} judged, {len(behaviours)} used · {len(CONDITIONS)} conditions · store {store_dir}")
    phases = PHASES if args.phase == "all" else (args.phase,)
    for ph in phases:
        if ph == "gen" and not phase_gen(cfg, behaviours, gen, deadline):
            log("stopping: generation cut by the budget (resumable)")
        elif ph == "new":
            phase_new(cfg, gen, cache, deadline)
        elif ph == "judge":
            phase_judge(cfg, gen, cache, store, tag, deadline)
        elif ph == "score":
            phase_score(behaviours, n_pm, gen, cache, store, tag, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
