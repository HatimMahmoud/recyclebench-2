#!/usr/bin/env python3
"""rpe_gate.py -- run the production intent gate over inversion-arm candidates, offline.

Why this exists
---------------
`rpe_lineage.py` writes `admitted: false` on every candidate: it never ran the gate, to save
DeBERTa time. `rpe_closedloop.py` reads that flag as "the gate rejected it", so in every
lineage and per-benchmark closed-loop artifact A_gate and A_both are empty and every A_loop
member is listed as `loop_not_gate`, i.e. as if the gate had rejected all of them. It never
looked. The convergent check therefore had no gate-comparable number (A_gate, the RSR analogue)
and no primary number (A_both), only the upper bound (A_loop).

This scores every candidate with the same `RealGate` the recycle arm used (lexical ->
cosine -> BERTScore -> bidirectional NLI; L2 off, as in the frozen recycle manifests) against
the ORIGINAL canonical request, and stores the decision in an append-only sidecar. The
manifests stay frozen. `rpe_closedloop.py --gate-store` then reads the flag from here.

Only the layers the gate actually evaluated are stored: a short-circuited layer is absent,
never the 1.0 placeholder `GateDecision.to_gate_scores()` writes (THESIS_KB section 9).

Offline: no Ollama calls. The three encoders load from the local HF cache.

  python3 tools/rpe_gate.py --dry-run          # how many candidates are still unscored
  python3 tools/rpe_gate.py                    # score them (resumable)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recyclebench.config import Config                     # noqa: E402
from recyclebench.types import Behaviour, Variant          # noqa: E402

DEFAULT_GLOBS = ("results/runs/rpe_lineage*/rpe_manifest_*.jsonl",
                 "results/runs/rpe_bench*/*/rpe_manifest_*.jsonl")
STORE = "cache/rpe_gate.jsonl"
EMBED, NLI = "all-MiniLM-L6-v2", "microsoft/deberta-large-mnli"


def signature(cfg) -> str:
    """Everything that decides admission. A changed threshold gets fresh keys."""
    from recyclebench.backends.real_gate import _DEFAULT_MIN_LEXICAL_DIVERGENCE
    g = cfg.gate
    lex = getattr(g, "min_lexical_divergence", _DEFAULT_MIN_LEXICAL_DIVERGENCE)
    return (f"lex>={lex}|cos>={g.cosine}:{EMBED}|bert>={g.bertscore}|nli>={g.nli_entail}:{NLI}"
            f"|intent:off")


def key(sig: str, prompt: str, text: str) -> str:
    return hashlib.sha256(f"{sig}|{prompt}|{text}".encode("utf-8")).hexdigest()[:32]


def load_store(path) -> dict:
    """-> {key: record}. Tolerates a truncated last line (a hard kill mid-write)."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["k"]] = r
    return out


def lookup(store: dict, sig: str, prompt: str, text: str):
    """-> the stored decision for (prompt, text) under this gate, or None if never scored."""
    return store.get(key(sig, prompt, text))


def manifest_candidates(paths):
    """-> [(prompt, text)] for every candidate in the manifests, de-duplicated, in file order."""
    seen, out = set(), []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            prompt = r.get("prompt") or ""
            for d in r.get("details") or []:
                t = d.get("text")
                if prompt and t and (prompt, t) not in seen:
                    seen.add((prompt, t))
                    out.append((prompt, t))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/real_llama_lineage_all.yaml",
                    help="supplies the gate thresholds (identical across lineage configs)")
    ap.add_argument("--manifests", nargs="*", default=None,
                    help=f"probe manifests; default: {', '.join(DEFAULT_GLOBS)}")
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.manifests] if args.manifests else sorted(
        p for g in DEFAULT_GLOBS for p in ROOT.glob(g))
    cfg = Config.load(args.config)
    sig = signature(cfg)
    store = load_store(ROOT / args.store)
    cands = manifest_candidates(paths)
    todo = [(p, t) for p, t in cands if key(sig, p, t) not in store]
    print(f"gate     : {sig}")
    print(f"manifests: {len(paths)}   candidates {len(cands)}   scored {len(cands) - len(todo)}"
          f"   to score {len(todo)}")
    if args.dry_run or not todo:
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never block on a HEAD request
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from recyclebench.backends.real_gate import RealGate
    gate = RealGate(cfg.gate, cfg, embed_model=EMBED, nli_model=NLI)
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    t0, adm = time.time(), 0
    dest = ROOT / args.store
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as fh:
        for i, (prompt, text) in enumerate(todo, 1):
            if deadline and time.time() > deadline:
                print(f"deadline reached, stopping cleanly at {i - 1}/{len(todo)}", flush=True)
                break
            d = gate.decide(Behaviour(bid="rpe", canonical_prompt=prompt),
                            Variant(bid="rpe", text=text, transform_class="rpe", surface_distance=0.0))
            adm += int(d.admitted)
            fh.write(json.dumps({"k": key(sig, prompt, text), "sig": sig,
                                 "prompt": prompt[:200], "text": text[:200],
                                 "admitted": bool(d.admitted), "rejected_by": d.rejected_by or None,
                                 "layers": {lr.name: round(float(lr.score), 6) for lr in d.layers}},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 50 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}   admitted so far {adm}   {rate:.2f} s/candidate", flush=True)
    print(f"wrote    : {args.store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
