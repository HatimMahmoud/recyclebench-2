"""
run_probe.py -- execute the independent (RPE) probe over a set of ProbeItems.

For each P- behaviour:
    1. invert the OLD model's response -> candidate prompts   (ProbeEngine)
    2. gate candidates for intent preservation               (Gate)
    3. re-query the NEW model with each admitted candidate    (target connector)
    4. judge the new model's response; recovered = 1 if ANY admitted candidate
       makes the new model comply, else 0; None if nothing was admitted (unresolved)

This mirrors the paraphrase arm exactly, differing ONLY in step 1's generator --
which is the whole point (same gate, same re-query, same judge => comparable RSR).

Connectors/judge are injected so this runs offline in tests. In production, pass:
    new_model_fn(prompt) -> response_text          (Ollama new model)
    judge_fn(prompt, response) -> 1|0|-1           (your judge ensemble; 1=comply)
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Callable, Optional
import json
from pathlib import Path

from .contract import ProbeItem
from .rpe_engine import ProbeEngine
try:
    from .gate import LexicalGate  # offline stub, optional
except Exception:
    LexicalGate = None


def run_probe(
    items: list[ProbeItem],
    engine: ProbeEngine,
    new_model_fn: Callable[[str], str],
    judge_fn: Callable[[str, str], int],
    gate=None,
    n_candidates: int = 5,
    verbose: bool = True,
) -> list[ProbeItem]:
    if gate is None:
        if LexicalGate is None:
            raise ValueError('no gate supplied and offline LexicalGate stub not present')
        gate = LexicalGate()
    for k, it in enumerate(items):
        # 1. invert the OLD model's harmful response
        cands = engine.invert(it.old_response, n_candidates=n_candidates)
        it.rpe_candidates = cands

        # 2. gate for intent preservation against the canonical P- prompt
        admitted = []
        for c in cands:
            ok, score = gate.check(it.prompt, c)
            if ok:
                admitted.append(c)
        it.rpe_admitted = admitted

        # 3+4. re-query new model + judge; OR-reduce over admitted candidates
        if not admitted:
            it.rpe_recovered = None                     # unresolved (fail-safe)
        else:
            recovered = 0
            for c in admitted:
                resp = new_model_fn(c)
                verdict = judge_fn(c, resp)             # 1 comply, 0 refuse, -1 undecided
                if verdict == 1:
                    recovered = 1
                    break
            it.rpe_recovered = recovered

        if verbose:
            tag = {1: "RECOVERED", 0: "held", None: "unresolved"}[it.rpe_recovered]
            print(f"[{k+1}/{len(items)}] {it.bid}: {len(cands)} cand, "
                  f"{len(admitted)} admitted -> {tag}")
    return items


def write_manifest(items: list[ProbeItem], path: Path) -> None:
    """Write a JSONL manifest mirroring the recycle-run format, for the analyzer."""
    with Path(path).open("w", encoding="utf-8") as f:
        for it in items:
            rec = {
                "bid": it.bid,
                "benchmark": it.benchmark,
                "rpe_recovered": it.rpe_recovered,
                "para_recovered": it.para_recovered,
                "n_candidates": len(it.rpe_candidates),
                "n_admitted": len(it.rpe_admitted),
                "admitted": it.rpe_admitted,
                "prompt": it.prompt,
                "old_response": it.old_response,
                "details": it.rpe_details,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
