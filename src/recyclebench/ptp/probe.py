"""Synthetic corpus generation by vocabulary probing (Suhail et al. section 3.1).

For each seed token v the forward model continues it for L tokens under stochastic
decoding, giving y^(v) = (v, y_2, ... y_L). Collectively these expose how the forward
model maps contexts to continuations across its vocabulary, which is what the inverse
model must learn to run backwards. Appendix E shows this stage, not the fine-tune, is
what establishes the inverse mapping.

Two deliberate deviations from the paper, both forced by a laptop and both recorded in
the corpus manifest so they end up in the write-up rather than being discovered later:

  * **Vocabulary subset.** The paper probes every token (|V| ~ 150k x 256 ~ 38M tokens,
    ~1 h on an A100). Locally that is days, so a subset is probed. Section C.4 argues
    full-vocabulary probing is what covers the transition dynamics, so a subset narrows
    exactly the thing that makes the method work -- which is why the run has a stop
    rule rather than an assumption of success.
  * **Longer sequences, fewer seeds.** Per-call overhead dominates on a local server:
    8,000 calls of 64 tokens costs far more wall-clock than 2,000 calls of 256 for the
    same token budget. Same corpus size, fewer round trips.

Generation uses Ollama's native /api/generate with raw=true, NOT the OpenAI chat
endpoint the rest of the project uses. The paper generates "without instructions,
relying solely on next-token prediction"; a chat template would wrap every seed in an
instruction frame and the model would answer it rather than continue it.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence


@dataclass
class ProbeSpec:
    model_id: str = "qwen2.5:3b"
    base_url: str = "http://localhost:11434"
    n_seeds: int = 2000
    seq_len: int = 256
    temperature: float = 0.9
    top_k: int = 5
    top_p: float = 0.95
    seed: int = 20261025
    meta: dict = field(default_factory=dict)


def seed_tokens(tokenizer, n: int, rng: random.Random,
                corpus: Optional[Sequence[str]] = None) -> list[int]:
    """Choose which vocabulary tokens to probe.

    With `corpus`, rank by frequency in it and take the top n -- the subset most
    relevant to the domain being inverted. Without, sample uniformly from the
    vocabulary, which is closer to the paper's full sweep.
    """
    if corpus:
        from collections import Counter
        c: Counter = Counter()
        for text in corpus:
            c.update(tokenizer(text, add_special_tokens=False)["input_ids"])
        ranked = [t for t, _ in c.most_common()]
        if len(ranked) >= n:
            return ranked[:n]
        pool = [t for t in range(len(tokenizer)) if t not in set(ranked)]
        rng.shuffle(pool)
        return ranked + pool[:n - len(ranked)]
    pool = list(range(len(tokenizer)))
    rng.shuffle(pool)
    return pool[:n]


def generate_raw(spec: ProbeSpec, prompt: str, n_predict: int, timeout: float = 120.0) -> str:
    """One raw completion from Ollama -- no chat template (`raw: true`)."""
    import requests
    r = requests.post(
        f"{spec.base_url}/api/generate",
        json={"model": spec.model_id, "prompt": prompt, "raw": True, "stream": False,
              "options": {"temperature": spec.temperature, "top_k": spec.top_k,
                          "top_p": spec.top_p, "num_predict": n_predict}},
        timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


class CorpusStore:
    """Append-only JSONL of probed sequences, so probing is resumable."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[int] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self.done.add(json.loads(line)["seed_token"])
                    except Exception:
                        continue

    def append(self, seed_token: int, ids: Sequence[int]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"seed_token": seed_token, "ids": list(ids)}) + "\n")
        self.done.add(seed_token)

    def sequences(self) -> list[list[int]]:
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line)["ids"])
        return out

    def __len__(self) -> int:
        return len(self.done)


def probe(spec: ProbeSpec, tokenizer, store: CorpusStore, tokens: Sequence[int],
          deadline: Optional[float] = None,
          on_log: Optional[Callable[[str], None]] = None) -> dict:
    """Probe each seed token once; skip ones already in the store."""
    log = on_log or print
    todo = [t for t in tokens if t not in store.done]
    log(f"[probe] {len(todo)} seeds to probe of {len(tokens)} "
        f"({len(store.done)} already stored) · {spec.model_id} · {spec.seq_len} tokens each")
    t0, done, failed = time.time(), 0, 0
    for i, tok_id in enumerate(todo, 1):
        if deadline and time.time() > deadline:
            log(f"[probe] deadline reached, stopping cleanly at {i - 1}/{len(todo)}")
            break
        seed_text = tokenizer.decode([tok_id])
        if not seed_text.strip():
            continue                                  # whitespace-only seed: nothing to continue
        try:
            text = generate_raw(spec, seed_text, spec.seq_len)
        except Exception as e:
            failed += 1
            if failed <= 5:
                log(f"[probe] seed {tok_id} failed: {type(e).__name__}: {e}")
            continue
        ids = [tok_id] + tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = ids[:spec.seq_len]
        if len(ids) < 8:
            continue                                  # empty continuation: not informative
        store.append(tok_id, ids)
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            rate = el / max(1, done)
            log(f"[probe] {done}/{len(todo)}  {rate:.2f}s/seed  "
                f"eta {(len(todo) - done) * rate / 60:.0f} min")
    return {"probed": done, "failed": failed, "stored_total": len(store),
            "seconds": time.time() - t0}


def chunk_for_training(sequences: Sequence[Sequence[int]], max_len: int) -> list[list[int]]:
    """Reverse each probed sequence and trim to the model's context."""
    from .reversal import reverse_ids
    return [reverse_ids(s)[:max_len] for s in sequences if len(s) >= 8]
