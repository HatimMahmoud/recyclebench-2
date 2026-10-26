"""Restrict the inverse model's vocabulary to tokens the corpus actually contains.

Why this is needed when probing a SUBSET of the vocabulary
-----------------------------------------------------------
PTP probes every vocabulary token, so every embedding row gets a gradient and the
output softmax is trained over the whole vocabulary. Probing a subset breaks that
silently. Measured on the 15k-seed corpus:

    distinct tokens seen        90,869 of 151,665   (59.9%)
    never receive a gradient    60,796              (40.1%)
    embedding parameters        78M of 105M total   (74% of the model)

So three quarters of the model is an embedding table, 40% of whose rows are frozen at
their random init -- and the softmax ranks 60k classes it has never been trained to
rank. Untrained logits compete with trained ones at generation time, which is exactly
how the first runs produced degenerate output ("1..\\n\\n1. \\n\\n1.."). The model was
being asked to choose among tokens it had no information about.

Restricting the vocabulary to observed tokens is the right adaptation, not a shortcut:
it makes a subset-probed run resemble the paper's full-vocabulary condition, where
every row is trained. It is recorded as a deviation either way.

Ids are remapped to a compact range; everything else is unchanged. `encode`/`decode`
move between the tokenizer's ids and the compact ones, so the forward model, the
corpus and the evaluation all keep speaking the original vocabulary.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

UNK = 0          # compact id 0 is reserved for anything unseen
PAD = 1          # compact id 1 pads batches


class CompactVocab:
    """Bidirectional map between tokenizer ids and a compact contiguous range."""

    def __init__(self, ids: Sequence[int]):
        self.ids = list(dict.fromkeys(int(i) for i in ids))      # dedupe, keep order
        self.to_compact = {t: i + 2 for i, t in enumerate(self.ids)}   # 0=UNK, 1=PAD
        self.from_compact = {v: k for k, v in self.to_compact.items()}

    def __len__(self) -> int:
        return len(self.ids) + 2

    def encode(self, ids: Iterable[int]) -> list[int]:
        g = self.to_compact.get
        return [g(int(t), UNK) for t in ids]

    def decode(self, ids: Iterable[int]) -> list[int]:
        """Back to tokenizer ids; UNK and PAD vanish rather than becoming real tokens."""
        g = self.from_compact.get
        return [t for t in (g(int(i)) for i in ids) if t is not None]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"ids": self.ids}))

    @staticmethod
    def load(path: Path) -> "CompactVocab":
        return CompactVocab(json.loads(Path(path).read_text())["ids"])

    @staticmethod
    def build(sequences: Iterable[Sequence[int]], min_count: int = 1,
              extra: Optional[Iterable[Sequence[int]]] = None) -> "CompactVocab":
        """Tokens appearing at least `min_count` times, plus everything in `extra`.

        `extra` (the fine-tuning pairs) is included unconditionally: a token the model
        must emit at inference but has no embedding for is unrecoverable, whereas a
        rare token costs one embedding row.
        """
        c: Counter = Counter()
        for s in sequences:
            c.update(int(t) for t in s)
        keep = {t for t, n in c.items() if n >= min_count}
        for s in (extra or []):
            keep.update(int(t) for t in s)
        return CompactVocab(sorted(keep))


def coverage(vocab: CompactVocab, sequences: Iterable[Sequence[int]]) -> float:
    """Share of corpus token occurrences the compact vocabulary covers."""
    tot = hit = 0
    known = vocab.to_compact
    for s in sequences:
        for t in s:
            tot += 1
            hit += int(t) in known
    return hit / max(1, tot)
