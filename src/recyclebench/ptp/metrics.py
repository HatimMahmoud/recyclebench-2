"""Reconstruction metrics, as defined in Suhail et al. Appendix B.

Split deliberately into two families, because PTP's central empirical claim is that
they diverge:

  lexical   exact match, BLEU, ROUGE-L, token F1   -- did we recover the wording?
  semantic  cosine, BERTScore F1                   -- did we recover the meaning?

For RecycleBench the semantic family is the one that matters: the estimand is whether
a reconstruction lands in the response-equivalence class, not whether it reproduces
the original string. The lexical family is reported anyway, because the gap between
them is itself the finding (prompt-level dispersed, response-level concentrated).

BLEU and ROUGE-L are implemented here rather than pulled in as dependencies: both are
short, and the project already carries enough optional ML deps.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_TOK = re.compile(r"\w+|[^\w\s]")


def tokens(s: str) -> list[str]:
    return _TOK.findall((s or "").lower())


def exact_match(a: str, b: str) -> float:
    return float(tokens(a) == tokens(b))


def token_f1(a: str, b: str) -> float:
    ta, tb = Counter(tokens(a)), Counter(tokens(b))
    overlap = sum((ta & tb).values())
    if not overlap:
        return 0.0
    p, r = overlap / max(1, sum(ta.values())), overlap / max(1, sum(tb.values()))
    return 2 * p * r / (p + r)


def _ngrams(xs: Sequence[str], n: int) -> Counter:
    return Counter(tuple(xs[i:i + n]) for i in range(len(xs) - n + 1))


def bleu(hyp: str, ref: str, max_n: int = 4) -> float:
    """Sentence BLEU with add-one smoothing on n > 1 (Chen & Cherry smoothing 1)."""
    h, r = tokens(hyp), tokens(ref)
    if not h or not r:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        hn, rn = _ngrams(h, n), _ngrams(r, n)
        match = sum((hn & rn).values())
        total = max(1, sum(hn.values()))
        if n > 1:                                   # smoothing: never a hard zero
            match, total = match + 1, total + 1
        if match == 0:
            return 0.0
        logs.append(math.log(match / total))
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / max(1, len(h)))
    return bp * math.exp(sum(logs) / max_n)


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hyp: str, ref: str, beta: float = 1.2) -> float:
    h, r = tokens(hyp), tokens(ref)
    if not h or not r:
        return 0.0
    l = _lcs(h, r)
    if not l:
        return 0.0
    p, rc = l / len(h), l / len(r)
    return ((1 + beta ** 2) * p * rc) / (rc + beta ** 2 * p)


def lexical_suite(hyp: str, ref: str) -> dict:
    return {"exact_match": exact_match(hyp, ref), "bleu": bleu(hyp, ref),
            "rouge_l": rouge_l(hyp, ref), "token_f1": token_f1(hyp, ref)}


def semantic_suite(hyps: Sequence[str], refs: Sequence[str],
                   embed_model: str = "all-MiniLM-L6-v2") -> dict:
    """Cosine over sentence embeddings. Loads from the local HF cache first: a network
    blip must not kill an otherwise offline evaluation."""
    from sentence_transformers import SentenceTransformer, util
    try:
        emb = SentenceTransformer(embed_model, local_files_only=True)
    except Exception:
        emb = SentenceTransformer(embed_model)
    a = emb.encode(list(hyps), convert_to_tensor=True, show_progress_bar=False)
    b = emb.encode(list(refs), convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(a, b).diagonal().tolist()
    return {"cosine_mean": sum(sims) / max(1, len(sims)), "cosine": sims}
