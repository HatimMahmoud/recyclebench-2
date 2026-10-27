"""Token-level sequence reversal for previous-token prediction (PTP).

Suhail et al. (2026), Remark 1: reversal is applied **after** tokenisation, never to
the raw text. Reversing characters or words first produces a different token sequence
under the model's tokenizer even when it came from the same text, so the inverse model
would be trained on a token inventory the forward model never emits.

This module is deliberately tiny and heavily tested, because getting it wrong is
silent: training still converges, the loss curve looks healthy, and the inverse model
learns a mapping that has nothing to do with the forward model's dynamics.

    forward   f      : Context_<t -> V      P(y_t | y_<t)      next-token prediction
    inverse   f_inv  : Context_>=t -> V     P(y_{t-1} | y_>=t) previous-token prediction

PTP implements f_inv by training an ordinary causal LM on token-reversed sequences, so
"predict the next token of the reversed sequence" *is* "predict the previous token of
the original".
"""
from __future__ import annotations

from typing import Sequence


def reverse_ids(ids: Sequence[int]) -> list[int]:
    """Reverse a token-id sequence. The whole of PTP's data transform."""
    return list(ids)[::-1]


def reverse_text(text: str, tokenizer) -> list[int]:
    """Tokenise in the forward direction, then reverse the ids (Remark 1).

    Never reverse `text` itself: `tokenizer(text[::-1])` is a different and
    meaningless token sequence.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return reverse_ids(ids)


def detokenize_reversed(ids: Sequence[int], tokenizer) -> str:
    """Un-reverse reversed ids and decode back to text."""
    return tokenizer.decode(reverse_ids(ids), skip_special_tokens=True)


def build_training_sequence(prompt: str, response: str, tokenizer,
                            eop: int | None = None) -> list[int]:
    """The fine-tuning sequence of PTP section 3.4.

    s     = (x_1..x_n, y_1..y_T)             prompt then response, one sequence
    s_rev = Reverse(s) = (y_T..y_1, x_n..x_1)

    Training a causal LM on s_rev therefore walks backwards from the end of the
    response into the prompt: by the time it reaches the prompt tokens it is
    conditioned on the entire response, which is exactly the inference-time
    conditioning.

    `eop` appends an end-of-prompt token after x_1. Without it the inverse problem has
    no defined output length: s_rev simply stops, nothing marks the prompt's start, and
    decoding runs to a fixed budget and continues past the prompt into invented text.
    Measured on the 27M local run, that overrun was most of the reported error --
    prompts average 10.8 tokens against a 40-token budget, so precision was capped near
    0.27 however good the model was. Adding the token raised best-of-32 token F1 from
    0.217 to 0.335 and precision from 0.140 to 0.346 with nothing else changed.
    """
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] + \
        tokenizer(response, add_special_tokens=False)["input_ids"]
    out = reverse_ids(ids)
    return out + [eop] if eop is not None else out


def split_reversed(ids_rev: Sequence[int], n_prompt_tokens: int) -> tuple[list[int], list[int]]:
    """Split a reversed (prompt+response) sequence into (reversed response, reversed prompt).

    In s_rev the response comes first and the prompt last, so the prompt is the final
    `n_prompt_tokens` entries.
    """
    ids_rev = list(ids_rev)
    if n_prompt_tokens <= 0:
        return ids_rev, []
    return ids_rev[:-n_prompt_tokens], ids_rev[-n_prompt_tokens:]
