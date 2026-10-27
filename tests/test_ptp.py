"""Tests for the PTP inverse-model implementation.

The one that matters is `test_reversal_is_after_tokenisation`. Suhail et al. Remark 1
requires reversal at the token level; doing it on raw text instead is silent --
training still converges and the loss curve looks healthy, but the inverse model has
learned a mapping unrelated to the forward model's dynamics.

The end-to-end test trains a tiny model to memorise a handful of pairs. That proves
the IMPLEMENTATION (reversal, batching, objective, reversed decoding), not the method:
PTP's quality comes from vocabulary-scale synthetic pre-training, which no unit test
can stand in for.
"""
from __future__ import annotations

import pytest

from recyclebench.ptp import metrics as M
from recyclebench.ptp.reversal import (build_training_sequence, detokenize_reversed,
                                       reverse_ids, reverse_text, split_reversed)

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen2.5 tokenizer not in the local HF cache")


def test_reverse_ids_is_an_involution():
    ids = [7985, 264, 5492, 911, 279, 9396, 13]
    assert reverse_ids(reverse_ids(ids)) == ids
    assert reverse_ids(ids)[0] == ids[-1]


def test_reversal_is_after_tokenisation(tok):
    """Remark 1: reversing the TEXT gives a different token inventory."""
    text = "Write me the lyrics of a specific song."
    ids = tok(text, add_special_tokens=False)["input_ids"]

    by_token = reverse_text(text, tok)
    assert by_token == list(reversed(ids)), "must reverse ids, not characters"
    assert sorted(by_token) == sorted(ids), "token inventory must be preserved exactly"

    by_text = tok(text[::-1], add_special_tokens=False)["input_ids"]
    assert sorted(by_text) != sorted(ids), (
        "reversing raw text should change the token inventory -- if this ever fails, "
        "the Remark 1 trap is not being exercised by this tokenizer")


def test_boundary_token_marks_where_the_prompt_starts(tok):
    """Without a stop condition the inverse problem has no defined output length.

    s_rev ends at the prompt's FIRST token, and nothing marks it, so decoding runs to
    a fixed budget and continues past the prompt into invented text. On the 27M local
    run that overrun was most of the measured error: prompts average 10.8 tokens
    against a 40-token budget, capping precision near 0.27 however good the model was.
    Adding the marker lifted best-of-32 precision 0.140 -> 0.346 and token F1
    0.217 -> 0.335 with nothing else changed, so the marker must stay where it is.
    """
    prompt, response = "Name three colours.", "Red, green and blue."
    eop = 151643                                  # Qwen's <|endoftext|>
    plain = build_training_sequence(prompt, response, tok)
    marked = build_training_sequence(prompt, response, tok, eop=eop)

    assert marked == plain + [eop], "the marker goes after x_1, at the very end"
    assert marked[-1] == eop
    assert eop not in plain, "an unmarked sequence must have no stop condition at all"

    n_p = len(tok(prompt, add_special_tokens=False)["input_ids"])
    _, prompt_rev = split_reversed(marked[:-1], n_p)
    assert tok.decode(reverse_ids(prompt_rev)) == prompt, (
        "appending the marker must not disturb the prompt span")


def test_detokenize_round_trip(tok):
    text = "Explain how to reset a password."
    assert detokenize_reversed(reverse_text(text, tok), tok) == text


def test_training_sequence_puts_prompt_last(tok):
    """s_rev = (y_T..y_1, x_n..x_1): the prompt must be at the END of the reversed seq."""
    prompt, response = "Name three colours.", "Red, green and blue."
    n_p = len(tok(prompt, add_special_tokens=False)["input_ids"])
    rev = build_training_sequence(prompt, response, tok)

    resp_rev, prompt_rev = split_reversed(rev, n_p)
    assert tok.decode(reverse_ids(prompt_rev)) == prompt
    assert tok.decode(reverse_ids(resp_rev)) == response


def test_split_reversed_handles_empty_prompt():
    assert split_reversed([1, 2, 3], 0) == ([1, 2, 3], [])


def test_metrics_are_bounded_and_ordered():
    assert M.exact_match("a b", "a b") == 1.0
    assert M.exact_match("a b", "a c") == 0.0
    assert M.token_f1("a b c", "a b c") == pytest.approx(1.0)
    assert 0.0 < M.token_f1("a b c", "a b d") < 1.0
    assert M.bleu("a b c d", "a b c d") == pytest.approx(1.0)
    assert M.rouge_l("a b c", "a b c") == pytest.approx(1.0)
    for f in (M.bleu, M.rouge_l, M.token_f1):
        assert f("", "a b") == 0.0


@pytest.mark.slow
def test_end_to_end_tiny_model_learns_to_invert(tok):
    """Train a ~3M model to memorise 6 pairs, then invert them.

    This is a wiring test. It proves reversal, batching, the NLL objective and reversed
    decoding compose into something that can recover a prompt from a response. It says
    nothing about whether PTP works at scale.
    """
    from recyclebench.ptp.model import PTPConfig, build_model, pick_device
    from recyclebench.ptp.train import evaluate_inversion, make_finetune_corpus, train

    pairs = [("name a colour", "blue"), ("name a fruit", "apple"),
             ("name a country", "france"), ("name an animal", "otter"),
             ("name a metal", "copper"), ("name a planet", "venus")]
    cfg = PTPConfig(vocab_size=len(tok), hidden_size=128, intermediate_size=256,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                    max_position_embeddings=64)
    model = build_model(cfg)
    seqs = make_finetune_corpus(pairs, tok)
    device = pick_device("cpu")                       # deterministic and cheap in CI

    rep = train(model, seqs * 40, device=device, epochs=1, batch_size=6, lr=3e-3,
                pad_id=0, log_every=10_000)
    assert rep.steps > 0
    assert rep.best_loss < 6.0, f"loss never came down: {rep}"

    got = evaluate_inversion(model, pairs, tok, device=device, max_new_tokens=8)
    assert got["n"] == len(pairs)
    assert 0.0 <= got["token_f1"] <= 1.0
