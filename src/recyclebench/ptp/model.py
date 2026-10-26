"""The PTP inverse model: an ordinary causal decoder, trained from scratch.

Suhail et al. use Qwen3-0.6B initialised from scratch (He init). Nothing about the
architecture is special -- the method lives entirely in the *data* (token-reversed
sequences) and the objective (standard next-token NLL on them). So this builds a
plain Llama-style decoder at whatever size fits the machine, and the same code serves
a 30M smoke-test model and a 0.6B cluster model.

Deliberately NOT a pretrained checkpoint. The paper's Appendix D fine-tunes a
pretrained T5 for PTP and it collapses (exact match 2.82 vs 64.77): left-to-right
pretraining priors fight reversed generation. A pretrained decoder would carry the
same penalty, and would also break the method's claim to be data-free -- the inverse
would be predicting prompts from its own pretraining knowledge rather than from the
forward model's dynamics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PTPConfig:
    """Size of the inverse decoder. Defaults are the local smoke-test size."""
    vocab_size: int = 151936
    hidden_size: int = 384
    intermediate_size: int = 1024
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    num_key_value_heads: int = 6
    max_position_embeddings: int = 512
    tie_word_embeddings: bool = True

    @staticmethod
    def smoke(vocab_size: int) -> "PTPConfig":
        """~10M non-embedding params: enough to prove the loop, not the method."""
        return PTPConfig(vocab_size=vocab_size, hidden_size=256, intermediate_size=768,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=512)

    @staticmethod
    def local(vocab_size: int) -> "PTPConfig":
        """~40M non-embedding params: the largest that trains overnight on an M1 Max."""
        return PTPConfig(vocab_size=vocab_size, hidden_size=512, intermediate_size=1536,
                         num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=8,
                         max_position_embeddings=512)

    @staticmethod
    def mid(vocab_size: int) -> "PTPConfig":
        """~92M non-embedding: 3.4x `local`, to separate capacity from corpus size.

        With the corpus held fixed, this is the only remaining axis between a failing
        local run and the paper's 0.6B.
        """
        return PTPConfig(vocab_size=vocab_size, hidden_size=768, intermediate_size=2304,
                         num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=12,
                         max_position_embeddings=512)

    @staticmethod
    def paper(vocab_size: int) -> "PTPConfig":
        """Qwen3-0.6B shape, as published. For the cluster, not for a laptop."""
        return PTPConfig(vocab_size=vocab_size, hidden_size=1024, intermediate_size=3072,
                         num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                         max_position_embeddings=1024)


def build_model(cfg: PTPConfig, init: str = "scaled"):
    """A randomly-initialised LlamaForCausalLM of the requested shape."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    lc = LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        tie_word_embeddings=cfg.tie_word_embeddings,
        bos_token_id=None, eos_token_id=None, pad_token_id=None,
    )
    model = LlamaForCausalLM(lc)
    _init_weights(model, cfg, init)
    return model


def _init_weights(model, cfg: PTPConfig, scheme: str = "scaled") -> None:
    """Initialise the decoder.

    The paper says He init. Applied literally -- kaiming_normal_ on every Linear --
    that diverges here: He gives std = sqrt(2/fan_in), and the projections that WRITE
    INTO THE RESIDUAL STREAM (o_proj, down_proj) then compound variance layer over
    layer until activations overflow. Observed directly: loss fell 7.5 -> 2.7 over ten
    epochs, climbed back to 3.5, and the weights went NaN.

    So "scaled" (the default) is the standard transformer scheme -- normal(0, 0.02),
    with residual projections scaled by 1/sqrt(2L) (GPT-2 onwards, and what Llama
    actually uses). "he" reproduces the paper's literal wording and is kept so the
    difference can be reported rather than hidden.
    """
    import torch.nn as nn

    n_layers = cfg.num_hidden_layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if scheme == "he":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            else:
                std = 0.02
                if name.endswith(("o_proj", "down_proj")):   # writes into the residual
                    std = 0.02 / math.sqrt(2 * n_layers)
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


def pick_device(requested: str | None = None) -> str:
    import torch
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def n_params(model) -> tuple[int, int]:
    """(total, non-embedding) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for n, p in model.named_parameters() if "embed_tokens" in n)
    return total, total - emb
