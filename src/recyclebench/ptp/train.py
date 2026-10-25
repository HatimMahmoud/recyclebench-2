"""Training and reversed-decoding for the PTP inverse model.

The objective is ordinary autoregressive NLL -- the method is in the data, which is
token-reversed (see `reversal`). Two stages, as in the paper:

  pre-training  on synthetic sequences probed from the forward model (section 3.3)
  fine-tuning   on reversed (prompt, response) pairs (section 3.4)

Appendix E is explicit that the pre-training stage is what establishes the inverse
mapping and that fine-tuning alone on a few hundred real pairs does not. So a run that
skips pre-training is expected to fail, and `evaluate_inversion` exists to say so with
a number rather than a vibe.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from .reversal import build_training_sequence, reverse_ids


@dataclass
class TrainReport:
    steps: int
    final_loss: float
    best_loss: float
    val_loss: Optional[float]
    seconds: float
    device: str
    diverged: bool = False

    def __str__(self) -> str:
        v = "n/a" if self.val_loss is None else f"{self.val_loss:.4f}"
        tag = "  DIVERGED" if self.diverged else ""
        return (f"steps {self.steps} · loss {self.final_loss:.4f} (best {self.best_loss:.4f}) · "
                f"val {v} · {self.seconds:.0f}s on {self.device}{tag}")


def batches(seqs: Sequence[Sequence[int]], batch_size: int, pad_id: int, device,
            max_len: Optional[int] = None):
    """Right-pad a list of id sequences into (input_ids, labels) with -100 on padding."""
    import torch
    order = list(range(len(seqs)))
    random.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [list(seqs[j]) for j in order[i:i + batch_size]]
        if max_len:
            chunk = [c[:max_len] for c in chunk]
        n = max(len(c) for c in chunk)
        inp = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), n), -100, dtype=torch.long)
        for r, c in enumerate(chunk):
            inp[r, :len(c)] = torch.tensor(c, dtype=torch.long)
            lab[r, :len(c)] = torch.tensor(c, dtype=torch.long)
        yield inp.to(device), lab.to(device)


def train(model, seqs: Sequence[Sequence[int]], *, device: str, epochs: int = 1,
          batch_size: int = 8, lr: float = 1e-3, pad_id: int = 0,
          max_len: Optional[int] = None, val: Optional[Sequence[Sequence[int]]] = None,
          log_every: int = 50, warmup: int = 200,
          on_log: Optional[Callable[[str], None]] = None) -> TrainReport:
    """Standard next-token NLL on already-reversed sequences.

    lr 1e-3 for pre-training and 5e-4 for fine-tuning, per Algorithm 1, but behind a
    linear warmup: at full lr from step 0 this diverges.

    A non-finite loss ABORTS. Silently training on through a NaN produced a model whose
    every tensor was NaN and an evaluation that scored 0.000 on every metric -- which
    reads exactly like "the method does not work at this scale" and is not that at all.
    """
    import torch
    log = on_log or print
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # cap warmup at 10% of the run, so a short run still reaches full lr
    total_steps = max(1, epochs * math.ceil(len(seqs) / max(1, batch_size)))
    warm = max(1, min(warmup, total_steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / warm))
    t0, step, last, best = time.time(), 0, float("nan"), float("inf")
    for ep in range(epochs):
        for inp, lab in batches(seqs, batch_size, pad_id, device, max_len):
            out = model(input_ids=inp, labels=lab)
            loss = out.loss
            if not torch.isfinite(loss):
                log(f"  ABORT at step {step}: loss is {float(loss)} -- diverged. "
                    f"Lower lr (now {lr}), lengthen warmup (now {warmup}), or check init.")
                return TrainReport(step, float(loss), best, None, time.time() - t0,
                                   device, diverged=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                log(f"  ABORT at step {step}: gradient norm is {float(gn)} -- diverged.")
                return TrainReport(step, last, best, None, time.time() - t0,
                                   device, diverged=True)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            last = float(loss.detach().cpu())
            best = min(best, last)
            if step % log_every == 0:
                log(f"  step {step} · epoch {ep + 1}/{epochs} · loss {last:.4f} "
                    f"· lr {sched.get_last_lr()[0]:.2e}")
    vl = evaluate_loss(model, val, device=device, batch_size=batch_size,
                       pad_id=pad_id, max_len=max_len) if val else None
    return TrainReport(step, last, best, vl, time.time() - t0, device)


def evaluate_loss(model, seqs, *, device, batch_size=8, pad_id=0, max_len=None) -> float:
    import torch
    model.to(device).eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for inp, lab in batches(list(seqs), batch_size, pad_id, device, max_len):
            tot += float(model(input_ids=inp, labels=lab).loss.detach().cpu()) * inp.shape[0]
            n += inp.shape[0]
    model.train()
    return tot / max(1, n)


def invert_ids(model, response_ids: Sequence[int], *, device: str, max_new_tokens: int = 48,
               temperature: float = 0.0, top_k: int = 0, eos_id: Optional[int] = None) -> list[int]:
    """PTP inference (section 3.5).

    Feed Reverse(y) as context, autoregressively generate the reversed prompt, then
    un-reverse. Greedy at temperature 0; sampling (temperature/top-k) gives the
    multiple distinct reconstructions the paper's section 5 relies on.
    """
    import torch
    model.to(device).eval()
    ctx = list(reverse_ids(response_ids))
    out: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([ctx + out], dtype=torch.long, device=device)
            logits = model(input_ids=inp).logits[0, -1]
            if temperature and temperature > 0:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[-1]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
            else:
                nxt = int(torch.argmax(logits))
            if eos_id is not None and nxt == eos_id:
                break
            out.append(nxt)
    return reverse_ids(out)          # reversed prompt -> prompt order


def evaluate_inversion(model, pairs, tokenizer, *, device: str, max_new_tokens: int = 48,
                       limit: Optional[int] = None) -> dict:
    """Closed-loop sanity: can the model recover prompts it was trained to invert?

    Reports exact match and token F1 against the true prompt. This measures the
    IMPLEMENTATION, not the method -- on a handful of memorised pairs a high score
    only says the loop is wired correctly.
    """
    from .metrics import exact_match, token_f1
    rows = list(pairs)[:limit] if limit else list(pairs)
    em, f1 = [], []
    for prompt, response in rows:
        rid = tokenizer(response, add_special_tokens=False)["input_ids"]
        got = tokenizer.decode(invert_ids(model, rid, device=device,
                                          max_new_tokens=max_new_tokens),
                               skip_special_tokens=True)
        em.append(exact_match(got, prompt))
        f1.append(token_f1(got, prompt))
    return {"n": len(rows),
            "exact_match": sum(em) / max(1, len(em)),
            "token_f1": sum(f1) / max(1, len(f1))}


def make_finetune_corpus(pairs: Iterable[tuple[str, str]], tokenizer) -> list[list[int]]:
    """Reversed (prompt + response) sequences, the section 3.4 fine-tuning corpus."""
    return [build_training_sequence(p, r, tokenizer) for p, r in pairs]
