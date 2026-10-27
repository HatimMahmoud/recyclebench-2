"""Give the inverse model a stop condition, and re-measure.

`tools/ptp_decode_sweep.py` showed that the local PTP numbers are dominated by a
missing termination signal rather than by model quality:

    true prompts average 10.8 tokens; decoding always emits 40
    best-of-32   precision 0.140   recall 0.522   token F1 0.217
                 F1 truncated to the true length (oracle): 0.347

Recall says over half the prompt's tokens are recovered. Precision says nearly three
quarters of what is emitted is overrun past the prompt's start, because nothing tells
the model the prompt has ended. In s_rev = Reverse(prompt + response) the sequence
runs backwards from the response's last token to the prompt's FIRST token, so the
prompt's start is the end of the sequence -- and it is unmarked. Generation therefore
runs for a fixed budget and sails on into invented text.

This appends one end-of-prompt token to every fine-tuning sequence and decodes with it
as the stop token, which makes the task well posed. It is an implementation fix, not a
tuning choice: without a stop condition the inverse problem has no defined output
length, and the oracle-length figure above is unreachable by any honest decoder.

The comparison is kept clean -- same pre-trained checkpoint, same 347 fine-tuning
pairs, same hyper-parameters, same held-out 86, same decode sweep. The only difference
is the boundary token. The marker reuses the tokenizer's <|endoftext|> id, which the
probed corpus never emits, appended to the compact vocabulary as one extra row so the
pre-trained embedding rows keep their indices.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recyclebench.ptp.model import PTPConfig, build_model, pick_device   # noqa: E402
from recyclebench.ptp.reversal import build_training_sequence            # noqa: E402
from recyclebench.ptp.train import train                                 # noqa: E402
from recyclebench.ptp.vocab import CompactVocab                          # noqa: E402

OUT = ROOT / "cache" / "ptp"
EOP_TOKENIZER_ID = 151643          # <|endoftext|>, absent from the probed corpus


def widen(state, extra: int, std: float = 0.02):
    """Grow the embedding (and untied head) by `extra` rows, keeping row indices."""
    import torch
    for k in ("model.embed_tokens.weight", "lm_head.weight"):
        if k in state:
            w = state[k]
            pad = w.new_empty(extra, w.shape[1]).normal_(0.0, std)
            state[k] = torch.cat([w, pad], dim=0)
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--pretrained", default=str(OUT / "local27m_25k" / "inverse_pretrained.pt"))
    ap.add_argument("--size", default="local", choices=("smoke", "local", "mid", "paper"))
    ap.add_argument("--ft-epochs", type=int, default=12)
    ap.add_argument("--n-pairs", type=int, default=None,
                    help="train on this many fine-tuning pairs (sampled with a fixed "
                         "seed). With --ft-epochs chosen to hold total steps constant, "
                         "this separates data diversity from optimisation compute.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-ckpt", default=str(OUT / "inverse_boundary.pt"))
    ap.add_argument("--out-vocab", default=str(OUT / "compact_vocab_eop.json"))
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    device = pick_device()

    cv = CompactVocab.load(OUT / "compact_vocab.json")
    assert EOP_TOKENIZER_ID not in cv.to_compact, "boundary id already in the vocabulary"
    cv2 = CompactVocab(list(cv.ids) + [EOP_TOKENIZER_ID])
    assert all(cv2.to_compact[t] == cv.to_compact[t] for t in cv.ids), \
        "appending the boundary id must not move existing rows"
    eop = cv2.to_compact[EOP_TOKENIZER_ID]
    cv2.save(Path(args.out_vocab))
    print(f"vocab {len(cv)} -> {len(cv2)} · end-of-prompt compact id {eop} · device {device}",
          flush=True)

    pairs = [json.loads(l) for l in
             (OUT / "ftpairs_qwen2.5_3b.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    held = json.loads((OUT / "heldout.json").read_text())
    held_prompts = {h["prompt"] for h in held}
    trn = [p for p in pairs if p["prompt"] not in held_prompts]
    if args.n_pairs and args.n_pairs < len(trn):
        import random as _r
        trn = _r.Random(11).sample(trn, args.n_pairs)
    print(f"{len(trn)} fine-tuning pairs, {len(held)} held out (same split as every run)",
          flush=True)

    cfg = getattr(PTPConfig, args.size)(len(cv2))
    model = build_model(cfg)
    state = widen(torch.load(args.pretrained, map_location="cpu"), len(cv2) - len(cv))
    model.load_state_dict(state)
    print(f"loaded {args.pretrained} and widened the embedding by one row", flush=True)

    seqs = []
    for p in trn:
        s = cv2.encode(build_training_sequence(p["prompt"], p["response"], tok))
        seqs.append((s + [eop])[:cfg.max_position_embeddings])
    rep = train(model, seqs, device=device, epochs=args.ft_epochs,
                batch_size=args.batch_size, lr=5e-4, pad_id=0,
                max_len=cfg.max_position_embeddings, log_every=50,
                on_log=lambda s: print("  " + s, flush=True))
    print(f"[finetune] {rep}", flush=True)
    torch.save(model.state_dict(), args.out_ckpt)
    print(f"saved {args.out_ckpt}", flush=True)

    (OUT / f"boundary_report_{_dt.datetime.now():%Y%m%d_%H%M}.json").write_text(json.dumps({
        "pretrained": args.pretrained, "checkpoint": args.out_ckpt,
        "vocab": args.out_vocab, "eop_compact_id": eop,
        "eop_tokenizer_id": EOP_TOKENIZER_ID, "n_train": len(trn), "n_heldout": len(held),
        "ft_epochs": args.ft_epochs, "steps": rep.steps, "final_loss": rep.final_loss,
        "best_loss": rep.best_loss, "diverged": rep.diverged,
    }, indent=1))
    print("\nnow score it against the no-boundary run:\n"
          f"  python3 -u tools/ptp_decode_sweep.py --checkpoint {args.out_ckpt} "
          f"--vocab {args.out_vocab} --eos {eop} --n-samples 32", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
