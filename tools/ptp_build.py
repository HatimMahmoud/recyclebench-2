#!/usr/bin/env python3
"""ptp_build.py -- build a PTP inverse model end to end (Suhail et al., 2026).

Phases, each resumable and separately time-boxed:

  probe       vocabulary probing of the forward model -> synthetic corpus   (section 3.1)
  pretrain    NLL on token-reversed synthetic sequences                     (section 3.3)
  ftdata      (prompt, response) pairs from the forward model, benign       (section 4.3)
  finetune    NLL on reversed prompt+response sequences                     (section 3.4)
  eval        held-out inversion quality -> GATE A                          (section 5)

**This local run is a feasibility test, not the deliverable.** The paper probes every
vocabulary token (~38M tokens/epoch, ~1 h on an A100); an M1 Max does a subset in
hours. Appendix E shows the pre-training stage is what establishes the inverse mapping,
so a subset narrows exactly what makes the method work. The point is to find out
whether a reduced-scale PTP learns anything at all, with a pre-specified stop rule, and
to prove the pipeline before it costs cluster time.

Everything is validated on BENIGN data (Alpaca instructions). An instrument is checked
on harmless inputs before it is pointed at the safety corpus.

Forward model defaults to qwen2.5:3b because its tokenizer is available locally and it
is fast. That model is also the project's paraphraser, so this build is NOT
decorrelated from the recycle arm -- fine for a feasibility test, not for a reported
convergent check. The cluster build uses a third family (see TODO_HUMAN.md B1).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recyclebench.ptp.metrics import lexical_suite, semantic_suite      # noqa: E402
from recyclebench.ptp.model import (PTPConfig, build_model, n_params,   # noqa: E402
                                    pick_device)
from recyclebench.ptp.probe import (CorpusStore, ProbeSpec, chunk_for_training,  # noqa: E402
                                    generate_raw, probe, seed_tokens)
from recyclebench.ptp.train import (evaluate_loss, invert_ids,          # noqa: E402
                                    make_finetune_corpus, train)
from recyclebench.ptp.vocab import CompactVocab, coverage              # noqa: E402

PHASES = ("probe", "pretrain", "ftdata", "finetune", "eval")
EOP_TOKENIZER_ID = 151643      # Qwen's <|endoftext|>; the probed corpus never emits it
OUT = ROOT / "cache" / "ptp"


def tokenizer_for(name: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(name)


def log(msg: str) -> None:
    print(f"{_dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def save(model, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    log(f"saved {path.relative_to(ROOT)}")


def load_into(model, path: Path, device: str):
    import torch
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        log(f"loaded {path.relative_to(ROOT)}")
        return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=PHASES + ("all",))
    ap.add_argument("--forward-model", default="qwen2.5:3b")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--size", default="local", choices=("smoke", "local", "mid", "paper"))
    ap.add_argument("--n-seeds", type=int, default=3000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--ft-epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--vocab-min-count", type=int, default=0,
                    help="restrict the model vocabulary to corpus tokens seen >= N times "
                         "(0 = full tokenizer vocabulary, as the paper assumes)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent probe requests; needs OLLAMA_NUM_PARALLEL >= this")
    ap.add_argument("--minutes", type=float, default=None, help="budget for THIS phase")
    ap.add_argument("--pretrained", default=None,
                    help="pre-trained checkpoint to load (default cache/ptp/"
                         "inverse_pretrained.pt, which holds whichever size ran last)")
    ap.add_argument("--finetuned", default=None,
                    help="where to write/read the fine-tuned checkpoint")
    ap.add_argument("--boundary", action="store_true",
                    help="append an end-of-prompt token so decoding has a stop "
                         "condition (see reversal.build_training_sequence); adds one "
                         "vocabulary row, so checkpoints are not interchangeable with "
                         "a run made without it")
    ap.add_argument("--gate-token-f1", type=float, default=0.30,
                    help="Gate A: held-out inversion token F1 floor")
    args = ap.parse_args(argv)

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer_for(args.tokenizer)
    device = pick_device(args.device)
    vocab_p = OUT / "compact_vocab.json"
    cv = None
    if args.vocab_min_count > 0:
        if vocab_p.exists():
            cv = CompactVocab.load(vocab_p)
        else:
            store0 = CorpusStore(OUT / f"corpus_{args.forward_model.replace(':', '_')}.jsonl")
            seqs0 = store0.sequences()
            ftp = OUT / f"ftpairs_{args.forward_model.replace(':', '_')}.jsonl"
            extra = []
            if ftp.exists():
                for line in ftp.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        extra.append(tok(r["prompt"] + r["response"],
                                         add_special_tokens=False)["input_ids"])
            cv = CompactVocab.build(seqs0, min_count=args.vocab_min_count, extra=extra)
            cv.save(vocab_p)
        log(f"compact vocabulary: {len(cv):,} of {len(tok):,} tokenizer ids "
            f"(min_count {args.vocab_min_count})")
    # `eop` is the id the DECODER stops on, i.e. compact when a compact vocabulary is
    # in use; the fine-tuning corpus is built in tokenizer ids and compacted after, so
    # that stage is given EOP_TOKENIZER_ID instead.
    eop = None
    if args.boundary:
        if cv is None:
            eop = EOP_TOKENIZER_ID
        else:
            if EOP_TOKENIZER_ID not in cv.to_compact:   # appended last: rows keep their ids
                cv = CompactVocab(list(cv.ids) + [EOP_TOKENIZER_ID])
                cv.save(OUT / "compact_vocab_eop.json")
            eop = cv.to_compact[EOP_TOKENIZER_ID]
        log(f"end-of-prompt token: tokenizer id {EOP_TOKENIZER_ID} -> decoder id {eop} "
            f"(vocabulary {len(cv) if cv else len(tok):,})")
    cfg = getattr(PTPConfig, args.size)(len(cv) if cv else len(tok))
    corpus_p = OUT / f"corpus_{args.forward_model.replace(':', '_')}.jsonl"
    ft_p = OUT / f"ftpairs_{args.forward_model.replace(':', '_')}.jsonl"
    # Explicit paths matter once more than one size has been trained: the default
    # names hold whichever run wrote them last, and loading a 92M checkpoint into a
    # 27M model fails late, after the Ollama phases have already been paid for.
    pre_w = Path(args.pretrained) if args.pretrained else OUT / "inverse_pretrained.pt"
    ft_w = Path(args.finetuned) if args.finetuned else OUT / "inverse_finetuned.pt"
    report_p = OUT / "report.json"
    report = json.loads(report_p.read_text()) if report_p.exists() else {}

    log(f"forward {args.forward_model} · tokenizer {args.tokenizer} (|V| {len(tok):,}) · "
        f"size {args.size} · device {device}")
    phases = PHASES if args.phase == "all" else (args.phase,)

    for ph in phases:
        if deadline and time.time() > deadline:
            log(f"[{ph}] skipped: budget exhausted")
            continue

        if ph == "probe":
            spec = ProbeSpec(model_id=args.forward_model, n_seeds=args.n_seeds,
                             seq_len=args.seq_len)
            store = CorpusStore(corpus_p)
            rng = random.Random(spec.seed)
            bench = []
            for f in ("data/sorrybench.csv", "data/harmbench.csv"):
                p = ROOT / f
                if p.exists():
                    bench += [l.split(",")[0] for l in
                              p.read_text(errors="ignore").splitlines()[1:2000]]
            toks = seed_tokens(tok, args.n_seeds, rng, corpus=bench or None)
            report["probe"] = probe(spec, tok, store, toks, deadline=deadline,
                                    workers=args.workers, on_log=log)
            report["probe"]["corpus"] = str(corpus_p.relative_to(ROOT))
            report["probe"]["deviation"] = (
                f"vocabulary subset: {args.n_seeds} of {len(tok)} tokens, {args.seq_len} "
                "tokens each; one corpus reused across epochs rather than regenerated "
                "on the fly (paper Appendix A prefers on-the-fly)")

        elif ph == "pretrain":
            store = CorpusStore(corpus_p)
            raw = store.sequences()
            if cv:
                log(f'  corpus coverage under the compact vocabulary: {coverage(cv, raw):.3%}')
                raw = [cv.encode(x) for x in raw]
            seqs = chunk_for_training(raw, cfg.max_position_embeddings)
            if len(seqs) < 50:
                log(f"[pretrain] only {len(seqs)} sequences -- run --phase probe first")
                continue
            rng = random.Random(7)
            rng.shuffle(seqs)
            cut = max(10, int(0.05 * len(seqs)))
            val, trn = seqs[:cut], seqs[cut:]
            model = build_model(cfg)
            tot, non_emb = n_params(model)
            log(f"[pretrain] {len(trn):,} sequences ({sum(len(s) for s in trn):,} tokens) · "
                f"model {tot/1e6:.1f}M params ({non_emb/1e6:.1f}M non-embedding)")
            rep = train(model, trn, device=device, epochs=args.epochs,
                        batch_size=args.batch_size, lr=1e-3, pad_id=0,
                        max_len=cfg.max_position_embeddings, val=val,
                        log_every=100, on_log=log)
            log(f"[pretrain] {rep}")
            save(model, pre_w)
            report["pretrain"] = {"steps": rep.steps, "final_loss": rep.final_loss,
                                  "best_loss": rep.best_loss, "val_loss": rep.val_loss,
                                  "seconds": rep.seconds, "n_train": len(trn),
                                  "n_val": len(val), "params_total": tot,
                                  "params_non_embedding": non_emb, "epochs": args.epochs}

        elif ph == "ftdata":
            prompts = json.loads((OUT / "finetune_prompts.json").read_text())
            done = set()
            if ft_p.exists():
                done = {json.loads(l)["prompt"] for l in
                        ft_p.read_text(encoding="utf-8").splitlines() if l.strip()}
            todo = [p for p in prompts if p not in done]
            log(f"[ftdata] {len(todo)} benign pairs to generate ({len(done)} stored)")
            spec = ProbeSpec(model_id=args.forward_model, temperature=0.7, seq_len=160)
            # Threaded for the same reason probing is: a single stream leaves a 3B model
            # decoding one token at a time while the machine idles. Ollama batches
            # concurrent requests only up to OLLAMA_NUM_PARALLEL, a SERVER-side setting.
            import threading
            from concurrent.futures import ThreadPoolExecutor
            state = {"n": 0, "stop": False}
            lock = threading.Lock()
            fh = ft_p.open("a", encoding="utf-8")

            def one_pair(prompt: str) -> None:
                if state["stop"]:
                    return
                if deadline and time.time() > deadline:
                    state["stop"] = True
                    return
                try:
                    r = generate_raw(spec, prompt, 160)
                except Exception as e:
                    log(f"[ftdata] {type(e).__name__}: {e}")
                    return
                if len(r.strip()) < 10:
                    return
                with lock:
                    fh.write(json.dumps({"prompt": prompt, "response": r.strip()}) + "\n")
                    fh.flush()
                    state["n"] += 1
                    n_done = state["n"]
                if n_done % 25 == 0:
                    log(f"[ftdata] {n_done}/{len(todo)}")

            try:
                if args.workers > 1:
                    with ThreadPoolExecutor(args.workers) as ex:
                        list(ex.map(one_pair, todo))
                else:
                    for prompt in todo:
                        one_pair(prompt)
            finally:
                fh.close()
            if state["stop"]:
                log(f"[ftdata] budget reached at {state['n']}/{len(todo)}")
            report["ftdata"] = {"generated": state["n"], "file": str(ft_p.relative_to(ROOT)),
                                "workers": args.workers}

        elif ph == "finetune":
            pairs = [json.loads(l) for l in ft_p.read_text(encoding="utf-8").splitlines()
                     if l.strip()] if ft_p.exists() else []
            if len(pairs) < 40:
                log(f"[finetune] only {len(pairs)} pairs -- run --phase ftdata first")
                continue
            held_p = OUT / "heldout.json"
            if held_p.exists():
                # Reuse the existing split. Re-deriving it would silently move the
                # held-out set whenever pairs are added, and every PTP figure so far is
                # scored on these 86 -- a changed denominator would break every
                # comparison without showing up anywhere in the output.
                held = json.loads(held_p.read_text())
                frozen = {h["prompt"] for h in held}
                trn = [x for x in pairs if x["prompt"] not in frozen]
                log(f"[finetune] reusing the frozen held-out set ({len(held)} pairs)")
            else:
                rng = random.Random(11)
                rng.shuffle(pairs)
                cut = max(20, int(0.2 * len(pairs)))
                held, trn = pairs[:cut], pairs[cut:]
                held_p.write_text(json.dumps(held, indent=1))
            model = build_model(cfg)
            if not load_into(model, pre_w, device):
                log("[finetune] NOTE: no pre-trained weights -- this is the Appendix E "
                    "ablation (fine-tune only), which the paper reports as failing")
            seqs = make_finetune_corpus([(p["prompt"], p["response"]) for p in trn], tok,
                                        eop=EOP_TOKENIZER_ID if args.boundary else None)
            if cv:
                seqs = [cv.encode(x) for x in seqs]
            seqs = [s[:cfg.max_position_embeddings] for s in seqs]
            rep = train(model, seqs, device=device, epochs=args.ft_epochs,
                        batch_size=args.batch_size, lr=5e-4, pad_id=0,
                        max_len=cfg.max_position_embeddings, log_every=50, on_log=log)
            log(f"[finetune] {rep}")
            save(model, ft_w)
            report["finetune"] = {"steps": rep.steps, "final_loss": rep.final_loss,
                                  "best_loss": rep.best_loss, "seconds": rep.seconds,
                                  "n_train": len(trn), "n_heldout": len(held)}

        elif ph == "eval":
            held = json.loads((OUT / "heldout.json").read_text()) \
                if (OUT / "heldout.json").exists() else []
            if not held:
                log("[eval] no held-out set -- run --phase finetune first")
                continue
            model = build_model(cfg)
            if not load_into(model, ft_w, device):
                log("[eval] no fine-tuned weights")
                continue
            hyps, refs, rows = [], [], []
            for i, r in enumerate(held, 1):
                if deadline and time.time() > deadline:
                    log(f"[eval] budget reached at {i-1}/{len(held)}")
                    break
                rid = tok(r["response"], add_special_tokens=False)["input_ids"]
                rid = rid[:cfg.max_position_embeddings - 48]
                if cv:
                    rid = cv.encode(rid)
                out_ids = invert_ids(model, rid, device=device, max_new_tokens=40,
                                     eos_id=eop)
                if cv:
                    out_ids = cv.decode(out_ids)
                got = tok.decode(out_ids, skip_special_tokens=True)
                hyps.append(got)
                refs.append(r["prompt"])
                rows.append({"true_prompt": r["prompt"], "reconstructed": got,
                             **lexical_suite(got, r["prompt"])})
                if i % 10 == 0:
                    log(f"[eval] {i}/{len(held)}")
            if not rows:
                continue
            agg = {k: sum(x[k] for x in rows) / len(rows)
                   for k in ("exact_match", "bleu", "rouge_l", "token_f1")}
            try:
                agg["cosine_mean"] = semantic_suite(hyps, refs)["cosine_mean"]
            except Exception as e:
                log(f"[eval] semantic metrics unavailable: {type(e).__name__}")
            passed = agg["token_f1"] >= args.gate_token_f1
            log("[eval] " + " · ".join(f"{k} {v:.3f}" for k, v in agg.items()))
            log(f"[eval] GATE A ({args.gate_token_f1:.2f} token F1): "
                f"{'PASS' if passed else 'FAIL'}")
            report["eval"] = {"n": len(rows), **agg, "gate_token_f1": args.gate_token_f1,
                              "gate_passed": bool(passed), "examples": rows[:25]}

        report["generated"] = _dt.datetime.now().strftime("%Y%m%d_%H%M")
        report["config"] = {"vocab_min_count": args.vocab_min_count, "boundary": args.boundary,
                            "model_vocab": len(cv) if cv else len(tok),
                            "forward_model": args.forward_model, "tokenizer": args.tokenizer,
                            "size": args.size, "vocab": len(tok), "device": device,
                            "n_seeds": args.n_seeds, "seq_len": args.seq_len}
        report_p.write_text(json.dumps(report, indent=2))
    log(f"report -> {report_p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
