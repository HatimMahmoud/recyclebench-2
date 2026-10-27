"""Does PTP's local failure survive proper decoding, or is it a decoding artefact?

`tools/ptp_diagnose.py` found a large teacher-forcing gap: on pairs it was fine-tuned
on, the 27M inverse predicts 60.5% of prompt tokens correctly given the true prefix,
but free-running greedy decoding recovers only 0.170 token F1. The mapping is
substantially learned; greedy decoding is throwing it away. That is exposure bias, and
it is a decoding problem, not a scale problem.

Two reasons a single greedy decode is also the WRONG measurement for this project:

  * the paper's section 5 reconstructs each prompt several times and uses the set;
  * RSR/PRR are "at least one admitted candidate re-opens the behaviour" estimands.
    Downstream, every candidate goes through the intent gate and is re-screened, so a
    run that produces one usable preimage among N has succeeded. Scoring one greedy
    decode measures something the pipeline never does.

So this sweeps decoding and reports the metric the pipeline actually consumes:

  greedy          temperature 0, one decode -- what every PTP figure so far reports
  best-of-N       N sampled decodes, scored at the best candidate (oracle selection)
  mean-of-N       the average candidate, i.e. what you get without an oracle
  distinct        how many of the N candidates differ, so "best-of" is not just noise

best-of-N is an oracle over candidates and would flatter any method if the downstream
step could not test them all. It is reported alongside mean-of-N and the trivial
baselines under the same N so the inflation is visible rather than hidden. Nested
prefixes of one sample set give N = 1, 2, 4, 8, 16, 32, so the curve is monotone by
construction and the marginal value of each extra candidate is readable.

The headline number is neither: it is the HIT RATE -- the share of items for which at
least one of N candidates would clear a threshold the pipeline already enforces. Two
are reported, because they answer different questions:

  cosine >= 0.75   the intent gate's own cosine layer. This is the share of responses
                   whose prompt could be reconstructed well enough to be admitted as a
                   preimage, which is what the PTP arm would actually contribute.
  token F1 >= 0.30 Gate A, the pre-registered go/no-go for the PTP arm.

Gate A was registered against a single greedy decode. If best-of-N clears it and greedy
does not, that is a change of measurement protocol and has to be registered as a
deviation before it is reported as a pass -- it is not a result that can be adopted
silently.

Candidates for one item are decoded as a batch, so N costs little more than 1.
No model calls -- pure local compute.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recyclebench.ptp.metrics import semantic_suite, token_f1, tokens  # noqa: E402
from recyclebench.ptp.model import PTPConfig, build_model, pick_device  # noqa: E402
from recyclebench.ptp.reversal import reverse_ids                      # noqa: E402
from recyclebench.ptp.vocab import CompactVocab                        # noqa: E402

OUT = ROOT / "cache" / "ptp"
NS = (1, 2, 4, 8, 16, 32)


def sample_batch(model, ctx_ids, n, *, device, max_new_tokens, temperature, top_k,
                 eos_id=None):
    """Decode `n` candidates for one context in parallel.

    Greedy when temperature == 0, in which case the n rows are identical and only the
    first is meaningful. Uses a KV cache: without one, each new token re-runs the
    whole ~500-token context and the sweep takes hours instead of minutes.

    With `eos_id` each row is cut at its own first end-of-prompt token. Rows finish at
    different steps, so the batch runs to the full budget and the tails are discarded
    afterwards rather than the loop breaking early.
    """
    import torch
    model.to(device).eval()
    ctx = torch.tensor([list(ctx_ids)] * n, dtype=torch.long, device=device)
    out = [[] for _ in range(n)]
    past, cur = None, ctx
    with torch.no_grad():
        for _ in range(max_new_tokens):
            res = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = res.past_key_values
            logits = res.logits[:, -1, :]
            if temperature and temperature > 0:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            for r in range(n):
                out[r].append(int(nxt[r, 0]))
            cur = nxt
    if eos_id is not None:
        out = [o[:o.index(eos_id)] if eos_id in o else o for o in out]
    return [reverse_ids(o) for o in out]        # reversed prompt -> prompt order


def prf(hyp: str, ref: str) -> tuple[float, float]:
    """Token precision and recall, so overrun and absence can be told apart.

    Nothing marks where the prompt begins, so decoding runs a fixed 40 tokens and
    sails past the prompt's start into whatever the model continues with. Those extra
    tokens cost precision but not recall, so low precision with decent recall means a
    missing stop condition, while low recall means the content was never recovered.
    """
    from collections import Counter
    ta, tb = Counter(tokens(hyp)), Counter(tokens(ref))
    ov = sum((ta & tb).values())
    return ov / max(1, sum(ta.values())), ov / max(1, sum(tb.values()))


def score(cands, truth, cv, tok, n_truth_tokens=None):
    """Score every candidate against the true prompt.

    Returns F1, the decoded texts, (precision, recall) and an ORACLE-LENGTH F1: the
    candidate truncated to the true prompt's token count. The oracle is not a
    reportable result -- it uses the answer -- but it bounds how much of the loss a
    stop condition could recover.
    """
    texts, oracle = [], []
    for ids in cands:
        if cv:
            ids = cv.decode(ids)
        texts.append(tok.decode(ids, skip_special_tokens=True))
        # ids are in prompt order and were generated backwards from the prompt's end,
        # so the first-generated tokens are the trailing ones.
        cut = ids[-n_truth_tokens:] if n_truth_tokens else ids
        oracle.append(tok.decode(cut, skip_special_tokens=True))
    return ([token_f1(x, truth) for x in texts], texts,
            [prf(x, truth) for x in texts],
            [token_f1(x, truth) for x in oracle])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--checkpoint", default=str(OUT / "local27m_25k" / "inverse_finetuned.pt"))
    ap.add_argument("--vocab", default=str(OUT / "compact_vocab.json"),
                    help="compact vocabulary the checkpoint was trained with")
    ap.add_argument("--eos", type=int, default=None,
                    help="compact id of the end-of-prompt token, if the run has one")
    ap.add_argument("--size", default="local", choices=("smoke", "local", "mid", "paper"))
    ap.add_argument("--n-samples", type=int, default=max(NS))
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    device = pick_device()
    cv = CompactVocab.load(Path(args.vocab)) if Path(args.vocab).exists() else None
    cfg = getattr(PTPConfig, args.size)(len(cv) if cv else len(tok))

    import torch
    model = build_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    held = json.loads((OUT / "heldout.json").read_text())
    if args.limit:
        held = held[:args.limit]
    print(f"device {device} · {args.checkpoint} · {len(held)} held-out pairs · "
          f"N={args.n_samples} temp {args.temperature} top-k {args.top_k}", flush=True)

    rows = []
    for i, r in enumerate(held, 1):
        rid = tok(r["response"], add_special_tokens=False)["input_ids"]
        rid = rid[:cfg.max_position_embeddings - args.max_new_tokens - 8]
        if cv:
            rid = cv.encode(rid)
        ctx = reverse_ids(rid)

        n_true = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])

        g = sample_batch(model, ctx, 1, device=device, max_new_tokens=args.max_new_tokens,
                         temperature=0.0, top_k=0, eos_id=args.eos)
        g_f1, g_txt, g_prf, g_or = score(g, r["prompt"], cv, tok, n_true)

        s = sample_batch(model, ctx, args.n_samples, device=device,
                         max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k, eos_id=args.eos)
        s_f1, s_txt, s_prf, s_or = score(s, r["prompt"], cv, tok, n_true)

        rows.append({"prompt": r["prompt"], "greedy_f1": g_f1[0], "greedy_text": g_txt[0],
                     "greedy_prf": g_prf[0], "greedy_oracle_f1": g_or[0],
                     "sample_f1": s_f1, "sample_text": s_txt,
                     "sample_prf": s_prf, "sample_oracle_f1": s_or,
                     "n_true_tokens": n_true, "distinct": len(set(s_txt))})
        if i == 1:
            print(f"  example · true: {r['prompt'][:70]}", flush=True)
            print(f"            greedy: {g_txt[0][:70]}", flush=True)
            print(f"            best:   {max(zip(s_f1, s_txt))[1][:70]}", flush=True)
        if i % 10 == 0:
            print(f"  {i}/{len(held)}", flush=True)

    report = {"generated": _dt.datetime.now().strftime("%Y%m%d_%H%M"),
              "checkpoint": args.checkpoint, "size": args.size, "device": device,
              "vocab": args.vocab, "eos": args.eos,
              "n_items": len(rows), "temperature": args.temperature, "top_k": args.top_k,
              "greedy": {}, "best_of": {}, "mean_of": {}, "distinct_mean": {}}

    report["greedy"]["token_f1"] = sum(x["greedy_f1"] for x in rows) / len(rows)
    for n in NS:
        if n > args.n_samples:
            continue
        best = [max(x["sample_f1"][:n]) for x in rows]
        mean = [sum(x["sample_f1"][:n]) / n for x in rows]
        report["best_of"][n] = sum(best) / len(best)
        report["mean_of"][n] = sum(mean) / len(mean)
        report["distinct_mean"][n] = sum(len(set(x["sample_text"][:n])) for x in rows) / len(rows)

    # Per-candidate cosine, so the hit rates can use the gate's own threshold. One
    # flat encode of every candidate, then regrouped -- 86 x 32 is a single batch.
    refs = [x["prompt"] for x in rows]
    try:
        report["greedy"]["cosine"] = semantic_suite([x["greedy_text"] for x in rows],
                                                    refs)["cosine_mean"]
        flat_h, flat_r = [], []
        for x in rows:
            flat_h.extend(x["sample_text"])
            flat_r.extend([x["prompt"]] * len(x["sample_text"]))
        sims = semantic_suite(flat_h, flat_r)["cosine"]
        k = args.n_samples
        for j, x in enumerate(rows):
            x["sample_cos"] = sims[j * k:(j + 1) * k]
    except Exception as e:
        print(f"  semantic metrics unavailable: {type(e).__name__}: {e}", flush=True)
        for x in rows:
            x["sample_cos"] = None

    # Hit rates: share of items with >= 1 candidate over a threshold the pipeline uses.
    report["hit_rate_cos_075"] = {}
    report["hit_rate_f1_030"] = {}
    report["best_of_cosine"] = {}
    for n in NS:
        if n > args.n_samples:
            continue
        report["hit_rate_f1_030"][n] = sum(
            max(x["sample_f1"][:n]) >= 0.30 for x in rows) / len(rows)
        if rows[0].get("sample_cos"):
            report["hit_rate_cos_075"][n] = sum(
                max(x["sample_cos"][:n]) >= 0.75 for x in rows) / len(rows)
            report["best_of_cosine"][n] = sum(
                max(x["sample_cos"][:n]) for x in rows) / len(rows)
    report["greedy"]["hit_f1_030"] = sum(x["greedy_f1"] >= 0.30 for x in rows) / len(rows)

    # Precision/recall and the oracle-length ceiling: is the residue absent, or buried?
    report["greedy"]["precision"] = sum(x["greedy_prf"][0] for x in rows) / len(rows)
    report["greedy"]["recall"] = sum(x["greedy_prf"][1] for x in rows) / len(rows)
    report["greedy"]["oracle_len_f1"] = sum(x["greedy_oracle_f1"] for x in rows) / len(rows)
    report["best_of_recall"], report["best_of_precision"] = {}, {}
    report["best_of_oracle_len_f1"] = {}
    for n in NS:
        if n > args.n_samples:
            continue
        # scored at the candidate that maximises the metric in question
        report["best_of_recall"][n] = sum(
            max(p[1] for p in x["sample_prf"][:n]) for x in rows) / len(rows)
        report["best_of_precision"][n] = sum(
            max(p[0] for p in x["sample_prf"][:n]) for x in rows) / len(rows)
        report["best_of_oracle_len_f1"][n] = sum(
            max(x["sample_oracle_f1"][:n]) for x in rows) / len(rows)
    report["mean_true_prompt_tokens"] = sum(x["n_true_tokens"] for x in rows) / len(rows)
    report["generated_tokens"] = args.max_new_tokens

    report["examples"] = [{k: x[k] for k in ("prompt", "greedy_text", "greedy_f1")}
                          | {"best_text": max(zip(x["sample_f1"], x["sample_text"]))[1],
                             "best_f1": max(x["sample_f1"])}
                          for x in rows[:15]]

    dest = Path(args.out) if args.out else OUT / f"decode_sweep_{report['generated']}.json"
    dest.write_text(json.dumps(report, indent=1))

    print(f"\n{'decode':16s}{'token F1':>10s}{'distinct':>10s}")
    print(f"{'greedy (N=1)':16s}{report['greedy']['token_f1']:>10.3f}{1:>10.1f}")
    for n in sorted(report["best_of"]):
        print(f"{'best-of-' + str(n):16s}{report['best_of'][n]:>10.3f}"
              f"{report['distinct_mean'][n]:>10.1f}")
    for n in sorted(report["mean_of"]):
        print(f"{'mean-of-' + str(n):16s}{report['mean_of'][n]:>10.3f}")
    if report["best_of_cosine"]:
        print(f"\n{'decode':16s}{'cosine':>10s}{'hit cos>=.75':>14s}{'hit F1>=.30':>13s}")
        print(f"{'greedy (N=1)':16s}{report['greedy']['cosine']:>10.3f}"
              f"{'—':>14s}{report['greedy']['hit_f1_030']:>13.3f}")
        for n in sorted(report["best_of_cosine"]):
            print(f"{'best-of-' + str(n):16s}{report['best_of_cosine'][n]:>10.3f}"
                  f"{report['hit_rate_cos_075'][n]:>14.3f}{report['hit_rate_f1_030'][n]:>13.3f}")
    print(f"\n{'decode':16s}{'precision':>11s}{'recall':>9s}{'oracle-len F1':>15s}")
    print(f"{'greedy (N=1)':16s}{report['greedy']['precision']:>11.3f}"
          f"{report['greedy']['recall']:>9.3f}{report['greedy']['oracle_len_f1']:>15.3f}")
    for n in sorted(report["best_of_recall"]):
        print(f"{'best-of-' + str(n):16s}{report['best_of_precision'][n]:>11.3f}"
              f"{report['best_of_recall'][n]:>9.3f}"
              f"{report['best_of_oracle_len_f1'][n]:>15.3f}")
    emitted = (f"stops at the end-of-prompt token (budget {report['generated_tokens']})"
               if args.eos is not None else f"always emits {report['generated_tokens']}")
    print(f"  (true prompts average {report['mean_true_prompt_tokens']:.1f} tokens; "
          f"decoding {emitted})")
    print("\nbaselines on this set: token F1  generic 0.246 / echo 0.236   "
          "cosine  generic 0.193 / echo 0.713")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
