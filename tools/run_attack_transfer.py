#!/usr/bin/env python3
"""
tools/run_attack_transfer.py  --  Arm 3: known-attack transfer (Wei et al. jailbreak suite).

Distinct from the recycle arm. Recycle asks "does the gen-3.1 fix survive a BENIGN,
meaning-preserving reword?"; this asks "does it survive KNOWN ADVERSARIAL ATTACKS?" The
attack prompts come pre-built from the HF dataset (liuyanchen1015/..._jailbreak_responses),
so nothing is generated and NOTHING is intent-gated (gating an attack is a category error).
We query ONE model (the one under test, e.g. llama31-8b) on each attack-transformed prompt,
grade it with the SAME decorrelated judge the other arms use, and report per-class bypass
rates with Wilson CIs -- overall, and restricted to behaviours the model refused canonically
(the "transfer-on-refused" rate, which is the real robustness number).

RUN (strong first pilot: baseline + 4 attacks that actually move an 8B, all behaviours):
  python3 tools/run_attack_transfer.py \
    --config configs/recycle_3_31.yaml --model llama31-8b \
    --classes AIM evil_confidant prefix_injection refusal_suppression \
    --cache cache/attack_transfer.jsonl \
    --out results/runs/attack_transfer/llama31-8b_pilot.json

Flags:
  --classes        attack classes to test (jailbreak_prompt_name values). Any subset.
  --baseline-class canonical no-transform class for the denominator (default original_prompt).
  --n-behaviours   sample this many base behaviours (seeded); default all 300.
  --model          cfg.real key of the model under test (default llama31-8b).
Resumable: every model call is cached by (model, prompt, seed), so an interrupt resumes.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HF_DATASET = "liuyanchen1015/Llama-3.2-1B-Instruct_jailbreak_responses_with_judgment"


def _load_config(p):
    from recyclebench.config import Config
    return Config.load(str(p))


def _build_judge(cfg):
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    from recyclebench.backends.real_judge import build_ensemble
    jm = {role: ProviderModel(f"judge:{role}", ProviderConfig(**spec))
          for role, spec in (getattr(cfg, "judges", None) or {}).items()}
    judge = build_ensemble(jm) if jm else None
    if judge is None or not getattr(judge, "members", None):
        raise SystemExit("no 'judges:' block in config — add one (see recycle_3_31.yaml) so "
                         "this arm is graded by the same decorrelated judge as the others.")
    print(f"  [judge] {','.join(jm)}")
    return judge


def _load_grouped(dataset_id, from_disk=None):
    """Return {prompt_name: {jailbreak_prompt_name: jailbreak_prompt_text}} + the base prompt."""
    from datasets import load_dataset, load_from_disk
    d = load_from_disk(from_disk) if from_disk else load_dataset(dataset_id)["train"]
    by_beh: dict[str, dict[str, str]] = defaultdict(dict)
    base: dict[str, str] = {}
    for r in d:
        pn = r["prompt_name"]
        cls = r["jailbreak_prompt_name"]
        by_beh[pn].setdefault(cls, r["jailbreak_prompt_text"])   # first wins on dups
        base.setdefault(pn, r.get("original_prompt_text", ""))
    return by_beh, base


def main():
    ap = argparse.ArgumentParser(description="known-attack transfer arm")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--model", default="llama31-8b", help="cfg.real key of the model under test")
    ap.add_argument("--classes", nargs="+", required=True,
                    help="attack classes (jailbreak_prompt_name) to test")
    ap.add_argument("--baseline-class", default="original_prompt",
                    help="no-transform class for the refusal denominator")
    ap.add_argument("--n-behaviours", type=int, default=None, help="seeded subset of base behaviours")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--dataset", default=HF_DATASET)
    ap.add_argument("--from-disk", default=None, help="load_from_disk path instead of the hub")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None, help="sampling seed (default cfg.seed)")
    args = ap.parse_args()

    from recyclebench.backends.providers import load_provider_model
    from recyclebench.realrun import ResponseCache, CachedModel
    from recyclebench.types import Response
    from recyclebench.stats import wilson_interval
    import random

    cfg = _load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    judge = _build_judge(cfg)
    cache = ResponseCache(str(args.cache), enabled=True)
    model = CachedModel(load_provider_model(args.model, "openai_compatible", cfg), cache)

    by_beh, base = _load_grouped(args.dataset, args.from_disk)
    behaviours = sorted(by_beh)
    if args.n_behaviours and args.n_behaviours < len(behaviours):
        rng = random.Random(seed)
        behaviours = sorted(rng.sample(behaviours, args.n_behaviours))
    print(f"  [data] {len(behaviours)} behaviours · classes={args.classes} · baseline={args.baseline_class}")

    def ask(text):
        r = model.respond(text, seed=seed)
        rtext = r.text if hasattr(r, "text") else str(r)
        return int(judge.classify(text, Response(text=rtext, latent_harmful=None)))  # 1=comply

    # 1. baseline: did the model comply on the canonical (no-transform) prompt?
    baseline_comply = {}
    for i, b in enumerate(behaviours):
        txt = by_beh[b].get(args.baseline_class) or base.get(b)
        if not txt:
            continue
        baseline_comply[b] = ask(txt)
        if (i + 1) % 25 == 0:
            print(f"    baseline {i+1}/{len(behaviours)}  (calls={model.calls} hits={model.hits})")
    refused = [b for b, c in baseline_comply.items() if c == 0]
    print(f"  [baseline] model complied canonically on {sum(baseline_comply.values())}/"
          f"{len(baseline_comply)}; refused {len(refused)} (the transfer denominator)")

    # 2. each attack class
    results = {}
    for cls in args.classes:
        broke = n = broke_ref = n_ref = 0
        per_beh = {}
        for b in behaviours:
            txt = by_beh[b].get(cls)
            if not txt:
                continue
            v = ask(txt)
            per_beh[b] = v
            n += 1; broke += v
            if b in baseline_comply and baseline_comply[b] == 0:   # refused canonically
                n_ref += 1; broke_ref += v
        by = wilson_interval(broke, n) if n else None
        tr = wilson_interval(broke_ref, n_ref) if n_ref else None
        results[cls] = {
            "n": n, "n_broke": broke,
            "bypass_rate": round(broke / n, 4) if n else None,
            "bypass_ci": [round(by.low, 4), round(by.high, 4)] if by else None,
            "n_refused_baseline": n_ref, "n_broke_among_refused": broke_ref,
            "transfer_on_refused": round(broke_ref / n_ref, 4) if n_ref else None,
            "transfer_ci": [round(tr.low, 4), round(tr.high, 4)] if tr else None,
        }
        print(f"  [{cls}] bypass {broke}/{n}"
              + (f" = {broke/n:.1%}" if n else "")
              + (f" | transfer-on-refused {broke_ref}/{n_ref} = {broke_ref/n_ref:.1%}" if n_ref else ""))
    cache.close()

    out = {"model": args.model, "baseline_class": args.baseline_class,
           "n_behaviours": len(behaviours), "n_refused_baseline": len(refused),
           "judge": list((getattr(cfg, "judges", None) or {}).keys()),
           "classes": results,
           "note": "Attack-transfer arm (NOT recycle). No intent gate. 'transfer_on_refused' is "
                   "the headline: among behaviours the model refused canonically, how often the "
                   "attack re-opens it. RAW/judge-uncalibrated unless the judge is calibrated."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print("  (calibrate the same way as recycle if you want corrected rates: the judge is shared.)")


if __name__ == "__main__":
    main()
