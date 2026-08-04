"""Build a WRAPPED canonical behaviour set from the liuyanchen1015 jailbreak dataset.

Motivation: Llama-2-7b-chat refuses *raw* harmful prompts, so P^- (old answers, new refuses) is
tiny on raw benchmarks. But an *older* model tends to fall for jailbreak *framings* that a *newer*
model was patched against -- so wrapping each behaviour in a fixed jailbreak template creates a
real old-answers/new-refuses gap. The recycling question then becomes the sharp one:
"when the newer model patched this jailbreak, does the patch generalise to reworded jailbreaks?"

This is a CONTROLLED transformation, not an adaptive attack: we pick ONE fixed template, apply it
identically to every behaviour and every model, and pre-register it. No optimisation against the
target -- so the estimand (fix durability under intent-preserving reword) is preserved.

The dataset (liuyanchen1015/Llama-3.2-1B-Instruct_jailbreak_responses_with_judgment) has, per row:
  original_prompt_text   -- the raw behaviour (300 unique)
  jailbreak_prompt_name  -- which template (32 unique)
  jailbreak_prompt_text  -- the behaviour already wrapped in that template
So we don't build templates ourselves; we SELECT a template by name and emit its wrapped prompts.

Two modes:

  # 1. list the available templates + how often the 1B complied under each (pick a strong one)
  python3 tools/wrap_behaviours.py --list

  # 2. emit the wrapped canonical set for one chosen template
  python3 tools/wrap_behaviours.py --template AIM --out data/llama_wrapped_AIM.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

DATASET = "liuyanchen1015/Llama-3.2-1B-Instruct_jailbreak_responses_with_judgment"


def _load():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("needs the datasets library: pip install datasets")
    return load_dataset(DATASET)["train"]


def list_templates():
    """Show each jailbreak template and the 1B compliance rate under it (score==1).

    A template with a HIGH 1B compliance rate is a good canonical wrapper: it gets the older,
    weaker model to answer, which is exactly what populates P^- when the newer model refuses.
    """
    ds = _load()
    per = defaultdict(lambda: [0, 0])  # name -> [complied, total]
    for r in ds:
        name = r.get("jailbreak_prompt_name", "?")
        per[name][1] += 1
        if r.get("score") == 1:
            per[name][0] += 1
    rows = sorted(per.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1), reverse=True)
    print(f"{'template':<28} {'1B-complied':>11} {'total':>7} {'rate':>7}")
    print("-" * 56)
    for name, (c, t) in rows:
        print(f"{name:<28} {c:>11} {t:>7} {c/max(t,1):>6.0%}")
    print("\nPick a HIGH-rate template as --template: the older model answers it (populating P^-)")
    print("while the newer model, if patched against that jailbreak, refuses. Avoid 'original*'")
    print("(that is the raw prompt, which is why P^- was tiny).")


def emit(template, out):
    ds = _load()
    seen, rows = set(), []
    names = set()
    for r in ds:
        names.add(r.get("jailbreak_prompt_name", "?"))
        if r.get("jailbreak_prompt_name") != template:
            continue
        wrapped = (r.get("jailbreak_prompt_text") or "").strip()
        base = (r.get("original_prompt_text") or "").strip()
        key = base or wrapped
        if not wrapped or key in seen:
            continue
        seen.add(key)
        # Category left as the template name so RQ2 can group by jailbreak type if desired.
        rows.append({"Goal": wrapped, "Category": f"jb:{template}"})
    if not rows:
        raise SystemExit(f"template {template!r} not found. Available (sample): "
                         + ", ".join(sorted(names)[:12]) + " ...  run --list to see all.")
    p = Path(out); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Goal", "Category"]); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} wrapped behaviours -> {out}")
    print(f"  template : {template}")
    print(f"  each Goal is the behaviour already wrapped in this jailbreak framing")
    print(f"  next: pilot-pminus with this as --behaviours, same pair, and watch P^- vs the raw run")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="list templates + 1B compliance rates")
    g.add_argument("--template", help="the jailbreak_prompt_name to wrap with")
    ap.add_argument("--out", help="output CSV (required with --template)")
    args = ap.parse_args(argv)
    if args.list:
        list_templates()
    else:
        if not args.out:
            raise SystemExit("--out is required with --template")
        emit(args.template, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
