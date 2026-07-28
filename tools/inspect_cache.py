"""Show the prompt sent and each model's response, reconstructed from the pilot cache.

The cache stores only ``{"k": sha256(model|prompt|seed), "t": response}`` -- the prompt and
model name are NOT persisted. But they are recomputable: you have the prompts (the behaviours
CSV), the model names (the config keys / --old/--new), and the per-draw seed is deterministic.
This tool recomputes each key and looks up the cached response, so it reconstructs the full
prompt -> response mapping for both models WITHOUT any new API calls.

Usage:
  python3 tools/inspect_cache.py \
    --config configs/real_ollama.yaml \
    --behaviours data/harmbench.csv \
    --old llama32-1b --new llama32-3b \
    --cache cache/ollama_pilot.jsonl \
    --limit 10            # show the first 10 behaviours (omit for all)

Add --only-pminus to show only behaviours where the 1B complied and the 3B refused (the P^-
candidates), or --refusals-only / --compliances-only to filter by the 1B's behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# import the SAME primitives the pilot used, so keys/seeds match exactly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recyclebench.config import Config                       # noqa: E402
from recyclebench.datasets import load_jailbreakbench        # noqa: E402
from recyclebench.realrun import _key                        # noqa: E402
from recyclebench.screening import _draw_seed                # noqa: E402
from recyclebench.judge import looks_like_refusal            # noqa: E402


def _refused(text: str) -> bool:
    # use the SAME classifier the pilot used, so the inspector's partition matches the run
    # (and inherits the curly-apostrophe / expanded-pattern fix)
    return looks_like_refusal(text)


def load_cache(path: Path) -> dict:
    """key -> response text."""
    d = {}
    if not path.exists():
        raise SystemExit(f"cache not found: {path}")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        d[row["k"]] = row["t"]
    return d


def responses_for(cache: dict, model: str, prompt: str, bid: str, k: int) -> list[str | None]:
    """The k canonical responses for (model, behaviour), or None where not yet cached."""
    out = []
    for draw in range(k):
        seed = _draw_seed(bid, -1, draw)          # same seed the pilot used (index=-1 = canonical)
        out.append(cache.get(_key(model, prompt, seed)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/real_ollama.yaml")
    ap.add_argument("--behaviours", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-pminus", action="store_true",
                    help="only behaviours where old complied AND new refused (P^- candidates)")
    ap.add_argument("--only-negflip", action="store_true",
                    help="only behaviours where old REFUSED AND new complied (safety regression)")
    ap.add_argument("--refusals-only", action="store_true", help="only where the old model refused")
    ap.add_argument("--compliances-only", action="store_true",
                    help="only where the old model complied")
    ap.add_argument("--width", type=int, default=240, help="max chars of response text shown")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    k = cfg.budget.k_canonical_samples
    behaviours = load_jailbreakbench(args.behaviours)
    cache = load_cache(Path(args.cache))

    shown = 0
    counts = {"old_screened": 0, "old_complied": 0, "new_screened": 0, "p_minus": 0}
    for b in behaviours:
        old_r = responses_for(cache, args.old, b.canonical_prompt, b.bid, k)
        if all(x is None for x in old_r):
            continue                                          # 1B not screened yet -> skip
        counts["old_screened"] += 1
        old_complied = any((x is not None and not _refused(x)) for x in old_r)
        counts["old_complied"] += int(old_complied)

        new_r = responses_for(cache, args.new, b.canonical_prompt, b.bid, k)
        new_present = any(x is not None for x in new_r)
        counts["new_screened"] += int(new_present)
        new_refused = new_present and all((x is None or _refused(x)) for x in new_r)
        new_complied = any((x is not None and not _refused(x)) for x in new_r)
        is_pminus = old_complied and new_refused
        is_negflip = (not old_complied) and new_complied     # 1B refused, 3B complied
        counts["p_minus"] += int(is_pminus)
        counts["neg_flip"] = counts.get("neg_flip", 0) + int(is_negflip)

        # filters
        if args.only_pminus and not is_pminus:
            continue
        if args.only_negflip and not is_negflip:
            continue
        if args.refusals_only and old_complied:
            continue
        if args.compliances_only and not old_complied:
            continue
        if args.limit is not None and shown >= args.limit:
            continue

        shown += 1
        tag = ("  <-- P^- candidate" if is_pminus
               else "  <-- NEGATIVE FLIP (regression)" if is_negflip else "")
        print("=" * 88)
        print(f"[{b.bid}]  category={b.category}{tag}")
        print(f"PROMPT SENT:\n  {b.canonical_prompt}")
        print(f"\n{args.old} (old)  [complied={old_complied}]:")
        for i, x in enumerate(old_r):
            body = "(not cached yet)" if x is None else x.strip()[:args.width]
            print(f"  draw {i}: {body}")
        print(f"\n{args.new} (new)  [screened={new_present}]:")
        for i, x in enumerate(new_r):
            if x is None:
                body = f"({args.old} refused -> {args.new} not queried)" if not old_complied else "(not cached yet)"
            else:
                body = x.strip()[:args.width]
            print(f"  draw {i}: {body}")

    print("=" * 88)
    print(f"SUMMARY  {args.old} screened={counts['old_screened']}  "
          f"{args.old} complied={counts['old_complied']}  "
          f"{args.new} screened={counts['new_screened']}  P^- so far={counts['p_minus']}")
    if args.limit is not None and shown >= args.limit:
        print(f"(showed first {shown}; drop --limit to see all)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
