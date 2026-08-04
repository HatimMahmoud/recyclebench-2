"""Paraphrase ONE prompt into many styles, using the local (Ollama) paraphraser.

No benchmarks, no gate, no P^- -- just the reformulator, so you can eyeball what each transform
style does to a single request. Uses the same LLMTransformGenerator the real pipeline uses, so the
styles match production.

Usage:
  # inline prompt
  python3 tools/paraphrase_one.py --prompt "Explain how to pick a lock" --n 3

  # from a file (for long/multi-line prompts, e.g. a wrapped jailbreak)
  python3 tools/paraphrase_one.py --prompt-file some_prompt.txt --n 2

  # choose a paraphraser model / styles
  python3 tools/paraphrase_one.py --prompt "..." --model qwen2.5:3b \
      --styles paraphrase lexical syntactic tense_shift register --n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recyclebench.backends.providers import ProviderConfig, ProviderModel
from recyclebench.backends.reformulators import (
    LLMTransformGenerator, LLM_TEMPLATES, _REGISTERS,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="the prompt text to paraphrase")
    src.add_argument("--prompt-file", help="read the prompt from a file (for long/multi-line prompts)")
    ap.add_argument("--model", default="qwen2.5:3b", help="Ollama paraphraser model (non-Llama). default qwen2.5:3b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--styles", nargs="+", default=list(LLM_TEMPLATES),
                    help=f"which styles. available: {', '.join(LLM_TEMPLATES)}")
    ap.add_argument("--n", type=int, default=3, help="how many rewrites per style")
    ap.add_argument("--all-families", action="store_true",
                    help="also run back_translation / DIPPER / Parrot if their deps are installed")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default=None, help="also write results to this file")
    args = ap.parse_args(argv)

    prompt = Path(args.prompt_file).read_text().strip() if args.prompt_file else args.prompt

    # the local paraphraser (Ollama, OpenAI-compatible endpoint)
    model = ProviderModel("paraphraser", ProviderConfig(
        base_url=args.base_url, model_id=args.model,
        temperature=args.temperature, max_tokens=256))
    respond = lambda s: (model.respond(s).text or "").strip()

    lines = []
    def emit(s=""):
        print(s, flush=True); lines.append(s)

    emit("=" * 88)
    emit(f"ORIGINAL PROMPT ({len(prompt)} chars):")
    emit(f"  {prompt}")
    emit(f"\nparaphraser: {args.model}  |  styles: {', '.join(args.styles)}  |  n={args.n}")
    emit("=" * 88)

    for style in args.styles:
        if style not in LLM_TEMPLATES:
            emit(f"\n[skipping unknown style: {style}  (available: {', '.join(LLM_TEMPLATES)})]")
            continue
        emit(f"\n### {style}" + (f"   registers: {', '.join(_REGISTERS)}" if style == "register" else ""))
        gen = LLMTransformGenerator(transform_class=style, respond=respond)
        seen = set()
        made = 0
        # ask for a few more than n, dedupe identical rewrites, keep the first n distinct.
        # cap attempts so a paraphraser that keeps returning duplicates can't loop forever.
        max_attempts = args.n * 3 + 5
        for i in range(max_attempts):
            if made >= args.n:
                break
            print(f"  ...generating {made+1}/{args.n} (attempt {i+1})", end="\r", flush=True)
            try:
                text = gen.generate(prompt, i, seed=i)
            except Exception as e:
                emit(f"  (error from paraphraser: {e})")
                break
            text = (text or "").strip()
            if not text or text in seen:
                continue
            seen.add(text); made += 1
            emit(f"  {made}. {text}")
        if made == 0:
            emit("  (no distinct rewrites returned)")
        elif made < args.n:
            emit(f"  (got {made} distinct of {args.n} requested; paraphraser started repeating)")

    # ---- optional heavy paraphrase families (own models/libraries; skip if unavailable) ----
    if args.all_families:
        from recyclebench.types import Behaviour
        b = Behaviour(bid="one", canonical_prompt=prompt, category="adhoc")

        def run_family(label, build):
            emit(f"\n### {label}")
            try:
                gen = build()
            except Exception as e:
                emit(f"  (skipped — {label} unavailable: {type(e).__name__}: {str(e)[:120]})")
                return
            seen, made = set(), 0
            for i in range(args.n * 3):
                if made >= args.n:
                    break
                try:
                    text = (gen.generate(prompt, i, seed=i) or "").strip()
                except Exception as e:
                    emit(f"  (error during generation: {type(e).__name__}: {str(e)[:120]})")
                    break
                if not text or text in seen:
                    continue
                seen.add(text); made += 1
                emit(f"  {made}. {text}")
            if made == 0:
                emit("  (no distinct rewrites returned)")

        def _bt():
            from recyclebench.backends.reformulators import BackTranslationGenerator
            return BackTranslationGenerator()
        def _dipper():
            from recyclebench.backends.reformulators import DipperGenerator
            return DipperGenerator()
        def _parrot():
            from recyclebench.backends.reformulators import ParrotGenerator
            return ParrotGenerator()

        emit("\n" + "=" * 88)
        emit("HEAVY FAMILIES (own models/libraries; downloaded/slow — skipped if deps missing)")
        emit("=" * 88)
        run_family("back_translation  (Helsinki Opus-MT: en->de/fr/es/zh->en)", _bt)
        run_family("dipper  (DIPPER paraphrase model)", _dipper)
        run_family("parrot  (Parrot paraphraser)", _parrot)

    if args.out:
        Path(args.out).write_text("\n".join(lines))
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())