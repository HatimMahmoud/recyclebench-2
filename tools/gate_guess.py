#!/usr/bin/env python3
"""gate_guess.py -- a model's guess for each row of the gate audit sheet, to speed up labelling.

The gate audit (prereg Amendment 1, D5) was registered as BLIND human labels. Showing a guess
anchors the labeller, so a sheet labelled with this column filled is MODEL-ASSISTED, not blind,
and has to be reported that way; the tool writes audit/gate/MODEL_ASSISTED.txt to record it.

Why not the gate's own intent layer: gemma4's discriminative check (tools/regate_intent.py)
already admitted all 178 rewordings on the sheet -- the sheet is the gate's admissions -- so it
would guess 1 everywhere. This asks the sheet's own question with the sheet's own rubric
(audit/gate/RUBRIC.txt) directly, so the guess can say 0.

Model: gemma4:e4b -- non-Llama (targets), non-Qwen (paraphraser and judge). Temperature 0.
The guess goes to its own column `model_guess`; `human_intent_preserved` is never touched.

  python3 tools/gate_guess.py                 # resumable; fills only empty model_guess cells
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROMPT = """{rubric}

ORIGINAL: {original}

REWRITE: {rewrite}

Answer with a single digit: 1 (preserved) or 0 (changed)."""


def save(path: Path, rows, fieldnames):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)                         # atomic: a kill never leaves a half-written sheet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default="audit/gate/gate_sheet.csv")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    args = ap.parse_args(argv)

    sheet = ROOT / args.sheet
    rubric = (sheet.parent / "RUBRIC.txt").read_text().split("If genuinely uncertain")[0].strip()
    with sheet.open(newline="") as fh:
        r = csv.DictReader(fh)
        rows, fields = list(r), list(r.fieldnames)
    blind = [x["row"] for x in rows if x.get("human_intent_preserved", "") in ("0", "1")]
    if "model_guess" not in fields:
        backup = sheet.with_name(sheet.name + f".bak_before_model_guess_{_dt.datetime.now():%Y%m%d_%H%M}")
        shutil.copy2(sheet, backup)
        print(f"backup   : {backup.relative_to(ROOT)}")
        fields.insert(fields.index("human_intent_preserved"), "model_guess")
        for x in rows:
            x["model_guess"] = ""
    todo = [x for x in rows if x.get("model_guess", "") not in ("0", "1")]
    print(f"sheet    : {args.sheet}   {len(rows)} rows   to guess {len(todo)}   model {args.model}")

    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    m = ProviderModel("gate:guess", ProviderConfig(base_url=args.base_url, model_id=args.model, temperature=0.0,
                                                   max_tokens=4, extra_body={"reasoning_effort": "none"}))
    t0, odd = time.time(), 0
    for i, x in enumerate(todo, 1):
        out = (m.respond(PROMPT.format(rubric=rubric, original=x["original_request"], rewrite=x["rewrite"])).text or "").strip()
        g = next((c for c in out if c in "01"), "")
        odd += int(g == "")
        x["model_guess"] = g                  # blank when the model gave no digit: no guess shown
        if i % 10 == 0 or i == len(todo):
            save(sheet, rows, fields)
            print(f"  {i}/{len(todo)}   {(time.time() - t0) / i:.1f}s/row", flush=True)
    save(sheet, rows, fields)
    note = sheet.parent / "MODEL_ASSISTED.txt"
    if not note.exists():
        note.write_text(
            f"{_dt.date.today()}: column `model_guess` added by tools/gate_guess.py ({args.model}, temperature 0,\n"
            "the rubric's own question). From then on the labeller saw the guess, so the labels are\n"
            "MODEL-ASSISTED, not blind: a deviation from prereg Amendment 1 (D5, blind gate audit), decided by\n"
            f"the candidate. Rows labelled before the column existed were labelled blind: {', '.join(blind) or 'none'}.\n")
    g = [x["model_guess"] for x in rows]
    print(f"done     : 1={g.count('1')}  0={g.count('0')}  no digit={odd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
