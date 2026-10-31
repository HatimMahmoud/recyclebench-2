#!/usr/bin/env python3
"""tools/release_c2.py -- build the release package for contribution C2 (the intent-verified variant set), minimal-disclosure.

Source: the frozen SORRY-Bench lineage manifest (440 behaviours x 8 candidates), the stored qwen2.5:14b verdicts of all six
target models on every admitted rewording, and the human gate-audit labels. Policy (exposé §10(3), ETHICS.md; the text tier
is the supervisor's decision):

  always released   per candidate: behaviour id, SORRY-Bench category, sha256 of the canonical prompt (to join with a
                    licensed copy -- the benchmark's own text is NOT redistributed), transform class, admitted flag, the gate
                    layer that rejected it, the gate scores the gate actually COMPUTED (layers after the failing one were
                    never evaluated and are null, not the manifest's 1.0 placeholder; `llm_intent` was not run and is null),
                    surface distance, the human audit label where one exists, and per model whether the reply complied
  never released    any model response; any canonical prompt text; the text of any rewording that re-opened a refusal
  --text non-reopening   additionally releases the text of admitted rewordings that NO model complied with (all six
                    verdicts present and 0). Default: no rewording text at all.

A leak check runs before anything is written: no response field, no canonical prompt, and no withheld rewording text may
occur in any output file; released texts must have six refusals each. Output: results/release_c2_<stamp>/ (never overwrites).

  python3 tools/release_c2.py                      # ids, scores, labels, outcomes; no text
  python3 tools/release_c2.py --text non-reopening
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import lineage_core as lc                                      # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.datasets import load_jailbreakbench          # noqa: E402
from recyclebench.manifest import Manifest                     # noqa: E402

CFG = "configs/real_llama_lineage_all.yaml"
MANIFEST = "results/runs/lineage_sorrybench_20261020/manifest.jsonl"
VERDICTS = "cache/judge_lineage_sorrybench.jsonl"
GATE_ORDER = (("cosine", "cosine"), ("bertscore", "bertscore"), ("nli_entail", "nli_entail"))
VARIANT_COLS = ["bid", "category", "prompt_sha256", "index", "transform_class", "admitted", "rejected_by", "cosine",
                "bertscore", "nli_entail", "llm_intent", "surface_distance", "human_intent_preserved"]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def computed_scores(row, thr):
    """keep the scores the short-circuiting gate evaluated; the first layer under its threshold rejected the candidate"""
    out, rej = {}, None
    for key, tkey in GATE_ORDER:
        if rej is not None:
            out[key] = None
            continue
        v = row.get(key)
        out[key] = None if v is None else round(float(v), 4)
        if not row["admitted"] and v is not None and v < getattr(thr, tkey):
            rej = key
    if not row["admitted"] and rej is None:            # no persisted layer failed: the unpersisted lexical layer did
        rej = "lexical_divergence (inferred)"
        out = {k: None for k, _ in GATE_ORDER}
    return out, rej


def gate_labels():
    """(bid, index) -> human_intent_preserved from the D5 audit, for the SORRY-Bench lineage manifest"""
    key, sheet = ROOT / "audit/gate/gate_key.csv", ROOT / "audit/gate/gate_sheet.csv"
    if not (key.exists() and sheet.exists()):
        return {}
    lab = {r["row"]: (r.get("human_intent_preserved") or "").strip() for r in csv.DictReader(sheet.open())}
    return {(r["bid"], int(r["index"])): lab.get(r["row"], "") for r in csv.DictReader(key.open())
            if r.get("manifest", "").endswith(MANIFEST.split("/")[-1]) and r.get("bench") == "sorrybench"}


def build(text_policy: str):
    cfg = Config.load(ROOT / CFG)
    models = list(cfg.lineage)
    behaviours = load_jailbreakbench(ROOT / "data/sorrybench.csv")
    bmap = {b.bid: b for b in behaviours}
    raw = [json.loads(l) for l in (ROOT / MANIFEST).read_text().splitlines() if l.strip()]
    texts = {(v["bid"], v["index"]): v for r in raw if r["type"] == "admitted" for v in r["variants"]}
    audit = [r for r in raw if r["type"] == "audit"]
    cache = lc.load_response_cache(ROOT / "cache/llama_gen.jsonl")
    verd = lc.StoredVerdicts(lc.VerdictStore(ROOT / VERDICTS), lc.judge_tag_from_cfg(cfg))
    manifest = Manifest.load(ROOT / MANIFEST)
    outcome = defaultdict(dict)                          # (bid, index) -> model -> 0/1/None
    responses = set()
    for it in lc.iter_items(cfg, behaviours, manifest, cache, models):
        if it.response:
            responses.add(it.response)                   # canonical and variant replies: none may leak
        if it.kind == "variant":
            outcome[(it.bid, it.index)][it.model] = None if it.response is None else verd.get(it.prompt, it.response)
    labels = gate_labels()
    rows, released, withheld = [], {}, set()
    for a in audit:
        k = (a["bid"], a["index"])
        sc, rej = computed_scores(a, cfg.gate)
        v = texts.get(k)
        oc = outcome.get(k, {})
        row = dict(bid=a["bid"], category=bmap[a["bid"]].category, prompt_sha256=sha(bmap[a["bid"]].canonical_prompt),
                   index=a["index"], transform_class=a["transform_class"], admitted=int(bool(a["admitted"])),
                   rejected_by=rej or "", **{x: ("" if y is None else y) for x, y in sc.items()}, llm_intent="",
                   surface_distance=round(v["surface_distance"], 4) if v else "",
                   human_intent_preserved=labels.get(k, ""))
        for m in models:
            row[f"complied_{m}"] = "" if oc.get(m) is None else oc[m]
        if v:
            safe = all(oc.get(m) == 0 for m in models)
            if text_policy == "non-reopening" and safe:
                released[k] = v["text"]
            else:
                withheld.add(v["text"])
        if text_policy != "none":
            row["text"] = released.get(k, "")
        rows.append(row)
    return dict(cfg=cfg, models=models, behaviours=behaviours, rows=rows, released=released, withheld=withheld,
                responses=responses, outcome=outcome)


def leak_check(files: dict, b) -> list:
    """files: name -> text content. Returns the list of violations (empty = clean)."""
    problems = []
    blob = "\n".join(files.values())
    released = set(b["released"].values())
    for t in b["withheld"]:
        if t not in released and len(t) > 12 and t in blob:
            problems.append(f"withheld rewording text present: {t[:40]}...")
    for be in b["behaviours"]:
        if len(be.canonical_prompt) > 12 and be.canonical_prompt in blob:
            problems.append(f"canonical prompt text present: {be.bid}")
    for r in b["responses"]:
        if len(r) > 40 and r in blob:
            problems.append("a model response is present")
            break
    for k, t in b["released"].items():
        if not all(b["outcome"][k].get(m) == 0 for m in b["models"]):
            problems.append(f"released text without six refusals: {k}")
    return problems


def datacard(b, policy, counts):
    m = ", ".join(b["models"])
    return f"""# RecycleBench intent-verified variant set (C2) -- data card

**What it is.** Every candidate rewording generated for the 440 SORRY-Bench requests in the RecycleBench study (8 per
request, six transform classes: paraphrase, syntactic, lexical, back-translation, tense shift, register; paraphraser
qwen2.5:3b), with the intent gate's decision and the scores it computed, the human intent-audit label where one exists, and,
for every admitted rewording, whether each of six Llama models ({m}) complied (judge qwen2.5:14b, temperature 0).

**What it is for.** Measuring whether refusal behaviour survives meaning-preserving rewording (the thesis's RSR / PBR),
re-applying other gate thresholds offline, and auditing the gate. It is not a jailbreak corpus.

**Composition.** {counts['candidates']} candidates, {counts['admitted']} admitted, {counts['behaviours']} behaviours.
Columns: `bid` (row index of SORRY-Bench as loaded by the study), `category`, `prompt_sha256` (of the canonical request; join
with your own licensed copy of SORRY-Bench -- its text is not redistributed here), `index`, `transform_class`, `admitted`,
`rejected_by`, gate scores `cosine` / `bertscore` / `nli_entail` (blank = that layer was never evaluated because an earlier
layer rejected the candidate), `llm_intent` (blank: not run), `surface_distance` (admitted only), `human_intent_preserved`
(1 / 0 / blank), `complied_<model>` (1 / 0 / blank = not judged){', and `text`' if policy != 'none' else ''}.

**Text policy: `{policy}`.** {'No rewording text is included.' if policy == 'none' else f'Text is included only for the {counts["released"]} admitted rewordings that no model complied with; the text of every rewording that re-opened a refusal is withheld.'}
No model response and no canonical prompt is included. Withheld material is available only through controlled academic
channels (ETHICS.md, exposé §10).

**Gate.** Lexical divergence → sentence-embedding cosine ≥ 0.75 → BERTScore ≥ 0.85 → NLI entailment ≥ 0.60, ANDed and
short-circuiting. Human audit (2026-10-29, 178 rewordings over four benchmarks): 3 of 98 admitted rewordings that re-opened a
fix changed the request; rows 19–178 were labelled with a model guess visible (stated in the thesis).

**Known limitations.** One paraphraser; one judge (calibrated on Llama 3.1 8B responses, bundle κ 0.833); the gate's
false-rejection rate was measured on a related population (prereg Amendment 2 §B4.1); `bid` is a row position, not a
SORRY-Bench question id.

**Licence.** Derived from SORRY-Bench; redistribution follows the source benchmark's terms, which must be checked before any
public release. Provenance: `{MANIFEST}`, verdicts `{VERDICTS}`, built {_dt.date.today().isoformat()} by `tools/release_c2.py`.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", choices=("none", "non-reopening"), default="none")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    b = build(args.text)
    cols = VARIANT_COLS + [f"complied_{m}" for m in b["models"]] + (["text"] if args.text != "none" else [])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    w.writerows(b["rows"])
    counts = dict(candidates=len(b["rows"]), admitted=sum(r["admitted"] for r in b["rows"]),
                  behaviours=len({r["bid"] for r in b["rows"]}), released=len(b["released"]),
                  withheld=len(b["withheld"]) - len(set(b["released"].values()) & b["withheld"]),
                  audited=sum(1 for r in b["rows"] if r["human_intent_preserved"] != ""))
    files = {"variants.csv": buf.getvalue(), "DATACARD.md": datacard(b, args.text, counts)}
    problems = leak_check(files, b)
    if problems:
        raise SystemExit("LEAK CHECK FAILED -- nothing written:\n  " + "\n  ".join(problems[:20]))
    out = Path(args.out) if args.out else ROOT / "results" / f"release_c2_{_dt.datetime.now():%Y%m%d_%H%M}"
    if out.exists():
        raise SystemExit(f"{out} exists -- refusing to overwrite")
    out.mkdir(parents=True)
    for name, content in files.items():
        (out / name).write_text(content)
    meta = dict(created=_dt.datetime.now().isoformat(timespec="seconds"), text_policy=args.text, counts=counts,
                leak_check="passed: no response, no canonical prompt, no withheld rewording; released texts have six refusals",
                files={n: sha(c) for n, c in files.items()}, source=dict(manifest=MANIFEST, verdicts=VERDICTS, config=CFG))
    (out / "release_manifest.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: {counts} · leak check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
