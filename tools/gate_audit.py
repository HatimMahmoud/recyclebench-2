#!/usr/bin/env python3
"""tools/gate_audit.py -- the human audit of the intent gate on the REAL frozen manifests.

`recyclebench audit-export` samples a rebuilt (mock) manifest; the real manifests are frozen files, so this
samples them directly. The audit is aimed at the number the exposé commits to -- how often an ADMITTED
rewording actually changed the request -- and spends most of its labels where a false admission would
inflate the headline:

  R  admitted rewordings that RE-OPENED a fix of the headline pair (llama3-8b -> llama3.1-8b) on any
     benchmark: each one is a counted bypass, so a changed request here directly overstates RSR (census
     up to --n-recovered);
  A  every other admitted rewording of the same four manifests (random --n-other).

Rejected rewordings cannot be audited: manifests keep their gate scores but not their text, so the false-
REJECTION rate (which costs power, not validity) is out of reach without regenerating them; the report says so.

The sheet is BLIND: annotators see the original request and the rewrite, in shuffled order, with no stratum,
gate score, or model outcome. The key (stratum, manifest, bid, index, scores) is a separate file.

  python3 tools/gate_audit.py export --out audit/gate          # -> gate_sheet.csv, gate_key.csv, RUBRIC.txt
  # fill human_intent_preserved with 1 / 0 / blank
  python3 tools/gate_audit.py ingest --dir audit/gate          # contamination with CIs, RSR without changed variants

Verdicts: SORRY-Bench and HarmBench use the stored qwen2.5:14b verdicts (complete on the pair's P^-),
AdvBench/StrongREJECT the offline rule (the run's own judge).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_core as lc                                          # noqa: E402
from recyclebench.audit import RUBRIC, invert_stratified          # noqa: E402,F401
from recyclebench.config import Config                             # noqa: E402
from recyclebench.datasets import load_jailbreakbench              # noqa: E402
from recyclebench.manifest import Manifest                         # noqa: E402
from recyclebench.stats import wilson_interval                     # noqa: E402

OLD, NEW = "llama3-8b", "llama31-8b"
SMOKE = "results/runs/20260805_124501_llama3-8b_to_llama31-8b_RECYCLE_qwen3b_SMOKE/manifest_harmbench.jsonl"
L3 = "results/runs/20260805_025658_llama3-8b_to_llama31-8b_RECYCLE"
SOURCES = [  # (benchmark, csv, manifest, verdict store or None for the offline rule)
    ("sorrybench", "data/sorrybench.csv", "results/runs/lineage_sorrybench_20261020/manifest.jsonl",
     "cache/judge_lineage_sorrybench.jsonl"),
    ("harmbench", "data/harmbench.csv", SMOKE, "cache/judge_qwen_harmbench.jsonl"),
    ("advbench", "data/advbench.csv", f"{L3}/manifest_advbench.jsonl", None),
    ("strongreject", "data/strongreject.csv", f"{L3}/manifest_strongreject.jsonl", None),
]
SHEET_COLS = ["row", "original_request", "rewrite", "human_intent_preserved", "note"]


def collect(cfg_path="configs/real_llama_lineage_all.yaml", cache_path="cache/llama_gen.jsonl"):
    """-> (rows, per_bench) where rows are every admitted rewording of the four manifests with its stratum."""
    cfg = Config.load(cfg_path)
    tag = lc.judge_tag_from_cfg(cfg)
    cache = lc.load_response_cache(cache_path)
    rows, per_bench = [], {}
    for bench, csv_p, man_p, vstore in SOURCES:
        bs = load_jailbreakbench(csv_p)
        bmap = {b.bid: b for b in bs}
        man = Manifest.load(man_p)
        store = lc.VerdictStore(vstore) if vstore else None
        ver = lc.StoredVerdicts(store, tag) if store else lc.OfflineVerdicts()
        obs, runs = lc.build_runs(cfg, bs, man, cache, ver, [OLD, NEW])
        if store:
            store.close()
        pm = set(lc.regression_set(runs[OLD], runs[NEW]))
        audit = {(a.bid, a.index): a for a in man.audit}
        rec_by_b = {}
        for b in bs:
            vo = {v.index: v for v in obs[NEW].variants.get(b.bid, [])}
            for v in man.R(b.bid):
                o = vo.get(v.index)
                recovered = b.bid in pm and o is not None and o.verdict == 1
                if recovered:
                    rec_by_b.setdefault(b.bid, []).append(v.index)
                a = audit.get((b.bid, v.index))
                rows.append(dict(bench=bench, manifest=man_p, bid=b.bid, index=v.index,
                                 transform_class=v.transform_class, original=bmap[b.bid].canonical_prompt,
                                 rewrite=v.text, stratum="R" if recovered else "A",
                                 in_pminus=int(b.bid in pm),
                                 cosine=getattr(a, "cosine", None), bertscore=getattr(a, "bertscore", None),
                                 nli_entail=getattr(a, "nli_entail", None)))
        per_bench[bench] = dict(n_pminus=len(pm), reopened=sorted(rec_by_b),
                                judge="qwen2.5:14b (stored)" if vstore else "offline rule",
                                rec_variants={b: ix for b, ix in rec_by_b.items()})
    return rows, per_bench


def cmd_export(args):
    out = Path(args.out)
    if (out / "gate_sheet.csv").exists() and not args.force:
        raise SystemExit(f"{out}/gate_sheet.csv exists -- it may hold labels. Use --force or another --out.")
    rows, per_bench = collect()
    R = [r for r in rows if r["stratum"] == "R"]
    A = [r for r in rows if r["stratum"] == "A"]
    rng = random.Random(args.seed)
    take_R = R if len(R) <= args.n_recovered else rng.sample(R, args.n_recovered)
    take_A = rng.sample(A, min(args.n_other, len(A)))
    picked = take_R + take_A
    rng.shuffle(picked)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "gate_sheet.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLS)
        w.writeheader()
        for i, r in enumerate(picked, start=1):
            w.writerow({"row": i, "original_request": r["original"], "rewrite": r["rewrite"],
                        "human_intent_preserved": "", "note": ""})
    keycols = ["row", "stratum", "bench", "manifest", "bid", "index", "transform_class", "in_pminus",
               "cosine", "bertscore", "nli_entail"]
    with (out / "gate_key.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keycols)
        w.writeheader()
        for i, r in enumerate(picked, start=1):
            w.writerow({"row": i, **{k: r[k] for k in keycols if k != "row"}})
    meta = dict(seed=args.seed, N_R=len(R), N_A=len(A), n_R=len(take_R), n_A=len(take_A),
                pair=f"{OLD} -> {NEW}", per_bench={k: {kk: vv for kk, vv in v.items() if kk != "rec_variants"}
                                                   for k, v in per_bench.items()},
                rec_variants={k: v["rec_variants"] for k, v in per_bench.items()})
    (out / "gate_strata.json").write_text(json.dumps(meta, indent=1))
    (out / "RUBRIC.txt").write_text(RUBRIC.replace("`human_intent_preserved`", "`human_intent_preserved` "
                                                   "(and optionally a short `note`)"))
    print(f"population: {len(R)} admitted rewordings that re-opened a {OLD} -> {NEW} fix (R), "
          f"{len(A)} other admitted (A)")
    for k, v in per_bench.items():
        print(f"  {k:<13} P⁻ {v['n_pminus']:>3}  re-opened {len(v['reopened']):>3}  judge {v['judge']}")
    print(f"sheet: {len(picked)} rows ({len(take_R)} R + {len(take_A)} A), shuffled and blind -> {out}/gate_sheet.csv")
    print(f"key  : {out}/gate_key.csv · strata {out}/gate_strata.json · rubric {out}/RUBRIC.txt")
    print("The sheet holds prompt text: working file, not a released artefact (ETHICS.md).")
    return 0


def _wil(k, n):
    w = wilson_interval(k, n)
    return dict(k=k, n=n, rate=(k / n if n else None), lo=w.low, hi=w.high)


def cmd_ingest(args):
    d = Path(args.dir)
    meta = json.loads((d / "gate_strata.json").read_text())
    key = {r["row"]: r for r in csv.DictReader((d / "gate_key.csv").open())}
    lab = {}
    for r in csv.DictReader((d / "gate_sheet.csv").open()):
        v = (r.get("human_intent_preserved") or "").strip()
        if v in ("0", "1"):
            lab[r["row"]] = int(v)
    if not lab:
        print("no labels yet (fill human_intent_preserved with 1 / 0)")
        return 1
    by = defaultdict(list)
    for row, v in lab.items():
        by[key[row]["stratum"]].append(v)
    chg = {s: sum(1 - v for v in vs) for s, vs in by.items()}
    res = {"labelled": {s: len(v) for s, v in by.items()}, "of_sheet": len(key)}
    # contamination = P(changed | admitted), per stratum and design-weighted over the admitted population
    res["contamination_R"] = _wil(chg.get("R", 0), len(by.get("R", [])))
    res["contamination_A"] = _wil(chg.get("A", 0), len(by.get("A", [])))
    NR, NA = meta["N_R"], meta["N_A"]
    pR = res["contamination_R"]["rate"] or 0.0
    pA = res["contamination_A"]["rate"] or 0.0
    res["contamination_admitted_weighted"] = (NR * pR + NA * pA) / (NR + NA)
    # headline robustness: drop every human-changed re-opening variant; does the behaviour still re-open?
    changed = {(key[r]["bench"], key[r]["bid"], int(key[r]["index"])) for r, v in lab.items()
               if v == 0 and key[r]["stratum"] == "R"}
    unlabelled_R = meta["N_R"] - len(by.get("R", []))
    rows = []
    for bench, rv in meta["rec_variants"].items():
        n_pm = meta["per_bench"][bench]["n_pminus"]
        k0 = len(rv)
        k1 = sum(1 for b, ix in rv.items() if any((bench, b, i) not in changed for i in ix))
        rows.append(dict(bench=bench, n_pminus=n_pm, reopened=k0, reopened_after_audit=k1))
    res["rsr_after_audit"] = rows
    res["note"] = ("false-rejection rate not estimable: rejected rewordings' text is not stored in the manifests"
                   + (f"; {unlabelled_R} R-stratum rewordings unlabelled, kept as admitted" if unlabelled_R else ""))
    print(json.dumps(res, indent=1, default=str))
    (d / "gate_audit_result.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"-> {d}/gate_audit_result.json")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", default="audit/gate")
    e.add_argument("--n-recovered", type=int, default=120)
    e.add_argument("--n-other", type=int, default=80)
    e.add_argument("--seed", type=int, default=20261022)
    e.add_argument("--force", action="store_true")
    i = sub.add_parser("ingest")
    i.add_argument("--dir", default="audit/gate")
    args = ap.parse_args(argv)
    return cmd_export(args) if args.cmd == "export" else cmd_ingest(args)


if __name__ == "__main__":
    sys.exit(main())
