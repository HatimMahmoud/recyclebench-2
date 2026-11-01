#!/usr/bin/env python3
"""tools/gate_recheck.py -- how much did the displayed model guess anchor the D5 gate-audit labels?

Rows 19-178 of `audit/gate/gate_sheet.csv` were labelled with a gemma4:e4b guess on screen (`audit/gate/MODEL_ASSISTED.txt`);
rows 1-18 were blind. This re-check (registered in prereg Amendment 2 §B4.4) re-labels a sample BLIND -- no guess, no
original label, shuffled -- in three strata:
  B   blind-baseline: 10 random rows of 1-18 (labelled blind the first time)      -> ordinary test-retest disagreement
  G0  every row of 19-178 where the model guessed "changed" (0)                   -> where anchoring would matter most
  G1  random rows of 19-178 where the model guessed "preserved" (1), to 40 rows in total
Scoring: per stratum, agreement of the blind re-label with the original label (Wilson 95%) and the direction of every
change; the excess disagreement of the model-assisted strata over the baseline is the anchoring estimate; and the D5
headline (re-opening rewordings that changed the request) recomputed with the blind re-labels where they exist.
Limitation: the labeller saw these rows two days earlier.

  python3 tools/gate_recheck.py export --out audit/gate_recheck
  python3 tools/judge_label_tui.py audit/gate_recheck/recheck_sheet.csv
  python3 tools/gate_recheck.py score --dir audit/gate_recheck
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recyclebench.audit import RUBRIC                               # noqa: E402
from recyclebench.stats import wilson_interval                      # noqa: E402

SHEET = ROOT / "audit/gate/gate_sheet.csv"
KEY = ROOT / "audit/gate/gate_key.csv"
SEED = 20261101
SHEET_COLS = ["row", "original_request", "rewrite", "human_intent_preserved", "note"]
KEY_COLS = ["row", "stratum", "original_row", "original_label", "model_guess", "audit_stratum"]


def population():
    rows = list(csv.DictReader(SHEET.open(newline="")))
    key = {r["row"]: r for r in csv.DictReader(KEY.open(newline=""))}
    for r in rows:
        r["audit_stratum"] = key.get(r["row"], {}).get("stratum", "")
    lab = [r for r in rows if (r.get("human_intent_preserved") or "").strip() in ("0", "1")]
    B = [r for r in lab if int(r["row"]) <= 18]
    G = [r for r in lab if int(r["row"]) > 18]
    G0 = [r for r in G if (r.get("model_guess") or "").strip() == "0"]
    G1 = [r for r in G if (r.get("model_guess") or "").strip() == "1"]
    return B, G0, G1


def cmd_export(args):
    out = Path(args.out)
    if (out / "recheck_sheet.csv").exists() and not args.force:
        raise SystemExit(f"{out}/recheck_sheet.csv exists -- it may hold labels. Use --force or another --out.")
    B, G0, G1 = population()
    rng = random.Random(args.seed)
    take = [dict(r, stratum="B") for r in rng.sample(B, min(args.n_baseline, len(B)))] + [dict(r, stratum="G0") for r in G0]
    take += [dict(r, stratum="G1") for r in rng.sample(G1, max(0, min(args.total - len(take), len(G1))))]
    rng.shuffle(take)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "recheck_sheet.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLS)
        w.writeheader()
        for i, r in enumerate(take, 1):
            w.writerow(dict(row=i, original_request=r["original_request"], rewrite=r["rewrite"], human_intent_preserved="", note=""))
    with (out / "recheck_key.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEY_COLS)
        w.writeheader()
        for i, r in enumerate(take, 1):
            w.writerow(dict(row=i, stratum=r["stratum"], original_row=r["row"], original_label=r["human_intent_preserved"].strip(),
                            model_guess=(r.get("model_guess") or "").strip(), audit_stratum=r["audit_stratum"]))
    meta = dict(seed=args.seed, N_B=len(B), N_G0=len(G0), N_G1=len(G1),
                n_B=sum(r["stratum"] == "B" for r in take), n_G0=sum(r["stratum"] == "G0" for r in take),
                n_G1=sum(r["stratum"] == "G1" for r in take))
    (out / "recheck_strata.json").write_text(json.dumps(meta, indent=1))
    (out / "RUBRIC.txt").write_text(RUBRIC)
    print(f"population: {len(B)} blind rows, {len(G0)} model-assisted rows guessed 'changed', {len(G1)} guessed 'preserved'")
    print(f"sheet: {len(take)} rows ({meta['n_B']} B + {meta['n_G0']} G0 + {meta['n_G1']} G1), blind and shuffled -> {out}/recheck_sheet.csv")


def _w(k, n):
    iv = wilson_interval(k, n)
    return dict(k=k, n=n, rate=round(iv.point, 4) if n else None, ci=[round(iv.low, 4), round(iv.high, 4)] if n else None)


def score(d: Path) -> dict | None:
    sheet, key, strata = d / "recheck_sheet.csv", d / "recheck_key.csv", d / "recheck_strata.json"
    if not (sheet.exists() and key.exists() and strata.exists()):
        return None
    keys = {r["row"]: r for r in csv.DictReader(key.open())}
    lab = [(keys[r["row"]], int(r["human_intent_preserved"])) for r in csv.DictReader(sheet.open())
           if (r.get("human_intent_preserved") or "").strip() in ("0", "1")]
    meta = json.loads(strata.read_text())
    res = dict(labelled=len(lab), of=meta["n_B"] + meta["n_G0"] + meta["n_G1"], strata={})
    for st in ("B", "G0", "G1"):
        xs = [(k, h) for k, h in lab if k["stratum"] == st]
        res["strata"][st] = dict(agreement=_w(sum(int(k["original_label"]) == h for k, h in xs), len(xs)),
                                 changes=[f"row {k['original_row']}: {k['original_label']} -> {h} (guess {k['model_guess'] or '-'})"
                                          for k, h in xs if int(k["original_label"]) != h])
    agree = lambda st: res["strata"][st]["agreement"]
    ma = [(k, h) for k, h in lab if k["stratum"] in ("G0", "G1")]
    if agree("B")["n"] and ma:
        res["anchoring_excess_disagreement"] = round((1 - sum(int(k["original_label"]) == h for k, h in ma) / len(ma))
                                                     - (1 - agree("B")["rate"]), 4)
        res["toward_guess"] = sum(1 for k, h in ma if int(k["original_label"]) != h and k["model_guess"] == k["original_label"])
    reopen = [(k, h) for k, h in lab if k["audit_stratum"] == "R"]
    res["reopening_rows_relabelled"] = dict(n=len(reopen), changed_request_now=sum(1 for _, h in reopen if h == 0),
                                            changed_request_before=sum(1 for k, _ in reopen if k["original_label"] == "0"))
    return res


def cmd_score(args):
    d = Path(args.dir)
    res = score(d)
    if res is None:
        raise SystemExit(f"no sheet/key/strata in {d}")
    dest = d / f"recheck_result_{_dt.datetime.now():%Y%m%d_%H%M}.json"     # dated: a re-score never replaces one
    if dest.exists():
        raise SystemExit(f"{dest} exists -- refusing to overwrite")
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", default="audit/gate_recheck")
    e.add_argument("--n-baseline", type=int, default=10)
    e.add_argument("--total", type=int, default=40)
    e.add_argument("--seed", type=int, default=SEED)
    e.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--dir", default="audit/gate_recheck")
    args = ap.parse_args(argv)
    {"export": cmd_export, "score": cmd_score}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
