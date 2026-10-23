#!/usr/bin/env python3
"""tools/judge_rpe.py -- re-judge the RPE (output-inversion) arm with the decorrelated judge.

Every RPE run on disk was graded either by the target model itself (SORRY-Bench: target-as-judge, circular)
or by whatever judge the probe was given, and the probe stores only a boolean `broke`. This replays the
config's `judges:` block (qwen2.5:14b rubric) over every gate-admitted candidate's stored reply, exactly as
tools/run_independent_probe.py calls it -- judge.classify(candidate_text, reply) -- and writes each verdict to
an append-only store keyed like tools/judge_lineage.py (judge tag + prompt + response), so it is resumable and
a different judge never reuses these verdicts.

NO target-model calls: the replies are already in the RPE manifests. Only the judge runs.

  python3 tools/judge_rpe.py --dry-run
  python3 tools/judge_rpe.py --out cache/judge_rpe_qwen.jsonl --stop-at 07:45
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_core as lc                                      # noqa: E402
import study_data as sd                                        # noqa: E402
from judge_lineage import build_judge                          # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.judge import looks_like_refusal              # noqa: E402
from recyclebench.types import Response                        # noqa: E402


def admitted_replies(spec):
    """-> [(bid, candidate, reply, stored_broke)] for every admitted candidate with a stored reply."""
    p = sd.ROOT / spec.path
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not (r.get("prompt") and str(r.get("bid", "")).startswith("jbb-")):
            continue
        for x in r.get("details") or []:
            if x.get("admitted") and x.get("response") is not None:
                out.append((r["bid"], x["text"], x["response"], x.get("broke")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/recycle_3_31.yaml", help="config whose `judges:` block to replay")
    ap.add_argument("--out", default="cache/judge_rpe_qwen.jsonl")
    ap.add_argument("--stop-at", default=None, metavar="HH:MM", help="stop gracefully at this local time")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    tag = lc.judge_tag_from_cfg(cfg)
    if not tag:
        raise SystemExit("config has no `judges:` block")
    store = lc.VerdictStore(args.out)
    deadline = None
    if args.stop_at:
        hh, mm = (int(x) for x in args.stop_at.split(":"))
        deadline = _dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= _dt.datetime.now():
            deadline += _dt.timedelta(days=1)

    per_run = {sp.id: admitted_replies(sp) for sp in sd.RPE_SPECS}
    todo, seen = [], set()
    for rid, rows in per_run.items():
        for bid, cand, reply, _ in rows:
            k = lc.verdict_key(tag, cand, reply)
            if k not in seen and store.get(k) is None:
                seen.add(k)
                todo.append((k, cand, reply))
    print(f"judge   : {tag}")
    for rid, rows in per_run.items():
        print(f"  {rid:<26} {len(rows):>4} admitted replies")
    print(f"remaining: {len(todo)} unique (candidate, reply) pairs · stored {len(store)} in {args.out}")

    if not args.dry_run and todo:
        judge = build_judge(cfg)
        t0, done = time.time(), 0
        try:
            for k, cand, reply in todo:
                if deadline and _dt.datetime.now() >= deadline:
                    print(f"reached --stop-at {args.stop_at} after {done}; re-run to resume")
                    break
                store.put(k, int(judge.classify(cand, Response(text=reply, latent_harmful=None))))
                done += 1
                if done % 10 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)}  {(time.time() - t0) / done:.1f}s/item", flush=True)
        except KeyboardInterrupt:
            print(f"interrupted after {done}; re-run to resume")
        finally:
            store.close()

    # stored verdict vs this judge vs offline rule, per run (bundle = behaviour re-opened by any admitted reply)
    print(f"\n{'run':<26} {'replies':>7} {'stored':>7} {'judge':>7} {'offline':>7}   behaviours re-opened stored/judge/offline")
    for rid, rows in per_run.items():
        if not rows:
            continue
        c, bund = Counter(), {}
        for bid, cand, reply, broke in rows:
            q = store.get(lc.verdict_key(tag, cand, reply))
            off = int(not looks_like_refusal(reply))
            c["n"] += 1
            c["stored"] += int(bool(broke))
            c["judge"] += int(q == 1)
            c["judged"] += int(q is not None)
            c["offline"] += off
            b = bund.setdefault(bid, [0, 0, 0, True])
            b[0] |= int(bool(broke))
            b[1] |= int(q == 1)
            b[2] |= off
            b[3] &= q is not None
        complete = all(b[3] for b in bund.values())
        print(f"{rid:<26} {c['n']:>7} {c['stored']:>7} {c['judge']:>7}{'' if c['judged'] == c['n'] else '*':1}"
              f"{c['offline']:>7}   {sum(b[0] for b in bund.values())}/{sum(b[1] for b in bund.values())}"
              f"{'' if complete else '*'}/{sum(b[2] for b in bund.values())} of {len(bund)}")
    print("(* = not every reply judged yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
