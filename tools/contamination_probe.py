#!/usr/bin/env python3
"""tools/contamination_probe.py -- verbatim-completion probe of the benchmarks (exposé §8 contamination audit).

Registered in PREREG_CONTAMINATION_20261031.md. For every benchmark prompt the model sees the first half of
its words as RAW text (Ollama /api/generate, `raw: true`, no chat template) and continues greedily
(temperature 0, Ollama's other defaults as in the study). A prompt counts as **reproduced** when the
continuation repeats the next words of the true prompt exactly (case and punctuation ignored) for at least
min(8, len(suffix)) words, with a suffix of at least 5 words. The per-benchmark reproduction rate is set
against SORRY-Bench and the held-out set, which postdate every probed model's pre-training cutoff and give the
floor of generic, non-memorised completion.

Only numbers are written (no prompt or continuation text): results/runs/contamination_<stamp>/summary.{json,txt}.
The raw continuations live in the separate store (default cache/contamination/), resumable.

  python3 tools/contamination_probe.py --phase all --models llama3-8b llama31-8b llama2-7b
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recyclebench.config import Config                                # noqa: E402
from recyclebench.datasets import load_jailbreakbench                 # noqa: E402

STORE = ROOT / "cache" / "contamination" / "continuations.jsonl"
BENCHES = ("advbench", "harmbench", "strongreject", "sorrybench", "heldout_jbb_guided")
CONTROLS = ("sorrybench", "heldout_jbb_guided")       # released after every probed model's cutoff
MIN_SUFFIX, MATCH_WORDS, NUM_PREDICT = 5, 8, 32


def log(msg):
    print(f"{_dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def words(s):
    return [w for w in (re.sub(r"[^\w\s]", " ", s.lower())).split() if w]


def split(prompt):
    """(prefix text, suffix words) at the word midpoint, or None if the suffix is too short."""
    raw = prompt.split()
    cut = len(raw) // 2
    suf = words(" ".join(raw[cut:]))
    if len(suf) < MIN_SUFFIX:
        return None
    return " ".join(raw[:cut]), suf


def matched(cont, suf):
    """How many leading suffix words the continuation reproduces in order."""
    c = words(cont)
    n = 0
    for a, b in zip(c, suf):
        if a != b:
            break
        n += 1
    return n


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.mem = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.mem[d["k"]] = d["t"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a")

    @staticmethod
    def key(model, prefix):
        return hashlib.sha256(f"{model}|{prefix}|{NUM_PREDICT}".encode()).hexdigest()[:32]

    def put(self, k, t):
        self.mem[k] = t
        self.fh.write(json.dumps({"k": k, "t": t}) + "\n")
        self.fh.flush()


def generate(model_id, prefix, base="http://localhost:11434"):
    body = json.dumps(dict(model=model_id, prompt=prefix, raw=True, stream=False,
                           options=dict(temperature=0, num_predict=NUM_PREDICT))).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(f"{base}/api/generate", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["response"]
        except Exception as exc:                     # transient Ollama failure
            last = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"{model_id}: generate failed: {last}")


def items():
    out = {}
    for b in BENCHES:
        rows = []
        for beh in load_jailbreakbench(ROOT / "data" / f"{b}.csv"):
            s = split(beh.canonical_prompt)
            if s:
                rows.append(s)
        out[b] = list(dict.fromkeys((p, tuple(s)) for p, s in rows))     # HarmBench has duplicates
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=("gen", "score", "all"))
    ap.add_argument("--config", default="configs/real_llama_lineage_all.yaml")
    ap.add_argument("--models", nargs="*", default=["llama3-8b", "llama31-8b", "llama2-7b"])
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts per benchmark (smoke test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = Config.load(ROOT / args.config)
    ids = {m: cfg.real[m]["model_id"] for m in args.models}
    its = {b: v[:args.limit] if args.limit else v for b, v in items().items()}
    st = Store(args.store)
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    log(f"contamination probe · {sum(map(len, its.values()))} prompts × {len(ids)} models · store {args.store}")
    if args.phase in ("gen", "all"):
        for m, mid in ids.items():
            todo = [p for v in its.values() for p, _ in v if Store.key(m, p) not in st.mem]
            log(f"[gen {m}] {len(todo)} continuations")
            for k, p in enumerate(todo, 1):
                if deadline and time.time() > deadline:
                    log(f"[gen {m}] budget reached at {k - 1}/{len(todo)}")
                    return 0
                st.put(Store.key(m, p), generate(mid, p))
                if k % 200 == 0:
                    log(f"[gen {m}] {k}/{len(todo)}")
    if args.phase in ("score", "all"):
        from recyclebench.stats import wilson_interval
        res = dict(created=_dt.datetime.now().isoformat(timespec="seconds"), prereg="PREREG_CONTAMINATION_20261031.md",
                   rule=f"reproduced = next min({MATCH_WORDS}, |suffix|) words exact; |suffix| >= {MIN_SUFFIX}; "
                        f"greedy, raw, num_predict {NUM_PREDICT}", controls=list(CONTROLS), models={})
        lines = [f"contamination probe · {res['created']} · {res['rule']}", ""]
        for m in ids:
            per = {}
            for b, v in its.items():
                got = [(st.mem.get(Store.key(m, p)), list(s)) for p, s in v]
                got = [(c, s) for c, s in got if c is not None]
                hits = [matched(c, s) >= min(MATCH_WORDS, len(s)) for c, s in got]
                frac = [matched(c, s) / len(s) for c, s in got]
                w = wilson_interval(sum(hits), len(hits))
                per[b] = dict(n=len(hits), of=len(v), reproduced=sum(hits),
                              rate=round(w.point, 4) if hits else None,
                              ci=[round(w.low, 4), round(w.high, 4)] if hits else None,
                              mean_suffix_matched=round(sum(frac) / len(frac), 4) if frac else None)
            res["models"][m] = per
            lines.append(m)
            for b, d in per.items():
                tag = " (control)" if b in CONTROLS else ""
                r = f"{d['reproduced']}/{d['n']} = {d['rate']:.3f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}]" if d["n"] else "-"
                lines.append(f"  {b + tag:<32}{r:>30}   mean suffix matched {d['mean_suffix_matched'] or 0:.3f}")
        out = Path(args.out) if args.out else ROOT / "results" / "runs" / f"contamination_{_dt.datetime.now():%Y%m%d_%H%M}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(res, indent=1))
        (out / "summary.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
