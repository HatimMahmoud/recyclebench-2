#!/usr/bin/env python3
"""tools/regate_intent.py -- apply gate layer (d), the discriminative intent check, to frozen manifests post hoc.

No run so far had an intent model configured, so every admitted rewording passed only lexical / cosine /
BERTScore / NLI. Layer (d) is ANDed after them, so applying it later can only REMOVE admissions: RSR and PBR on
the 4-layer gate are recomputable from the response cache with no new target-model calls.

The check is the package's own DiscriminativeIntentClassifier (recyclebench.backends.intent_layer): the rewording
is shown with its behaviour's canonical prompt and 3 deterministic distractors from the same benchmark, and the
intent model must pick the right one. Verdicts go to an append-only store keyed on (model, benchmark, bid,
variant index, text), so the run is resumable and a different intent model never reuses them.

  python3 tools/regate_intent.py --dry-run
  python3 tools/regate_intent.py --model gemma4:e4b --stop-at 07:45
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recyclebench.backends.intent_layer import DiscriminativeIntentClassifier, llm_classify_fn  # noqa: E402
from recyclebench.datasets import load_jailbreakbench                                         # noqa: E402
from recyclebench.manifest import Manifest                                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SMOKE = "results/runs/20260805_124501_llama3-8b_to_llama31-8b_RECYCLE_qwen3b_SMOKE/manifest_harmbench.jsonl"
L3 = "results/runs/20260805_025658_llama3-8b_to_llama31-8b_RECYCLE"
MANIFESTS = [  # (benchmark, csv, manifest) -- the four behind the pooled headline, in priority order
    ("sorrybench", "data/sorrybench.csv", "results/runs/lineage_sorrybench_20261020/manifest.jsonl"),
    ("harmbench", "data/harmbench.csv", SMOKE),
    ("advbench", "data/advbench.csv", f"{L3}/manifest_advbench.jsonl"),
    ("strongreject", "data/strongreject.csv", f"{L3}/manifest_strongreject.jsonl"),
]


def intent_key(model, bench, bid, index, text):
    return hashlib.sha256(f"{model}\x1f{bench}\x1f{bid}\x1f{index}\x1f{text}".encode()).hexdigest()[:32]


class IntentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.mem = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    self.mem[d["k"]] = d
                except (json.JSONDecodeError, KeyError):
                    pass
        self.fh = None

    def get(self, k):
        return self.mem.get(k)

    def put(self, rec):
        if self.fh is None:
            self.fh = self.path.open("a")
        self.mem[rec["k"]] = rec
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def admitted(model_id):
    """-> [(key, bench, behaviour, variant, classifier)] in priority order."""
    out = []
    for bench, csv_p, man_p in MANIFESTS:
        bs = load_jailbreakbench(str(ROOT / csv_p))
        man = Manifest.load(ROOT / man_p)
        for b in bs:
            for v in man.R(b.bid):
                out.append((intent_key(model_id, bench, b.bid, v.index, v.text), bench, b, v, bs))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma4:e4b", help="Ollama model id of the intent checker (non-Llama, non-Qwen)")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--out", default=None, help="default cache/intent_<model>.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stop-at", default=None, metavar="HH:MM")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out or ROOT / f"cache/intent_{args.model.replace(':', '_').replace('/', '_')}.jsonl")
    store = IntentStore(out)
    items = admitted(args.model)
    todo = [it for it in items if store.get(it[0]) is None]
    print(f"intent model {args.model} · {len(items)} admitted rewordings · {len(items) - len(todo)} done · "
          f"{len(todo)} to check -> {out}")
    if args.dry_run or not todo:
        return 0
    deadline = None
    if args.stop_at:
        hh, mm = (int(x) for x in args.stop_at.split(":"))
        deadline = _dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= _dt.datetime.now():
            deadline += _dt.timedelta(days=1)
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    m = ProviderModel("gate:intent", ProviderConfig(base_url=args.base_url, model_id=args.model,
                                                    temperature=0.0, max_tokens=4,
                                                    extra_body={"reasoning_effort": "none"}))
    fn = llm_classify_fn(lambda prompt: m.respond(prompt).text)
    clf_by_bench = {}
    t0, done, kept = time.time(), 0, 0
    try:
        for k, bench, b, v, bs in todo[:args.limit] if args.limit else todo:
            if deadline and _dt.datetime.now() >= deadline:
                print(f"reached --stop-at after {done}; re-run to resume")
                break
            clf = clf_by_bench.setdefault(bench, DiscriminativeIntentClassifier(bs, fn, n_distract=3,
                                                                                family=f"llm:{args.model}"))
            iv = clf.evaluate(b, v)
            store.put(dict(k=k, model=args.model, bench=bench, bid=b.bid, index=v.index, admitted=int(iv.admitted),
                           note=iv.note))
            done += 1
            kept += int(iv.admitted)
            if done % 50 == 0:
                r = (time.time() - t0) / done
                print(f"  {done}/{len(todo)}  kept {kept}  {r:.2f}s/item  eta {r * (len(todo) - done) / 60:.0f} min", flush=True)
    except KeyboardInterrupt:
        print(f"interrupted after {done}; re-run to resume")
    finally:
        store.close()
    print(f"done: {done} checked this run, {kept} kept ({kept / max(done, 1):.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
