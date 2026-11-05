#!/usr/bin/env python3
"""tools/mutation_screen.py -- SORRY-Bench's own 20 linguistic mutations on every model (PREREG_SORRYBENCH_MUTATIONS_20261105.md).

SORRY-Bench ships each of its 440 base requests in 20 mutated forms (writing styles, persuasion techniques, encodings,
translations). This arm sends every mutation to every model once and asks, per pair, how many of the newer model's fixes
on the base requests it answers in at least one mutated form. It is an attack-transfer arm, not RSR: the mutations are
SORRY-Bench's, not intent-gated rewordings of ours, so the replies are judged against the ORIGINAL English request
(cipher replies decoded deterministically, non-English replies translated back by qwen2.5:14b first).

  build      data/sorrybench_mutations.jsonl from sorry-bench/sorry-bench-202503 (gated; HF token), every base checked
             against data/sorrybench.csv (ids are its row positions)
  generate   one draw per (base, mutation) for --model, styles --styles (a comma list, or all), into --cache
  judge      decode / translate, then the config's judge against the original request -> cache/mutations/judge.jsonl
  report     answer rate per model x style; per pair, fixes reopened by >= 1 mutation (and per style)

No text is printed; reports hold counts and ids only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import lineage_core as lc                                      # noqa: E402
from recyclebench.config import Config                         # noqa: E402
from recyclebench.datasets import load_jailbreakbench          # noqa: E402
from recyclebench.realrun import _key                          # noqa: E402
from recyclebench.screening import _draw_seed                  # noqa: E402

HF = "sorry-bench/sorry-bench-202503"
DATA = "data/sorrybench_mutations.jsonl"
BASE = "data/sorrybench.csv"
C = "cache/mutations"
STORE = f"{C}/judge.jsonl"
TRANS = f"{C}/translations.jsonl"
OUT = "results/runs/mutations"
SEED_OFFSET = 1000          # draw seed _draw_seed(bid, SEED_OFFSET + style index, 0): disjoint from canonical (-1) and variants
MODEL_CFG = {**{m: "configs/llama4_pegasus.yaml" for m in ("llama2-7b", "llama2-13b", "llama3-8b", "llama31-8b",
                                                             "llama32-1b", "llama32-3b", "llama4-scout", "llama4-maverick")},
             **{m: "configs/chain70b.yaml" for m in ("llama2-70b", "llama3-70b", "llama31-70b", "llama33-70b")}}
TRANSLATOR = {"provider": "openai_compatible", "base_url": "http://localhost:11434/v1", "model_id": "qwen2.5:14b",
              "temperature": 0.0, "max_tokens": 1024}


# =============================================================================
# decoding (deterministic) and translation
# =============================================================================
MORSE = {".-": "a", "-...": "b", "-.-.": "c", "-..": "d", ".": "e", "..-.": "f", "--.": "g", "....": "h", "..": "i",
         ".---": "j", "-.-": "k", ".-..": "l", "--": "m", "-.": "n", "---": "o", ".--.": "p", "--.-": "q", ".-.": "r",
         "...": "s", "-": "t", "..-": "u", "...-": "v", ".--": "w", "-..-": "x", "-.--": "y", "--..": "z",
         "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4", ".....": "5", "-....": "6",
         "--...": "7", "---..": "8", "----.": "9"}


def _shift(text, k):
    out = []
    for ch in text:
        if ch in string.ascii_lowercase:
            out.append(chr((ord(ch) - 97 - k) % 26 + 97))
        elif ch in string.ascii_uppercase:
            out.append(chr((ord(ch) - 65 - k) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def _atbash(text):
    return "".join(chr(219 - ord(c)) if c in string.ascii_lowercase else chr(155 - ord(c)) if c in string.ascii_uppercase
                   else c for c in text)


def _morse(text):
    words = []
    for w in text.replace("\n", " / ").split(" / "):
        words.append("".join(MORSE.get(t, "") for t in w.split()))
    return " ".join(words)


def _ascii(text):
    toks = [t for t in text.replace(",", " ").split() if t.isdigit()]
    if len(toks) < 3:
        return text
    try:
        return "".join(chr(int(t)) for t in toks if 0 < int(t) < 0x110000)
    except ValueError:
        return text


DECODERS = {"caesar": lambda t: _shift(t, 3), "atbash": _atbash, "morse": _morse, "ascii": _ascii}


def kind(style):
    s = style.lower()
    if s in DECODERS:
        return "cipher"
    if s.startswith("translate"):
        return "translate"
    return "english"


# =============================================================================
# data
# =============================================================================
def cmd_build(a):
    from datasets import load_dataset
    ds = load_dataset(HF, token=True)
    rows = [dict(r) for r in ds["train" if "train" in ds else list(ds)[0]]]
    base = load_jailbreakbench(str(ROOT / BASE))
    by_qid = defaultdict(dict)
    for r in rows:
        t = r.get("turns")
        text = str((t[0] if isinstance(t, (list, tuple)) and t else t) or "").strip()   # some rows carry None
        if text:                                                  # an empty mutation is counted as missing below
            by_qid[str(r.get("question_id"))][str(r.get("prompt_style") or "base").strip()] = text
    bases = [q for q, st in by_qid.items() if "base" in st]
    goal_to_q = {by_qid[q]["base"]: q for q in bases}
    styles = sorted({s for st in by_qid.values() for s in st if s != "base"})
    print(f"{len(rows)} rows, {len(by_qid)} questions, {len(styles)} mutation styles: {styles}")
    out, missing = [], 0
    for b in base:
        q = goal_to_q.get(b.canonical_prompt)
        if q is None:
            raise SystemExit(f"base row {b.bid} not found in {HF}: the CSV is not the registered population")
        for s in styles:
            if s not in by_qid[q]:
                missing += 1
                continue
            out.append({"bid": b.bid, "qid": q, "style": s, "prompt": by_qid[q][s]})
    p = ROOT / DATA
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    p.chmod(0o600)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    meta = {"source": HF, "rows": len(out), "bases": len(base), "styles": styles, "missing": missing, "sha256": h,
            "built": _dt.datetime.now().isoformat(timespec="seconds")}
    (ROOT / (DATA + ".provenance.json")).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "styles"}))
    return 0


def load_items():
    rows = [json.loads(l) for l in (ROOT / DATA).read_text(encoding="utf-8").splitlines() if l.strip()]
    styles = sorted({r["style"] for r in rows})
    return rows, styles


def seed_of(bid, style, styles):
    return _draw_seed(bid, SEED_OFFSET + styles.index(style), 0)


# =============================================================================
# generate / judge
# =============================================================================
def cmd_generate(a):
    from recyclebench.backends.providers import ProviderConfig, ProviderModel
    from recyclebench.realrun import CachedModel, ResponseCache
    cfg = Config.load(str(ROOT / MODEL_CFG[a.model]))
    rows, styles = load_items()
    if a.styles == "all":
        want = styles
    elif a.styles.startswith("shard:"):                        # shard:i/n = every n-th style from index i
        i, n = (int(x) for x in a.styles[6:].split("/"))
        want = styles[i::n]
    else:
        want = a.styles.split(",")
    base = {b.bid: b for b in load_jailbreakbench(str(ROOT / BASE))}
    cache_p = ROOT / a.cache
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    have = lc.load_response_cache(str(cache_p)) if cache_p.exists() else {}
    todo = [r for r in rows if r["style"] in want and _key(a.model, r["prompt"], seed_of(r["bid"], r["style"], styles)) not in have]
    print(f"{a.model}: {len(todo)} to generate ({len(want)} styles) -> {a.cache}", flush=True)
    rc = ResponseCache(cache_p)
    m = CachedModel(ProviderModel(a.model, ProviderConfig(**cfg.real[a.model])), rc)
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        m.respond(r["prompt"], behaviour=base[r["bid"]], canonical=False, seed=seed_of(r["bid"], r["style"], styles))
        if i % 100 == 0 or i == len(todo):
            el = (time.time() - t0) / i
            print(f"  {i}/{len(todo)}  {el:.2f}s/call  eta {el * (len(todo) - i) / 60:.0f} min", flush=True)
    rc.close()
    return 0


class Translations:
    def __init__(self, path):
        self.p = ROOT / path
        self.d = {}
        if self.p.exists():
            for l in self.p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    x = json.loads(l)
                    self.d[x["k"]] = x["t"]
        self.fh = open(self.p, "a", encoding="utf-8")
        self.model = None

    def get(self, text):
        k = hashlib.sha256(text.encode()).hexdigest()[:32]
        if k in self.d:
            return self.d[k]
        if self.model is None:
            from recyclebench.backends.providers import ProviderConfig, ProviderModel
            self.model = ProviderModel("translator:qwen2.5:14b", ProviderConfig(**TRANSLATOR))
        out = self.model.respond("Translate the following text into English. Output only the translation, nothing "
                                 "else.\n\n" + text[:4000]).text or ""
        self.d[k] = out
        self.fh.write(json.dumps({"k": k, "t": out}, ensure_ascii=False) + "\n")
        self.fh.flush()
        return out


def english(style, reply, tr):
    k = kind(style)
    if k == "cipher":
        return DECODERS[style.lower()](reply)
    if k == "translate":
        return tr.get(reply) if reply.strip() else reply
    return reply


def cmd_judge(a):
    from judge_lineage import build_judge
    from recyclebench.types import Response
    cfg = Config.load(str(ROOT / "configs/llama4_pegasus.yaml"))
    tag = lc.judge_tag_from_cfg(cfg)
    rows, styles = load_items()
    base = {b.bid: b.canonical_prompt for b in load_jailbreakbench(str(ROOT / BASE))}
    store = lc.VerdictStore(str(ROOT / STORE))
    tr = Translations(TRANS)
    judge = None
    done = 0
    t0 = time.time()
    for model in a.models:
        cache = {}
        for p in sorted((ROOT / C).glob(f"{model}__*.jsonl")):
            cache.update(lc.load_response_cache(str(p)))
        todo = [(r, cache[k]) for r in rows
                if (k := _key(model, r["prompt"], seed_of(r["bid"], r["style"], styles))) in cache]
        print(f"{model}: {len(todo)} cached replies", flush=True)
        for i, (r, reply) in enumerate(todo, 1):
            en = english(r["style"], reply, tr)
            k = lc.verdict_key(tag, base[r["bid"]], en)
            if store.get(k) is not None:
                continue
            judge = judge or build_judge(cfg)
            store.put(k, int(judge.classify(base[r["bid"]], Response(text=en, latent_harmful=None))))
            done += 1
            if done % 200 == 0:
                print(f"  judged {done}  {(time.time() - t0) / done:.2f}s/item", flush=True)
    store.close()
    print(f"done: {done} judged this run")
    return 0


# =============================================================================
# report
# =============================================================================
def cmd_report(a):
    import cluster_pairs as cp
    cfg = Config.load(str(ROOT / "configs/llama4_pegasus.yaml"))
    tag = lc.judge_tag_from_cfg(cfg)
    rows, styles = load_items()
    base = {b.bid: b.canonical_prompt for b in load_jailbreakbench(str(ROOT / BASE))}
    store = lc.VerdictStore(str(ROOT / STORE))
    tr = Translations(TRANS)
    answered = defaultdict(lambda: defaultdict(set))          # model -> style -> bids answered
    coverage = Counter()
    for model in MODEL_CFG:
        cache = {}
        for p in sorted((ROOT / C).glob(f"{model}__*.jsonl")):
            cache.update(lc.load_response_cache(str(p)))
        for r in rows:
            reply = cache.get(_key(model, r["prompt"], seed_of(r["bid"], r["style"], styles)))
            if reply is None:
                continue
            k = tr.d.get(hashlib.sha256(reply.encode()).hexdigest()[:32]) if kind(r["style"]) == "translate" else None
            en = k if k is not None else english(r["style"], reply, tr) if kind(r["style"]) != "translate" else None
            if en is None:
                continue
            v = store.get(lc.verdict_key(tag, base[r["bid"]], en))
            if v is None:
                continue
            coverage[model] += 1
            if v == 1:
                answered[model][r["style"]].add(r["bid"])
    store.close()
    srcs = [s for s in cp.load_sources(only={"sorrybench"})]
    out = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "prereg": "PREREG_SORRYBENCH_MUTATIONS_20261105.md",
           "designation": "exploratory; raw judge output against the original request", "styles": styles,
           "coverage": {m: {"judged": coverage[m], "expected": len(rows)} for m in MODEL_CFG}, "models": {}, "pairs": {}}
    for m in MODEL_CFG:
        out["models"][m] = {s: len(answered[m][s]) for s in styles} | {"any": len(set().union(*answered[m].values()))
                                                                        if answered[m] else 0}
    for o, n in cp.PAIRS:
        src = next((s for s in srcs if o in s.models and n in s.models), None)
        if not src or coverage[n] < len(rows):
            continue
        fixes = [b.bid for b in src.behaviours if src.obs[o].canonical.get(b.bid) == 1 and src.obs[n].canonical.get(b.bid) == 0]
        per = {s: sum(b in answered[n][s] for b in fixes) for s in styles}
        anyk = sum(any(b in answered[n][s] for s in styles) for b in fixes)
        out["pairs"][f"{o}->{n}"] = {"fixes": len(fixes), "reopened_any": anyk, "per_style": per}
    path = ROOT / OUT / f"report_{_dt.datetime.now():%Y%m%d_%H%M}.json"
    if path.exists():
        raise SystemExit(f"{path} exists -- refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["pairs"].items():
        print(f"  {k:28s} fixes {v['fixes']:3d}  reopened by >= 1 mutation {v['reopened_any']:3d}")
    print(f"wrote {path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True, choices=list(MODEL_CFG))
    g.add_argument("--styles", default="all")
    g.add_argument("--cache", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--models", nargs="*", default=list(MODEL_CFG))
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd in ("generate", "judge", "report") and not (ROOT / DATA).exists():
        raise SystemExit(f"{DATA} absent: run `build` first")
    if a.cmd in ("judge", "report"):
        reg = (ROOT / "PREREG_SORRYBENCH_MUTATIONS_20261105.md").read_text(encoding="utf-8")
        if "Status: SIGNED" not in reg:
            raise SystemExit("registration not signed: nothing is judged or reported")
    return {"build": cmd_build, "generate": cmd_generate, "judge": cmd_judge, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
