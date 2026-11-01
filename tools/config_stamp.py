#!/usr/bin/env python3
"""tools/config_stamp.py -- the full-configuration stamp the registration hash leaves out (prereg Amendment 2 §B5.5).

`prereg.decision_fields` hashes seed, lineage, classes, benchmarks, gate, budget and decision rule. It does not cover which
model a config key points at, its decoding settings, the judge, paraphraser and intent models, the Ollama runtime, or the
weights. The response cache is keyed on (config key, prompt, seed), so any of those could change without a cache miss.

`stamp` records, for every config with a `real:` section:
  * sha256 of the file and of each model block (target, judge, paraphraser, intent), with the model id and decoding;
  * the registration hash, for reference;
  * for every Ollama model referenced: the weights blob, the manifest date and the Ollama version.
and checks the one silent failure the cache design allows: the SAME config key defined with DIFFERENT model blocks in two
configs (they would share cached replies). `verify` compares the current state with a stamp and lists every drift.

  python3 tools/config_stamp.py stamp                    # -> results/config_stamp_<stamp>.json
  python3 tools/config_stamp.py verify results/config_stamp_<stamp>.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MANIFESTS = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"


def sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else json.dumps(b, sort_keys=True).encode()).hexdigest()


def blocks(raw: dict) -> dict:
    """role/key -> model block, for every block that names a model"""
    out = {f"target:{k}": v for k, v in (raw.get("real") or {}).items()}
    out.update({f"judge:{k}": v for k, v in (raw.get("judges") or {}).items()})
    for role in ("paraphraser", "intent"):
        if isinstance(raw.get(role), dict) and raw[role].get("model_id"):
            out[role] = raw[role]
    return out


def ollama_facts(model_ids) -> dict:
    ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    facts = {"ollama_version": ver, "models": {}}
    for mid in sorted(model_ids):
        name, _, tag = mid.partition(":")
        man = MANIFESTS / name / (tag or "latest")
        mf = subprocess.run(["ollama", "show", mid, "--modelfile"], capture_output=True, text=True).stdout
        blob = next((l.split("sha256-")[-1] for l in mf.splitlines() if l.startswith("FROM") and "sha256-" in l), None)
        facts["models"][mid] = dict(weights_blob=blob[:16] if blob else None,
                                    manifest_date=_dt.datetime.fromtimestamp(man.stat().st_mtime).isoformat(timespec="minutes")
                                    if man.exists() else None)
    return facts


def stamp() -> dict:
    from recyclebench.config import Config
    from recyclebench.prereg import decision_fields
    configs, mids, by_key = {}, set(), defaultdict(dict)
    for p in sorted((ROOT / "configs").glob("*.yaml")):
        raw = yaml.safe_load(p.read_text()) or {}
        if not raw.get("real"):
            continue
        bl = blocks(raw)
        try:
            reg = sha(decision_fields(Config.load(p)))[:32]
        except Exception as e:                                        # noqa: BLE001
            reg = f"n/a ({type(e).__name__})"
        configs[str(p.relative_to(ROOT))] = dict(
            file_sha256=sha(p.read_bytes()), decision_fields_sha=reg,
            blocks={k: dict(model_id=v.get("model_id"), sha256=sha(v)[:16],
                            decoding={x: v.get(x) for x in ("temperature", "top_p", "max_tokens", "system_prompt", "extra_body")
                                      if v.get(x) is not None}) for k, v in bl.items()})
        for k, v in bl.items():
            if v.get("model_id"):
                mids.add(v["model_id"])
            if k.startswith("target:"):
                by_key[k.split(":", 1)[1]][str(p.relative_to(ROOT))] = sha(v)[:16]
    collisions = {k: v for k, v in by_key.items() if len(set(v.values())) > 1}
    return dict(created=_dt.datetime.now().isoformat(timespec="seconds"), configs=configs,
                cache_key_collisions=collisions, ollama=ollama_facts(mids),
                note="cache keys are (config key, prompt, seed): a collision means two configs would share cached replies "
                     "for different model blocks")


def verify(old: dict, new: dict) -> list:
    drift = []
    for c, o in old["configs"].items():
        n = new["configs"].get(c)
        if n is None:
            drift.append(f"{c}: missing now")
            continue
        if n["file_sha256"] != o["file_sha256"]:
            changed = [b for b in o["blocks"] if n["blocks"].get(b, {}).get("sha256") != o["blocks"][b]["sha256"]]
            drift.append(f"{c}: file changed" + (f"; model blocks changed: {', '.join(changed)}" if changed else ""))
    if new["ollama"]["ollama_version"] != old["ollama"]["ollama_version"]:
        drift.append(f"Ollama {old['ollama']['ollama_version']} -> {new['ollama']['ollama_version']}")
    for mid, f in old["ollama"]["models"].items():
        g = new["ollama"]["models"].get(mid)
        if not g or g["weights_blob"] != f["weights_blob"]:
            drift.append(f"{mid}: weights {f['weights_blob']} -> {(g or {}).get('weights_blob')}")
    for k in set(new["cache_key_collisions"]) - set(old["cache_key_collisions"]):
        drift.append(f"new cache-key collision: {k}")
    return drift


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stamp")
    s.add_argument("--out", default=None)
    v = sub.add_parser("verify")
    v.add_argument("stamp_file")
    args = ap.parse_args(argv)
    if args.cmd == "stamp":
        st = stamp()
        out = Path(args.out) if args.out else ROOT / "results" / f"config_stamp_{_dt.datetime.now():%Y%m%d_%H%M}.json"
        if out.exists():
            raise SystemExit(f"{out} exists -- refusing to overwrite")
        out.write_text(json.dumps(st, indent=1))
        print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: {len(st['configs'])} configs, "
              f"{len(st['ollama']['models'])} Ollama models, {st['ollama']['ollama_version']}")
        for k, v in st["cache_key_collisions"].items():
            print(f"  CACHE-KEY COLLISION {k}: " + "; ".join(f"{c} {h}" for c, h in v.items()))
        return 0
    drift = verify(json.loads(Path(args.stamp_file).read_text()), stamp())
    print("no drift" if not drift else "DRIFT:\n  " + "\n  ".join(drift))
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
