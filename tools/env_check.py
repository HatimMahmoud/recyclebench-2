#!/usr/bin/env python3
"""tools/env_check.py -- is this environment the one the registered figures came from?

Compares the installed numeric stack against `requirements-lock.txt` and reports the differences,
plus the BLAS thread settings and the platform. Run it in a cluster job before a long run: a
different numpy or statsmodels is a different instrument, and finding that out afterwards costs the
run.

    python3 tools/env_check.py              # report
    python3 tools/env_check.py --strict     # exit 1 if anything drifted
    python3 tools/env_check.py --json       # machine-readable, for a run's provenance

Why it exists: requirements.txt floor-pins every dependency, so the container and the laptop never
had the same stack. `test_study_is_byte_reproducible_under_same_seed` passes locally and reportedly
fails in the image (docs/pegasus/LESSONS.md §6.5); until the stacks match, that difference cannot be
attributed to threading rather than to a library version, so this check is a precondition for any
cross-environment reproducibility claim.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-lock.txt"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")


def read_lock(path: Path = LOCK) -> dict:
    if not path.exists():
        raise SystemExit(f"no lock file at {path}")
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9._-]+)==([\w.+-]+)", line)
        if not m:
            raise SystemExit(f"{path}: cannot parse {line!r} (expected name==version)")
        out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def installed(name: str):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def report(lock: dict) -> dict:
    pkgs = {}
    for name, want in sorted(lock.items()):
        got = installed(name)
        pkgs[name] = {"locked": want, "installed": got,
                      "status": "missing" if got is None else
                                ("ok" if got == want else "drifted")}
    threads = {v: os.environ.get(v) for v in THREAD_VARS}
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "lock": str(LOCK.relative_to(ROOT)),
            "packages": pkgs, "threads": threads,
            "drifted": sorted(k for k, v in pkgs.items() if v["status"] != "ok"),
            "threads_unpinned": sorted(k for k, v in threads.items() if v not in ("1",))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="exit 1 if any package drifted")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--lock", default=str(LOCK))
    args = ap.parse_args(argv)

    r = report(read_lock(Path(args.lock)))
    if args.json:
        print(json.dumps(r, indent=1))
        return 1 if (args.strict and r["drifted"]) else 0

    print(f"python   : {r['python']}  ({r['machine']}, {r['platform']})")
    print(f"lock     : {r['lock']}")
    for name, p in r["packages"].items():
        mark = {"ok": " ", "drifted": "!", "missing": "?"}[p["status"]]
        got = p["installed"] or "NOT INSTALLED"
        print(f"  {mark} {name:14} locked {p['locked']:10} installed {got}")
    print("threads  : " + ", ".join(f"{k}={v or 'unset'}" for k, v in r["threads"].items()))
    if r["threads_unpinned"]:
        print("           unpinned BLAS threads make jobs slower and are the leading suspect for "
              "the\n           container-only reproducibility failure; pegasus/env.sh pins them.")
    if r["drifted"]:
        print(f"\nDRIFT: {', '.join(r['drifted'])}")
        print("This is not the stack the registered figures came from. Install with\n"
              "  pip install -r requirements.txt -c requirements-lock.txt\n"
              "or, if the new versions are intended, re-run the suite, confirm the registered "
              "figures reproduce, and update the lock deliberately.")
        return 1 if args.strict else 0
    print("\nenvironment matches the lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
