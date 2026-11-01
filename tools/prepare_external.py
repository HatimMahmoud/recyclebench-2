#!/usr/bin/env python3
"""tools/prepare_external.py -- bring-your-own-data adapter: any JSON/JSONL/CSV corpus -> a screened-ready CSV.

Every arm of this study reads one shape: a CSV with a ``Goal`` column (``recyclebench.datasets.
load_jailbreakbench``). Getting a new corpus into the study should therefore be *one mapping file*,
not a new converter per source. This reads a recipe (YAML) that names the source, the field holding
the request, and what becomes ``Category``, and emits that CSV plus a provenance sidecar.

What it guarantees
------------------
* **Deterministic.** Same recipe + same source bytes -> byte-identical CSV. Records keep recipe
  file order, then in-file order; nothing is shuffled or sampled here.
* **Pinned.** A github source is fetched at an immutable commit, into
  ``data/external/<name>_<commit7>/``, and every file's sha256 is recorded. A later run verifies
  those hashes and refuses to proceed on a mismatch.
* **Provenance.** ``<out>.provenance.json`` records the recipe hash, source commit, licence, every
  file hash, the field mapping, row counts at each filter step, and the overlap report.
* **Overlap-aware.** Reports (``--drop-overlap``: removes) behaviours that duplicate an
  already-screened benchmark, so a "new" corpus cannot silently re-screen seen items. Same
  normalisation and difflib cutoff as ``tools/prepare_heldout.py``.
* **Quiet.** Prints counts, hashes and field names only -- never prompt text, so a log or a
  terminal scrollback carries no harmful content.

Usage
-----
  python3 tools/prepare_external.py --recipe datasets/sciencesafety.yaml
  python3 tools/prepare_external.py --recipe datasets/sciencesafety.yaml --drop-overlap
  python3 tools/prepare_external.py --recipe R.yaml --no-fetch     # offline; sources must be present
  python3 tools/prepare_external.py --recipe R.yaml --dry-run      # report only, write nothing

Recipe
------
See ``docs/BYOD.md`` and ``datasets/sciencesafety.yaml``. The whole schema:

    name: sciencesafety                  # used for the external download dir
    license: MIT
    source:
      kind: github | local
      repo: owner/name                   # github
      commit: <full sha>                 # github, immutable
      root: data/ScienceSafetyData       # optional path prefix for `paths`
      paths: [Dataset/bio.json, ...]     # github: repo-relative; local: anything readable
    format: json | jsonl | csv           # json = a list of objects per file
    mapping:
      goal: Prompt                       # REQUIRED: the field holding the request
      category: "@stem"                  # a field name, or @stem/@parent/@literal:<x>
      extra: {Task: Task}                # extra CSV column <- source field
    out: data/sciencesafety.csv
    dedup:
      against: [sorrybench, harmbench]   # benchmark stems in data/
      cutoff: 0.85                       # difflib ratio; exact matches always reported
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def norm(s: str) -> str:
    """Normalisation used for overlap detection (identical to tools/prepare_heldout.py)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def load_records(path: Path, fmt: str) -> list[dict]:
    """Read one source file into a list of dicts, preserving file order."""
    if fmt == "json":
        d = json.loads(path.read_text())
        if isinstance(d, dict):                       # a mapping of id -> record
            d = [dict(v, _key=k) if isinstance(v, dict) else {"_key": k, "value": v}
                 for k, v in d.items()]
        if not isinstance(d, list):
            raise SystemExit(f"{path}: expected a JSON list or object, got {type(d).__name__}")
        return [r for r in d if isinstance(r, dict)]
    if fmt == "jsonl":
        out = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                if isinstance(r, dict):
                    out.append(r)
        return out
    if fmt == "csv":
        with path.open(newline="") as fh:
            return list(csv.DictReader(fh))
    raise SystemExit(f"unknown format: {fmt!r} (json | jsonl | csv)")


def category_of(spec: str, rec: dict, path: Path) -> str:
    """Resolve the category spec: a field name, or @stem / @parent / @literal:<x>."""
    if not spec:
        return "unknown"
    if spec == "@stem":
        return path.stem
    if spec == "@parent":
        return path.parent.name
    if spec.startswith("@literal:"):
        return spec.split(":", 1)[1]
    return str(rec.get(spec, "") or "").strip() or "unknown"


# ---------------------------------------------------------------------------
# source resolution
# ---------------------------------------------------------------------------
def resolve_sources(recipe: dict, name: str, fetch: bool) -> tuple[list[Path], dict]:
    """Return the local source files in recipe order, plus the provenance block for them."""
    src = recipe["source"]
    kind = src.get("kind", "github")
    root = src.get("root") or ""
    paths = list(src["paths"])

    if kind == "local":
        # Path-join, so an absolute `root` (or an absolute entry in `paths`) stays absolute;
        # only a genuinely relative result is resolved against the repo root.
        files = []
        for p in paths:
            q = (Path(root) / p) if root else Path(p)
            files.append(q if q.is_absolute() else ROOT / q)
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise SystemExit("missing local source files:\n  " + "\n  ".join(missing))
        return files, {"kind": "local", "root": str(root), "paths": paths}

    if kind != "github":
        raise SystemExit(f"unknown source kind: {kind!r} (github | local)")

    # a github source is always repo-relative, so plain string joining is correct here
    rel = [f"{root.strip('/')}/{p}".strip("/") if root else p for p in paths]
    repo, commit = src["repo"], src["commit"]
    if len(commit) < 40:
        raise SystemExit("source.commit must be a full 40-char sha (an immutable pin), "
                         f"got {commit!r}")
    base = ROOT / "data" / "external" / f"{name}_{commit[:7]}"
    files = []
    fetched = []
    for r in rel:
        dest = base / r
        if not dest.exists():
            if not fetch:
                raise SystemExit(f"{dest} is absent and --no-fetch was given")
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = RAW.format(repo=repo, commit=commit, path=r)
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    dest.write_bytes(resp.read())
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                raise SystemExit(f"fetch failed for {r}: {e}")
            fetched.append(r)
        files.append(dest)
    return files, {"kind": "github", "repo": repo, "commit": commit,
                   "paths": rel, "local_dir": str(base.relative_to(ROOT)),
                   "fetched_now": fetched}


# ---------------------------------------------------------------------------
# overlap against already-screened benchmarks
# ---------------------------------------------------------------------------
def overlap_report(goals: list[str], against: list[str], cutoff: float,
                   data_dir: Path) -> dict:
    """Exact and near duplicates of each goal in any named benchmark CSV."""
    seen: dict[str, str] = {}                       # normalised goal -> benchmark stem
    absent = []
    for stem in against:
        p = data_dir / f"{stem}.csv"
        if not p.exists():
            absent.append(stem)
            continue
        with p.open(newline="") as fh:
            for row in csv.DictReader(fh):
                g = (row.get("Goal") or "").strip()
                if g:
                    seen.setdefault(norm(g), stem)
    keys = list(seen)
    exact, near = {}, {}
    for i, g in enumerate(goals):
        n = norm(g)
        if n in seen:
            exact[i] = seen[n]
            continue
        if cutoff and cutoff < 1.0:
            m = difflib.get_close_matches(n, keys, n=1, cutoff=cutoff)
            if m:
                near[i] = seen[m[0]]
    return {"benchmarks": against, "absent": absent, "cutoff": cutoff,
            "n_reference_behaviours": len(seen),
            "exact": {str(k): v for k, v in exact.items()},
            "near": {str(k): v for k, v in near.items()},
            "n_exact": len(exact), "n_near": len(near)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build(recipe_path: Path, *, fetch: bool = True, drop_overlap: bool = False,
          out: Path | None = None, dry_run: bool = False) -> dict:
    recipe_text = recipe_path.read_text()
    recipe = yaml.safe_load(recipe_text)
    name = recipe["name"]
    fmt = recipe.get("format", "json")
    mapping = recipe["mapping"]
    goal_field = mapping["goal"]
    cat_spec = mapping.get("category", "")
    extra = dict(mapping.get("extra") or {})
    dedup = dict(recipe.get("dedup") or {})
    dest = Path(out) if out else (ROOT / recipe["out"])
    if not dest.is_absolute():
        dest = ROOT / dest

    files, src_prov = resolve_sources(recipe, name, fetch)
    file_hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                   sha256_file(p) for p in files}

    # --- map records -------------------------------------------------------
    rows, n_read, n_empty = [], 0, 0
    per_file = {}
    for p in files:
        recs = load_records(p, fmt)
        kept = 0
        for rec in recs:
            n_read += 1
            goal = str(rec.get(goal_field, "") or "").strip()
            if not goal:
                n_empty += 1
                continue
            row = {"Goal": goal, "Category": category_of(cat_spec, rec, p)}
            for col, field in extra.items():
                row[col] = str(rec.get(field, "") or "").strip()
            rows.append(row)
            kept += 1
        per_file[p.name] = {"records": len(recs), "kept": kept}

    if not rows:
        raise SystemExit(f"no usable rows: is mapping.goal ({goal_field!r}) the right field?")

    # --- drop exact duplicates within the corpus (keep first occurrence) ---
    inner_dupes, uniq, seen = [], [], set()
    for i, r in enumerate(rows):
        n = norm(r["Goal"])
        if n in seen:
            inner_dupes.append(i)
            continue
        seen.add(n)
        uniq.append(r)
    rows = uniq

    # --- overlap with already-screened benchmarks --------------------------
    ov = None
    if dedup.get("against"):
        ov = overlap_report([r["Goal"] for r in rows], list(dedup["against"]),
                            float(dedup.get("cutoff", 0.85)), ROOT / "data")
        if drop_overlap:
            bad = {int(k) for k in ov["exact"]} | {int(k) for k in ov["near"]}
            rows = [r for i, r in enumerate(rows) if i not in bad]

    cols = ["Goal", "Category"] + list(extra)
    prov = {
        "tool": "tools/prepare_external.py",
        "name": name,
        "recipe": str(recipe_path.relative_to(ROOT) if recipe_path.is_relative_to(ROOT)
                      else recipe_path),
        "recipe_sha256": sha256_text(recipe_text),
        "license": recipe.get("license", "unstated"),
        "source": src_prov,
        "file_sha256": file_hashes,
        "format": fmt,
        "mapping": {"goal": goal_field, "category": cat_spec, "extra": extra},
        "columns": cols,
        "counts": {"records_read": n_read, "dropped_empty_goal": n_empty,
                   "dropped_duplicate_within_corpus": len(inner_dupes),
                   "dropped_overlap": (len({int(k) for k in ov["exact"]}
                                           | {int(k) for k in ov["near"]})
                                       if (ov and drop_overlap) else 0),
                   "written": len(rows)},
        "per_file": per_file,
        "overlap": ov,
        "overlap_dropped": bool(drop_overlap),
        "out": str(dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest),
    }

    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        prov["out_sha256"] = sha256_file(dest)
        side = dest.with_suffix(dest.suffix + ".provenance.json")
        side.write_text(json.dumps(prov, indent=2))
        prov["provenance_file"] = str(side.relative_to(ROOT) if side.is_relative_to(ROOT)
                                      else side)
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, help="dataset recipe YAML")
    ap.add_argument("--out", help="override the recipe's `out:` path")
    ap.add_argument("--no-fetch", action="store_true",
                    help="never download; every source file must already be present")
    ap.add_argument("--drop-overlap", action="store_true",
                    help="remove behaviours duplicating an already-screened benchmark "
                         "(default: report only)")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args(argv)

    rp = Path(args.recipe)
    if not rp.is_absolute():
        rp = ROOT / rp
    if not rp.exists():
        raise SystemExit(f"recipe not found: {rp}")

    prov = build(rp, fetch=not args.no_fetch, drop_overlap=args.drop_overlap,
                 out=args.out, dry_run=args.dry_run)

    c = prov["counts"]
    print(f"recipe        : {prov['recipe']}  ({prov['recipe_sha256'][:16]})")
    s = prov["source"]
    if s["kind"] == "github":
        print(f"source        : github {s['repo']} @ {s['commit'][:7]}  ({prov['license']})")
        print(f"                -> {s['local_dir']}  ({len(prov['file_sha256'])} files)")
        if s["fetched_now"]:
            print(f"                fetched now: {len(s['fetched_now'])}")
    else:
        print(f"source        : local, {len(prov['file_sha256'])} files  ({prov['license']})")
    print(f"mapping       : Goal <- {prov['mapping']['goal']!r}   "
          f"Category <- {prov['mapping']['category']!r}"
          + (f"   extra: {list(prov['mapping']['extra'])}" if prov['mapping']['extra'] else ""))
    print(f"records read  : {c['records_read']}")
    print(f"  empty goal  : -{c['dropped_empty_goal']}")
    print(f"  dup in-set  : -{c['dropped_duplicate_within_corpus']}")
    if prov["overlap"]:
        o = prov["overlap"]
        verb = "dropped" if prov["overlap_dropped"] else "reported only"
        print(f"  overlap     : {o['n_exact']} exact + {o['n_near']} near (cutoff "
              f"{o['cutoff']}) vs {o['n_reference_behaviours']} screened behaviours "
              f"[{verb}]")
        if o["absent"]:
            print(f"                (no CSV for: {', '.join(o['absent'])})")
    print(f"written       : {c['written']} behaviours")
    if "out_sha256" in prov:
        print(f"out           : {prov['out']}  ({prov['out_sha256'][:16]})")
        print(f"provenance    : {prov['provenance_file']}")
    else:
        print("out           : (dry run, nothing written)")

    by_cat: dict[str, int] = {}
    for k, v in prov["per_file"].items():
        by_cat[k] = v["kept"]
    print("per source file: " + ", ".join(f"{k} {v}" for k, v in by_cat.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
