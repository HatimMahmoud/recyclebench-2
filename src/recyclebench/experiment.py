"""One configuration-driven experiment: dataset x science config x execution target.

The pipeline already had every scientific piece (screening, the judge, the gate, the frozen
manifest, the estimators) and every piece was already resumable. What it did not have was one
*description* of a run that both a laptop and a Slurm job can execute, so each corpus grew its own
bespoke shell queue -- `run_sciencesafety_20261101.sh` and its RPE follow-on -- which re-implemented
stage ordering, time budgets, resume markers and, because `pilot-pminus` wrote a fixed filename, a
backup/restore dance around shared scratch files.

An `ExperimentSpec` is that description. It composes three things that stay separable:

    dataset:    a BYOD recipe (tools/prepare_external.py) or an existing CSV
    science:    a registered config (configs/*.yaml) -- judge, gate, budgets, seed, lineage
    execution:  where it runs and with what resources; nothing scientific

Nothing here decides anything scientific. Thresholds, budgets, denominators, rubrics and seeds
remain the config's; this module only sequences existing steps, keeps their artifacts inside one run
directory, and refuses to resume a run whose inputs have changed underneath it.

Three verbs:

    validate    inputs resolve, config loads, required providers exist          (no writes)
    plan        the ordered stages, what each would do, what is already done    (no writes)
    run         execute the pending stages, resumably

`plan` and `validate` make no model calls, no downloads, no submissions and write nothing at all --
`Plan.writes` is empty by construction and a test asserts it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]

# The canonical stage order. `screen` must precede `subset`, and `subset` must precede `recycle`,
# because the recycling set is the JUDGE's P^- and not the offline-rule partition (docs/BYOD.md
# trap 1). `partition` is a control and deliberately runs after `recycle`, so a run that loses its
# time budget still has the primary estimand.
STAGES = ("prepare", "screen", "subset", "recycle", "partition", "audit")


class SpecError(SystemExit):
    """A spec that cannot run. Raised with an actionable message, never a traceback."""


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def code_revision() -> str:
    """git HEAD; else RECYCLEBENCH_CODE_REVISION; else 'unknown'. Recorded so a resume can refuse a change.

    The cluster image has no git, so inside a job `git rev-parse` fails: pegasus/run.sbatch and
    experiment.sbatch read HEAD on the host and pass it in (the Scout recycle runs of 2026-11-02
    recorded 'unknown' before this fallback existed).
    """
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("RECYCLEBENCH_CODE_REVISION") or "unknown"


@dataclass
class Execution:
    """Where the experiment runs. Nothing here may affect a number."""
    target: str = "local"                  # local | cluster
    run_root: str = "results/runs"
    cache_dir: str = "cache"
    shard: Optional[str] = None            # a task id: gives this task its own cache shard
    partition: Optional[str] = None        # cluster only, passed through to the scheduler
    gpus: Optional[int] = None
    mem: Optional[str] = None
    time_limit: Optional[str] = None
    stop_at: Optional[str] = None          # wall-clock HH:MM after which no new stage starts

    @staticmethod
    def from_dict(d: dict) -> "Execution":
        unknown = set(d) - {f for f in Execution.__dataclass_fields__}
        if unknown:
            raise SpecError(f"unknown execution keys: {sorted(unknown)}")
        e = Execution(**d)
        if e.target not in ("local", "cluster"):
            raise SpecError(f"execution.target must be local or cluster, got {e.target!r}")
        return e


@dataclass
class ExperimentSpec:
    name: str
    science_config: str
    old: str
    new: str
    behaviours: Optional[str] = None            # an existing CSV ...
    recipe: Optional[str] = None                # ... or a BYOD recipe that produces one
    execution: Execution = field(default_factory=Execution)
    stages: tuple = STAGES
    audit_bundles: int = 60
    min_p_minus_for_recycle: int = 5
    path: Optional[str] = None                  # the spec file, for provenance

    # ---------------- construction ----------------
    @staticmethod
    def load(path) -> "ExperimentSpec":
        import yaml
        p = Path(path)
        if not p.exists():
            raise SpecError(f"no experiment spec at {p}")
        try:
            d = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as e:
            raise SpecError(f"{p} is not valid YAML: {e}")
        if not isinstance(d, dict):
            raise SpecError(f"{p} must contain a mapping, got {type(d).__name__}")
        return ExperimentSpec.from_dict(d, path=str(p))

    @staticmethod
    def from_dict(d: dict, path: Optional[str] = None) -> "ExperimentSpec":
        known = {"name", "science_config", "old", "new", "behaviours", "recipe",
                 "execution", "stages", "audit_bundles", "min_p_minus_for_recycle"}
        unknown = set(d) - known
        if unknown:
            raise SpecError(f"unknown spec keys: {sorted(unknown)} (known: {sorted(known)})")
        for req in ("name", "science_config", "old", "new"):
            if not d.get(req):
                raise SpecError(f"spec is missing required key {req!r}")
        if not d.get("behaviours") and not d.get("recipe"):
            raise SpecError("give either behaviours: (an existing CSV) or recipe: (a BYOD recipe)")
        if d.get("behaviours") and d.get("recipe"):
            raise SpecError("give behaviours: or recipe:, not both -- "
                            "a recipe already names its output CSV")
        stages = tuple(d.get("stages") or STAGES)
        bad = [s for s in stages if s not in STAGES]
        if bad:
            raise SpecError(f"unknown stages {bad} (known, in order: {list(STAGES)})")
        return ExperimentSpec(
            name=str(d["name"]), science_config=str(d["science_config"]),
            old=str(d["old"]), new=str(d["new"]),
            behaviours=d.get("behaviours"), recipe=d.get("recipe"),
            execution=Execution.from_dict(d.get("execution") or {}),
            stages=stages, audit_bundles=int(d.get("audit_bundles", 60)),
            min_p_minus_for_recycle=int(d.get("min_p_minus_for_recycle", 5)), path=path)

    # ---------------- resolved paths ----------------
    def _abs(self, rel) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    @property
    def run_dir(self) -> Path:
        return self._abs(self.execution.run_root) / self.name

    @property
    def behaviours_csv(self) -> Path:
        """The canonical CSV. With a recipe, its declared `out:` -- read without running it."""
        if self.behaviours:
            return self._abs(self.behaviours)
        import yaml
        rp = self._abs(self.recipe)
        if not rp.exists():
            raise SpecError(f"no recipe at {rp}")
        r = yaml.safe_load(rp.read_text()) or {}
        if not r.get("out"):
            raise SpecError(f"{rp} has no out: key, so the CSV path is unknown")
        return self._abs(r["out"])

    @property
    def cache_path(self) -> Path:
        """This experiment's own response cache, sharded per task when a shard id is given.

        Two concurrent jobs must never append to one JSONL (LESSONS.md §6.2). A shard keeps them
        apart; `tools/cache_consolidate.py` merges shards afterwards.
        """
        d = self._abs(self.execution.cache_dir)
        if self.execution.shard:
            return d / "shards" / f"{self.name}__{self.execution.shard}.jsonl"
        return d / f"{self.name}.jsonl"

    @property
    def verdict_path(self) -> Path:
        d = self._abs(self.execution.cache_dir)
        if self.execution.shard:
            return d / "shards" / f"judge_{self.name}__{self.execution.shard}.jsonl"
        return d / f"judge_{self.name}.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / f"manifest_{self.name}.jsonl"

    @property
    def pminus_csv(self) -> Path:
        return self.run_dir / f"{self.name}_p_minus_judged.csv"

    @property
    def audit_dir(self) -> Path:
        return ROOT / "audit" / self.name

    # ---------------- fingerprint ----------------
    def fingerprint(self) -> dict:
        """What a resume must agree with: the inputs that would change the numbers.

        The dataset digest, the effective science config, the pair, the seed/budgets the cache keys
        derive from, and the code revision. A resume whose fingerprint differs is refused, because
        appending new draws to a cache keyed on a different seed silently mixes two experiments.
        """
        from .config import Config
        cfgp = self._abs(self.science_config)
        fp = {"spec_name": self.name, "pair": [self.old, self.new],
              "code_revision": code_revision(),
              "science_config": str(cfgp.relative_to(ROOT) if cfgp.is_relative_to(ROOT) else cfgp)}
        fp["science_config_sha256"] = _sha256_file(cfgp) if cfgp.exists() else None
        csv = self.behaviours_csv
        fp["behaviours"] = str(csv.relative_to(ROOT) if csv.is_relative_to(ROOT) else csv)
        fp["behaviours_sha256"] = _sha256_file(csv) if csv.exists() else None
        if cfgp.exists():
            cfg = Config.load(cfgp)
            fp["seed"] = cfg.seed
            fp["k_canonical_samples"] = cfg.budget.k_canonical_samples
            fp["samples_per_variant"] = cfg.budget.samples_per_variant
            fp["paraphrases_per_behaviour"] = cfg.budget.paraphrases_per_behaviour
            fp["transform_classes"] = list(cfg.transform_classes)
            fp["judges"] = cfg.judges or None
            fp["gate"] = {k: getattr(cfg.gate, k) for k in
                          ("cosine", "bertscore", "nli_entail", "llm_intent")
                          if hasattr(cfg.gate, k)}
        return fp

    # ---------------- environment ----------------
    def runtime(self) -> dict:
        return {"when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "host": platform.node(), "platform": platform.platform(),
                "python": sys.version.split()[0],
                "target": self.execution.target,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "ollama_url": os.environ.get("RECYCLEBENCH_OLLAMA_URL"),
                "shard": self.execution.shard}


# =============================================================================
# Stage state
# =============================================================================
@dataclass
class StageState:
    name: str
    done: bool
    artifacts: list
    missing: list
    recorded_fingerprint: Optional[dict] = None

    @property
    def valid(self) -> bool:
        return self.done and not self.missing


class StageStore:
    """`<run>/stages/<stage>.json`, written only after the stage's artifacts exist.

    A marker that is written before validation turns a timed-out stage into a "completed" one, and
    the next stage then reads a partial artifact. So `complete()` checks every declared artifact
    and refuses to record the stage if one is missing.
    """

    def __init__(self, run_dir: Path):
        self.dir = run_dir / "stages"

    def path(self, stage: str) -> Path:
        return self.dir / f"{stage}.json"

    def read(self, stage: str) -> Optional[dict]:
        p = self.path(stage)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None          # a truncated marker counts as not done

    def state(self, stage: str, artifacts) -> StageState:
        rec = self.read(stage)
        arts = [str(a) for a in artifacts]
        missing = [a for a in arts if not Path(a).exists()]
        return StageState(stage, rec is not None, arts, missing,
                          (rec or {}).get("fingerprint"))

    def complete(self, stage: str, artifacts, fingerprint: dict, extra: Optional[dict] = None):
        missing = [str(a) for a in artifacts if not Path(a).exists()]
        if missing:
            raise RuntimeError(f"stage {stage} produced no {missing} -- not recording it as done")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path(stage).write_text(json.dumps(
            {"stage": stage, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "artifacts": [str(a) for a in artifacts], "fingerprint": fingerprint,
             **(extra or {})}, indent=1))


def check_resume(spec: ExperimentSpec, store: StageStore) -> list:
    """Reasons this run may not be resumed. Empty list = safe.

    Compares each completed stage's recorded fingerprint against the current one and reports the
    keys that moved. `code_revision` is reported as a warning rather than a blocker: editing a tool
    between stages is normal, while changing the dataset, the seed or the judge is not.
    """
    now = spec.fingerprint()
    soft = {"code_revision"}
    problems = []
    for stage in spec.stages:
        rec = store.read(stage)
        if not rec or not rec.get("fingerprint"):
            continue
        was = rec["fingerprint"]
        moved = [k for k in sorted(set(was) | set(now)) if was.get(k) != now.get(k)]
        hard = [k for k in moved if k not in soft]
        for k in hard:
            problems.append(f"stage {stage}: {k} changed since it ran "
                            f"({was.get(k)!r} -> {now.get(k)!r})")
    return problems


# =============================================================================
# Plan
# =============================================================================
@dataclass
class Step:
    stage: str
    argv: list                     # the exact command, as a plan prints it and a run executes it
    artifacts: list                # files/dirs that must exist for the stage to count as done
    why_skipped: Optional[str] = None
    state: Optional[StageState] = None

    @property
    def runnable(self) -> bool:
        return self.why_skipped is None and not (self.state and self.state.valid)


@dataclass
class Plan:
    spec: ExperimentSpec
    steps: list
    problems: list                 # resume blockers
    notes: list

    @property
    def writes(self) -> list:
        """Files a plan itself writes. Always empty: planning is read-only, and a test asserts it."""
        return []

    def render(self) -> str:
        s = self.spec
        out = [f"experiment : {s.name}",
               f"spec       : {s.path or '(inline)'}",
               f"science    : {s.science_config}",
               f"pair       : {s.old} -> {s.new}",
               f"behaviours : {s.behaviours_csv}"
               + ("" if s.behaviours_csv.exists() else "   (MISSING -- the prepare stage writes it)"),
               f"run dir    : {s.run_dir}",
               f"cache      : {s.cache_path}",
               f"verdicts   : {s.verdict_path}",
               f"target     : {s.execution.target}"
               + (f"  shard={s.execution.shard}" if s.execution.shard else ""), ""]
        for n in self.notes:
            out.append(f"note: {n}")
        if self.notes:
            out.append("")
        for st in self.steps:
            if st.why_skipped:
                mark = f"SKIP    ({st.why_skipped})"
            elif st.state and st.state.valid:
                mark = "done"
            elif st.state and st.state.done and st.state.missing:
                mark = f"RERUN   (recorded done, but missing {st.state.missing})"
            else:
                mark = "pending"
            out.append(f"  {st.stage:10} {mark}")
            if st.runnable:
                out.append(f"             $ {' '.join(st.argv)}")
        if self.problems:
            out += ["", "RESUME BLOCKED:"] + [f"  - {p}" for p in self.problems]
        return "\n".join(out)


def _py() -> str:
    return sys.executable or "python3"


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT) if Path(p).is_absolute() and Path(p).is_relative_to(ROOT) else p)


def build_plan(spec: ExperimentSpec) -> Plan:
    """The ordered stages with their exact commands. Reads the filesystem; writes nothing."""
    store = StageStore(spec.run_dir)
    cfg, run, csv = spec.science_config, spec.run_dir, spec.behaviours_csv
    cache, verd = spec.cache_path, spec.verdict_path
    notes, steps = [], []

    pilot_json = run / "pminus_pilot.json"
    p_minus = None
    if pilot_json.exists():
        try:
            p_minus = json.loads(pilot_json.read_text()).get("p_minus")
        except json.JSONDecodeError:
            notes.append(f"{_rel(pilot_json)} is not readable JSON; the screen stage will rerun")

    def add(stage, argv, artifacts, why=None):
        if stage not in spec.stages:
            return
        steps.append(Step(stage, argv, [str(a) for a in artifacts], why,
                          store.state(stage, artifacts)))

    add("prepare",
        [_py(), "tools/prepare_external.py", "--recipe", str(spec.recipe or "")],
        [csv], None if spec.recipe else "no recipe: behaviours CSV is supplied directly")
    add("screen",
        [_py(), "-m", "recyclebench", "pilot-pminus", "--config", cfg,
         "--old", spec.old, "--new", spec.new, "--behaviours", _rel(csv),
         "--cache", _rel(cache), "--run-dir", _rel(run)],
        [pilot_json])
    add("subset",
        [_py(), "tools/pminus_subset.py", "--pilot", _rel(pilot_json),
         "--behaviours", _rel(csv), "--out", _rel(spec.pminus_csv)],
        [spec.pminus_csv],
        None if pilot_json.exists() else "needs the screen stage first")

    why_recycle = None
    if not spec.pminus_csv.exists() and p_minus is None:
        why_recycle = "needs the judged P^- subset first"
    elif p_minus is not None and p_minus < spec.min_p_minus_for_recycle:
        why_recycle = (f"|P^-| = {p_minus} < min_p_minus_for_recycle "
                       f"{spec.min_p_minus_for_recycle}: RSR would be undefined or unpowered")
    add("recycle",
        [_py(), "-m", "recyclebench", "run-real", "--config", cfg,
         "--behaviours", _rel(spec.pminus_csv), "--manifest", _rel(spec.manifest_path),
         "--cache", _rel(cache), "--run-dir", _rel(run)],
        [spec.manifest_path, run / "summary_real.json"], why_recycle)
    add("partition",
        [_py(), "tools/pminus_report.py", "--config", cfg, "--old", spec.old, "--new", spec.new,
         "--cache", _rel(cache), _rel(csv), "--screen-inverse", "--dump", _rel(run / "cells")],
        [run / "cells"])
    add("audit",
        [_py(), "tools/judge_export_real.py", "--config", cfg, "--new-model", spec.new,
         "--behaviours", _rel(spec.pminus_csv), "--manifest", _rel(spec.manifest_path),
         "--cache", _rel(cache), "--out", _rel(spec.audit_dir / "judge_sheet.csv"),
         "--n-bundles", str(spec.audit_bundles)],
        [spec.audit_dir / "judge_sheet.csv"],
        None if spec.manifest_path.exists() else "needs the frozen manifest from recycle")

    if spec.execution.shard:
        notes.append(f"sharded run: appends to {_rel(cache)} and {_rel(verd)} only, so it cannot "
                     f"collide with another job; merge with tools/cache_consolidate.py afterwards")
    if not Path(spec.science_config).is_absolute() and not (ROOT / spec.science_config).exists():
        notes.append(f"science config {spec.science_config} does not exist")
    return Plan(spec, steps, check_resume(spec, store), notes)


# =============================================================================
# Validate
# =============================================================================
def validate(spec: ExperimentSpec) -> list:
    """Everything checkable without running anything. Returns a list of problems ('' = ok).

    Deliberately includes "is the judge actually configured": a config with no `judges:` block
    falls back to the refusal-prefix rule, which is pilot-grade and disagrees with the LLM judge,
    so a run that silently used it would produce figures nobody can reconcile. That must be an
    explicit choice, not a default discovered afterwards.
    """
    from .config import Config
    problems = []
    cfgp = spec._abs(spec.science_config)
    if not cfgp.exists():
        return [f"science config not found: {cfgp}"]
    try:
        cfg = Config.load(cfgp)
    except Exception as e:                                     # noqa: BLE001 - report, don't raise
        return [f"{cfgp} does not load: {type(e).__name__}: {e}"]

    real = getattr(cfg, "real", None) or {}
    for who, key in (("old", spec.old), ("new", spec.new)):
        if key not in real:
            problems.append(f"{who} model {key!r} has no `real:` entry in {spec.science_config} "
                            f"(available: {sorted(real)})")
    if not cfg.judges:
        problems.append(f"{spec.science_config} has no `judges:` block, so screening would fall "
                        f"back to the offline refusal-prefix rule (pilot-grade). Configure a judge "
                        f"or state the fallback deliberately.")
    if "recycle" in spec.stages and not cfg.paraphraser:
        problems.append(f"stage 'recycle' needs a `paraphraser:` entry in {spec.science_config}")

    csv = spec.behaviours_csv
    if not csv.exists() and "prepare" not in spec.stages:
        problems.append(f"behaviours CSV {csv} does not exist and no prepare stage would create it")
    if csv.exists():
        try:
            from .datasets import load_jailbreakbench
            n = len(load_jailbreakbench(str(csv)))
            if n == 0:
                problems.append(f"{csv} parsed to 0 behaviours (is the column named 'Goal'?)")
        except Exception as e:                                 # noqa: BLE001
            problems.append(f"{csv} does not parse: {type(e).__name__}: {e}")
    if spec.recipe and not spec._abs(spec.recipe).exists():
        problems.append(f"recipe not found: {spec._abs(spec.recipe)}")
    if "recycle" in spec.stages and "subset" not in spec.stages and not spec.pminus_csv.exists():
        problems.append("stage 'recycle' without 'subset' needs the judged P^- CSV to exist "
                        f"already ({_rel(spec.pminus_csv)})")
    return problems


# =============================================================================
# Run
# =============================================================================
def _past_stop(stop_at: Optional[str]) -> bool:
    """Portable wall-clock stop. The shell queues used BSD `date -j -f`, which is macOS-only."""
    if not stop_at:
        return False
    try:
        hh, mm = (int(x) for x in stop_at.split(":"))
    except ValueError:
        raise SpecError(f"execution.stop_at must be HH:MM, got {stop_at!r}")
    now = datetime.now()
    stop = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return now >= stop


def run(spec: ExperimentSpec, only=None, force: bool = False, env=None) -> int:
    """Execute the pending stages in order. Returns a process exit code.

    Resumable by construction: every stage's own tool skips what its caches already hold, and a
    stage is recorded done only once its artifacts exist.
    """
    plan = build_plan(spec)
    if plan.problems and not force:
        print(plan.render())
        print("\nNot running. The inputs changed since a completed stage, so appending to this "
              "run would mix two experiments.\nStart a new `name:`, or pass --force if you are "
              "certain the change is immaterial.")
        return 1

    spec.run_dir.mkdir(parents=True, exist_ok=True)
    spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
    spec.cache_path.touch(exist_ok=True)           # fill_responses reads before appending
    fp = spec.fingerprint()
    (spec.run_dir / "experiment.json").write_text(json.dumps(
        {"spec": {k: v for k, v in vars(spec).items() if k != "execution"},
         "execution": vars(spec.execution), "fingerprint": fp, "runtime": spec.runtime(),
         "paths": {"cache": str(spec.cache_path), "verdicts": str(spec.verdict_path),
                   "manifest": str(spec.manifest_path), "p_minus_csv": str(spec.pminus_csv),
                   "audit": str(spec.audit_dir)}}, indent=1, default=str))

    store = StageStore(spec.run_dir)
    environ = {**os.environ, "PYTHONPATH": "src", "TRANSFORMERS_VERBOSITY": "error",
               "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               # the Pegasus known-issues page: unpinned BLAS threads make jobs slower and are the
               # leading suspect for the container-only reproducibility failure (LESSONS.md §6.5)
               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
               "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
               "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
               **(env or {})}

    ran = 0
    for step in plan.steps:
        if only and step.stage not in only:
            continue
        if step.why_skipped:
            print(f"--- {step.stage}: SKIP ({step.why_skipped})")
            continue
        if step.state and step.state.valid:
            print(f"--- {step.stage}: already done")
            continue
        if _past_stop(spec.execution.stop_at):
            print(f"--- {step.stage}: SKIPPED, past stop_at {spec.execution.stop_at}")
            continue
        print(f"\n=== {step.stage}: {' '.join(step.argv)}", flush=True)
        r = subprocess.run(step.argv, cwd=ROOT, env=environ)
        if r.returncode != 0:
            print(f"\n{step.stage} FAILED (rc={r.returncode}). Nothing was recorded as done; "
                  f"rerun the same command to resume.")
            return r.returncode
        try:
            store.complete(step.stage, step.artifacts, fp)
        except RuntimeError as e:
            print(f"\n{e}\nThe command exited 0 but its artifact is absent: treating the stage as "
                  f"incomplete rather than done.")
            return 1
        ran += 1
        print(f"--- {step.stage}: done")

    print(f"\n{ran} stage(s) ran. Artifacts in {_rel(spec.run_dir)}.")
    print("Figures are RAW: this corpus has no judge calibration of its own until the audit sheet "
          "in\n" + _rel(spec.audit_dir) + " is labelled (tools/primary_calibration.py).")
    return 0
