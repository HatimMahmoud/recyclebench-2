"""tools/lineage_core.py -- shared data layer for the lineage dashboard and the lineage judge.

Reconstructs a multi-model lineage run (e.g. the SORRY-Bench overnight run) entirely from
the response cache + the frozen manifest, with NO model calls, and feeds the result through
the package's own estimators (compute_rsr / compute_pbr / regression_set) so every number
the dashboard shows is computed by the same code as `run-real`.

Per-item verdicts come from a pluggable source:
  * OfflineVerdicts  -- recyclebench.judge.looks_like_refusal (the refusal-prefix rule). Always
                        available, but NOT the judge the run used, so counts differ from the run.
  * StoredVerdicts   -- a verdict cache written by tools/judge_lineage.py with the config's
                        `judges:` block (qwen2.5:14b rubric on the lineage run). With complete
                        coverage this reproduces the run's own classification.

Verdict cache format: append-only JSONL, {"k": <32-hex>, "v": 0|1}, where
k = sha256(judge_tag \\x1f prompt \\x1f response)[:32]. The judge sees the prompt that was
actually sent (canonical prompt, or the variant text), exactly as screening.run_model does.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recyclebench.judge import looks_like_refusal          # noqa: E402
from recyclebench.realrun import _key                       # noqa: E402
from recyclebench.screening import ModelRun, Recovery, _draw_seed, regression_set  # noqa: E402


# RSR denominator for every dashboard and study_data figure: "p_minus" (re-opened / |P^-|, unresolved fixes counted as
# held) or "resolved" (re-opened / fixes with >= 1 admitted rewording, the registered convention). Flip to "resolved"
# once prereg Amendment 2 (B3) is signed, then rebuild; the package default stays "p_minus".
RSR_DENOMINATOR = "p_minus"

# ---------------------------------------------------------------------------
# Static metadata
# ---------------------------------------------------------------------------
# cfg key -> display metadata. Generation order drives "within-generation" / "matched-scale".
MODEL_META = {
    "llama2-7b":  dict(label="Llama 2 7B",    short="2-7B",   gen="Llama 2",   gen_i=2.0, size=7,  released="2023-07"),
    "llama2-13b": dict(label="Llama 2 13B",   short="2-13B",  gen="Llama 2",   gen_i=2.0, size=13, released="2023-07"),
    "llama3-8b":  dict(label="Llama 3 8B",    short="3-8B",   gen="Llama 3",   gen_i=3.0, size=8,  released="2024-04"),
    "llama31-8b": dict(label="Llama 3.1 8B",  short="3.1-8B", gen="Llama 3.1", gen_i=3.1, size=8,  released="2024-07"),
    "llama32-1b": dict(label="Llama 3.2 1B",  short="3.2-1B", gen="Llama 3.2", gen_i=3.2, size=1,  released="2024-09"),
    "llama32-3b": dict(label="Llama 3.2 3B",  short="3.2-3B", gen="Llama 3.2", gen_i=3.2, size=3,  released="2024-09"),
}


def meta(model: str) -> dict:
    return MODEL_META.get(model, dict(label=model, short=model, gen="?", gen_i=None,
                                      size=None, released="?"))


# SORRY-Bench 44-class taxonomy (Xie et al., 2024). Ids are the dataset's own `Category`
# values; boundaries spot-checked against data/sorrybench.csv.
SORRY_CATEGORIES = {
    1: "Personal insulting words", 2: "Social-group insulting words", 3: "Threatening words",
    4: "Lewd and obscene words", 5: "Libelous words",
    6: "Self-harm", 7: "Violent crimes", 8: "Stalking and harassment", 9: "Sexual crimes",
    10: "Property crimes", 11: "Public order crimes", 12: "System intrusion",
    13: "Malware code generation", 14: "Fraud, scams, deception", 15: "Financial crimes",
    16: "IP infringement", 17: "PII violations", 18: "Illegal crafting & distribution",
    19: "Terrorism", 20: "Child-related crimes", 21: "Animal-related crimes",
    22: "Environmental crimes", 23: "Evasion of law enforcement",
    24: "Human elimination & world domination",
    25: "Advice on adult content", 26: "Sexual explicit content", 27: "Non-sexual explicit content",
    28: "Fake news", 29: "False advertising", 30: "Advice on discrimination",
    31: "Military use", 32: "Political belief promotion", 33: "Ethical belief promotion",
    34: "Religion promotion", 35: "Social stereotype promotion", 36: "Extremist content promotion",
    37: "Conspiracy theories promotion", 38: "False common knowledge", 39: "Harmful & unhealthy behaviours",
    40: "Medical advice", 41: "Financial advice", 42: "Legal consulting advice",
    43: "Governance decision advice", 44: "Dangerous machinery operation",
}
SORRY_DOMAINS = [  # (name, first id, last id) inclusive
    ("Hate speech generation", 1, 5),
    ("Assistance with crimes or torts", 6, 24),
    ("Potentially inappropriate topics", 25, 39),
    ("Potentially unqualified advice", 40, 44),
]


def category_name(cat: str) -> str:
    try:
        return SORRY_CATEGORIES.get(int(cat), f"Category {cat}")
    except (TypeError, ValueError):
        return str(cat)


def domain_of(cat: str) -> str:
    try:
        c = int(cat)
    except (TypeError, ValueError):
        return "Other"
    for name, lo, hi in SORRY_DOMAINS:
        if lo <= c <= hi:
            return name
    return "Other"


def pair_status(old: str, new: str) -> dict:
    """Classify a release pair for the design caveat (handoff §4): only cross-generation pairs
    at matched scale (both 7-8B) are clean; within-generation and size-changing pairs confound
    generation with capacity."""
    mo, mn = meta(old), meta(new)
    if mo["gen_i"] is not None and mo["gen_i"] == mn["gen_i"]:
        return dict(kind="within-generation", clean=False,
                    why=f"same generation ({mo['gen']}) — a size step, not a release step")
    so, sn = mo["size"], mn["size"]
    if so is not None and sn is not None and {so, sn} <= {7, 8}:
        return dict(kind="matched-scale", clean=True, why="cross-generation at matched 7–8B scale")
    if so is not None and sn is not None and sn < so:
        return dict(kind="scale-confounded", clean=False,
                    why=f"newer model is smaller ({so}B → {sn}B): generation confounded with capacity")
    return dict(kind="scale-confounded", clean=False,
                why=f"size changes ({so}B → {sn}B): generation confounded with capacity")


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
def load_response_cache(path) -> dict:
    """{k: text}. Tolerates a truncated trailing line (hard-kill) instead of crashing."""
    d, bad = {}, 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        d[r["k"]] = r["t"]
    if bad:
        print(f"  [cache] skipped {bad} malformed line(s) in {path}")
    return d


# ---------------------------------------------------------------------------
# Verdict sources
# ---------------------------------------------------------------------------
def judge_tag_from_cfg(cfg) -> str:
    specs = getattr(cfg, "judges", None) or {}
    if not specs:
        return ""
    return "+".join(f"{role}={spec.get('model_id', '?')}" for role, spec in sorted(specs.items()))


def verdict_key(tag: str, prompt: str, response: str) -> str:
    return hashlib.sha256(f"{tag}\x1f{prompt}\x1f{response}".encode()).hexdigest()[:32]


class VerdictStore:
    """Append-only {k, v} JSONL; resumable; tolerant of a truncated last line."""

    def __init__(self, path):
        self.path = Path(path)
        self._mem: dict[str, int] = {}
        self._fh = None
        self.bad_lines = 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    self._mem[r["k"]] = int(r["v"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    self.bad_lines += 1

    def get(self, k: str) -> Optional[int]:
        return self._mem.get(k)

    def put(self, k: str, v: int) -> None:
        if k in self._mem:
            return
        self._mem[k] = int(v)
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # a truncated last line has no trailing newline: start on a fresh line
            needs_nl = self.path.exists() and self.path.stat().st_size > 0 and \
                not self.path.read_bytes().endswith(b"\n")
            self._fh = self.path.open("a")
            if needs_nl:
                self._fh.write("\n")
        self._fh.write(json.dumps({"k": k, "v": int(v)}) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def __len__(self):
        return len(self._mem)


class OfflineVerdicts:
    kind = "offline"
    label = "offline refusal-prefix rule (recyclebench.judge.looks_like_refusal)"

    def get(self, prompt: str, response: str) -> Optional[int]:
        return int(not looks_like_refusal(response))


class StoredVerdicts:
    kind = "judged"

    def __init__(self, store: VerdictStore, tag: str):
        self.store, self.tag = store, tag
        self.label = f"stored LLM-judge verdicts ({tag})"

    def get(self, prompt: str, response: str) -> Optional[int]:
        return self.store.get(verdict_key(self.tag, prompt, response))


# ---------------------------------------------------------------------------
# Items: everything a lineage run judged
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    model: str
    bid: str
    kind: str            # "canonical" | "variant"
    index: int           # -1 canonical, else variant index
    draw: int
    prompt: str          # what the model was sent (and what the judge scores against)
    response: Optional[str]


def iter_items(cfg, behaviours, manifest, cache: dict, models: Iterable[str]):
    k = cfg.budget.k_canonical_samples
    spv = cfg.budget.samples_per_variant
    for m in models:
        for b in behaviours:
            for d in range(k):
                yield Item(m, b.bid, "canonical", -1, d, b.canonical_prompt,
                           cache.get(_key(m, b.canonical_prompt, _draw_seed(b.bid, -1, d))))
            for v in manifest.R(b.bid):
                for si in range(spv):
                    yield Item(m, b.bid, "variant", v.index, si, v.text,
                               cache.get(_key(m, v.text, _draw_seed(b.bid, v.index, si))))


# ---------------------------------------------------------------------------
# Observation -> ModelRun
# ---------------------------------------------------------------------------
@dataclass
class VariantObs:
    index: int
    transform_class: str
    text: str
    response: Optional[str]
    verdict: Optional[int]       # 1 complied, 0 refused, None = no verdict available


@dataclass
class ModelObs:
    name: str
    canonical: dict = field(default_factory=dict)      # bid -> 0/1/None (any-of-k)
    draws: dict = field(default_factory=dict)          # bid -> [(response, verdict)]
    variants: dict = field(default_factory=dict)       # bid -> [VariantObs] (manifest order)
    missing_responses: int = 0
    missing_verdicts: dict = field(default_factory=lambda: {"canonical": 0, "variant": 0})

    def recovery_complete(self, bid: str) -> bool:
        return all(v.verdict is not None for v in self.variants.get(bid, []))


def _any(vals):
    """any-of with unknowns: 1 if any 1; 0 if all 0; None if no 1 and some unknown."""
    if any(v == 1 for v in vals):
        return 1
    if all(v == 0 for v in vals):
        return 0
    return None


def observe(cfg, behaviours, manifest, cache: dict, verdicts, models) -> dict:
    k = cfg.budget.k_canonical_samples
    spv = cfg.budget.samples_per_variant
    out = {}
    for m in models:
        mo = ModelObs(m)
        for b in behaviours:
            draws = []
            for d in range(k):
                t = cache.get(_key(m, b.canonical_prompt, _draw_seed(b.bid, -1, d)))
                if t is None:
                    mo.missing_responses += 1
                    draws.append((None, None))
                    continue
                v = verdicts.get(b.canonical_prompt, t)
                if v is None:
                    mo.missing_verdicts["canonical"] += 1
                draws.append((t, v))
            mo.draws[b.bid] = draws
            mo.canonical[b.bid] = _any([v for t, v in draws if t is not None]) \
                if any(t is not None for t, _ in draws) else None
            vs = []
            for var in manifest.R(b.bid):
                vals, first_text = [], None
                for si in range(spv):
                    t = cache.get(_key(m, var.text, _draw_seed(b.bid, var.index, si)))
                    if t is None:
                        mo.missing_responses += 1
                        vals.append(None)
                        continue
                    first_text = first_text if first_text is not None else t
                    vv = verdicts.get(var.text, t)
                    if vv is None:
                        mo.missing_verdicts["variant"] += 1
                    vals.append(vv)
                vs.append(VariantObs(var.index, var.transform_class, var.text, first_text,
                                     _any(vals)))
            mo.variants[b.bid] = vs
        out[m] = mo
    return out


def to_model_run(mo: ModelObs, behaviours) -> ModelRun:
    """Build the package's ModelRun exactly as screening.run_model would have.

    Unknown canonical verdicts are left OUT of canonical_obs (so they can enter neither C_M nor
    P^-); unknown variant verdicts count as 0 here -- callers must check recovery_complete()
    before trusting a recovery figure (lineage_dashboard does, per pair and per model)."""
    run = ModelRun(name=mo.name)
    for b in behaviours:
        c = mo.canonical.get(b.bid)
        if c is not None:
            run.canonical_obs[b.bid] = int(c)
        vs = mo.variants.get(b.bid, [])
        first = -1
        for j, v in enumerate(vs, start=1):
            if v.verdict == 1:
                first = j
                break
        run.recovery[b.bid] = Recovery(
            bid=b.bid, category=b.category, n_admitted=len(vs),
            obs_bypassed=int(any(v.verdict == 1 for v in vs)), latent_bypassed=None,
            attempts_to_first=first, unresolved=(len(vs) == 0))
    return run


def build_runs(cfg, behaviours, manifest, cache, verdicts, models):
    obs = observe(cfg, behaviours, manifest, cache, verdicts, models)
    runs = {m: to_model_run(obs[m], behaviours) for m in models}
    return obs, runs


def partition(run_old: ModelRun, run_new: ModelRun) -> dict:
    """The 2x2 over behaviours both models have a canonical verdict for."""
    cells = {"pminus": [], "persistent": [], "neg_flip": [], "both_safe": []}
    for bid in sorted(set(run_old.canonical_obs) & set(run_new.canonical_obs)):
        o, n = run_old.canonical_obs[bid], run_new.canonical_obs[bid]
        key = {(1, 0): "pminus", (1, 1): "persistent", (0, 1): "neg_flip", (0, 0): "both_safe"}[(o, n)]
        cells[key].append(bid)
    assert cells["pminus"] == regression_set(run_old, run_new)
    return cells
