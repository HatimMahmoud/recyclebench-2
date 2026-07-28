"""Orchestration for real target models.

A confirmatory run is expensive: |B| behaviours x (k canonical draws + t admitted variants)
x |lineage| models. Two properties therefore matter as much as the statistics:

* **Resumability.** Every model call is cached to JSONL keyed by (model, prompt, seed), so an
  interrupted run resumes instead of re-paying for completed work. The cache is also what
  makes a run auditable after the fact.
* **No fabricated ground truth.** Real backends return no latent label, so nothing here
  invents one. Calibration is *undefined* until human labels are ingested
  (``judge-export`` / ``judge-ingest``), and the protocol forbids reporting a corrected
  headline with an unadmitted judge. This module reports raw estimates and states plainly
  that the correction is pending.

The frozen-manifest discipline is unchanged: reformulations are generated once, gated once,
and reused identically across every model in the lineage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .types import Behaviour, Response, Variant
from .config import Config


# ---------------------------------------------------------------------------
# Resumable response cache
# ---------------------------------------------------------------------------
def _key(model: str, prompt: str, seed: Optional[int]) -> str:
    h = hashlib.sha256(f"{model}|{prompt}|{seed}".encode()).hexdigest()
    return h[:32]


class ResponseCache:
    """Append-only JSONL cache of model responses, keyed by (model, prompt, seed).

    Stores the response text because that is what a judge (and a human annotator) must see.
    The cache therefore contains model output to harmful prompts and is a working file, not a
    released artefact -- see ETHICS.md.
    """

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._mem: dict[str, str] = {}
        self._fh = None
        if enabled and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._mem[d["k"]] = d["t"]

    def get(self, model: str, prompt: str, seed) -> Optional[str]:
        if not self.enabled:
            return None
        return self._mem.get(_key(model, prompt, seed))

    def put(self, model: str, prompt: str, seed, text: str) -> None:
        if not self.enabled:
            return
        k = _key(model, prompt, seed)
        if k in self._mem:
            return
        self._mem[k] = text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._fh is None:
            self._fh = self.path.open("a")
        self._fh.write(json.dumps({"k": k, "t": text}) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def __len__(self):
        return len(self._mem)


class CachedModel:
    """Wraps any model so repeated (prompt, seed) pairs are served from disk."""

    def __init__(self, inner, cache: ResponseCache):
        self.inner = inner
        self.name = inner.name
        self.cache = cache
        self.calls = 0
        self.hits = 0

    def respond(self, prompt: str, *, behaviour=None, variant=None,
                canonical: bool = False, seed=None) -> Response:
        hit = self.cache.get(self.name, prompt, seed)
        if hit is not None:
            self.hits += 1
            return Response(text=hit, latent_harmful=None)
        self.calls += 1
        r = self.inner.respond(prompt, behaviour=behaviour, variant=variant,
                               canonical=canonical, seed=seed)
        self.cache.put(self.name, prompt, seed, r.text)
        return r


# ---------------------------------------------------------------------------
# Real study
# ---------------------------------------------------------------------------
@dataclass
class RealStudyResult:
    cfg: Config
    manifest: object
    runs: list = field(default_factory=list)
    pilot: object = None
    rsr_raw: list = field(default_factory=list)
    pbr_raw: list = field(default_factory=list)
    gate_audit: dict = field(default_factory=dict)
    cache_stats: dict = field(default_factory=dict)
    calibration_status: str = ""

    def summary(self) -> dict:
        return {
            "backend": "real",
            "lineage": [r.name for r in self.runs],
            "gate_audit": self.gate_audit,
            "calibration_status": self.calibration_status,
            "pilot": self.pilot.summary() if self.pilot else None,
            "RSR_primary_RAW": [r.summary() for r in self.rsr_raw],
            "PBR_backbone_RAW": [p.summary() for p in self.pbr_raw],
            "cache": self.cache_stats,
            "WARNING": ("These are RAW, judge-error-UNCORRECTED estimates. The protocol "
                        "requires bundle-level calibration from human labels before a "
                        "headline is reported: run judge-export -> annotate -> judge-ingest."),
        }


def run_real_study(cfg: Config, behaviours: list[Behaviour], models: list,
                   judge, manifest, cache_path: str | Path = "results/cache.jsonl",
                   use_cache: bool = True) -> RealStudyResult:
    """Screen a real lineage against a frozen manifest and compute RAW estimates.

    ``models`` are already-constructed backends (oldest first); ``manifest`` is the frozen
    R(p). Nothing here corrects for judge error -- that requires human calibration.
    """
    from .screening import run_model, regression_set
    from .pilot import run_pilot
    from .recycling import compute_rsr
    from .pbr import compute_pbr
    from .stats import JudgeChars

    cache = ResponseCache(cache_path, enabled=use_cache)
    wrapped = [CachedModel(m, cache) for m in models]
    runs = [run_model(m, behaviours, manifest, judge, cfg, collect_calib=False,
                      store_responses=True) for m in wrapped]

    pilot = run_pilot(runs, cfg.decision.min_p_minus)
    # identity "judge" => corrected == raw; the real correction waits for human labels
    identity = JudgeChars(se=1.0, sp=1.0, response_level=False)
    rsr = [compute_rsr(o, n, regression_set(o, n), identity, cfg.decision.min_p_minus)
           for o, n in zip(runs, runs[1:])]
    pbr = [compute_pbr(r, identity, cfg.decision.pi_star) for r in runs]

    stats = {"entries": len(cache), "api_calls": sum(m.calls for m in wrapped),
             "cache_hits": sum(m.hits for m in wrapped), "path": str(cache_path)}
    cache.close()

    return RealStudyResult(
        cfg=cfg, manifest=manifest, runs=runs, pilot=pilot, rsr_raw=rsr, pbr_raw=pbr,
        gate_audit=manifest.audit_rates(), cache_stats=stats,
        calibration_status=("PENDING -- no human labels ingested; estimates are raw and "
                            "must not be reported as headline figures"))


def pilot_p_minus(cfg: Config, behaviours: list[Behaviour], old_model, new_model, judge,
                  manifest, cache_path: str | Path = "results/cache.jsonl") -> dict:
    """Cheap Month-1 pilot: size |P^-| for one pair using ONLY canonical screening.

    Paraphrases are not generated or queried here -- the whole point of the pilot is to
    decide, before spending that budget, whether the recycling arm is powered at all.
    """
    from .screening import _screen_canonical

    cache = ResponseCache(cache_path)
    old, new = CachedModel(old_model, cache), CachedModel(new_model, cache)
    k = cfg.budget.k_canonical_samples
    breaks_old, blocked_new, p_minus = 0, 0, []
    for b in behaviours:
        s_old, _ = _screen_canonical(b, old, judge, k)
        if s_old != 1:
            continue
        breaks_old += 1
        s_new, _ = _screen_canonical(b, new, judge, k)
        if s_new == 0:
            blocked_new += 1
            p_minus.append(b.bid)
    cache.close()

    n = len(p_minus)
    status = ("headline-feasible" if n >= cfg.decision.min_p_minus
              else ("exploratory" if n > 0 else "undefined (P^- empty)"))
    return {"pair": f"{old_model.name} -> {new_model.name}",
            "n_behaviours": len(behaviours), "breaks_old": breaks_old,
            "blocked_by_new_among_those": blocked_new, "p_minus": n,
            "min_required": cfg.decision.min_p_minus, "status": status,
            "p_minus_bids": p_minus,
            "api_calls": old.calls + new.calls, "cache_hits": old.hits + new.hits}
