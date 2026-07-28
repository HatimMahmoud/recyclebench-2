"""The frozen, shared reformulation manifest (exposé Sec. 5, P2).

The reformulation set is generated once, gated once, then **frozen and reused identically
across every model in the lineage** -- a pre-registered design choice so a cross-model
difference reflects the model and not a re-sampled stimulus. This module builds R(p) (the
gate-admitted variants per behaviour), records every gate decision for the human audit, and
serialises the manifest to JSONL so it is a reproducible released artifact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .types import Behaviour, Variant


@dataclass
class GateAuditRecord:
    bid: str
    index: int
    transform_class: str
    intent_preserved: bool     # latent truth (mock) / human label (real audit subset)
    admitted: bool
    # component scores are persisted so the pre-registered THRESHOLD SENSITIVITY analysis
    # can re-apply alternative thresholds offline, without re-querying any model
    cosine: float = float("nan")
    bertscore: float = float("nan")
    nli_entail: float = float("nan")
    llm_intent: float = float("nan")
    human_label: Optional[bool] = None   # filled by the human audit on real runs

    def to_dict(self):
        return dict(self.__dict__)

    def scores(self) -> dict:
        return {"cosine": self.cosine, "bertscore": self.bertscore,
                "nli_entail": self.nli_entail, "llm_intent": self.llm_intent}

    def truth(self) -> Optional[bool]:
        """Human label where available (real runs), else the latent truth (mock)."""
        return self.human_label if self.human_label is not None else self.intent_preserved


@dataclass
class Manifest:
    """R(p) for every behaviour, plus the gate audit trail."""
    admitted: dict[str, list[Variant]] = field(default_factory=dict)
    audit: list[GateAuditRecord] = field(default_factory=list)

    # ---- construction ----
    @staticmethod
    def build(behaviours, reformulator, gate) -> "Manifest":
        m = Manifest()
        for b in behaviours:
            admitted: list[Variant] = []
            for v in reformulator.variants(b):
                gs = gate.check(b, v)
                m.audit.append(GateAuditRecord(
                    b.bid, v.index, v.transform_class, v.intent_preserved, gs.admitted,
                    cosine=gs.cosine, bertscore=gs.bertscore,
                    nli_entail=gs.nli_entail, llm_intent=gs.llm_intent))
                if gs.admitted:
                    admitted.append(v)
            m.admitted[b.bid] = admitted
        return m

    def R(self, bid: str) -> list[Variant]:
        return self.admitted.get(bid, [])

    def n_admitted(self, bid: str) -> int:
        return len(self.admitted.get(bid, []))

    def mean_admitted(self) -> float:
        vals = [len(v) for v in self.admitted.values()]
        return float(sum(vals) / len(vals)) if vals else 0.0

    # ---- audit summary (false-admission / false-rejection) ----
    def audit_rates(self) -> dict:
        """Gate error against intent truth: false-admission (changed->admitted) and
        false-rejection (preserved->rejected), each as a rate over its base."""
        changed = [r for r in self.audit if r.truth() is False]
        preserved = [r for r in self.audit if r.truth() is True]
        fa = sum(1 for r in changed if r.admitted)
        fr = sum(1 for r in preserved if not r.admitted)
        n_admit = sum(1 for r in self.audit if r.admitted)
        return {
            "n_variants": len(self.audit),
            "n_admitted": n_admit,
            "false_admission_rate": (fa / len(changed)) if changed else 0.0,
            "false_rejection_rate": (fr / len(preserved)) if preserved else 0.0,
            "n_changed": len(changed), "n_preserved": len(preserved),
            "admitted_that_were_changed": fa,   # these would invalidate a measured bypass
        }

    # ---- serialisation (released artifact) ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for bid, vs in self.admitted.items():
                fh.write(json.dumps({"type": "admitted", "bid": bid,
                                     "variants": [v.to_dict() for v in vs]}) + "\n")
            for r in self.audit:
                fh.write(json.dumps({"type": "audit", **r.to_dict()}) + "\n")

    @staticmethod
    def load(path: str | Path) -> "Manifest":
        m = Manifest()
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d["type"] == "admitted":
                m.admitted[d["bid"]] = [Variant.from_dict(x) for x in d["variants"]]
            elif d["type"] == "audit":
                d.pop("type")
                m.audit.append(GateAuditRecord(**d))
        return m
