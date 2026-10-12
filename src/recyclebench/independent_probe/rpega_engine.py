"""Genetic Reverse-Prompt-Engineering engine (RPE_GA), wrapping
hanklee97121/RPE_Reverse_Prompt_Engineering (MIT) as a ProbeEngine.
Same .invert() contract as the MVP RPEEngine. The GA refines candidates
(cross-mutation + ROUGE fitness vs the target output) and returns the single
best-recovered prompt -- fixing the 'best-of-N can't pick the good one' problem.

Vendored upstream at: tools/vendor/rpe_ga/RPE_ga.py
Deps: openai spacy evaluate rouge_score numpy ; python -m spacy download en_core_web_sm
"""
from __future__ import annotations
from dataclasses import dataclass
import os, sys


@dataclass
class RPEGAEngine:
    model_id: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    X: int = 5
    N: int = 5
    n_answers: int = 1
    spacy_model: str = "en_core_web_sm"
    vendor_path: str = "tools/vendor/rpe_ga"
    name: str = "rpe-ga"
    _RPE: object = None

    def _ensure_loaded(self):
        if self._RPE is not None:
            return
        os.environ["OPENAI_BASE_URL"] = self.base_url
        os.environ["OPENAI_API_KEY"] = self.api_key
        vp = os.path.abspath(self.vendor_path)
        if vp not in sys.path:
            sys.path.insert(0, vp)
        from RPE_ga import Reverse_Prompt_Engineering
        self._RPE = Reverse_Prompt_Engineering(
            X=self.X, N=self.N, model=self.model_id, tag_model=self.spacy_model)

    def invert(self, old_response: str, n_candidates: int = 5) -> list[str]:
        self._ensure_loaded()
        rpe = self._RPE
        N = max(self.N, min(n_candidates, 10))
        rpe.reset(X=self.X, N=N)
        rpe.generate_answers(prompt=None, answers=[old_response], n=self.n_answers)
        best = rpe(ini_prompts=None, ini_scores=None, ini_total_scores=None, ini_answers=None)
        best = (best or "").strip()
        return [best] if best else []
