"""Real target-model adapters.

Any OpenAI-compatible endpoint works, which covers the intended local setup: serve each
Llama generation with vLLM (``vllm serve <model> --port 8001``) and point one adapter at
each port. Together/OpenAI/Anthropic are supported for hosted runs.

Decoding is pinned (temperature, top_p, seed, max_tokens) because the protocol requires
fixed decoding with k-sampling; only the sampling seed varies across the k draws.

No harmful content is generated here -- prompts come from the benchmark CSV the *user*
supplies, and raw completions are not persisted unless ``store_raw`` is explicitly enabled.
"""
from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..types import Behaviour, Response, Variant


@dataclass
class ProviderConfig:
    provider: str = "openai_compatible"   # openai_compatible | together | anthropic
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "RECYCLEBENCH_API_KEY"
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout: float = 120.0
    max_retries: int = 4
    system_prompt: Optional[str] = None


class ProviderModel:
    """A real chat model behind an HTTP API, exposing the same ``respond`` contract."""

    def __init__(self, name: str, pc: ProviderConfig):
        self.name = name
        self.pc = pc
        self._client = None

    # -- lazy client construction so importing the package never requires the SDKs --
    def _client_obj(self):
        if self._client is not None:
            return self._client
        key = os.environ.get(self.pc.api_key_env, "not-needed-for-local")
        if self.pc.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=key, timeout=self.pc.timeout)
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url=self.pc.base_url,
                                  timeout=self.pc.timeout)
        return self._client

    def _chat(self, prompt: str, seed: Optional[int]) -> str:
        c = self._client_obj()
        last = None
        for attempt in range(self.pc.max_retries):
            try:
                if self.pc.provider == "anthropic":
                    kw = {}
                    if self.pc.system_prompt:
                        kw["system"] = self.pc.system_prompt
                    r = c.messages.create(model=self.pc.model_id,
                                          max_tokens=self.pc.max_tokens,
                                          temperature=self.pc.temperature,
                                          messages=[{"role": "user", "content": prompt}], **kw)
                    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                msgs = ([{"role": "system", "content": self.pc.system_prompt}]
                        if self.pc.system_prompt else [])
                msgs.append({"role": "user", "content": prompt})
                kw = dict(model=self.pc.model_id, messages=msgs,
                          temperature=self.pc.temperature, top_p=self.pc.top_p,
                          max_tokens=self.pc.max_tokens)
                if seed is not None:
                    kw["seed"] = seed          # honoured by vLLM / OpenAI
                r = c.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception as exc:           # transient API/network failure
                last = exc
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"{self.name}: provider call failed after retries: {last}")

    def respond(self, prompt: str, *, behaviour: Optional[Behaviour] = None,
                variant: Optional[Variant] = None, canonical: bool = False,
                seed: Optional[int] = None) -> Response:
        text = self._chat(prompt, seed)
        self._trace(prompt, text, behaviour=behaviour, variant=variant,
                    canonical=canonical, seed=seed)
        return Response(text=text, latent_harmful=None)   # no latent truth for real models

    def _trace(self, prompt: str, text: str, *, behaviour: Optional[Behaviour],
               variant: Optional[Variant], canonical: bool, seed: Optional[int]) -> None:
        target = os.environ.get("RECYCLEBENCH_TRACE_MODEL")
        if not target:
            return
        names = {self.name, self.pc.model_id}
        if target not in names and target != "*":
            return

        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "name": self.name,
            "model_id": self.pc.model_id,
            "seed": seed,
            "canonical": canonical,
            "bid": getattr(behaviour, "bid", None),
            "variant_index": getattr(variant, "index", None),
            "transform_class": getattr(variant, "transform_class", None),
            "prompt": prompt,
            "response": text,
        }
        out = os.environ.get("RECYCLEBENCH_TRACE_PATH")
        if out:
            path = Path(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            return

        print("\n" + "=" * 88, flush=True)
        print(f"TRACE {row['ts']}  {self.name} ({self.pc.model_id})  seed={seed}", flush=True)
        if row["bid"] is not None:
            print(f"bid={row['bid']} canonical={canonical} variant={row['variant_index']} "
                  f"class={row['transform_class']}", flush=True)
        print("PROMPT:", flush=True)
        print(prompt, flush=True)
        print("OUTPUT:", flush=True)
        print(text, flush=True)


def load_provider_model(name: str, backend: str, cfg) -> ProviderModel:
    """Build a ProviderModel from ``cfg.real[name]`` (see configs/real.yaml)."""
    spec = (getattr(cfg, "real", {}) or {}).get(name)
    if spec is None:
        raise KeyError(f"no real-backend spec for model '{name}'. Add it under 'real:' "
                       f"in your config (see configs/real.yaml).")
    pc = ProviderConfig(**spec)
    return ProviderModel(name, pc)
