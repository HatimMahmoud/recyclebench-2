# Pre-registration — cache staleness check (reproducibility)

**Registered 2026-10-31 14:07, before any regeneration** (no smoke test; the tool's sampling was counted offline only).
Responds to an external review point: cache keys hold (config key, prompt, seed) but not the weights, the Ollama runtime or
the decoding settings, so a re-pulled model or a changed runtime would be served stale replies without notice. Manifest
dates show llama2:13b and llama3.2:1b/3b re-pulled on 2026-10-10, after their July/August caches were written.

* **Sample** (`tools/cache_staleness.py`, seed 20261101): for each of the six lineage models, 20 cached canonical draw-0
  replies from SORRY-Bench (stratum S, screened 2026-10-19/20) and 20 from HarmBench / AdvBench / StrongREJECT pooled
  (stratum O, July–August), drawn uniformly from the cached items. 240 replies.
* **Procedure:** regenerate each with the study's request (config `configs/real_llama_lineage_all.yaml`, temperature 0,
  max_tokens 512, the same seed, Ollama defaults); judge cached and regenerated replies with the study's judge
  (qwen2.5:14b rubric) into a separate store (`cache/staleness_20261031/`). The main cache is not modified.
* **Reported per model × stratum:** byte-identical rate, whitespace-normalised identical rate, judge-verdict agreement
  (Wilson 95%), the list of verdict flips; the Ollama version and the weights blob of each model.
* **Reading rule, fixed now:** verdict agreement is the quantity that matters for the estimates. A model × stratum cell with
  verdict agreement below 90% (point estimate) is flagged: its cached results are reported as *not reproducible with the
  current weights/runtime* in the threats to validity, with the flip direction. Byte-identity below 100% alone is reported
  but not flagged (runtime nondeterminism at temperature 0 can change wording without changing the decision). No cached
  result is replaced or re-estimated on the basis of this check.
