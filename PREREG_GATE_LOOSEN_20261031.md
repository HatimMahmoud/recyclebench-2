# Pre-registration — loosened gate thresholds (Amendment 1 A3.5, the side the manifests could not support)

**Registered 2026-10-31 14:12, before any re-decision or reply** (the scoring path was dry-run: the strict row reproduces
compose v2's per-style figures exactly). Amendment 1 A3.5 registered a gate-threshold sensitivity of ±0.05 / ±0.10; only
tightening was possible, because the frozen manifests keep no rejected text.

* **Population:** the compose v2 store's six single-style conditions (production paraphraser qwen2.5:3b, the six registered
  classes, production gate against the original request): 882 candidates over the anchor's 49 judged fixes. Style pairs are
  excluded (two-step rewrites). The store is read, never written; own store `cache/gate_loosen_20261031/`.
* **Loosening:** cosine 0.75, BERTScore 0.85, NLI 0.60, each lowered jointly by δ ∈ {0.05, 0.10}; the lexical floor unchanged.
  The production RealGate, built with the δ = 0.10 thresholds, re-decides every candidate the strict gate rejected at a
  semantic layer (`tools/gate_loosen.py`); δ = 0.05 follows from the same layer scores.
* **Replies and verdicts:** every newly admitted candidate is answered once by llama3.1-8b with compose v2's settings (config
  `configs/recycle_3_31.yaml`, seed 20261029 + index, temperature 0) and judged by qwen2.5:14b against the candidate text.
* **Estimands per δ:** admissions; RSR over the resolved fixes for each single style and for "any single style", Wilson 95%,
  and corrected with the 60-bundle calibration; newly resolved fixes.
* **Reading rule, fixed now:** reported beside the strict figures as the loosened half of A3.5. Newly admitted rewordings carry
  no human intent label at the time of scoring; the false-rejection sheet (Amendment 2 §B4.1) samples the same rejected pool,
  so once it is labelled the share of loosened admissions that preserve intent is reported with it. A rise in RSR under
  loosening is not read as evidence about the anchor without that share. Exploratory.
