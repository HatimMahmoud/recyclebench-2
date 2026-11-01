# Pre-registration — quantisation sensitivity of the anchor (Amendment 2 D7)

**Registered 2026-10-31 14:10, before any reply of the Q4_K_M model exists** (the scoring path was dry-run on an empty store).
The registered lineage served `llama3:8b` at **Q4_0** and `llama3.1:8b` at **Q4_K_M** (Ollama default tags), so the
anchor pair differs in quantisation as well as generation. Only the older model's quantisation enters P⁻; every reply of
the newer model to every rewording is already cached and judged.

* **Model:** `llama3:8b-instruct-q4_K_M` (pulled 2026-10-31 14:07), config `configs/quant_check.yaml` (cache key
  `llama3-8b-q4km`, own store `cache/quant_check_20261031/`).
* **Screen:** the 440 SORRY-Bench requests, draw 0 (seed `_draw_seed(bid, -1, 0)`), temperature 0, max_tokens 512, Ollama
  defaults; judged by qwen2.5:14b.
* **Arms, identical but for the older model's weights:** A = registered llama3-8b (Q4_0) draw 0; B = Q4_K_M draw 0. The
  newer model is llama3.1-8b with its registered any-of-3 canonical verdicts in both arms. Using draw 0 for the older model
  in both arms (instead of the registered any-of-3) keeps the comparison matched and halves the generation; the registered
  k = 3 figures (P⁻ 49, 23/49) are printed beside them.
* **Estimands:** older-model answer rate in both arms with McNemar on the 440; P⁻ per arm and their Jaccard overlap; RSR per
  arm over its own P⁻ from the newer model's stored rewording verdicts (lineage manifest), raw and corrected (60-bundle
  calibration), resolved denominator primary, |P⁻| beside it.
* **Reading rule, fixed now:** the anchor's conclusion is *robust to the quantisation mismatch* if arm B's corrected RSR
  carries the same π\* verdict as arm A's; the difference B − A is reported with both intervals either way. If the older
  model's answer rate differs between quantisations (McNemar p < 0.05), the write-up states that quantisation changes
  which requests count as fixes. Exploratory; never pooled with the registered estimates.
