# Pre-registration — benign over-refusal on XSTest (RQ4 reference)

**Registered 2026-10-31 01:25, before any reply of this design exists** (a 4-prompt smoke test of the tooling went
to a scratch store and is not used). Exploratory addition that delivers the exposé's RQ4 reference ("anchored to
the model's own benign over-refusal rate"), which the study has so far replaced by the fixed π* = 0.15.
Decided by the candidate; the supervisor delegated the design.

* **Prompts:** XSTest v2 (Röttger et al., NAACL 2024; `paul-rottger/exaggerated-safety`, `xstest_prompts.csv`),
  250 safe + 200 unsafe contrast prompts, unchanged → `data/xstest.csv` (Goal, Category = XSTest type, Label).
* **Models, in this order (a wall-clock cut keeps whole models):** llama3-8b, llama31-8b, llama2-7b (the matched-scale
  7–8B chain), then llama2-13b, llama32-1b, llama32-3b, mistral7b-v02, mistral7b-v03.
* **Generation:** one draw per prompt, seed `_draw_seed(bid, -1, 0)` (the study's first canonical draw), temperature 0,
  Ollama defaults as in every study request, **max_tokens 256** (the study uses 512; a refusal shows in the opening,
  and 256 halves the time). Store `cache/overrefusal_xstest/`.
* **Judge:** the study's qwen2.5:14b rubric judge against the prompt; refusal = verdict 0. The offline refusal rule
  (`looks_like_refusal`) is reported beside it. Judge rates are raw: the judge was calibrated on harmful prompts only.
* **Estimands:** per model, refusal rate on the 250 safe prompts (benign over-refusal, Wilson 95%) and on the 200
  unsafe prompts (sanity: should be high); safe refusal by XSTest type (descriptive).
* **Comparisons:** for each matched pair of models, exact McNemar on the 250 safe prompts (newer vs older), descriptive
  (no correction; not a confirmatory test).
* **Use in the thesis:** (i) the RQ4 reference: compare each newer model's PBR and RSR with its own benign over-refusal;
  (ii) the alternative explanation "the newer model simply refuses more": if the newer model's over-refusal rises
  sharply, part of P⁻ may be indiscriminate refusal rather than learned harm detection. Never pooled with RSR/PBR.
* Results: `results/runs/overrefusal_xstest_<stamp>/summary.{json,txt}` (numbers only).

*Edit log: 2026-10-31 01:27, after launch: the stated registration time was corrected from 01:30 to 01:25 (the time of writing). No other change.*
