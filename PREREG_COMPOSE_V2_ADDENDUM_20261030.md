# Addendum to PREREG_COMPOSE_V2_20261029.md — a second pair (registered 2026-10-30, before its data)

The registered design is applied **unchanged** to a second pair, to test whether the anchor's result (no style
pair re-opens more than its singles) holds beyond it:

* **Pair:** llama2-7b → llama3.1-8b — the matched-scale 7–8B chain's skip pair (post-hoc pair in THESIS_KB §7.3).
* **Behaviours:** its qwen2.5:14b-judged SORRY-Bench P⁻ (23 behaviours, full-benchmark ids), from the same lineage
  manifest and verdict cache.
* Everything else as registered: 21 conditions, 3 candidates, generator, gate against the original request, judge
  against the candidate text, the primary comparison (pair vs either single, McNemar, Holm over 15) and the
  secondary ones. With 23 behaviours the tests have little power; a null there is weak evidence on its own.
* Separate store `cache/compose_v2__llama2-7b__llama31-8b/`, results `results/runs/compose_v2_llama2-7b__to__llama31-8b_<stamp>/`.
  The anchor run is not touched. Exploratory.
