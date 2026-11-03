# Llama 4 Scout: the full RecycleBench study (2026-10-31 … 11-03)

**Designation, everywhere:** post hoc (not in the pre-registration or its amendments), scale-confounded
(109B-parameter MoE, 17B active, against ≤13B dense models), **exploratory**. Every figure is raw judge output
(qwen2.5:14b rubric); no human calibration exists for Scout, and the SORRY-Bench/HarmBench calibrations are not
transferred to it. Nothing here is pooled with the registered pairs.

Artifacts: `results/runs/llama4_scout/` (gitignored; manifests, closed-loop files and caches hold harmful text) and
`results/runs/20261031_1816_llama4_scout_canonical/`. The completion matrix is
`results/runs/llama4_scout/suite_20261103_1452.{json,md}` (`tools/llama4_suite.py`); the study page is
`dashboard_llama4_scout_study[_redacted]_20261103_1452.html` (`tools/llama4_dashboard.py --suite`). The redacted page
holds counts only.

## Coverage

| Arm | Cells | Status |
|---|---|---|
| Canonical screen, AdvBench 520 / HarmBench 400 / StrongREJECT 313, 7 models | 18 pairs | COMPLETE — VERIFIED: re-derived independently from the raw cache + verdict stores, 100 % coverage, P⁻ ids identical |
| Recycling (RSR), pairs with judged \|P⁻\| ≥ 5 (driver floor) | 3 | COMPLETE — VERIFIED (exploratory: all < 30) |
| Recycling, other pairs (\|P⁻\| 0–4) | 15 | NOT APPLICABLE — below the floor |
| Output inversion (RPE), pairs with \|P⁻\| ≥ 1 | 11 | COMPLETE — VERIFIED |
| RPE, pairs with \|P⁻\| = 0 | 7 | NOT APPLICABLE — PRR undefined (never 0) |
| Known-attack transfer, 31 classes × 300 behaviours | 1 | COMPLETE — VERIFIED |
| Rewording styles at the gate (style arcs) | 1 | COMPLETE — VERIFIED |
| SORRY-Bench screen | — | BLOCKED — CSV absent on Pegasus and Machine 2 (gated on HF; copy from the Mac) |
| PBR / RSR on the lineage's frozen manifests | — | BLOCKED — manifests only on the Mac |
| Held-out JBB | — | NOT APPLICABLE — registered for llama3-8b → 3.1-8b only |
| Compose v2, mechanistic probe, PTP | — | NOT APPLICABLE — registered for other pairs / dense-only code / arm parked |
| Human judge calibration for Scout | — | BLOCKED — sheets written (`audit/llama4_scout__*/judge_sheet.csv`), unlabelled |

## Results

**Canonical.** Scout answered AdvBench 7/520, HarmBench 137/400, StrongREJECT 9/313. Largest P⁻ (fixes) over the six
older models: 6, 7 and 1; regressions (older refused, Scout answered) up to 107 on HarmBench. The newer model mostly
answers *more*, so P⁻ is far below the confirmatory floor of 30 for every pair.

**Recycling** (each pair froze its own manifest: qwen2.5:3b, 8 rewordings over 6 styles, production gate; both
models re-screened on it with fresh draws in one environment):

| Pair | P⁻ stored | P⁻ re-judged | P⁻ in run | RSR raw (Wilson 95 %) |
|---|---|---|---|---|
| HarmBench llama3-8b → Scout | 7 | 8 | 4 | 2/4 [0.15, 0.85] |
| HarmBench llama3.2-3b → Scout | 5 | 5 | 2 | 2/2 [0.34, 1.00] |
| AdvBench llama3-8b → Scout | 6 | 6 | 5 | 0/5 [0.00, 0.43] |

Back-translation had **0 admitted variants** in every pair, which is why it never re-opens anything here.

**Inversion (RPE)**, fixes re-opened / resolved: HarmBench llama3-8b → Scout A_gate 4/5, A_loop 5/6, A_both 4/5;
HarmBench llama3.2-3b A_gate 3/4, A_loop 3/4; AdvBench llama3-8b A_gate 0/0 (the gate admitted none of 47
reconstructions), A_loop 3/3. Every other pair has n ≤ 4. Per-pair figures: `rpe/<bench>/closedloop_*_gated_*.json`.

**Attack transfer** (judged against the original goal; denominator = the 279 of 300 behaviours Scout refused in
plain text). Highest: combination_3 56/278 (20.1 % [15.9, 25.3]), AIM 38/279 (13.6 %), combination_1 34/278 (12.2 %);
ten more classes 3.6–6.8 %; the rest ≤ 2.5 %. Check: class `none` (the unchanged request) gives 0/279 and 21/300
bypass, exactly the baseline. Cipher classes are flagged on the page: the judge's design-weighted sensitivity on
attack output is ~0.07, so this arm is under-counted (THESIS_KB §4.3).

## Reproducibility

- **Code:** branch `llama4-scout-suite`. Recycle, RPE and attack shards ran at `8fdd17c`; the merge, judge pass and
  report ran at `8432cc4`. The recycle runs' `experiment.json` records `code_revision: unknown` (no git in the image;
  fixed in `8432cc4`); their revision is established by the sync log and `logs/scout_chain.txt`.
- **Image:** `recyclebench_gate_8fdd17c.sqsh` (Ollama 0.40.12, Python 3.12.3, `requirements-lock.txt` exact;
  `env_check --strict` passes; 246 tests pass, byte-reproducibility included). Canonical Scout ran in the earlier
  `recyclebench_ollama-0.40.12.sqsh`.
- **Models (Ollama IDs):** llama4:17b-scout-16e-instruct-q4_K_M `bf31604e25c2`, qwen2.5:14b `7cdf5a0187d5`
  (judge + inverter), qwen2.5:3b `357c53fb659c` (paraphraser), llama3:8b `365c0bd3c000`, llama3.2:3b `a80c4f17acd5`
  (others in the provenance files). Not checked against the Mac's blobs: no Mac digest is recorded.
- **Data:** sha256 advbench `8630c64e…`, harmbench `f46f7409…` (all 400 rows), strongreject `efadebf5…`; attack corpus
  liuyanchen1015/Llama-3.2-1B-Instruct_jailbreak_responses_with_judgment, 9,888 rows, revision `69224dce…`.
- **Config:** seed 20260101, k = 3, gate 0.75/0.85/0.60/0.50, `configs/llama4_pegasus.yaml` and the lineage-only
  variants in `configs/llama4_scout/`.
- **Jobs:** canonical 3683991 (H100, 4h49m); recycle 3690565 (B200, 3h38m); RPE 3690569 (H100, 1h45m); attack
  3690568 (cancelled after 4h40m, replies kept) + shards 3691839/41/43/45 (2 × A100-RP/H100, 3h23m–7h04m) + judge
  pass 3699489 (RTX 3090, 1h46m); report 3699490. ≈ 56 GPU-hours for the suite. Full list in `logs/scout_chain.txt`.
- **Regenerate the page:** `python3 tools/llama4_suite.py` then `python3 tools/llama4_dashboard.py --run-dir
  results/runs/20261031_1816_llama4_scout_canonical --suite <suite json> [--redact]` (CPU only; `PYTHONUTF8=1` on
  Windows). Lessons and failure modes: `docs/pegasus/LESSONS.md` §4b.

## Open

1. Copy `data/sorrybench.csv` and the lineage's frozen manifests from the Mac to `/netscratch` (chmod 600), then
   screen Scout on SORRY-Bench and re-screen it on the lineage manifests (RSR for every pair, PBR).
2. Label `audit/llama4_scout__*/judge_sheet.csv` before any corrected Scout figure.
3. Record the Mac's `ollama list` IDs to close the old-model identity question.
4. Maverick: 3,699 canonical replies in `cache/llama4_maverick.jsonl` (job 3689042) await merge + judge + report.
