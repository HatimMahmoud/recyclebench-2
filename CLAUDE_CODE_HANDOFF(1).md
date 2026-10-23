# RecycleBench — Full Project Handoff (Claude Code)

**Written:** 2026-10-21 · **For:** continuing RecycleBench in Claude Code with full repo access.
**Author:** the prior Claude (cloud session, no repo access — the reason for the move).
**Scope:** the entire project — thesis framing, methodology, implementation, every experiment, and open work.

---

## 0. How to continue in Claude Code (do this first)

macOS (M1 Max, 32 GB). Needs a Claude Pro/Max/Team/Enterprise or Console account.

1. **Install** (native installer, recommended):
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   ```
   (Alt: `brew install --cask claude-code`, or `npm install -g @anthropic-ai/claude-code` with Node 22+.)
   New terminal → `claude --version`, optionally `claude doctor`.
2. **Put this file in the repo root:**
   ```bash
   cd ~/Downloads/recyclebench\ 2/ && cp /path/to/CLAUDE_CODE_HANDOFF.md .
   ```
3. **Launch** from inside the repo: `claude` (first run = browser login).
4. **Persist context:** run `/init` to generate a `CLAUDE.md`, then paste §12 (Conventions) + §13 (Gotchas)
   into it. Keep this handoff as the one-time full brief.
5. **First prompt (suggested):**
   > Read `CLAUDE_CODE_HANDOFF.md`. Then read `tools/make_dashboard.py` and locate `build_view`,
   > `partition`, `filterBench`, and how section modules are injected. Start with the ★ open task in
   > §14.1 (native `--lineage` mode). Show me the diff before writing; don't hit Ollama without asking.

---

## 1. The thesis

**Title:** *Security of Open-Weight LLMs — Measuring Cross-Generational Robustness to Paraphrase
Recycling of Jailbreaks.* An intent-gated jailbreak-paraphrase recycling protocol, evaluated on the
Llama lineage.
**Candidate:** Hatim Mohammed Arhoumah Mahmoud (M.Sc. CS, Intelligent Systems, RPTU Kaiserslautern-Landau,
part-time; employed at IBM). **Supervisor:** Islam Mesabah, M.Sc. **Examiner:** Prof. Dr. Sebastian
Vollmer. In collaboration with IBM Frankfurt. Exposé dated 2026-06-18.

**One-paragraph thesis.** The security of a deployed LLM rests on the assumption that its safety
alignment *generalises* — that a model refuses on the basis of harmful **intent**, not **surface form**.
RecycleBench treats that assumption as a directly measurable property. Guiding question (cross-
generational): *when a newer Llama generation blocks a jailbreak an older one answered, did it learn the
intent, or just the wording it was shown?* We isolate **P⁻**, the regression set (prompts the OLD model
answered and the NEW model refuses — the fixes), and measure how much of P⁻ a **benign, intent-preserving
paraphrase** re-opens (**RSR**). High RSR ⇒ surface-level patching; low RSR ⇒ consistent with genuine
generalisation. A deliberately **weak attacker** (fluent reword, not adversarial suffix), so recovery is a
conservative lower bound on shallowness. The contribution is **measurement and integration**, not a new
attack/defence — assembling established controls (intent gate, calibrated judge, judge-error correction)
at a matched unit of analysis, pre-registered, into one released instrument and a corrected estimate.

**Research questions.**
- **RQ1 (primary — recycling):** can the M_n→M_{n+1} safety improvement be bypassed by meaning-preserving
  paraphrase? Estimate RSR over P⁻ per pre-registered pair and pooled, 95% CI, McNemar's exact test.
  Headline only where the pilot shows |P⁻| adequate; else exploratory + PBR carries weight.
- **RQ2 (interpretation):** do recycled successes reflect memorisation or intent generalisation? Reported
  per harm category to locate brittleness. Neither direction claimed as mechanistic proof.
- **RQ3 (recycling drivers):** which paraphrase classes most recover patched jailbreaks? Mixed-effects
  logistic regression of per-variant recovery on transform class + quantified surface distance, random
  intercept per behaviour. (This is the statistical backbone of the "style arcs" dashboard section and the
  factorial-composition idea in §10/§14.5.)
- **RQ4 (backbone / instrument validation):** how paraphrase-robust is each model within itself? PBR over
  the always-large own-refusal set, vs a pre-registered reference; the reportable floor when P⁻ is small.

---

## 2. The three arms + metrics

**Arm 1 — Recycle (primary).** Over P⁻, apply benign intent-preserving paraphrases, re-test the new model.
- **RSR** = fraction of P⁻ re-opened by ≥1 gated paraphrase. `RSR = |{p∈P⁻ : ∃p'∈G(p), M_{n+1} answers p'}| / |P⁻|`. Robust-fix rate = 1−RSR.
- **PBR** = of a model's own currently-refused set C_M, fraction re-opened. Always estimable → backbone/floor.
- **Intent gate (ANDed):** cosine ≥ 0.75, BERTScore ≥ 0.85, NLI-entail ≥ 0.60, LLM-intent ≥ 0.50.
- **6 transform classes:** `paraphrase, syntactic, lexical, back_translation, tense_shift, register`.
  8 paraphrases/behaviour. Paraphraser `qwen2.5:3b` (decorrelated). We standardized these this project.

**Arm 2 — RPE (output-inversion).** Reconstruct the request from the OLD model's compliant output, gate it,
re-test the new model. Convergent cross-check that the effect isn't a paraphraser artifact; standalone,
it's refusal-coverage auditing. Inverter `qwen2.5:14b`.

**Arm 3 — Attack-transfer.** Known adversarial attacks (Wei et al., 31 classes), **not** intent-gated —
a distinct, stronger threat model. Headline = transfer-on-refused. **Key finding:** judge against the
*original* intent; scoring encoded outputs against encoded prompts yields spurious bypasses that vanish
under intent-faithful judging.

---

## 3. Rigor

- **P⁻ floor** `min_p_minus = 30`: below → exploratory, PBR carries headline.
- **Judge.** *Proposal* specifies an **ensemble** (StrongREJECT fine-tuned evaluator + HarmBench-style
  classifier + Llama Guard 3), reused not built, calibrated to human labels. *Implementation* currently
  uses a single **decorrelated rubric judge `qwen2.5:14b`** (temp 0). **⚠ Reconcile this** in the write-up
  (the harness/report should either adopt the ensemble or justify the single-judge substitution). Judge
  verdicts not cached.
- **Rogan–Gladen correction:** π̂ = (p+sp−1)/(se+sp−1), from human-audited se/sp; admit only κ ≥ 0.60.
  Correction + calibration at the **behaviour-bundle** level (a behaviour counts once if any admitted
  variant succeeds; per-response specificity compounds to sp^t across t admitted variants, so bundle-level
  is the correct unit). Anchor judge: se 0.862, sp 0.967, κ 0.830.
- **CIs** = parametric bootstrap at bundle level; **prereg** / **prereg-verify** freeze design & detect drift.
- **Gate audit** (lineage run): 3,520 variants, 1,851 admitted, false-admission **0.0**, admitted-but-
  changed **0** — the rebuttal to "your paraphrases changed the ask."

---

## 4. Experiment space & lineage design (⚠ important nuance)

From the experiment-count memo (Appendix A): the instruct-only Llama family has **G=6** generations, sizes
s=(3,2,3,2,1,2), **N=13** models (15 with the two Llama-3.2-Vision). Pairing counts:
- All cross-generational ordered pairs **E_cross = 69** (91 with Vision) — the full combinatorial space.
- All ordered pairs incl. within-generation **E_all = 78**.
- Adjacent-generation only **E_adj = 22** (30 with Vision).

**The scale confound.** E_cross mixes generation with *size* (e.g. 7B→405B is a ~60× capacity jump), and
date-ordering can place a 1B model in the "newer, safer" role the estimand assumes. The recycling construct
is only well-posed when M_new is plausibly the stronger, better-aligned model. The memo therefore
recommends **matched-scale generational chains**:
- **7–8B chain:** Llama-2-7B → Llama-3-8B → Llama-3.1-8B (2 consecutive pairs)
- **70B chain:** Llama-2-70B → Llama-3-70B → Llama-3.1-70B → Llama-3.3-70B (3 consecutive pairs)

≈5 matched-scale pairs = the pre-registered minimum-viable scope. Cost: `calls ≈ E·B·k·(1+c̄+p̄t̄)` → even
≈5 pairs, B≈1000, k=3 is tens of thousands of calls. Recommendation: **report E_cross=69 as the space,
pre-register the ≈5 matched-scale subset.**

**⚠ What we actually ran vs. the recommended design.** The overnight lineage was
`llama2-7b → 2-13b → 3-8b → 31-8b → 32-1b → 32-3b` — an **adjacent-step breadth** chain that is **not**
matched-scale: it includes within-generation pairs (2-7b→2-13b, 32-1b→32-3b) and puts small models (1B, 3B)
in the "newer" role (31-8b→32-1b). That is exactly the scale confound the memo warns about — and it's
**why only the matched-scale anchor (3-8b→31-8b, both 8B) is firm and the other four rungs are
exploratory** (small P⁻ *and* confounded). **For the thesis:** the clean story is the 7–8B matched-scale
chain (2-7b→3-8b→31-8b) + the anchor; present the full six-model run as breadth with the confound stated.
The **70B chain is not run** (no local hardware; 70B/405B/MoE are stretch/GPU-rental/out-of-scope on the
M1 Max). Don't attempt 70B locally.

---

## 5. Repo layout & CLI

```
~/Downloads/recyclebench 2/
  src/recyclebench/
    datasets.py     # loaders (load_jailbreakbench)
    screening.py    # builds C_M and P⁻; _draw_seed
    realrun.py      # the recycle run; _key (cache key)
    gate.py         # intent gate G (4 checks)
    judge.py        # judge (+ Llama Guard in proposal); looks_like_refusal
    metrics.py      # RSR, PBR, Rogan–Gladen correction
    config.py       # Config.load
  tools/
    paraphrase_one.py     # the paraphrase operator T
    make_dashboard.py     # single-pair HTML dashboard (★ edit target, §14.1)
    inspect_cache.py      # cache observability
    build_all_dashboards.py
    calibration_section.py attack_transfer_section.py rpe_dashboard_section.py
    style_arcs_section.py positioning_section.py      # dashboard section modules
  configs/   # recycle_3_31.yaml; real_llama_lineage_all.yaml; recycle_27_213.yaml,
             #   recycle_213_38.yaml, recycle_318_321.yaml, recycle_321_323.yaml (cloned, not all run)
  data/      # sorrybench.csv, harmbench.csv, advbench.csv, strongreject.csv (+ JBB, DAN per proposal)
  cache/llama_gen.jsonl   # {"k": sha256, "t": response} — NO model field
  results/runs/
    recycle_3_31_sorrybench/      # calibrated ANCHOR (manifest_sorrybench.jsonl, corrected_*.json)
    lineage_sorrybench_20261020/  # overnight lineage — only manifest.jsonl (+ dated dashboards)
    attack_transfer/              # llama31-8b_judged_original.json, llama31-70b_judged.json
    rpe_3_31_sorrybench/
```

Models (lineage order): `llama2-7b → llama2-13b → llama3-8b → llama31-8b → llama32-1b → llama32-3b`.
Benchmarks: AdvBench, HarmBench, JBB-Behaviors, StrongREJECT, SORRY-Bench (primary), DAN/in-the-wild.

**CLI (`recyclebench …`):** `pilot-pminus` (size \|P⁻\|, cell CSVs) · `run-real` (full real run; manifest,
RSR+PBR; derives pairs from config `lineage:`) · `run` (simulated) · `pilot` · `manifest` ·
`judge-export`/`judge-ingest` (→ se/sp, κ, admit) · `gate-audit`/`audit-export`/`audit-ingest` ·
`validate` (MC coverage) · `prereg`/`prereg-verify` · `figures`/`version`.
Config blocks: `seed`, `lineage`, `real`, `transform_classes`, `paraphraser`, `judges`, `gate`,
`budget` (k_canonical_samples=3, paraphrases_per_behaviour=8), `decision` (min_p_minus=30, kappa_floor=0.60).
**All configs in a lineage must share identical `judges`/`gate`/`transform_classes`.**

---

## 6. Dashboard system

`make_dashboard.py` (single pair) builds the 2×2 / Sankey flow from the **cache** for `--old/--new`; the
manifest supplies admitted paraphrase cards. It **injects section modules** before `</body>` (calibration
at a `<!--CALSLOT-->` marker in `.wrap`). Section modules are self-contained `<style>`+`<section>` fragments
scoped under `.cal-root`/`.atk-root`/`.sa-root`/`.pos-root`/`.rpe-root`.

**Current section layout** (after merging 01+02): `01` Overview & calibrated headline · `02` Results ·
`03` By benchmark · `04` RPE · `05` Style arcs · `06` Attack · `06·b` 70B · `07` Positioning. There's a
**scroll ⇄ tabs** icon toggle (`collectBlocks()` / `activate(k)` / `setView`). The per-benchmark view
system — `build_view(...)` renders one full view in `<div class="bview" data-view="{name}">`, `filterBench(t)`
toggles them — is the mechanism to reuse for §14.1.

> ⚠ The `make_dashboard.py` I had in the cloud was an **older** copy (03/04/05 numbering, no toggle). Your
> local file is the source of truth — read it before editing.

Fixes made this project: `attack_transfer_section.py` `_pct`→`f"{100*x:.1f}%"`, bar width absolute;
`calibration_section.py` reads `corrected_*.json` (`d.RSR`, `d.PBR_new`, `d.judge.{se,sp,kappa,…}`);
`style_arcs_section.py` concentric arcs over transform_class × nli_entail. `build_all_dashboards.py` =
batch (one dated dashboard per `results/runs/*`, `PAIR_OVERRIDE` for odd names). `dashboard_standalone.html`
= strict client-side viewer; must not replace the make_dashboard path.

---

## 7. Experiments & state

- **Pilot (3-8b → 3.1-8b, pooled AdvBench+HarmBench+StrongREJECT):** |P⁻|=33, recovered 18, **RSR 0.545**
  (clears floor; HarmBench arm McNemar p=6e-5). This is the slides' "first result." **Done.**
- **Anchor (3-8b → 3.1-8b, SORRY-Bench):** |P⁻|≈47 (49 on lineage re-screen), **RSR ≈0.47–0.49 raw /
  0.53–0.55 corrected**, McNemar p=2.4e-7. Judge admitted (se 0.862/sp 0.967/κ 0.830). The deep calibrated
  case; the report's headline. **Done.** *(Note: pilot and anchor are the same pair on different
  benchmarks — both ~0.55, don't conflate.)*
- **Attack-transfer (3.1-8b):** calibrated se 0.857/sp 1.0/κ 0.833 (admitted); all 31 attack classes
  transfer ≤~5% of refused under intent-faithful judging. **Done.**
- **llama32jb dataset:** anchor gives |P⁻|=3 (old refuses 293/300); non-discriminating → documented
  negative, not an RSR. **Done.**
- **SORRY-Bench lineage (5 adjacent pairs):** overnight run. **Done (RAW).** See §4 for the design caveat.
- **Per-pair lineage configs** cloned but not individually run (the run used `real_llama_lineage_all.yaml`).

---

## 8. Results (lineage, RAW — corrected via shared anchor judge se 0.862 / sp 0.967)

| Release pair | P⁻ | re-op. | RSR raw | RSR corr | McNemar p | status |
|---|--:|--:|--:|--:|--:|---|
| llama2-7b → 2-13b | 18 | 15 | 0.833 | 0.97 | 6.1e-5 | exploratory (within-gen) |
| llama2-13b → 3-8b | 3 | 3 | 1.000 | 1.00 | 0.25 | exploratory (tiny) |
| **llama3-8b → 3.1-8b** | **49** | **23** | **0.469** | **0.53** | 2.4e-7 | **anchor / matched-scale / firm** |
| llama3.1-8b → 3.2-1b | 12 | 5 | 0.417 | 0.46 | 0.0625 | exploratory (scale confound) |
| llama3.2-1b → 3.2-3b | 13 | 8 | 0.615 | 0.70 | 0.0078 | exploratory (within-gen) |
| **Pooled** | **95** | **54** | **0.568** | **0.65** | — | lineage headline |

PBR (own-refusal, raw) rises across generations: 2-7b 0.126 · 2-13b 0.101 · 3-8b 0.197 · 3.1-8b 0.168 ·
3.2-1b 0.196 · 3.2-3b 0.231.

---

## 9. Thesis structure (from the exposé)

- **Pipeline P1–P4:** P1 behaviour set + judge reuse/calibration + pilot P⁻ · P2 gated paraphrase operator
  T · P3 calibrated low-bias evaluation (RSR/PBR) · P4 statistics & sensitivity. Offline, deterministic,
  cached, frozen stimuli.
- **Contributions:** **C1** pre-registered human-calibrated paraphrase-robustness instrument + bundle-level
  judge-error-corrected PBR. **C2** released dataset of intent-verified paraphrase variants with an audited
  gate. **C3 (primary)** the cross-generational recycling study (RSR over P⁻). **C4 (backbone)** corrected
  re-measurement across generations — the *gap* between naïve and corrected estimates. Stance: no
  methodological-novelty claim; measurement + integration at a matched unit, pre-registered.
- **Tiered deliverables (completion never depends on an outcome):** **L0** validated instrument + dataset
  (done even if nothing is bypassed) · **L1** any PBR value is an informative security result · **L2**
  stretch (more lineage pairs, largest-model runs). **Minimum viable thesis:** instrument + complete PBR on
  ≤8B models, single consumer GPU.
- **Timeline (6 months):** M1 protocol frozen · M2 instrument v1 · M3 core result secured · M4 analysis
  complete · M5 stretch/robustness · M6 submission. Core result secured by M3–4; stretch is droppable.
  Two mid-term presentations planned with Prof. Vollmer.
- **Prior-art positioning (hedged novelty):** Qi et al. "shallow" = *token depth* (orthogonal to our
  surface-vs-intent axis); Surface-Form Sensitivity [31] = within-model variance, nearest neighbour but no
  release pair / no P⁻; Mitra [38] = cross-gen transfer on **Gemma** but replays attacks **verbatim**
  (can't separate "learned intent" from "memorised string" — reformulation is exactly what does); leaderboards
  rank static safety, none uses the release pair as the unit. Claim: no public artifact centres on
  intent-preserving recovery over the adjacent-release regression set P⁻.

---

## 10. Supervisor threads

- **Vollmer's evolutionary idea → our factorial counter-proposal.** He suggested an *evolutionary* approach
  (combine two paraphrase classes, fold RPE into the loop with fitness selection). We recommended a **fixed
  factorial composition** instead: 21 conditions = 6 singles + C(6,2)=15 pairs, **no fitness selection**,
  **RPE kept decoupled** — avoids a scoop risk (cf. Mitra), preserves the weak-attacker framing, the gate,
  RPE independence, and clean calibration. Write-up: `Composition_Ablation_Memo.pdf`. **Pending greenlight.**
  (This is the natural RQ3 extension — compounding surface axes.)
- **Documentation requests:** disliked a usage-manual format → wanted it **research-oriented** →
  `RecycleBench_Report` (two-column paper). Also a developer `WALKTHROUGH.md`/`.html`.

---

## 11. Deliverables produced (cloud outputs; import what you need)

- **Proposal set (current):** `expose_v6.pdf` (full exposé), `slides_expose_v6_1.pdf` (defense slides),
  `CrossGen_Experiment_Count_Memo.pdf` (Appendix A — experiment-space sizing).
- `RecycleBench_Report.pdf`/`.tex` — 4-page research report incl. the lineage table, pooled headline, gate
  audit. **Current.** (`RecycleBench_Documentation.pdf` = earlier usage manual, superseded.)
- `Composition_Ablation_Memo.pdf`/`.tex` — the factorial composition counter-proposal.
- `WALKTHROUGH.md`/`.html` — developer docs.
- `build_lineage_dashboard.py` (iframe lineage wrapper — **superseded by §14.1**, reference only) +
  `dashboard_lineage_sorrybench_*.html` (hand-built summary, not make_dashboard output).
- Dashboard system edits (section modules, `build_all_dashboards.py`, `dashboard_standalone.html`).

---

## 12. Conventions & invariants (paste into CLAUDE.md)

- **Never overwrite dashboards/HTML** — new **dated** file per run (`…_<YYYYmmdd_HHMM>.html`).
- **Cache record** `{"k": sha256, "t": response}` — no model field; model hashed into `k` via
  `_key(model, prompt, _draw_seed(bid, index, draw))` (`index=-1` canonical). Don't grep the cache for model
  names (matches response text, not keys). Judge verdicts not cached. `--config` supplies only `k` (=3) and
  `seed`; a different `seed` changes canonical draws → cache misses.
- **Intent gate (ANDed):** cosine ≥0.75, BERT ≥0.85, NLI ≥0.60, llm_intent ≥0.50. **P⁻ floor 30.**
- **Judge correction:** Rogan–Gladen, bundle level, admit κ ≥0.60. Anchor judge se 0.862/sp 0.967/κ 0.830.
- **Lineage configs share** identical `judges`/`gate`/`transform_classes`. Section modules self-contained,
  injected before `</body>`.
- **Deliverable prefs:** compiled/runnable artifacts and PDFs; send only changed files (no routine zips).

---

## 13. Gotchas (these bit us)

- **zsh word-splitting:** user's shell is zsh; `set -- $var` / unquoted `$var` don't split. Use
  `"${a}:${b}"` + `${pair%%:*}`/`${pair##*:}`, or an array loop.
- **Ollama swap-thrashing:** default `OLLAMA_MAX_LOADED_MODELS=1` reloads a ~10 GB model every judge/target
  alternation (~1 variant/30 s). Run `OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=1h`; `caffeinate -i`,
  lid open, plugged in. **A run at 0% CPU may be swap-thrashing, not hung** — check `llama_gen.jsonl`
  growth + `ollama ps` before killing. **Never double-launch** `run-real` (two writers corrupt cache/manifest).
- **Dashboard rung prints `P^-=0`** → use `--config configs/real_llama_lineage_all.yaml` (carries the seed).
- **Lineage folder isn't auto-dashboarded** (manifest is `manifest.jsonl`, no pair in name, single-pair tool).
- **No 70B/405B/MoE locally** — M1 Max 32 GB. Those are stretch/GPU-rental/out-of-scope.

---

## 14. Open tasks

### 14.1 ★ Native `--lineage` mode in `make_dashboard.py` (immediate)
One SORRY-Bench page, five pairs as **native tabs** in house style — no iframes. Reuse `build_view` +
`filterBench`: (1) add `--lineage` (read ordered `lineage:` from `--config`) or `--pairs a:b,…`; (2) per
pair `partition(cache,[sorrybench.csv],old,new,k)` → cells, compute recovered/rsr (same loop as per-bench
sub-views, via `after_state`), `build_view(...,name=f"{old}→{new}",active=(i==0))`; (3) tab bar reusing
`.btabs/.btab` with `data-t="{old}→{new}"` (existing `filterBench` works unchanged); (4) shared
`--manifest results/runs/lineage_sorrybench_20261020/manifest.jsonl`. **Accept:** one file, 5 native tabs,
per-tab P⁻ = **18, 3, 49, 12, 13**. Watch element-id collisions (follow the multi-view pattern).

### 14.2 Teach `build_all_dashboards.py` about lineage folders
`manifest.jsonl` (no suffix) → fan out over the config's consecutive `lineage:` pairs.

### 14.3 Retire the iframe wrapper
Delete `build_lineage_dashboard.py` once §14.1 lands.

### 14.4 Reconcile judge design (proposal ensemble vs implemented qwen2.5:14b)
Decide and document: adopt the StrongREJECT+HarmBench+Llama-Guard ensemble, or justify the single
decorrelated judge. Keep report and exposé consistent.

### 14.5 (Deferred) Formal lineage judge calibration
`judge-export → hand-label → judge-ingest` on pooled lineage bundles, if a lineage-specific se/sp is wanted
over the shared-judge correction (4/5 rungs exploratory → shared-judge is proportionate).

### 14.6 (Deferred, pending Vollmer) Factorial composition ablation
21-condition fixed factorial (6 singles + 15 pairs), no fitness selection, RPE decoupled — per
`Composition_Ablation_Memo`. This is the RQ3 driver extension.

### 14.7 (Design) Matched-scale framing
When writing results, lead with the matched-scale 7–8B chain (2-7b→3-8b→31-8b) + anchor as the clean story;
present the full six-model lineage as breadth with the scale confound stated (§4).

---

## 15. Suggested order of work
1. §14.1 native `--lineage`, verify per-tab P⁻ = 18/3/49/12/13.
2. §14.2 build_all lineage awareness. 3. §14.3 delete the iframe wrapper.
4. §14.4 judge reconciliation; fold into `RecycleBench_Report`.
5. Deferred: §14.5 / §14.6 on Vollmer's call; §14.7 at write-up time.
