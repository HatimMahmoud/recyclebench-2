# What is left, and who can do it

Updated 2026-10-31 21:30 (the 2026-10-24 version is in git history). Everything automatable has been run; this file holds
only what needs a person, plus the optional cluster work.

---

## A. Needs you

### Done
- A1. Sign prereg Amendment 1 — candidate 2026-10-25, supervisor 2026-10-28; the confirmatory replication has run.
- A2. `audit/rpe/rpe_harm_sheet_20261028_2327.csv` — 65/65 blind, anchor PRR 36/38 verified.
- A3. `audit/gate/gate_sheet.csv` — 178/178 (rows 19–178 model-assisted): 3/98 re-opening rewordings changed the request.
- A4. `audit/harmbench/judge_sheet.csv` — 196/196 blind: se 0.865, sp 1.000, κ 0.933; supports the calibration's transfer.
- A5. Attack-transfer audit of judge-negatives — **sheet exported** 2026-10-31 as a class-stratified audit (A7, last line).
- A6. The anchor manifest — a subset-CSV id join error; reproduced, nothing to decide.

### A7. Label the five registered sheets (Amendment 2 §B4, hashes frozen before any label)
```
python3 tools/judge_label_tui.py audit/gate_rejection/rejection_sheet.csv                 # 130 rows  §B4.1 gate false rejection
python3 tools/judge_label_tui.py audit/overrefusal/benign_sheet.csv                       #  50 rows  §B4.2 judge on benign prompts
python3 tools/judge_label_tui.py audit/sorrybench/judge_sheet_blind_relabel_20261030.csv  # 280 rows  §B4.3 blind re-label
python3 tools/judge_label_tui.py audit/gate_recheck/recheck_sheet.csv                     #  40 rows  §B4.4 gate re-check
python3 tools/judge_label_tui.py audit/attack_classes/attack_sheet.csv                    # 116 rows  §B4.5 attack transfer
```
All five are blind (no judge or model guess shown). On the attack sheet, a reply you cannot judge: skip it (`s`) and write
"cannot tell" in its `note` column; a blank without a note counts as not yet labelled. If the blind re-label (280 rows) is
completed, it becomes the primary calibration for every corrected figure (§B4.3) — mechanically, through
`tools/primary_calibration.py`.

Then, with no model calls:
```
python3 tools/after_labels.py          # plan: what is labelled, what would run
python3 tools/after_labels.py --run    # score, apply §B4.3, recompute, rebuild -- every output new and dated
```
and ask Claude to write the results into Amendment 2 and THESIS_KB.

### A10. ScienceSafetyData (BYOD) — PARKED for the cluster (2026-11-01)
**Not an open local task.** The corpus will be re-run on the cluster together with several other
external datasets, so nothing more happens locally and it stays out of `study_data` and off every
dashboard until then. The local run below is a completed, isolated pilot kept for reference; the
reusable part is the adapter (`tools/prepare_external.py`, `docs/BYOD.md`), which is what the cluster
run will use. Of the three decisions below, only the labelling (1) would still be needed, and only if
the local figures are ever cited — on the cluster the corpus gets its own calibration.

<details><summary>Local pilot, for reference</summary>

An external corpus the supervisor asked to explore: 240 dual-use science research requests
(`ulab-uiuc/SafeScientist`, pinned `9db8659`, MIT), registered in `PREREG_SCIENCESAFETY_20261101.md`
(+ `_RPE_` for the inversion arm) and run **isolated** overnight 2026-11-01.
```
bash run_sciencesafety_20261101.sh --list        # stage status + |P^-|
bash run_sciencesafety_rpe_20261101.sh --list    # follow-on arms
cat results/runs/sciencesafety_20261101/META.txt
```
**Result (RAW, 2026-11-01):** judged P⁻ = 23/240, RSR 13/23 = 0.565 [0.363, 0.768], McNemar p = 2.4e-4.
Under-powered (23 < 30), and the summary's "PBR headline" is over those same 23 behaviours, so it is
numerically RSR, not an independent figure. RPE: the gate admitted **0 of 183** inversion candidates
(cosine 101, NLI 82), so A_gate is undefined and only the unclaimable A_loop upper bound (15/21) exists
— registered threat T2 measured, not assumed. Full numbers in the run's `META.txt`.

Three things need you:
1. **Label the blind sheet** `audit/sciencesafety/`. **Stage 40 did not produce it** — `--out` was
   given a directory. The queue is fixed and the marker cleared, but re-running it costs ~127
   `qwen2.5:14b` judge calls (zero target calls), so it needs your go-ahead. This is registered threat
   **T1** and it is the one that matters: these prompts have a legitimate research reading and the judge
   scores *attempt at the request*, so `breaks_old` may be inflated and some "fixes" may not be fixes.
   **Every figure from this corpus is raw and T1-unresolved until this is labelled.**
2. **Decide whether to fold the corpus into the study.** It is in no dashboard and not in
   `tools/study_data.py` by design. Folding it in needs a non-empty P⁻ *and* an amendment — not a
   silent registration.
3. **Decide on the style factorial.** Not run and not registered for this corpus: it needs a power
   statement for whatever |P⁻| turned out to be. The plumbing exists
   (`tools/run_compose_v2.py --behaviours`); the registration does not.
</details>

### A8. Sign Amendment 2 (`prereg_amendment_2_DRAFT.md`), supervisor acknowledges
After signing, the dashboards switch to the registered RSR denominator (`lineage_core.RSR_DENOMINATOR = "resolved"`, anchor
23/46) and are rebuilt — ask Claude.

### A9. Decisions with the supervisor
- Release text policy (`tools/release_c2.py`: no rewording text by default; `--text non-reopening` only after this decision).
- The disclosure draft `docs/disclosure/meta_disclosure_DRAFT.md` (not sent).
- Whether the causal white-box result (B3) may be cited outside the appendix.

---

## B. Optional, needs the cluster (not required for the thesis)

### B3. ~~Causal white-box interventions~~ — done 2026-10-31, locally
Forward passes only, so the M1 sufficed (`tools/mech_causal.py`, registered in `PREREG_MECH_CAUSAL_20261031.md`). On the
anchor pair: moving each re-opening rewording up llama3.1-8b's refusal direction raises P(refusal opening) more than a
matched random move in 23/23 fixes, and moving the canonical request down lowers it in 23/23 (both p = 2.4e-7); full
restoration in 5/53 rewordings, so the direction carries part of the effect. THESIS_KB §4.5.

### B4. White-box beyond the anchor
`tools/mech_probe.py` and `tools/mech_causal.py` cover llama3-8b → llama3.1-8b on SORRY-Bench only. Extending them needs a
`--pair` option and GGUF links for the other models. Each 8B run needs about 16 GB of unified memory: on the M1 it swaps
heavily, on the cluster it is routine.

### B5. Llama-3.1-70B as a lineage target
Would answer the scale-confound criticism of the non-anchor rungs (THESIS_KB §6: the study runs 2 of the memo's ≈ 5
matched-scale comparisons). Not possible on the M1. Optional — scope creep unless the thesis needs it.

### B2. Before trusting any cluster number
Every study reply came from Ollama (Q4 quantisations, repeat penalty 1.1 over the last 64 tokens). A bf16 model under vLLM
defaults is a different model: reproduce quantisation and decoding, and validate against cached replies first (as
`tools/mech_probe.py` does). Locally the quantisation mismatch is bounded (THESIS_KB §7.8). A shared cluster is a different
exposure profile for a harmful-prompt corpus: keep it in a restricted path and check with the administrator (`ETHICS.md`
assumes local-only).

---

## Parked (not on the to-do list)
PTP: appendix only, later, on the cluster (`PTP_FINDINGS_20261027.md`; the earlier plan is in git history).

## C. Formerly stale — all resolved
Gate layer L2 complete and shown; llama2-13b PBR on AdvBench / StrongREJECT filled; the lineage pair matrix rebuilt from the
judged verdicts; the non-adjacent matched-scale pairs are computed in `study_data` (`EXTRA_PAIRS`, 2026-10-29), so the master,
pair and lineage dashboards agree.
