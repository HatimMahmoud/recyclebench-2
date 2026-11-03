# External corpora on every model (Pegasus runbook)

This run takes GuidedBench (200), ScienceSafetyData (240) and JBB-Behaviors (100) through the canonical screen,
recycling (RSR), inversion (RPE) and style arcs. It covers all eight models, the six dense Llama models plus Llama 4
Scout and Maverick, over 19 release pairs.

- **Registration:** `PREREG_EXTERNAL_CLUSTER_20261103.md`. Sign it and commit before submitting.
- **Designation:** exploratory, with raw figures.
- **Single source of truth:** `tools/external_suite.py`. It holds the corpora, models, pairs, configs, specs,
  eligibility and the completion matrix.

## Before the first submission

1. **Code.** Get this branch onto Pegasus as a `git bundle` (LESSONS §4b.1). Check out the signed commit in `$RB_REPO`.
2. **Image.** Use the validated gate image, `recyclebench_gate_8fdd17c.sqsh`, which is the default of
   `external_submit.sh`. Code is mounted, not baked in, so no rebuild is needed.
3. **Models.** All eight tags in `configs/llama4_pegasus.yaml`, plus qwen2.5:14b and qwen2.5:3b, are already in
   `$RB_SCRATCH/ollama`. The gate's Hugging Face models are in `$HF_HOME`.
4. **Data.** The first job, D, checks `data/<corpus>.csv` against the registered fingerprints: row count plus a hash of
   the ordered request texts.
   - **ScienceSafetyData:** D builds it if absent, from the pinned recipe.
   - **JBB-Behaviors:** D builds it if absent, from Hugging Face.
   - **GuidedBench:** has no builder in the repo, so place `data/guidedbench.csv` (columns `Goal,Category`, 200 rows)
     yourself.
   - **On a mismatch:** D stops the chain (order matters: ids are row positions and seeds derive from them). The Mac's
     file is the registered one (file sha256 in the registration).
5. **Plan check.** Run `bash pegasus/external_submit.sh` on the login node. It prints the twelve `sbatch` commands
   and submits nothing.

## Run

```bash
bash pegasus/external_submit.sh --submit          # the whole chain, with explicit dependencies
squeue -u $USER                                   # job names rb-ext-*
```

| job | what | GPU shape | time asked |
|---|---|---|---|
| D | data fingerprints (+ JBB / ScienceSafety builders) | CPU | 0:30 |
| G-dense · G-scout · G-mav | canonical draws (k = 3), each into its own `cache/external/canonical__<set>.jsonl` | 80 GB · 80 GB · 2 × B200/H200 | 8 h · 6 h · 8 h |
| J | merge the three caches, judge every draw of every model, canonical report (all 19 pairs) | 80 GB | 5 h |
| R-dense · R-scout · R-mav | recycle, eligible pairs (judged \|P⁻\| ≥ 5) of each group | 80 GB · 1 card > 80 GB · 2 × B200/H200 | 8 h each |
| P-prep | invert every fix (old answer → 8 candidate requests) and gate every candidate | 80 GB | 6 h |
| P-loop · P-mav | closed loop for dense + Scout pairs · Maverick pairs | 80 GB · 2 × B200/H200 | 8 h · 6 h |
| F | style arcs, inversion reports, completion matrix (runs even if an arm failed) | CPU | 1 h |

**Why the shapes differ** (LESSONS §4b.6). Recycling judges inline beside the target. Scout (67 GB) plus the judge
therefore needs more than 80 GB on one card. Maverick (245 GB) needs two large cards. Generation and the inversion arm
load one model at a time, so 80 GB is enough there, except for Maverick.

The time requests come from the Scout suite's measured rates: 3 s per Scout call, 6.6 s on long answers, about 10 s with
an inline judge. They are upper bounds, not estimates. Every step resumes, so a timeout costs only a resubmission.

**One corpus at a time.** `RB_CORPORA=sciencesafety bash pegasus/external_submit.sh --submit` runs the same chain
for a subset of the registered corpora; every job inherits the variable. The matrix still lists all three, so a
corpus left out shows as PENDING or BLOCKED. A later run of the remaining corpora reuses the same caches (keys
differ by prompt) and writes a new dated canonical report.

## Check, resume, finish

- **Progress.** `results/runs/external/matrix_<stamp>.md` (job F, or `python3 tools/external_suite.py matrix` at any
  time; CPU, no model calls). Every corpus × pair × arm cell gets one of five statuses:
  - COMPLETE: verified;
  - PENDING: executable, not run yet;
  - NOT APPLICABLE: e.g. |P⁻| below the floor, or an undefined PRR;
  - BLOCKED: a prerequisite is missing;
  - FAILED: needs repair.

  The run is finished when no cell is PENDING or FAILED.
- **Resume after a timeout or a fix.** Resubmit only what is left, for example:

  ```bash
  bash pegasus/external_submit.sh --submit --only R-scout,F
  ```

  A dependency on a job that is not resubmitted is dropped, so don't resubmit a job whose prerequisite is still running.
- **Rules that hold throughout:**
  - **One job per output:** never run two jobs that write the same output. The submitter gives every job its own cache
    and directory.
  - **No overwrites:** nothing overwrites. Reports, matrices and closed-loop files are dated, and the driver refuses a
    resume whose fingerprint (data, config, seed, budgets) moved.
- **What stays local.** Everything lands in `cache/external/` and `results/runs/external/`. Both are gitignored, because
  they hold replies to harmful requests. Bring back the matrix and the canonical report (counts only). Bring back the
  rest with `rsync --ignore-existing`, never by committing it.

## Not in this run

These are listed in the matrix's static rows, with reasons:
- known-attack transfer;
- the ScienceSafetyData `Attack/` split;
- compose v2;
- the mechanistic probe;
- model-level PBR;
- human calibration (sheets are exported to `audit/external__*/`, unlabelled);
- the 70B models.
