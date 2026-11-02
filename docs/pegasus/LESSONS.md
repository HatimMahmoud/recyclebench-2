# Running recyclebench on Pegasus: what we learned

Written 2026-11-01 after the first Pegasus runs (image build, model pulls, the Llama 4 Scout canonical screen,
the first Maverick attempt). Audience: whoever cleans up the code so that cluster runs work without hand-holding.
The how-to is [pegasus/README.md](../../pegasus/README.md); the cluster documentation, condensed, is in [kb/](kb/README.md).

## 1. What exists

| File | Role |
|---|---|
| `docker/Dockerfile` | `ollama/ollama:0.40.12` + venv with `requirements.txt`, `openai`, `pytest` (`WITH_GATE=1` adds the gate's torch stack). Repo is mounted, not copied. |
| `.dockerignore` | build context = `requirements.txt` only (no cache/data/results, which hold harmful text) |
| `.gitattributes` | LF for `*.sh`, `*.sbatch`, `Dockerfile` in a Windows checkout |
| `pegasus/env.sh` | every path and Ollama setting, sourced by all scripts; override `RB_*` before sourcing |
| `pegasus/build_image.sh` | submits `podman build` + `enroot import` as an sbatch job (~2 min) → `$RB_IMAGE` |
| `pegasus/run.sbatch` | runs any repo command in the image: `sbatch [resources] pegasus/run.sbatch <cmd…>` |
| `pegasus/in_container.sh` | inside the job: start `ollama serve` on a per-job port, wait, write provenance, run the command, stop Ollama |
| `pegasus/llama4_canonical.sh` | generate → judge → report for one model (`RB_MODEL`), steps selectable (`RB_STEPS`), side cache (`RB_CACHE`) |
| `pegasus/speed_check.sh` | one load + timed benign prompts: load seconds and steady tokens/s for a model on a GPU |
| `configs/llama4_pegasus.yaml` | the six-model lineage config + `llama4-scout` + `llama4-maverick`; seed/judge/gate identical, so cached responses hit |
| `tools/llama4_report.py` | canonical 2×2 of every earlier lineage model → `--new` model, from cache + verdict stores |
| `tools/llama4_dashboard.py` | one HTML page for a Llama 4 canonical screen (judged), same design system as the lineage dashboard |
| `src/recyclebench/backends/providers.py` | `resolve_base_url`: `RECYCLEBENCH_OLLAMA_URL` replaces `http://localhost:11434` (tested in `tests/test_real_backend.py`) |
| `tools/prepare_behaviours.py --keep-duplicates` | HarmBench must keep all 400 rows (ids are row positions) |

## 2. The run flow

```
login node:  sbatch …                         (nothing else runs here)
  └─ run.sbatch  ─ srun --container-image=$RB_IMAGE --container-mounts=/netscratch,/ds:ro,$RB_REPO
       └─ in_container.sh ─ ollama serve on 127.0.0.1:(20000 + jobid % 20000)
                            RECYCLEBENCH_OLLAMA_URL=http://127.0.0.1:<port>
                            provenance → $RB_LOGS/provenance_<job>.txt
            └─ your command (python3 tools/…, bash pegasus/llama4_canonical.sh, ollama pull …)
```

Typical commands (from the repo on the login node):

```bash
bash pegasus/build_image.sh                                                          # once per Ollama version
sbatch -p RTXA6000 --gpus=0 --mem=24G --time=03:00:00 pegasus/run.sbatch ollama pull <tag>   # CPU only
sbatch -p H100-PCI,H100,RTXB6000,A100-80GB,A100-RP --gpus=1 --mem=96G --time=12:00:00 \
       pegasus/run.sbatch bash pegasus/llama4_canonical.sh                                   # Scout
RB_MODEL=llama4-maverick RB_CACHE=cache/llama4_maverick.jsonl RB_STEPS=generate \
  sbatch --export=ALL -p B200,H200 --gpus=2 --mem=300G --time=08:00:00 \
       pegasus/run.sbatch bash pegasus/llama4_canonical.sh                                   # Maverick
sbatch -p RTXA6000 --gpus=0 --mem=16G --time=00:20:00 pegasus/run.sbatch \
       python3 tools/llama4_dashboard.py --run-dir results/runs/<run>                        # dashboard
```

## 3. Measured numbers

| What | Value |
|---|---|
| image build (podman + enroot import) | 2 min, 5.3 GB `.sqsh` |
| `ollama pull` from a compute node | 32–56 MB/s (Scout 67 GB ≈ 35 min, Maverick 245 GB ≈ 1 h 50 min) |
| `/netscratch` sequential read | ~216 MB/s → Scout load 7.5–8 min, Maverick ~20 min |
| Scout, warm, one A100-80GB / H100 | 31–37 generated tokens/s, 250–470 prompt tokens/s |
| Scout, first request after a load | 3.2 tokens/s (one-off warm-up) |
| Scout full canonical screen (AdvBench 520 + HarmBench 400 + StrongREJECT 313, k = 3) | 3,699 calls in 4.3 h on one H100 (3.0 s/call, HarmBench 6.6 s/call because Scout answers at length) + 31 min qwen2.5:14b judge |
| dashboard build | 25 s, CPU only |

## 4. Mistakes made, and what the code now does about them

1. **Work on the login node.** Early checks (CSV hashes, cache hit counts, a `dd` read test) ran on the login
   node. Not allowed. Every script now runs through `run.sbatch` or `sbatch --wrap`; the host Python has no numpy
   anyway.
2. **`srun` for long work.** Two image builds died with the SSH session. `build_image.sh` uses `sbatch`.
3. **`batch` partition for CPU jobs.** Had no free RAM. CPU jobs use `-p RTXA6000 --gpus=0`.
4. **Ollama's 5-minute load timeout.** Scout never loaded; the job retried every 5 min for 28 min of A100 time
   before it was cancelled. `OLLAMA_LOAD_TIMEOUT=30m`.
5. **Ollama prunes blobs at startup.** A server started by one job deleted the finished-but-unreferenced layers
   of a pull running in another job; Maverick's manifest then pointed at a missing blob and its first GPU job
   failed. `OLLAMA_NOPRUNE=1`. Repair = `ollama pull` again (fetches only what is missing). Prefer a single pull
   job at a time anyway.
6. **A side cache that did not exist.** `fill_responses.py` reads the cache before appending;
   `llama4_canonical.sh` now `touch`es `$CACHE` first.
7. **Long `--time` requests.** A 16 h request for 2×H200 was estimated to start 10 days later; 8 h started the
   next day. Check with `sbatch --test-only`, request what the job needs, rely on resumability.
8. **Smoke test before speed check.** A full smoke test burned its GPU hour on the load timeout. Order now:
   hello-world / `speed_check.sh` (30 min cap) → small subset (`RB_BENCH=… RB_DATA=<dir with a 5-row CSV>`) → full run.
9. **Wrong Grafana filter.** A saved link filtered `var-partition=batch`, so jobs on other partitions looked
   missing. Use `$__all`.

## 4b. The Llama 4 Scout suite (2026-11-01/02): what broke, what it costs

Run as `llama4-scout-suite` (`pegasus/scout_{prefetch,recycle,rpe,attack}.sh`, `experiments/llama4_scout/`,
`tools/llama4_suite.py`); the chain is logged in `/netscratch/$USER/recyclebench/logs/scout_chain.txt`.

1. **Code reaches Pegasus as a `git bundle`.** The clone's HTTPS remote needs browser auth. `git bundle create
   x.bundle <have>..<branch> main` locally, `scp` it to `/netscratch`, `git fetch x.bundle ...` in a CPU job. Every
   run then records a real `code_revision`. For a build or a validation, clone the exact commit into
   `/netscratch/.../src/<rev>` and point `RB_REPO` at it: it does not have to wait for a job using the home clone.
2. **This Slurm ignores `SBATCH_DEPENDENCY`**, and `--dependency=afterok:<id>` on a job that finished long ago is
   refused ("Job dependency problem"). Pass `--dependency` explicitly, only on jobs still known to Slurm.
3. **A stopped local orchestrator can survive.** One that was stopped still submitted a duplicate chain an hour
   later, including a rebuild of the validated image path. Anything that may run twice now takes an atomic
   `mkdir` lock on `/netscratch/$USER/recyclebench/locks/<name>.d` and exits if it is held (the lock is released by
   an EXIT trap). Use the same lock for **twin jobs**: one job per hardware shape (1 GPU > 80 GB, or 2 × 80 GB),
   whichever starts first runs, the other exits in a minute; cancel the pending twin when one starts.
4. **The gate image needs memory.** `pip` installs cleanly, then podman is OOM-killed committing the torch/CUDA layer
   at 24G: `RB_BUILD_MEM=96G`. The image is 10.9 GB, and **unpacking it at job start needs ~16 GB**: an 8 GB job is
   OOM-killed in `unsquashfs` before the command runs.
5. **The gate layer was not locked.** `WITH_GATE=1` installed torch/sentence-transformers without the lock file, so
   numpy/scipy/pandas could move. It is now `-c requirements-lock.txt`; `env_check --strict` passes in the image.
   **The byte-reproducibility test now passes in the container** (246 passed, 0 failed); locking and thread pinning
   changed together, so which of the two fixed it is not separable.
6. **One Scout plus the judge needs more than 80 GB.** On one 80 GB card Ollama swaps Scout (67 GB, ~8 min from
   `/netscratch`) and qwen2.5:14b. Phase-separated tools (RPE) tolerate it; tools that judge inline (run-real,
   attack transfer) need 1 card > 80 GB (B200, RTXB6000, H200) or 2 × 80 GB. **Ollama 0.40.12 runs on B200**
   (compute 10.0). `pegasus/gpu_guard.sh` stops a job whose judge is not 100 % on the GPU.
7. **Attack transfer is the long arm:** ~10 s per call (Scout + inline judge) on 2 × A100-80GB, ~9,900 calls. Shard by
   class (classes are independent): each shard seeds its own cache from the arm's cache, then merge
   (`cache_consolidate.py`) and run `scout_attack.sh` once more, which hits the cache and only judges, so every class
   shares one baseline and one judging environment.
8. **The driver plans once per invocation.** On a fresh run `subset` and `recycle` are skipped as "needs screen
   first" even after screen succeeded; `scout_recycle.sh` re-invokes `experiment run` until a pass runs nothing.
9. **A job-id shard breaks resume.** `experiment.sbatch` named the cache shard after `$SLURM_JOB_ID`, so a
   resubmission started an empty shard. `RB_SHARD=` (set, empty) runs unsharded; a shard is for concurrent jobs only.
10. **Fair-share moves estimates by days.** After a night of use, a B200 start went from "now" to 11 Nov while seven
    B200 GPUs sat idle (a higher-priority array). Check `sbatch --test-only` for several shapes and submit twins.
11. **Python output in job logs is block-buffered.** Progress lines arrive in bursts or at exit: use `python3 -u`.
12. **On Windows, use `PYTHONUTF8=1`** for any local rebuild: under cp1252 one StrongREJECT prompt decodes differently,
    its cache key misses, and the count silently drops 313 → 312.

## 5. Things that differ from the Mac runs (record them, don't hide them)

- CUDA vs Metal, Ollama 0.40.12 here vs whatever the Mac has: new responses are a separate environment.
  Cached responses are untouched (keys use config key + prompt + seed, not the URL or hardware).
- Llama 4's Ollama template adds a long default system prompt: a two-word prompt counted ~365 prompt tokens.
  Not yet recorded per run; add `ollama show --modelfile <tag>` to the provenance.
- Temperature 0 with three seeded draws still gives differing draws on GPU (e.g. AdvBench jbb-0000: R, C, C), so
  "any of k draws" matters.
- Tags are not digests: `llama3:8b` pulled on Pegasus in 2026 may not be the blob the Mac pulled months earlier.
  Compare `ollama list` IDs before letting Pegasus regenerate anything for the six old models. **Still open:** no Mac
  digest is recorded anywhere in the repo; the Scout recycle and RPE runs regenerated old-model replies on Pegasus,
  so they are internally consistent (both models, one environment) but not tied to the Mac's old-model blobs.
- The judge is not environment-invariant either. Re-judging the identical cached canonical replies on a B200 gave
  HarmBench llama3-8b → Scout P⁻ = 8 against the stored 7 (one verdict moved); AdvBench and HarmBench llama32-3b
  reproduced exactly (6, 5). Report re-judged figures beside stored ones, never silently replace them.

## 6. Cleanup list for the next person

Ordered by how much they get in the way of a seamless run.

1. **Merging results back is manual.** `tools/cache_consolidate.py` now merges verdict stores too (13744c3); still
   missing is one `sync_from_pegasus.sh` (rsync `results/runs/` with `--ignore-existing`, caches into a staging dir,
   consolidate).
2. **Parallel jobs and the shared cache.** Two jobs must never append to the same JSONL (`fill_responses.py`
   has a lock file; the verdict stores have none). Today the second job writes a side cache and skips judging, and
   someone merges and judges later by hand. Make that a supported mode: per-job cache shards
   (`cache/shards/<job>.jsonl`) merged at the end, or a lock on the verdict stores too.
3. **`llama4_canonical.sh` is generic but named for Llama 4.** It works for any config key (`RB_MODEL`). Rename
   (e.g. `pegasus/canonical_screen.sh`) and take the config from `RB_CONFIG`.
4. **Provenance gaps.** Add `ollama show --modelfile` per model, the image digest, the git commit, and copy the
   provenance file into every run dir (only the report step does it now).
5. ~~**Container test failure.**~~ Resolved 2026-11-01 (§4b.5): in the locked gate image with pinned BLAS threads
   the full suite passes, the byte-reproducibility test included.
6. **The 70B lineage is downloaded but not configured.** `llama2:70b`, `llama3:70b`, `llama3.1:70b`, `llama3.3:70b`
   are in `/netscratch/…/ollama`; add config keys, `MODEL_META` entries in `tools/lineage_core.py`, and pick
   48 GB+ GPUs (a 70B q4 is ~40–43 GB).
7. **SORRY-Bench is missing on Pegasus.** Gated on Hugging Face: accept the licence, export `HF_TOKEN`, run
   `tools/prepare_sorrybench.py` in a job. The scripts skip a benchmark whose CSV is absent.
8. **Rewording arm (RSR/PBR) for Llama 4.** Done for Scout's three eligible pairs on their own frozen manifests
   (§4b). Still blocked: PBR and RSR for every pair on the *lineage's* frozen manifests, which live only on the Mac.
   Copy them over (they hold harmful text: `/netscratch`, `chmod 600`).
9. **`fill_responses.py` should create a missing cache itself** instead of relying on the caller's `touch`.
10. **Pin and record the Ollama version in one place.** It is `RB_OLLAMA_VERSION` in `env.sh` and the Dockerfile
    `ARG`; the build script passes one to the other. Keep it that way; bump both by rebuilding the image.

## 7. Rules worth repeating

- Nothing runs on the login node. Not even a five-line Python check.
- Ask before anything that calls a model (`CLAUDE.md`): GPU jobs cost real compute.
- Never overwrite results; dashboards and run dirs are dated.
- `/netscratch` has no backup: sync `cache/` and `results/` back after every run.
- Benchmark CSVs and anything with responses are harmful text: `/netscratch`, `chmod 600`, never in git or an image.
