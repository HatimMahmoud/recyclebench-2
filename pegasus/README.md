# Running recyclebench on DFKI Pegasus

Pegasus docs: <https://pegasus.dfki.de/docs/>, condensed in [docs/pegasus/kb/](../docs/pegasus/kb/README.md).
What went wrong the first time and what is still to clean up: [docs/pegasus/LESSONS.md](../docs/pegasus/LESSONS.md).

House rules these scripts follow: **nothing runs on the login node** (that includes IDEs, AI coding agents and
"just a quick Python check"); code in `$HOME`, everything large in `/netscratch/$USER`; containers through
enroot/pyxis; every job is an `sbatch` job.

## Layout

| What | Where |
|---|---|
| repo (code, `cache/`, `results/`) | `~/repos/recyclebench-2` |
| benchmark CSVs (harmful text, chmod 600) | `/netscratch/$USER/recyclebench/data/`, symlinked into `data/` |
| Ollama models | `/netscratch/$USER/recyclebench/ollama` |
| image | `/netscratch/$USER/recyclebench/images/recyclebench_ollama-<ver>.sqsh` |
| job, Ollama and provenance logs | `/netscratch/$USER/recyclebench/logs/` |

`/netscratch` has no backup: copy `cache/` and `results/` back to the main machine after a run and merge the
cache there (`tools/cache_consolidate.py`). Never run jobs on two machines against the same cache file at once.

## One-time setup

```bash
mkdir -p /netscratch/$USER/recyclebench/logs
bash pegasus/build_image.sh                       # sbatch job, ~2 min: docker/Dockerfile -> .sqsh
sbatch -p RTXA6000 --gpus=0 --mem=24G --time=03:00:00 pegasus/run.sbatch \
    bash -c 'ollama pull llama4:17b-scout-16e-instruct-q4_K_M && ollama pull qwen2.5:14b'
```

Run one pull job at a time. If a model fails with `failed to describe model … no such file`, pull it again:
only the missing blobs are fetched.

Benchmarks (in a job, not on the login node): `tools/prepare_behaviours.py --benchmark {advbench,strongreject}`
and `--benchmark harmbench --keep-duplicates` reproduce the CSVs the cached runs used. SORRY-Bench is gated on
Hugging Face: accept its licence, put a read token in `$HF_TOKEN`, then `tools/prepare_sorrybench.py`. Never
rebuild `data/heldout_jbb_guided.csv`; copy the registered file (sha256 in `data/heldout_jbb_guided.audit.json`).

## Running one experiment spec (the supported path)

`pegasus/experiment.sbatch` runs an experiment spec through the *same* driver as a laptop, so the
stage order, the guards and the artifacts are identical; only the wrapper and the resources differ.

```bash
RB_PLAN=1 sbatch -p RTXA6000 --gpus=0 --mem=8G --time=00:10:00 \
    pegasus/experiment.sbatch experiments/<spec>.yaml        # plan only: no calls, no writes
sbatch -p A100-80GB --gpus=1 --mem=96G --time=08:00:00 \
    pegasus/experiment.sbatch experiments/<spec>.yaml        # run, resumable
RB_STAGE=screen sbatch ... pegasus/experiment.sbatch experiments/<spec>.yaml   # one stage only
```

The job passes `--shard $SLURM_JOB_ID`, so it appends only to `cache/shards/<name>__<job>.jsonl`
and can never collide with another job on the shared cache (LESSONS §6.2). Merge afterwards:

```bash
python3 tools/cache_consolidate.py --target cache/<name>.jsonl \
    --sources cache/shards/<name>__*.jsonl --apply
```

That tool now merges **verdict stores** (`{k, v}`) as well as response caches (`{k, t}`), detects
which kind each file is, and refuses to merge one kind into the other.

Before a long run, confirm the environment matches the stack the registered figures came from:

```bash
sbatch -p RTXA6000 --gpus=0 --mem=8G --time=00:10:00 pegasus/run.sbatch \
    python3 tools/env_check.py --strict
```

## Running a canonical screen of one model

`pegasus/canonical_screen.sh` replaces `llama4_canonical.sh` (LESSONS §6.3): the config is
`RB_CONFIG`, so it is not Llama-4-specific.

```bash
RB_CONFIG=configs/llama4_pegasus.yaml RB_MODEL=llama4-scout \
sbatch -p H100-PCI,H100,A100-80GB --gpus=1 --mem=96G --time=12:00:00 \
    pegasus/run.sbatch bash pegasus/canonical_screen.sh
```

## Running (original commands)

`pegasus/run.sbatch` runs any command inside the image with a private Ollama server on a per-job port
(`RECYCLEBENCH_OLLAMA_URL`; another job on the node may hold 11434). Provenance (Ollama version, GPU, model
digests) goes to `logs/provenance_<job>.txt`.

```bash
# 1. cheap check first: load time and steady tokens/s (30 min cap)
sbatch -p H100 --gpus=1 --mem=96G --time=00:30:00 pegasus/run.sbatch bash pegasus/speed_check.sh
# 2. Llama 4 Scout canonical screen + judge + 2x2 (one 80 GB+ GPU, ~5 h)
sbatch -p H100-PCI,H100,RTXB6000,A100-80GB,A100-RP --gpus=1 --mem=96G --time=12:00:00 \
    pegasus/run.sbatch bash pegasus/llama4_canonical.sh
# 3. Maverick (245 GB: one node, 2x B200/H200) beside another job: own cache, judge later
RB_MODEL=llama4-maverick RB_CACHE=cache/llama4_maverick.jsonl RB_STEPS=generate \
    sbatch --export=ALL -p B200,H200 --gpus=2 --mem=300G --time=08:00:00 \
    pegasus/run.sbatch bash pegasus/llama4_canonical.sh
# 4. dashboard for a finished screen (CPU only)
sbatch -p RTXA6000 --gpus=0 --mem=16G --time=00:20:00 pegasus/run.sbatch \
    python3 tools/llama4_dashboard.py --run-dir results/runs/<stamp>_llama4_scout_canonical
squeue -u $USER ; sacct -j <job> ; tail -f /netscratch/$USER/recyclebench/logs/<name>_<job>.out
```

Smoke test on a subset: `RB_BENCH=advbench RB_DATA=<dir holding a 5-row advbench.csv>`; the ids are row
positions, so a prefix of the real CSV reuses its responses.

Choosing resources: `sbatch --test-only <same options>` prints the estimated start. Short `--time` starts much
sooner (backfill), and every step here resumes from the caches, so ask for what the job needs. Every partition is
a GPU partition; CPU-only jobs pass `--gpus=0`. Ollama serves one request at a time (`OLLAMA_NUM_PARALLEL=1`, as on
the Mac), so extra GPUs only help when the model does not fit on one.

Responses produced here come from CUDA, not the M1's Metal backend, and possibly another Ollama version: cached
responses are unaffected, but treat new ones as a separate environment.
