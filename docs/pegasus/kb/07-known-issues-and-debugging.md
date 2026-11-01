# Known issues and debugging

Sources: [Known issues](https://pegasus.dfki.de/docs/slurm-cluster/known-issues/),
[Debugging](https://pegasus.dfki.de/docs/slurm-cluster/debugging/). Paraphrased notes, taken 2026-11-01.

## Known issues (from the docs)

| Symptom | Cause | Fix |
|---|---|---|
| pip builds from source, `No matching distribution found for cython>=3.0.0`, `inconsistent Name` | very old pip in the image | `python -m pip install --upgrade pip` first |
| `CUDA error: invalid device ordinal` with `--ntasks` + `--gpus-per-task` | Slurm bug (SchedMD 18756) | set `CUDA_VISIBLE_DEVICES` from `SLURM_GPUS_ON_NODE`, or `--nodes=1 --ntasks=N --gpus=N`, or use torchrun with `--gpus` only |
| NCCL / peer-to-peer fails between tasks | each task sees only its own GPU | add `--gpu-bind=none` |
| software refuses to run as root | your user is root inside the container (not on the host) | `--no-container-remap-root` |
| job slow, CPU graph mostly red (system time) | libraries spawn a thread per node CPU | `MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 USE_OPENMP=1`; size worker pools from `SLURM_CPUS_PER_TASK` / `SLURM_CPUS_ON_NODE` |
| `multiprocessing.cpu_count()` too high | returns the node's CPUs, not the job's | `int(os.environ['SLURM_CPUS_ON_NODE'])` |
| job keeps "running" after its processes exited | unknown, rare | `scancel <job>` (Peggy may flag it) |
| `OMP: Error #178: Function Can't open SHM failed`, DataLoader worker aborted | unknown | `ulimit -n 4000`, more `--mem`, fewer workers + the thread variables above |

## Debugging with an IDE

- Treat debugging like an interactive job: one session, stop it when done.
- Recommended: run **code-server** (browser VS Code) inside a job instead of attaching a remote IDE:
  1. Self-signed cert, valid one year:

```
mkdir -p ~/.cert
openssl req -x509 -newkey rsa:4096 -days 365 -nodes \
  -keyout ~/.cert/pegasus.key -out ~/.cert/pegasus.crt -sha256 \
  -subj "/C=DE/ST=Rheinland-Pfalz/L=Kaiserslautern/O=DFKI/OU=Pegasus/CN=kl.dfki.de"
```

  2. A `start_code_server.sh` (template on the docs page) that sets an argon2 password hash (`echo -n <pw> |
     argon2 $(openssl rand -base64 32) -e`; never your cluster password), picks port
     `(SLURM_JOB_ID + 10007) % 16384 + 49152`, unpacks the newest `/netscratch/software/code-server-*-linux-amd64.tar.gz`
     and starts it with `--auth password --cert … --bind-addr=$HOSTNAME.kl.dfki.de:$PORT`. `chmod u+x,go-rwx` it.
  3. Start it:

```
srun --container-image=/enroot/nvcr.io_nvidia_pytorch_22.03-py3.sqsh \
  --container-mounts=/netscratch/software:/netscratch/software:ro,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" --time=01:00:00 start_code_server.sh
```

- If code-server misbehaves, stop it and delete `.code-server` in the working directory; a running VS Code
  Remote-SSH session can make it fail with `ENOENT … connect`.

## Issues this project hit (not in the docs)

| Symptom | Cause | Fix (in the repo) |
|---|---|---|
| `srun` job killed while queued / building | SSH session dropped | submit with `sbatch` (`pegasus/build_image.sh`) |
| "Requested nodes are busy" for a CPU job on `batch` | no free RAM on batch nodes | `-p RTXA6000 --gpus=0` |
| Ollama `Load failed … timed out waiting for llama-server to start`, retried every 5 min | 67 GB model read from `/netscratch` at ~216 MB/s takes > 5 min (default load timeout) | `OLLAMA_LOAD_TIMEOUT=30m` (`pegasus/env.sh`) |
| a model's blob missing after a successful pull (`failed to describe model … no such file`) | every `ollama serve` prunes blobs no manifest references; a server started by another job mid-pull deleted the pull's finished layers | `OLLAMA_NOPRUNE=1`; repair by re-running `ollama pull` (fetches only missing blobs) |
| first request after a load: 3 tokens/s | one-off warm-up | ignore; steady state ~31–37 tokens/s on A100/H100 for Scout |
| `ModuleNotFoundError: numpy` running a repo tool on the login node | the host Python is bare, and you must not run tools there anyway | run in the image via `pegasus/run.sbatch` |
| test `test_study_is_byte_reproducible_under_same_seed` fails in the container, passes on Windows/macOS | not diagnosed (likely threaded BLAS or a newer numpy/statsmodels) | open, see `docs/pegasus/LESSONS.md` |
