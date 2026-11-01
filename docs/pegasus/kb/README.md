# Pegasus knowledge base

Condensed notes of the [Pegasus documentation](https://pegasus.dfki.de/docs/) (DFKI's Slurm GPU cluster), taken
2026-11-01 from pages last changed 2026-09-04, plus what this project learned running on it. They are
paraphrased and reorganised by topic; when they disagree with the official pages, the official pages win.
Project-specific lessons and the cleanup list are in [../LESSONS.md](../LESSONS.md).

| File | Covers (official pages) |
|---|---|
| [01-access-and-rules.md](01-access-and-rules.md) | Guidelines & Rules, Getting started, Questions / Contact |
| [02-storage.md](02-storage.md) | Storage |
| [03-containers.md](03-containers.md) | Software stack, Custom software |
| [04-batch-jobs.md](04-batch-jobs.md) | Slurm cluster (hello world), Batch jobs |
| [05-resources-and-partitions.md](05-resources-and-partitions.md) | Resource allocation, Partitions |
| [06-monitoring-logging-peggy.md](06-monitoring-logging-peggy.md) | Monitoring your jobs, Logging, Peggy |
| [07-known-issues-and-debugging.md](07-known-issues-and-debugging.md) | Known issues, Debugging |
| [08-multi-gpu-and-examples.md](08-multi-gpu-and-examples.md) | Multi-GPU & distributed jobs, Examples |

## The ten things to know first

1. Log in to one of `login1–3.pegasus.kl.dfki.de` with an SSH key, and only one at a time.
2. Nothing computes on a login node, IDEs and AI coding agents included. Use `sbatch`.
3. Everything runs in Enroot containers through Pyxis (`--container-image`, `--container-mounts`).
4. `$HOME` (10 GB, backed up) is for code; `/netscratch/$USER` (no backup) for images, models, data, logs.
5. Pick the GPU by partition; `-p A,B` accepts whichever starts first; `--gpus=0` for CPU-only work.
6. GPU jobs get 24 GiB RAM unless you ask; containers need their `.sqsh` size on top.
7. Short `--time` starts sooner (backfill); `sbatch --test-only` shows the estimate.
8. Watch jobs on `monitoring.pegasus.kl.dfki.de` (intranet/VPN), or `squeue -u $USER` / `sacct -j <id>`.
9. Services listen on ports ≥ 10000.
10. Be reachable on Mattermost (Deep Learning channel) while jobs run.
