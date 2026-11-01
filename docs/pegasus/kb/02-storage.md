# Storage

Source: [Storage](https://pegasus.dfki.de/docs/guidelines/storage/). Paraphrased notes, taken 2026-11-01.

> Pegasus storage is for running experiments, not for keeping data. Anything that matters must also live
> outside Pegasus.

## Where things go

| Path | Use | Backup |
|---|---|---|
| `$HOME` (10 GB quota) | source code, small final results | every 3 h, kept 30 days |
| `/netscratch/$USER` | BeeGFS cluster FS: environments, images, models, data, logs, results | **none** |
| `/fscratch/$USER` | smaller, faster cluster FS for low-latency or very distributed jobs | none |
| `/ds`, `/ds-*` | shared datasets: read, don't write | local snapshots every 3 h, kept 30 days |

- Do **not** put in `$HOME`: logs, checkpoints, virtualenvs/conda envs, datasets, model weights.
- Avoid many small files, and huge numbers of files in one directory: network file systems punish both.
  For datasets, prefer few large files (datadings, webdataset).
- Prefer mounting `/ds` read-only into containers over copying data; write results to mounted `/netscratch`.
- After an account is deactivated, all personal directories go to cold storage on the 1st of the month after next;
  retention is meant to be 6 months but may be shorter.

## Cleanup

- `ncdu` shows where space goes (press `D` to delete from inside it).
- `springclean` finds serial files such as training snapshots:

```
/netscratch/springclean /netscratch/$USER --dry-run
/netscratch/springclean -h
```

## Datasets in `/ds`

- A large collection of common datasets already exists. New shared datasets: add a folder with a short
  `README.md` (source URL, citation) and mention it in the chat channel; ask first if it is big.

## Git

- Use DFKI GitLab for personal repositories; never commit large or binary files (datasets, models).

## What this project stores where

| What | Where |
|---|---|
| repo clone, `cache/`, `results/` | `~/repos/recyclebench-2` (cache ≈ 70 MB of text) |
| benchmark CSVs (harmful text, `chmod 600`) | `/netscratch/$USER/recyclebench/data/`, symlinked into the repo's `data/` |
| Ollama models (~475 GB with the 70B lineage, Scout, Maverick) | `/netscratch/$USER/recyclebench/ollama` |
| container image (5.3 GB) | `/netscratch/$USER/recyclebench/images/` |
| job, Ollama and provenance logs | `/netscratch/$USER/recyclebench/logs/` |

`/netscratch` has no backup: copy `cache/` and `results/` back to the main machine after every run.
