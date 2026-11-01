# Monitoring, logging and Peggy

Sources: [Monitoring your jobs](https://pegasus.dfki.de/docs/slurm-cluster/monitoring-your-jobs/),
[Logging](https://pegasus.dfki.de/docs/slurm-cluster/logging/), [Peggy](https://pegasus.dfki.de/docs/slurm-cluster/peggy/).
Paraphrased notes, taken 2026-11-01.

## Dashboards (DFKI intranet or VPN only)

| Dashboard | URL |
|---|---|
| Current jobs | `http://monitoring.pegasus.kl.dfki.de/d/slurm-current-jobs/current-jobs?refresh=15s&autofitpanels` |
| Job archive | `http://monitoring.pegasus.kl.dfki.de/d/slurm-job-archive/job-archive?autofitpanels` |
| Job details | `http://monitoring.pegasus.kl.dfki.de/d/slurm-job-details/job-details?var-jobid=<JOBID>` |
| Resources (what is free) | `http://monitoring.pegasus.kl.dfki.de/d/slurm-resources/resources?refresh=15s` |
| Job statistics | `http://monitoring.pegasus.kl.dfki.de/d/slurm-job-statistics/job-statistics?refresh=5m` |
| All | `http://monitoring.pegasus.kl.dfki.de/dashboards` |

- Filters: user, id, name (PostgreSQL patterns: `%` wildcard, `a|b`), state, **partition**, node. A job is invisible
  if the partition filter is set to something else (a saved link with `var-partition=batch` hides every job on other
  partitions), and pending jobs may not appear at all; use `squeue -u $USER` for those.
- Job details shows GPU/CPU/memory over the job's life. Watch for: overloaded CPUs (red = system time), full
  memory or swap, idle GPUs, long pauses. Power and network figures are per node, not per job.
- You can `ssh` to a node while your job runs there and use `htop`.

## E-mail notifications

```
srun --mail-type=ALL --mail-user=you@example.org [command]      # also works with sbatch
```

`ALL` = BEGIN, END, FAIL, INVALID_DEPEND, REQUEUE, STAGE_OUT; `TIME_LIMIT_90` warns at 90 % of the time limit.

## Console logging (opt-in, work in progress)

- `touch ~/.logoptin` enables logging of job console output to the job-details dashboard; `rm ~/.logoptin` disables.
  Checked at job start only.
- **Logs are visible to everyone with dashboard access.** Never print secrets or (for this project) prompt/response
  text while opted in.
- `.logoptin` is YAML, e.g. `mode: enabled | metadata | disabled`. Switch at run time by printing
  `/LOGCOMMAND {"mode":"metadata"}`.
- One JSON object per line on stdout (`{"epoch": 1, "loss": 7.2}`) becomes plotted metadata; a `"progress"` key is
  the preferred x-axis.

## Peggy

- An automated checker that reads the dashboard time series and messages you (e-mail via `--mail-user`, or
  Mattermost) about jobs that do not stop, frozen GPUs in multi-GPU jobs, heavy system overhead, or I/O stalls.
- An issue must persist 30 min to several hours before a message; repeats at most every 12 h unless it gets worse.
