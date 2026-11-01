# Batch jobs (sbatch)

Source: [Batch jobs](https://pegasus.dfki.de/docs/slurm-cluster/batch-jobs/),
[Slurm cluster](https://pegasus.dfki.de/docs/slurm-cluster/). Paraphrased notes, taken 2026-11-01.

## Why sbatch, not srun

- The job runs without your console: no `tmux` needed, and it survives SSH drops and login-node crashes.
  (In this project an `srun` image build was killed twice by dropped SSH sessions; `sbatch` fixed it.)
- Output goes to a file (default `slurm-<jobid>.out`; set `--output`, which accepts `%j` job id, `%x` job name,
  `%u` user, `%a` array index, …).
- Job arrays run sweeps in one submission and are kinder to the scheduler.
- Defaults can live in the script as `#SBATCH` lines.

## Basics

```
#!/bin/bash
echo "hello world!"
```

```
sbatch --time 1 --mem 16M hello_sbatch
```

- A shebang is required. `#SBATCH` lines must all sit at the top; parsing stops at the first command.
  Command-line options override them.
- `#SBATCH` lines are not shell-expanded: ``#SBATCH --container-workdir="`pwd`"`` sets the literal string.
  Pass such options on the command line, or `cd "$SLURM_SUBMIT_DIR"` inside the script.
- With `--container-image` on `sbatch`, the whole script runs in the container. To mix host steps and different
  images, call `srun --container-image=…` inside the script (this project's `pegasus/run.sbatch` does that).

## Job arrays

```
#!/bin/bash
#SBATCH --array 0-4%3          # 5 tasks, at most 3 at once
echo "index $SLURM_ARRAY_TASK_ID"
```

Each task reads `$SLURM_ARRAY_TASK_ID`; map it to parameters (one file per index, an index into a JSON list,
bash arrays of argument sets, or a `ParameterGrid` in Python).

## No script file

```
sbatch [options] --wrap "command \$ESCAPED_VAR"      # escape variables meant for the compute node
sbatch <<EOF                                          # script on stdin; expanded at submission time
#!/bin/sh
#SBATCH --container-image=/enroot/…
python train.py
EOF
```

## Dependencies (used heavily in this project)

```
R=$(sbatch --parsable … repair.sh)
sbatch --dependency=afterok:$R …      # start only if $R succeeded
sbatch --dependency=afterany:$R …     # start after $R ends, whatever happened
```

A chain like download → verify → GPU run never wastes GPU time on a broken prerequisite.

## Hello world

`srun echo "hello world!"` creates a job, reserves resources, runs the command on a worker and prints the output.
