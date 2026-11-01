# Access, house rules and contact

Sources: [Guidelines & Rules](https://pegasus.dfki.de/docs/guidelines/),
[Getting started](https://pegasus.dfki.de/docs/guidelines/getting-started/),
[Questions / Contact](https://pegasus.dfki.de/docs/guidelines/questions-contact/).
Paraphrased notes, taken 2026-11-01 (the pages were last changed 2026-09-04). The official docs win if they disagree.

## Rules (breaking them can get the account suspended)

- **Be considerate.** Request what the job needs and use it well; never hold resources "for later".
- **Be responsible.** Watch your jobs; a job that wastes compute is your problem, and access can be restricted
  if you interfere with the cluster.
- **Be reachable.** Operators contact users on DFKI Mattermost (Deep Learning channel). If you cannot be reached
  within 1–2 hours about a problem job, it may be cancelled.
- **Keep directories tidy.** `$HOME` has a 10 GB quota; `/netscratch/$USER` has no hard limit yet but is shared,
  so clean it regularly and keep permanent use within a few TB. Cleanup requests must be followed promptly.
- **Ask, then contribute back.** Try to solve it yourself, ask in the channel, update the docs with what you learn.

## Prerequisites

- MyDFKI / intranet account, access to DFKI GitLab and Mattermost.
- A Pegasus account: email `cluster@dfki.de` from a DFKI address (credentials go only to that address).
- Basic shell tools: `ssh`, `htop`, `tmux`/`screen`.

## Logging in

- Head (login) nodes: `login1`, `login2`, `login3` `.pegasus.kl.dfki.de`. Check the host key fingerprint on first use.
- Use an ed25519 SSH key; `$HOME` is shared across all machines, so key setup is one-time.
- **Use one login node at a time.** End your processes on one before moving to another.
- Login-node limits: 500 sessions per user, 6 CPU cores, 10 GB RAM; processes older than two months are killed nightly.
- **Never run compute or heavy tools on a login node, remote IDEs (VS Code Remote, Claude Code) included.**
  For this project: the login node is for `sbatch`, `squeue`, `sacct`, `scancel`, `scp`, `ls`, reading logs.
  Anything that executes code, however small, goes in a job.

## Data transfer and long sessions

- Do not mount cluster storage on your machine; use `scp`, `rsync`, SFTP or `sshfs`.
- Talk to the operators before transfers of roughly 20 GB or more.
- `screen`/`tmux` keeps interactive work alive across disconnects, but prefer `sbatch` (see [04-batch-jobs.md](04-batch-jobs.md)).

## Interactive jobs

- Keep them rare; limit to `--time=04:00:00 --immediate=3600` (these are also the maximums).

## Network

- From the Saarbrücken VPN only ports 22, 80, 443 and ≥ 10000 are reachable, so services (Jupyter, TensorBoard,
  Ollama if exposed) must use ports ≥ 10000.

## Good practice

- Wrap job commands in shell scripts; track experiments (sacred, MLflow, …).
- Write jobs that can resume after interruption and checkpoint regularly.
- An onboarding tutorial lives in the Pegasus tutorial Git repository (`./bin/tutorial` after cloning).

## Asking for help

- Read the docs first. Quick questions: Mattermost Deep Learning channel; answer in threads, paste text and code
  blocks rather than screenshots, link the job in the monitoring dashboard.
- Include: what you tried to do, the command line, console/error output, job ids / nodes, what you expected.
- Private matters: Christian Schulze (admin, accounts, hardware, security), Joachim Folz (monitoring, logging,
  libraries, containers, docs).
