# Resources and partitions

Sources: [Resource allocation](https://pegasus.dfki.de/docs/slurm-cluster/resource-allocation/),
[Partitions](https://pegasus.dfki.de/docs/slurm-cluster/partitions/). Paraphrased notes, taken 2026-11-01.

## Requesting resources

```
-n, --ntasks=ntasks         number of tasks
-N, --nodes=N               number of nodes
-c, --cpus-per-task=ncpus
    --cpus-per-gpu=n
-G, --gpus=n                GPUs for the job (--gpus=0: no GPU)
    --gpus-per-node=n
    --gpus-per-task=n
    --mem=MB                memory per node (e.g. --mem=96G)
    (--mem-per-gpu)         DO NOT USE: buggy
    --mem-per-cpu=MB
    --time=1-00:00          d-hh:mm
```

- The GPU **type** is chosen by partition (`-p NAME`, several allowed: `-p H200,B200` runs on whichever starts first).
- One task on one GPU: `--gpus=1`. Multi-task: `--ntasks` + `--gpus-per-task` + `--gpu-bind=none`
  (Slurm's documented `--gpus`/`--gpus-per-task` combination does not work here).
- **Default memory for GPU jobs is 24 GiB per node.** Containers need ~their `.sqsh` size on top. Ask for more
  explicitly (`--mem=96G`), or the job runs badly or is OOM-killed.
- `scontrol show partition NAME` shows a partition's defaults and limits (`MaxTime`, `DefMemPerNode`, nodes).
- Interactive: `srun [...] --time=04:00:00 --immediate=3600 --pty /bin/bash` (those are the maximums).
  Interactive jobs left open are the main cause of idle hardware.

## Partitions

Every partition is a GPU partition; `batch` is the default. Some groups own sub-partitions (higher priority,
same default time, no maximum); this project's account `dsa-external` (group `dsa`) can use `L40S-DSA`,
`A100-RP`, `H100-RP`.

CPU and memory "per GPU" assume a node split evenly across its GPUs: a starting point, not a limit.

| Partition | GPU | Arch | GPU mem (GB) | GPUs/node | CPUs/GPU | RAM/GPU (GB) | Time default – max |
|---|---|---|---|---|---|---|---|
| `A100-40GB` / `A100-80GB` | A100-SXM4 | Ampere | 40 / 80 | 8 | 32 | 112 / 224 | 1 – 3 days |
| `A100-PCI` | A100-PCIE | Ampere | 40 | 8 | 12 | 48 | 1 – 3 days |
| `B200` | B200-SXM6 | Blackwell | 183 | 8 | 28 | 224 | 1 – 1 day |
| `H100` | H100-SXM5 | Hopper | 80 | 8 | 28 | 224 | 1 – 1 day |
| `H100-PCI` | H100 NVL | Hopper | 94 | 6 | 32 | 170 | 1 – 1 day |
| `H200` | H200-SXM5 | Hopper | 141 | 8 | 28 | 224 | 1 – 1 day |
| `H200-PCI` | H200 NVL | Hopper | 141 | 6 | 32 | 170 | 1 – 1 day |
| `L40S` | L40S | Ada Lovelace | 48 | 8 | 16 | 125 | 1 – 3 days |
| `RTX3090` | RTX 3090 | Ampere | 24 | 8 | 12 | 64 | 1 – 3 days |
| `RTXA6000` | RTX A6000 | Ampere | 48 | 8 | 12 | 108 | 1 – 3 days |
| `RTXB6000` | RTX Pro 6000 Blackwell | Blackwell | 96 | 8 | 64 | 224 | 1 – 1 day |
| `V100-32GB` | V100-SXM2 | Volta | 32 | 8 | 10 | 64 | 1 – 3 days |
| `batch` (default) | RTX 6000 | Turing | 24 | 10 | 7 | 64 | 1 – 3 days |

Interconnect: NVSwitch on A100-SXM, B200, H100, H200 (best multi-GPU); NVLink on H100-PCI, H200-PCI, V100;
PCIe elsewhere. InfiniBand only on A100-SXM and V100 (multi-node). Minimum NGC image: A100 20.06;
A100-PCI/RTX3090/RTXA6000 20.10; H100/H200/L40S 22.09; B200 25.05; RTXB6000 25.06.

The page also lists ResNet-50 and Transformer training throughput per partition (e.g. ImageNet images/s on a full
node: H100 12,879; H200 13,472; B200 ~22,000; RTXB6000 9,150; A100 6,500; RTXA6000 3,450).

## Seeing what is free

```
sinfo
sinfo -o "| %15P | %15f | %8c | %10m | %15G | %10O | %8t | %N"
sinfo -o "%P %C %e"            # CPUs allocated/idle/other/total and free memory per partition
sinfo -p H100,H200 -o "%P %G %F"
```

Resources dashboard: `http://monitoring.pegasus.kl.dfki.de/d/slurm-resources/resources?refresh=15s`.

## Getting a job started sooner (learned in this project)

- `sbatch --test-only [same options] script` prints an estimated start time without submitting. Compare
  partitions and GPU counts before choosing.
- **A shorter `--time` starts much sooner.** Backfill fits short jobs into gaps: the same 2×H200 request was
  estimated 10 days out with `--time=16:00:00` and next day with `--time=08:00:00`. Make jobs resumable and ask
  for the time they actually need.
- List several acceptable partitions in one job (`-p B200,H200`, `-p H100-PCI,H100,RTXB6000,A100-80GB,A100-RP`).
- CPU-only work (image builds, `ollama pull`) still goes to a GPU partition with `--gpus=0`. On 2026-10-30
  `batch` had no free RAM ("Requested nodes are busy"); `RTXA6000` with `--gpus=0` started at once.
