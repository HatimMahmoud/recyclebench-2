# Multi-GPU jobs and examples

Sources: [Multi-GPU & distributed jobs](https://pegasus.dfki.de/docs/slurm-cluster/multi-gpu-distributed-jobs/),
[Examples](https://pegasus.dfki.de/docs/slurm-cluster/examples/). Paraphrased notes, taken 2026-11-01.

## Distributed training (PyTorch)

- Recommended: DistributedDataParallel with one Slurm task per GPU. DataParallel needs `-N 1`.
- Don't run multi-GPU jobs on `batch`: it mixes GPU models, from slow to crashing.
- `srun -K` (`--kill-on-bad-exit`) stops all tasks when one dies.
- The cluster presets `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK` and NCCL settings
  (`NCCL_SOCKET_IFNAME=bond,eth`, `NCCL_IB_HCA=mlx5`); call `torch.distributed.init_process_group('NCCL')`,
  pick the device from `LOCAL_RANK`, shard data with `DistributedSampler`, print and checkpoint on rank 0 only.

```
srun -p RTXA6000 --ntasks 4 --gpus-per-task 1 --gpu-bind=none [command]    # 1 task per GPU (DDP)
srun -p RTXA6000 --ntasks 1 --gpus-per-task 4 [command]                    # all GPUs in one task
```

## Examples from the docs

Single GPU on the default partition:

```
srun -K --job-name="Project_Experiment" --gpus=1 \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-workdir="`pwd`" python train.py
```

16 GPUs, 6 CPUs each:

```
srun -K --job-name="Project_Experiment" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" \
  -p RTXA6000 --ntasks=16 --gpus-per-task=1 --gpu-bind=none --cpus-per-gpu=6 python train.py
```

## Multi-GPU inference with Ollama (this project)

Ollama is not a Slurm-distributed program: one task, one server, several GPUs. When a model does not fit on one
GPU, the server splits its layers across all GPUs the job was given (`--gpus=N`, one node). Example: Llama 4
Maverick (q4_K_M, 245 GB) on `-p B200,H200 --gpus=2` (2×183 GB or 2×141 GB). More GPUs than the model needs
do not speed up a sequential (`OLLAMA_NUM_PARALLEL=1`) run.
