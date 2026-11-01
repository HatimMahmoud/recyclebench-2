# Containers: software stack and custom images

Sources: [Software stack](https://pegasus.dfki.de/docs/slurm-cluster/software-stack/),
[Custom software](https://pegasus.dfki.de/docs/slurm-cluster/custom-software/). Paraphrased notes, taken 2026-11-01.

## The model

- Nodes have no user software. Everything runs in containers: **Enroot** images started through Slurm by the
  **Pyxis** plugin. Docker is not available.
- Pyxis adds options to `srun` (and `sbatch`):

```
--container-image=[USER@][REGISTRY#]IMAGE[:TAG]|PATH   docker image (enroot URI) or a .sqsh path
--container-mounts=SRC:DST[:FLAGS][,SRC:DST...]        bind mounts; flags joined with "+", e.g. ro+rprivate
--container-workdir=PATH
--container-save=PATH                                  save the modified container as a new .sqsh
--no-container-remap-root                              don't map your user to root inside the container
```

- Standard invocation:

```
srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
  [your command]
```

- Giving `--container-image` a Docker Hub name works only for small images (a few GiB); larger ones fail with
  `no space left on device`. Import them first with `enroot import` (below).
- Container start takes ~20–30 s (the `.sqsh` is copied over the network), longer under load.
- A container needs about its `.sqsh` size in RAM, on top of the job's own memory.

## Ready-made images

- `ls -1 /enroot/`; each image has a `.packages` file to grep, e.g. `grep torch==2 /enroot/*.packages`.
- Most are NVIDIA NGC images. Minimum NGC version per GPU: A100 20.06, RTX3090/RTXA6000 20.10, A10 21.05,
  H100/H200/L40S 22.09, B200 25.05, RTXB6000 25.06 (see [05-resources-and-partitions.md](05-resources-and-partitions.md)).
- `/enroot/podman+enroot.sqsh` is the image for building your own images.

## Ways to get custom software (simplest first)

1. **Install script at job start** (recommended when it installs in seconds). Upgrade pip first: old images ship
   a pip that misreads PyPI metadata. Run it with `--task-prolog="`pwd`/install.sh"` or as a wrapper that ends in
   `"$@"`. For multi-task jobs, let only `SLURM_LOCALID == 0` install and have the others wait on a done-file:

```
#!/bin/bash
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  apt update; apt install -y [...] ; apt clean
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  touch "${DONEFILE}"
else
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi
"$@"
```

2. **Modify an existing image** interactively and save it (`--mem` must exceed twice the image size):

```
srun --time=04:00:00 --immediate=3600 \
  --container-image=/enroot/[image].sqsh \
  --container-save=/netscratch/$USER/[modified image].sqsh \
  --pty /bin/bash
```

3. **Import from Docker Hub:**

```
srun enroot import -o /netscratch/$USER/nvidia_cuda_10.2-runtime-centos7.sqsh docker://nvidia/cuda:10.2-runtime-centos7
```

4. **Build a Dockerfile with Podman, then import** (what this project does). `/dev/fuse` must be mounted:

```
srun --mem=[enough] --time=04:00:00 --immediate=3600 \
  --container-image=/enroot/podman+enroot.sqsh \
  --container-mounts=/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" --pty bash
# inside:
podman build -t temp .
enroot import -o /netscratch/$USER/[name].sqsh podman://temp
```

   Rootless Podman breaks APT's sandbox (`Operation not permitted`, `Could not switch group`). Put this in the
   Dockerfile before any `apt` line:

```
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/sandbox-disable
```

   A locally built image can be moved over with `docker save -o image.tar …`, then `podman load -i image.tar` and
   `enroot import` on the cluster. Native Podman on the nodes is old and unsupported for GPU work; its images vanish
   when the job ends.

5. **virtualenv / conda** in `/netscratch/$USER` (never `$HOME`), on a small CUDA base image. Least recommended.

## How this project does it

- `docker/Dockerfile`: `ollama/ollama:<pinned version>` + Python venv with `requirements.txt` (+ `openai`, `pytest`;
  `WITH_GATE=1` adds the gate's torch stack). The repo is **mounted**, not baked in, so code edits need no rebuild.
- `pegasus/build_image.sh` submits the Podman build as an `sbatch` job (a plain `srun` dies with the SSH session).
  Build time ≈ 2 min; result `/netscratch/$USER/recyclebench/images/recyclebench_ollama-<ver>.sqsh` (5.3 GB).
- `.dockerignore` sends only `requirements.txt` into the build context, so no cache, data or results
  (harmful text) ever enter an image.
