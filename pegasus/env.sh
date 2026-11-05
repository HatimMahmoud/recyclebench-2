# Shared paths and Ollama settings for the Pegasus scripts. Source it; override any RB_* first.
# Layout follows the Pegasus storage rules: code in $HOME, everything large in /netscratch.
export RB_REPO=${RB_REPO:-$HOME/repos/recyclebench-2}
export RB_SCRATCH=${RB_SCRATCH:-/netscratch/$USER/recyclebench}
export RB_OLLAMA_VERSION=${RB_OLLAMA_VERSION:-0.40.12}
export RB_IMAGE=${RB_IMAGE:-$RB_SCRATCH/images/recyclebench_ollama-$RB_OLLAMA_VERSION.sqsh}
export RB_LOGS=$RB_SCRATCH/logs
# The image has no git: read HEAD here on the host (srun passes the environment into the container, where
# this line keeps the inherited value), so the experiment driver can record its code revision.
export RECYCLEBENCH_CODE_REVISION=${RECYCLEBENCH_CODE_REVISION:-$(git -C "$RB_REPO" rev-parse HEAD 2>/dev/null || true)}

export OLLAMA_MODELS=$RB_SCRATCH/ollama          # ~475 GB with Scout, Maverick, the 70B lineage; no backup, re-pullable
export OLLAMA_MAX_LOADED_MODELS=2                # target + judge resident (CLAUDE.md)
export OLLAMA_KEEP_ALIVE=1h
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-1}   # sequential, as on the Mac: batching can change greedy output.
                                                 # Only a job whose outputs are never compared token for token
                                                 # (the PTP synthetic corpus) may raise it.
export OLLAMA_CONTEXT_LENGTH=8192                # newer Ollama sizes context by VRAM; prompts need < 2k
export OLLAMA_LOAD_TIMEOUT=30m                   # Scout is 67 GB on /netscratch (~216 MB/s): the 5 min default never loads it
export OLLAMA_NOPRUNE=1                          # jobs share one model dir: a server starting while another job
                                                 # pulls would delete that pull's finished-but-unlisted blobs
export HF_HOME=$RB_SCRATCH/hf

# One BLAS thread per process. The Pegasus known-issues page recommends it (a library that spawns a
# thread per node CPU makes jobs slower, mostly system time), and unpinned threads are the leading
# suspect for the container-only failure of test_study_is_byte_reproducible_under_same_seed
# (docs/pegasus/LESSONS.md §6.5). Harmless here: nothing in this project is BLAS-bound.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
