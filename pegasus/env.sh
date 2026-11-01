# Shared paths and Ollama settings for the Pegasus scripts. Source it; override any RB_* first.
# Layout follows the Pegasus storage rules: code in $HOME, everything large in /netscratch.
export RB_REPO=${RB_REPO:-$HOME/repos/recyclebench-2}
export RB_SCRATCH=${RB_SCRATCH:-/netscratch/$USER/recyclebench}
export RB_OLLAMA_VERSION=${RB_OLLAMA_VERSION:-0.40.12}
export RB_IMAGE=${RB_IMAGE:-$RB_SCRATCH/images/recyclebench_ollama-$RB_OLLAMA_VERSION.sqsh}
export RB_LOGS=$RB_SCRATCH/logs

export OLLAMA_MODELS=$RB_SCRATCH/ollama          # ~475 GB with Scout, Maverick, the 70B lineage; no backup, re-pullable
export OLLAMA_MAX_LOADED_MODELS=2                # target + judge resident (CLAUDE.md)
export OLLAMA_KEEP_ALIVE=1h
export OLLAMA_NUM_PARALLEL=1                     # sequential, as on the Mac: batching can change greedy output
export OLLAMA_CONTEXT_LENGTH=8192                # newer Ollama sizes context by VRAM; prompts need < 2k
export OLLAMA_LOAD_TIMEOUT=30m                   # Scout is 67 GB on /netscratch (~216 MB/s): the 5 min default never loads it
export OLLAMA_NOPRUNE=1                          # jobs share one model dir: a server starting while another job
                                                 # pulls would delete that pull's finished-but-unlisted blobs
export HF_HOME=$RB_SCRATCH/hf
