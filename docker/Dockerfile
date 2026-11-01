# Runtime image for recyclebench on a Pegasus GPU node: a pinned Ollama plus the Python deps.
# The repo is mounted at run time (pegasus/run.sbatch), so code edits never need a rebuild.
# Build on the cluster with pegasus/build_image.sh (podman build -> enroot import).
ARG OLLAMA_VERSION=0.40.12
FROM docker.io/ollama/ollama:${OLLAMA_VERSION}
ARG OLLAMA_VERSION
LABEL org.opencontainers.image.title="recyclebench" ollama.version="${OLLAMA_VERSION}"

# rootless podman cannot drop privileges for apt's sandbox user (Pegasus docs, custom software)
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/sandbox-disable \
 && apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv curl ca-certificates \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt "openai>=1.30" "pytest>=7.4" \
 && pip freeze > /opt/requirements.lock

# The intent gate (run-real, tools/rpe_gate.py) needs torch + transformers; screening and
# judging do not. Build with --build-arg WITH_GATE=1 only when a job builds a manifest.
ARG WITH_GATE=0
RUN if [ "$WITH_GATE" = "1" ]; then \
      pip install "sentence-transformers>=2.2" "transformers>=4.40" "bert-score>=0.3.13" \
      && pip freeze > /opt/requirements.lock; \
    fi

ENTRYPOINT []
CMD ["bash"]
