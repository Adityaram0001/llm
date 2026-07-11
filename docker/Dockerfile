# LLM-Lab training image — deps baked in, code+data pulled at start (see docker/entrypoint.sh).
#
# Build & push FROM THE MAC (pods are x86_64 — the --platform flag is MANDATORY on Apple
# Silicon, an arm64 image will not run on any rental GPU box):
#
#   docker buildx build --platform linux/amd64 \
#     -f docker/Dockerfile -t docker.io/<dockerhub-user>/llmlab:0.1 --push .
#
# Rebuild only when requirements.txt or the base image changes — NOT for code changes
# (code is git-cloned at container start, so day-to-day iteration needs no rebuild).
#
# Base must be a CUDA >= 12.8 torch build — RTX 5090 is Blackwell/sm_120 and older
# cu12.1/cu12.4 wheels cannot launch kernels on it.
ARG BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
      git tmux rsync curl ca-certificates unzip \
    && curl -fsSL https://rclone.org/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*

# Deps only — the base image already has a matching torch; pip leaves it alone.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY docker/entrypoint.sh /usr/local/bin/llmlab-entrypoint
RUN chmod +x /usr/local/bin/llmlab-entrypoint

WORKDIR /workspace
# RunPod & friends override the entrypoint with their own init; llmlab-entrypoint also
# works when invoked manually in a shell (see docs/CLOUD.md).
ENTRYPOINT ["/usr/local/bin/llmlab-entrypoint"]
CMD ["bash"]
