# flashGNM toolchain image: CUDA + the uv environment. Used as the local
# devcontainer and as the Modal image (modal.Image.from_dockerfile). The image ships the
# *environment* only; the repo lives on /workspace (bind mount locally, added at runtime
# on Modal).
#
# CUDA 12.8 matches the torch 2.8.0 wheels in uv.lock (host driver >= 570).
# Triton 3.4 is only packaged for linux/x86_64, so build for amd64, also on Apple Silicon:
#   docker buildx build --platform linux/amd64 -t FlashGNM .
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# build-essential + python3-dev: Triton compiles a small C launcher per kernel at runtime.
RUN apt-get update && apt-get install -y \
    python3 python3-dev \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Build the venv on the distro python (3.12 on ubuntu 24.04).
ENV UV_PYTHON_DOWNLOADS=never
# The uv cache and /opt/venv sit on different filesystems (cache volume vs image).
ENV UV_LINK_MODE=copy

WORKDIR /opt/FlashGNM-lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Keep compiled Triton kernels across container restarts (volume in devcontainer.json).
ENV TRITON_CACHE_DIR=/root/.cache/triton_cache

WORKDIR /workspace
