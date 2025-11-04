# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

# Enable faster installs and deterministic builds
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=2

WORKDIR /app

# Copy dependency list first for caching
COPY requirements.txt .

# Leverage BuildKit cache mounts for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy source code last (changes here won't invalidate dependency layer)
COPY scripts ./scripts
COPY processed_npz ./processed_npz

CMD ["bash"]
