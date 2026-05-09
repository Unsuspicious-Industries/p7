ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG PROPOSITION7_COMMIT=unknown

LABEL org.opencontainers.image.title="proposition7 benchmark artifact"
LABEL org.opencontainers.image.description="Reusable proposition7 benchmark environment with all configs for artifact evaluation"
LABEL org.opencontainers.image.revision="${PROPOSITION7_COMMIT}"
LABEL org.opencontainers.image.p7.config_dir="benchmarks/configs"
LABEL org.opencontainers.image.p7.mount_source="/workspace"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PROPOSITION7_ARTIFACT_COMMIT=${PROPOSITION7_COMMIT}
ENV UV_PROJECT_ENVIRONMENT=/opt/proposition7-venv
ENV PATH="/opt/proposition7-venv/bin:${PATH}"
ENV PYTHONPATH="/workspace/src:/workspace"
ENV HF_HOME=/cache/huggingface

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && mkdir -p /cache/huggingface \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir uv==0.11.1 \
    && uv sync --locked --extra transformers --extra outlines --extra dev

COPY . .

RUN chmod +x /workspace/scripts/*.sh \
    && python -m py_compile benchmarks/run.py benchmarks/api.py benchmarks/agg.py benchmarks/providers.py

CMD ["/bin/bash"]
