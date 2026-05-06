# syntax=docker/dockerfile:1
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG P7_COMMIT=unknown

LABEL org.opencontainers.image.title="p7 benchmark artifact"
LABEL org.opencontainers.image.description="Reproducible p7 benchmark runner for artifact evaluation"
LABEL org.opencontainers.image.revision="${P7_COMMIT}"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV P7_ARTIFACT_COMMIT=${P7_COMMIT}

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY benchmarks ./benchmarks
COPY tests ./tests
COPY artifact ./artifact

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[transformers,outlines,dev]" \
    && chmod +x /workspace/artifact/entrypoint.sh \
    && python -m py_compile benchmarks/run.py benchmarks/api.py benchmarks/agg.py

ENTRYPOINT ["/workspace/artifact/entrypoint.sh"]
CMD ["smoke"]
