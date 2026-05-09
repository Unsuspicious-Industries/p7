ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG PROPOSITION7_COMMIT=unknown
ARG P7_JOB_CONFIG=benchmarks/configs/paper.toml
ARG P7_JOB_ARGS=--resume

LABEL org.opencontainers.image.title="proposition7 benchmark artifact"
LABEL org.opencontainers.image.description="Reproducible proposition7 benchmark runner for artifact evaluation"
LABEL org.opencontainers.image.revision="${PROPOSITION7_COMMIT}"
LABEL org.opencontainers.image.p7.job_config="${P7_JOB_CONFIG}"
LABEL org.opencontainers.image.p7.job_args="${P7_JOB_ARGS}"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PROPOSITION7_ARTIFACT_COMMIT=${PROPOSITION7_COMMIT}
ENV P7_JOB_CONFIG=${P7_JOB_CONFIG}
ENV P7_JOB_ARGS=${P7_JOB_ARGS}
ENV PATH="/workspace/.venv/bin:${PATH}"

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY benchmarks ./benchmarks
COPY tests ./tests
COPY artifact ./artifact

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir uv==0.11.1 \
    && uv sync --locked --extra transformers --extra outlines --extra dev \
    && chmod +x /workspace/artifact/entrypoint.sh \
    && python -m py_compile benchmarks/run.py benchmarks/api.py benchmarks/agg.py

ENTRYPOINT ["/workspace/artifact/entrypoint.sh"]
CMD ["job"]
