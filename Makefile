SHELL := /usr/bin/env bash

ROOT := $(CURDIR)
IMAGE_NAME ?= proposition7-benchmark-artifact:latest
OUTPUT_TAR ?= $(ROOT)/dist/proposition7-benchmark-artifact.tar
OUTPUT_SIF ?= $(ROOT)/dist/proposition7-benchmark-artifact.sif
PYTHON_FILES := benchmarks/api.py benchmarks/run.py benchmarks/agg.py scripts/modal_sandbox_run.py
SHELL_FILES := scripts/build_artifact_image.sh scripts/build_job_image.sh scripts/run_artifact_sif.sh
TEST_FILES := tests/api.py tests/benchmarks_api.py tests/environment.py tests/grammar.py tests/llm_stop_tokens.py
JOB_CONFIG ?= benchmarks/configs/sas26_reproduction.toml

.PHONY: help build artifact docker job-image sif test paper sif-paper modal-full process clean-dist

help:
	@printf 'Targets:\n'
	@printf '  make build        Build sdist and wheel\n'
	@printf '  make artifact     Build Docker tar and, when possible, SIF\n'
	@printf '  make docker       Build Docker tar only\n'
	@printf '  make job-image    Build Docker tar for one baked benchmark job\n'
	@printf '  make sif          Build Docker tar and force SIF export\n'
	@printf '  make test         Run benchmark-facing test suite\n'
	@printf '  make paper        Run paper artifact with Docker\n'
	@printf '  make sif-paper    Run paper artifact with Apptainer/Singularity\n'
	@printf '  make modal-full   Launch full Modal small-Qwen benchmark\n'
	@printf '  make process      Process a raw.jsonl file into CSV summaries\n'

build:
	python -m build

artifact:
	IMAGE_NAME="$(IMAGE_NAME)" OUTPUT_TAR="$(OUTPUT_TAR)" OUTPUT_SIF="$(OUTPUT_SIF)" ./scripts/build_artifact_image.sh

docker:
	IMAGE_NAME="$(IMAGE_NAME)" OUTPUT_TAR="$(OUTPUT_TAR)" OUTPUT_SIF="$(OUTPUT_SIF)" EXPORT_SINGULARITY=never ./scripts/build_artifact_image.sh

job-image:
	bash ./scripts/build_job_image.sh "$(JOB_CONFIG)"

sif:
	IMAGE_NAME="$(IMAGE_NAME)" OUTPUT_TAR="$(OUTPUT_TAR)" OUTPUT_SIF="$(OUTPUT_SIF)" EXPORT_SINGULARITY=always ./scripts/build_artifact_image.sh

test:
	python -m py_compile $(PYTHON_FILES)
	bash -n $(SHELL_FILES)
	python -m pytest -q $(TEST_FILES)

paper:
	docker load -i "$(OUTPUT_TAR)"
	docker run --rm --gpus all -v "$(ROOT)/artifact-output:/workspace/benchmarks/out" -v "$(ROOT)/.env:/workspace/.env:ro" "$(IMAGE_NAME)" paper --resume

sif-paper:
	SIF_PATH="$(OUTPUT_SIF)" OUT_DIR="$(ROOT)/artifact-output" ENV_FILE="$(ROOT)/.env" ./scripts/run_artifact_sif.sh paper --resume

modal-full:
	python scripts/modal_sandbox_run.py --config benchmarks/configs/modal_qwen_full.toml --run-dir /workspace/benchmarks/out/modal-qwen-full --local-out-dir dist/modal-qwen-full --timeout 86400

process:
	@if [ -z "$(IN)" ]; then printf 'Usage: make process IN=benchmarks/out/paper/raw.jsonl OUT=benchmarks/out/paper/processed\n' >&2; exit 2; fi
	python benchmarks/agg.py --in "$(IN)" --out-dir "$(or $(OUT),$(ROOT)/processed)"

clean-dist:
	rm -rf "$(ROOT)/dist/proposition7-benchmark-artifact.tar" "$(ROOT)/dist/proposition7-benchmark-artifact.sif"
