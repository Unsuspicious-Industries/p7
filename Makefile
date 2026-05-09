SHELL := /usr/bin/env bash

ROOT := $(CURDIR)
IMAGE_NAME ?= proposition7-benchmark-artifact:latest
OUTPUT_TAR ?= $(ROOT)/dist/proposition7-benchmark-artifact.tar
OUTPUT_BUNDLE ?= $(ROOT)/dist/proposition7-review-bundle.tar.gz
PYTHON_FILES := benchmarks/api.py benchmarks/run.py benchmarks/agg.py benchmarks/providers.py
SHELL_FILES := scripts/build_artifact_image.sh
TEST_FILES := tests/api.py tests/benchmarks_api.py tests/environment.py tests/grammar.py tests/llm_stop_tokens.py
RUN_CONFIG ?= benchmarks/configs/sas26_reproduction.toml
RUN_ARGS ?= --resume

.PHONY: help build artifact docker dry-run paper test process clean-dist

help:
	@printf 'Targets:\n'
	@printf '  make build        Build sdist and wheel\n'
	@printf '  make artifact     Build Docker image tar, source bundle, and manifest\n'
	@printf '  make docker       Alias for make artifact\n'
	@printf '  make dry-run      Dry-run RUN_CONFIG in the artifact image\n'
	@printf '  make paper        Run RUN_CONFIG in Docker with output/cache mounts\n'
	@printf '  make test         Run benchmark-facing test suite\n'
	@printf '  make process      Process a raw.jsonl file into CSV summaries\n'

build:
	python -m build

artifact:
	IMAGE_NAME="$(IMAGE_NAME)" OUTPUT_TAR="$(OUTPUT_TAR)" OUTPUT_BUNDLE="$(OUTPUT_BUNDLE)" ./scripts/build_artifact_image.sh

docker:
	$(MAKE) artifact

test:
	python -m py_compile $(PYTHON_FILES)
	bash -n $(SHELL_FILES)
	python -m pytest -q $(TEST_FILES)

dry-run:
	docker load -i "$(OUTPUT_TAR)"
	docker run --rm "$(IMAGE_NAME)" python benchmarks/run.py --config "$(RUN_CONFIG)" --dry-run

paper:
	docker load -i "$(OUTPUT_TAR)"
	mkdir -p "$(ROOT)/artifact-output" "$(ROOT)/hf-cache"
	docker run --rm --gpus all --env-file "$(ROOT)/.env" -v "$(ROOT)/artifact-output:/workspace/benchmarks/out" -v "$(ROOT)/hf-cache:/cache/huggingface" "$(IMAGE_NAME)" python benchmarks/run.py --config "$(RUN_CONFIG)" $(RUN_ARGS)

process:
	@if [ -z "$(IN)" ]; then printf 'Usage: make process IN=benchmarks/out/paper/raw.jsonl OUT=benchmarks/out/paper/processed\n' >&2; exit 2; fi
	python benchmarks/agg.py --in "$(IN)" --out-dir "$(or $(OUT),$(ROOT)/processed)"

clean-dist:
	rm -rf "$(ROOT)/dist/proposition7-benchmark-artifact.tar" "$(ROOT)/dist/proposition7-review-bundle.tar.gz" "$(ROOT)/dist/proposition7-review-manifest.txt" "$(ROOT)/dist/proposition7-review.git-status.txt" "$(ROOT)/dist/proposition7-review.source-diff.patch"
