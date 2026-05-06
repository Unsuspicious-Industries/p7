#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-p7-benchmark-artifact:latest}"
OUTPUT_TAR="${OUTPUT_TAR:-$ROOT/dist/p7-benchmark-artifact.tar}"
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
DOCKERFILE="$ROOT/artifact/Dockerfile"

if ! command -v docker >/dev/null 2>&1; then
  printf 'docker is required to build the artifact image\n' >&2
  exit 127
fi

P7_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"

mkdir -p "$(dirname "$OUTPUT_TAR")"

docker build \
  --pull \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "P7_COMMIT=$P7_COMMIT" \
  --tag "$IMAGE_NAME" \
  --file "$DOCKERFILE" \
  "$ROOT"

docker save "$IMAGE_NAME" --output "$OUTPUT_TAR"

printf 'Built %s\n' "$IMAGE_NAME"
printf 'Saved %s\n' "$OUTPUT_TAR"
printf '\nRun smoke suite:\n'
printf '  docker load -i %s\n' "$OUTPUT_TAR"
printf '  docker run --rm -v "$PWD/artifact-output:/workspace/benchmarks/out" %s smoke\n' "$IMAGE_NAME"
printf '\nRun paper suite on GPU host:\n'
printf '  docker run --rm --gpus all -v "$PWD/artifact-output:/workspace/benchmarks/out" -v "$PWD/.env:/workspace/.env:ro" %s paper --resume\n' "$IMAGE_NAME"
