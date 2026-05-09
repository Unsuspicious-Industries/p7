#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-proposition7-benchmark-artifact:latest}"
OUTPUT_TAR="${OUTPUT_TAR:-$ROOT/dist/proposition7-benchmark-artifact.tar}"
OUTPUT_BUNDLE="${OUTPUT_BUNDLE:-$ROOT/dist/proposition7-review-bundle.tar.gz}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-$ROOT/dist/proposition7-review-manifest.txt}"
GIT_STATUS_FILE="${GIT_STATUS_FILE:-$ROOT/dist/proposition7-review.git-status.txt}"
SOURCE_DIFF_FILE="${SOURCE_DIFF_FILE:-$ROOT/dist/proposition7-review.source-diff.patch}"
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
DOCKERFILE="$ROOT/artifact/Dockerfile"

docker_cmd=(docker)

ensure_docker() {
  if command -v podman >/dev/null 2>&1; then
    docker_cmd=(podman)
    return 0
  fi
  if command -v docker >/dev/null 2>&1; then
    docker_cmd=(docker)
    if "${docker_cmd[@]}" info >/dev/null 2>&1; then
      return 0
    fi
  fi
  if command -v nix >/dev/null 2>&1; then
    docker_cmd=(nix shell nixpkgs#podman -c podman)
    return 0
  fi
  printf 'podman or docker is required to build the artifact image\n' >&2
  exit 127
}

create_source_bundle() {
  local parent root_name
  parent="$(dirname "$ROOT")"
  root_name="$(basename "$ROOT")"

  tar -C "$parent" \
    --exclude="$root_name/.git" \
    --exclude="$root_name/.github" \
    --exclude="$root_name/.env" \
    --exclude="$root_name/.env.*" \
    --exclude="$root_name/.mypy_cache" \
    --exclude="$root_name/**/.cache" \
    --exclude="$root_name/.nix-pip" \
    --exclude="$root_name/.pytest_cache" \
    --exclude="$root_name/.ruff_cache" \
    --exclude="$root_name/.venv" \
    --exclude="$root_name/.venv-nix" \
    --exclude="$root_name/.vscode" \
    --exclude="$root_name/**/__pycache__" \
    --exclude="$root_name/**/*.pyc" \
    --exclude="$root_name/artifact-output" \
    --exclude="$root_name/benchmarks/out" \
    --exclude="$root_name/backup" \
    --exclude="$root_name/build" \
    --exclude="$root_name/checkpoints" \
    --exclude="$root_name/**/checkpoints" \
    --exclude="$root_name/dist" \
    --exclude="$root_name/evals" \
    --exclude="$root_name/final" \
    --exclude="$root_name/hf-cache" \
    --exclude="$root_name/**/hf-cache" \
    --exclude="$root_name/huggingface" \
    --exclude="$root_name/**/huggingface" \
    --exclude="$root_name/model-cache" \
    --exclude="$root_name/**/model-cache" \
    --exclude="$root_name/models" \
    --exclude="$root_name/processed" \
    --exclude="$root_name/result" \
    --exclude="$root_name/unified" \
    --exclude="$root_name/vast_model_matrix" \
    --exclude="$root_name/**/*.safetensors" \
    --exclude="$root_name/**/*.bin" \
    --exclude="$root_name/**/*.ckpt" \
    --exclude="$root_name/**/*.gguf" \
    --exclude="$root_name/**/*.onnx" \
    --exclude="$root_name/**/*.pt" \
    --exclude="$root_name/**/*.pth" \
    -czf "$OUTPUT_BUNDLE" \
    "$root_name"
}

write_manifest() {
  local config_abs config_rel config_sha

  git -C "$ROOT" status --short > "$GIT_STATUS_FILE" 2>&1 || true
  git -C "$ROOT" diff --binary > "$SOURCE_DIFF_FILE" 2>&1 || true

  {
    printf 'image_name=%s\n' "$IMAGE_NAME"
    printf 'docker_archive=%s\n' "$OUTPUT_TAR"
    printf 'source_bundle=%s\n' "$OUTPUT_BUNDLE"
    printf 'dockerfile=%s\n' "artifact/Dockerfile"
    printf 'config_dir=benchmarks/configs\n'
    printf 'image_contains=source_tree,benchmark_configs,python_dependencies\n'
    printf 'image_excludes=api_keys,model_weights,model_caches,benchmark_outputs,backup\n'
    printf 'model_cache_mount=/cache/huggingface\n'
    printf 'output_mount=/workspace/benchmarks/out\n'
    for config_abs in "$ROOT"/benchmarks/configs/*.toml; do
      [ -e "$config_abs" ] || continue
      config_rel="${config_abs#"$ROOT"/}"
      config_sha="$(sha256sum "$config_abs" | cut -d ' ' -f 1)"
      printf 'config=%s sha256=%s\n' "$config_rel" "$config_sha"
    done
    printf 'uv_lock_sha256=%s\n' "$(sha256sum "$ROOT/uv.lock" | cut -d ' ' -f 1)"
    printf 'dockerfile_sha256=%s\n' "$(sha256sum "$DOCKERFILE" | cut -d ' ' -f 1)"
    printf 'dockerignore_sha256=%s\n' "$(sha256sum "$ROOT/.dockerignore" | cut -d ' ' -f 1)"
    printf 'git_commit=%s\n' "$PROPOSITION7_COMMIT"
    printf 'git_status_file=%s\n' "$GIT_STATUS_FILE"
    printf 'source_diff_file=%s\n' "$SOURCE_DIFF_FILE"
    printf 'base_image=%s\n' "$BASE_IMAGE"
    printf 'built_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$OUTPUT_MANIFEST"
}

ensure_docker

PROPOSITION7_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"

mkdir -p "$(dirname "$OUTPUT_TAR")"
mkdir -p "$(dirname "$OUTPUT_BUNDLE")"
mkdir -p "$(dirname "$OUTPUT_MANIFEST")"

"${docker_cmd[@]}" build \
  --pull=false \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "PROPOSITION7_COMMIT=$PROPOSITION7_COMMIT" \
  --tag "$IMAGE_NAME" \
  --file "$DOCKERFILE" \
  "$ROOT"

"${docker_cmd[@]}" save "$IMAGE_NAME" --output "$OUTPUT_TAR"
create_source_bundle
write_manifest

printf 'Built Docker image: %s\n' "$IMAGE_NAME"
printf 'Saved Docker archive: %s\n' "$OUTPUT_TAR"
printf 'Saved source bundle: %s\n' "$OUTPUT_BUNDLE"
printf 'Saved manifest: %s\n' "$OUTPUT_MANIFEST"
printf '\nDry-run the SAS reproduction config:\n'
printf '  docker load -i %s\n' "$OUTPUT_TAR"
printf '  docker run --rm %s python benchmarks/run.py --config benchmarks/configs/sas26_reproduction.toml --dry-run\n' "$IMAGE_NAME"
printf '\nRun the SAS reproduction config on a GPU host:\n'
printf '  mkdir -p artifact-output hf-cache\n'
printf '  docker run --rm --gpus all --env-file .env -v "$PWD/artifact-output:/workspace/benchmarks/out" -v "$PWD/hf-cache:/cache/huggingface" %s python benchmarks/run.py --config benchmarks/configs/sas26_reproduction.toml --resume\n' "$IMAGE_NAME"
