#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="${1:-${CONFIG:-benchmarks/configs/sas26_reproduction.toml}}"
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
JOB_ARGS="${JOB_ARGS:---resume}"
EXPORT_SINGULARITY="${EXPORT_SINGULARITY:-never}"
PODMAN_SOCKET="${PODMAN_SOCKET:-/tmp/p7-job-podman-$$.sock}"
podman_service_pid=""
docker_cmd=(docker)

cleanup() {
  if [ -n "$podman_service_pid" ]; then
    kill "$podman_service_pid" >/dev/null 2>&1 || true
    wait "$podman_service_pid" >/dev/null 2>&1 || true
  fi
  if [ -n "${PODMAN_SOCKET:-}" ]; then
    rm -f "$PODMAN_SOCKET"
  fi
}
trap cleanup EXIT

start_nix_podman_socket() {
  if ! command -v nix >/dev/null 2>&1; then
    return 1
  fi
  policy_dir="$HOME/.config/containers"
  policy_file="$policy_dir/policy.json"
  if [ ! -f "$policy_file" ]; then
    mkdir -p "$policy_dir"
    printf '{"default":[{"type":"insecureAcceptAnything"}]}' > "$policy_file"
  fi
  rm -f "$PODMAN_SOCKET"
  nix shell nixpkgs#podman -c podman system service --time=0 "unix://$PODMAN_SOCKET" \
    >"${PODMAN_SOCKET}.log" 2>&1 &
  podman_service_pid="$!"
  for _ in $(seq 1 30); do
    if [ -S "$PODMAN_SOCKET" ]; then
      export DOCKER_HOST="unix://$PODMAN_SOCKET"
      return 0
    fi
    sleep 1
  done
  printf 'failed to start Nix-provided Podman Docker-compatible socket\n' >&2
  if [ -f "${PODMAN_SOCKET}.log" ]; then
    cat "${PODMAN_SOCKET}.log" >&2
  fi
  return 1
}

ensure_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    if command -v nix >/dev/null 2>&1; then
      docker_cmd=(nix shell nixpkgs#docker -c docker)
    else
      printf 'docker is required to build a job image\n' >&2
      exit 127
    fi
  fi
  if "${docker_cmd[@]}" info >/dev/null 2>&1; then
    return 0
  fi
  if start_nix_podman_socket && "${docker_cmd[@]}" info >/dev/null 2>&1; then
    printf 'Using Nix-provided rootless Podman socket as Docker daemon: %s\n' "$DOCKER_HOST"
    return 0
  fi
  printf 'docker daemon is not reachable and Nix Podman fallback failed\n' >&2
  exit 127
}

ensure_docker

case "$CONFIG" in
  /*) config_abs="$CONFIG" ;;
  *) config_abs="$ROOT/$CONFIG" ;;
esac

if [ ! -f "$config_abs" ]; then
  printf 'config not found: %s\n' "$CONFIG" >&2
  exit 2
fi

case "$config_abs" in
  "$ROOT"/*) config_rel="${config_abs#"$ROOT"/}" ;;
  *)
    printf 'config must live under repository root: %s\n' "$config_abs" >&2
    exit 2
    ;;
esac

config_base="$(basename "$config_rel")"
slug="${config_base%.toml}"
slug="$(printf '%s' "$slug" | tr -c 'A-Za-z0-9_.-' '-')"

IMAGE_NAME="${IMAGE_NAME:-proposition7-benchmark-${slug}:latest}"
OUTPUT_TAR="${OUTPUT_TAR:-$ROOT/dist/${slug}.docker.tar}"
OUTPUT_SIF="${OUTPUT_SIF:-$ROOT/dist/${slug}.sif}"
DRY_RUN_LOG="${DRY_RUN_LOG:-$ROOT/dist/${slug}.dry-run.txt}"
MANIFEST="${MANIFEST:-$ROOT/dist/${slug}.manifest.txt}"
GIT_STATUS_FILE="${GIT_STATUS_FILE:-$ROOT/dist/${slug}.git-status.txt}"
SOURCE_DIFF_FILE="${SOURCE_DIFF_FILE:-$ROOT/dist/${slug}.source-diff.patch}"

PROPOSITION7_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
CONFIG_SHA256="$(sha256sum "$config_abs" | cut -d ' ' -f 1)"
UV_LOCK_SHA256="$(sha256sum "$ROOT/uv.lock" | cut -d ' ' -f 1)"
DOCKERFILE_SHA256="$(sha256sum "$ROOT/artifact/Dockerfile" | cut -d ' ' -f 1)"

mkdir -p "$(dirname "$OUTPUT_TAR")"

"${docker_cmd[@]}" build \
  --load \
  --pull \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "PROPOSITION7_COMMIT=$PROPOSITION7_COMMIT" \
  --build-arg "P7_JOB_CONFIG=$config_rel" \
  --build-arg "P7_JOB_ARGS=$JOB_ARGS" \
  --label "org.opencontainers.image.p7.config_sha256=$CONFIG_SHA256" \
  --label "org.opencontainers.image.p7.uv_lock_sha256=$UV_LOCK_SHA256" \
  --label "org.opencontainers.image.p7.dockerfile_sha256=$DOCKERFILE_SHA256" \
  --tag "$IMAGE_NAME" \
  --file "$ROOT/artifact/Dockerfile" \
  "$ROOT"

"${docker_cmd[@]}" run --rm "$IMAGE_NAME" job --dry-run | tee "$DRY_RUN_LOG"
"${docker_cmd[@]}" save "$IMAGE_NAME" --output "$OUTPUT_TAR"
git -C "$ROOT" status --short > "$GIT_STATUS_FILE" 2>&1 || true
git -C "$ROOT" diff --binary > "$SOURCE_DIFF_FILE" 2>&1 || true

{
  printf 'image_name=%s\n' "$IMAGE_NAME"
  printf 'output_tar=%s\n' "$OUTPUT_TAR"
  printf 'job_config=%s\n' "$config_rel"
  printf 'job_args=%s\n' "$JOB_ARGS"
  printf 'config_sha256=%s\n' "$CONFIG_SHA256"
  printf 'uv_lock_sha256=%s\n' "$UV_LOCK_SHA256"
  printf 'dockerfile_sha256=%s\n' "$DOCKERFILE_SHA256"
  printf 'git_commit=%s\n' "$PROPOSITION7_COMMIT"
  printf 'git_status_file=%s\n' "$GIT_STATUS_FILE"
  printf 'source_diff_file=%s\n' "$SOURCE_DIFF_FILE"
  printf 'base_image=%s\n' "$BASE_IMAGE"
  printf 'dry_run_log=%s\n' "$DRY_RUN_LOG"
  printf 'built_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$MANIFEST"

case "$EXPORT_SINGULARITY" in
  never)
    ;;
  auto|always)
    if command -v apptainer >/dev/null 2>&1; then
      apptainer build "$OUTPUT_SIF" "docker-archive:$OUTPUT_TAR"
    elif command -v singularity >/dev/null 2>&1; then
      singularity build "$OUTPUT_SIF" "docker-archive:$OUTPUT_TAR"
    elif command -v nix >/dev/null 2>&1; then
      nix shell nixpkgs#apptainer -c apptainer build "$OUTPUT_SIF" "docker-archive:$OUTPUT_TAR"
    elif [ "$EXPORT_SINGULARITY" = "always" ]; then
      printf 'apptainer, singularity, or nix is required to export a SIF image\n' >&2
      exit 127
    fi
    ;;
  *)
    printf 'EXPORT_SINGULARITY must be one of: auto, always, never\n' >&2
    exit 2
    ;;
esac

printf 'Built job image: %s\n' "$IMAGE_NAME"
printf 'Saved Docker archive: %s\n' "$OUTPUT_TAR"
printf 'Saved manifest: %s\n' "$MANIFEST"
printf 'Saved dry-run log: %s\n' "$DRY_RUN_LOG"
if [ -f "$OUTPUT_SIF" ]; then
  printf 'Saved SIF image: %s\n' "$OUTPUT_SIF"
fi

printf '\nReviewer run command:\n'
printf '  docker load -i %s\n' "$OUTPUT_TAR"
printf '  docker run --rm --gpus all -v "$PWD/artifact-output:/workspace/benchmarks/out" -v "$PWD/.env:/workspace/.env:ro" %s\n' "$IMAGE_NAME"
printf '\nReviewer dry-run command:\n'
printf '  docker run --rm %s job --dry-run\n' "$IMAGE_NAME"
