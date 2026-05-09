#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-proposition7-benchmark-artifact:latest}"
OUTPUT_TAR="${OUTPUT_TAR:-$ROOT/dist/proposition7-benchmark-artifact.tar}"
OUTPUT_SIF="${OUTPUT_SIF:-$ROOT/dist/proposition7-benchmark-artifact.sif}"
EXPORT_SINGULARITY="${EXPORT_SINGULARITY:-auto}"
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
DOCKERFILE="$ROOT/artifact/Dockerfile"
PODMAN_SOCKET="${PODMAN_SOCKET:-/tmp/p7-artifact-podman-$$.sock}"

apptainer_cmd=()
docker_cmd=(docker)
podman_service_pid=""

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

resolve_apptainer() {
  if command -v apptainer >/dev/null 2>&1; then
    apptainer_cmd=(apptainer)
    return 0
  fi
  if command -v singularity >/dev/null 2>&1; then
    apptainer_cmd=(singularity)
    return 0
  fi
  if command -v nix >/dev/null 2>&1; then
    apptainer_cmd=(nix shell nixpkgs#apptainer -c apptainer)
    return 0
  fi
  return 1
}

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
  if command -v podman >/dev/null 2>&1; then
    docker_cmd=(podman)
    export DOCKER_HOST=""
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
    export DOCKER_HOST=""
    return 0
  fi
  printf 'podman or docker is required to build the artifact image\n' >&2
  exit 127
}

ensure_docker

PROPOSITION7_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"

mkdir -p "$(dirname "$OUTPUT_TAR")"

  "${docker_cmd[@]}" build \
    --load \
    --pull=false \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "PROPOSITION7_COMMIT=$PROPOSITION7_COMMIT" \
    --tag "$IMAGE_NAME" \
    --file "$DOCKERFILE" \
    "$ROOT"

"${docker_cmd[@]}" save "$IMAGE_NAME" --output "$OUTPUT_TAR"

case "$EXPORT_SINGULARITY" in
  auto)
    if resolve_apptainer; then
      "${apptainer_cmd[@]}" build "$OUTPUT_SIF" "docker-archive:$OUTPUT_TAR"
    fi
    ;;
  always)
    if ! resolve_apptainer; then
      printf 'apptainer, singularity, or nix is required to export a SIF image\n' >&2
      exit 127
    fi
    "${apptainer_cmd[@]}" build "$OUTPUT_SIF" "docker-archive:$OUTPUT_TAR"
    ;;
  never)
    ;;
  *)
    printf 'EXPORT_SINGULARITY must be one of: auto, always, never\n' >&2
    exit 2
    ;;
esac

printf 'Built %s\n' "$IMAGE_NAME"
printf 'Saved %s\n' "$OUTPUT_TAR"
if [ -f "$OUTPUT_SIF" ]; then
  printf 'Saved %s\n' "$OUTPUT_SIF"
elif [ "$EXPORT_SINGULARITY" = "auto" ]; then
  printf 'Skipped SIF export: no apptainer/singularity binary found and no nix fallback available\n'
fi
printf '\nRun paper suite on GPU host:\n'
printf '  docker load -i %s\n' "$OUTPUT_TAR"
printf '  docker run --rm --gpus all -v "$PWD/artifact-output:/workspace/benchmarks/out" -v "$PWD/.env:/workspace/.env:ro" %s paper --resume\n' "$IMAGE_NAME"
if [ -f "$OUTPUT_SIF" ] || [ "$EXPORT_SINGULARITY" != "never" ]; then
  printf '\nRun paper suite with Apptainer/Singularity on a GPU host:\n'
  printf '  ./scripts/run_artifact_sif.sh paper --resume\n'
fi
