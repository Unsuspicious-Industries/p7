#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${SIF_PATH:-$ROOT/dist/proposition7-benchmark-artifact.sif}"
OUT_DIR="${OUT_DIR:-$ROOT/artifact-output}"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
MODE="${1:-paper}"

runtime_cmd=()

resolve_runtime() {
  if command -v apptainer >/dev/null 2>&1; then
    runtime_cmd=(apptainer)
    return 0
  fi
  if command -v singularity >/dev/null 2>&1; then
    runtime_cmd=(singularity)
    return 0
  fi
  if command -v nix >/dev/null 2>&1; then
    runtime_cmd=(nix shell nixpkgs#apptainer -c apptainer)
    return 0
  fi
  return 1
}

if ! resolve_runtime; then
  printf 'apptainer, singularity, or nix is required\n' >&2
  exit 127
fi

if [ ! -f "$SIF_PATH" ]; then
  printf 'SIF artifact not found: %s\n' "$SIF_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

shift || true

binds=(
  "$OUT_DIR:/workspace/benchmarks/out"
)
if [ -f "$ENV_FILE" ]; then
  binds+=("$ENV_FILE:/workspace/.env:ro")
fi

bind_arg=$(IFS=, ; printf '%s' "${binds[*]}")

exec "${runtime_cmd[@]}" run \
  --cleanenv \
  --containall \
  --writable-tmpfs \
  --bind "$bind_arg" \
  --nv \
  "$SIF_PATH" \
  "$MODE" \
  "$@"
