#!/usr/bin/env bash
set -euo pipefail

cd /workspace

command="${1:-smoke}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "$command" in
  smoke)
    exec python benchmarks/run.py --config benchmarks/configs/smoke.toml "$@"
    ;;
  paper)
    exec python benchmarks/run.py --config benchmarks/configs/paper.toml "$@"
    ;;
  run)
    if [ "$#" -lt 1 ]; then
      printf 'usage: run <config.toml> [runner args...]\n' >&2
      exit 2
    fi
    config="$1"
    shift
    exec python benchmarks/run.py --config "$config" "$@"
    ;;
  test)
    python -m py_compile benchmarks/run.py benchmarks/api.py benchmarks/agg.py
    exec python -m pytest -q tests/benchmarks_api.py "$@"
    ;;
  shell)
    exec /bin/bash "$@"
    ;;
  *)
    exec "$command" "$@"
    ;;
esac
