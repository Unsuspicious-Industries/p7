#!/usr/bin/env bash
set -euo pipefail

cd /workspace

command="${1:-job}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "$command" in
  paper)
    exec python benchmarks/run.py --config benchmarks/configs/paper.toml "$@"
    ;;
  deploy)
    exec python benchmarks/run.py --config benchmarks/configs/deploy_constrained.toml "$@"
    ;;
  openrouter)
    exec python benchmarks/run.py --config benchmarks/configs/openrouter_closed.toml "$@"
    ;;
  preferred)
    exec python benchmarks/run.py --config benchmarks/configs/vast_preferred.toml "$@"
    ;;
  job)
    config="${P7_JOB_CONFIG:-}"
    if [ -z "$config" ]; then
      printf 'P7_JOB_CONFIG is required for job images\n' >&2
      exit 2
    fi
    if [ "$#" -eq 0 ] && [ -n "${P7_JOB_ARGS:-}" ]; then
      # P7_JOB_ARGS is intentionally simple shell words, e.g. "--resume".
      # shellcheck disable=SC2086
      set -- ${P7_JOB_ARGS}
    fi
    exec python benchmarks/run.py --config "$config" "$@"
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
  smoke)
    python -m py_compile benchmarks/run.py benchmarks/api.py benchmarks/agg.py benchmarks/providers.py
    exec python benchmarks/run.py --config "${P7_JOB_CONFIG:-benchmarks/configs/paper.toml}" --dry-run
    ;;
  test)
    python -m py_compile benchmarks/run.py benchmarks/api.py benchmarks/agg.py benchmarks/providers.py
    exec python -m pytest -q tests/benchmarks_api.py "$@"
    ;;
  shell)
    exec /bin/bash "$@"
    ;;
  *)
    exec "$command" "$@"
    ;;
esac
