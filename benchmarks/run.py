#!/usr/bin/env python3
"""Benchmark runner: orchestrates jobs from a TOML config.

Utility/infrastructure helpers live in benchmarks/utils.py.
Prompt construction and interaction logic live in benchmarks/api.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing
import os
import re
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import proposition7
from benchmarks.api import (
    FatalBenchmarkInvariantError,
    append_jsonl,
    load_tasks,
    grammar_name,
    run_interaction,
)
from benchmarks.providers import OpenRouterModel, OutlinesSyntaxModel
from benchmarks.utils import (
    Job,
    build_jobs,
    clean_hf_model_cache,
    concurrent_runs_for_model,
    model_cache_key,
    model_kwargs_from_args,
    parse_model_concurrency,
    release_cached_models,
    release_torch_cuda_cache,
)


class JobTimeoutError(TimeoutError):
    pass


ARTIFACT_SCHEMA_VERSION = 1
RESUME_KEY_FIELDS = [
    "backend",
    "model",
    "task_id",
    "task_hash",
    "resolution_hash",
    "mode",
    "try",
]


@dataclass(frozen=True)
class BenchmarkMatrix:
    name: str
    backend: str
    models: list[str]
    modes: list[str]


@dataclass(frozen=True)
class BenchmarkConfig:
    source_path: Path
    raw: dict[str, Any]
    schema_version: int
    run_name: str
    output_root: Path
    save_traces: bool
    selectors: list[str]
    task_ids: list[str]
    max_tasks: int
    max_tokens_override: int
    tries: int
    seed: int
    timeout: float
    think_budget: int
    temperature: float
    model_concurrency: str
    low_space: bool
    device: str
    torch_dtype: str
    device_map: str
    model_kwargs: dict[str, Any]
    openrouter_env: str
    matrices: list[BenchmarkMatrix]


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    config_copy: Path
    raw_jsonl: Path
    traces_jsonl: Path
    results_json: Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the reproducible proposition7 benchmark artifact from a TOML config."
    )
    p.add_argument("--config", required=True, help="Benchmark config TOML path")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume the existing run directory instead of creating a new one",
    )
    p.add_argument(
        "--run-dir",
        default="",
        help="Optional explicit run directory; must not exist unless --resume is set",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print the planned run without executing jobs",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# TOML parsing helpers
# ---------------------------------------------------------------------------

def _as_dict(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"{field} must be a TOML table")
    return dict(value)


def _as_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise SystemExit(f"{field} must be a string or a list of strings")
    return [item.strip() for item in value]


def _as_bool(value: Any, field: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise SystemExit(f"{field} must be a boolean")
    return value


def _as_int(value: Any, field: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise SystemExit(f"{field} must be an integer") from e


def _as_float(value: Any, field: str, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise SystemExit(f"{field} must be a number") from e


def _as_str(value: Any, field: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise SystemExit(f"{field} must be a string")
    return value


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    run = _as_dict(raw.get("run"), "run")
    tasks = _as_dict(raw.get("tasks"), "tasks")
    execution = _as_dict(raw.get("execution"), "execution")
    local = _as_dict(raw.get("local"), "local")
    openrouter = _as_dict(raw.get("openrouter"), "openrouter")
    matrices_raw = raw.get("matrix")
    if not isinstance(matrices_raw, list) or not matrices_raw:
        raise SystemExit("config must define at least one [[matrix]] table")

    matrices: list[BenchmarkMatrix] = []
    seen_matrix_names: set[str] = set()
    seen_matrix_jobs: set[tuple[str, str, str]] = set()
    for index, item in enumerate(matrices_raw, start=1):
        entry = _as_dict(item, f"matrix[{index}]")
        name = _as_str(entry.get("name"), f"matrix[{index}].name", f"matrix-{index}")
        backend = _as_str(entry.get("backend"), f"matrix[{index}].backend", "local")
        if backend not in {"local", "openrouter"}:
            raise SystemExit(f"matrix[{index}].backend must be 'local' or 'openrouter'")
        models = _as_list(entry.get("models"), f"matrix[{index}].models")
        modes = _as_list(entry.get("modes"), f"matrix[{index}].modes")
        if not models:
            raise SystemExit(f"matrix[{index}].models must not be empty")
        if not modes:
            raise SystemExit(f"matrix[{index}].modes must not be empty")
        if name in seen_matrix_names:
            raise SystemExit(f"Duplicate matrix name: {name}")
        seen_matrix_names.add(name)
        for model_name in models:
            for mode in modes:
                key = (backend, model_name, mode)
                if key in seen_matrix_jobs:
                    raise SystemExit(
                        "Duplicate matrix job definition for "
                        f"backend={backend} model={model_name} mode={mode}"
                    )
                seen_matrix_jobs.add(key)
        matrices.append(
            BenchmarkMatrix(name=name, backend=backend, models=models, modes=modes)
        )

    model_kwargs = local.get("model_kwargs") or {}
    if not isinstance(model_kwargs, dict):
        raise SystemExit("local.model_kwargs must be a TOML table")

    if "parallel_tasks" in execution:
        raise SystemExit(
            "execution.parallel_tasks was renamed to execution.model_concurrency"
        )

    return BenchmarkConfig(
        source_path=path,
        raw=dict(raw),
        schema_version=_as_int(raw.get("schema_version"), "schema_version", 1),
        run_name=_as_str(run.get("name"), "run.name", path.stem),
        output_root=Path(_as_str(run.get("output_root"), "run.output_root", str(OUT))),
        save_traces=_as_bool(run.get("save_traces"), "run.save_traces", False),
        selectors=_as_list(tasks.get("selectors"), "tasks.selectors") or ["all"],
        task_ids=_as_list(tasks.get("ids"), "tasks.ids"),
        max_tasks=_as_int(tasks.get("max_tasks"), "tasks.max_tasks", 0),
        max_tokens_override=_as_int(
            tasks.get("max_tokens_override"), "tasks.max_tokens_override", 0
        ),
        tries=_as_int(execution.get("tries"), "execution.tries", 1),
        seed=_as_int(execution.get("seed"), "execution.seed", 7),
        timeout=_as_float(execution.get("timeout"), "execution.timeout", 0.0),
        think_budget=_as_int(execution.get("think_budget"), "execution.think_budget", 128),
        temperature=_as_float(execution.get("temperature"), "execution.temperature", 0.0),
        model_concurrency=_as_str(
            execution.get("model_concurrency"), "execution.model_concurrency", "1"
        ),
        low_space=_as_bool(execution.get("low_space"), "execution.low_space", False),
        device=_as_str(local.get("device"), "local.device", "cpu"),
        torch_dtype=_as_str(local.get("torch_dtype"), "local.torch_dtype", "auto"),
        device_map=_as_str(local.get("device_map"), "local.device_map", ""),
        model_kwargs=dict(model_kwargs),
        openrouter_env=_as_str(openrouter.get("env_file"), "openrouter.env_file", ".env"),
        matrices=matrices,
    )


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "benchmark-run"


def unique_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base_dir.with_name(f"{base_dir.name}-{stamp}")
    index = 2
    while candidate.exists():
        candidate = base_dir.with_name(f"{base_dir.name}-{stamp}-{index}")
        index += 1
    return candidate


def resolve_run_paths(
    config: BenchmarkConfig,
    *,
    resume: bool,
    explicit_run_dir: str,
) -> RunPaths:
    if explicit_run_dir:
        run_dir = Path(explicit_run_dir)
        if run_dir.exists() and not resume:
            raise SystemExit(
                f"Run directory already exists and will not be overwritten: {run_dir}"
            )
    else:
        base_dir = config.output_root / slugify_name(config.run_name)
        run_dir = base_dir if resume else unique_run_dir(base_dir)
    return RunPaths(
        run_dir=run_dir,
        config_copy=run_dir / "config.toml",
        raw_jsonl=run_dir / "raw.jsonl",
        traces_jsonl=run_dir / "traces.jsonl",
        results_json=run_dir / "results.json",
    )


def ensure_run_directory(
    config: BenchmarkConfig,
    paths: RunPaths,
    *,
    resume: bool,
) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    if paths.config_copy.exists():
        with paths.config_copy.open("rb") as handle:
            existing = tomllib.load(handle)
        if json.dumps(existing, sort_keys=True) != json.dumps(config.raw, sort_keys=True):
            raise SystemExit(
                "Refusing to mix different configs in the same run directory: "
                f"{paths.run_dir}"
            )
    else:
        paths.config_copy.write_text(
            config.source_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if not paths.raw_jsonl.exists():
        paths.raw_jsonl.write_text("", encoding="utf-8")
    if config.save_traces and not paths.traces_jsonl.exists():
        paths.traces_jsonl.write_text("", encoding="utf-8")


def write_results_summary(
    config: BenchmarkConfig,
    paths: RunPaths,
    *,
    task_count: int,
    total_jobs: int,
    planned_pending_jobs: int,
) -> None:
    summary = {
        "run": config.run_name,
        "tasks": task_count,
        "matrices": [matrix.name for matrix in config.matrices],
        "total_jobs": total_jobs,
        "planned_pending_jobs": planned_pending_jobs,
        "raw_jsonl": str(paths.raw_jsonl),
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    paths.results_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Mode validation
# ---------------------------------------------------------------------------

def normalize_modes(raw_modes: list[str], backend: str) -> list[str]:
    modes = [m.strip() for m in raw_modes if m.strip()]
    valid_modes = frozenset(proposition7.BENCHMARK_MODES)
    unknown = set(modes) - valid_modes
    if unknown:
        raise SystemExit(f"Unknown modes: {', '.join(sorted(unknown))}")
    if backend == "openrouter":
        unsupported = [
            m for m in modes
            if m not in {"unconstrained", "unconstrained_cleaned", "unconstrained_thinking"}
        ]
        if unsupported:
            raise SystemExit(
                "OpenRouter backend only supports unconstrained modes: "
                + ", ".join(unsupported)
            )
    if not modes:
        raise SystemExit("No runnable modes selected")
    return modes


def selected_tasks(config: BenchmarkConfig) -> list[Any]:
    tasks = load_tasks(config.selectors)
    task_ids = set(config.task_ids)
    if task_ids:
        tasks = [task for task in tasks if task.task_id in task_ids]
        missing = task_ids - {task.task_id for task in tasks}
        if missing:
            raise SystemExit(f"Unknown task ids: {', '.join(sorted(missing))}")
    if config.max_tasks > 0:
        tasks = tasks[: config.max_tasks]
    if config.max_tokens_override > 0:
        tasks = [replace(task, max_tokens=config.max_tokens_override) for task in tasks]
    if not tasks:
        raise SystemExit("No tasks selected")
    return tasks


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def matrix_args(config: BenchmarkConfig, matrix: BenchmarkMatrix) -> argparse.Namespace:
    return argparse.Namespace(
        backend=matrix.backend,
        timeout=config.timeout,
        seed=config.seed,
        think_budget=config.think_budget,
        temperature=config.temperature,
        device=config.device,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        model_kwargs=config.model_kwargs,
        model_kwargs_json="",
        low_space=config.low_space,
        model_concurrency=parse_model_concurrency(config.model_concurrency),
        openrouter_env=config.openrouter_env,
        matrix_name=matrix.name,
    )


def make_model(args: argparse.Namespace, model_name: str, gname: str, mode: str) -> Any:
    model_kwargs = model_kwargs_from_args(args)
    if mode in {"outlines", "outlines_mixed"}:
        return OutlinesSyntaxModel(
            model_name,
            grammar_name=gname,
            device=args.device,
            **model_kwargs,
        )
    if args.backend == "openrouter":
        return OpenRouterModel(model_name, env_path=args.openrouter_env)
    return proposition7.get_model_class(model_name).from_pretrained(
        model_name,
        grammar=proposition7.get_grammar(gname),
        device=args.device,
        **model_kwargs,
    )


def cached_model(
    args: argparse.Namespace,
    job: Job,
    cache: dict[tuple[str, ...], Any],
) -> Any:
    key = model_cache_key(args, job)
    if key not in cache:
        cache[key] = make_model(args, job.model_name, job.grammar_name, job.mode)
    elif args.backend == "local" and job.mode not in {"outlines", "outlines_mixed"}:
        # Grammar can differ per-job even within a single model; update in place.
        cache[key].grammar = proposition7.get_grammar(job.grammar_name)
    return cache[key]


# ---------------------------------------------------------------------------
# Resume / existing-record handling
# ---------------------------------------------------------------------------

def read_existing_record_keys(
    path: Path, backend: str
) -> set[tuple[str, str, str, str, str, str, int]]:
    if not path.exists() or path.is_dir():
        return set()
    keys: set[tuple[str, str, str, str, str, str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = row.get("model")
            task_id = row.get("task_id")
            mode = row.get("mode")
            task_hash = str(row.get("task_hash") or "")
            resolution_hash = str(row.get("resolution_hash") or "")
            record_backend = str(row.get("backend") or backend)
            if record_backend != backend:
                continue
            if not (model and task_id and mode):
                continue
            try:
                attempt = int(row.get("try", 0))
            except (TypeError, ValueError):
                attempt = 0
            keys.add((backend, str(model), str(task_id), task_hash, resolution_hash, str(mode), attempt))
    return keys


# ---------------------------------------------------------------------------
# Per-job execution
# ---------------------------------------------------------------------------

def error_record(job: Job, args: argparse.Namespace, error: Exception) -> dict[str, Any]:
    return {
        "task_id": job.task.task_id,
        "language": job.task.language,
        "grammar": job.task.grammar,
        "category": job.task.category,
        "task_hash": job.task.task_hash,
        "resolution_hash": job.task.resolution_hash,
        "resolution_mode": str(job.task.resolution.get("mode", "exact")),
        "mode": job.mode,
        "expected": job.task.expected,
        "output": f"ERROR: {error}",
        "exact": False,
        "parse_ok": False,
        "parse_complete": False,
        "semantic_ok": False,
        "resolution_error": "",
        "resolution_observed": None,
        "resolution_expected": None,
        "error": str(error),
        "parse_error": str(error),
        "stop_reason": "model_error",
        "tokens": 0,
        "seed": args.seed + job.attempt,
        "passed": False,
        "seconds": 0.0,
        "model": job.model_name,
        "backend": args.backend,
        "try": job.attempt,
    }


def timeout_record(job: Job, args: argparse.Namespace, seconds: float) -> dict[str, Any]:
    return {
        "task_id": job.task.task_id,
        "language": job.task.language,
        "grammar": job.task.grammar,
        "category": job.task.category,
        "task_hash": job.task.task_hash,
        "resolution_hash": job.task.resolution_hash,
        "resolution_mode": str(job.task.resolution.get("mode", "exact")),
        "mode": job.mode,
        "expected": job.task.expected,
        "output": "ERROR: timeout",
        "exact": False,
        "parse_ok": False,
        "parse_complete": False,
        "semantic_ok": False,
        "resolution_error": "",
        "resolution_observed": None,
        "resolution_expected": None,
        "error": "timeout",
        "parse_error": "timeout",
        "stop_reason": "timeout",
        "tokens": 0,
        "seed": args.seed + job.attempt,
        "passed": False,
        "seconds": seconds,
        "model": job.model_name,
        "backend": args.backend,
        "try": job.attempt,
    }


def run_with_timeout(seconds: float, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        return fn()

    def raise_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise JobTimeoutError("timeout")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_job(
    args: argparse.Namespace,
    job: Job,
    trace_enabled: bool,
    model: Optional[Any] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_buffer: list[dict[str, Any]] = []

    def execute() -> dict[str, Any]:
        nonlocal model
        model = model or make_model(args, job.model_name, job.grammar_name, job.mode)
        return run_interaction(
            model,
            job.task,
            job.mode,
            seed=args.seed + job.attempt,
            think_budget=args.think_budget,
            temperature=args.temperature,
            trace=trace_buffer if trace_enabled else None,
        )

    try:
        record = run_with_timeout(args.timeout, execute)
    except FatalBenchmarkInvariantError:
        raise
    except JobTimeoutError:
        record = timeout_record(job, args, args.timeout)
    except Exception as error:
        record = error_record(job, args, error)

    record["model"] = job.model_name
    record["backend"] = args.backend
    record["try"] = job.attempt
    for trace_record in trace_buffer:
        trace_record["model"] = job.model_name
        trace_record["backend"] = args.backend
        trace_record["try"] = job.attempt
    return record, trace_buffer


def write_outputs(
    out_path: Path,
    trace_path: Optional[Path],
    record: dict[str, Any],
    traces: list[dict[str, Any]],
) -> None:
    append_jsonl(out_path, record)
    if trace_path is not None:
        for trace_record in traces:
            append_jsonl(trace_path, trace_record)


# ---------------------------------------------------------------------------
# Concurrency / worker management
# ---------------------------------------------------------------------------

def group_jobs_by_model(jobs: list[Job]) -> list[tuple[str, list[Job]]]:
    order: list[str] = []
    groups: dict[str, list[Job]] = {}
    for job in jobs:
        if job.model_name not in groups:
            order.append(job.model_name)
            groups[job.model_name] = []
        groups[job.model_name].append(job)
    return [(model_name, groups[model_name]) for model_name in order]


def split_model_jobs(jobs: list[Job], concurrent_runs: int) -> list[list[Job]]:
    chunks: list[list[Job]] = [[] for _ in range(max(1, concurrent_runs))]
    for index, job in enumerate(jobs):
        chunks[index % len(chunks)].append(job)
    return [chunk for chunk in chunks if chunk]


def run_worker_chunk(
    worker_id: int,
    args: argparse.Namespace,
    jobs: list[Job],
    trace_enabled: bool,
    out_path: Optional[str] = None,
    trace_path: Optional[str] = None,
    write_lock: Any = None,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    cache: dict[tuple[str, ...], Any] = {}
    load_errors: dict[tuple[str, ...], Exception] = {}
    outputs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    model: Optional[Any] = None
    try:
        for job in jobs:
            started = time.monotonic()
            print(f"[start worker={worker_id}] {job.label()}", flush=True)
            key = model_cache_key(args, job)
            load_error = load_errors.get(key)
            if load_error is not None:
                record, traces = error_record(job, args, load_error), []
            else:
                try:
                    model = cached_model(args, job, cache)
                except Exception as error:
                    load_errors[key] = error
                    record, traces = error_record(job, args, error), []
                else:
                    record, traces = run_job(args, job, trace_enabled, model=model)
                    if record.get("error") == "timeout":
                        model = None
                        release_cached_models(cache)
                    else:
                        release_torch_cuda_cache()
            if out_path is None:
                outputs.append((record, traces))
            else:
                def _write() -> None:
                    write_outputs(
                        Path(out_path),
                        Path(trace_path) if trace_path else None,
                        record,
                        traces,
                    )
                if write_lock is None:
                    _write()
                else:
                    with write_lock:
                        _write()
            print(
                f"[done worker={worker_id}] {job.label()}"
                f" error={record['error']} seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
    finally:
        model = None
        release_cached_models(cache)
    return outputs


def run_model_concurrency(
    args: argparse.Namespace,
    jobs: list[Job],
    out_path: Path,
    trace_path: Optional[Path],
) -> None:
    trace_enabled = trace_path is not None
    for model_name, model_jobs in group_jobs_by_model(jobs):
        model_concurrency = concurrent_runs_for_model(args, model_name, len(model_jobs))
        print(
            f"[model-concurrency] model={model_name} jobs={len(model_jobs)}"
            f" concurrent_runs={model_concurrency}",
            flush=True,
        )
        chunks = split_model_jobs(model_jobs, model_concurrency)
        for chunk in chunks:
            if any(job.model_name != model_name for job in chunk):
                raise FatalBenchmarkInvariantError(
                    f"model concurrency mixed jobs from multiple models for {model_name}"
                )
        if getattr(args, "_test_inline_workers", False):
            for index, chunk in enumerate(chunks):
                for record, traces in run_worker_chunk(index, args, chunk, trace_enabled):
                    write_outputs(out_path, trace_path, record, traces)
            release_cached_models({})
            if args.low_space and args.backend == "local":
                clean_hf_model_cache(model_name)
            continue
        mp_context = multiprocessing.get_context("spawn")
        manager = multiprocessing.Manager()
        write_lock = manager.Lock()
        with ProcessPoolExecutor(max_workers=model_concurrency, mp_context=mp_context) as executor:
            futures = [
                executor.submit(
                    run_worker_chunk,
                    index,
                    args,
                    chunk,
                    trace_enabled,
                    str(out_path),
                    str(trace_path) if trace_path is not None else None,
                    write_lock,
                )
                for index, chunk in enumerate(chunks)
            ]
            for future in futures:
                future.result()
        futures.clear()
        manager.shutdown()
        release_cached_models({})
        if args.low_space and args.backend == "local":
            clean_hf_model_cache(model_name)


# ---------------------------------------------------------------------------
# Matrix execution
# ---------------------------------------------------------------------------

def run_matrix(
    config: BenchmarkConfig,
    matrix: BenchmarkMatrix,
    tasks: list[Any],
    paths: RunPaths,
    *,
    resume: bool,
    dry_run: bool,
) -> tuple[int, int]:
    args = matrix_args(config, matrix)
    modes = normalize_modes(matrix.modes, matrix.backend)
    jobs = build_jobs(tasks, matrix.models, modes, config.tries, grammar_name)
    planned_jobs = len(jobs)

    if resume:
        existing = read_existing_record_keys(paths.raw_jsonl, matrix.backend)
        if existing:
            before = len(jobs)
            jobs = [job for job in jobs if job.key(matrix.backend) not in existing]
            print(
                f"[resume] matrix={matrix.name} skipped {before - len(jobs)} existing jobs",
                flush=True,
            )

    print(
        f"[matrix] {matrix.name}: backend={matrix.backend} models={len(matrix.models)}"
        f" modes={','.join(modes)} planned={planned_jobs} pending={len(jobs)}",
        flush=True,
    )

    if dry_run:
        return planned_jobs, len(jobs)
    if not jobs:
        return planned_jobs, 0

    trace_path = paths.traces_jsonl if config.save_traces else None
    print(
        f"[run] matrix={matrix.name} jobs={len(jobs)}"
        f" model_concurrency={args.model_concurrency}",
        flush=True,
    )
    run_model_concurrency(args, jobs, paths.raw_jsonl, trace_path)
    return planned_jobs, 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_benchmark_config(Path(args.config))
    if config.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise SystemExit(
            f"Unsupported schema_version {config.schema_version}; expected {ARTIFACT_SCHEMA_VERSION}"
        )

    tasks = selected_tasks(config)
    paths = resolve_run_paths(config, resume=args.resume, explicit_run_dir=args.run_dir)

    total_jobs = 0
    pending_jobs = 0
    for matrix in config.matrices:
        planned, pending = run_matrix(
            config, matrix, tasks, paths, resume=args.resume, dry_run=True,
        )
        total_jobs += planned
        pending_jobs += pending

    print(
        f"[plan] run={config.run_name} tasks={len(tasks)}"
        f" matrices={len(config.matrices)} total_jobs={total_jobs}"
        f" pending_jobs={pending_jobs} run_dir={paths.run_dir}",
        flush=True,
    )
    if args.dry_run:
        return

    ensure_run_directory(config, paths, resume=args.resume)
    for matrix in config.matrices:
        run_matrix(config, matrix, tasks, paths, resume=args.resume, dry_run=False)
    write_results_summary(
        config,
        paths,
        task_count=len(tasks),
        total_jobs=total_jobs,
        planned_pending_jobs=pending_jobs,
    )


if __name__ == "__main__":
    main()
