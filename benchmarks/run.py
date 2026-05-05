#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing
import os
import platform
import re
import signal
import shutil
import socket
import subprocess
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

import p7
from benchmarks.agg import dedupe_rows, delta_rows, load_rows, summarize
from benchmarks.api import append_jsonl, load_tasks, grammar_name, run_interaction
from benchmarks.providers import OpenRouterModel, OutlinesSyntaxModel


@dataclass(frozen=True)
class Job:
    model_name: str
    grammar_name: str
    task: Any
    mode: str
    attempt: int

    def label(self) -> str:
        return f"{self.model_name} {self.task.task_id} {self.mode} try={self.attempt} grammar={self.grammar_name}"

    def key(self, backend: str) -> tuple[str, str, str, str, str, str, int]:
        return (
            backend,
            self.model_name,
            self.task.task_id,
            self.task.task_hash,
            self.task.resolution_hash,
            self.mode,
            self.attempt,
        )


class JobTimeoutError(TimeoutError):
    pass

VRAM = None

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
    parallel_tasks: str
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the reproducible p7 benchmark artifact from a TOML config."
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
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{field} must be an integer") from error
    return parsed


def _as_float(value: Any, field: str, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{field} must be a number") from error
    return parsed


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
            raise SystemExit(
                f"matrix[{index}].backend must be 'local' or 'openrouter'"
            )
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
        matrices.append(BenchmarkMatrix(name=name, backend=backend, models=models, modes=modes))

    model_kwargs = local.get("model_kwargs")
    if model_kwargs is None:
        model_kwargs = {}
    if not isinstance(model_kwargs, dict):
        raise SystemExit("local.model_kwargs must be a TOML table")

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
        think_budget=_as_int(
            execution.get("think_budget"), "execution.think_budget", 128
        ),
        parallel_tasks=_as_str(
            execution.get("parallel_tasks"), "execution.parallel_tasks", "1"
        ),
        low_space=_as_bool(execution.get("low_space"), "execution.low_space", False),
        device=_as_str(local.get("device"), "local.device", "cpu"),
        torch_dtype=_as_str(local.get("torch_dtype"), "local.torch_dtype", "auto"),
        device_map=_as_str(local.get("device_map"), "local.device_map", ""),
        model_kwargs=dict(model_kwargs),
        openrouter_env=_as_str(openrouter.get("env_file"), "openrouter.env_file", ".env"),
        matrices=matrices,
    )


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
    if resume:
        return


def normalize_modes(raw_modes: list[str], backend: str) -> list[str]:
    aliases = {
        "constrained": "constrained_direct",
        "mixed": "constrained_mixed",
    }
    modes = [aliases.get(mode.strip(), mode.strip()) for mode in raw_modes if mode.strip()]
    valid_modes = {
        "constrained_direct",
        "constrained_mixed",
        "outlines",
        "unconstrained",
        "unconstrained_raw",
    }
    unknown = set(modes) - valid_modes
    if unknown:
        raise SystemExit(f"Unknown modes: {', '.join(sorted(unknown))}")
    if backend == "openrouter":
        skipped = [
            mode for mode in modes if mode not in {"unconstrained", "unconstrained_raw"}
        ]
        if skipped:
            print(
                "[skip] OpenRouter backend only supports unconstrained modes: "
                + ", ".join(skipped),
                flush=True,
            )
        modes = [mode for mode in modes if mode in {"unconstrained", "unconstrained_raw"}]
    if not modes:
        raise SystemExit("No runnable modes selected")
    return modes


def selected_modes(args: argparse.Namespace) -> list[str]:
    return normalize_modes(
        [x.strip() for x in args.modes.split(",") if x.strip()], args.backend
    )


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


def matrix_args(config: BenchmarkConfig, matrix: BenchmarkMatrix) -> argparse.Namespace:
    return argparse.Namespace(
        backend=matrix.backend,
        timeout=config.timeout,
        seed=config.seed,
        think_budget=config.think_budget,
        device=config.device,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        model_kwargs=config.model_kwargs,
        model_kwargs_json="",
        low_space=config.low_space,
        parallel_tasks=parse_parallel_tasks(config.parallel_tasks),
        openrouter_env=config.openrouter_env,
        matrix_name=matrix.name,
    )


def git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def build_results_artifact(
    config: BenchmarkConfig,
    paths: RunPaths,
    matrices: list[BenchmarkMatrix],
    matrix_job_counts: dict[str, int],
    *,
    status: str,
    total_jobs: int,
) -> dict[str, Any]:
    raw_rows = load_rows(paths.raw_jsonl)
    rows = dedupe_rows(raw_rows)
    summary = summarize(rows, ("backend", "model", "mode", "language"))
    category_summary = summarize(
        rows, ("backend", "model", "mode", "language", "category")
    )
    created_at = datetime.now(timezone.utc).isoformat()
    if paths.results_json.exists():
        try:
            existing = json.loads(paths.results_json.read_text(encoding="utf-8"))
            created_at = str(existing.get("run", {}).get("created_at") or created_at)
        except Exception:
            pass
    completed_jobs = len(rows)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": {
            "name": config.run_name,
            "directory": str(paths.run_dir),
            "status": status,
            "created_at": created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config_file": str(paths.config_copy),
            "records_file": str(paths.raw_jsonl),
            "traces_file": str(paths.traces_jsonl) if config.save_traces else None,
            "resume_key_fields": RESUME_KEY_FIELDS,
            "raw_record_count": len(raw_rows),
            "deduped_record_count": len(rows),
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "pending_jobs": max(total_jobs - completed_jobs, 0),
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "git_commit": git_commit(ROOT.parent),
        },
        "config": config.raw,
        "matrices": [
            {
                "name": matrix.name,
                "backend": matrix.backend,
                "models": matrix.models,
                "modes": normalize_modes(matrix.modes, matrix.backend),
                "planned_jobs": matrix_job_counts[matrix.name],
            }
            for matrix in matrices
        ],
        "summary": {
            "by_model_mode_language": summary,
            "by_model_mode_language_category": category_summary,
            "deltas": delta_rows(summary, ("backend", "model", "language")),
            "deltas_by_category": delta_rows(
                category_summary, ("backend", "model", "language", "category")
            ),
        },
        "records": rows,
    }


def write_results_artifact(
    config: BenchmarkConfig,
    paths: RunPaths,
    matrices: list[BenchmarkMatrix],
    matrix_job_counts: dict[str, int],
    *,
    status: str,
    total_jobs: int,
) -> None:
    artifact = build_results_artifact(
        config,
        paths,
        matrices,
        matrix_job_counts,
        status=status,
        total_jobs=total_jobs,
    )
    paths.results_json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_jobs(tasks: list[Any], models: list[str], modes: list[str], tries: int) -> list[Job]:
    jobs: list[Job] = []
    for model_name in models:
        for task in tasks:
            gname = grammar_name(task.grammar)
            for mode in modes:
                for attempt in range(tries):
                    jobs.append(Job(model_name, gname, task, mode, attempt))
    return jobs


def model_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if hasattr(args, "model_kwargs") and getattr(args, "model_kwargs") is not None:
        loaded = getattr(args, "model_kwargs")
        if not isinstance(loaded, dict):
            raise SystemExit("model_kwargs must be a dict")
        kwargs.update(loaded)
    if args.model_kwargs_json:
        try:
            loaded = json.loads(args.model_kwargs_json)
        except json.JSONDecodeError as error:
            raise SystemExit(f"Invalid --model-kwargs-json: {error}") from error
        if not isinstance(loaded, dict):
            raise SystemExit("--model-kwargs-json must decode to a JSON object")
        kwargs.update(loaded)

    if args.torch_dtype and args.torch_dtype != "none":
        if args.torch_dtype == "auto":
            kwargs.setdefault("torch_dtype", "auto")
        else:
            import torch

            kwargs.setdefault("torch_dtype", getattr(torch, args.torch_dtype))
    if args.device_map:
        kwargs.setdefault("device_map", args.device_map)
    return kwargs


def parse_parallel_tasks(value: Any) -> int | str:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        workers = int(text)
    except ValueError as error:
        raise SystemExit("--parallel-tasks must be a positive integer or 'auto'") from error
    if workers < 1:
        raise SystemExit("--parallel-tasks must be >= 1 or 'auto'")
    return workers


def model_param_billions(model_name: str) -> Optional[float]:
    lower = model_name.lower()
    known = {
        "gpt2": 0.124,
        "distilgpt2": 0.082,
        "Qwen/Qwen3.5-0.8" : 0.8
    }
    if lower in known:
        return known[lower]

    match = re.search(r"(?:^|[^a-z0-9])e?(\d+(?:\.\d+)?)\s*b(?![a-z])", lower)
    if match:
        return float(match.group(1))
    match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*m(?![a-z])", lower)
    if match:
        return float(match.group(1)) / 1000.0
    return None


def dtype_bytes(args: argparse.Namespace) -> float:
    if args.torch_dtype == "float32" or args.torch_dtype == "none":
        return 4.0
    return 2.0


def gpu_vram_gib(args: argparse.Namespace) -> Optional[float]:
    if VRAM is not None:
        return float(VRAM)
    configured = getattr(args, "vram", None)
    if configured is not None:
        return float(configured)
    if str(args.device).startswith("cpu"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device_index = torch.cuda.current_device()
        return torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
    except Exception:
        return None


def auto_parallel_tasks(args: argparse.Namespace, model_name: str, job_count: int) -> int:
    if "gemma" in model_name.lower():
        print(
            f"[parallel] auto model={model_name} workers=1 reason=gemma_memory_cap",
            flush=True,
        )
        return 1

    vram_gib = gpu_vram_gib(args)
    params_b = model_param_billions(model_name)
    print(f" VRAM: {vram_gib}GB | PARAMS: {params_b}B")
    if vram_gib is None or params_b is None:
        print(
            f"[parallel] auto model={model_name} workers=1 reason=unknown_vram_or_model_size",
            flush=True,
        )
        return 1

    bytes_per_param = dtype_bytes(args)
    weight_gib = params_b * bytes_per_param * (1_000_000_000 / (1024 ** 3))
    per_worker_gib = max(0.5, weight_gib + 0.15)
    usable_vram_gib = max(0.0, vram_gib * 0.98)
    workers = max(1, int(usable_vram_gib // per_worker_gib))
    workers = max(1,int(min(workers, 32, job_count) / 1.5))
    print(
        f"[parallel] auto model={model_name} params={params_b:.3g}B vram={vram_gib:.1f}GiB per_worker≈{per_worker_gib:.1f}GiB workers={workers}",
        flush=True,
    )
    return workers


def parallel_workers_for_model(
    args: argparse.Namespace,
    model_name: str,
    job_count: int,
) -> int:
    configured = parse_parallel_tasks(args.parallel_tasks)
    if configured == "auto":
        return auto_parallel_tasks(args, model_name, job_count)
    return min(int(configured), job_count)


def make_model(args: argparse.Namespace, model_name: str, gname: str, mode: str) -> Any:
    model_kwargs = model_kwargs_from_args(args)
    if mode == "outlines":
        return OutlinesSyntaxModel(model_name, grammar_name=gname, device=args.device)
    if args.backend == "openrouter":
        return OpenRouterModel(model_name, env_path=args.openrouter_env)
    return p7.get_model_class(model_name).from_pretrained(
        model_name,
        grammar=p7.get_grammar(gname),
        device=args.device,
        **model_kwargs,
    )


def model_cache_key(args: argparse.Namespace, job: Job) -> tuple[str, ...]:
    if job.mode == "outlines":
        return ("outlines", job.model_name, job.grammar_name)
    if args.backend in {"local", "openrouter"}:
        return (args.backend, job.model_name)
    return (args.backend, job.model_name, job.grammar_name)


def hf_hub_cache_dir() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def hf_model_cache_name(model_name: str) -> str:
    return f"models--{model_name.replace('/', '--')}"


def release_torch_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_cached_models(cache: dict[tuple[str, ...], Any]) -> None:
    cache.clear()
    release_torch_cuda_cache()


def clean_hf_model_cache(keep_model_name: str) -> None:
    hub = hf_hub_cache_dir()
    if not hub.exists() or not hub.is_dir():
        print(f"[low-space] HF cache not found: {hub}", flush=True)
        return

    keep = hf_model_cache_name(keep_model_name)
    try:
        before_free = shutil.disk_usage(hub).free
    except OSError:
        before_free = 0

    removed = 0
    for path in sorted(hub.iterdir()):
        if not path.is_dir() or not path.name.startswith("models--"):
            continue
        if path.name == keep:
            continue
        try:
            shutil.rmtree(path)
            removed += 1
        except Exception as error:
            print(f"[low-space] failed to remove {path}: {error}", flush=True)

    try:
        after_free = shutil.disk_usage(hub).free
    except OSError:
        after_free = before_free
    freed_gb = max(0, after_free - before_free) / (1024 ** 3)
    print(
        f"[low-space] kept {keep_model_name}; removed {removed} cached HF model dirs; freed {freed_gb:.2f} GiB",
        flush=True,
    )


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


def read_existing_record_keys(path: Path, backend: str) -> set[tuple[str, str, str, str, str, str, int]]:
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
            trace=trace_buffer if trace_enabled else None,
        )

    try:
        record = run_with_timeout(args.timeout, execute)
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


def cached_model(
    args: argparse.Namespace,
    job: Job,
    cache: dict[tuple[str, ...], Any],
) -> Any:
    key = model_cache_key(args, job)
    if key not in cache:
        cache[key] = make_model(args, job.model_name, job.grammar_name, job.mode)
    elif args.backend == "local" and job.mode != "outlines":
        cache[key].grammar = p7.get_grammar(job.grammar_name)
    return cache[key]


def run_sequential(
    args: argparse.Namespace,
    jobs: list[Job],
    out_path: Path,
    trace_path: Optional[Path],
) -> None:
    cache: dict[tuple[str, ...], Any] = {}
    load_errors: dict[tuple[str, ...], Exception] = {}
    trace_enabled = trace_path is not None
    current_model_name: Optional[str] = None
    model: Optional[Any] = None
    for job in jobs:
        started = time.monotonic()
        print(f"[start] {job.label()}", flush=True)
        if (
            args.low_space
            and args.backend == "local"
            and job.model_name != current_model_name
        ):
            if current_model_name is None:
                print(f"[low-space] preparing model {job.model_name}", flush=True)
            else:
                print(
                    f"[low-space] switching model {current_model_name} -> {job.model_name}",
                    flush=True,
                )
            model = None
            release_cached_models(cache)
            clean_hf_model_cache(job.model_name)
            current_model_name = job.model_name
        key = model_cache_key(args, job)
        if key in load_errors:
            record, traces = error_record(job, args, load_errors[key]), []
        else:
            try:
                model = cached_model(args, job, cache)
            except Exception as error:
                load_errors[key] = error
                record, traces = error_record(job, args, error), []
            else:
                record, traces = run_job(args, job, trace_enabled, model=model)
        write_outputs(out_path, trace_path, record, traces)
        print(f"[done] {job.label()} error={record['error']} seconds={time.monotonic() - started:.1f}", flush=True)
    model = None
    release_cached_models(cache)


def group_jobs_by_model(jobs: list[Job]) -> list[tuple[str, list[Job]]]:
    order: list[str] = []
    groups: dict[str, list[Job]] = {}
    for job in jobs:
        if job.model_name not in groups:
            order.append(job.model_name)
            groups[job.model_name] = []
        groups[job.model_name].append(job)
    return [(model_name, groups[model_name]) for model_name in order]


def split_jobs(jobs: list[Job], workers: int) -> list[list[Job]]:
    chunks: list[list[Job]] = [[] for _ in range(max(1, workers))]
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
                    if key not in cache:
                        model = cached_model(args, job, cache)
                    else:
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
                f"[done worker={worker_id}] {job.label()} error={record['error']} seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
    finally:
        model = None
        release_cached_models(cache)
    return outputs


def run_parallel_by_model(
    args: argparse.Namespace,
    jobs: list[Job],
    out_path: Path,
    trace_path: Optional[Path],
) -> None:
    trace_enabled = trace_path is not None
    for model_name, model_jobs in group_jobs_by_model(jobs):
        if args.low_space and args.backend == "local":
            print(f"[low-space] preparing model {model_name}", flush=True)
            clean_hf_model_cache(model_name)
        group_workers = parallel_workers_for_model(args, model_name, len(model_jobs))
        print(
            f"[parallel] model={model_name} jobs={len(model_jobs)} workers={group_workers}",
            flush=True,
        )
        chunks = split_jobs(model_jobs, group_workers)
        if getattr(args, "_test_inline_workers", False):
            for index, chunk in enumerate(chunks):
                for record, traces in run_worker_chunk(index, args, chunk, trace_enabled):
                    write_outputs(out_path, trace_path, record, traces)
            release_cached_models({})
            continue
        mp_context = multiprocessing.get_context("spawn")
        manager = multiprocessing.Manager()
        write_lock = manager.Lock()
        with ProcessPoolExecutor(max_workers=group_workers, mp_context=mp_context) as executor:
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
    jobs = build_jobs(tasks, matrix.models, modes, config.tries)
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
        f"[matrix] {matrix.name}: backend={matrix.backend} models={len(matrix.models)} "
        f"modes={','.join(modes)} planned={planned_jobs} pending={len(jobs)}",
        flush=True,
    )

    if dry_run:
        return planned_jobs, len(jobs)
    if not jobs:
        return planned_jobs, 0

    trace_path = paths.traces_jsonl if config.save_traces else None
    print(
        f"[run] matrix={matrix.name} jobs={len(jobs)} parallel_tasks={args.parallel_tasks}",
        flush=True,
    )
    if args.parallel_tasks == 1:
        run_sequential(args, jobs, paths.raw_jsonl, trace_path)
    else:
        run_parallel_by_model(args, jobs, paths.raw_jsonl, trace_path)
    return planned_jobs, 0


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(Path(args.config))
    if config.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise SystemExit(
            f"Unsupported schema_version {config.schema_version}; expected {ARTIFACT_SCHEMA_VERSION}"
        )

    tasks = selected_tasks(config)
    paths = resolve_run_paths(config, resume=args.resume, explicit_run_dir=args.run_dir)

    matrix_job_counts: dict[str, int] = {}
    total_jobs = 0
    pending_jobs = 0
    for matrix in config.matrices:
        planned, pending = run_matrix(
            config,
            matrix,
            tasks,
            paths,
            resume=args.resume,
            dry_run=True,
        )
        matrix_job_counts[matrix.name] = planned
        total_jobs += planned
        pending_jobs += pending

    print(
        f"[plan] run={config.run_name} tasks={len(tasks)} matrices={len(config.matrices)} "
        f"total_jobs={total_jobs} pending_jobs={pending_jobs} run_dir={paths.run_dir}",
        flush=True,
    )
    if args.dry_run:
        return

    ensure_run_directory(config, paths, resume=args.resume)
    write_results_artifact(
        config,
        paths,
        config.matrices,
        matrix_job_counts,
        status="running",
        total_jobs=total_jobs,
    )

    try:
        for matrix in config.matrices:
            run_matrix(
                config,
                matrix,
                tasks,
                paths,
                resume=args.resume,
                dry_run=False,
            )
            write_results_artifact(
                config,
                paths,
                config.matrices,
                matrix_job_counts,
                status="running",
                total_jobs=total_jobs,
            )
    except KeyboardInterrupt:
        write_results_artifact(
            config,
            paths,
            config.matrices,
            matrix_job_counts,
            status="interrupted",
            total_jobs=total_jobs,
        )
        raise
    except Exception:
        write_results_artifact(
            config,
            paths,
            config.matrices,
            matrix_job_counts,
            status="failed",
            total_jobs=total_jobs,
        )
        raise

    write_results_artifact(
        config,
        paths,
        config.matrices,
        matrix_job_counts,
        status="completed",
        total_jobs=total_jobs,
    )
    print(f"[done] results written to {paths.results_json}", flush=True)


if __name__ == "__main__":
    main()
