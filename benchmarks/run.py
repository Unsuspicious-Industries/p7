#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import signal
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT.parent))

import p7
from benchmarks.api import DEFAULT_MODELS, append_jsonl, load_tasks, grammar_name, run_interaction
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run p7 benchmark suite.")
    p.add_argument("--tasks", default="all", help="Comma list of task selectors (all/core/task_id/grammar/category)")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma list of HF model ids")
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--seed", type=int, default=7, help="Base RNG seed; attempt index is added")
    p.add_argument("--max-tasks", type=int, default=0, help="0 means all")
    p.add_argument("--max-tokens-override", type=int, default=0, help="Override task max_tokens for smoke runs")
    p.add_argument("--task-ids", default="", help="Optional comma-separated task ids to run")
    p.add_argument("--out", default=str(OUT / "raw.jsonl"))
    p.add_argument("--append", action="store_true", help="Append to existing output files instead of overwriting")
    p.add_argument("--trace-out", default="", help="Optional JSONL path for structured per-task token logs")
    p.add_argument("--resume", "--skip-existing", dest="resume", action="store_true", help="Append to existing outputs and skip completed jobs")
    p.add_argument("--backend", choices=["local", "openrouter"], default="local")
    p.add_argument("--timeout", type=float, default=0.0, help="Maximum seconds allowed for a single benchmark job; 0 disables timeout")
    p.add_argument(
        "--modes",
        default="constrained_direct,constrained_mixed,outlines,unconstrained,unconstrained_raw",
        help="Comma list from {constrained_direct,constrained_mixed,outlines,unconstrained,unconstrained_raw}",
    )
    p.add_argument("--think-budget", type=int, default=128, help="Token budget for mixed reasoning mode")
    p.add_argument("--openrouter-env", default=".env", help="Path to .env with OPENROUTER_API_KEY")
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32", "none"], help="Local/Modal torch_dtype passed to from_pretrained")
    p.add_argument("--device-map", default="", help="Optional local/Modal device_map, e.g. auto")
    p.add_argument("--model-kwargs-json", default="", help="Extra JSON object passed to from_pretrained")
    p.add_argument("--low-space", action="store_true", help="Release loaded models and delete other HF model caches before switching models")
    p.add_argument("--parallel-tasks", default="1", help="Run up to N jobs concurrently per model, or 'auto' to size workers from model size and GPU VRAM")
    p.add_argument("--dry", action="store_true")
    return p.parse_args()


def selected_modes(args: argparse.Namespace) -> list[str]:
    aliases = {
        "constrained": "constrained_direct",
        "mixed": "constrained_mixed",
    }
    modes = [aliases.get(x.strip(), x.strip()) for x in args.modes.split(",") if x.strip()]
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
    if args.backend == "openrouter":
        skipped = [
            mode for mode in modes if mode not in {"unconstrained", "unconstrained_raw"}
        ]
        if skipped:
            print("[skip] OpenRouter backend only supports unconstrained modes", flush=True)
        modes = [
            mode for mode in modes if mode in {"unconstrained", "unconstrained_raw"}
        ]
    if not modes:
        raise SystemExit("No runnable modes selected")
    return modes


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
    workers = min(workers, 32, job_count)
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


def main() -> None:
    args = parse_args()
    args.parallel_tasks = parse_parallel_tasks(args.parallel_tasks)
    if args.resume:
        args.append = True
    names = [x.strip() for x in args.tasks.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    tasks = load_tasks(names)
    task_ids = {x.strip() for x in args.task_ids.split(",") if x.strip()}
    if task_ids:
        tasks = [task for task in tasks if task.task_id in task_ids]
        missing = task_ids - {task.task_id for task in tasks}
        if missing:
            raise SystemExit(f"Unknown task ids: {', '.join(sorted(missing))}")
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if args.max_tokens_override > 0:
        tasks = [replace(task, max_tokens=args.max_tokens_override) for task in tasks]
    if not tasks:
        raise SystemExit("No tasks selected")
    if not models:
        raise SystemExit("No models selected")
    modes = selected_modes(args)
    jobs = build_jobs(tasks, models, modes, args.tries)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    trace_enabled = bool(args.trace_out)
    trace_path = Path(args.trace_out) if trace_enabled else None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        existing = read_existing_record_keys(out_path, args.backend)
        if existing:
            before = len(jobs)
            jobs = [job for job in jobs if job.key(args.backend) not in existing]
            print(f"[resume] skipped {before - len(jobs)} existing jobs from {out_path}", flush=True)

    if args.dry:
        print(f"tasks={len(tasks)} models={len(models)} modes={','.join(modes)} tries={args.tries} jobs={len(jobs)} parallel_tasks={args.parallel_tasks}")
        print(f"first_task={tasks[0].task_id if tasks else 'none'}")
        return

    if not jobs:
        print("[run] no pending jobs", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.append:
        if out_path.exists() and out_path.is_dir():
            raise SystemExit(f"Cannot append to {out_path}: path is a directory")
        if trace_path is not None:
            if trace_path.exists() and trace_path.is_dir():
                raise SystemExit(f"Cannot append to {trace_path}: path is a directory")
            trace_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.write_text("", encoding="utf-8")
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("", encoding="utf-8")

    print(f"[run] backend={args.backend} jobs={len(jobs)} parallel_tasks={args.parallel_tasks}", flush=True)
    if args.parallel_tasks == 1:
        run_sequential(args, jobs, out_path, trace_path)
    else:
        run_parallel_by_model(args, jobs, out_path, trace_path)


if __name__ == "__main__":
    main()
