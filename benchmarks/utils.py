"""Utility helpers for benchmark orchestration.

Separated from run.py so the orchestration logic stays readable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    model_name: str
    grammar_name: str
    task: Any
    mode: str
    attempt: int

    def label(self) -> str:
        return (
            f"{self.model_name} {self.task.task_id} {self.mode}"
            f" try={self.attempt} grammar={self.grammar_name}"
        )

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


def build_jobs(
    tasks: list[Any],
    models: list[str],
    modes: list[str],
    tries: int,
    grammar_name_fn: Any,
) -> list[Job]:
    """Cross-product of models × tasks × modes × tries."""
    jobs: list[Job] = []
    for model_name in models:
        for task in tasks:
            for mode in modes:
                gname = grammar_name_fn(task.grammar)
                for attempt in range(tries):
                    jobs.append(
                        Job(
                            model_name=model_name,
                            grammar_name=gname,
                            task=task,
                            mode=mode,
                            attempt=attempt,
                        )
                    )
    return jobs


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

# Global VRAM override — set from CLI or environment before running.
VRAM: Optional[float] = None


def model_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Merge model kwargs from args into a dict suitable for from_pretrained."""
    kwargs: dict[str, Any] = dict(getattr(args, "model_kwargs", {}) or {})
    extra_json = getattr(args, "model_kwargs_json", "") or ""
    if extra_json.strip():
        try:
            extra = json.loads(extra_json)
        except json.JSONDecodeError as e:
            raise SystemExit(f"model_kwargs_json is not valid JSON: {e}") from e
        if not isinstance(extra, dict):
            raise SystemExit("model_kwargs_json must be a JSON object")
        kwargs.update(extra)
    # Forward device / dtype from top-level args if not already overridden.
    for key, attr in [("torch_dtype", "torch_dtype"), ("device_map", "device_map")]:
        if key not in kwargs:
            val = getattr(args, attr, None)
            if val:
                kwargs[key] = val
    return kwargs


def model_param_billions(model_name: str) -> Optional[float]:
    """Heuristically extract parameter count (in billions) from a model name."""
    lower = model_name.lower()
    known = {"gpt2": 0.124, "distilgpt2": 0.082, "qwen/qwen3.5-0.8b": 0.8}
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
    if getattr(args, "torch_dtype", "auto") in ("float32", "none"):
        return 4.0
    return 2.0


def gpu_vram_gib(args: argparse.Namespace) -> Optional[float]:
    if VRAM is not None:
        return float(VRAM)
    configured = getattr(args, "vram", None)
    if configured is not None:
        return float(configured)
    if str(getattr(args, "device", "cpu")).startswith("cpu"):
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        device_index = torch.cuda.current_device()
        return torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
    except Exception:
        return None


def parse_model_concurrency(value: Any) -> int | str:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        n = int(text)
    except ValueError as e:
        raise SystemExit("model_concurrency must be a positive integer or 'auto'") from e
    if n < 1:
        raise SystemExit("model_concurrency must be >= 1 or 'auto'")
    return n


def auto_model_concurrency(
    args: argparse.Namespace, model_name: str, job_count: int
) -> int:
    if "gemma" in model_name.lower():
        print(
            f"[model-concurrency] auto model={model_name} concurrent_runs=1 reason=gemma_memory_cap",
            flush=True,
        )
        return 1
    vram_gib = gpu_vram_gib(args)
    params_b = model_param_billions(model_name)
    if vram_gib is None or params_b is None:
        print(
            f"[model-concurrency] auto model={model_name} concurrent_runs=1"
            " reason=unknown_vram_or_model_size",
            flush=True,
        )
        return 1
    bytes_per_param = dtype_bytes(args)
    weight_gib = params_b * bytes_per_param * (1_000_000_000 / (1024**3))
    per_worker_gib = max(0.5, weight_gib + 0.15)
    usable_vram_gib = max(0.0, vram_gib * 0.75)
    concurrent_runs = max(1, int(usable_vram_gib // per_worker_gib))
    concurrent_runs = max(1, min(concurrent_runs, 32, job_count))
    print(
        f"[model-concurrency] auto model={model_name}"
        f" params={params_b:.3g}B vram={vram_gib:.1f}GiB"
        f" usable≈{usable_vram_gib:.1f}GiB safety=1.33x"
        f" per_run≈{per_worker_gib:.1f}GiB concurrent_runs={concurrent_runs}",
        flush=True,
    )
    return concurrent_runs


def concurrent_runs_for_model(
    args: argparse.Namespace,
    model_name: str,
    job_count: int,
) -> int:
    configured = parse_model_concurrency(args.model_concurrency)
    if configured == "auto":
        return auto_model_concurrency(args, model_name, job_count)
    return min(int(configured), job_count)


def model_cache_key(args: argparse.Namespace, job: Job) -> tuple[str, ...]:
    if job.mode in {"outlines", "outlines_mixed"}:
        return ("outlines", job.model_name, job.grammar_name)
    if getattr(args, "backend", "local") in {"local", "openrouter"}:
        return (args.backend, job.model_name)
    return (args.backend, job.model_name, job.grammar_name)


# ---------------------------------------------------------------------------
# CUDA / HF cache utilities
# ---------------------------------------------------------------------------

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


def hf_hub_cache_dir() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def hf_model_cache_name(model_name: str) -> str:
    return f"models--{model_name.replace('/', '--')}"


def clean_hf_model_cache(model_name: str) -> None:
    """Delete the cached HF model files to free disk space (low_space mode).

    JANK: this permanently removes the cached weights; re-running will re-download.
    Only enable via low_space=true in the config when disk is genuinely constrained.
    """
    cache_dir = hf_hub_cache_dir() / hf_model_cache_name(model_name)
    if cache_dir.exists():
        print(f"[low-space] removing cache {cache_dir}", flush=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    else:
        print(f"[low-space] cache not found, skipping: {cache_dir}", flush=True)
