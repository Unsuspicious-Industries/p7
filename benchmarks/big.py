#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import p7


BIG_MODELS = [
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-E4B-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "openai/gpt-oss-20b",
    "moonshotai/Moonlight-16B-A3B",
    "google/gemma-4-26B-A4B-it",
    "THUDM/GLM-4.7-Flash",
    "Qwen/Qwen3.6-27B",
    
]

MODES = ["constrained_mixed", "unconstrained_raw", "closed_unconstrained"]

CLOSED_MODELS = [
    "openai/gpt-5.4-mini",
    "openai/gpt-5.3-codex",
    "anthropic/claude-4.5-haiku",
    "google/gemini-3.0-flash-latest",
    "qwen/qwen3.5-35b-a3b",
    "google/gemma-4-31b-it",
]

MIN_RAM_GB = 90


def check_ram() -> float:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except Exception:
        pass
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    return 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run big model benchmarks (90GB+ RAM required)"
    )
    p.add_argument("--tasks", default="all", help="Comma list of task selectors")
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--max-tokens-override", type=int, default=0)
    p.add_argument("--task-ids", default="")
    p.add_argument("--out", default=str(ROOT / "out" / "big.jsonl"))
    p.add_argument("--append", action="store_true")
    p.add_argument("--trace-out", default="")
    p.add_argument("--resume", "--skip-existing", dest="resume", action="store_true")
    p.add_argument("--timeout", type=float, default=0.0)
    p.add_argument("--think-budget", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--device-map", default="")
    p.add_argument("--model-kwargs-json", default="")
    p.add_argument("--low-space", action="store_true")
    p.add_argument("--parallel-tasks", default="1")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--openrouter-env", default=".env")
    p.add_argument("--backend", default="local", choices=["local", "openrouter"])
    p.add_argument(
        "--skip-ram-check", action="store_true", help="Skip 90GB RAM requirement check"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_ram_check:
        available_ram = check_ram()
        print(
            f"[ram] available={available_ram:.1f}GB required={MIN_RAM_GB}GB", flush=True
        )
        assert available_ram >= MIN_RAM_GB, (
            f"Need at least {MIN_RAM_GB}GB RAM, found {available_ram:.1f}GB"
        )

    from benchmarks.api import load_tasks
    from benchmarks.run import (
        build_jobs,
        cached_model,
        run_job,
        write_outputs,
        read_existing_record_keys,
        release_cached_models,
    )

    names = [x.strip() for x in args.tasks.split(",") if x.strip()]
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
        from dataclasses import replace

        tasks = [replace(task, max_tokens=args.max_tokens_override) for task in tasks]

    open_models = [m for m in BIG_MODELS if m not in CLOSED_MODELS]
    models_for_modes = {
        "constrained_mixed": open_models,
        "unconstrained_raw": open_models,
        "closed_unconstrained": CLOSED_MODELS,
    }

    all_jobs = []
    for mode in MODES:
        models = models_for_modes.get(mode, [])
        if not models:
            continue
        jobs = build_jobs(tasks, models, [mode], args.tries)
        all_jobs.extend(jobs)

    if not all_jobs:
        raise SystemExit("No jobs to run")

    OUT = ROOT / "out"
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        args.append = True
        existing = read_existing_record_keys(out_path, args.backend)
        if existing:
            before = len(all_jobs)
            all_jobs = [
                job for job in all_jobs if job.key(args.backend) not in existing
            ]
            print(
                f"[resume] skipped {before - len(all_jobs)} existing jobs from {out_path}",
                flush=True,
            )

    if args.append:
        if out_path.exists() and out_path.is_dir():
            raise SystemExit(f"Cannot append to {out_path}: path is a directory")
    else:
        out_path.write_text("", encoding="utf-8")

    print(
        f"[run] jobs={len(all_jobs)} parallel_tasks={args.parallel_tasks}", flush=True
    )

    if args.dry:
        print(f"tasks={len(tasks)} modes={','.join(MODES)} tries={args.tries}")
        return

    cache = {}
    for job in all_jobs:
        started = None
        try:
            started = job.label()
            print(f"[start] {started}", flush=True)

            if job.mode == "closed_unconstrained":
                from benchmarks.providers import OpenRouterModel

                model = OpenRouterModel(job.model_name, env_path=args.openrouter_env)
                record, traces = run_job(args, job, bool(args.trace_out), model=model)
            else:
                model = cached_model(args, job, cache)
                record, traces = run_job(args, job, bool(args.trace_out), model=model)

            write_outputs(
                out_path,
                Path(args.trace_out) if args.trace_out else None,
                record,
                traces,
            )
            print(f"[done] {started} error={record['error']}", flush=True)
        except Exception as e:
            if started:
                print(f"[error] {started}: {e}", flush=True)
            from benchmarks.run import error_record

            record = error_record(job, args, e)
            write_outputs(
                out_path, Path(args.trace_out) if args.trace_out else None, record, []
            )

    release_cached_models(cache)
    print(f"[done] wrote results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
