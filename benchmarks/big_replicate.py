#!/usr/bin/env python3
"""
Replicate big.py evaluations with OPENROUTER models.

Usage:
    # Dry run to see commands
    python big_replicate.py --dry-run

    # Actually run
    python big_replicate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODES = ["unconstrained"]  # openrouter only supports unconstrained mode

BIG_MODELS_VIA_OPENROUTER = [
    "qwen/qwen3.5-9b",
    "google/gemma-4-4b-it",
    "deepseek-ai/deepseek-r1-distill-qwen-7b",
    "moonshotai/moonlight-16b-a3b",
    "google/gemma-4-26b-it",
    "qwen/qwen3.6-27b-instruct",
]

CLOSED_MODELS = [
    "openai/gpt-5.4-mini",
    "openai/gpt-5.3-codex",
    "anthropic/claude-4.5-haiku",
    "google/gemini-3.0-flash-latest",
    "qwen/qwen3.5-35b-a3b",
    "google/gemma-4-31b-it",
]

ALL_MODELS = BIG_MODELS_VIA_OPENROUTER + CLOSED_MODELS


def get_task_ids(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description="Replicate big.py with openrouter")
    p.add_argument(
        "--task-ids",
        default="final/big_leaderboard/task_ids.txt",
        help="Path to task ids file",
    )
    p.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help="Comma-separated openrouter model ids",
    )
    p.add_argument(
        "--tries",
        type=int,
        default=1,
        help="Number of tries per task",
    )
    p.add_argument(
        "--backend",
        default="openrouter",
        help="Backend to use",
    )
    p.add_argument(
        "--out",
        default="final/big_openrouter/raw.jsonl",
        help="Output jsonl path",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = p.parse_args()

    task_ids = get_task_ids(Path(args.task_ids))
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"Tasks: {len(task_ids)}")
    print(f"Models: {len(models)}")
    print(f"Total jobs: {len(task_ids) * len(models) * args.tries}")
    print()

    jobs = []
    for model in models:
        for task_id in task_ids:
            for try_num in range(args.tries):
                jobs.append((model, task_id, try_num))

    print("Sample commands:")
    for model, task_id, try_num in jobs[:3]:
        cmd = [
            sys.executable,
            "benchmarks/run.py",
            "--tasks",
            task_id,
            "--models",
            model,
            "--modes",
            "unconstrained",
            "--backend",
            args.backend,
            "--tries",
            str(args.tries),
            "--out",
            args.out,
            "--resume",
        ]
        print(
            f"  {sys.executable} benchmarks/run.py --tasks {task_id} --models {model} ..."
        )

    if args.dry_run:
        print("\n[Dry run] Not executing any commands")
        return

    print(f"\n[Info] To run these evaluations, you would execute:")
    print(f"  python benchmarks/run.py --tasks all --models {args.models} ")
    print(f"    --modes unconstrained --backend openrouter --out {args.out} --resume")
    print(f"\nOr you can use benchmarks/big.py but with --backend openrouter")


if __name__ == "__main__":
    main()
