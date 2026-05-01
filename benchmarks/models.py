#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / "out" / "vast_model_matrix"


LOCAL_ASCENDING_MODELS = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-1.5B",
    "Qwen/Qwen3.5-0.8B",
    "Ministral-3-8B-Instruct-2512-GGUF",
    "microsoft/Phi-4-mini-instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

DEFAULT_CLOSED_MODELS = [
    "openai/gpt-5.4-mini",
    "openai/gpt-5.3-codex",
    "anthropic/claude-4.5-haiku",
    "google/gemini-3.0-flash-latest",
    "qwen/qwen3.5-35b-a3b",
    "google/gemma-4-31b-it"
]


@dataclass(frozen=True)
class Phase:
    name: str
    backend: str
    modes: str
    models: list[str]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_model_file(path: str) -> list[str]:
    if not path:
        return []
    models: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            models.append(stripped)
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full benchmark matrix for a Vast.ai/local GPU host."
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task selectors (all/core/task_id/grammar/category)",
    )
    parser.add_argument("--tries", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 means all tasks")
    parser.add_argument(
        "--task-ids", default="", help="Optional comma-separated task ids"
    )
    parser.add_argument(
        "--max-tokens-override",
        type=int,
        default=0,
        help="Override max_tokens for smoke runs",
    )
    parser.add_argument(
        "--models",
        default=",".join(LOCAL_ASCENDING_MODELS),
        help="Comma-separated local/HF model ids",
    )
    parser.add_argument(
        "--models-file",
        default="",
        help="Optional newline-separated local/HF model ids",
    )
    parser.add_argument(
        "--closed-models",
        default=",".join(DEFAULT_CLOSED_MODELS),
        help="Comma-separated OpenRouter model ids for closed-model baseline",
    )
    parser.add_argument(
        "--closed-models-file",
        default="",
        help="Optional newline-separated OpenRouter model ids",
    )
    parser.add_argument(
        "--without-closed",
        action="store_true",
        help="Disable the OpenRouter closed-model phase",
    )
    parser.add_argument(
        "--without-mixed",
        action="store_true",
        help="Disable the p7 mixed reasoning phase",
    )
    parser.add_argument(
        "--without-outlines",
        action="store_true",
        help="Disable the Outlines syntax-only baseline",
    )
    parser.add_argument(
        "--with-traces",
        action="store_true",
        help="Write per-phase token trace JSONL files",
    )
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32", "none"],
    )
    parser.add_argument(
        "--device-map", default="auto", help="Use empty string to disable device_map"
    )
    parser.add_argument(
        "--model-kwargs-json",
        default="",
        help="Extra JSON object passed to from_pretrained",
    )
    parser.add_argument(
        "--low-space",
        action="store_true",
        help="Pass --low-space to benchmark phases so HF model caches are cleaned before model switches",
    )
    parser.add_argument(
        "--parallel-tasks",
        default="1",
        help="Pass --parallel-tasks to benchmark phases; use 'auto' to size from model/GPU",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-job timeout in seconds; 0 disables",
    )
    parser.add_argument("--think-budget", type=int, default=128)
    parser.add_argument("--openrouter-env", default=".env")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite phase raw files instead of appending only missing jobs",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing raw JSONL files",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="Continue if a phase command fails"
    )
    return parser.parse_args()


def run_command(command: list[str], *, dry_run: bool, keep_going: bool) -> int:
    print(shlex.join(command), flush=True)
    if dry_run:
        return 0
    result = subprocess.run(command, cwd=REPO, check=False)
    if result.returncode != 0:
        message = (
            f"command failed with exit code {result.returncode}: {shlex.join(command)}"
        )
        if keep_going:
            print(f"[warn] {message}", flush=True)
        else:
            raise SystemExit(message)
    return result.returncode


def benchmark_command(
    args: argparse.Namespace, phase: Phase, raw: Path, phase_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        "benchmarks/run.py",
        "--tasks",
        args.tasks,
        "--models",
        ",".join(phase.models),
        "--tries",
        str(args.tries),
        "--backend",
        phase.backend,
        "--modes",
        phase.modes,
        "--out",
        str(raw),
        "--timeout",
        str(args.timeout),
        "--think-budget",
        str(args.think_budget),
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
    ]
    if args.device_map:
        command.extend(["--device-map", args.device_map])
    if args.model_kwargs_json:
        command.extend(["--model-kwargs-json", args.model_kwargs_json])
    if args.low_space:
        command.append("--low-space")
    if args.parallel_tasks != "1":
        command.extend(["--parallel-tasks", args.parallel_tasks])
    if args.max_tasks:
        command.extend(["--max-tasks", str(args.max_tasks)])
    if args.task_ids:
        command.extend(["--task-ids", args.task_ids])
    if args.max_tokens_override:
        command.extend(["--max-tokens-override", str(args.max_tokens_override)])
    if args.with_traces:
        command.extend(["--trace-out", str(phase_dir / "token_logs.jsonl")])
    if not args.no_resume:
        command.append("--resume")
    if phase.backend == "openrouter":
        command.extend(["--openrouter-env", args.openrouter_env])
    return command


def aggregate_command(raw: Path, phase_dir: Path) -> list[str]:
    return [
        sys.executable,
        "benchmarks/agg.py",
        "--in",
        str(raw),
        "--out-dir",
        str(phase_dir),
    ]


def phase_paths(args: argparse.Namespace, phase: Phase) -> tuple[Path, Path]:
    phase_dir = Path(args.out_dir) / phase.name
    return phase_dir, phase_dir / "raw.jsonl"


def run_phase(args: argparse.Namespace, phase: Phase) -> None:
    phase_dir, raw = phase_paths(args, phase)
    if raw.exists() and raw.is_dir():
        raise SystemExit(f"Expected raw file at {raw}, but found a directory")

    print(
        f"[phase] {phase.name}: backend={phase.backend} modes={phase.modes} models={len(phase.models)}",
        flush=True,
    )
    if not args.aggregate_only:
        command = benchmark_command(args, phase, raw, phase_dir)
        run_command(command, dry_run=args.dry_run, keep_going=args.keep_going)

    if args.dry_run:
        print(shlex.join(aggregate_command(raw, phase_dir)), flush=True)
        return
    if raw.exists() and raw.stat().st_size > 0:
        run_command(
            aggregate_command(raw, phase_dir), dry_run=False, keep_going=args.keep_going
        )
    else:
        print(f"[warn] no raw data to aggregate for {phase.name}: {raw}", flush=True)


def combine_and_aggregate(args: argparse.Namespace, phases: list[Phase]) -> None:
    combined_dir = Path(args.out_dir) / "combined"
    combined_raw = combined_dir / "raw.jsonl"
    phase_raws = [phase_paths(args, phase)[1] for phase in phases]

    if args.dry_run:
        print(
            f"[combine] {' '.join(str(path) for path in phase_raws)} -> {combined_raw}",
            flush=True,
        )
        print(shlex.join(aggregate_command(combined_raw, combined_dir)), flush=True)
        return

    combined_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with combined_raw.open("w", encoding="utf-8") as output:
        for raw in phase_raws:
            if not raw.exists() or raw.stat().st_size == 0:
                continue
            with raw.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        output.write(line if line.endswith("\n") else line + "\n")
                        written += 1

    if written:
        run_command(
            aggregate_command(combined_raw, combined_dir),
            dry_run=False,
            keep_going=args.keep_going,
        )
    else:
        print(
            f"[warn] no raw data to aggregate for combined report: {combined_raw}",
            flush=True,
        )


def build_phases(args: argparse.Namespace) -> list[Phase]:
    open_models = [*split_csv(args.models), *read_model_file(args.models_file)]
    closed_models = [
        *split_csv(args.closed_models),
        *read_model_file(args.closed_models_file),
    ]
    phases: list[Phase] = []

    if open_models:
        phases.append(
            Phase("open_unconstrained", "local", "unconstrained", open_models)
        )
        phases.append(
            Phase("open_unconstrained_raw", "local", "unconstrained_raw", open_models)
        )
        phases.append(
            Phase("p7_constrained_direct", "local", "constrained_direct", open_models)
        )
        if not args.without_mixed:
            phases.append(
                Phase("p7_constrained_mixed", "local", "constrained_mixed", open_models)
            )
        if not args.without_outlines:
            phases.append(
                Phase("outlines_constrained", "local", "outlines", open_models)
            )

    if not args.without_closed:
        if not closed_models:
            raise SystemExit(
                "No closed models selected; provide --closed-models or --closed-models-file"
            )
        phases.append(
            Phase("closed_unconstrained", "openrouter", "unconstrained", closed_models)
        )

    return phases


def main() -> None:
    args = parse_args()
    phases = build_phases(args)
    if not phases:
        raise SystemExit("No phases selected")
    for phase in phases:
        run_phase(args, phase)
    combine_and_aggregate(args, phases)


if __name__ == "__main__":
    main()
