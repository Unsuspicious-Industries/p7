#!/usr/bin/env python3
"""
Deploy script to complete the benchmark matrix on remote.
Generates commands to fill gaps in the model × mode matrix.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_unified(path: Path = Path("final/unified/raw.jsonl")) -> dict[tuple, int]:
    """Load current state and return (model, mode) -> count"""
    if not path.exists():
        return {}

    matrix = defaultdict(int)
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                matrix[(r["model"], r["mode"])] += 1
    return matrix


MODELS_LOCAL = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.6-27B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "microsoft/Phi-4-mini-instruct",
]

MODELS_OPENROUTER = [
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-31b-it",
    "openai/gpt-oss-20b",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.3-codex",
    "qwen/qwen3.5-35b-a3b",
]


def get_target_models() -> list[str]:
    return MODELS_LOCAL + MODELS_OPENROUTER


def get_target_modes() -> list[str]:
    return [
        "constrained_direct",
        "constrained_mixed",
        "unconstrained",
        "unconstrained_raw",
    ]


def main():
    p = argparse.ArgumentParser(description="Generate completion commands")
    p.add_argument("--dry-run", action="store_true", help="Just print gaps")
    p.add_argument(
        "--out-commands", default="final/unified/commands.sh", help="Output script"
    )
    args = p.parse_args()

    matrix = load_unified()
    models = get_target_models()
    modes = get_target_modes()

    TARGET_TASKS = 94

    # Find gaps
    gaps = []
    for model in models:
        for mode in modes:
            current = matrix.get((model, mode), 0)
            if current < TARGET_TASKS:
                gaps.append((model, mode, TARGET_TASKS - current))

    print(f"=== GAPS FOUND: {len(gaps)} cells ===")
    for model, mode, missing in sorted(gaps, key=lambda x: (x[1], -x[2])):
        print(f"  {model[:40]:<40} {mode:<20} missing: {missing}")

    # Recalculate - constrained only for local models, unconstrained for all
    # Local models need: constrained_direct, constrained_mixed, unconstrained, unconstrained_raw
    # OpenRouter models need: unconstrained only (constrained not possible)

    openrouter_gaps = []
    local_gaps = []

    for model, mode, missing in gaps:
        if model in MODELS_OPENROUTER:
            # OpenRouter: only unconstrained modes
            if mode in ("unconstrained", "unconstrained_raw"):
                openrouter_gaps.append((model, mode, missing))
        else:
            # Local: all modes
            if mode in ("constrained_direct", "constrained_mixed"):
                local_gaps.append((model, mode, missing))
            else:
                openrouter_gaps.append((model, mode, missing))

    print(f"\n=== LOCAL (constrained) - {len(local_gaps)} commands ===")
    for model, mode, missing in local_gaps:
        print(f"  {model[:35]:<35} {mode:<20} {missing}")

    print(f"\n=== OPENROUTER - {len(openrouter_gaps)} commands ===")
    for model, mode, missing in openrouter_gaps:
        print(f"  {model[:35]:<35} {mode:<20} {missing}")

    # Generate commands
    with open(args.out_commands, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Benchmark completion script\n")
        f.write("# Run on remote with OpenRouter API access\n\n")

        f.write("# === OPENROUTER (unconstrained) ===\n")
        for model, mode, missing in openrouter_gaps:
            f.write(f"# Missing: {missing}\n")
            f.write(
                f"python benchmarks/run.py --tasks all --models {model} --modes {mode} --backend openrouter --out final/unified/raw.jsonl --resume\n\n"
            )

        f.write("# === LOCAL (constrained) - run locally ===\n")
        for model, mode, missing in local_gaps:
            f.write(f"# Missing: {missing}\n")
            f.write(
                f"python benchmarks/run.py --tasks all --models {model} --modes {mode} --backend local --out final/unified/raw.jsonl --resume\n\n"
            )

    print(f"\nCommands written to {args.out_commands}")


if __name__ == "__main__":
    main()
