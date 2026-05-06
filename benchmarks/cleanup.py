#!/usr/bin/env python3
"""Clean benchmark data by removing system errors (CUDA OOM, API errors, etc.)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Tuple


ERROR_PATTERNS = [
    "CUDA out of memory",
    "CUDA error",
    "is not a local folder",
    "not a valid model ID",
    "Couldn't instantiate",
    "HTTP error",
    "Connection error",
    "filter_out_non_signature_kwargs",
    "tiktoken",
    "sentencepiece",
]

MODELS_TO_EXCLUDE = {
    "Qwen/Qwen3.5-1.5B",
    "Ministral-3-8B-Instruct-2512-GGUF",
}


def is_system_error(record: dict) -> bool:
    """Return True if record has system-level errors (exclude from benchmarks)."""
    error = record.get("error", "")
    output = record.get("output", "")
    combined = f"{error} {output}"
    return any(p in combined for p in ERROR_PATTERNS)


def is_valid_model(record: dict) -> bool:
    """Return True if model should be included."""
    return record.get("model", "") not in MODELS_TO_EXCLUDE


def clean_records(records: list[dict]) -> Tuple[list[dict], dict]:
    """Remove error records. Returns (cleaned, stats)."""
    cleaned = []
    stats = {
        "total": len(records),
        "removed_system_error": 0,
        "removed_invalid_model": 0,
        "by_reason": Counter(),
    }

    for rec in records:
        if not is_valid_model(rec):
            stats["removed_invalid_model"] += 1
            continue
        if is_system_error(rec):
            stats["removed_system_error"] += 1
            reason = rec.get("error", "")[:60]
            stats["by_reason"][reason] += 1
            continue
        cleaned.append(rec)

    return cleaned, stats


def load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict]) -> None:
    """Save records to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def print_stats(stats: dict) -> None:
    """Print cleanup statistics."""
    print(f"\n{'=' * 60}")
    print("CLEANUP STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total records: {stats['total']}")
    print(f"Removed (system errors): {stats['removed_system_error']}")
    print(f"Removed (invalid model): {stats['removed_invalid_model']}")
    print(f"Remaining: {stats['total'] - stats['removed_system_error'] - stats['removed_invalid_model']}")
    if stats["by_reason"]:
        print("\nRemoval reasons (top 10):")
        for reason, count in stats["by_reason"].most_common(10):
            print(f"  {reason}: {count}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cleanup.py <input.jsonl> [output.jsonl]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path

    print(f"Loading {input_path}...")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records")

    cleaned, stats = clean_records(records)
    print_stats(stats)

    if output_path != input_path:
        save_jsonl(output_path, cleaned)
        print(f"Saved cleaned data to {output_path}")
    else:
        print("Dry run (no output file specified)")
