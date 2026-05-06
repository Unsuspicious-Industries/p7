#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ERRORS_TO_DROP = {
    "CUDA out of memory.",
    "`tiktoken` is required to read a `tiktoken` file.",
    "Couldn't instantiate the backend tokenizer",
    "THUDM/GLM-4.7-Flash is not a local folder",
    "module 'transformers.models.siglip2.image_processing_siglip2' has no attribute",
}


def is_error_record(record: dict) -> bool:
    error = record.get("error", "")
    if not error:
        return False
    for drop_err in ERRORS_TO_DROP:
        if drop_err in error:
            return True
    return False


def load_and_clean(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if not is_error_record(rec):
                    records.append(rec)
    return records


def compute_stats(records: list[dict]) -> dict:
    by_entry = defaultdict(lambda: {"total": 0, "passed": 0, "exact": 0})

    for rec in records:
        model = rec.get("model", "?")
        mode = rec.get("mode", "?")

        if mode in ("constrained_direct", "constrained_mixed"):
            entry = f"{model} (constrained)"
        elif mode == "unconstrained":
            entry = f"{model} (unconstrained)"
        elif mode == "unconstrained_cleaned":
            entry = f"{model} (unconstrained cleaned)"
        else:
            entry = f"{model} ({mode})"

        by_entry[entry]["total"] += 1
        if rec.get("passed"):
            by_entry[entry]["passed"] += 1
        if rec.get("exact"):
            by_entry[entry]["exact"] += 1

    results = {}
    for entry, stats in by_entry.items():
        results[entry] = {
            "pass_rate": 100 * stats["passed"] / stats["total"]
            if stats["total"]
            else 0,
            "exact_rate": 100 * stats["exact"] / stats["total"]
            if stats["total"]
            else 0,
            "total": stats["total"],
        }
    return results


def load_closed_baseline(path: Path) -> list[dict]:
    return load_and_clean(path)


def main():
    p = argparse.ArgumentParser(description="Clean big_vast and generate leaderboard")
    p.add_argument(
        "--big-vast", default="final/big_vast", help="Path to big_vast jsonl"
    )
    p.add_argument(
        "--closed",
        default="final/vast_model_matrix/closed_unconstrained/raw.jsonl",
        help="Path to closed_unconstrained raw.jsonl",
    )
    p.add_argument(
        "--out-dir", default="final/big_leaderboard", help="Output directory"
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and cleaning {args.big_vast}...")
    big_records = load_and_clean(Path(args.big_vast))
    print(f"  Loaded {len(big_records)} records after cleaning")

    print(f"Loading closed baseline from {args.closed}...")
    closed_records = load_and_clean(Path(args.closed))
    print(f"  Loaded {len(closed_records)} records after cleaning")

    print("\nComputing stats...")
    big_stats = compute_stats(big_records)
    closed_stats = compute_stats(closed_records)

    all_entries = {**big_stats, **closed_stats}

    sorted_entries = sorted(
        all_entries.items(), key=lambda x: x[1]["pass_rate"], reverse=True
    )

    print(f"\n=== BIG LEADERBOARD ===")
    print(f"{'Rank':<5} {'Entry':<50} {'Pass%':>8} {'Exact%':>8} {'N':>6}")
    print("-" * 85)
    for rank, (entry, stats) in enumerate(sorted_entries, 1):
        print(
            f"{rank:<5} {entry:<50} {stats['pass_rate']:>8.1f} {stats['exact_rate']:>8.1f} {stats['total']:>6}"
        )

    leaderboard_path = out_dir / "leaderboard.csv"
    with open(leaderboard_path, "w") as f:
        f.write("rank,entry,pass_rate,exact_rate,total\n")
        for rank, (entry, stats) in enumerate(sorted_entries, 1):
            f.write(
                f"{rank},{entry},{stats['pass_rate']:.2f},{stats['exact_rate']:.2f},{stats['total']}\n"
            )

    print(f"\nLeaderboard saved to {leaderboard_path}")

    big_raw_path = out_dir / "big_cleaned.jsonl"
    with open(big_raw_path, "w") as f:
        for rec in big_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Cleaned big_vast saved to {big_raw_path}")

    closed_raw_path = out_dir / "closed_cleaned.jsonl"
    with open(closed_raw_path, "w") as f:
        for rec in closed_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Cleaned closed baseline saved to {closed_raw_path}")

    tasks = sorted(set(r["task_id"] for r in big_records))
    task_list_path = out_dir / "task_ids.txt"
    with open(task_list_path, "w") as f:
        for t in tasks:
            f.write(f"{t}\n")
    print(f"Task list ({len(tasks)} tasks) saved to {task_list_path}")


if __name__ == "__main__":
    main()
