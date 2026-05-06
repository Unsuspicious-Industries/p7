#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from review import process_jsonl_file
from agg import (
    SUMMARY_METRICS,
    dedupe_rows,
    load_rows,
    record_key,
    summarize,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate evaluations from raw benchmark data."
    )
    p.add_argument(
        "--in-dir",
        default="benchmarks/out",
        help="Directory containing raw.jsonl files",
    )
    p.add_argument(
        "--out-dir",
        default="benchmarks/out/evals",
        help="Output directory for evaluation reports",
    )
    p.add_argument(
        "--exclude-modes", default="outlines", help="Comma-separated modes to exclude"
    )
    return p.parse_args()


def count_tokens(text: str, model_name: str = "") -> int:
    return len(text.split())


def load_and_fix_rows(path: Path) -> list[dict]:
    rows = load_rows(path)
    for row in rows:
        output = row.get("output", "")
        row["token_count"] = count_tokens(output)
        row["char_count"] = len(output)
    return rows


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_modes = set(args.exclude_modes.split(","))

    raw_files = list(in_dir.rglob("raw.jsonl"))
    print(f"Found {len(raw_files)} raw.jsonl files")

    print("Running review to fix misclassified output blocks...")
    total_fixed = 0
    for rf in raw_files:
        stats = process_jsonl_file(rf, dry_run=False)
        total_fixed += stats["fixed"]
    if total_fixed > 0:
        print(f"Fixed {total_fixed} misclassified output block(s)")
    else:
        print("No misclassified output blocks found")

    all_rows = []
    for rf in raw_files:
        rows = load_and_fix_rows(rf)
        for row in rows:
            if row.get("mode") in exclude_modes:
                continue
            all_rows.append(row)

    print(f"Loaded {len(all_rows)} rows after filtering")

    if not all_rows:
        print("No data to evaluate")
        return

    deduped = dedupe_rows(all_rows)
    print(f"Deduped to {len(deduped)} rows")

    summary = summarize(deduped, ("backend", "model", "mode", "language"))
    summary_cols = ["backend", "model", "mode", "language", *SUMMARY_METRICS]
    write_csv(out_dir / "summary.csv", summary, summary_cols)

    category_summary = summarize(
        deduped, ("backend", "model", "mode", "language", "category")
    )
    category_cols = [
        "backend",
        "model",
        "mode",
        "language",
        "category",
        *SUMMARY_METRICS,
    ]
    write_csv(out_dir / "summary_by_category.csv", category_summary, category_cols)

    with open(out_dir / "raw_fixed.jsonl", "w") as f:
        for row in deduped:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'summary_by_category.csv'}")
    print(f"Wrote {out_dir / 'raw_fixed.jsonl'}")


if __name__ == "__main__":
    main()
