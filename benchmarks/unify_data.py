#!/usr/bin/env python3
"""Unify all benchmark data into a single dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ERROR_PATTERNS = [
    "is not a local folder",
    "CUDA out of memory",
    "CUDA error",
    "not a valid model ID",
    "filter_out_non_signature_kwargs",
    "tiktoken",
    "Couldn't instantiate the backend tokenizer",
    "sentencepiece",
]

MODELS_TO_EXCLUDE = {
    "Qwen/Qwen3.5-1.5B",
    "Ministral-3-8B-Instruct-2512-GGUF",
}


def is_api_error(record: dict) -> bool:
    error = record.get("error", "")
    output = record.get("output", "")
    combined = f"{error} {output}"
    for pattern in ERROR_PATTERNS:
        if pattern in combined:
            return True
    return False


def is_valid_model(record: dict) -> bool:
    return record.get("model", "") not in MODELS_TO_EXCLUDE


def clean_record(record: dict) -> dict | None:
    if is_api_error(record):
        return None
    if not is_valid_model(record):
        return None

    rec = dict(record)
    rec["_source"] = record.get("_source", "unknown")
    return rec


def record_key(rec: dict) -> tuple:
    try:
        attempt = int(rec.get("try", 0))
    except (TypeError, ValueError):
        attempt = 0
    return (
        rec.get("model", ""),
        rec.get("mode", ""),
        rec.get("task_id", ""),
        rec.get("task_hash", ""),
        rec.get("resolution_hash", ""),
        attempt,
    )


def main():
    p = argparse.ArgumentParser(description="Unify all benchmark data")
    p.add_argument("--out", default="final/unified/raw.jsonl", help="Output path")
    args = p.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    sources = {}

    # Source 1: vast_model_matrix/clean
    print("Loading vast_model_matrix/clean...")
    with open("final/vast_model_matrix/clean/raw.jsonl") as f:
        records = [json.loads(l) for l in f if l.strip()]
    for r in records:
        r["_source"] = "vast_matrix"
    all_records.extend(records)
    sources["vast_matrix"] = len(records)
    print(f"  Loaded {len(records)} records")

    # Source 2: big_leaderboard/big_cleaned
    print("Loading big_leaderboard/big_cleaned...")
    with open("final/big_leaderboard/big_cleaned.jsonl") as f:
        records = [json.loads(l) for l in f if l.strip()]
    for r in records:
        r["_source"] = "big_vast"
    all_records.extend(records)
    sources["big_vast"] = len(records)
    print(f"  Loaded {len(records)} records")

    # Source 3: big_leaderboard/closed_cleaned
    print("Loading big_leaderboard/closed_cleaned...")
    with open("final/big_leaderboard/closed_cleaned.jsonl") as f:
        records = [json.loads(l) for l in f if l.strip()]
    for r in records:
        r["_source"] = "closed_baseline"
    all_records.extend(records)
    sources["closed_baseline"] = len(records)
    print(f"  Loaded {len(records)} records")

    print(f"\nTotal records before cleaning: {len(all_records)}")

    # Clean and dedupe
    print("\nCleaning records...")
    cleaned = []
    for r in all_records:
        cleaned_rec = clean_record(r)
        if cleaned_rec:
            cleaned.append(cleaned_rec)

    print(f"  After API error removal: {len(cleaned)}")

    # Dedupe - keep best result (highest pass_rate) per key
    print("Deduplicating (keeping best pass rate)...")
    deduped = {}
    for r in cleaned:
        key = record_key(r)
        passed = 1 if r.get("passed") or r.get("error") == "ok" else 0
        exact = 1 if r.get("exact") else 0
        score = (passed, exact)  # prefer passed > exact

        if key not in deduped:
            deduped[key] = r
        else:
            existing = deduped[key]
            existing_passed = (
                1 if existing.get("passed") or existing.get("error") == "ok" else 0
            )
            existing_exact = 1 if existing.get("exact") else 0
            existing_score = (existing_passed, existing_exact)

            if score > existing_score:
                deduped[key] = r

    final_records = list(deduped.values())
    print(f"  After dedupe: {len(final_records)}")

    # Summary by model and mode
    print("\n=== UNIFIED DATASET SUMMARY ===")
    print(f"Total records: {len(final_records)}")

    by_model = Counter(r["model"] for r in final_records)
    print(f"\nRecords by model ({len(by_model)} models):")
    for model, cnt in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"  {model}: {cnt}")

    by_mode = Counter(r["mode"] for r in final_records)
    print(f"\nRecords by mode:")
    for mode, cnt in sorted(by_mode.items()):
        print(f"  {mode}: {cnt}")

    by_backend = Counter(r.get("backend", "local") for r in final_records)
    print(f"\nRecords by backend:")
    for backend, cnt in sorted(by_backend.items()):
        print(f"  {backend}: {cnt}")

    # Matrix view
    print("\n=== MODEL × MODE MATRIX ===")
    matrix = defaultdict(lambda: defaultdict(int))
    for r in final_records:
        matrix[r["model"]][r["mode"]] += 1

    all_modes = sorted(
        set(m for model_data in matrix.values() for m in model_data.keys())
    )
    header = "Model".ljust(50) + "".join(m[:15].ljust(15) for m in all_modes)
    print(header)
    print("-" * len(header))
    for model in sorted(matrix.keys()):
        row = model.ljust(50)
        for mode in all_modes:
            row += str(matrix[model][mode]).rjust(15)
        print(row[:80])

    # Write output
    print(f"\nWriting to {output_path}...")
    with open(output_path, "w") as f:
        for rec in final_records:
            f.write(json.dumps(rec) + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
