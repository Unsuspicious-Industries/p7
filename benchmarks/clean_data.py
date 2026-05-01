#!/usr/bin/env python3
"""Clean and unify the vast_model_matrix dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
    "Qwen/Qwen3.5-1.5B",  # Invalid HF model
    "Ministral-3-8B-Instruct-2512-GGUF",  # Invalid HF model
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


def load_and_clean(path: Path) -> list[dict]:
    records = []
    removed = {"api_error": 0, "invalid_model": 0}

    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if is_api_error(rec):
                    removed["api_error"] += 1
                    continue
                if not is_valid_model(rec):
                    removed["invalid_model"] += 1
                    continue
                records.append(rec)

    return records, removed


def main():
    p = argparse.ArgumentParser(description="Clean and unify vast_model_matrix data")
    p.add_argument(
        "--input",
        default="final/vast_model_matrix/combined/raw.jsonl",
        help="Input raw.jsonl",
    )
    p.add_argument(
        "--output",
        default="final/vast_model_matrix/clean/raw.jsonl",
        help="Output cleaned raw.jsonl",
    )
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}...")
    records, removed = load_and_clean(input_path)

    print(f"  Total records: {len(records)}")
    print(f"  Removed (API errors): {removed['api_error']}")
    print(f"  Removed (invalid models): {removed['invalid_model']}")

    # Summary by model
    print("\nRecords by model:")
    by_model = Counter(r["model"] for r in records)
    for model, cnt in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"  {model}: {cnt}")

    # Summary by mode
    print("\nRecords by mode:")
    by_mode = Counter(r["mode"] for r in records)
    for mode, cnt in sorted(by_mode.items()):
        print(f"  {mode}: {cnt}")

    # Summary by error
    print("\nRemaining errors:")
    errors = Counter(r["error"] for r in records)
    for err, cnt in errors.most_common(10):
        print(f"  {err}: {cnt}")

    # Write cleaned data
    print(f"\nWriting cleaned data to {output_path}...")
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
