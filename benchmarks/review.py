#!/usr/bin/env python3
"""Review and fix jsonl files where model output is incorrectly classified as reasoning blocks.

Some models wrap their actual output in language-specific XML tags (e.g., <fun>, <imp>, <stlc>).
The system incorrectly classifies these as reasoning/think blocks instead of valid output.
This script detects such cases, corrects the metadata, and rewrites the jsonl files in place.

Conditions for a record to be considered misclassified:
1. reasoning_blocks contains a block wrapped in a known language tag
2. The inner content of that tag matches (or closely matches) the output field
3. output_extracted is False (or reasoning_blocks incorrectly contains the output)

Fixes applied:
- Set output_extracted = True
- Remove the tagged output block from reasoning_blocks
- Recalculate formal_tokens from the output
- Set think_tokens to count tokens in remaining reasoning blocks only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LANGUAGE_TAGS = ("fun", "imp", "stlc")

TAG_PATTERN = re.compile(
    r"^<(" + "|".join(LANGUAGE_TAGS) + r")>\s*(.*?)\s*</\1>$",
    re.DOTALL,
)


def extract_from_tag(text: str) -> tuple[str, str] | None:
    """Extract (tag_name, inner_content) from a language-tagged block.

    Returns None if the text is not wrapped in a known language tag.
    """
    match = TAG_PATTERN.match(text.strip())
    if match:
        return match.group(1), match.group(2).strip()
    return None


def is_likely_output(inner: str, language: str) -> bool:
    """Quick sanity check that inner content looks like valid output."""
    inner = inner.strip()
    if not inner:
        return False
    lang = language.lower()
    if lang == "fun":
        return inner.startswith("let ")
    if lang == "imp":
        return inner.startswith("{")
    if lang == "stlc":
        return inner.startswith("\u03bb")
    return len(inner) >= 3


def fix_record(record: dict) -> str:
    """Fix a single record if output is misclassified as reasoning block.

    Returns a reason string if fixed, empty string if no fix needed.
    """
    reasoning_blocks = record.get("reasoning_blocks", [])
    if not reasoning_blocks:
        return ""

    language = record.get("language", "")
    output = record.get("output", "")

    tagged_idx = None
    tagged_content = None

    for i, block in enumerate(reasoning_blocks):
        if not isinstance(block, str):
            continue
        result = extract_from_tag(block)
        if result is None:
            continue
        tag_name, inner = result
        if not is_likely_output(inner, language):
            continue

        # Check if the inner content matches the current output
        if inner.strip() == output.strip():
            tagged_idx = i
            tagged_content = inner
            break

    if tagged_idx is None:
        return ""

    # Build new reasoning_blocks without the tagged output block
    new_reasoning = []
    for i, block in enumerate(reasoning_blocks):
        if i == tagged_idx:
            continue
        new_reasoning.append(block)

    record["reasoning_blocks"] = new_reasoning
    record["output_extracted"] = True
    record["formal_tokens"] = len(output.split())
    record["think_tokens"] = sum(
        len(b.split()) for b in new_reasoning if isinstance(b, str)
    )

    return f"removed <{language}> tag from reasoning_blocks[{tagged_idx}], set output_extracted=True"


def process_jsonl_file(filepath: Path, dry_run: bool = False) -> dict:
    stats = {"total": 0, "fixed": 0, "fixes": []}

    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    for idx, record in enumerate(records):
        stats["total"] += 1
        reason = fix_record(record)
        if reason:
            stats["fixed"] += 1
            stats["fixes"].append(
                {
                    "index": idx,
                    "task_id": record.get("task_id", "unknown"),
                    "reason": reason,
                }
            )

    if not dry_run and stats["fixed"] > 0:
        with open(filepath, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review and fix jsonl files where model output is misclassified as reasoning blocks."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing jsonl files to process (searched recursively)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be fixed, do not modify files",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: directory '{directory}' does not exist", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(directory.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"No jsonl files found in '{directory}'")
        sys.exit(0)

    print(f"Found {len(jsonl_files)} jsonl file(s) in '{directory}'")
    if args.dry_run:
        print("DRY RUN mode - no files will be modified\n")

    total_stats = {"total": 0, "fixed": 0}

    for filepath in jsonl_files:
        stats = process_jsonl_file(filepath, dry_run=args.dry_run)
        total_stats["total"] += stats["total"]
        total_stats["fixed"] += stats["fixed"]

        if stats["fixed"] > 0:
            action = "Would fix" if args.dry_run else "Fixed"
            rel = filepath
            try:
                rel = filepath.relative_to(Path.cwd())
            except ValueError:
                pass
            print(f"\n{action} {stats['fixed']} record(s) in {rel}:")
            for fix in stats["fixes"][:5]:
                print(f"  [{fix['index']}] {fix['task_id']}: {fix['reason']}")
            if len(stats["fixes"]) > 5:
                print(f"  ... and {len(stats['fixes']) - 5} more")

    print(f"\n{'=' * 60}")
    action = "Would fix" if args.dry_run else "Fixed"
    print(
        f"Total: {action} {total_stats['fixed']} of {total_stats['total']} records across {len(jsonl_files)} file(s)"
    )


if __name__ == "__main__":
    main()
