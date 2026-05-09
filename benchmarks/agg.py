#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process proposition7 benchmark runs into CSV summaries."
    )
    p.add_argument("--in", dest="inp", default=str(OUT / "raw.jsonl"))
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Count duplicate raw records instead of keeping the last record per job",
    )
    p.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep system-error rows and excluded models instead of filtering them out",
    )
    return p.parse_args()


def pct(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return 100.0 * n / d


SUMMARY_METRICS = [
    "attempts",
    "exact_rate",
    "pass_rate",
    "ok_rate",
    "parse_error_rate",
    "non_completable_rate",
    "incomplete_rate",
    "semantic_mismatch_rate",
    "timeout_rate",
    "other_error_rate",
    "avg_tokens",
    "avg_seconds",
]


def record_key(row: dict) -> tuple[str, str, str, str, str, str, int]:
    try:
        attempt = int(row.get("try", 0))
    except (TypeError, ValueError):
        attempt = 0
    return (
        str(row.get("backend", "")),
        str(row.get("model", "")),
        str(row.get("task_id", "")),
        str(row.get("task_hash", "")),
        str(row.get("resolution_hash", "")),
        str(row.get("mode", "")),
        attempt,
    )


def task_type(row: dict[str, Any]) -> str:
    category = str(row.get("category", "") or "")
    if ":" in category:
        return category.split(":", 1)[1]
    return category or str(row.get("language", "") or "")


def is_system_error(row: dict[str, Any]) -> bool:
    combined = f"{row.get('error', '')} {row.get('output', '')}"
    return any(pattern in combined for pattern in ERROR_PATTERNS)


def is_valid_model(row: dict[str, Any]) -> bool:
    return str(row.get("model", "") or "") not in MODELS_TO_EXCLUDE


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        output = str(copy.get("output", "") or "")
        copy["task_type"] = task_type(copy)
        copy["char_count"] = len(output)
        copy["token_count"] = len(output.split())
        enriched.append(copy)
    return enriched


def clean_records(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "input_records": len(rows),
        "removed_system_error": 0,
        "removed_invalid_model": 0,
        "remaining_records": 0,
        "removal_reasons": {},
    }
    reasons: Counter[str] = Counter()

    for row in rows:
        if not is_valid_model(row):
            stats["removed_invalid_model"] += 1
            reasons["invalid_model"] += 1
            continue
        if is_system_error(row):
            stats["removed_system_error"] += 1
            reasons[str(row.get("error", "") or "system_error")[:80]] += 1
            continue
        cleaned.append(row)

    stats["remaining_records"] = len(cleaned)
    stats["removal_reasons"] = dict(reasons)
    return cleaned, stats


def dedupe_rows(rows: list[dict]) -> list[dict]:
    deduped: dict[tuple[str, str, str, str, str, str, int], dict] = {}
    order: list[tuple[str, str, str, str, str, str, int]] = []
    for row in rows:
        key = record_key(row)
        if key not in deduped:
            order.append(key)
        deduped[key] = row
    return [deduped[key] for key in order]


def summarize(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    by = defaultdict(list)
    for row in rows:
        by[tuple(row.get(dim, "") for dim in dimensions)].append(row)

    summary = []
    for key, group in sorted(by.items()):
        n = len(group)
        c = Counter(row["error"] for row in group)
        exact = sum(1 for row in group if row["exact"])
        known = {
            "ok",
            "parse_error",
            "non_completable",
            "incomplete",
            "semantic_mismatch",
            "timeout",
        }
        other_errors = sum(count for error, count in c.items() if error not in known)
        tok = sum(float(row.get("tokens", 0)) for row in group) / max(n, 1)
        sec = sum(float(row.get("seconds", 0.0)) for row in group) / max(n, 1)
        summary.append(
            {
                **dict(zip(dimensions, key)),
                "attempts": n,
                "exact_rate": round(pct(exact, n), 2),
                "pass_rate": round(pct(c.get("ok", 0), n), 2),
                "ok_rate": round(pct(c.get("ok", 0), n), 2),
                "parse_error_rate": round(pct(c.get("parse_error", 0), n), 2),
                "non_completable_rate": round(pct(c.get("non_completable", 0), n), 2),
                "incomplete_rate": round(pct(c.get("incomplete", 0), n), 2),
                "semantic_mismatch_rate": round(
                    pct(c.get("semantic_mismatch", 0), n), 2
                ),
                "timeout_rate": round(pct(c.get("timeout", 0), n), 2),
                "other_error_rate": round(pct(other_errors, n), 2),
                "avg_tokens": round(tok, 2),
                "avg_seconds": round(sec, 2),
            }
        )
    return summary


def load_rows(path: Path) -> list[dict]:
    if not path.exists() or path.is_dir():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def summary_columns(dimensions: tuple[str, ...]) -> list[str]:
    return [*dimensions, *SUMMARY_METRICS]


def write_summary(
    path: Path, rows: list[dict[str, Any]], dimensions: tuple[str, ...]
) -> list[dict[str, Any]]:
    summary = summarize(rows, dimensions)
    write_csv(path, summary, summary_columns(dimensions))
    return summary


def delta_rows(summary: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    pair = defaultdict(list)
    for row in summary:
        pair[tuple(row.get(dim, "") for dim in dimensions)].append(row)

    rows = []
    for key, vals in sorted(pair.items()):
        try:
            constrained = [
                val for val in vals if val.get("mode") == "constrained_direct"
            ]
            raw_unconstrained = [
                val for val in vals if val.get("mode") == "unconstrained"
            ]
            assisted_unconstrained = [
                val for val in vals if val.get("mode") == "unconstrained_cleaned"
            ]
            unconstrained = raw_unconstrained or assisted_unconstrained
        except KeyError:
            continue
        if not constrained or not unconstrained:
            continue
        c = constrained[0]
        u = unconstrained[0]
        rows.append(
            {
                **dict(zip(dimensions, key)),
                "unconstrained_mode": u["mode"],
                "constrained_exact": c["exact_rate"],
                "unconstrained_exact": u["exact_rate"],
                "exact_delta": round(c["exact_rate"] - u["exact_rate"], 2),
                "constrained_pass_rate": c["pass_rate"],
                "unconstrained_pass_rate": u["pass_rate"],
                "pass_rate_delta": round(c["pass_rate"] - u["pass_rate"], 2),
                "constrained_parse_error": c["parse_error_rate"],
                "unconstrained_parse_error": u["parse_error_rate"],
                "parse_error_delta": round(
                    u["parse_error_rate"] - c["parse_error_rate"], 2
                ),
                "constrained_non_completable": c["non_completable_rate"],
                "unconstrained_non_completable": u["non_completable_rate"],
            }
        )
    return rows


def mode_pair_rows(
    summary: list[dict],
    dimensions: tuple[str, ...],
    *,
    primary_mode: str,
    baseline_mode: str,
) -> list[dict]:
    pair = defaultdict(list)
    for row in summary:
        pair[tuple(row.get(dim, "") for dim in dimensions)].append(row)

    rows = []
    for key, vals in sorted(pair.items()):
        try:
            primary = [val for val in vals if val.get("mode") == primary_mode]
            baseline = [val for val in vals if val.get("mode") == baseline_mode]
        except KeyError:
            continue
        if not primary or not baseline:
            continue
        left = primary[0]
        right = baseline[0]
        rows.append(
            {
                **dict(zip(dimensions, key)),
                "primary_mode": primary_mode,
                "baseline_mode": baseline_mode,
                "primary_exact": left["exact_rate"],
                "baseline_exact": right["exact_rate"],
                "exact_delta": round(left["exact_rate"] - right["exact_rate"], 2),
                "primary_pass_rate": left["pass_rate"],
                "baseline_pass_rate": right["pass_rate"],
                "pass_rate_delta": round(left["pass_rate"] - right["pass_rate"], 2),
                "primary_parse_error": left["parse_error_rate"],
                "baseline_parse_error": right["parse_error_rate"],
                "parse_error_delta": round(
                    right["parse_error_rate"] - left["parse_error_rate"], 2
                ),
                "primary_non_completable": left["non_completable_rate"],
                "baseline_non_completable": right["non_completable_rate"],
                "primary_avg_tokens": left["avg_tokens"],
                "baseline_avg_tokens": right["avg_tokens"],
                "avg_tokens_delta": round(left["avg_tokens"] - right["avg_tokens"], 2),
                "primary_avg_seconds": left["avg_seconds"],
                "baseline_avg_seconds": right["avg_seconds"],
                "avg_seconds_delta": round(
                    left["avg_seconds"] - right["avg_seconds"], 2
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    in_path = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = enrich_rows(load_rows(in_path))
    cleaned_rows, clean_stats = (
        (
            raw_rows,
            {
                "input_records": len(raw_rows),
                "removed_system_error": 0,
                "removed_invalid_model": 0,
                "remaining_records": len(raw_rows),
                "removal_reasons": {},
            },
        )
        if args.no_clean
        else clean_records(raw_rows)
    )
    rows = cleaned_rows if args.no_dedupe else dedupe_rows(cleaned_rows)

    task_dimensions = (
        "backend",
        "model",
        "mode",
        "task_id",
        "language",
        "task_type",
        "category",
        "resolution_mode",
    )
    model_language_dimensions = ("backend", "model", "language")
    mode_task_type_dimensions = ("backend", "mode", "task_type")

    summaries = {
        "summary_overall.csv": ("backend",),
        "summary_by_mode.csv": ("backend", "mode"),
        "summary_by_language.csv": ("backend", "language"),
        "summary_by_task_type.csv": ("backend", "task_type"),
        "summary_by_model.csv": ("backend", "model"),
        "summary_by_model_mode.csv": ("backend", "model", "mode"),
        "summary_by_model_language.csv": model_language_dimensions,
        "summary_by_mode_language.csv": ("backend", "mode", "language"),
        "summary_by_mode_task_type.csv": mode_task_type_dimensions,
        "summary_by_task.csv": task_dimensions,
    }

    summary_outputs: dict[str, list[dict[str, Any]]] = {}
    for file_name, dimensions in summaries.items():
        summary_outputs[file_name] = write_summary(
            out_dir / file_name, rows, dimensions
        )

    delta_metric_cols = [
        "unconstrained_mode",
        "constrained_exact",
        "unconstrained_exact",
        "exact_delta",
        "constrained_pass_rate",
        "unconstrained_pass_rate",
        "pass_rate_delta",
        "constrained_parse_error",
        "unconstrained_parse_error",
        "parse_error_delta",
        "constrained_non_completable",
        "unconstrained_non_completable",
    ]
    model_language_delta = delta_rows(
        summary_outputs["summary_by_model_language.csv"], model_language_dimensions
    )
    mode_task_type_delta = delta_rows(
        summary_outputs["summary_by_mode_task_type.csv"], ("backend", "task_type")
    )
    mixed_vs_unconstrained_thinking_by_model_language = mode_pair_rows(
        summary_outputs["summary_by_model_language.csv"],
        model_language_dimensions,
        primary_mode="constrained_mixed",
        baseline_mode="unconstrained_thinking",
    )
    mixed_vs_unconstrained_thinking_by_task_type = mode_pair_rows(
        summary_outputs["summary_by_mode_task_type.csv"],
        ("backend", "task_type"),
        primary_mode="constrained_mixed",
        baseline_mode="unconstrained_thinking",
    )
    write_csv(
        out_dir / "delta_by_model_language.csv",
        model_language_delta,
        ["backend", "model", "language", *delta_metric_cols],
    )
    write_csv(
        out_dir / "delta_by_task_type.csv",
        mode_task_type_delta,
        ["backend", "task_type", *delta_metric_cols],
    )
    comparison_metric_cols = [
        "primary_mode",
        "baseline_mode",
        "primary_exact",
        "baseline_exact",
        "exact_delta",
        "primary_pass_rate",
        "baseline_pass_rate",
        "pass_rate_delta",
        "primary_parse_error",
        "baseline_parse_error",
        "parse_error_delta",
        "primary_non_completable",
        "baseline_non_completable",
        "primary_avg_tokens",
        "baseline_avg_tokens",
        "avg_tokens_delta",
        "primary_avg_seconds",
        "baseline_avg_seconds",
        "avg_seconds_delta",
    ]
    write_csv(
        out_dir / "comparison_mixed_vs_unconstrained_thinking_by_model_language.csv",
        mixed_vs_unconstrained_thinking_by_model_language,
        ["backend", "model", "language", *comparison_metric_cols],
    )
    write_csv(
        out_dir / "comparison_mixed_vs_unconstrained_thinking_by_task_type.csv",
        mixed_vs_unconstrained_thinking_by_task_type,
        ["backend", "task_type", *comparison_metric_cols],
    )

    benchmark_record_columns = [
        "backend",
        "model",
        "mode",
        "task_id",
        "language",
        "task_type",
        "category",
        "resolution_mode",
        "error",
        "passed",
        "exact",
        "parse_ok",
        "parse_complete",
        "tokens",
        "seconds",
        "try",
        "task_hash",
        "resolution_hash",
        "grammar",
        "expected",
        "output",
        "output_extracted",
        "semantic_ok",
        "resolution_error",
        "resolution_observed",
        "resolution_expected",
        "parse_error",
        "stop_reason",
        "diagnostics",
        "thoughts",
        "think_tokens",
        "formal_tokens",
        "reasoning_blocks",
        "seed",
        "raw_output",
        "_source",
        "task_type",
        "char_count",
        "token_count",
    ]
    write_csv(out_dir / "benchmark_records.csv", rows, benchmark_record_columns)
    write_jsonl(out_dir / "cleaned_raw.jsonl", rows)
    write_json(
        out_dir / "process_stats.json",
        {
            **clean_stats,
            "deduped_records": len(rows),
            "raw_records": len(raw_rows),
        },
    )

    for file_name in [
        "summary_overall.csv",
        "summary_by_mode.csv",
        "summary_by_language.csv",
        "summary_by_task_type.csv",
        "summary_by_model.csv",
        "summary_by_model_mode.csv",
        "summary_by_model_language.csv",
        "summary_by_mode_language.csv",
        "summary_by_mode_task_type.csv",
        "summary_by_task.csv",
        "delta_by_model_language.csv",
        "delta_by_task_type.csv",
        "comparison_mixed_vs_unconstrained_thinking_by_model_language.csv",
        "comparison_mixed_vs_unconstrained_thinking_by_task_type.csv",
        "benchmark_records.csv",
        "cleaned_raw.jsonl",
        "process_stats.json",
    ]:
        print(f"wrote {out_dir / file_name}")


if __name__ == "__main__":
    main()
