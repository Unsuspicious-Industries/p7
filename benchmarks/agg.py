#!/usr/bin/env python3
"""Aggregate benchmark raw.jsonl into analysis CSVs.

Usage:
    python -m benchmarks.agg --out-dir benchmarks/out/mini-4b
    python -m benchmarks.agg --in path/to/raw.jsonl --out-dir path/to/out
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

# Records whose error or output contain these strings are infrastructure
# failures, not model failures — exclude from analysis.
_SYSTEM_ERROR_PATTERNS = [
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

_MODELS_TO_EXCLUDE: set[str] = set()

_KNOWN_ERRORS = {"ok", "parse_error", "non_completable", "incomplete", "task_failed", "timeout"}


# ---------------------------------------------------------------------------
# Parsing CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="",
                   help="Path to raw.jsonl (default: <out-dir>/raw.jsonl)")
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--no-dedupe", action="store_true",
                   help="Skip deduplication; count all raw records")
    p.add_argument("--no-clean", action="store_true",
                   help="Keep system-error rows and excluded models")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _stdev(vals: list[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def _pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else round(100.0 * n / d, 2)


def _round(v: Optional[float], n: int = 4) -> Optional[float]:
    return None if v is None else round(v, n)


# ---------------------------------------------------------------------------
# Loading / cleaning / deduplication
# ---------------------------------------------------------------------------

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


def _is_system_error(row: dict) -> bool:
    combined = f"{row.get('error', '')} {row.get('output', '')}"
    return any(p in combined for p in _SYSTEM_ERROR_PATTERNS)


def _is_valid_model(row: dict) -> bool:
    return str(row.get("model") or "") not in _MODELS_TO_EXCLUDE


def clean_records(rows: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Remove system errors and excluded models. Returns (cleaned, stats)."""
    cleaned: list[dict] = []
    n_sys = n_model = 0
    reasons: Counter[str] = Counter()
    for row in rows:
        if not _is_valid_model(row):
            n_model += 1
            continue
        if _is_system_error(row):
            n_sys += 1
            reasons[str(row.get("error") or "")[:80]] += 1
            continue
        cleaned.append(row)
    return cleaned, {
        "input_records": len(rows),
        "removed_system_error": n_sys,
        "removed_invalid_model": n_model,
        "remaining_records": len(cleaned),
        "removal_reasons": dict(reasons),
    }


def _record_key(row: dict) -> tuple:
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


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """Keep the last-written record for each (backend, model, task, hash, mode, try)."""
    deduped: dict[tuple, dict] = {}
    order: list[tuple] = []
    for row in rows:
        key = _record_key(row)
        if key not in deduped:
            order.append(key)
        deduped[key] = row
    return [deduped[k] for k in order]


# ---------------------------------------------------------------------------
# Enrichment (adds derived scalar fields to each record)
# ---------------------------------------------------------------------------

def _task_type(row: dict) -> str:
    cat = str(row.get("category") or "")
    if ":" in cat:
        return cat.split(":", 1)[1]
    return cat or str(row.get("language") or "")


def _step_scalar_stats(row: dict) -> dict[str, Any]:
    """Derive scalar statistics from per-step arrays. Tolerates absent arrays."""
    pre = row.get("step_pre_entropies") or []
    post = row.get("step_entropies") or []
    retries = row.get("step_retries") or []

    if not pre:
        return {
            "avg_pre_entropy": None,
            "avg_post_entropy": None,
            "avg_entropy_reduction": None,
            "avg_retries": None,
            "max_retries": None,
            "pct_steps_retried": None,
        }

    avg_pre = sum(pre) / len(pre)
    avg_post = sum(post) / len(post) if post else None
    avg_red = (avg_pre - avg_post) if avg_post is not None else None
    avg_ret = sum(retries) / len(retries) if retries else None
    max_ret = max(retries) if retries else None
    pct_ret = _pct(sum(1 for r in retries if r > 0), len(retries)) if retries else None

    return {
        "avg_pre_entropy": _round(avg_pre),
        "avg_post_entropy": _round(avg_post),
        "avg_entropy_reduction": _round(avg_red),
        "avg_retries": _round(avg_ret),
        "max_retries": max_ret,
        "pct_steps_retried": pct_ret,
    }


def enrich_rows(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        copy = dict(row)
        copy["task_type"] = _task_type(copy)
        copy["char_count"] = len(str(copy.get("output") or ""))
        copy.update(_step_scalar_stats(copy))
        enriched.append(copy)
    return enriched


# ---------------------------------------------------------------------------
# Summarization (aggregate metrics per dimension group)
# ---------------------------------------------------------------------------

SUMMARY_METRICS = [
    "attempts",
    "exact_rate",
    "pass_rate",
    "parse_error_rate",
    "non_completable_rate",
    "incomplete_rate",
    "task_failed_rate",
    "timeout_rate",
    "other_error_rate",
    "avg_tokens",
    "avg_seconds",
    "avg_think_tokens",
    "avg_formal_tokens",
]


def summarize(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    """Aggregate accuracy/performance metrics per dimension group."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by[tuple(str(row.get(d) or "") for d in dimensions)].append(row)

    result = []
    for key, group in sorted(by.items()):
        n = len(group)
        c = Counter(row.get("error") for row in group)
        exact = sum(1 for row in group if row.get("exact"))
        other = sum(cnt for err, cnt in c.items() if err not in _KNOWN_ERRORS)
        result.append({
            **dict(zip(dimensions, key)),
            "attempts": n,
            "exact_rate": _pct(exact, n),
            "pass_rate": _pct(c.get("ok", 0), n),
            "parse_error_rate": _pct(c.get("parse_error", 0), n),
            "non_completable_rate": _pct(c.get("non_completable", 0), n),
            "incomplete_rate": _pct(c.get("incomplete", 0), n),
            "task_failed_rate": _pct(c.get("task_failed", 0), n),
            "timeout_rate": _pct(c.get("timeout", 0), n),
            "other_error_rate": _pct(other, n),
            "avg_tokens": _round(_mean([float(r.get("tokens") or 0) for r in group]), 2),
            "avg_seconds": _round(_mean([float(r.get("seconds") or 0) for r in group]), 2),
            "avg_think_tokens": _round(_mean([float(r.get("think_tokens") or 0) for r in group]), 2),
            "avg_formal_tokens": _round(_mean([float(r.get("formal_tokens") or 0) for r in group]), 2),
        })
    return result


def summary_columns(dimensions: tuple[str, ...]) -> list[str]:
    return [*dimensions, *SUMMARY_METRICS]


# ---------------------------------------------------------------------------
# Delta: constrained_direct vs unconstrained
# ---------------------------------------------------------------------------

def delta_rows(summary: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    """Compute constrained_direct vs unconstrained deltas per dimension group."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for row in summary:
        by[tuple(str(row.get(d) or "") for d in dimensions)].append(row)

    result = []
    for key, vals in sorted(by.items()):
        constrained = [v for v in vals if v.get("mode") == "constrained_direct"]
        unconstrained = (
            [v for v in vals if v.get("mode") == "unconstrained"]
            or [v for v in vals if v.get("mode") == "unconstrained_cleaned"]
        )
        if not constrained or not unconstrained:
            continue
        c, u = constrained[0], unconstrained[0]
        result.append({
            **dict(zip(dimensions, key)),
            "unconstrained_mode": u["mode"],
            "constrained_pass_rate": c["pass_rate"],
            "unconstrained_pass_rate": u["pass_rate"],
            "pass_rate_delta": round(c["pass_rate"] - u["pass_rate"], 2),
            "constrained_exact": c["exact_rate"],
            "unconstrained_exact": u["exact_rate"],
            "exact_delta": round(c["exact_rate"] - u["exact_rate"], 2),
            "constrained_parse_error": c["parse_error_rate"],
            "unconstrained_parse_error": u["parse_error_rate"],
            "parse_error_delta": round(u["parse_error_rate"] - c["parse_error_rate"], 2),
            "constrained_non_completable": c["non_completable_rate"],
            "unconstrained_non_completable": u["non_completable_rate"],
            "constrained_avg_tokens": c["avg_tokens"],
            "unconstrained_avg_tokens": u["avg_tokens"],
        })
    return result


def mode_pair_rows(
    summary: list[dict],
    dimensions: tuple[str, ...],
    *,
    primary_mode: str,
    baseline_mode: str,
) -> list[dict]:
    """Compare two specific modes across dimension groups."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for row in summary:
        by[tuple(str(row.get(d) or "") for d in dimensions)].append(row)

    result = []
    for key, vals in sorted(by.items()):
        primary = [v for v in vals if v.get("mode") == primary_mode]
        baseline = [v for v in vals if v.get("mode") == baseline_mode]
        if not primary or not baseline:
            continue
        left, right = primary[0], baseline[0]
        result.append({
            **dict(zip(dimensions, key)),
            "primary_mode": primary_mode,
            "baseline_mode": baseline_mode,
            "primary_pass_rate": left["pass_rate"],
            "baseline_pass_rate": right["pass_rate"],
            "pass_rate_delta": round(left["pass_rate"] - right["pass_rate"], 2),
            "primary_exact": left["exact_rate"],
            "baseline_exact": right["exact_rate"],
            "exact_delta": round(left["exact_rate"] - right["exact_rate"], 2),
            "primary_avg_tokens": left["avg_tokens"],
            "baseline_avg_tokens": right["avg_tokens"],
            "avg_tokens_delta": round((left["avg_tokens"] or 0) - (right["avg_tokens"] or 0), 2),
            "primary_avg_seconds": left["avg_seconds"],
            "baseline_avg_seconds": right["avg_seconds"],
        })
    return result


# ---------------------------------------------------------------------------
# Entropy summary (constrained modes only, from step arrays)
# ---------------------------------------------------------------------------

_ENTROPY_COLUMNS = [
    "n_records",
    "n_steps",
    "mean_pre_entropy",
    "std_pre_entropy",
    "mean_post_entropy",
    "std_post_entropy",
    "mean_entropy_reduction",
    "mean_retries",
    "pct_steps_retried",
]


def entropy_summary_rows(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    """Aggregate per-step entropy and retry stats by dimension group.

    Pools all steps across records in a group, so the result is a true
    per-step mean rather than a mean-of-means.  Omits groups with no
    step data (unconstrained runs have empty arrays).
    """
    by: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by[tuple(str(row.get(d) or "") for d in dimensions)].append(row)

    result = []
    for key, group in sorted(by.items()):
        pre_all: list[float] = []
        post_all: list[float] = []
        ret_all: list[float] = []
        for row in group:
            pre_all.extend(row.get("step_pre_entropies") or [])
            post_all.extend(row.get("step_entropies") or [])
            ret_all.extend(float(r) for r in (row.get("step_retries") or []))

        if not pre_all:
            continue  # no step data — skip (unconstrained or pre-entropy era)

        reductions = [p - q for p, q in zip(pre_all, post_all)] if post_all else []
        result.append({
            **dict(zip(dimensions, key)),
            "n_records": len(group),
            "n_steps": len(pre_all),
            "mean_pre_entropy": _round(_mean(pre_all)),
            "std_pre_entropy": _round(_stdev(pre_all)),
            "mean_post_entropy": _round(_mean(post_all)),
            "std_post_entropy": _round(_stdev(post_all)),
            "mean_entropy_reduction": _round(_mean(reductions)),
            "mean_retries": _round(_mean(ret_all)),
            "pct_steps_retried": _round(
                _pct(sum(1 for r in ret_all if r > 0), len(ret_all))
            ),
        })
    return result


# ---------------------------------------------------------------------------
# Stop-reason breakdown
# ---------------------------------------------------------------------------

def stop_reason_rows(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    """Count stop reasons per dimension group, sorted by frequency."""
    by: dict[tuple, Counter] = defaultdict(Counter)
    totals: dict[tuple, int] = defaultdict(int)
    for row in rows:
        key = tuple(str(row.get(d) or "") for d in dimensions)
        by[key][str(row.get("stop_reason") or "")] += 1
        totals[key] += 1

    result = []
    for key, counts in sorted(by.items()):
        total = totals[key]
        for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            result.append({
                **dict(zip(dimensions, key)),
                "stop_reason": reason,
                "count": count,
                "pct": _pct(count, total),
            })
    return result


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _sanitize_csv_row(row: dict, columns: list[str]) -> dict:
    """Flatten multi-line strings to single lines for CSV safety.

    Newlines inside fields produce multi-line quoted cells that confuse
    many tools.  We keep originals in the JSONL; the CSV is for analysis.
    """
    out = {}
    for col in columns:
        v = row.get(col, "")
        if isinstance(v, str):
            v = v.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
        out[col] = v
    return out


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_sanitize_csv_row(row, columns))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


# ---------------------------------------------------------------------------
# Flat CSV column layout for benchmark_records.csv
# ---------------------------------------------------------------------------

RECORD_COLUMNS = [
    # identity
    "backend", "model", "mode", "task_id", "language", "grammar", "category",
    "task_type", "resolution_mode", "task_hash", "resolution_hash", "try", "seed",
    # outcome
    "error", "passed", "exact", "parse_ok", "parse_complete", "semantic_ok",
    "resolution_error", "parse_error", "stop_reason",
    # content
    "expected", "output", "output_extracted", "raw_output",
    # generation counts
    "tokens", "seconds", "think_tokens", "formal_tokens",
    # constrained step statistics (scalars; arrays are in cleaned_raw.jsonl)
    "avg_pre_entropy", "avg_post_entropy", "avg_entropy_reduction",
    "avg_retries", "max_retries", "pct_steps_retried",
    # misc
    "char_count",
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _write_summary(out_dir: Path, filename: str, rows: list[dict],
                   dimensions: tuple[str, ...]) -> list[dict]:
    summary = summarize(rows, dimensions)
    write_csv(out_dir / filename, summary, summary_columns(dimensions))
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    in_path = Path(args.inp) if args.inp else out_dir / "raw.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows(in_path)
    if not raw_rows:
        raise SystemExit(f"No records found in {in_path}")

    enriched = enrich_rows(raw_rows)
    if args.no_clean:
        cleaned = enriched
        clean_stats: dict[str, Any] = {
            "input_records": len(enriched), "removed_system_error": 0,
            "removed_invalid_model": 0, "remaining_records": len(enriched),
            "removal_reasons": {},
        }
    else:
        cleaned, clean_stats = clean_records(enriched)

    rows = cleaned if args.no_dedupe else dedupe_rows(cleaned)

    # ── Summary tables ──────────────────────────────────────────────────────
    dims_model_mode     = ("backend", "model", "mode")
    dims_model_lang     = ("backend", "model", "language")
    dims_mode_lang      = ("backend", "mode", "language")
    dims_mode_tasktype  = ("backend", "mode", "task_type")
    dims_task           = ("backend", "model", "mode", "task_id", "language",
                           "task_type", "category", "resolution_mode")

    s_model_mode    = _write_summary(out_dir, "summary_by_model_mode.csv",    rows, dims_model_mode)
    s_model_lang    = _write_summary(out_dir, "summary_by_model_language.csv", rows, dims_model_lang)
    s_mode_lang     = _write_summary(out_dir, "summary_by_mode_language.csv",  rows, dims_mode_lang)
    s_mode_tasktype = _write_summary(out_dir, "summary_by_mode_task_type.csv", rows, dims_mode_tasktype)
    _write_summary(out_dir, "summary_by_task.csv", rows, dims_task)

    # ── Delta: constrained_direct vs unconstrained ───────────────────────────
    delta_cols = [
        "unconstrained_mode",
        "constrained_pass_rate", "unconstrained_pass_rate", "pass_rate_delta",
        "constrained_exact", "unconstrained_exact", "exact_delta",
        "constrained_parse_error", "unconstrained_parse_error", "parse_error_delta",
        "constrained_non_completable", "unconstrained_non_completable",
        "constrained_avg_tokens", "unconstrained_avg_tokens",
    ]
    write_csv(
        out_dir / "delta_by_model_language.csv",
        delta_rows(s_model_lang, dims_model_lang),
        ["backend", "model", "language", *delta_cols],
    )
    write_csv(
        out_dir / "delta_by_task_type.csv",
        delta_rows(s_mode_tasktype, ("backend", "task_type")),
        ["backend", "task_type", *delta_cols],
    )

    # ── Mixed vs unconstrained_thinking comparison (optional) ────────────────
    mixed_vs_think_cols = [
        "primary_mode", "baseline_mode",
        "primary_pass_rate", "baseline_pass_rate", "pass_rate_delta",
        "primary_exact", "baseline_exact", "exact_delta",
        "primary_avg_tokens", "baseline_avg_tokens", "avg_tokens_delta",
        "primary_avg_seconds", "baseline_avg_seconds",
    ]
    write_csv(
        out_dir / "comparison_mixed_vs_thinking_by_model_language.csv",
        mode_pair_rows(s_model_lang, dims_model_lang,
                       primary_mode="constrained_mixed",
                       baseline_mode="unconstrained_thinking"),
        ["backend", "model", "language", *mixed_vs_think_cols],
    )

    # ── Per-step entropy analysis (constrained modes only) ───────────────────
    write_csv(
        out_dir / "entropy_by_model_mode.csv",
        entropy_summary_rows(rows, dims_model_mode),
        ["backend", "model", "mode", *_ENTROPY_COLUMNS],
    )

    # ── Stop-reason breakdown ────────────────────────────────────────────────
    write_csv(
        out_dir / "stop_reasons.csv",
        stop_reason_rows(rows, ("backend", "model", "mode")),
        ["backend", "model", "mode", "stop_reason", "count", "pct"],
    )

    # ── Individual records (flat CSV; arrays in JSONL) ───────────────────────
    write_csv(out_dir / "benchmark_records.csv", rows, RECORD_COLUMNS)
    write_jsonl(out_dir / "cleaned_raw.jsonl", rows)
    write_json(out_dir / "process_stats.json", {
        **clean_stats,
        "raw_records": len(raw_rows),
        "deduped_records": len(rows),
    })

    outputs = [
        "summary_by_model_mode.csv",
        "summary_by_model_language.csv",
        "summary_by_mode_language.csv",
        "summary_by_mode_task_type.csv",
        "summary_by_task.csv",
        "delta_by_model_language.csv",
        "delta_by_task_type.csv",
        "comparison_mixed_vs_thinking_by_model_language.csv",
        "entropy_by_model_mode.csv",
        "stop_reasons.csv",
        "benchmark_records.csv",
        "cleaned_raw.jsonl",
        "process_stats.json",
    ]
    for name in outputs:
        print(f"  wrote {out_dir / name}")


if __name__ == "__main__":
    main()
