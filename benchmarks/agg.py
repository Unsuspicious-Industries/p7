#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate p7 benchmark runs.")
    p.add_argument("--in", dest="inp", default=str(OUT / "raw.jsonl"))
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Count duplicate raw records instead of keeping the last record per job",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def delta_rows(summary: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    pair = defaultdict(list)
    for row in summary:
        pair[tuple(row.get(dim, "") for dim in dimensions)].append(row)

    rows = []
    for key, vals in sorted(pair.items()):
        constrained = [
            val for val in vals if val["mode"] in {"constrained", "constrained_direct"}
        ]
        raw_unconstrained = [val for val in vals if val["mode"] == "unconstrained_raw"]
        assisted_unconstrained = [val for val in vals if val["mode"] == "unconstrained"]
        unconstrained = raw_unconstrained or assisted_unconstrained
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


def main() -> None:
    args = parse_args()
    in_path = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows(in_path)

    rows = raw_rows if args.no_dedupe else dedupe_rows(raw_rows)

    summary = summarize(rows, ("backend", "model", "mode", "language"))
    category_summary = summarize(
        rows, ("backend", "model", "mode", "language", "category")
    )

    sum_csv = out_dir / "summary.csv"
    summary_cols = ["backend", "model", "mode", "language", *SUMMARY_METRICS]
    write_csv(sum_csv, summary, summary_cols)

    category_csv = out_dir / "summary_by_category.csv"
    category_cols = [
        "backend",
        "model",
        "mode",
        "language",
        "category",
        *SUMMARY_METRICS,
    ]
    write_csv(category_csv, category_summary, category_cols)

    delta_rows_ = delta_rows(summary, ("backend", "model", "language"))
    category_delta_rows = delta_rows(
        category_summary, ("backend", "model", "language", "category")
    )

    delta_csv = out_dir / "delta.csv"
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
    write_csv(
        delta_csv, delta_rows_, ["backend", "model", "language", *delta_metric_cols]
    )

    category_delta_csv = out_dir / "delta_by_category.csv"
    write_csv(
        category_delta_csv,
        category_delta_rows,
        ["backend", "model", "language", "category", *delta_metric_cols],
    )

    md = out_dir / "report.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Report\n\n")
        f.write(f"Input rows: {len(raw_rows)}\n")
        if len(rows) != len(raw_rows):
            f.write(f"Deduped rows: {len(rows)}\n")
        f.write("\n")
        f.write("## Key metric\n\n")
        f.write("- `exact_rate`: exact text match with expected answer\n")
        f.write("- `parse_error_rate`: output not parseable by target grammar\n")
        f.write(
            "- `non_completable_rate`: output is a dead grammar prefix; this should be 0 for constrained decoding\n"
        )
        f.write("- `incomplete_rate`: parseable but not complete\n")
        f.write("- `semantic_mismatch_rate`: parseable/complete but wrong answer\n")
        f.write("- `timeout_rate`: job hit the configured timeout\n")
        f.write("- `other_error_rate`: uncategorized runtime/model errors\n\n")
        f.write("## Constrained vs unconstrained delta\n\n")
        f.write("Uses `unconstrained_raw` as the baseline when present.\n\n")
        for r in delta_rows_:
            f.write(
                f"- {r['backend']} {r['model']} {r['language']} vs {r['unconstrained_mode']}: exact delta {r['exact_delta']} pts, "
                f"parse-error reduction {r['parse_error_delta']} pts\n"
            )

    print(f"wrote {sum_csv}")
    print(f"wrote {category_csv}")
    print(f"wrote {delta_csv}")
    print(f"wrote {category_delta_csv}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
