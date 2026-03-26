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
    return p.parse_args()


def pct(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return 100.0 * n / d


def main() -> None:
    args = parse_args()
    in_path = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by = defaultdict(list)
    for r in rows:
        key = (r["model"], r["mode"], r["language"])
        by[key].append(r)

    summary = []
    for (model, mode, lang), group in sorted(by.items()):
        n = len(group)
        c = Counter(r["error"] for r in group)
        exact = sum(1 for r in group if r["exact"])
        tok = sum(float(r.get("tokens", 0)) for r in group) / max(n, 1)
        sec = sum(float(r.get("seconds", 0.0)) for r in group) / max(n, 1)
        summary.append(
            {
                "model": model,
                "mode": mode,
                "language": lang,
                "attempts": n,
                "exact_rate": round(pct(exact, n), 2),
                "ok_rate": round(pct(c.get("ok", 0), n), 2),
                "parse_error_rate": round(pct(c.get("parse_error", 0), n), 2),
                "incomplete_rate": round(pct(c.get("incomplete", 0), n), 2),
                "semantic_mismatch_rate": round(pct(c.get("semantic_mismatch", 0), n), 2),
                "avg_tokens": round(tok, 2),
                "avg_seconds": round(sec, 2),
            }
        )

    sum_csv = out_dir / "summary.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        cols = list(summary[0].keys()) if summary else [
            "model",
            "mode",
            "language",
            "attempts",
            "exact_rate",
            "ok_rate",
            "parse_error_rate",
            "incomplete_rate",
            "semantic_mismatch_rate",
            "avg_tokens",
            "avg_seconds",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary:
            w.writerow(r)

    pair = defaultdict(list)
    for r in summary:
        pair[(r["model"], r["language"])].append(r)
    delta_rows = []
    for (model, lang), vals in sorted(pair.items()):
        a = [v for v in vals if v["mode"] == "constrained"]
        b = [v for v in vals if v["mode"] == "unconstrained"]
        if not a or not b:
            continue
        c = a[0]
        u = b[0]
        delta_rows.append(
            {
                "model": model,
                "language": lang,
                "constrained_exact": c["exact_rate"],
                "unconstrained_exact": u["exact_rate"],
                "exact_delta": round(c["exact_rate"] - u["exact_rate"], 2),
                "constrained_parse_error": c["parse_error_rate"],
                "unconstrained_parse_error": u["parse_error_rate"],
                "parse_error_delta": round(u["parse_error_rate"] - c["parse_error_rate"], 2),
            }
        )

    delta_csv = out_dir / "delta.csv"
    with delta_csv.open("w", newline="", encoding="utf-8") as f:
        cols = list(delta_rows[0].keys()) if delta_rows else [
            "model",
            "language",
            "constrained_exact",
            "unconstrained_exact",
            "exact_delta",
            "constrained_parse_error",
            "unconstrained_parse_error",
            "parse_error_delta",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in delta_rows:
            w.writerow(r)

    md = out_dir / "report.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Report\n\n")
        f.write(f"Input rows: {len(rows)}\n\n")
        f.write("## Key metric\n\n")
        f.write("- `exact_rate`: exact text match with expected answer\n")
        f.write("- `parse_error_rate`: output not parseable by target grammar\n")
        f.write("- `incomplete_rate`: parseable but not complete\n")
        f.write("- `semantic_mismatch_rate`: parseable/complete but wrong answer\n\n")
        f.write("## Constrained vs unconstrained delta\n\n")
        for r in delta_rows:
            f.write(
                f"- {r['model']} {r['language']}: exact delta {r['exact_delta']} pts, "
                f"parse-error reduction {r['parse_error_delta']} pts\n"
            )

    print(f"wrote {sum_csv}")
    print(f"wrote {delta_csv}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
