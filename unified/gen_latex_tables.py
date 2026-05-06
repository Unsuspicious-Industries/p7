#!/usr/bin/env python3
"""
Generate per-table LaTeX files from summary CSV (pure Python, no pandas).
"""

import csv
import sys
from pathlib import Path


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    return s.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')


def csv_to_latex(
    csv_path: Path,
    caption: str = "",
    label: str = "",
    col_format: str = None,
    highlight_min: str = None,
    highlight_max: str = None,
    rotate_header: bool = False,
) -> str:
    """Convert CSV to LaTeX table."""
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return ""
        headers = reader.fieldnames

    if col_format is None:
        col_format = "l" + "r" * (len(headers) - 1)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_format}}}")
    lines.append("    \\toprule")

    # Header
    if rotate_header:
        header = " & ".join(f"\\rotatebox{{90}}{{{escape_latex(h)}}}" for h in headers)
    else:
        header = " & ".join(escape_latex(h) for h in headers)
    lines.append(f"    {header} \\\\")
    lines.append("    \\midrule")

    # Rows
    for row in rows:
        values = []
        for i, h in enumerate(headers):
            val = escape_latex(row.get(h, ""))
            # Highlighting
            if highlight_min and h == highlight_min:
                # Find min value
                min_val = min(float(row.get(h, "inf")) for r in rows if r.get(h))
                if float(row.get(h, "inf")) == min_val:
                    val = f"\\textbf{{{val}}}"
            if highlight_max and h == highlight_max:
                max_val = max(float(row.get(h, "-inf")) for r in rows if r.get(h))
                if float(row.get(h, "-inf")) == max_val:
                    val = f"\\textbf{{{val}}}"
            values.append(val)
        row_str = " & ".join(values)
        lines.append(f"    {row_str} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    if caption and not label:
        lines.append(f"  \\caption{{{caption}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_per_table_latex(out_dir: Path):
    """Generate one .tex file per table."""
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Generating per-table LaTeX files]")

    # 1. Error breakdown by mode (KEY PAPER TABLE)
    error_csv = out_dir / "error_breakdown.csv"
    if error_csv.exists():
        tex = csv_to_latex(
            error_csv,
            caption="Error distribution by generation mode. Constrained modes eliminate parse errors (0\\%), shifting failures to Task Invalid (logic errors).",
            label="tab:error_breakdown",
            highlight_max="pass_rate",
        )
        (tables_dir / "table1_error_breakdown.tex").write_text(tex)
        print(f"  Generated {tables_dir / 'table1_error_breakdown.tex'}")

    # 2. Summary by mode and language
    summary_csv = out_dir / "summary.csv"
    if summary_csv.exists():
        # Create simplified table
        data = {}
        with summary_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["mode"], row["language"])
                if key not in data:
                    data[key] = {"mode": row["mode"], "language": row["language"]}
                # Average pass_rate and parse_error_rate
                for col in ["pass_rate", "parse_error_rate", "semantic_mismatch_rate"]:
                    if col not in data[key]:
                        data[key][col] = []
                    try:
                        data[key][col].append(float(row[col]))
                    except ValueError:
                        pass

        # Write simplified CSV
        simple_path = tables_dir / "summary_simple.csv"
        with simple_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "language", "pass_rate", "parse_error_rate", "semantic_mismatch_rate"])
            writer.writeheader()
            for key in sorted(data.keys()):
                row = data[key].copy()
                for col in ["pass_rate", "parse_error_rate", "semantic_mismatch_rate"]:
                    if col in row:
                        row[col] = f"{sum(row[col])/len(row[col]):.2f}"
                writer.writerow(row)

        tex = csv_to_latex(
            simple_path,
            caption="Summary of benchmark results by mode and language.",
            label="tab:summary_mode_lang",
        )
        (tables_dir / "table2_summary_mode_lang.tex").write_text(tex)
        print(f"  Generated {tables_dir / 'table2_summary_mode_lang.tex'}")

    # 3. Delta: Mixed vs Direct
    summary_csv = out_dir / "summary.csv"
    if summary_csv.exists():
        # Pivot: model -> {constrained_direct, constrained_mixed}
        models = {}
        with summary_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["mode"] in ["constrained_direct", "constrained_mixed"]:
                    model = row["model"]
                    mode = row["mode"]
                    if model not in models:
                        models[model] = {}
                    try:
                        models[model][mode] = float(row["pass_rate"])
                    except ValueError:
                        pass

        # Calculate delta
        delta_data = []
        for model, modes in sorted(models.items()):
            if "constrained_direct" in modes and "constrained_mixed" in modes:
                delta = modes["constrained_mixed"] - modes["constrained_direct"]
                delta_data.append({
                    "model": model,
                    "direct": f"{modes['constrained_direct']:.2f}",
                    "mixed": f"{modes['constrained_mixed']:.2f}",
                    "delta": f"{delta:+.2f}",
                })

        if delta_data:
            delta_path = tables_dir / "mixed_vs_direct.csv"
            with delta_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["model", "direct", "mixed", "delta"])
                writer.writeheader()
                for row in delta_data:
                    writer.writerow(row)

            tex = csv_to_latex(
                delta_path,
                caption="Constrained Mixed vs Direct: Reasoning improves pass rate.",
                label="tab:mixed_vs_direct",
                highlight_max="delta",
            )
            (tables_dir / "table3_mixed_vs_direct.tex").write_text(tex)
            print(f"  Generated {tables_dir / 'table3_mixed_vs_direct.tex'}")

    # 4. Global pass rate comparison
    summary_csv = out_dir / "summary.csv"
    if summary_csv.exists():
        mode_stats = {}
        with summary_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mode = row["mode"]
                if mode not in mode_stats:
                    mode_stats[mode] = {"mode": mode, "pass_rates": []}
                try:
                    mode_stats[mode]["pass_rates"].append(float(row["pass_rate"]))
                except ValueError:
                    pass

        global_path = tables_dir / "global_pass_rate.csv"
        with global_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "pass_rate"])
            writer.writeheader()
            for mode, data in sorted(mode_stats.items(), key=lambda x: -sum(x[1]["pass_rates"])/len(x[1]["pass_rates"])):
                avg = sum(data["pass_rates"]) / len(data["pass_rates"])
                writer.writerow({"mode": mode, "pass_rate": f"{avg:.2f}"})

        tex = csv_to_latex(
            global_path,
            caption="Global pass rate comparison across all modes.",
            label="tab:global_pass_rate",
            highlight_max="pass_rate",
        )
        (tables_dir / "table4_global_pass_rate.tex").write_text(tex)
        print(f"  Generated {tables_dir / 'table4_global_pass_rate.tex'}")

    # 5. Per-language breakdown for each constrained mode
    summary_csv = out_dir / "summary.csv"
    if summary_csv.exists():
        for mode in ["constrained_direct", "constrained_mixed"]:
            lang_data = {}
            with summary_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["mode"] == mode:
                        lang = row["language"]
                        if lang not in lang_data:
                            lang_data[lang] = {"language": lang, "pass_rates": []}
                        try:
                            lang_data[lang]["pass_rates"].append(float(row["pass_rate"]))
                        except ValueError:
                            pass

            if lang_data:
                lang_path = tables_dir / f"lang_breakdown_{mode}.csv"
                with lang_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["language", "pass_rate"])
                    writer.writeheader()
                    for lang, data in sorted(lang_data.items()):
                        avg = sum(data["pass_rates"]) / len(data["pass_rates"])
                        writer.writerow({"language": lang, "pass_rate": f"{avg:.2f}"})

                mode_label = "Constrained Direct" if "direct" in mode else "Constrained Mixed"
                tex = csv_to_latex(
                    lang_path,
                    caption=f"{mode_label}: Per-language performance breakdown.",
                    label=f"tab:lang_breakdown_{mode}",
                )
                (tables_dir / f"table5_lang_breakdown_{mode.replace('_', '')}.tex").write_text(tex)
                filename = f"table5_lang_breakdown_{mode.replace('_', '')}.tex"
                print(f"  Generated {tables_dir / filename}")

    print(f"\n  All table files saved to {tables_dir}")


if __name__ == "__main__":
    out_dir = Path("evals")
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])

    generate_per_table_latex(out_dir)
