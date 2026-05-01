#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({"font.size": 14, "figure.dpi": 150})
sns.set_style("whitegrid")

MODES = [
    "constrained_direct",
    "constrained_mixed",
    "unconstrained_raw",
    "unconstrained",
]
MODE_COLORS = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]


def main():
    p = argparse.ArgumentParser(description="Generate pass rate visualization.")
    p.add_argument(
        "--in-dir", default="vast_model_matrix/evals", help="Evaluation data directory"
    )
    p.add_argument(
        "--out-dir", default="vast_model_matrix/evals/figures", help="Output directory"
    )
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_dir / "summary.csv")
    data = df[df["mode"].isin(MODES)]
    data = data[~data["model"].str.contains("gpt2|Baguettotron", na=False)]

    agg = data.groupby("mode")["pass_rate"].mean().reset_index()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(agg["mode"], agg["pass_rate"], color=MODE_COLORS, width=0.6)
    plt.title(
        "Pass Rate by Mode (All Models, All Languages)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.ylabel("Pass Rate (%)", fontsize=16)
    plt.xlabel("")
    plt.xticks(rotation=0, fontsize=14)
    plt.ylim(0, max(agg["pass_rate"]) * 1.2 + 5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pass_rate_by_mode.png", bbox_inches="tight")
    plt.close()

    print(f"Generated {out_dir / 'pass_rate_by_mode.png'}")


if __name__ == "__main__":
    main()
