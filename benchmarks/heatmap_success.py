#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
sns.set_style("whitegrid")

MODES = [
    "constrained_direct",
    "constrained_mixed",
    "unconstrained_raw",
    "unconstrained",
]
MODE_COLORS = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]


def main():
    p = argparse.ArgumentParser(description="Generate success rate heatmap.")
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

    pivot = data.pivot_table(
        values="pass_rate", index="model", columns="mode", aggfunc="mean"
    )
    pivot = pivot[MODES]

    plt.figure(figsize=(8, max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Pass Rate (%)"},
    )
    plt.title("Pass Rate: Model × Mode (all languages)", fontsize=13, fontweight="bold")
    plt.xlabel("Mode", fontsize=11)
    plt.ylabel("Model", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_success.png", bbox_inches="tight")
    plt.close()

    print(f"Generated {out_dir / 'heatmap_success.png'}")


if __name__ == "__main__":
    main()
