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

MODE_LABELS = {
    "constrained_direct": "Constrained (Direct)",
    "constrained_mixed": "Constrained (Mixed)",
    "unconstrained_raw": "Unconstrained (Raw)",
    "unconstrained": "Unconstrained",
}


def load_data(in_dir: Path) -> pd.DataFrame:
    return pd.read_csv(in_dir / "summary.csv")


def filter_modes(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["mode"].isin(MODES)].copy()


def make_entry_name(row: pd.Series) -> str:
    mode = row["mode"]
    model = row["model"]
    if mode in ("constrained_direct", "constrained_mixed"):
        return f"{model} (constrained)"
    elif mode == "unconstrained_raw":
        return f"{model} (unconstrained raw)"
    else:
        return f"{model} (unconstrained)"


def plot_model_mode_heatmap(df: pd.DataFrame, out_dir: Path, metric: str = "pass_rate"):
    data = filter_modes(df)
    data["entry"] = data.apply(make_entry_name, axis=1)

    constrained = data[data["mode"].isin(["constrained_direct", "constrained_mixed"])]
    unconstrained_raw = data[data["mode"] == "unconstrained_raw"]

    constrained_pivot = constrained.pivot_table(
        values=metric, index="model", columns="mode", aggfunc="mean"
    )
    constrained_pivot = constrained_pivot[["constrained_direct", "constrained_mixed"]]

    unconstrained_raw_pivot = unconstrained_raw.pivot_table(
        values=metric, index="model", columns="mode", aggfunc="mean"
    )

    if not constrained_pivot.empty:
        plt.figure(figsize=(6, max(4, len(constrained_pivot) * 0.5)))
        sns.heatmap(
            constrained_pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            cbar_kws={"label": f"{metric.replace('_', ' ').title()} (%)"},
        )
        plt.title("Constrained: Model × Mode", fontsize=13, fontweight="bold")
        plt.xlabel("")
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig(out_dir / "heatmap_constrained.png", bbox_inches="tight")
        plt.close()
        print(f"  Generated heatmap_constrained.png")

    if not unconstrained_raw_pivot.empty:
        plt.figure(figsize=(4, max(4, len(unconstrained_raw_pivot) * 0.5)))
        sns.heatmap(
            unconstrained_raw_pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            cbar_kws={"label": f"{metric.replace('_', ' ').title()} (%)"},
        )
        plt.title("Unconstrained Raw: Model × Mode", fontsize=13, fontweight="bold")
        plt.xlabel("")
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig(out_dir / "heatmap_unconstrained_raw.png", bbox_inches="tight")
        plt.close()
        print(f"  Generated heatmap_unconstrained_raw.png")


def plot_global_ranking(df: pd.DataFrame, out_dir: Path, metric: str = "pass_rate"):
    data = filter_modes(df)
    data["entry"] = data.apply(make_entry_name, axis=1)

    rank = data.groupby("entry")[metric].mean().reset_index()
    rank = rank.sort_values(metric, ascending=False)

    plt.figure(figsize=(10, max(6, len(rank) * 0.4)))
    bars = plt.barh(range(len(rank)), rank[metric], color="steelblue")
    plt.yticks(range(len(rank)), rank["entry"])
    plt.xlabel(metric.replace("_", " ").title() + " (%)")
    plt.title(
        "Global Leaderboard: All Model+Mode Combinations",
        fontsize=13,
        fontweight="bold",
    )
    plt.gca().invert_yaxis()
    plt.xlim(0, 105)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_dir / "global_ranking.png", bbox_inches="tight")
    plt.close()
    print(f"  Generated global_ranking.png")

    return rank


def plot_mode_ranking(df: pd.DataFrame, out_dir: Path, metric: str = "pass_rate"):
    data = filter_modes(df)

    constrained_modes = ["constrained_direct", "constrained_mixed"]
    unconstrained_raw_modes = ["unconstrained_raw"]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, data["model"].nunique() * 0.4)))

    cdata = data[data["mode"].isin(constrained_modes)]
    crank = cdata.groupby("model")[metric].mean().reset_index()
    crank = crank.sort_values(metric, ascending=True)

    axes[0].barh(range(len(crank)), crank[metric], color="#1f77b4")
    axes[0].set_yticks(range(len(crank)))
    axes[0].set_yticklabels(crank["model"])
    axes[0].set_xlabel(metric.replace("_", " ").title() + " (%)")
    axes[0].set_title("Constrained Modes Avg", fontweight="bold")
    axes[0].set_xlim(0, 105)

    udata = data[data["mode"].isin(unconstrained_raw_modes)]
    urank = udata.groupby("model")[metric].mean().reset_index()
    urank = urank.sort_values(metric, ascending=True)

    axes[1].barh(range(len(urank)), urank[metric], color="#d62728")
    axes[1].set_yticks(range(len(urank)))
    axes[1].set_yticklabels(urank["model"])
    axes[1].set_xlabel(metric.replace("_", " ").title() + " (%)")
    axes[1].set_title("Unconstrained Raw Avg", fontweight="bold")
    axes[1].set_xlim(0, 105)

    plt.suptitle("Model Rankings by Category", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "model_ranking_by_mode.png", bbox_inches="tight")
    plt.close()
    print(f"  Generated model_ranking_by_mode.png")


def plot_mode_comparison(df: pd.DataFrame, out_dir: Path, metric: str = "pass_rate"):
    data = filter_modes(df)
    agg = data.groupby("mode")[metric].mean().reset_index()
    agg = agg.sort_values(metric, ascending=False)

    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        [MODE_LABELS.get(m, m) for m in agg["mode"]], agg[metric], color=colors
    )
    plt.title(
        f"{metric.replace('_', ' ').title()} by Mode", fontsize=13, fontweight="bold"
    )
    plt.ylabel(metric.replace("_", " ").title() + " (%)")
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, max(agg[metric]) * 1.2 + 5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_by_mode.png", bbox_inches="tight")
    plt.close()
    print(f"  Generated {metric}_by_mode.png")


def main():
    p = argparse.ArgumentParser(description="Generate heatmaps and rankings.")
    p.add_argument(
        "--in-dir",
        required=True,
        help="Evaluation data directory (contains summary.csv)",
    )
    p.add_argument(
        "--out-dir",
        help="Output directory for heatmaps (default: <in-dir>/heatmaps)",
    )
    p.add_argument(
        "--metric",
        default="pass_rate",
        choices=["pass_rate", "exact_rate", "ok_rate"],
        help="Metric to visualize",
    )
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = in_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {in_dir}...")
    df = load_data(in_dir)
    print(f"Loaded {len(df)} rows\nGenerating visualizations in {out_dir}...")

    print("\n[Heatmaps]")
    plot_model_mode_heatmap(df, out_dir, args.metric)

    print("\n[Rankings]")
    plot_global_ranking(df, out_dir, args.metric)
    plot_mode_ranking(df, out_dir, args.metric)

    print("\n[Mode Comparison]")
    plot_mode_comparison(df, out_dir, args.metric)

    print(f"\nDone! All outputs in {out_dir}")


if __name__ == "__main__":
    main()
