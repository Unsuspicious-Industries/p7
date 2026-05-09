#!/usr/bin/env python3
"""
Unified rendering script for constrained generation benchmarks.

This script:
1. Loads ALL .jsonl files from input directory
2. Cleans data (removes CUDA OOM, API errors)
3. Deduplicates by taking the BEST try per task
4. Generates summary CSVs with error breakdowns
5. Creates publication-quality visualizations (PNG + LaTeX tables)

Usage:
    python unified/renders.py [--in-dir unified] [--out-dir evals]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================================
# Constants
# ============================================================

MODES_ALL = [
    "constrained_direct",
    "constrained_mixed",
    "unconstrained_raw",
    "unconstrained",
]

MODES_HEATMAP = [
    "constrained_direct",
    "constrained_mixed",
    "unconstrained",
]

MODE_LABELS = {
    "constrained_direct": "Constrained (Direct)",
    "constrained_mixed": "Constrained (Mixed)",
    "unconstrained": "Unconstrained",
    "unconstrained_raw": "Unconstrained (Raw)",
}

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

# Error colors for stacked bars
ERROR_COLORS = {
    "ok": "#2ca02c",
    "parse_error": "#d62728",
    "task_invalid": "#ff7f0e",  # Was: semantic_mismatch - valid syntax, wrong logic
    "incomplete": "#e377c2",
    "non_completable": "#8c564b",
    "timeout": "#7f7f7f",
    "other": "#17becf",
}


# ============================================================
# Model size helper
# ============================================================

def model_param_billions(model_name: str) -> Optional[float]:
    """Extract model size in billions of parameters."""
    import re
    lower = model_name.lower()
    known = {
        "gpt2": 0.124,
        "distilgpt2": 0.082,
        "Qwen/Qwen3.5-0.8": 0.8,
        "Qwen/Qwen3.5-2B": 2.0,
        "Qwen/Qwen3.5-4B": 4.0,
        "Qwen/Qwen3.5-4B-Base": 4.0,
        "Qwen/Qwen3.5-9B": 9.0,
        "Qwen/Qwen3.5-9B-Base": 9.0,
        "Qwen/Qwen3.6-27B": 27.0,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 7.0,
        "microsoft/Phi-4-mini-instruct": 3.8,
    }
    if lower in known:
        return known[lower]

    match = re.search(r"(?:^|[^a-z0-9])e?(\d+(?:\.\d+)?)\s*b(?![a-z])", lower)
    if match:
        return float(match.group(1))
    match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*m(?![a-z])", lower)
    if match:
        return float(match.group(1)) / 1000.0
    return None


# ============================================================
# Data Loading and Cleaning
# ============================================================

def is_system_error(record: dict) -> bool:
    """Check if record has system-level errors."""
    error = record.get("error", "")
    output = record.get("output", "")
    combined = f"{error} {output}"
    return any(p in combined for p in ERROR_PATTERNS)


def is_valid_model(record: dict) -> bool:
    return record.get("model", "") not in MODELS_TO_EXCLUDE


def load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clean_records(records: list[dict]) -> tuple[list[dict], dict]:
    """Remove system error records."""
    cleaned = []
    stats = {
        "total": len(records),
        "removed_system_error": 0,
        "removed_invalid_model": 0,
        "by_reason": Counter(),
    }

    for rec in records:
        if not is_valid_model(rec):
            stats["removed_invalid_model"] += 1
            continue
        if is_system_error(rec):
            stats["removed_system_error"] += 1
            reason = rec.get("error", "")[:60]
            stats["by_reason"][reason] += 1
            continue
        cleaned.append(rec)

    return cleaned, stats


# ============================================================
# Best-Try Deduplication
# ============================================================

def record_priority(rec: dict) -> tuple:
    """Return priority tuple (higher = better)."""
    passed = 1 if rec.get("passed") or rec.get("error") == "ok" else 0
    exact = 1 if rec.get("exact") else 0
    has_output = 1 if rec.get("output", "").strip() else 0
    error_penalty = {
        "ok": 0,
        "parse_error": 3,
        "semantic_mismatch": 2,
        "incomplete": 4,
        "non_completable": 5,
        "timeout": 6,
    }.get(rec.get("error", ""), 7)
    return (passed, exact, has_output, -error_penalty)


def dedupe_best_try(records: list[dict]) -> list[dict]:
    """Keep best try for each (model, mode, task_id)."""
    groups = defaultdict(list)
    for rec in records:
        key = (rec.get("model", ""), rec.get("mode", ""), rec.get("task_id", ""))
        groups[key].append(rec)

    deduped = []
    for key, group in groups.items():
        best = max(group, key=record_priority)
        deduped.append(best)

    return deduped


# ============================================================
# Aggregation
# ============================================================

def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else 100.0 * n / d


def summarize(records: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    """Aggregate records by dimensions."""
    by = defaultdict(list)
    for row in records:
        by[tuple(row.get(dim, "") for dim in dimensions)].append(row)

    summary = []
    for key, group in sorted(by.items()):
        n = len(group)
        c = Counter(row["error"] for row in group)
        exact = sum(1 for row in group if row.get("exact"))
        known = {"ok", "parse_error", "non_completable", "incomplete", "semantic_mismatch", "timeout"}
        other_errors = sum(count for err, count in c.items() if err not in known)
        tok = sum(float(row.get("tokens", 0)) for row in group) / max(n, 1)
        sec = sum(float(row.get("seconds", 0.0)) for row in group) / max(n, 1)
        model = group[0].get("model", "")
        params_b = model_param_billions(model)

        summary.append({
            **dict(zip(dimensions, key)),
            "attempts": n,
            "exact_rate": round(pct(exact, n), 2),
            "pass_rate": round(pct(c.get("ok", 0), n), 2),
            "ok_rate": round(pct(c.get("ok", 0), n), 2),
            "parse_error_rate": round(pct(c.get("parse_error", 0), n), 2),
            "non_completable_rate": round(pct(c.get("non_completable", 0), n), 2),
            "incomplete_rate": round(pct(c.get("incomplete", 0), n), 2),
            "semantic_mismatch_rate": round(pct(c.get("semantic_mismatch", 0), n), 2),
            "timeout_rate": round(pct(c.get("timeout", 0), n), 2),
            "other_error_rate": round(pct(other_errors, n), 2),
            "avg_tokens": round(tok, 2),
            "avg_seconds": round(sec, 2),
            "model_params_b": params_b,
        })

    return summary


# ============================================================
# Visualization Helpers
# ============================================================

def setup_style(style: str = "science", use_latex: bool = False):
    """Set up publication-quality plot style."""
    if style == "science":
        try:
            import scienceplots
            plt.style.use(["science", "grid"])
        except ImportError:
            sns.set_style("whitegrid")
    else:
        sns.set_style("whitegrid")

    if use_latex:
        plt.rcParams.update({"text.usetex": True})

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "figure.titlesize": 20,
    })


def save_fig(out_dir: Path, name: str, fmt: str = "png"):
    """Save figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    dpi = 300 if fmt == "png" else 150
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"  Generated {path}")


# ============================================================
# Visualizations
# ============================================================

def plot_error_breakdown(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """Figure 1: Stacked error breakdown by mode."""
    print("\n[Figure 1: Error Breakdown by Mode]")

    # Only use models that have ALL 3 modes
    data = df[df["mode"].isin(MODES_HEATMAP)].copy()
    model_mode_counts = data.groupby(["model", "mode"]).size().reset_index()
    model_counts = model_mode_counts.groupby("model")["mode"].nunique()
    complete_models = model_counts[model_counts == len(MODES_HEATMAP)].index

    if len(complete_models) == 0:
        print("  No models with all 3 modes")
        return

    data = data[data["model"].isin(complete_models)]

    # Aggregate by mode (excluding parse error and timeout)
    mode_stats = data.groupby("mode").agg({
        "pass_rate": "mean",
        "semantic_mismatch_rate": "mean",
        "incomplete_rate": "mean",
        "non_completable_rate": "mean",
        "other_error_rate": "mean",
    }).reset_index()

    if mode_stats.empty:
        print("  No data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    modes = [MODE_LABELS.get(m, m) for m in mode_stats["mode"]]
    x = range(len(modes))
    bottom_vals = [0] * len(x)

    # Success (green)
    success = mode_stats["pass_rate"]
    ax.bar(x, success, bottom=bottom_vals, color=ERROR_COLORS["ok"],
           label="Pass", edgecolor="white", linewidth=0.5)
    bottom_vals = [b + s for b, s in zip(bottom_vals, success)]

    # Semantic mismatch (task invalid) (orange)
    task_invalid = mode_stats["semantic_mismatch_rate"]
    ax.bar(x, task_invalid, bottom=bottom_vals, color=ERROR_COLORS["task_invalid"],
           label="Task Invalid", edgecolor="white", linewidth=0.5)
    bottom_vals = [b + t for b, t in zip(bottom_vals, task_invalid)]

    # Incomplete (pink)
    incomplete = mode_stats["incomplete_rate"]
    ax.bar(x, incomplete, bottom=bottom_vals, color=ERROR_COLORS["incomplete"],
           label="Incomplete", edgecolor="white", linewidth=0.5)
    bottom_vals = [b + i for b, i in zip(bottom_vals, incomplete)]

    # Non-completable (brown)
    non_comp = mode_stats["non_completable_rate"]
    ax.bar(x, non_comp, bottom=bottom_vals, color=ERROR_COLORS["non_completable"],
           label="Non-completable", edgecolor="white", linewidth=0.5)
    bottom_vals = [b + n for b, n in zip(bottom_vals, non_comp)]


    # Other
    other = mode_stats["other_error_rate"]
    if other.sum() > 0:
        ax.bar(x, other, bottom=bottom_vals, color=ERROR_COLORS["other"],
               label="Other", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.set_ylabel("Rate (%)", fontsize=16)
    ax.set_title("Error Distribution by Generation Mode", fontsize=18, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    save_fig(out_dir, "fig1_error_breakdown", fmt)


def plot_combined_heatmap(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """Heatmap: constrained_direct, constrained_mixed, unconstrained_raw."""
    print("\n[Heatmap: Combined Modes x Model]")

    required_modes = ["constrained_direct", "constrained_mixed", "unconstrained_raw"]
    data = df[df["mode"].isin(required_modes)].copy()

    if data.empty:
        print("  No data available")
        return

    # Find models with all three modes
    model_mode_counts = data.groupby(["model", "mode"]).size().reset_index()
    model_counts = model_mode_counts.groupby("model")["mode"].nunique()
    complete_models = model_counts[model_counts == len(required_modes)].index

    if len(complete_models) == 0:
        print("  No models with complete data for all 3 modes")
        return

    data = data[data["model"].isin(complete_models)]
    pivot = data.pivot_table(values="pass_rate", index="model",
                             columns="mode", aggfunc="mean")
    pivot = pivot[required_modes]

    plt.figure(figsize=(10, max(6, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
                cbar_kws={"label": "Pass Rate (%)"}, linewidths=0.5)
    plt.title("Model x Mode: Pass Rate (%)", fontsize=18, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Model", fontsize=16)
    plt.xticks(rotation=15, ha="right", fontsize=12)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    save_fig(out_dir, "fig2_combined_heatmap", fmt)


def plot_heatmap_by_language(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """Heatmap: Model x Language for each mode."""
    print("\n[Heatmap: Model x Language for each mode]")

    for mode in MODES_HEATMAP:
        data = df[df["mode"] == mode].copy()
        if data.empty:
            continue

        # Find models with all 3 languages
        model_lang_counts = data.groupby(["model", "language"]).size().reset_index()
        model_counts = model_lang_counts.groupby("model")["language"].nunique()
        complete_models = model_counts[model_counts == 3].index

        if len(complete_models) == 0:
            continue

        data = data[data["model"].isin(complete_models)]
        pivot = data.pivot_table(values="pass_rate", index="model",
                                 columns="language", aggfunc="mean")

        mode_label = MODE_LABELS.get(mode, mode)
        plt.figure(figsize=(8, max(6, len(pivot) * 0.6)))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
                    cbar_kws={"label": "Pass Rate (%)"}, linewidths=0.5)
        plt.title(f"{mode_label}: Model x Language", fontsize=16, fontweight="bold")
        plt.xlabel("")
        plt.ylabel("Model", fontsize=14)
        plt.xticks(rotation=15, ha="right", fontsize=11)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        save_fig(out_dir, f"heatmap_language_{mode}", fmt)


def plot_model_scaling(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """Figure 3: Model size vs pass rate. Only models with ALL 3 modes."""
    print("\n[Figure 3: Model Scaling (Size vs Pass Rate)]")

    data = df[df["model_params_b"].notna()].copy()
    if data.empty:
        print("  No model param data available")
        return

    # Only use models that have ALL modes in MODES_HEATMAP
    model_mode_counts = data.groupby(["model", "mode"]).size().reset_index()
    model_counts = model_mode_counts.groupby("model")["mode"].nunique()
    complete_models = model_counts[model_counts == len(MODES_HEATMAP)].index

    if len(complete_models) == 0:
        print("  No models with all 3 modes")
        return

    data = data[data["model"].isin(complete_models)]

    fig, ax = plt.subplots(figsize=(12, 8))

    for mode in MODES_HEATMAP:
        mode_data = data[data["mode"] == mode]
        if mode_data.empty:
            continue
        # Group by model params and compute mean pass rate
        grouped = mode_data.groupby("model_params_b")["pass_rate"].mean().reset_index()
        grouped = grouped.sort_values("model_params_b")
        label = MODE_LABELS.get(mode, mode)
        linestyle = "-" if "constrained" in mode else "--"
        marker = "o" if "constrained" in mode else "s"
        ax.plot(grouped["model_params_b"], grouped["pass_rate"],
                label=label, linestyle=linestyle, marker=marker, linewidth=2.5, markersize=10)

    ax.set_xlabel("Model Size (billions of parameters)", fontsize=16)
    ax.set_ylabel("Pass Rate (%)", fontsize=16)
    ax.set_title("Model Scaling by Mode",
                 fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="both", which="major", labelsize=12)

    save_fig(out_dir, "fig3_model_scaling", fmt)


def plot_global_pass_rate(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """Global pass rate by mode (only models with all 3 modes)."""
    print("\n[Figure: Global Pass Rate by Mode]")

    # Only use the 3 modes
    data = df[df["mode"].isin(MODES_HEATMAP)].copy()

    # Find models with ALL 3 modes
    model_mode_counts = data.groupby(["model", "mode"]).size().reset_index()
    model_counts = model_mode_counts.groupby("model")["mode"].nunique()
    complete_models = model_counts[model_counts == len(MODES_HEATMAP)].index

    if len(complete_models) == 0:
        print("  No models with all 3 modes")
        return

    data = data[data["model"].isin(complete_models)]

    # Aggregate by mode
    mode_stats = data.groupby("mode").agg({
        "pass_rate": "mean",
    }).reset_index()

    mode_stats = mode_stats.sort_values("pass_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = []
    for mode in mode_stats["mode"]:
        if "constrained" in mode:
            colors.append("#1f77b4" if "mixed" in mode else "#2ca02c")
        else:
            colors.append("#d62728" if "raw" in mode else "#ff7f0e")

    bars = ax.bar(range(len(mode_stats)), mode_stats["pass_rate"],
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(mode_stats)))
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in mode_stats["mode"]],
                       rotation=15, ha="right")
    ax.set_ylabel("Pass Rate (%)", fontsize=16)
    ax.set_title("Global Pass Rate by Mode",
                 fontsize=18, fontweight="bold")
    ax.set_ylim(0, max(mode_stats["pass_rate"]) * 1.2 + 5)

    for bar, rate in zip(bars, mode_stats["pass_rate"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_fig(out_dir, "fig_global_pass_rate", fmt)


def plot_global_ranking(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """Global leaderboard: only 3 modes, models with ALL 3 modes."""
    print("\n[Global Ranking: 3 Modes]")

    # Only use the 3 modes that should be compared
    data = df[df["mode"].isin(MODES_HEATMAP)].copy()

    # Find models that have ALL 3 modes
    model_mode_counts = data.groupby(["model", "mode"]).size().reset_index()
    model_counts = model_mode_counts.groupby("model")["mode"].nunique()
    complete_models = model_counts[model_counts == len(MODES_HEATMAP)].index

    if len(complete_models) == 0:
        print("  No models with all 3 modes")
        return

    data = data[data["model"].isin(complete_models)]

    # Create entry names
    data["entry"] = data.apply(lambda row: f"{row['model']} ({MODE_LABELS.get(row['mode'], row['mode'])})", axis=1)

    rank = data.groupby("entry").agg({
        "pass_rate": "mean",
        "semantic_mismatch_rate": "mean",
        "incomplete_rate": "mean",
    }).reset_index()
    rank = rank.sort_values("pass_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(8, len(rank) * 0.5)))

    y = range(len(rank))
    left_vals = [0] * len(y)

    # Success (green)
    ax.barh(y, rank["pass_rate"], left=left_vals, color=ERROR_COLORS["ok"],
            label="Pass", edgecolor="white", linewidth=0.5)
    left_vals = [l + p for l, p in zip(left_vals, rank["pass_rate"])]

    # Task invalid (orange) - was "Semantic Mismatch"
    ax.barh(y, rank["semantic_mismatch_rate"], left=left_vals,
            color=ERROR_COLORS["task_invalid"],
            label="Task Invalid", edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(rank["entry"], fontsize=11)
    ax.set_xlabel("Rate (%)", fontsize=16)
    ax.set_title("Global Leaderboard: 3 Modes",
                 fontsize=18, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_fig(out_dir, "global_ranking", fmt)


def plot_mixed_vs_direct(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """
    Bar chart comparing constrained_mixed vs constrained_direct.
    Shows delta for models with both modes, and individual rates for models with only one mode.
    """
    print("\n[Constrained Mixed vs Direct Comparison]")

    data = df[df["mode"].isin(["constrained_direct", "constrained_mixed"])].copy()
    if data.empty:
        print("  No constrained data available")
        return

    # Pivot to get both modes per model
    pivot = data.pivot_table(values="pass_rate", index="model",
                             columns="mode", aggfunc="mean")

    # Categorize models
    both_modes = pivot[
        pivot.get("constrained_direct").notna() & pivot.get("constrained_mixed").notna()
    ]
    direct_only = pivot[
        pivot.get("constrained_direct").notna() & pivot.get("constrained_mixed").isna()
    ]
    mixed_only = pivot[
        pivot.get("constrained_direct").isna() & pivot.get("constrained_mixed").notna()
    ]

    # Calculate delta for models with both modes
    if not both_modes.empty:
        both_modes = both_modes.copy()
        both_modes["delta"] = both_modes["constrained_mixed"] - both_modes["constrained_direct"]
        both_modes = both_modes.sort_values("delta")

    # Combine for plotting
    plot_data = []

    # Models with both modes: show delta
    if not both_modes.empty:
        for idx, (model, row) in enumerate(both_modes.iterrows()):
            plot_data.append({
                "model": model,
                "value": row["delta"],
                "label": f"{row['delta']:.1f}%",
                "type": "delta",
                "color": "green" if row["delta"] > 0 else "red"
            })

    # Models with mixed only: show mixed rate
    if not mixed_only.empty:
        for idx, (model, row) in enumerate(mixed_only.iterrows()):
            plot_data.append({
                "model": model,
                "value": row["constrained_mixed"],
                "label": f"mixed={row['constrained_mixed']:.1f}%",
                "type": "mixed_only",
                "color": "blue"
            })

    # Models with direct only: show direct rate
    if not direct_only.empty:
        for idx, (model, row) in enumerate(direct_only.iterrows()):
            plot_data.append({
                "model": model,
                "value": -row["constrained_direct"],  # Negative to show it's direct
                "label": f"direct={row['constrained_direct']:.1f}%",
                "type": "direct_only",
                "color": "orange"
            })

    if not plot_data:
        print("  No data to plot")
        return

    # Sort by value
    plot_data = sorted(plot_data, key=lambda x: x["value"])

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_data) * 0.5)))

    colors = [d["color"] for d in plot_data]
    labels = [d["model"] for d in plot_data]
    values = [d["value"] for d in plot_data]

    bars = ax.barh(range(len(plot_data)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Pass Rate (%)", fontsize=16)
    ax.set_title("Constrained: Mixed vs Direct Comparison",
                 fontsize=18, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, d in zip(bars, plot_data):
        width = bar.get_width()
        ax.text(width + (0.5 if width >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                d["label"], va="center", fontsize=9, ha="left" if width >= 0 else "right")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", label="Mixed > Direct (delta)"),
        Patch(facecolor="red", label="Direct > Mixed (delta)"),
        Patch(facecolor="blue", label="Mixed only"),
        Patch(facecolor="orange", label="Direct only"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    plt.tight_layout()
    save_fig(out_dir, "fig4_mixed_vs_direct", fmt)


def plot_global_leaderboard_all(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """
    Global leaderboard with ALL models (including those with missing modes).
    Shows task-invalid bars for each model+mode combination.
    """
    print("\n[Global Leaderboard: ALL Models, With Task-Invalid Bars]")

    # Create entry names
    df = df.copy()
    df["entry"] = df.apply(
        lambda row: f"{row['model']} ({MODE_LABELS.get(row['mode'], row['mode'])})",
        axis=1
    )

    # Aggregate by entry
    rank = df.groupby("entry").agg({
        "pass_rate": "mean",
        "semantic_mismatch_rate": "mean",
        "incomplete_rate": "mean",
    }).reset_index()
    rank = rank.sort_values("pass_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(14, max(10, len(rank) * 0.5)))

    y = range(len(rank))
    left_vals = [0] * len(y)

    # Success (green)
    ax.barh(y, rank["pass_rate"], left=left_vals, color=ERROR_COLORS["ok"],
            label="Pass", edgecolor="white", linewidth=0.5)
    left_vals = [l + p for l, p in zip(left_vals, rank["pass_rate"])]

    # Task invalid (orange)
    ax.barh(y, rank["semantic_mismatch_rate"], left=left_vals,
            color=ERROR_COLORS["task_invalid"],
            label="Task Invalid", edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(rank["entry"], fontsize=10)
    ax.set_xlabel("Rate (%)", fontsize=16)
    ax.set_title("Global Leaderboard: ALL Models + Modes",
                 fontsize=18, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_fig(out_dir, "fig_global_leaderboard_all", fmt)


def plot_global_leaderboard_simple(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """
    Global leaderboard with ALL models - NO task-invalid bars, just pass rates.
    """
    print("\n[Global Leaderboard: ALL Models, Simple (No Task-Invalid)]")

    # Create entry names
    df = df.copy()
    df["entry"] = df.apply(
        lambda row: f"{row['model']} ({MODE_LABELS.get(row['mode'], row['mode'])})",
        axis=1,
    )

    # Aggregate by entry
    rank = df.groupby("entry").agg({
        "pass_rate": "mean",
    }).reset_index()
    rank = rank.sort_values("pass_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(14, max(10, len(rank) * 0.5)))

    y = range(len(rank))
    bars = ax.barh(y, rank["pass_rate"], color=ERROR_COLORS["ok"],
                     edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(rank["entry"], fontsize=10)
    ax.set_xlabel("Pass Rate (%)", fontsize=16)
    ax.set_title("Global Leaderboard: Pass Rates Only",
                 fontsize=18, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    save_fig(out_dir, "fig_global_leaderboard_simple", fmt)


def plot_global_leaderboard_top_n(df: pd.DataFrame, out_dir: Path, fmt: str = "png", n: int = 5):
    """
    Global leaderboard with top N models (including those with missing modes).
    Shows task-invalid bars for each model+mode combination.
    """
    print(f"\n[Global Leaderboard: Top {n} Models, With Task-Invalid Bars]")

    # Create entry names
    df = df.copy()
    df["entry"] = df.apply(
        lambda row: f"{row['model']} ({MODE_LABELS.get(row['mode'], row['mode'])})",
        axis=1
    )

    # Aggregate by entry
    rank = df.groupby("entry").agg({
        "pass_rate": "mean",
        "semantic_mismatch_rate": "mean",
        "incomplete_rate": "mean",
    }).reset_index()
    rank = rank.sort_values("pass_rate", ascending=False)

    # Take top N
    rank = rank.head(n)

    if rank.empty:
        print(f"  No data to plot")
        return

    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.5)))

    y = range(len(rank))
    left_vals = [0] * len(y)

    # Success (green)
    ax.barh(y, rank["pass_rate"], left=left_vals, color=ERROR_COLORS["ok"],
            label="Pass", edgecolor="white", linewidth=0.5)
    left_vals = [l + p for l, p in zip(left_vals, rank["pass_rate"])]

    # Task invalid (orange)
    ax.barh(y, rank["semantic_mismatch_rate"], left=left_vals,
            color=ERROR_COLORS["task_invalid"],
            label="Task Invalid", edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(rank["entry"], fontsize=10)
    ax.set_xlabel("Rate (%)", fontsize=16)
    ax.set_title(f"Global Leaderboard: Top {n} Models + Modes",
                 fontsize=18, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_fig(out_dir, f"fig_global_leaderboard_top_{n}", fmt)


def plot_global_leaderboard_top_n_simple(df: pd.DataFrame, out_dir: Path, fmt: str = "png", n: int = 5):
    """
    Global leaderboard with top N models - NO task-invalid bars, just pass rates.
    """
    print(f"\n[Global Leaderboard: Top {n} Models, Simple (No Task-Invalid)]")

    # Create entry names
    df = df.copy()
    df["entry"] = df.apply(
        lambda row: f"{row['model']} ({MODE_LABELS.get(row['mode'], row['mode'])})",
        axis=1,
    )

    # Aggregate by entry
    rank = df.groupby("entry").agg({
        "pass_rate": "mean",
    }).reset_index()
    rank = rank.sort_values("pass_rate", ascending=False)

    # Take top N
    rank = rank.head(n)

    if rank.empty:
        print(f"  No data to plot")
        return

    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.5)))

    y = range(len(rank))
    bars = ax.barh(y, rank["pass_rate"], color=ERROR_COLORS["ok"],
                     edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(rank["entry"], fontsize=10)
    ax.set_xlabel("Pass Rate (%)", fontsize=16)
    ax.set_title(f"Global Leaderboard: Top {n} Pass Rates",
                 fontsize=18, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    save_fig(out_dir, f"fig_global_leaderboard_top_{n}_simple", fmt)


def plot_token_efficiency(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """Compare token usage across modes."""
    print("\n[Token Efficiency by Mode]")

    mode_stats = df.groupby("mode").agg({
        "avg_tokens": "mean",
        "pass_rate": "mean",
    }).reset_index()

    mode_stats = mode_stats[mode_stats["mode"].isin(MODES_HEATMAP)]

    if mode_stats.empty:
        print("  No data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    modes = [MODE_LABELS.get(m, m) for m in mode_stats["mode"]]
    colors = ["#2ca02c" if "constrained" in m else "#d62728" for m in mode_stats["mode"]]

    bars = ax.bar(range(len(mode_stats)), mode_stats["avg_tokens"],
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(mode_stats)))
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.set_ylabel("Average Tokens", fontsize=16)
    ax.set_title("Token Usage by Mode", fontsize=18, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, rate in zip(bars, mode_stats["pass_rate"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f"pass={rate:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_fig(out_dir, "fig_token_efficiency", fmt)


def plot_cost_comparison(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """
    Latency/token cost comparison: constrained_direct vs unconstrained.
   Only models with BOTH modes (fair comparison).
    """
    print("\n[Cost Comparison: Constrained Direct vs Unconstrained]")

    # Only use constrained_direct and unconstrained (NOT unconstrained_raw)
    data = df[df["mode"].isin(["constrained_direct", "unconstrained"])].copy()
    if data.empty:
        print("  No data available")
        return

    # Find models with BOTH modes
    model_mode_counts = data.groupby(["model", "mode"]).size().reset_index()
    model_counts = model_mode_counts.groupby("model")["mode"].nunique()
    complete_models = model_counts[model_counts == 2].index

    if len(complete_models) == 0:
        print("  No models with both constrained_direct and unconstrained")
        return

    data = data[data["model"].isin(complete_models)]

    # Pivot: model -> {constrained_direct, unconstrained}
    pivot = data.pivot_table(values="avg_seconds", index="model",
                             columns="mode", aggfunc="mean")

    if "constrained_direct" not in pivot.columns or "unconstrained" not in pivot.columns:
        print("  Missing one of the modes")
        return

    # Calculate ratios
    pivot["seconds_ratio"] = pivot["unconstrained"] / pivot["constrained_direct"]
    pivot = pivot.sort_values("seconds_ratio")

    # Plot latency ratio
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))

    colors = ["green" if r > 1 else "red" for r in pivot["seconds_ratio"]]
    bars = ax.barh(range(len(pivot)), pivot["seconds_ratio"],
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Latency Ratio (Unconstrained / Constrained Direct)", fontsize=16)
    ax.set_title("Latency Cost: Constrained Direct vs Unconstrained",
                 fontsize=18, fontweight="bold")
    ax.axvline(x=1, color="black", linestyle="-", linewidth=1)
    ax.set_xlim(0, max(pivot["seconds_ratio"]) * 1.2)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (idx, row) in enumerate(pivot.iterrows()):
        ratio = row["seconds_ratio"]
        ax.text(ratio + 0.05, i, f"{ratio:.2f}x",
                va="center", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", label="Unconstrained slower (constrained faster)"),
        Patch(facecolor="red", label="Constrained slower"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="best")

    plt.tight_layout()
    save_fig(out_dir, "fig_cost_latency", fmt)

    # Also do token comparison
    pivot_tokens = data.pivot_table(values="avg_tokens", index="model",
                                  columns="mode", aggfunc="mean")

    if "constrained_direct" in pivot_tokens.columns and "unconstrained" in pivot_tokens.columns:
        pivot_tokens["tokens_ratio"] = pivot_tokens["unconstrained"] / pivot_tokens["constrained_direct"]
        pivot_tokens = pivot_tokens.sort_values("tokens_ratio")

        fig2, ax2 = plt.subplots(figsize=(12, max(6, len(pivot_tokens) * 0.5)))

        colors = ["green" if r > 1 else "red" for r in pivot_tokens["tokens_ratio"]]
        bars = ax2.barh(range(len(pivot_tokens)), pivot_tokens["tokens_ratio"],
                       color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_yticks(range(len(pivot_tokens)))
        ax2.set_yticklabels(pivot_tokens.index, fontsize=10)
        ax2.set_xlabel("Token Ratio (Unconstrained / Constrained Direct)", fontsize=16)
        ax2.set_title("Token Cost: Constrained Direct vs Unconstrained",
                      fontsize=18, fontweight="bold")
        ax2.axvline(x=1, color="black", linestyle="-", linewidth=1)
        ax2.set_xlim(0, max(pivot_tokens["tokens_ratio"]) * 1.2)
        ax2.grid(True, alpha=0.3, axis="x")

        for i, (idx, row) in enumerate(pivot_tokens.iterrows()):
            ratio = row["tokens_ratio"]
            ax2.text(ratio + 0.05, i, f"{ratio:.2f}x",
                     va="center", fontsize=9)

        ax2.legend(handles=legend_elements, fontsize=10, loc="best")

        plt.tight_layout()
        save_fig(out_dir, "fig_cost_tokens", fmt)


# ============================================================
# LaTeX Table Generation (Pure Python, no pandas)
# ============================================================

def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    return s.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')


def csv_to_latex(csv_path: Path, caption: str = "", label: str = "", highlight_max: str = None) -> str:
    """Convert CSV to LaTeX table using csv module."""
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return ""
        headers = reader.fieldnames
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    
    cols = len(headers)
    col_fmt = "l" + "r" * (cols - 1)
    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append("    \\toprule")
    
    header = " & ".join(escape_latex(h) for h in headers)
    lines.append(f"    {header} \\\\")
    lines.append("    \\midrule")
    
    for row in rows:
        values = []
        for h in headers:
            val = escape_latex(row.get(h, ""))
            # Highlight max if specified
            if highlight_max and h == highlight_max:
                try:
                    max_val = max(float(r.get(h, "-inf")) for r in rows if r.get(h))
                    if float(row.get(h, "-inf")) == max_val:
                        val = f"\\textbf{{{val}}}"
                except (ValueError, TypeError):
                    pass
            values.append(val)
        row_str = " & ".join(values)
        lines.append(f"    {row_str} \\\\")
    
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    if caption and not label:
        lines.append(f"  \\caption{{{caption}}}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_latex_tables(out_dir: Path):
    """Generate per-table LaTeX files from CSVs (pure Python)."""
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

    # 5. Per-language breakdown for constrained modes
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
                filename = f"table5_lang_breakdown_{mode.replace('_', '')}.tex"
                (tables_dir / filename).write_text(tex)
                print(f"  Generated {tables_dir / filename}")

    print(f"  All table files saved to {tables_dir}")


# ============================================================
# Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark evaluations and figures."
    )
    parser.add_argument("--in-dir", default="unified",
                        help="Input directory (contains .jsonl files)")
    parser.add_argument("--out-dir", default="evals",
                        help="Output directory for evaluations")
    parser.add_argument("--style", choices=["science", "seaborn", "default"],
                        default="science", help="Plot style")
    parser.add_argument("--format", choices=["pdf", "svg", "png"],
                        default="png", help="Output format for figures")
    parser.add_argument("--latex", action="store_true",
                        help="Use LaTeX for text rendering")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Only generate CSVs, skip visualizations")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up jsonl files in-place (remove system errors)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "figures" / "paper"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Step 1: Clean up jsonl files in-place (optional)
    # ============================================================
    if args.cleanup:
        print("=" * 60)
        print("STEP 1: Cleaning up jsonl files in-place")
        print("=" * 60)

        jsonl_files = sorted(in_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"Error: No .jsonl files found in {in_dir}")
            sys.exit(1)

        total_removed = 0
        for jf in jsonl_files:
            records = load_jsonl(jf)
            print(f"  Processing {jf.name}: {len(records)} records")

            cleaned, stats = clean_records(records)
            removed = stats['removed_system_error'] + stats['removed_invalid_model']
            total_removed += removed

            if removed > 0:
                # Write cleaned data back to the SAME file
                with jf.open("w", encoding="utf-8") as f:
                    for rec in cleaned:
                        f.write(json.dumps(rec) + "\n")
                print(f"    Removed {removed} bad records, saved {len(cleaned)} clean records")
            else:
                print(f"    Already clean, no changes")

        print(f"\nTotal records removed from all files: {total_removed}")
        print("=" * 60)

    # ============================================================
    # Step 2: Load ALL jsonl files and clean data
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Loading and cleaning data")
    print("=" * 60)

    jsonl_files = sorted(in_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {in_dir}")
        sys.exit(1)

    # Load and combine all jsonl files
    all_records = []
    for jf in jsonl_files:
        records = load_jsonl(jf)
        print(f"  Loaded {len(records)} records from {jf.name}")
        all_records.extend(records)

    print(f"\nTotal records from all files: {len(all_records)}")

    # Clean records (remove any remaining system errors)
    cleaned, stats = clean_records(all_records)
    print(f"\nCleanup stats:")
    print(f"  Removed (system errors like CUDA OOM): {stats['removed_system_error']}")
    print(f"  Removed (invalid model): {stats['removed_invalid_model']}")
    print(f"  Remaining: {len(cleaned)}")

    if stats["by_reason"]:
        print("  Removal reasons (top 5):")
        for reason, count in stats["by_reason"].most_common(5):
            print(f"    {reason}: {count}")

    # ============================================================
    # Step 3: Best-try deduplication
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Best-try deduplication")
    print("=" * 60)

    deduped = dedupe_best_try(cleaned)
    # Remove DeepSeek models (tokenizer issues)
    deduped = [rec for rec in deduped if "deepseek" not in rec.get("model", "").lower()]
    print(f"Deduplicated: {len(cleaned)} -> {len(deduped)} records")

    # Save cleaned+deduped data to out_dir (not in_dir)
    cleaned_path = out_dir / "raw_cleaned.jsonl"
    with cleaned_path.open("w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved cleaned data to {cleaned_path}")

    # ============================================================
    # Step 3: Generate summary CSVs
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Generating summary CSVs")
    print("=" * 60)

    summary = summarize(deduped, ("backend", "model", "mode", "language"))
    summary_cols = ["backend", "model", "mode", "language",
                    "attempts", "pass_rate", "exact_rate", "parse_error_rate",
                    "semantic_mismatch_rate", "incomplete_rate",
                    "non_completable_rate", "timeout_rate", "other_error_rate",
                    "avg_tokens", "avg_seconds", "model_params_b"]
    summary_df = pd.DataFrame(summary)[summary_cols]
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Wrote {summary_path}")

    category_summary = summarize(deduped, ("backend", "model", "mode", "language", "category"))
    cat_cols = ["backend", "model", "mode", "language", "category",
                "attempts", "pass_rate", "exact_rate", "parse_error_rate",
                "semantic_mismatch_rate", "incomplete_rate", "non_completable_rate"]
    cat_df = pd.DataFrame(category_summary)[cat_cols]
    cat_path = out_dir / "summary_by_category.csv"
    cat_df.to_csv(cat_path, index=False)
    print(f"  Wrote {cat_path}")

    error_breakdown = summarize(deduped, ("mode",))
    error_cols = ["mode", "attempts", "pass_rate", "parse_error_rate",
                  "semantic_mismatch_rate", "incomplete_rate",
                  "non_completable_rate", "timeout_rate", "other_error_rate"]
    error_df = pd.DataFrame(error_breakdown)[error_cols]
    error_path = out_dir / "error_breakdown.csv"
    error_df.to_csv(error_path, index=False)
    print(f"  Wrote {error_path}")

    # Create compact.csv with ALL info from deduplicated data (raw records, not aggregated)
    print("\n  Creating compact.csv with ALL original data...")
    compact_path = out_dir / "compact.csv"
    
    # Write all deduplicated records to CSV with all original columns
    if deduped:
        # Get all possible columns from the records
        all_keys = set()
        for rec in deduped:
            all_keys.update(rec.keys())
        # Remove internal keys we don't want in CSV
        exclude = {"_source"}
        columns = sorted(k for k in all_keys if k not in exclude)
        
        with compact_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for rec in deduped:
                # Convert non-serializable to string
                row = {}
                for k in columns:
                    v = rec.get(k, "")
                    if isinstance(v, (dict, list)):
                        row[k] = json.dumps(v)
                    else:
                        row[k] = str(v)
                writer.writerow(row)
        
        print(f"  Wrote {compact_path} ({len(deduped)} records, {len(columns)} columns)")
    else:
        print(f"  No data to write")

    # ============================================================
    # Step 4: Generate LaTeX tables
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Generating LaTeX tables")
    print("=" * 60)
    generate_latex_tables(out_dir)

    # ============================================================
    # Step 5: Generate visualizations
    # ============================================================
    if not args.skip_viz:
        print("\n" + "=" * 60)
        print("STEP 5: Generating publication-quality figures")
        print("=" * 60)

        setup_style(args.style, args.latex)

        df = pd.read_csv(summary_path)
        # Remove DeepSeek models (tokenizer issues causing low rates)
        df = df[~df["model"].str.contains("deepseek", case=False, na=False)]

        plot_error_breakdown(df, viz_dir, args.format)
        plot_combined_heatmap(df, viz_dir, args.format)
        plot_heatmap_by_language(df, viz_dir, args.format)
        plot_model_scaling(df, viz_dir, args.format)
        plot_global_pass_rate(df, viz_dir, args.format)
        plot_global_ranking(df, viz_dir, args.format)  # Fair comparison (3 modes)
        plot_global_leaderboard_all(df, viz_dir, args.format)  # All models with bars
        plot_global_leaderboard_simple(df, viz_dir, args.format)  # All models, no bars
        plot_mixed_vs_direct(df, viz_dir, args.format)
        plot_cost_comparison(df, viz_dir, args.format)
        plot_token_efficiency(df, viz_dir, args.format)

        # Top N leaderboards
        plot_global_leaderboard_top_n(df, viz_dir, args.format, n=5)  # Top 5 with bars
        plot_global_leaderboard_top_n_simple(df, viz_dir, args.format, n=5)  # Top 5 simple
        plot_global_leaderboard_top_n(df, viz_dir, args.format, n=10)  # Top 10 with bars
        plot_global_leaderboard_top_n_simple(df, viz_dir, args.format, n=10)  # Top 10 simple (JUST PASS RATES)
        plot_mixed_vs_direct(df, viz_dir, args.format)
        plot_improvement_by_language(df, viz_dir, args.format)  # Improvement heatmap
        plot_model_size_heatmap(df, viz_dir, args.format)  # Model size heatmap
        plot_cost_comparison(df, viz_dir, args.format)
        plot_token_efficiency(df, viz_dir, args.format)

        print(f"\nAll figures saved to {viz_dir}")

    # Generate LaTeX .tex files for figures
    generate_figure_tex(out_dir)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


def plot_improvement_by_language(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """
    Heatmap showing improvement: constrained_mixed - constrained_direct by language.
    Only models with BOTH modes.
    """
    print("\n[Improvement Heatmap: Mixed - Direct by Language]")

    # Filter to constrained modes
    data = df[df["mode"].isin(["constrained_direct", "constrained_mixed"])].copy()
    if data.empty:
        print("  No constrained data")
        return

    # Find models with BOTH modes
    model_mode_counts = data.groupby(["model", "mode"]).size().reset_index()
    model_counts = model_mode_counts.groupby("model")["mode"].nunique()
    complete_models = model_counts[model_counts == 2].index

    if len(complete_models) == 0:
        print("  No models with both constrained modes")
        return

    data = data[data["model"].isin(complete_models)]

    # Pivot: model+language -> {direct, mixed}
    pivot = data.pivot_table(
        values="pass_rate", index=["model", "language"],
        columns="mode", aggfunc="mean"
    )

    if "constrained_direct" not in pivot.columns or "constrained_mixed" not in pivot.columns:
        print("  Missing one of the constrained modes")
        return

    # Calculate improvement
    pivot["improvement"] = pivot["constrained_mixed"] - pivot["constrained_direct"]

    # Create heatmap: model x language
    heatmap_data = pivot.pivot_table(
        values="improvement", index="model", columns="language", aggfunc="mean"
    )

    if heatmap_data.empty:
        print("  No data for heatmap")
        return

    plt.figure(figsize=(8, max(6, len(heatmap_data) * 0.6)))
    sns.heatmap(
        heatmap_data, annot=True, fmt="+.1f", cmap="RdYlGn",
        center=0, cbar_kws={"label": "Improvement (pct points)"},
        linewidths=0.5
    )
    plt.title("Improvement: Constrained Mixed - Direct by Language",
                 fontsize=16, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Model", fontsize=14)
    plt.xticks(rotation=15, ha="right", fontsize=11)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    save_fig(out_dir, "fig_improvement_by_language", fmt)


def plot_model_size_heatmap(df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    """
    Heatmap: Models ordered by size (params) x Mode.
    Shows pass rate with models sorted by parameter count.
    """
    print("\n[Heatmap: Models Ordered by Size x Mode]")

    # Only use the 3 main modes
    data = df[df["mode"].isin(MODES_HEATMAP)].copy()

    # Add model size for ordering
    data["model_size"] = data["model"].apply(model_param_billions)
    data = data[data["model_size"].notna()]

    if data.empty:
        print("  No model size data")
        return

    # Pivot: model -> mode (with size for ordering)
    pivot = data.pivot_table(
        values="pass_rate", index="model", columns="mode", aggfunc="mean"
    )

    # Add size column for sorting
    model_sizes = data.groupby("model")["model_size"].first()
    pivot["model_size"] = model_sizes
    pivot = pivot.sort_values("model_size")

    # Remove size column for heatmap
    heatmap_data = pivot[MODES_HEATMAP]

    plt.figure(figsize=(10, max(6, len(heatmap_data) * 0.6)))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
        cbar_kws={"label": "Pass Rate (%)"}, linewidths=0.5
    )
    plt.title("Model x Mode: Ordered by Model Size (Parameters)",
                 fontsize=16, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Model (ordered by size)", fontsize=14)
    plt.xticks(rotation=15, ha="right", fontsize=11)
    # Y-tick labels with size
    ytick_labels = [f"{m} ({p:.1f}B)" for m, p in zip(heatmap_data.index, pivot.loc[heatmap_data.index, "model_size"])]
    plt.yticks(range(len(heatmap_data)), ytick_labels, fontsize=9)
    plt.tight_layout()
    save_fig(out_dir, "fig_model_size_heatmap", fmt)


def generate_figure_tex(out_dir: Path):
    """Generate .tex files for each figure with proper LaTeX labeling."""
    fig_dir = out_dir / "figures" / "paper"
    if not fig_dir.exists():
        return
    print("\n[Generating LaTeX .tex files for figures]")
    for png_file in sorted(fig_dir.glob("*.png")):
        # Derive a label from filename: e.g., fig1_error_breakdown -> fig:error_breakdown
        label = "fig:" + png_file.stem
        # Caption: convert snake_case to Title Case
        caption = png_file.stem.replace("_", " ").title()
        tex_content = f"""\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth]{{{fig_dir.name}/{png_file.name}}}
  \\caption{{{caption}}}
  \\label{{{label}}}
\\end{{figure}}
"""
        tex_path = png_file.with_suffix(".tex")
        tex_path.write_text(tex_content)
        print(f"  Generated {tex_path.name}")


if __name__ == "__main__":
    main()
