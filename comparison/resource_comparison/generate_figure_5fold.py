#!/usr/bin/env python3
"""Generate 5-fold CV CPU benchmark figures with mean +- sd error bars.

Reads aggregated JSONs produced by aggregate_5fold.py from
``results/cpu_5fold/`` and draws four figures: training time, inference time,
training memory, and inference memory.  Every bar shows the mean across the
five folds and a vertical error bar for the standard deviation.
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Re-use the canonical method order / colours from the main benchmark.
from generate_figure import METHOD_COLORS, METHOD_ORDER, SUBSET_NOMINAL

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "cpu_5fold")
FIGS_DIR = os.path.join(HERE, "figs")


def load_aggregated(results_dir):
    records = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        records.append(d)
    return pd.DataFrame(records)


def make_figure(df, metric, ylabel, title, out_path, method_order=METHOD_ORDER):
    """Draw one grouped bar figure (mean +- sd) and save PDF + PNG."""
    mean_col = f"{metric}_mean"
    sd_col = f"{metric}_sd"
    if mean_col not in df.columns:
        raise RuntimeError(f"Metric {metric} not found in aggregated results")

    df = df.copy()
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values(["subset", "method"])

    # Recover nominal cell numbers from subset names.
    df["n_cells"] = df["subset"].map(SUBSET_NOMINAL)
    cell_numbers = sorted(df["n_cells"].unique())
    n_groups = len(cell_numbers)

    present_methods = [m for m in method_order if m in df["method"].values]
    n_methods = len(present_methods)
    bar_width = 0.10
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, method in enumerate(present_methods):
        sub = df[df["method"] == method]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        for j, nc in enumerate(cell_numbers):
            row = sub[sub["n_cells"] == nc]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            mean = row[mean_col]
            sd = row[sd_col]
            if not np.isfinite(mean) or mean <= 0:
                continue
            if not np.isfinite(sd) or sd < 0:
                sd = 0.0

            # Asymmetric error bar: keep the lower bound strictly positive
            # so it renders correctly on a log axis.
            lower_bound = max(mean - sd, mean * 0.05)
            yerr = np.array([[mean - lower_bound], [sd]])

            ax.bar(
                x[j] + offset,
                mean,
                width=bar_width,
                yerr=yerr,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.3,
                error_kw={
                    "ecolor": "#3a3a3a",
                    "elinewidth": 1.0,
                    "capthick": 1.0,
                    "capsize": 2.5,
                },
            )

    ax.set_xlabel("cell number")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)

    def sci_label(n):
        val = n / 1e5
        if val == int(val):
            return f"{int(val)}×10⁵"
        return f"{val}×10⁵"

    ax.set_xticklabels([sci_label(nc) for nc in cell_numbers])
    ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title)

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[m], edgecolor="white")
        for m in present_methods
    ]
    ax.legend(
        handles,
        present_methods,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        ncol=1,
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path} and {png_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--out-dir", default=FIGS_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_aggregated(args.results_dir)
    if df.empty:
        raise RuntimeError(f"No aggregated results found in {args.results_dir}")

    suffix = "_5fold"
    figures = [
        ("train_s", "Training time (s)", "Training time (mean ± SD, 5-fold CV)",
         f"compute_resource_benchmark_training_time{suffix}.pdf"),
        ("inference_s", "Inference time (s)", "Inference time (mean ± SD, 5-fold CV)",
         f"compute_resource_benchmark_inference_time{suffix}.pdf"),
        ("rss_train_mb", "Peak training memory (MB)", "Peak training memory (mean ± SD, 5-fold CV)",
         f"compute_resource_benchmark_training_memory{suffix}.pdf"),
        ("rss_inference_mb", "Peak inference memory (MB)", "Peak inference memory (mean ± SD, 5-fold CV)",
         f"compute_resource_benchmark_inference_memory{suffix}.pdf"),
    ]

    for metric, ylabel, title, fname in figures:
        out_path = os.path.join(args.out_dir, fname)
        make_figure(df, metric, ylabel, title, out_path)


if __name__ == "__main__":
    main()
