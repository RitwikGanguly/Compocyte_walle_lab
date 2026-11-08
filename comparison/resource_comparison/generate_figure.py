#!/usr/bin/env python3
"""Generate Figure 4d-style compute-resource benchmark figures.

Reads JSON results written by the individual method scripts and produces three
separate figures: training time, inference time, and max memory.

Two Compocyte backends are shown (GPU / CPU).  They use two clearly separated
hues from the same blue family so that the flagship GPU result reads as its
own colour and the CPU result cannot be mistaken for it, while the remaining
methods keep the colour-blind-safe matplotlib tab10 palette.
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Default to the CPU benchmark folder because that is the historical main
# benchmark.  The GPU benchmark script passes --results-dir results/gpu.
RESULTS_DIR = os.path.join(HERE, "results", "cpu")
FIGS_DIR = os.path.join(HERE, "figs")

# Nominal cell numbers used in the paper (total cells per subset)
SUBSET_NOMINAL = {
    "liu_100k": 100_000,
    "zhang_250k": 250_000,
    "oelen_e1_500k": 500_000,
}

METHOD_ORDER = [
    "Compocyte (GPU)",
    "Compocyte (CPU)",
    "scANVI",
    "CellTypist",
    "treeArches",
    "scPoli",
    "CHETAH",
    "transferData",
]

# Compocyte (GPU) keeps the strong saturated blue; Compocyte (CPU) moves to a
# clearly different cyan hue so the two bars are never confused.  All other
# methods keep the tab10 colours.
METHOD_COLORS = {
    "Compocyte (GPU)": "#1F77B4",   # saturated blue  (flaghip result)
    "Compocyte (CPU)": "#17BECF",   # cyan            (distinct from GPU)
    "scANVI": "#FF7F0E",            # orange
    "CellTypist": "#2CA02C",        # green
    "treeArches": "#D62728",        # red
    "scPoli": "#9467BD",            # purple
    "CHETAH": "#8C564B",            # brown
    "transferData": "#E377C2",      # pink
}


def load_results(results_dir):
    records = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        mem_bytes = d.get("rss_total_peak_bytes", 0)
        gpu_bytes = d.get("gpu_peak_bytes", 0)
        max_mem_mb = max(mem_bytes, gpu_bytes) / 1024 ** 2
        rss_train_bytes = d.get("rss_train_peak_bytes", 0)
        rss_inference_bytes = d.get("rss_inference_peak_bytes", 0)
        gpu_train_bytes = d.get("gpu_train_peak_bytes", 0)
        gpu_inference_bytes = d.get("gpu_inference_peak_bytes", 0)
        records.append({
            "method": d["method"],
            "subset": d["subset"],
            "n_cells": SUBSET_NOMINAL.get(d["subset"], int(d["n_train"])),
            "train_s": float(d["train_s"]),
            "inference_s": float(d["inference_s"]),
            "max_mem_mb": max_mem_mb,
            "rss_train_mb": rss_train_bytes / 1024 ** 2,
            "rss_inference_mb": rss_inference_bytes / 1024 ** 2,
            "gpu_train_mb": gpu_train_bytes / 1024 ** 2,
            "gpu_inference_mb": gpu_inference_bytes / 1024 ** 2,
            # A result carrying a "note" was not a clean, complete run
            # (e.g. the 500k transferData integration was terminated early),
            # so its bar is hatched to flag it in the figure.
            "estimated": bool(d.get("note")),
        })
    return pd.DataFrame(records)


def make_single_figure(df, metric, ylabel, title, out_path, method_order=METHOD_ORDER):
    """Draw one grouped bar figure and save PDF + PNG."""
    df = df.copy()
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values(["n_cells", "method"])

    cell_numbers = sorted(df["n_cells"].unique())
    n_groups = len(cell_numbers)

    # Offsets are computed over the methods actually present in the data, not
    # the full METHOD_ORDER.  The GPU run omits Compocyte (CPU), CHETAH and
    # transferData, so using the full 8-method count would leave empty slots
    # (gaps) between the drawn bars -- e.g. right after Compocyte (GPU),
    # whose slot is immediately followed by the absent Compocyte (CPU).
    # Using only the present methods keeps every group tightly packed while
    # preserving the canonical METHOD_ORDER.
    present_methods = [m for m in method_order if m in df["method"].values]
    n_methods = len(present_methods)
    bar_width = 0.10
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5))

    has_estimated = False
    for i, method in enumerate(present_methods):
        sub = df[df["method"] == method]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        for j, nc in enumerate(cell_numbers):
            row = sub[sub["n_cells"] == nc]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            height = row[metric]
            if not np.isfinite(height):
                continue
            estimated = bool(row["estimated"])
            has_estimated = has_estimated or estimated
            ax.bar(
                x[j] + offset,
                height,
                width=bar_width,
                color=METHOD_COLORS[method],
                hatch="///" if estimated else None,
                edgecolor="#3a3a3a" if estimated else "white",
                linewidth=0.7 if estimated else 0.3,
            )
            if estimated:
                # offset in points, so the marker hugs the bar top on a log axis
                ax.annotate(
                    "*",
                    xy=(x[j] + offset, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=15,
                    color="#3a3a3a",
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

    # Headroom so the tallest bar (and any "*" marker above it) never touches
    # the top of the axes.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.7)

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

    if has_estimated:
        ax.text(
            0.0, -0.17,
            "* hatched bar: run terminated before completion; value is a lower bound.",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="#3a3a3a",
        )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path} and {png_path}")


def make_memory_figure(df, phase, ylabel, title, out_path, method_order=METHOD_ORDER):
    """Draw one grouped bar figure for peak memory (training or inference phase).

    For each method two adjacent sub-bars are drawn whenever data exists:
    system RAM (solid fill) and GPU VRAM (hatched fill).  CPU-only methods
    show RAM only.  The two legends separate *method* (colour) from *memory
    type* (solid vs hatched) so the figure unambiguously reports RAM vs VRAM.
    """
    df = df.copy()
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values(["n_cells", "method"])

    cell_numbers = sorted(df["n_cells"].unique())
    n_groups = len(cell_numbers)
    present_methods = [m for m in method_order if m in df["method"].values]
    n_methods = len(present_methods)

    rss_col = f"rss_{phase}_mb"
    gpu_col = f"gpu_{phase}_mb"

    bar_width = 0.14
    sub_width = bar_width / 2
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5))

    has_estimated = False
    for i, method in enumerate(present_methods):
        sub = df[df["method"] == method]
        method_offset = (i - n_methods / 2 + 0.5) * bar_width
        for j, nc in enumerate(cell_numbers):
            row = sub[sub["n_cells"] == nc]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            rss_val = row[rss_col]
            gpu_val = row[gpu_col]
            estimated = bool(row["estimated"])
            has_estimated = has_estimated or estimated

            have_rss = np.isfinite(rss_val) and rss_val > 0
            have_gpu = np.isfinite(gpu_val) and gpu_val > 0
            n_sub = int(have_rss) + int(have_gpu)
            if n_sub == 0:
                continue

            if n_sub == 2:
                sub_offsets = [-sub_width / 2, sub_width / 2]
                sub_values = [(rss_val, False), (gpu_val, True)]
            else:
                sub_offsets = [0.0]
                sub_values = [(rss_val if have_rss else gpu_val, not have_rss)]

            for (val, is_vram), soff in zip(sub_values, sub_offsets):
                ax.bar(
                    x[j] + method_offset + soff,
                    val,
                    width=sub_width,
                    color=METHOD_COLORS[method],
                    hatch="///" if is_vram else None,
                    edgecolor="#3a3a3a" if (estimated or is_vram) else "white",
                    linewidth=0.7 if (estimated or is_vram) else 0.3,
                )
                if estimated:
                    ax.annotate(
                        "*",
                        xy=(x[j] + method_offset + soff, val),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=15,
                        color="#3a3a3a",
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
    ax.set_ylim(lo, hi * 1.7)

    # Legend 1: methods (colour encodes method)
    method_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[m], edgecolor="white")
        for m in present_methods
    ]
    leg1 = ax.legend(
        method_handles,
        present_methods,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        ncol=1,
        title="Method",
    )
    ax.add_artist(leg1)

    # Legend 2: memory type (solid = system RAM, hatched = GPU VRAM)
    mem_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", edgecolor="#3a3a3a", hatch="///"),
    ]
    ax.legend(
        mem_handles,
        ["RAM (system)", "VRAM (GPU)"],
        loc="upper left",
        bbox_to_anchor=(1.02, 0.62),
        frameon=False,
        ncol=1,
        title="Memory",
    )

    if has_estimated:
        ax.text(
            0.0, -0.17,
            "* hatched bar: run terminated before completion; value is a lower bound.",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="#3a3a3a",
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
    ap.add_argument(
        "--tag",
        default="",
        help="Optional suffix appended to figure filenames (e.g. 'gpu' -> *_gpu.pdf).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_results(args.results_dir)
    if df.empty:
        raise RuntimeError(f"No results found in {args.results_dir}")

    suffix = f"_{args.tag}" if args.tag else ""
    figures = [
        ("train_s", "Training time (s)", "Training time",
         f"compute_resource_benchmark_training_time{suffix}.pdf"),
        ("inference_s", "Inference time (s)", "Inference time",
         f"compute_resource_benchmark_inference_time{suffix}.pdf"),
    ]

    for metric, ylabel, title, fname in figures:
        out_path = os.path.join(args.out_dir, fname)
        make_single_figure(df, metric, ylabel, title, out_path)

    # Peak memory, split by phase (training vs inference) and by type (RAM vs VRAM)
    # so the training >> inference contrast and the RAM vs VRAM distinction are
    # both visible instead of being collapsed into a single "max memory" bar.
    memory_figures = [
        ("train", "Peak training memory (MB)", "Peak training memory",
         f"compute_resource_benchmark_training_memory{suffix}.pdf"),
        ("inference", "Peak inference memory (MB)", "Peak inference memory",
         f"compute_resource_benchmark_inference_memory{suffix}.pdf"),
    ]

    for phase, ylabel, title, fname in memory_figures:
        out_path = os.path.join(args.out_dir, fname)
        make_memory_figure(df, phase, ylabel, title, out_path)


if __name__ == "__main__":
    main()
