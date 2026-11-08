#!/usr/bin/env python3
"""Aggregate per-fold CPU benchmark JSONs into mean +- sd summary JSONs.

Reads fold results from ``results/cpu_5fold/folds/fold{k}/`` and writes one
aggregated JSON per (method, subset) into ``results/cpu_5fold/`` with
``*_mean`` and ``*_sd`` fields for every numeric metric.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MB = 1024 ** 2

# Metrics for which mean +- sd should be reported.
AGGREGATED_METRICS = [
    "train_s",
    "inference_s",
    "rss_train_peak_bytes",
    "rss_inference_peak_bytes",
    "rss_total_peak_bytes",
]


def load_results(folds_dir, n_folds):
    """Return list of result dicts from all fold directories."""
    records = []
    for k in range(n_folds):
        fold_dir = os.path.join(folds_dir, f"fold{k}")
        if not os.path.isdir(fold_dir):
            continue
        for path in glob.glob(os.path.join(fold_dir, "*.json")):
            with open(path) as f:
                d = json.load(f)
            records.append(d)
    return records


def group_records(records):
    """Group per-fold records by (method, subset)."""
    groups = defaultdict(list)
    for d in records:
        key = (d["method"], d["subset"])
        groups[key].append(d)
    return groups


def aggregate_group(items):
    """Compute mean +- sd across folds for one method/subset."""
    items = sorted(items, key=lambda x: x.get("fold", 0))
    n = len(items)
    if n == 0:
        raise ValueError("empty group")

    first = items[0]
    result = {
        "method": first["method"],
        "subset": first["subset"],
        "n_genes": first.get("n_genes"),
        "epochs": first.get("epochs"),
        "k_folds": n,
    }

    # Sample-size metrics
    for key in ("n_train", "n_test"):
        vals = np.array([x[key] for x in items], dtype=float)
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_sd"] = float(np.std(vals, ddof=1)) if n > 1 else 0.0

    # Timing / memory metrics
    for key in AGGREGATED_METRICS:
        vals = np.array([x[key] for x in items], dtype=float)
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_sd"] = float(np.std(vals, ddof=1)) if n > 1 else 0.0

    # Convenience MB columns for memory figures
    for phase in ("train", "inference", "total"):
        key = f"rss_{phase}_peak_bytes"
        if f"{key}_mean" in result:
            result[f"rss_{phase}_mb_mean"] = result[f"{key}_mean"] / MB
            result[f"rss_{phase}_mb_sd"] = result[f"{key}_sd"] / MB

    # Inherit hardware metadata from the first fold (same node for all folds).
    if "hardware" in first:
        result["hardware"] = first["hardware"]

    return result


def method_filename(method):
    return f"{method.replace(' ', '_').replace('(', '').replace(')', '')}"


def write_json(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    method = method_filename(result["method"])
    path = os.path.join(out_dir, f"{result['subset']}_{method}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-dir", default=os.path.join(HERE, "results", "cpu_5fold", "folds"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "cpu_5fold"))
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    records = load_results(args.folds_dir, args.n_folds)
    if not records:
        raise RuntimeError(f"No fold results found in {args.folds_dir}")

    groups = group_records(records)
    for (method, subset), items in sorted(groups.items()):
        agg = aggregate_group(items)
        path = write_json(agg, args.out_dir)
        print(f"Aggregated {method}/{subset} over {len(items)} fold(s) -> {path}")


if __name__ == "__main__":
    main()
