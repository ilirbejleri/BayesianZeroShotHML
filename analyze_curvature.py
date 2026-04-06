#!/usr/bin/env python3
"""
Cross-dataset curvature analysis for Paper 2.

Reads training logs and eval CSVs from all 7 datasets, generates
publication-quality figures for the curvature study:

  Figure 1: Curvature vs data fraction (one line per dataset)
  Figure 2: Curvature vs hierarchy complexity (scatter)
  Figure 3: Fan et al. comparison (table/bar chart)
  Figure 4: c ∝ log(N) fits per dataset
  Figure 5: Accuracy tables (all datasets)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

from hierarchy import compute_hierarchy_metrics, HIERARCHY_TREES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
FIG_DIR = Path("paper2/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["awa2", "cifar100", "cub200", "sun397",
            "stanford_cars", "tiered_imagenet", "inaturalist"]
DATASET_LABELS = {
    "awa2": "AwA2",
    "cifar100": "CIFAR-100",
    "cub200": "CUB-200",
    "sun397": "SUN397",
    "stanford_cars": "Stanford Cars",
    "tiered_imagenet": "tiered-ImageNet",
    "inaturalist": "iNaturalist",
}

FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
FRACTION_LABELS = ["1%", "5%", "10%", "25%", "50%", "100%"]

# Fan et al. (2025) reported curvature values for comparison
FAN_ET_AL_CURVATURES = {
    "tiered_imagenet": {"1-shot": 0.113, "5-shot": 0.127},
    "cifar100": {"CIFAR10-LT": 5.36e-4},  # CIFAR-10, not 100 — note difference
}

# Colors for each dataset
DATASET_COLORS = {
    "awa2": "#d62728",
    "cifar100": "#1f77b4",
    "cub200": "#2ca02c",
    "sun397": "#ff7f0e",
    "stanford_cars": "#9467bd",
    "tiered_imagenet": "#8c564b",
    "inaturalist": "#e377c2",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_logs():
    """Load training CSVs into a dict keyed by (model, dataset, fraction)."""
    logs = {}
    for path in sorted(LOG_DIR.glob("train_*.csv")):
        name = path.stem
        # Match: train_{model}_{dataset}_abl{frac}_dim{dim}
        match = re.match(
            r"train_(bayesian|euclidean)_(\w+)_abl([\d.]+)_dim\d+", name
        )
        if not match:
            # Also try Paper 1 format: train_{model}_abl{frac}_dim{dim}
            match2 = re.match(r"train_(bayesian|euclidean)_abl([\d.]+)_dim\d+", name)
            if match2:
                model = match2.group(1)
                dataset = "awa2"
                frac = float(match2.group(2))
            else:
                continue
        else:
            model = match.group(1)
            dataset = match.group(2)
            frac = float(match.group(3))

        df = pd.read_csv(path)
        logs[(model, dataset, frac)] = df
    return logs


def load_eval_results():
    """Load eval CSVs into a dict keyed by (model, dataset, fraction)."""
    results = {}
    for path in sorted(LOG_DIR.glob("eval_*.csv")):
        name = path.stem
        match = re.match(
            r"eval_(bayesian|euclidean)_(\w+)_abl([\d.]+)_dim\d+", name
        )
        if not match:
            match2 = re.match(r"eval_(bayesian|euclidean)_abl([\d.]+)_dim\d+", name)
            if match2:
                model = match2.group(1)
                dataset = "awa2"
                frac = float(match2.group(2))
            else:
                continue
        else:
            model = match.group(1)
            dataset = match.group(2)
            frac = float(match.group(3))

        df = pd.read_csv(path)
        results[(model, dataset, frac)] = df.iloc[0].to_dict()
    return results


# ---------------------------------------------------------------------------
# Figure 1: Curvature vs data fraction (main result)
# ---------------------------------------------------------------------------

def plot_curvature_vs_data_all(logs):
    """One line per dataset showing learned curvature vs training data fraction."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for ds in DATASETS:
        fracs, curvatures = [], []
        for frac in FRACTIONS:
            key = ("bayesian", ds, frac)
            if key in logs:
                fracs.append(frac)
                curvatures.append(logs[key]["curvature"].iloc[-1])

        if fracs:
            ax.plot(fracs, curvatures, "o-",
                    color=DATASET_COLORS[ds],
                    linewidth=2, markersize=6,
                    label=DATASET_LABELS[ds])

    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Learned Curvature $c$")
    ax.set_xscale("log")
    ax.set_xticks(FRACTIONS)
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.0%}" if x >= 0.1 else f"{x:.0%}")
    )
    ax.legend(title="Dataset", loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("Learned Curvature Scales with Data Availability Across Datasets")
    plt.tight_layout()

    fig.savefig(FIG_DIR / "curvature_vs_data_all.pdf")
    fig.savefig(FIG_DIR / "curvature_vs_data_all.png")
    print("  Saved curvature_vs_data_all.pdf/png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Curvature vs hierarchy complexity (scatter)
# ---------------------------------------------------------------------------

def plot_curvature_vs_hierarchy(logs):
    """Scatter: hierarchy depth on x, final curvature (100% data) on y."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    depths, curvatures, labels = [], [], []
    branching_factors = []

    for ds in DATASETS:
        key = ("bayesian", ds, 1.0)
        if key not in logs:
            continue
        metrics = compute_hierarchy_metrics(ds)
        c_final = logs[key]["curvature"].iloc[-1]

        depths.append(metrics["depth"])
        branching_factors.append(metrics["branching_factor"])
        curvatures.append(c_final)
        labels.append(DATASET_LABELS[ds])

    # Scatter: depth vs curvature
    ax = axes[0]
    for i, label in enumerate(labels):
        ds_key = [k for k, v in DATASET_LABELS.items() if v == label][0]
        ax.scatter(depths[i], curvatures[i], s=100, zorder=3,
                   color=DATASET_COLORS[ds_key], edgecolors="black", linewidth=0.5)
        ax.annotate(label, (depths[i], curvatures[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)

    # Fit line
    if len(depths) >= 3:
        slope, intercept, r_val, _, _ = stats.linregress(depths, curvatures)
        x_fit = np.linspace(min(depths) - 0.5, max(depths) + 0.5, 100)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color="gray", alpha=0.5,
                label=f"$R^2 = {r_val**2:.2f}$")
        ax.legend()

    ax.set_xlabel("Taxonomy Tree Depth")
    ax.set_ylabel("Learned Curvature $c$ (100% data)")
    ax.set_title("Curvature vs. Hierarchy Depth")
    ax.grid(True, alpha=0.3)

    # Scatter: branching factor vs curvature
    ax = axes[1]
    for i, label in enumerate(labels):
        ds_key = [k for k, v in DATASET_LABELS.items() if v == label][0]
        ax.scatter(branching_factors[i], curvatures[i], s=100, zorder=3,
                   color=DATASET_COLORS[ds_key], edgecolors="black", linewidth=0.5)
        ax.annotate(label, (branching_factors[i], curvatures[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)

    if len(branching_factors) >= 3:
        slope, intercept, r_val, _, _ = stats.linregress(branching_factors, curvatures)
        x_fit = np.linspace(min(branching_factors) - 0.5, max(branching_factors) + 0.5, 100)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color="gray", alpha=0.5,
                label=f"$R^2 = {r_val**2:.2f}$")
        ax.legend()

    ax.set_xlabel("Mean Branching Factor")
    ax.set_ylabel("Learned Curvature $c$ (100% data)")
    ax.set_title("Curvature vs. Branching Factor")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "curvature_vs_hierarchy.pdf")
    fig.savefig(FIG_DIR / "curvature_vs_hierarchy.png")
    print("  Saved curvature_vs_hierarchy.pdf/png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: c ∝ log(N) fit per dataset
# ---------------------------------------------------------------------------

def plot_log_n_fits(logs):
    """Fit c = a * log(N) + b per dataset and overlay."""
    fig, ax = plt.subplots(figsize=(8, 5))

    fit_results = {}

    for ds in DATASETS:
        ns, curvatures = [], []
        for frac in FRACTIONS:
            key = ("bayesian", ds, frac)
            if key not in logs:
                continue
            # Estimate N from the log: number of training samples
            c_final = logs[key]["curvature"].iloc[-1]
            # Use fraction to estimate N (approximate)
            ns.append(frac)
            curvatures.append(c_final)

        if len(ns) < 3:
            continue

        log_ns = np.log(ns)
        slope, intercept, r_val, _, _ = stats.linregress(log_ns, curvatures)
        fit_results[ds] = {"slope": slope, "intercept": intercept, "r2": r_val**2}

        ax.plot(ns, curvatures, "o-",
                color=DATASET_COLORS[ds], linewidth=2, markersize=6,
                label=f"{DATASET_LABELS[ds]} ($a$={slope:.2f}, $R^2$={r_val**2:.2f})")

        # Plot fit line
        x_fit = np.linspace(min(ns), max(ns), 100)
        y_fit = slope * np.log(x_fit) + intercept
        ax.plot(x_fit, y_fit, "--", color=DATASET_COLORS[ds], alpha=0.4)

    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Learned Curvature $c$")
    ax.set_xscale("log")
    ax.set_xticks(FRACTIONS)
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.0%}" if x >= 0.1 else f"{x:.0%}")
    )
    ax.legend(title="Dataset (slope, $R^2$)", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("$c \\propto \\log(N)$ Fit Per Dataset")
    plt.tight_layout()

    fig.savefig(FIG_DIR / "log_n_fits.pdf")
    fig.savefig(FIG_DIR / "log_n_fits.png")
    print("  Saved log_n_fits.pdf/png")
    plt.close(fig)

    return fit_results


# ---------------------------------------------------------------------------
# Figure 4: Fan et al. comparison
# ---------------------------------------------------------------------------

def plot_fan_comparison(logs):
    """Compare our learned curvature on shared datasets with Fan et al."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Our curvature at 100% data
    our_values = {}
    for ds in ["tiered_imagenet", "cifar100"]:
        key = ("bayesian", ds, 1.0)
        if key in logs:
            our_values[ds] = logs[key]["curvature"].iloc[-1]

    if not our_values:
        print("  [WARN] No shared-dataset results for Fan et al. comparison")
        plt.close(fig)
        return

    labels = []
    our_vals = []
    fan_vals = []

    if "tiered_imagenet" in our_values:
        labels.append("tiered-IN\n(ours, 100%)")
        our_vals.append(our_values["tiered_imagenet"])
        labels.append("tiered-IN\n(Fan, 1-shot)")
        fan_vals.append(FAN_ET_AL_CURVATURES["tiered_imagenet"]["1-shot"])
        labels.append("tiered-IN\n(Fan, 5-shot)")
        fan_vals.append(FAN_ET_AL_CURVATURES["tiered_imagenet"]["5-shot"])

    x = np.arange(len(labels))
    all_vals = our_vals + fan_vals
    colors = ["#d62728"] * len(our_vals) + ["#1f77b4"] * len(fan_vals)

    bars = ax.bar(x, all_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Learned Curvature $c$")
    ax.set_title("Curvature Comparison: Ours vs. Fan et al. (2025)")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, all_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fan_comparison.pdf")
    fig.savefig(FIG_DIR / "fan_comparison.png")
    print("  Saved fan_comparison.pdf/png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Accuracy tables
# ---------------------------------------------------------------------------

def print_accuracy_tables(evals):
    """Print and generate LaTeX accuracy tables for all datasets."""
    print("\n" + "=" * 80)
    print("  ACCURACY RESULTS — ALL DATASETS")
    print("=" * 80)

    for ds in DATASETS:
        print(f"\n--- {DATASET_LABELS.get(ds, ds)} ---")
        print(f"{'Model':<12} {'Frac':>6} {'Top-1':>8} {'Top-5':>8} {'N_test':>8}")
        print("-" * 50)
        for model in ["bayesian", "euclidean"]:
            for frac in FRACTIONS:
                key = (model, ds, frac)
                if key in evals:
                    e = evals[key]
                    print(f"{model:<12} {frac:>6.2f} {float(e['top1']):>7.2f}% "
                          f"{float(e['top5']):>7.2f}% {int(e['n_test']):>8d}")

    # LaTeX table for curvature across datasets
    print("\n" + "=" * 80)
    print("  LATEX: Curvature Table")
    print("=" * 80)
    print("% Final curvature c at each data fraction, per dataset")
    print(r"\begin{tabular}{l" + "c" * len(FRACTIONS) + "}")
    print(r"\toprule")
    header = " & ".join([f"\\textbf{{{fl}}}" for fl in FRACTION_LABELS])
    print(f"\\textbf{{Dataset}} & {header} \\\\")
    print(r"\midrule")


# ---------------------------------------------------------------------------
# Summary of curvature values
# ---------------------------------------------------------------------------

def print_curvature_summary(logs):
    """Print curvature at convergence for all datasets."""
    print("\n" + "=" * 80)
    print("  CURVATURE SUMMARY (epoch 200, Bayesian model)")
    print("=" * 80)
    print(f"{'Dataset':<20} " + "  ".join(f"{fl:>6}" for fl in FRACTION_LABELS))
    print("-" * 80)

    for ds in DATASETS:
        vals = []
        for frac in FRACTIONS:
            key = ("bayesian", ds, frac)
            if key in logs:
                vals.append(f"{logs[key]['curvature'].iloc[-1]:>6.3f}")
            else:
                vals.append(f"{'---':>6}")
        print(f"{DATASET_LABELS.get(ds, ds):<20} " + "  ".join(vals))

    print("\n  Hierarchy metrics:")
    print(f"{'Dataset':<20} {'Depth':>6} {'Branch':>8} {'Internal':>9}")
    print("-" * 50)
    for ds in DATASETS:
        m = compute_hierarchy_metrics(ds)
        print(f"{DATASET_LABELS.get(ds, ds):<20} {m['depth']:>6} "
              f"{m['branching_factor']:>8.2f} {m['internal_nodes']:>9}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading training logs...")
    logs = load_training_logs()
    print(f"  Found {len(logs)} training runs")

    # Count per-dataset
    datasets_found = set()
    for (model, ds, frac) in logs:
        datasets_found.add(ds)
    print(f"  Datasets: {', '.join(sorted(datasets_found))}")

    print("Loading eval results...")
    evals = load_eval_results()
    print(f"  Found {len(evals)} eval results")

    print("\nGenerating figures...")

    if any(("bayesian", ds, f) in logs for ds in DATASETS for f in FRACTIONS):
        plot_curvature_vs_data_all(logs)
        plot_curvature_vs_hierarchy(logs)
        fit_results = plot_log_n_fits(logs)
        plot_fan_comparison(logs)

        if fit_results:
            print("\n  c ∝ log(N) fit results:")
            for ds, res in fit_results.items():
                print(f"    {DATASET_LABELS.get(ds, ds)}: "
                      f"slope={res['slope']:.3f}  R²={res['r2']:.3f}")
    else:
        print("  [WARN] No Bayesian training logs found for curvature analysis")

    print_curvature_summary(logs)
    print_accuracy_tables(evals)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
