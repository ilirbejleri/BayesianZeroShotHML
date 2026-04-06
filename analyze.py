#!/usr/bin/env python3
"""
Analysis and visualization for Bayesian Hyperbolic ZSL experiments.

Reads training CSVs and eval CSVs from logs/, generates publication-quality
figures in paper/figures/, and prints summary tables.
"""

import os
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
FRACTION_LABELS = ["1%", "5%", "10%", "25%", "50%", "100%"]

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_training_logs():
    """Load all training CSVs into a dict keyed by (model, fraction)."""
    logs = {}
    for path in sorted(LOG_DIR.glob("train_*.csv")):
        name = path.stem  # e.g. train_bayesian_abl0.01_dim64
        match = re.match(r"train_(bayesian|euclidean)_abl([\d.]+)_dim\d+", name)
        if not match:
            continue
        model = match.group(1)
        frac = float(match.group(2))
        df = pd.read_csv(path)
        logs[(model, frac)] = df
    return logs


def load_eval_results():
    """Load all eval CSVs into a dict keyed by (model, fraction)."""
    results = {}
    for path in sorted(LOG_DIR.glob("eval_*.csv")):
        name = path.stem
        match = re.match(r"eval_(bayesian|euclidean)_abl([\d.]+)_dim\d+", name)
        if not match:
            continue
        model = match.group(1)
        frac = float(match.group(2))
        df = pd.read_csv(path)
        results[(model, frac)] = df.iloc[0].to_dict()
    return results


# ---------------------------------------------------------------------------
# Figure 1: Training loss curves
# ---------------------------------------------------------------------------

def plot_training_curves(logs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(FRACTIONS) - 1)) for i in range(len(FRACTIONS))]

    for ax, model, title in zip(axes, ["bayesian", "euclidean"],
                                 ["Bayesian Hyperbolic", "Euclidean Baseline"]):
        for frac, color, label in zip(FRACTIONS, colors, FRACTION_LABELS):
            key = (model, frac)
            if key not in logs:
                continue
            df = logs[key]
            ax.plot(df["epoch"], df["train_loss"], color=color, label=label,
                    linewidth=1.2, alpha=0.85)

        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_xlim(1, 200)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Training Loss (log scale)")
    axes[1].legend(title="Training Data", loc="upper right")
    fig.suptitle("Training Loss Convergence — Data Ablation Sweep", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.pdf")
    fig.savefig(FIG_DIR / "training_curves.png")
    print(f"  Saved training_curves.pdf/png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Curvature evolution (Bayesian only)
# ---------------------------------------------------------------------------

def plot_curvature_evolution(logs):
    fig, ax = plt.subplots(figsize=(6, 4))

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(FRACTIONS) - 1)) for i in range(len(FRACTIONS))]

    for frac, color, label in zip(FRACTIONS, colors, FRACTION_LABELS):
        key = ("bayesian", frac)
        if key not in logs:
            continue
        df = logs[key]
        ax.plot(df["epoch"], df["curvature"], color=color, label=label,
                linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learned Curvature $c$")
    ax.set_title("Poincar\u00e9 Ball Curvature Evolution")
    ax.legend(title="Training Data")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 200)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "curvature_evolution.pdf")
    fig.savefig(FIG_DIR / "curvature_evolution.png")
    print(f"  Saved curvature_evolution.pdf/png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: KL divergence over training
# ---------------------------------------------------------------------------

def plot_kl_evolution(logs):
    fig, ax = plt.subplots(figsize=(6, 4))

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(FRACTIONS) - 1)) for i in range(len(FRACTIONS))]

    for frac, color, label in zip(FRACTIONS, colors, FRACTION_LABELS):
        key = ("bayesian", frac)
        if key not in logs:
            continue
        df = logs[key]
        ax.plot(df["epoch"], df["kl_loss"], color=color, label=label,
                linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Posterior Collapse Monitoring (KL Term)")
    ax.legend(title="Training Data")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 200)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "kl_evolution.pdf")
    fig.savefig(FIG_DIR / "kl_evolution.png")
    print(f"  Saved kl_evolution.pdf/png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Data ablation — Top-1 and Top-5 accuracy
# ---------------------------------------------------------------------------

def plot_ablation_accuracy(evals):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, metric, title in zip(axes, ["top1", "top5"],
                                  ["Top-1 Accuracy (%)", "Top-5 Accuracy (%)"]):
        bay_vals = []
        euc_vals = []
        bay_fracs = []
        euc_fracs = []

        for frac in FRACTIONS:
            bkey = ("bayesian", frac)
            ekey = ("euclidean", frac)
            if bkey in evals:
                bay_vals.append(float(evals[bkey][metric]))
                bay_fracs.append(frac)
            if ekey in evals:
                euc_vals.append(float(evals[ekey][metric]))
                euc_fracs.append(frac)

        ax.plot(bay_fracs, bay_vals, "o-", color="#d62728", linewidth=2,
                markersize=7, label="Bayesian Hyperbolic", zorder=3)
        ax.plot(euc_fracs, euc_vals, "s--", color="#1f77b4", linewidth=2,
                markersize=7, label="Euclidean Baseline", zorder=3)

        ax.set_xlabel("Training Data Fraction")
        ax.set_ylabel(title)
        ax.set_xscale("log")
        ax.set_xticks(FRACTIONS)
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:.0%}" if x >= 0.1 else f"{x:.0%}"))
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate the gap at 1%
        if bay_fracs and euc_fracs:
            if 0.01 in bay_fracs and 0.01 in euc_fracs:
                b_idx = bay_fracs.index(0.01)
                e_idx = euc_fracs.index(0.01)
                gap = bay_vals[b_idx] - euc_vals[e_idx]
                if abs(gap) > 0.5:
                    mid = (bay_vals[b_idx] + euc_vals[e_idx]) / 2
                    ax.annotate(f"+{gap:.1f}%", xy=(0.01, mid),
                                fontsize=9, color="#d62728", fontweight="bold",
                                ha="left", va="center",
                                xytext=(0.015, mid))

    fig.suptitle("Zero-Shot Accuracy vs. Training Data — AwA2 Unseen Classes",
                 y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ablation_accuracy.pdf")
    fig.savefig(FIG_DIR / "ablation_accuracy.png")
    print(f"  Saved ablation_accuracy.pdf/png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Final curvature vs. dataset size
# ---------------------------------------------------------------------------

def plot_curvature_vs_data(logs):
    fig, ax = plt.subplots(figsize=(5, 3.5))

    fracs = []
    curvatures = []
    for frac in FRACTIONS:
        key = ("bayesian", frac)
        if key in logs:
            fracs.append(frac)
            curvatures.append(logs[key]["curvature"].iloc[-1])

    ax.plot(fracs, curvatures, "o-", color="#2ca02c", linewidth=2, markersize=8)
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Final Curvature $c$")
    ax.set_xscale("log")
    ax.set_title("Learned Curvature Scales with Data Availability")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(FRACTIONS)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0%}" if x >= 0.1 else f"{x:.0%}"))
    plt.tight_layout()
    fig.savefig(FIG_DIR / "curvature_vs_data.pdf")
    fig.savefig(FIG_DIR / "curvature_vs_data.png")
    print(f"  Saved curvature_vs_data.pdf/png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def print_summary_tables(logs, evals):
    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY (final epoch)")
    print("=" * 70)
    print(f"{'Model':<12} {'Frac':>6} {'Loss':>10} {'Align':>10} {'KL':>10} {'Curv':>8}")
    print("-" * 70)
    for model in ["bayesian", "euclidean"]:
        for frac in FRACTIONS:
            key = (model, frac)
            if key not in logs:
                continue
            row = logs[key].iloc[-1]
            print(f"{model:<12} {frac:>6.2f} {row['train_loss']:>10.6f} "
                  f"{row['align_loss']:>10.6f} {row['kl_loss']:>10.6f} "
                  f"{row['curvature']:>8.4f}")
        print()

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS (unseen classes)")
    print("=" * 70)
    print(f"{'Model':<12} {'Frac':>6} {'Top-1':>8} {'Top-5':>8} {'N_test':>8}")
    print("-" * 70)
    for model in ["bayesian", "euclidean"]:
        for frac in FRACTIONS:
            key = (model, frac)
            if key in evals:
                e = evals[key]
                print(f"{model:<12} {frac:>6.2f} {float(e['top1']):>7.2f}% "
                      f"{float(e['top5']):>7.2f}% {int(e['n_test']):>8d}")
            else:
                print(f"{model:<12} {frac:>6.2f}    ---      ---        ---")
        print()

    # LaTeX-ready table
    print("\n" + "=" * 70)
    print("  LATEX TABLE (copy into paper)")
    print("=" * 70)
    for metric in ["top1", "top5"]:
        mname = "Top-1" if metric == "top1" else "Top-5"
        print(f"\n% {mname} Accuracy Table")
        for model, label in [("euclidean", "Euclidean ZSL"),
                              ("bayesian", r"\textbf{Bayesian Hyp.\ (ours)}")]:
            vals = []
            for frac in FRACTIONS:
                key = (model, frac)
                if key in evals:
                    vals.append(f"{float(evals[key][metric]):.2f}")
                else:
                    vals.append("---")
            print(f"{label:<40s} & " + " & ".join(vals) + r" \\")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading training logs...")
    logs = load_training_logs()
    print(f"  Found {len(logs)} training runs")

    print("Loading eval results...")
    evals = load_eval_results()
    print(f"  Found {len(evals)} eval results")

    missing = []
    for model in ["bayesian", "euclidean"]:
        for frac in FRACTIONS:
            if (model, frac) not in evals:
                missing.append(f"{model} abl={frac}")
    if missing:
        print(f"  [WARN] Missing eval for: {', '.join(missing)}")

    print("\nGenerating figures...")
    plot_training_curves(logs)
    plot_curvature_evolution(logs)
    plot_kl_evolution(logs)
    plot_ablation_accuracy(evals)
    plot_curvature_vs_data(logs)

    print_summary_tables(logs, evals)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
