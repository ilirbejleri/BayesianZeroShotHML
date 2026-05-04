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

N_CLASSES = {
    "awa2": 50,
    "cifar100": 100,
    "cub200": 200,
    "sun397": 397,
    "stanford_cars": 196,
    "tiered_imagenet": 608,
    "inaturalist": 500,
}

EMBED_DIM = {
    "awa2": 85,
    "cifar100": 300,
    "cub200": 300,
    "sun397": 300,
    "stanford_cars": 300,
    "tiered_imagenet": 300,
    "inaturalist": 300,
}

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
    """One marker series per dataset (curvature vs fraction) overlaid with
    the per-dataset c = a*log(N) + b least-squares fit (light dashed line).
    This merges the role of the previous separate trajectory + fit figures
    so one plot now carries both."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for ds in DATASETS:
        fracs, curvatures = [], []
        for frac in FRACTIONS:
            key = ("bayesian", ds, frac)
            if key in logs:
                fracs.append(frac)
                curvatures.append(logs[key]["curvature"].iloc[-1])

        if not fracs:
            continue

        # Solid line + markers for the empirical trajectory
        ax.plot(fracs, curvatures, "o-",
                color=DATASET_COLORS[ds],
                linewidth=2, markersize=6,
                label=DATASET_LABELS[ds])

        # Light dashed line for the c = a*log(N) + b fit
        if len(fracs) >= 3:
            log_fracs = np.log(fracs)
            slope, intercept, _, _, _ = stats.linregress(log_fracs, curvatures)
            x_fit = np.linspace(min(fracs), max(fracs), 100)
            y_fit = slope * np.log(x_fit) + intercept
            ax.plot(x_fit, y_fit, "--",
                    color=DATASET_COLORS[ds], alpha=0.35, linewidth=1)

    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Learned Curvature $c$")
    ax.set_xscale("log")
    ax.set_xticks(FRACTIONS)
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.0%}" if x >= 0.1 else f"{x:.0%}")
    )
    ax.legend(title="Dataset", loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Learned Curvature vs.\\ Training Data Fraction "
                 "(solid: empirical; dashed: $c = a\\log N + b$ fit)")
    plt.tight_layout()

    fig.savefig(FIG_DIR / "curvature_vs_data_all.pdf")
    fig.savefig(FIG_DIR / "curvature_vs_data_all.png")
    print("  Saved curvature_vs_data_all.pdf/png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Curvature vs hierarchy complexity (scatter)
# ---------------------------------------------------------------------------

def _r2_with_and_without_awa2(xs, ys, dataset_keys):
    """Return (r2_all, r2_no_awa2) for the regression of ys on xs."""
    if len(xs) < 3:
        return None, None
    _, _, r_all, _, _ = stats.linregress(xs, ys)
    keep = [i for i, k in enumerate(dataset_keys) if k != "awa2"]
    if len(keep) < 3:
        return r_all ** 2, None
    xs_no = [xs[i] for i in keep]
    ys_no = [ys[i] for i in keep]
    _, _, r_no, _, _ = stats.linregress(xs_no, ys_no)
    return r_all ** 2, r_no ** 2


def plot_curvature_vs_hierarchy(logs):
    """Scatter: hierarchy depth/branching on x, c_100% on y. Reports R² with
    and without AwA2 to make the AwA2-as-outlier story explicit."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dataset_keys, depths, branching_factors, curvatures, labels = [], [], [], [], []
    for ds in DATASETS:
        key = ("bayesian", ds, 1.0)
        if key not in logs:
            continue
        metrics = compute_hierarchy_metrics(ds)
        c_final = logs[key]["curvature"].iloc[-1]
        dataset_keys.append(ds)
        depths.append(metrics["depth"])
        branching_factors.append(metrics["branching_factor"])
        curvatures.append(c_final)
        labels.append(DATASET_LABELS[ds])

    def _scatter_panel(ax, xs, xlabel, title):
        for i, label in enumerate(labels):
            ds_key = dataset_keys[i]
            ax.scatter(xs[i], curvatures[i], s=100, zorder=3,
                       color=DATASET_COLORS[ds_key],
                       edgecolors="black", linewidth=0.5)
            ax.annotate(label, (xs[i], curvatures[i]),
                        textcoords="offset points", xytext=(8, 4), fontsize=8)
        if len(xs) >= 3:
            slope, intercept, r_val, _, _ = stats.linregress(xs, curvatures)
            x_fit = np.linspace(min(xs) - 0.5, max(xs) + 0.5, 100)
            ax.plot(x_fit, slope * x_fit + intercept, "--",
                    color="gray", alpha=0.5)
            r2_all, r2_no = _r2_with_and_without_awa2(xs, curvatures, dataset_keys)
            ann = (f"$R^2$ all 7 datasets: {r2_all:.2f}\n"
                   f"$R^2$ excluding AwA2: {r2_no:.2f}")
            ax.text(0.02, 0.98, ann, transform=ax.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                              edgecolor="gray"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Learned Curvature $c$ (100% data)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    _scatter_panel(axes[0], depths, "Taxonomy Tree Depth",
                   "Curvature vs. Hierarchy Depth")
    _scatter_panel(axes[1], branching_factors, "Mean Branching Factor",
                   "Curvature vs. Branching Factor")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "curvature_vs_hierarchy.pdf")
    fig.savefig(FIG_DIR / "curvature_vs_hierarchy.png")
    print("  Saved curvature_vs_hierarchy.pdf/png")
    plt.close(fig)

    # Also report the same numbers for slope a (from log-N fits) vs depth/branching
    return dataset_keys, depths, branching_factors, curvatures


# ---------------------------------------------------------------------------
# Figure 3: c ∝ log(N) fit per dataset
# ---------------------------------------------------------------------------

def plot_log_n_fits(logs):
    """Fit c = a * log(N) + b per dataset, with analytical 95% CI on the slope."""
    fig, ax = plt.subplots(figsize=(8, 5))

    fit_results = {}

    for ds in DATASETS:
        ns, curvatures = [], []
        for frac in FRACTIONS:
            key = ("bayesian", ds, frac)
            if key not in logs:
                continue
            c_final = logs[key]["curvature"].iloc[-1]
            ns.append(frac)
            curvatures.append(c_final)

        if len(ns) < 3:
            continue

        log_ns = np.log(ns)
        # linregress returns standard error of the slope (Wald) -> 95% CI via t.
        result = stats.linregress(log_ns, curvatures)
        slope, intercept = result.slope, result.intercept
        r2 = result.rvalue ** 2
        # 95% CI on slope: ±t_{0.975, n-2} * stderr
        df_resid = max(len(ns) - 2, 1)
        t_crit = stats.t.ppf(0.975, df_resid)
        ci_half = t_crit * result.stderr
        slope_lo, slope_hi = slope - ci_half, slope + ci_half
        # p-value already in result.pvalue (test slope==0)

        fit_results[ds] = {
            "slope": slope, "intercept": intercept, "r2": r2,
            "stderr": result.stderr, "ci": (slope_lo, slope_hi),
            "p_value": result.pvalue,
        }

        ax.plot(ns, curvatures, "o-",
                color=DATASET_COLORS[ds], linewidth=2, markersize=6,
                label=f"{DATASET_LABELS[ds]} ($a$={slope:+.2f}, $R^2$={r2:.2f})")

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
# Mechanical pattern classifier (Section 5.2)
# ---------------------------------------------------------------------------

def classify_pattern(curvatures):
    """Mechanical 3-way pattern classifier.

    Inputs: list of curvatures aligned with FRACTIONS (low → high data).
    Returns: 'A' (monotonic-increase), 'B' (rises-then-plateau),
             'C' (peaks-early-then-declines).

    Decision rule (applied mechanically):
      - Pattern A: argmax is the LAST fraction (curvature still climbing
                   at full data) AND c_final > 1.5 * c_initial.
      - Pattern C: argmax index <= 2 (i.e., peak at 1%, 5% or 10% data),
                   AND c_final < 0.85 * max(c).
      - Pattern B: everything else (peak in the middle, or final value
                   close to the peak).
    """
    arr = np.asarray(curvatures, dtype=float)
    arg = int(np.argmax(arr))
    c_init, c_final, c_max = arr[0], arr[-1], arr.max()
    if arg == len(arr) - 1 and c_final > 1.5 * c_init:
        return "A"
    if arg <= 2 and c_final < 0.85 * c_max:
        return "C"
    return "B"


def report_pattern_classification(logs):
    """Apply classify_pattern to every dataset and print the mechanical
    classification, with the inputs that drove each decision."""
    print("\n" + "=" * 80)
    print("  MECHANICAL PATTERN CLASSIFICATION")
    print("=" * 80)
    print(f"{'Dataset':<18} {'argmax@':>10} {'c_init':>8} {'c_max':>8} "
          f"{'c_final':>9} {'final/max':>11} {'Pattern':>8}")
    print("-" * 80)
    classifications = {}
    for ds in DATASETS:
        vals = []
        for frac in FRACTIONS:
            key = ("bayesian", ds, frac)
            if key not in logs:
                vals = None
                break
            vals.append(logs[key]["curvature"].iloc[-1])
        if vals is None:
            continue
        pat = classify_pattern(vals)
        classifications[ds] = pat
        arg = int(np.argmax(vals))
        c_init, c_final, c_max = vals[0], vals[-1], max(vals)
        print(f"{DATASET_LABELS[ds]:<18} {FRACTION_LABELS[arg]:>10} "
              f"{c_init:>8.3f} {c_max:>8.3f} {c_final:>9.3f} "
              f"{c_final/c_max:>11.3f} {pat:>8}")
    return classifications


# ---------------------------------------------------------------------------
# Cross-dataset regressions on n_classes / classes-per-embed-dim
# ---------------------------------------------------------------------------

def report_n_classes_regressions(logs, fit_results):
    """Regress c_100% and slope `a` on number of classes and on
    classes/embedding-dim ratio. Reports R² with and without AwA2."""
    print("\n" + "=" * 80)
    print("  REGRESSION OF c_100% AND SLOPE ON CLASS COUNT / EMBED-DIM")
    print("=" * 80)

    rows = []
    for ds in DATASETS:
        key = ("bayesian", ds, 1.0)
        if key not in logs or ds not in fit_results:
            continue
        rows.append({
            "ds": ds,
            "n_classes": N_CLASSES[ds],
            "log_n_classes": np.log(N_CLASSES[ds]),
            "classes_per_dim": N_CLASSES[ds] / EMBED_DIM[ds],
            "c_100": logs[key]["curvature"].iloc[-1],
            "slope": fit_results[ds]["slope"],
        })

    def _report(xs, ys, dataset_keys, x_label, y_label):
        if len(xs) < 3:
            return
        slope, intercept, r, _, _ = stats.linregress(xs, ys)
        r2_all = r ** 2
        keep = [i for i, k in enumerate(dataset_keys) if k != "awa2"]
        if len(keep) >= 3:
            slope_no, intercept_no, r_no, _, _ = stats.linregress(
                [xs[i] for i in keep], [ys[i] for i in keep])
            r2_no = r_no ** 2
        else:
            slope_no, r2_no = np.nan, np.nan
        print(f"  {y_label} vs {x_label}:")
        print(f"    all 7 datasets:   slope={slope:+.4f}  R^2={r2_all:.3f}")
        print(f"    excluding AwA2:   slope={slope_no:+.4f}  R^2={r2_no:.3f}")

    keys = [r["ds"] for r in rows]
    _report([r["log_n_classes"] for r in rows], [r["c_100"] for r in rows],
            keys, "log(n_classes)", "c at 100% data")
    _report([r["log_n_classes"] for r in rows], [r["slope"] for r in rows],
            keys, "log(n_classes)", "slope a")
    _report([r["classes_per_dim"] for r in rows], [r["c_100"] for r in rows],
            keys, "classes / embed_dim", "c at 100% data")


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

def plot_convergence_check(logs):
    """Appendix figure: training loss + curvature vs epoch at the 1% and 10%
    ablation, per dataset. Lets reviewers verify that the optimizer
    converged at low-data fractions and isn't caught in a partially-trained
    regime that would explain the early-data curvature peaks."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    for col_idx, frac in enumerate([0.01, 0.10]):
        ax_loss, ax_curv = axes[0, col_idx], axes[1, col_idx]
        for ds in DATASETS:
            key = ("bayesian", ds, frac)
            if key not in logs:
                continue
            df = logs[key]
            ax_loss.plot(df["epoch"], df["train_loss"],
                         color=DATASET_COLORS[ds], linewidth=1.2,
                         label=DATASET_LABELS[ds] if col_idx == 0 else None)
            ax_curv.plot(df["epoch"], df["curvature"],
                         color=DATASET_COLORS[ds], linewidth=1.2)
        ax_loss.set_title(f"Ablation = {int(frac*100)}\\% data")
        ax_loss.set_ylabel("Train loss")
        ax_loss.grid(True, alpha=0.3)
        ax_curv.set_xlabel("Epoch")
        ax_curv.set_ylabel("Learned curvature $c$")
        ax_curv.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "convergence_check.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "convergence_check.png", bbox_inches="tight")
    print("  Saved convergence_check.pdf/png")
    plt.close(fig)


def print_accuracy_at_100(evals, logs):
    """Print a single combined table: c at 100%, top-1 accuracy, top-5 accuracy.
    Verifies the model is actually learning (not just collapsing curvature
    on an undertrained model). Also emits the LaTeX form."""
    print("\n" + "=" * 80)
    print("  ACCURACY AT 100% DATA (with converged curvature)")
    print("=" * 80)
    print(f"{'Dataset':<18} {'c_100%':>8} {'Top-1':>8} {'Top-5':>8} {'N_test':>8}")
    print("-" * 80)
    rows = []
    for ds in DATASETS:
        c_key = ("bayesian", ds, 1.0)
        if c_key not in logs or c_key not in evals:
            continue
        c_val = logs[c_key]["curvature"].iloc[-1]
        e = evals[c_key]
        top1 = float(e["top1"])
        top5 = float(e["top5"])
        n_test = int(e["n_test"])
        rows.append((ds, c_val, top1, top5, n_test))
        print(f"{DATASET_LABELS[ds]:<18} {c_val:>8.3f} {top1:>7.2f}% "
              f"{top5:>7.2f}% {n_test:>8d}")

    # LaTeX
    print("\n  LaTeX (accuracy table for paper):")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & $\bm{c_{100\%}}$ & \textbf{Top-1 (\%)} & "
          r"\textbf{Top-5 (\%)} & $\bm{N_{\text{test}}}$ \\")
    print(r"\midrule")
    for ds, c, t1, t5, n in rows:
        print(f"{DATASET_LABELS[ds]} & {c:.2f} & {t1:.2f} & {t5:.2f} & {n} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


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
        hier_data = plot_curvature_vs_hierarchy(logs)
        fit_results = plot_log_n_fits(logs)
        plot_fan_comparison(logs)
        plot_convergence_check(logs)

        if fit_results:
            print("\n  c = a*log(N) + b fit results (95% CI on slope):")
            for ds, res in fit_results.items():
                lo, hi = res["ci"]
                print(f"    {DATASET_LABELS.get(ds, ds):<18} "
                      f"a={res['slope']:+.3f}  "
                      f"95% CI=[{lo:+.3f}, {hi:+.3f}]  "
                      f"R²={res['r2']:.3f}  "
                      f"p={res['p_value']:.3f}")

        # Mechanical pattern classification
        report_pattern_classification(logs)

        # Cross-dataset regressions on n_classes
        report_n_classes_regressions(logs, fit_results)

        # Slope-vs-hierarchy with/without AwA2 (using already-loaded fits)
        if hier_data is not None and fit_results:
            ds_keys, depths, branchings, _ = hier_data
            slopes = [fit_results[k]["slope"] for k in ds_keys if k in fit_results]
            ds_keys_match = [k for k in ds_keys if k in fit_results]
            depths_match = [d for d, k in zip(depths, ds_keys) if k in fit_results]
            branchings_match = [b for b, k in zip(branchings, ds_keys) if k in fit_results]

            print("\n  Slope a vs taxonomy descriptors (with vs without AwA2):")
            r2_all, r2_no = _r2_with_and_without_awa2(
                depths_match, slopes, ds_keys_match)
            print(f"    a vs depth:     R^2 all={r2_all:.3f}  "
                  f"R^2 ex-AwA2={r2_no:.3f}")
            r2_all, r2_no = _r2_with_and_without_awa2(
                branchings_match, slopes, ds_keys_match)
            print(f"    a vs branching: R^2 all={r2_all:.3f}  "
                  f"R^2 ex-AwA2={r2_no:.3f}")
    else:
        print("  [WARN] No Bayesian training logs found for curvature analysis")

    print_curvature_summary(logs)
    print_accuracy_at_100(evals, logs)
    print_accuracy_tables(evals)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
