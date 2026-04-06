# Bayesian Zero-Shot Learning in Hyperbolic Space

This repository contains code and papers for two related research projects on learnable curvature in hyperbolic neural networks.

## Paper 1: Bayesian Hyperbolic Embeddings for Extreme Uncertainty in Zero-Shot Learning

Maps visual features to Wrapped Normal distributions on the Poincare ball for zero-shot classification on AwA2. Key finding: the learned curvature parameter scales as c ~ log(N) with training set size — an emergent, unforced relationship discovered through gradient descent.

**Status:** Complete. See [`paper/main.pdf`](paper/main.pdf).

## Paper 2: Learned Curvature as a Probe of Hierarchical Complexity

Tests whether the curvature-data relationship generalizes across datasets and correlates with ground-truth hierarchy metrics (tree depth, branching factor, Gromov delta-hyperbolicity).

Preliminary results on 3/7 datasets show that between-dataset curvature tracks taxonomy depth (AwA2 at depth 3 produces ~4x higher curvature than CIFAR-100 and Stanford Cars at depth 2), though within-dataset log(N) scaling varies in strength.

**Status:** In progress (3/7 datasets complete). See [`paper2/main.pdf`](paper2/main.pdf).

## Structure

```
models.py              # Bayesian hyperbolic + Euclidean baseline models
loss.py                # Geodesic alignment + KL loss
train.py / eval.py     # Paper 1 training and evaluation
train_multi.py / eval_multi.py  # Paper 2 multi-dataset training and evaluation
data.py / data_multi.py         # Data loaders
prepare_dataset.py     # Dataset download, feature extraction, ZSL splits
hierarchy.py           # Taxonomy trees and hierarchy metrics for all 7 datasets
analyze.py             # Paper 1 analysis and figures
analyze_curvature.py   # Paper 2 cross-dataset curvature analysis
scripts/               # SLURM job scripts for cluster training
paper/                 # Paper 1 (tex, figures, pdf)
paper2/                # Paper 2 (tex, figures, pdf)
```

## Datasets

| Dataset | Classes | Hierarchy Depth | Status |
|---------|---------|----------------|--------|
| AwA2 | 50 | 3 | Done (Paper 1) |
| CIFAR-100 | 100 | 2 | Done |
| Stanford Cars | 196 | 2 | Done |
| CUB-200 | 200 | 5 | Pending |
| SUN397 | 397 | 3 | Pending |
| tiered-ImageNet | 608 | 8 | Pending |
| iNaturalist | 500 | 7 | Pending |
