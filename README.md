# Bayesian Zero-Shot Learning in Hyperbolic Space

This repository contains code and papers for two related research projects on learnable curvature in hyperbolic neural networks.

## Paper 1: Bayesian Hyperbolic Embeddings for Extreme Uncertainty in Zero-Shot Learning

Maps visual features to Wrapped Normal distributions on the Poincare ball for zero-shot classification on AwA2. Key observation: the learned curvature parameter scales monotonically as c ~ log(N) with training set size on this single dataset — an emergent, unforced relationship discovered through gradient descent.

**Status:** Complete. See [`paper/main.pdf`](paper/main.pdf).

## Paper 2: An Empirical Study of Learned Poincare Curvature in Hyperbolic ZSL

Tests whether the c ~ log(N) scaling observed on AwA2 generalizes across diverse image classification datasets. **It does not.** Across six datasets, the per-dataset slope of c vs. log(N) varies from +1.31 (AwA2, R²=0.95) to **-0.17** (SUN397, R²=0.86), changing sign across datasets. Three reproducible response patterns emerge:

1. **Monotonic increase** (AwA2): curvature climbs from 1.18 to 6.63
2. **Rises then plateaus** (CUB-200, Stanford Cars, CIFAR-100): plateaus around c ∈ [1.6, 2.3]
3. **Peaks then declines** (SUN397, iNaturalist, tiered-ImageNet): curvature decreases as more data arrives

The strongest predictor of which pattern a dataset exhibits is the **class-embedding modality** (per-class binary attributes vs. GloVe word vectors), not taxonomy depth. The paper argues learned curvature reports a dataset's available semantic structure as seen through its embedding modality.

**Status:** 6 of 7 datasets complete; tiered-ImageNet still training. See [`paper2/main.pdf`](paper2/main.pdf).

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
scripts/
  train_job.sh / train_multi_job.sh    # SLURM training jobs
  ablation_sweep.sh / multi_sweep.sh   # Submit full ablation sweeps
  extract_features_job.sh              # Generic feature extraction job
  extract_tiered_imagenet.sh           # tiered-ImageNet extraction with TMPDIR optimization
  fix_inaturalist_names.py             # One-shot fix for iNaturalist class name embeddings
  pace-connect                         # SSH ControlMaster helper for PACE ICE
paper/                 # Paper 1 (tex, figures, pdf)
paper2/                # Paper 2 (tex, figures, pdf)
cluster_instructions.md  # PACE ICE workflow notes
```

## Datasets

| Dataset | Classes | Embedding | Embed dim | Final c (100% data) | Status |
|---------|---------|-----------|-----------|---------------------|--------|
| AwA2 | 50 | per-class attrs | 85 | **6.63** | Done |
| CUB-200 | 200 | GloVe | 300 | 2.01 | Done |
| CIFAR-100 | 100 | GloVe | 300 | 1.81 | Done |
| Stanford Cars | 196 | GloVe | 300 | 1.58 | Done |
| iNaturalist | 500 | GloVe (taxonomy) | 300 | 1.08 | Done |
| SUN397 | 397 | GloVe | 300 | **0.85** | Done |
| tiered-ImageNet | 608 | GloVe (WordNet) | 300 | (in progress) | Training |
