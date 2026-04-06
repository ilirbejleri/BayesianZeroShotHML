"""
Multi-dataset data loading for the curvature study (Paper 2).

Loads pre-extracted features and embeddings from a unified format produced
by prepare_dataset.py:

    data/<dataset_name>/
        features.npy          (N, 2048) float32
        labels.npy            (N,) int64, 0-indexed
        class_names.json      list of class name strings
        class_embeddings.npy  (C, D) float32  — attributes or GloVe
        splits.json           {"seen": [...], "unseen": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# All datasets supported by this loader
SUPPORTED_DATASETS = [
    "awa2", "cifar100", "cub200", "sun397",
    "stanford_cars", "tiered_imagenet", "inaturalist",
]


class UnifiedDataset(Dataset):
    """Dataset backed by pre-extracted ResNet-101 features and semantic embeddings."""

    def __init__(self, dataset_name: str, data_root: str = "data") -> None:
        base = Path(data_root) / dataset_name
        if not base.exists():
            raise FileNotFoundError(
                f"{base} not found. Run:\n"
                f"  python prepare_dataset.py --dataset {dataset_name}\n"
            )

        self.name = dataset_name
        self.features = np.load(base / "features.npy").astype(np.float32)
        self.labels = np.load(base / "labels.npy").astype(np.int64)

        with open(base / "class_names.json") as f:
            self.class_names: List[str] = json.load(f)

        self.attributes = np.load(base / "class_embeddings.npy").astype(np.float32)

        with open(base / "splits.json") as f:
            splits = json.load(f)
        self.seen_classes: List[int] = splits["seen"]
        self.unseen_classes: List[int] = splits["unseen"]

        print(f"[{dataset_name}] {len(self)} samples, "
              f"{len(self.class_names)} classes "
              f"({len(self.seen_classes)} seen / {len(self.unseen_classes)} unseen), "
              f"embed_dim={self.attributes.shape[1]}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        feat = torch.from_numpy(self.features[idx])
        label = int(self.labels[idx])
        attr = torch.from_numpy(self.attributes[label])
        return feat, label, attr


# ---------------------------------------------------------------------------
# Split helpers (same logic as data.py, driven by splits.json)
# ---------------------------------------------------------------------------

def build_zsl_splits(dataset: UnifiedDataset) -> Tuple[Subset, Subset]:
    """Return (train_subset, test_subset) from the dataset's seen/unseen split."""
    seen_set = set(dataset.seen_classes)
    unseen_set = set(dataset.unseen_classes)

    train_indices = [i for i, lbl in enumerate(dataset.labels) if lbl in seen_set]
    test_indices = [i for i, lbl in enumerate(dataset.labels) if lbl in unseen_set]
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def ablate_training_set(
    train_subset: Subset, fraction: float, seed: int = 42,
) -> Subset:
    """Sub-sample the training split to ``fraction`` of its original size."""
    assert 0.0 < fraction <= 1.0
    rng = np.random.RandomState(seed)
    n = len(train_subset)
    keep = max(1, int(n * fraction))
    chosen = rng.choice(n, size=keep, replace=False)
    return Subset(train_subset, chosen.tolist())


def load_class_embeddings(
    dataset: UnifiedDataset, normalize: bool = True,
) -> torch.Tensor:
    """Return class-level semantic embeddings, optionally L2-normalised."""
    emb = torch.from_numpy(dataset.attributes).float()
    if normalize:
        emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return emb


def build_seen_label_map(dataset: UnifiedDataset) -> Dict[int, int]:
    """Map dataset-level label indices to contiguous 0..n_seen-1."""
    mapping = {}
    for rank, cls_idx in enumerate(sorted(dataset.seen_classes)):
        mapping[cls_idx] = rank
    return mapping


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    dataset_name: str,
    data_root: str = "data",
    batch_size: int = 128,
    ablation_fraction: float = 1.0,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """Convenience function returning train/test loaders and class embeddings.

    Returns
    -------
    train_loader, test_loader, class_embeddings  (C, D)
    """
    dataset = UnifiedDataset(dataset_name, data_root)
    train_sub, test_sub = build_zsl_splits(dataset)

    if ablation_fraction < 1.0:
        train_sub = ablate_training_set(train_sub, ablation_fraction, seed=seed)

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_sub, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    class_emb = load_class_embeddings(dataset)
    return train_loader, test_loader, class_emb
