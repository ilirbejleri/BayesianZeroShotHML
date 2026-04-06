"""
Data loading and split construction for Zero-Shot / Few-Shot learning on AwA2.

Animals with Attributes 2 (AwA2) provides 37,322 images across 50 animal
classes, each annotated with 85 continuous semantic attributes.  The canonical
ZSL evaluation uses a *proposed split* (PS) of 40 seen / 10 unseen classes
following Xian et al. (2019).

This module handles:
  1. Loading pre-extracted visual features (ResNet-101 2048-d) shipped with AwA2
     *or* raw images when fine-tuning a backbone.
  2. Constructing the seen/unseen class splits.
  3. Building few-shot subsets at configurable data-ablation ratios for the
     extreme-low-data experiments.
  4. Providing WordNet-derived class embeddings aligned with the semantic
     hierarchy used by the hyperbolic loss.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ---------------------------------------------------------------------------
# AwA2 proposed-split class lists (Xian et al., 2019)
# ---------------------------------------------------------------------------

# fmt: off
SEEN_CLASSES: List[str] = [
    "antelope", "grizzly+bear", "killer+whale", "beaver", "dalmatian",
    "persian+cat", "horse", "german+shepherd", "blue+whale", "siamese+cat",
    "skunk", "mole", "tiger", "hippopotamus", "leopard",
    "moose", "spider+monkey", "humpback+whale", "elephant", "gorilla",
    "ox", "fox", "sheep", "seal", "chimpanzee",
    "hamster", "squirrel", "rhinoceros", "rabbit", "bat",
    "giraffe", "wolf", "chihuahua", "rat", "weasel",
    "otter", "buffalo", "zebra", "giant+panda", "deer",
]

UNSEEN_CLASSES: List[str] = [
    "bobcat", "pig", "collie", "cow", "dolphin",
    "polar+bear", "lion", "mouse", "raccoon", "walrus",
]
# fmt: on

ALL_CLASSES: List[str] = sorted(SEEN_CLASSES + UNSEEN_CLASSES)
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(ALL_CLASSES)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AwA2Features(Dataset):
    """Dataset backed by the pre-extracted ResNet-101 features shipped with AwA2.

    Expected directory layout (``root``):
        Animals_with_Attributes2/
            Features/
                ResNet101/
                    AwA2-features.txt   (N x 2048)
                    AwA2-filenames.txt  (N,)
                    AwA2-labels.txt     (N,)

    Class names are derived from AwA2-filenames.txt (each line is
    ``classname/image.jpg``).  The continuous attribute matrix is loaded
    from ``predicate-matrix-continuous.txt`` if present in ``root``,
    otherwise it must be supplied via the ``attributes_path`` argument.
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        attributes_path: Optional[str | pathlib.Path] = None,
    ) -> None:
        root = pathlib.Path(root)
        feat_dir = root / "Features" / "ResNet101"

        # Load pre-extracted features
        self.features = np.loadtxt(feat_dir / "AwA2-features.txt", dtype=np.float32)
        raw_labels = np.loadtxt(feat_dir / "AwA2-labels.txt", dtype=np.int64)
        # AwA2 labels are 1-indexed
        self.labels = raw_labels - 1

        # Derive class names from filenames (format: "classname_NNNNN.jpg")
        with open(feat_dir / "AwA2-filenames.txt") as f:
            filenames = [line.strip() for line in f]

        # Build ordered class name list: map each label to its class name
        label_to_name: Dict[int, str] = {}
        for fname, lbl in zip(filenames, self.labels):
            # "grizzly+bear_10001.jpg" → "grizzly+bear"
            class_name = fname.rsplit("_", 1)[0]
            label_to_name[int(lbl)] = class_name
        self.class_names = [label_to_name[i] for i in range(len(label_to_name))]

        # Continuous attribute matrix (50 x 85)
        attr_file = self._resolve_attributes_path(root, attributes_path)
        if attr_file is None:
            raise FileNotFoundError(
                "predicate-matrix-continuous.txt not found. "
                "Searched in:\n"
                f"  {root / 'predicate-matrix-continuous.txt'}\n"
                f"  {root / 'predicates' / 'predicate-matrix-continuous.txt'}\n"
                f"  {root.parent / 'predicate-matrix-continuous.txt'}\n"
                "Download it with:\n"
                "  wget https://cvml.ista.ac.at/AwA2/AwA2-base.zip\n"
                "  unzip -j AwA2-base.zip 'Animals_with_Attributes2/predicate-matrix-continuous.txt' "
                f"-d {root}\n"
                "Or pass --attributes_path explicitly."
            )
        self.attributes = np.loadtxt(attr_file, dtype=np.float32)

    @staticmethod
    def _resolve_attributes_path(
        root: pathlib.Path,
        explicit: Optional[str | pathlib.Path],
    ) -> Optional[pathlib.Path]:
        """Find the predicate-matrix file, checking several common locations."""
        if explicit is not None:
            p = pathlib.Path(explicit)
            if p.is_file():
                return p
        candidates = [
            root / "predicate-matrix-continuous.txt",
            root / "predicates" / "predicate-matrix-continuous.txt",
            root.parent / "predicate-matrix-continuous.txt",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        feat = torch.from_numpy(self.features[idx])
        label = int(self.labels[idx])
        attr = torch.from_numpy(self.attributes[label])
        return feat, label, attr


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _class_name_to_dataset_idx(class_names: List[str], target_names: List[str]) -> List[int]:
    """Map split class names to 0-based dataset label indices."""
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    return [name_to_idx[n] for n in target_names if n in name_to_idx]


def build_zsl_splits(
    dataset: AwA2Features,
) -> Tuple[Subset, Subset]:
    """Return (train_subset, test_subset) following the proposed split."""
    seen_idx = set(_class_name_to_dataset_idx(dataset.class_names, SEEN_CLASSES))
    unseen_idx = set(_class_name_to_dataset_idx(dataset.class_names, UNSEEN_CLASSES))

    train_indices = [i for i, lbl in enumerate(dataset.labels) if lbl in seen_idx]
    test_indices = [i for i, lbl in enumerate(dataset.labels) if lbl in unseen_idx]
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def ablate_training_set(
    train_subset: Subset,
    fraction: float,
    seed: int = 42,
) -> Subset:
    """Sub-sample the training split to ``fraction`` of its original size.

    Used for the data-ablation experiments (e.g. fraction=0.01 keeps 1% of
    training data to simulate extreme low-data regimes).
    """
    assert 0.0 < fraction <= 1.0
    rng = np.random.RandomState(seed)
    n = len(train_subset)
    keep = max(1, int(n * fraction))
    chosen = rng.choice(n, size=keep, replace=False)
    return Subset(train_subset, chosen.tolist())


# ---------------------------------------------------------------------------
# Semantic embeddings
# ---------------------------------------------------------------------------

def load_attribute_matrix(dataset: AwA2Features) -> torch.Tensor:
    """Return the (50, 85) continuous attribute matrix as a tensor."""
    return torch.from_numpy(dataset.attributes).float()


def load_class_embeddings(
    dataset: AwA2Features,
    normalize: bool = True,
) -> torch.Tensor:
    """Return class-level semantic embeddings (attribute vectors).

    Optionally L2-normalises each row so that cosine and dot-product
    comparisons are interchangeable.
    """
    emb = torch.from_numpy(dataset.attributes).float()
    if normalize:
        emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return emb


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    root: str,
    batch_size: int = 128,
    ablation_fraction: float = 1.0,
    num_workers: int = 4,
    seed: int = 42,
    attributes_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """Convenience function returning train/test loaders and class embeddings.

    Parameters
    ----------
    root : str
        Path to ``Animals_with_Attributes2/`` directory.
    batch_size : int
        Mini-batch size.
    ablation_fraction : float
        Fraction of the training set to keep (1.0 = full).
    num_workers : int
        Dataloader workers.
    seed : int
        Random seed for ablation sampling.
    attributes_path : str, optional
        Explicit path to predicate-matrix-continuous.txt.

    Returns
    -------
    train_loader, test_loader, class_embeddings
    """
    dataset = AwA2Features(root, attributes_path=attributes_path)
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
