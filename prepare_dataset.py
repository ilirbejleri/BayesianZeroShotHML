#!/usr/bin/env python3
"""
Prepare datasets for the curvature study (Paper 2).

Downloads data (where possible), extracts ResNet-101 features, builds
semantic embeddings (per-class attributes or GloVe), defines ZSL splits,
and saves everything in the unified format consumed by data_multi.py.

Usage:
    python prepare_dataset.py --dataset cifar100 --device cuda
    python prepare_dataset.py --dataset cub200 --device cuda
    python prepare_dataset.py --dataset awa2     # convert existing AwA2

The unified output format (per dataset) is:
    data/<name>/features.npy        (N, 2048) float32
    data/<name>/labels.npy          (N,) int64
    data/<name>/class_names.json    list of C strings
    data/<name>/class_embeddings.npy (C, D) float32
    data/<name>/splits.json         {"seen": [...], "unseen": [...]}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet101_Weights
from tqdm import tqdm


# ===================================================================
# Common utilities
# ===================================================================

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def extract_features(
    dataset: Dataset,
    batch_size: int = 64,
    device: str = "cuda",
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ResNet-101 features from an image dataset.

    Returns (features, labels) as numpy arrays.
    """
    model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model = model.to(device).eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    )

    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting ResNet-101 features"):
            images = batch[0].to(device)
            labels = batch[1]
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def load_glove(path: str = "data/glove/glove.6B.300d.txt") -> Dict[str, np.ndarray]:
    """Load GloVe word vectors into a dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"GloVe vectors not found at {path}.\n"
            "Download with:\n"
            "  mkdir -p data/glove\n"
            "  wget -P data/glove/ https://nlp.stanford.edu/data/glove.6B.zip\n"
            "  unzip data/glove/glove.6B.zip -d data/glove/\n"
        )
    print(f"Loading GloVe from {path}...")
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading GloVe", total=400000):
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if vec.shape[0] == 300:
                embeddings[word] = vec
    print(f"  Loaded {len(embeddings)} word vectors")
    return embeddings


def class_name_to_glove(name: str, glove: Dict[str, np.ndarray], dim: int = 300) -> np.ndarray:
    """Convert a class name to a GloVe embedding by averaging word vectors."""
    clean = name.lower().replace("_", " ").replace("+", " ").replace("-", " ").replace("/", " ")
    # Remove numbers (e.g., year from car models)
    words = [w for w in clean.split() if not w.isdigit()]
    vecs = [glove[w] for w in words if w in glove]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def save_unified(
    out_dir: str,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    embeddings: np.ndarray,
    seen: List[int],
    unseen: List[int],
) -> None:
    """Save dataset in the unified format."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "features.npy", features.astype(np.float32))
    np.save(out / "labels.npy", labels.astype(np.int64))
    np.save(out / "class_embeddings.npy", embeddings.astype(np.float32))

    with open(out / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    with open(out / "splits.json", "w") as f:
        json.dump({"seen": seen, "unseen": unseen}, f, indent=2)

    print(f"\nSaved to {out}:")
    print(f"  Features:   {features.shape}")
    print(f"  Classes:    {len(class_names)} ({len(seen)} seen / {len(unseen)} unseen)")
    print(f"  Embeddings: {embeddings.shape}")
    n_train = np.isin(labels, seen).sum()
    n_test = np.isin(labels, unseen).sum()
    print(f"  Train/test: {n_train} / {n_test} samples")


# ===================================================================
# AwA2 — convert existing data to unified format
# ===================================================================

def prepare_awa2(data_root: str = "data", device: str = "cpu", **kwargs) -> None:
    """Convert existing AwA2 pre-extracted features to unified format."""
    from data import AwA2Features, SEEN_CLASSES, UNSEEN_CLASSES

    awa2_dir = Path(data_root) / "Animals_with_Attributes2"
    if not awa2_dir.exists():
        raise FileNotFoundError(f"{awa2_dir} not found")

    dataset = AwA2Features(str(awa2_dir))

    # Map seen/unseen class names to indices
    name_to_idx = {n: i for i, n in enumerate(dataset.class_names)}
    seen = sorted([name_to_idx[n] for n in SEEN_CLASSES if n in name_to_idx])
    unseen = sorted([name_to_idx[n] for n in UNSEEN_CLASSES if n in name_to_idx])

    save_unified(
        out_dir=str(Path(data_root) / "awa2"),
        features=dataset.features,
        labels=dataset.labels,
        class_names=dataset.class_names,
        embeddings=dataset.attributes,
        seen=seen,
        unseen=unseen,
    )


# ===================================================================
# CIFAR-100
# ===================================================================

# Mapping: fine class name → superclass name
CIFAR100_FINE_TO_SUPER = {}
_CIFAR100_SUPER_MAP = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "computer_keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}
for _super, _fines in _CIFAR100_SUPER_MAP.items():
    for _fine in _fines:
        CIFAR100_FINE_TO_SUPER[_fine] = _super

# Hold out 4 superclasses (20 fine classes) as unseen — diverse selection
CIFAR100_UNSEEN_SUPERCLASSES = {
    "large_carnivores", "aquatic_mammals", "flowers", "vehicles_1",
}


def prepare_cifar100(
    data_root: str = "data",
    device: str = "cuda",
    batch_size: int = 256,
    num_workers: int = 4,
    glove: Optional[Dict] = None,
) -> None:
    """Prepare CIFAR-100 with ResNet-101 features and GloVe embeddings."""
    from torchvision.datasets import CIFAR100

    raw_dir = Path(data_root) / "cifar100_raw"
    out_dir = Path(data_root) / "cifar100"

    if (out_dir / "features.npy").exists():
        print("[cifar100] Already prepared, skipping. Delete data/cifar100/ to re-run.")
        return

    print("=== Preparing CIFAR-100 ===")
    train_ds = CIFAR100(str(raw_dir), train=True, download=True, transform=TRANSFORM)
    test_ds = CIFAR100(str(raw_dir), train=False, download=True, transform=TRANSFORM)
    combined = ConcatDataset([train_ds, test_ds])

    features, labels = extract_features(combined, batch_size, device, num_workers)
    class_names = train_ds.classes  # 100 fine-grained names

    if glove is None:
        glove = load_glove()

    embeddings = np.stack([
        class_name_to_glove(name, glove) for name in class_names
    ])

    # ZSL split by superclass
    seen, unseen = [], []
    for idx, name in enumerate(class_names):
        superclass = CIFAR100_FINE_TO_SUPER.get(name)
        if superclass in CIFAR100_UNSEEN_SUPERCLASSES:
            unseen.append(idx)
        else:
            seen.append(idx)

    save_unified(str(out_dir), features, labels, class_names, embeddings, seen, unseen)


# ===================================================================
# CUB-200-2011
# ===================================================================

def prepare_cub200(
    data_root: str = "data",
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    glove: Optional[Dict] = None,
) -> None:
    """Prepare CUB-200-2011 with ResNet-101 features.

    Uses 312-d per-class attributes if available, otherwise GloVe.
    Expects data at data/CUB_200_2011/ (download from Caltech).
    """
    cub_root = Path(data_root) / "CUB_200_2011"
    out_dir = Path(data_root) / "cub200"

    if (out_dir / "features.npy").exists():
        print("[cub200] Already prepared, skipping.")
        return

    if not cub_root.exists():
        raise FileNotFoundError(
            f"{cub_root} not found.\n"
            "Download CUB-200-2011 from:\n"
            "  https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz\n"
            f"Extract to {data_root}/CUB_200_2011/\n"
        )

    print("=== Preparing CUB-200-2011 ===")

    # Read class names
    with open(cub_root / "classes.txt") as f:
        class_entries = [line.strip().split() for line in f]
    class_names = [entry[1].split(".")[1] for entry in class_entries]

    # Read image paths and labels
    with open(cub_root / "images.txt") as f:
        image_entries = [line.strip().split() for line in f]
    with open(cub_root / "image_class_labels.txt") as f:
        label_entries = [line.strip().split() for line in f]

    image_paths = [cub_root / "images" / entry[1] for entry in image_entries]
    labels_1indexed = [int(entry[1]) for entry in label_entries]
    labels = np.array([l - 1 for l in labels_1indexed], dtype=np.int64)

    # Build a simple image dataset
    from PIL import Image

    class CUBImageDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.labels[idx]

    img_dataset = CUBImageDataset(image_paths, labels, TRANSFORM)
    features, extracted_labels = extract_features(img_dataset, batch_size, device, num_workers)

    # Try to load 312-d attributes
    attr_file = cub_root / "attributes" / "class_attribute_labels_continuous.txt"
    if attr_file.exists():
        embeddings = np.loadtxt(attr_file, dtype=np.float32)
        print(f"  Loaded 312-d attributes: {embeddings.shape}")
    else:
        print("  No attribute file found, using GloVe embeddings")
        if glove is None:
            glove = load_glove()
        embeddings = np.stack([class_name_to_glove(n, glove) for n in class_names])

    # ZSL split: use every 4th class as unseen (deterministic, 50 unseen / 150 seen)
    unseen = list(range(0, 200, 4))  # indices 0, 4, 8, ..., 196
    seen = [i for i in range(200) if i not in set(unseen)]

    save_unified(str(out_dir), features, extracted_labels, class_names, embeddings, seen, unseen)


# ===================================================================
# SUN397
# ===================================================================

def prepare_sun397(
    data_root: str = "data",
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    glove: Optional[Dict] = None,
) -> None:
    """Prepare SUN397 with ResNet-101 features and GloVe embeddings."""
    out_dir = Path(data_root) / "sun397"

    if (out_dir / "features.npy").exists():
        print("[sun397] Already prepared, skipping.")
        return

    try:
        from torchvision.datasets import SUN397 as SUN397Dataset
    except ImportError:
        raise ImportError("SUN397 requires torchvision >= 0.13")

    raw_dir = Path(data_root) / "sun397_raw"
    print("=== Preparing SUN397 ===")

    dataset = SUN397Dataset(str(raw_dir), download=True, transform=TRANSFORM)

    features, labels = extract_features(dataset, batch_size, device, num_workers)

    # Get class names from the dataset
    class_names = [str(c).replace("/", "_").lstrip("_") for c in dataset.classes]

    if glove is None:
        glove = load_glove()

    embeddings = np.stack([class_name_to_glove(n, glove) for n in class_names])

    # ZSL split: last 20% of classes (alphabetically) as unseen
    n_classes = len(class_names)
    n_unseen = n_classes // 5
    unseen = list(range(n_classes - n_unseen, n_classes))
    seen = list(range(n_classes - n_unseen))

    save_unified(str(out_dir), features, labels, class_names, embeddings, seen, unseen)


# ===================================================================
# Stanford Cars
# ===================================================================

def prepare_stanford_cars(
    data_root: str = "data",
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    glove: Optional[Dict] = None,
) -> None:
    """Prepare Stanford Cars with ResNet-101 features and GloVe embeddings.

    Expects Kaggle download (jessicali9530) at data/stanford_cars_raw/ with:
        car_devkit/devkit/cars_train_annos.mat   (annotations)
        car_devkit/devkit/cars_meta.mat          (class names)
        cars_train/cars_train/*.jpg              (8144 training images)
    """
    raw_dir = Path(data_root) / "stanford_cars_raw"
    out_dir = Path(data_root) / "stanford_cars"

    if (out_dir / "features.npy").exists():
        print("[stanford_cars] Already prepared, skipping.")
        return

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} not found.\n"
            "Download Stanford Cars from Kaggle:\n"
            "  https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset\n"
            f"Extract to {raw_dir}/\n"
        )

    print("=== Preparing Stanford Cars ===")
    from PIL import Image
    from scipy.io import loadmat

    # --- Find annotation .mat file ---
    anno_path = None
    for candidate in [
        "car_devkit/devkit/cars_train_annos.mat",
        "devkit/cars_train_annos.mat",
        "cars_train_annos.mat",
    ]:
        if (raw_dir / candidate).exists():
            anno_path = raw_dir / candidate
            break

    if anno_path is None:
        raise FileNotFoundError(
            f"cars_train_annos.mat not found in {raw_dir}.\n"
            "Expected at: {raw_dir}/car_devkit/devkit/cars_train_annos.mat"
        )

    # --- Find class names .mat file ---
    meta_path = None
    for candidate in [
        "car_devkit/devkit/cars_meta.mat",
        "devkit/cars_meta.mat",
        "cars_meta.mat",
    ]:
        if (raw_dir / candidate).exists():
            meta_path = raw_dir / candidate
            break

    # --- Find image directory (handle double-nesting) ---
    img_dir = None
    for candidate in [
        "cars_train/cars_train",  # double-nested (Kaggle format)
        "cars_train",
        "train",
    ]:
        d = raw_dir / candidate
        if d.exists() and any(d.glob("*.jpg")):
            img_dir = d
            break

    if img_dir is None:
        raise FileNotFoundError(
            f"No image directory with .jpg files found in {raw_dir}.\n"
            "Expected at: {raw_dir}/cars_train/cars_train/*.jpg"
        )

    print(f"  Annotations: {anno_path}")
    print(f"  Images: {img_dir}")

    # --- Load annotations ---
    mat = loadmat(str(anno_path), squeeze_me=True)
    annos = mat["annotations"]

    image_paths = []
    raw_labels = []
    for anno in annos:
        fname = str(anno["fname"])
        label = int(anno["class"])  # 1-indexed
        img_path = img_dir / fname
        if img_path.exists():
            image_paths.append(img_path)
            raw_labels.append(label)

    print(f"  Matched {len(image_paths)} / {len(annos)} images")

    # --- Load class names ---
    if meta_path:
        meta = loadmat(str(meta_path), squeeze_me=True)
        all_class_names = [str(n) for n in meta["class_names"]]  # 196 names
        print(f"  Class names: {len(all_class_names)} (e.g. '{all_class_names[0]}')")
    else:
        all_class_names = [f"car_class_{i}" for i in range(196)]

    # --- Remap labels to 0-indexed ---
    # raw_labels are 1-indexed (1..196)
    labels = np.array([l - 1 for l in raw_labels], dtype=np.int64)
    class_names = all_class_names  # already ordered 0..195

    # --- Build dataset for feature extraction ---
    class AnnotatedImageDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.labels[idx]

    dataset = AnnotatedImageDataset(image_paths, labels, TRANSFORM)
    features, _ = extract_features(dataset, batch_size, device, num_workers)

    n_classes = len(class_names)
    print(f"  Total classes: {n_classes}")

    if glove is None:
        glove = load_glove()

    embeddings = np.stack([class_name_to_glove(n, glove) for n in class_names])

    # ZSL split: last ~25% of classes as unseen
    n_unseen = n_classes // 4  # 49
    unseen = list(range(n_classes - n_unseen, n_classes))
    seen = list(range(n_classes - n_unseen))

    save_unified(str(out_dir), features, labels, class_names, embeddings, seen, unseen)


# ===================================================================
# tiered-ImageNet
# ===================================================================

def prepare_tiered_imagenet(
    data_root: str = "data",
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    glove: Optional[Dict] = None,
) -> None:
    """Prepare tiered-ImageNet.

    Expects pre-processed pkl/npz files or an ImageFolder structure at
    data/tiered_imagenet_raw/.  The standard source is:
      https://github.com/yaoyao-liu/mini-imagenet-tools (tiered variant)

    If pre-extracted features exist as .npz, loads directly.
    Otherwise extracts from images.
    """
    raw_dir = Path(data_root) / "tiered_imagenet_raw"
    out_dir = Path(data_root) / "tiered_imagenet"

    if (out_dir / "features.npy").exists():
        print("[tiered_imagenet] Already prepared, skipping.")
        return

    print("=== Preparing tiered-ImageNet ===")

    # Check for pre-extracted numpy files (common distribution format)
    npz_train = raw_dir / "train_features.npz"
    npz_test = raw_dir / "test_features.npz"

    if npz_train.exists() and npz_test.exists():
        print("  Loading from pre-extracted .npz files")
        train_data = np.load(npz_train)
        test_data = np.load(npz_test)
        features = np.concatenate([train_data["features"], test_data["features"]])
        labels = np.concatenate([train_data["labels"], test_data["labels"]])
        if "class_names" in train_data:
            class_names = list(train_data["class_names"])
        else:
            n_classes = labels.max() + 1
            class_names = [f"class_{i}" for i in range(n_classes)]
        train_classes = set(np.unique(train_data["labels"]).tolist())
        test_classes = set(np.unique(test_data["labels"]).tolist())
    else:
        # Try ImageFolder structure: train/, val/, test/ subdirectories
        splits_data = {}
        for split in ["train", "val", "test"]:
            split_dir = raw_dir / split
            if not split_dir.exists():
                continue
            from torchvision.datasets import ImageFolder
            ds = ImageFolder(str(split_dir), transform=TRANSFORM)
            feats, lbls = extract_features(ds, batch_size, device, num_workers)
            splits_data[split] = (feats, lbls, ds.classes)

        if not splits_data:
            raise FileNotFoundError(
                f"No data found in {raw_dir}.\n"
                "Download tiered-ImageNet and place train/test subdirectories there.\n"
            )

        all_class_names = splits_data[list(splits_data.keys())[0]][2]
        features = np.concatenate([v[0] for v in splits_data.values()])
        labels = np.concatenate([v[1] for v in splits_data.values()])
        class_names = all_class_names
        train_classes = set(np.unique(splits_data["train"][1]).tolist()) if "train" in splits_data else set()
        test_classes = set(np.unique(splits_data["test"][1]).tolist()) if "test" in splits_data else set()

    n_classes = len(class_names) if class_names else (labels.max() + 1)
    if not class_names:
        class_names = [f"class_{i}" for i in range(n_classes)]

    if glove is None:
        glove = load_glove()

    embeddings = np.stack([class_name_to_glove(n, glove) for n in class_names])

    # Use the natural train/test split if available
    if train_classes and test_classes:
        seen = sorted(train_classes)
        unseen = sorted(test_classes)
    else:
        n_unseen = n_classes // 4
        unseen = list(range(n_classes - n_unseen, n_classes))
        seen = list(range(n_classes - n_unseen))

    save_unified(str(out_dir), features, labels, class_names, embeddings, seen, unseen)


# ===================================================================
# iNaturalist
# ===================================================================

def prepare_inaturalist(
    data_root: str = "data",
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    glove: Optional[Dict] = None,
    max_classes: int = 500,
) -> None:
    """Prepare iNaturalist 2021 mini.

    Uses torchvision.datasets.INaturalist if available, otherwise expects
    an ImageFolder structure at data/inaturalist_raw/.
    """
    raw_dir = Path(data_root) / "inaturalist_raw"
    out_dir = Path(data_root) / "inaturalist"

    if (out_dir / "features.npy").exists():
        print("[inaturalist] Already prepared, skipping.")
        return

    print("=== Preparing iNaturalist ===")

    # Try torchvision INaturalist
    try:
        from torchvision.datasets import INaturalist
        dataset = INaturalist(
            str(raw_dir), version="2021_train_mini",
            download=True, transform=TRANSFORM,
        )
        features, labels = extract_features(dataset, batch_size, device, num_workers)

        # Subsample to max_classes if too many
        unique_labels = np.unique(labels)
        if len(unique_labels) > max_classes:
            rng = np.random.RandomState(42)
            keep_classes = set(rng.choice(unique_labels, size=max_classes, replace=False).tolist())
            mask = np.isin(labels, list(keep_classes))
            features = features[mask]
            labels = labels[mask]
            # Remap labels to 0..max_classes-1
            old_to_new = {old: new for new, old in enumerate(sorted(keep_classes))}
            labels = np.array([old_to_new[l] for l in labels], dtype=np.int64)
            unique_labels = np.unique(labels)

        n_classes = len(unique_labels)
        class_names = [f"species_{i}" for i in range(n_classes)]

    except Exception:
        # Fall back to ImageFolder
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"{raw_dir} not found.\n"
                "Download iNaturalist 2021 mini or place images in:\n"
                f"  {raw_dir}/<class_name>/<image>.jpg\n"
            )

        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(str(raw_dir), transform=TRANSFORM)

        # Subsample classes if too many
        if len(dataset.classes) > max_classes:
            rng = np.random.RandomState(42)
            keep_idx = set(rng.choice(len(dataset.classes), size=max_classes, replace=False).tolist())
            keep_samples = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl in keep_idx]
            from torch.utils.data import Subset
            dataset = Subset(dataset, keep_samples)

        features, labels = extract_features(dataset, batch_size, device, num_workers)

        # Remap labels
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        old_to_new = {old: new for new, old in enumerate(sorted(unique_labels))}
        labels = np.array([old_to_new[l] for l in labels], dtype=np.int64)
        class_names = [f"species_{i}" for i in range(n_classes)]

    if glove is None:
        glove = load_glove()

    embeddings = np.stack([class_name_to_glove(n, glove) for n in class_names])

    # ZSL split: hold out 20% of classes
    n_unseen = n_classes // 5
    rng = np.random.RandomState(42)
    all_indices = list(range(n_classes))
    rng.shuffle(all_indices)
    unseen = sorted(all_indices[:n_unseen])
    seen = sorted(all_indices[n_unseen:])

    save_unified(str(out_dir), features, labels, class_names, embeddings, seen, unseen)


# ===================================================================
# Registry and main
# ===================================================================

DATASET_PREPARERS = {
    "awa2": prepare_awa2,
    "cifar100": prepare_cifar100,
    "cub200": prepare_cub200,
    "sun397": prepare_sun397,
    "stanford_cars": prepare_stanford_cars,
    "tiered_imagenet": prepare_tiered_imagenet,
    "inaturalist": prepare_inaturalist,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for curvature study")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_PREPARERS.keys()) + ["all"],
                        help="Dataset to prepare")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for feature extraction (auto|cpu|mps|cuda)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--glove_path", type=str,
                        default="data/glove/glove.6B.300d.txt")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Pre-load GloVe if any dataset needs it
    glove = None
    datasets_needing_glove = {"cifar100", "sun397", "stanford_cars",
                              "tiered_imagenet", "inaturalist"}
    targets = list(DATASET_PREPARERS.keys()) if args.dataset == "all" else [args.dataset]

    if any(d in datasets_needing_glove for d in targets):
        glove = load_glove(args.glove_path)

    for name in targets:
        print(f"\n{'='*60}")
        print(f"  Preparing: {name}")
        print(f"{'='*60}")
        DATASET_PREPARERS[name](
            data_root=args.data_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            glove=glove,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
