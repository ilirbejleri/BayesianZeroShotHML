"""
One-shot fix for iNaturalist class names.

The original prepare_inaturalist used placeholder names ("species_0", ...)
which caused all GloVe embeddings to collapse to the same vector, leading
to a degenerate alignment loss.

This script reproduces the deterministic 500-class subsample (seed=42) from
the original 10000-class iNat 2021 mini dataset, looks up each kept class
in the raw directory listing, parses its full taxonomy, and rewrites
data/inaturalist/{class_names.json, class_embeddings.npy} accordingly.

features.npy and labels.npy are unchanged — they were already keyed to the
correct subsample of classes by index.
"""
import os
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from prepare_dataset import load_glove, class_name_to_glove

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "inaturalist_raw" / "2021_train_mini"
OUT_DIR = DATA_ROOT / "inaturalist"
MAX_CLASSES = 500

if not RAW_DIR.exists():
    raise FileNotFoundError(f"{RAW_DIR} not found")
if not OUT_DIR.exists():
    raise FileNotFoundError(f"{OUT_DIR} not found")

# 1) Get all class directory names sorted (matches torchvision INaturalist ordering)
all_dirs = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())
print(f"Found {len(all_dirs)} class directories (expected 10000)")

# 2) Reproduce the deterministic 500-class sample (seed=42) from sorted indices
rng = np.random.RandomState(42)
keep_classes = sorted(rng.choice(len(all_dirs), size=MAX_CLASSES, replace=False).tolist())
print(f"Sampled {len(keep_classes)} classes (sorted indices: {keep_classes[:3]}...)")

# 3) Get the directory names for the kept classes (in the sorted order — matches
#    the old_to_new mapping in the original prep code)
kept_dir_names = [all_dirs[i] for i in keep_classes]
print(f"First kept dir name: {kept_dir_names[0]}")
print(f"Last kept dir name:  {kept_dir_names[-1]}")


def dirname_to_taxonomy(dirname: str) -> str:
    """Strip leading '<idx>_' prefix and replace underscores with spaces."""
    parts = dirname.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        taxa_str = parts[1]
    else:
        taxa_str = dirname
    return taxa_str.replace("_", " ").lower()


# 4) Build new class names from the taxonomy
class_names = [dirname_to_taxonomy(d) for d in kept_dir_names]
print(f"Examples:")
for i in [0, 100, 250, 499]:
    print(f"  [{i}] {class_names[i]}")

# 5) Build new GloVe embeddings
print("Loading GloVe...")
glove = load_glove()
print("Building embeddings...")
embeddings = np.stack([class_name_to_glove(n, glove) for n in class_names]).astype(np.float32)
n_zero = (embeddings.sum(1) == 0).sum()
print(f"Embeddings: shape={embeddings.shape}, all-zero (no GloVe match): {n_zero}/{len(class_names)}")

# 6) Backup old files and write new ones
for f in ["class_names.json", "class_embeddings.npy"]:
    src = OUT_DIR / f
    if src.exists():
        bak = OUT_DIR / (f + ".placeholder.bak")
        src.rename(bak)
        print(f"  backed up {f} -> {bak.name}")

with open(OUT_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)
np.save(OUT_DIR / "class_embeddings.npy", embeddings)
print(f"Wrote new class_names.json and class_embeddings.npy")

# 7) Sanity check: embedding norms should differ across classes
norms = np.linalg.norm(embeddings, axis=1)
print(f"Norm stats: min={norms.min():.3f} max={norms.max():.3f} mean={norms.mean():.3f}")
print(f"Distinct norms (rounded): {len(np.unique(np.round(norms, 3)))}")

# Verify against existing labels
labels = np.load(OUT_DIR / "labels.npy")
print(f"\nVerify: features-side has {len(np.unique(labels))} unique labels, "
      f"new class_names has {len(class_names)} entries — must match")
assert len(class_names) == len(np.unique(labels)), "MISMATCH!"
print("OK")
