#!/usr/bin/env python3
"""
Evaluation for Bayesian Hyperbolic Zero-Shot Learning.

Computes Top-1 and Top-5 accuracy on *unseen* classes by:
  1. Embedding all class attribute vectors (seen + unseen) on the Poincaré ball.
  2. For each test image, computing geodesic distances to every unseen-class
     prototype and ranking by proximity.

For the Bayesian model, we marginalise predictions over K Monte-Carlo samples
from the visual encoder's wrapped-normal posterior.

Usage:
    python eval.py --data_root ./data/Animals_with_Attributes2 \
                   --checkpoint checkpoints/bayesian_abl1.0_dim64_ep100.pt \
                   --model bayesian --n_samples 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from data import (
    AwA2Features, build_zsl_splits, load_class_embeddings,
    SEEN_CLASSES, UNSEEN_CLASSES,
)
from models import BayesianHyperbolicZSL, EuclideanBaselineZSL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ZSL model on unseen classes")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--attributes_path", type=str, default=None,
                    help="Path to predicate-matrix-continuous.txt (if not inside data_root)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", choices=["bayesian", "euclidean"], default="bayesian")
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--n_samples", type=int, default=10,
                    help="MC samples for Bayesian prediction (ignored for euclidean)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_csv", type=str, default=None,
                    help="Path to write results CSV")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_bayesian(
    model: BayesianHyperbolicZSL,
    test_loader,
    unseen_attrs: torch.Tensor,
    unseen_label_set: set,
    label_to_unseen_idx: dict,
    device: torch.device,
    n_samples: int = 10,
) -> dict:
    """Evaluate the Bayesian hyperbolic model with MC marginalisation."""
    model.eval()
    ball = model.ball

    # Embed unseen class prototypes
    class_emb = model.embed_classes(unseen_attrs.to(device))  # (C_unseen, d)
    n_unseen = class_emb.shape[0]

    correct_1 = 0
    correct_5 = 0
    total = 0

    for features, labels, _ in tqdm(test_loader, desc="Eval (Bayesian)"):
        features = features.to(device)
        B = features.shape[0]

        # Draw n_samples from the posterior for each image
        mu, logvar, z_samples = model.visual_encoder(features, n_samples=n_samples)
        # z_samples: (n_samples, B, d)

        # Average distance across samples → (B, C_unseen)
        avg_dists = torch.zeros(B, n_unseen, device=device)
        for s in range(n_samples):
            z_s = z_samples[s]  # (B, d)
            d = ball.dist(z_s.unsqueeze(1), class_emb.unsqueeze(0))  # (B, C_unseen)
            avg_dists += d
        avg_dists /= n_samples

        # Top-k predictions (smallest distance = best)
        _, topk_indices = avg_dists.topk(5, dim=1, largest=False)

        for i in range(B):
            true_label = int(labels[i])
            if true_label not in label_to_unseen_idx:
                continue
            true_unseen = label_to_unseen_idx[true_label]
            preds = topk_indices[i].tolist()
            if preds[0] == true_unseen:
                correct_1 += 1
            if true_unseen in preds:
                correct_5 += 1
            total += 1

    top1 = correct_1 / max(total, 1) * 100
    top5 = correct_5 / max(total, 1) * 100
    return {"top1": top1, "top5": top5, "total": total}


@torch.no_grad()
def evaluate_euclidean(
    model: EuclideanBaselineZSL,
    test_loader,
    unseen_attrs: torch.Tensor,
    unseen_label_set: set,
    label_to_unseen_idx: dict,
    device: torch.device,
) -> dict:
    """Evaluate the Euclidean baseline."""
    model.eval()
    class_emb = model.embed_classes(unseen_attrs.to(device))  # (C_unseen, d)

    correct_1 = 0
    correct_5 = 0
    total = 0

    for features, labels, _ in tqdm(test_loader, desc="Eval (Euclidean)"):
        features = features.to(device)
        B = features.shape[0]
        vis = model.visual_encoder(features)  # (B, d)
        dists = torch.cdist(vis, class_emb)   # (B, C_unseen)
        _, topk_indices = dists.topk(5, dim=1, largest=False)

        for i in range(B):
            true_label = int(labels[i])
            if true_label not in label_to_unseen_idx:
                continue
            true_unseen = label_to_unseen_idx[true_label]
            preds = topk_indices[i].tolist()
            if preds[0] == true_unseen:
                correct_1 += 1
            if true_unseen in preds:
                correct_5 += 1
            total += 1

    top1 = correct_1 / max(total, 1) * 100
    top5 = correct_5 / max(total, 1) * 100
    return {"top1": top1, "top5": top5, "total": total}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = AwA2Features(args.data_root, attributes_path=args.attributes_path)
    _, test_subset = build_zsl_splits(dataset)
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    class_emb = load_class_embeddings(dataset)
    attr_dim = class_emb.shape[1]

    # Build unseen-class index mapping
    unseen_label_set = set()
    label_to_unseen_idx = {}
    name_to_idx = {n: i for i, n in enumerate(dataset.class_names)}
    for rank, name in enumerate(UNSEEN_CLASSES):
        ds_idx = name_to_idx[name]
        unseen_label_set.add(ds_idx)
        label_to_unseen_idx[ds_idx] = rank

    unseen_ds_indices = [name_to_idx[n] for n in UNSEEN_CLASSES]
    unseen_attrs = class_emb[unseen_ds_indices]  # (10, attr_dim)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    if args.model == "bayesian":
        model = BayesianHyperbolicZSL(
            visual_dim=2048, attr_dim=attr_dim,
            hidden_dim=saved_args.get("hidden_dim", args.hidden_dim),
            embed_dim=saved_args.get("embed_dim", args.embed_dim),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        results = evaluate_bayesian(
            model, test_loader, unseen_attrs,
            unseen_label_set, label_to_unseen_idx,
            device, n_samples=args.n_samples,
        )
    else:
        model = EuclideanBaselineZSL(
            visual_dim=2048, attr_dim=attr_dim,
            hidden_dim=saved_args.get("hidden_dim", args.hidden_dim),
            embed_dim=saved_args.get("embed_dim", args.embed_dim),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        results = evaluate_euclidean(
            model, test_loader, unseen_attrs,
            unseen_label_set, label_to_unseen_idx, device,
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  Model:    {args.model}")
    print(f"  Ckpt:     {args.checkpoint}")
    print(f"  Samples:  {results['total']}")
    print(f"  Top-1:    {results['top1']:.2f}%")
    print(f"  Top-5:    {results['top5']:.2f}%")
    print(f"{'='*50}\n")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "checkpoint", "ablation", "top1", "top5", "n_test"])
            w.writerow([
                args.model,
                args.checkpoint,
                saved_args.get("ablation_fraction", "?"),
                f"{results['top1']:.2f}",
                f"{results['top5']:.2f}",
                results["total"],
            ])
        print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
