#!/usr/bin/env python3
"""
Evaluation for the curvature study (Paper 2).

Identical to eval.py but parameterised by --dataset to support all 7 datasets.

Usage:
    python eval_multi.py --dataset cifar100 \
        --checkpoint checkpoints/bayesian_cifar100_abl1.0_dim64_ep200.pt \
        --model bayesian --n_samples 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from data_multi import UnifiedDataset, build_zsl_splits, load_class_embeddings
from models import BayesianHyperbolicZSL, EuclideanBaselineZSL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ZSL model (multi-dataset)")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", choices=["bayesian", "euclidean"], default="bayesian")
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_csv", type=str, default=None)
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
def evaluate_bayesian(model, test_loader, unseen_attrs, label_to_unseen_idx,
                      device, n_samples=10):
    model.eval()
    ball = model.ball
    class_emb = model.embed_classes(unseen_attrs.to(device))
    n_unseen = class_emb.shape[0]

    correct_1 = correct_5 = total = 0

    for features, labels, _ in tqdm(test_loader, desc="Eval (Bayesian)"):
        features = features.to(device)
        B = features.shape[0]
        mu, logvar, z_samples = model.visual_encoder(features, n_samples=n_samples)

        avg_dists = torch.zeros(B, n_unseen, device=device)
        for s in range(n_samples):
            z_s = z_samples[s]
            d = ball.dist(z_s.unsqueeze(1), class_emb.unsqueeze(0))
            avg_dists += d
        avg_dists /= n_samples

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

    return {"top1": correct_1 / max(total, 1) * 100,
            "top5": correct_5 / max(total, 1) * 100,
            "n_test": total}


@torch.no_grad()
def evaluate_euclidean(model, test_loader, unseen_attrs, label_to_unseen_idx,
                       device):
    model.eval()
    class_emb = model.embed_classes(unseen_attrs.to(device))

    correct_1 = correct_5 = total = 0

    for features, labels, _ in tqdm(test_loader, desc="Eval (Euclidean)"):
        features = features.to(device)
        B = features.shape[0]
        vis = model.visual_encoder(features)
        dists = torch.cdist(vis, class_emb)
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

    return {"top1": correct_1 / max(total, 1) * 100,
            "top5": correct_5 / max(total, 1) * 100,
            "n_test": total}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # Data
    dataset = UnifiedDataset(args.dataset, args.data_root)
    _, test_sub = build_zsl_splits(dataset)
    test_loader = torch.utils.data.DataLoader(
        test_sub, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    class_emb = load_class_embeddings(dataset)
    attr_dim = class_emb.shape[1]

    # Build unseen-class index mapping
    label_to_unseen_idx = {}
    unseen_indices = sorted(dataset.unseen_classes)
    for rank, cls_idx in enumerate(unseen_indices):
        label_to_unseen_idx[cls_idx] = rank

    unseen_attrs = class_emb[unseen_indices]

    # Model
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
            label_to_unseen_idx, device, args.n_samples,
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
            label_to_unseen_idx, device,
        )

    # Report
    print(f"\n{'='*50}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Model:    {args.model}")
    print(f"  Ckpt:     {args.checkpoint}")
    print(f"  Samples:  {results['n_test']}")
    print(f"  Top-1:    {results['top1']:.2f}%")
    print(f"  Top-5:    {results['top5']:.2f}%")
    print(f"{'='*50}\n")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "dataset", "checkpoint", "ablation",
                        "top1", "top5", "n_test"])
            w.writerow([
                args.model, args.dataset, args.checkpoint,
                saved_args.get("ablation_fraction", "?"),
                f"{results['top1']:.2f}", f"{results['top5']:.2f}",
                results["n_test"],
            ])
        print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
