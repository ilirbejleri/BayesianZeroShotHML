#!/usr/bin/env python3
"""
Training loop for Bayesian Hyperbolic Zero-Shot Learning.

Supports:
  - Full and ablated training-set sizes (--ablation_fraction)
  - Bayesian hyperbolic model and Euclidean baseline (--model {bayesian,euclidean})
  - CSV metric logging for downstream analysis
  - Checkpoint saving every N epochs
  - MPS (Apple Silicon), CUDA, or CPU backends

Example — local M2 run with 10 % data:
    python train.py --data_root ./data/Animals_with_Attributes2 \
                    --ablation_fraction 0.10 --epochs 50 --device mps

Example — SLURM cluster, full data:
    python train.py --data_root /scratch/$USER/AwA2 \
                    --ablation_fraction 1.0 --epochs 200 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
from tqdm import tqdm

from data import get_dataloaders, SEEN_CLASSES, UNSEEN_CLASSES, CLASS_TO_IDX
from models import BayesianHyperbolicZSL, EuclideanBaselineZSL
from loss import BayesianHyperbolicLoss, EuclideanAlignmentLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Bayesian Hyperbolic ZSL")
    # Data
    p.add_argument("--data_root", type=str, required=True,
                    help="Path to Animals_with_Attributes2/ directory")
    p.add_argument("--attributes_path", type=str, default=None,
                    help="Path to predicate-matrix-continuous.txt (if not inside data_root)")
    p.add_argument("--ablation_fraction", type=float, default=1.0,
                    help="Fraction of training data to use (0, 1]")
    # Model
    p.add_argument("--model", choices=["bayesian", "euclidean"], default="bayesian")
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--curvature", type=float, default=1.0,
                    help="Initial Poincaré ball curvature (learnable)")
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_hyp", type=float, default=1e-4,
                    help="Learning rate for manifold parameters (geoopt)")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    # Loss
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lambda_kl", type=float, default=0.01)
    p.add_argument("--free_nats", type=float, default=0.0)
    # System
    p.add_argument("--device", type=str, default="auto",
                    help="auto | cpu | mps | cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=25,
                    help="Save checkpoint every N epochs")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_seen_label_map(dataset_class_names: list) -> dict:
    """Map dataset-level label indices to contiguous 0..39 indices for seen classes."""
    seen_set = set(SEEN_CLASSES)
    mapping = {}
    counter = 0
    for idx, name in enumerate(dataset_class_names):
        if name in seen_set:
            mapping[idx] = counter
            counter += 1
    return mapping


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"[INFO] device = {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, test_loader, class_emb = get_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        ablation_fraction=args.ablation_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        attributes_path=args.attributes_path,
    )
    n_train = len(train_loader.dataset)
    print(f"[INFO] Training samples: {n_train}  (ablation={args.ablation_fraction})")

    # We only train on *seen* classes; build a contiguous label mapping
    from data import AwA2Features
    full_ds = AwA2Features(args.data_root, attributes_path=args.attributes_path)
    seen_map = build_seen_label_map(full_ds.class_names)
    n_seen = len(seen_map)
    seen_class_indices = sorted(seen_map.keys())

    # Attribute embeddings for seen classes only (used during training)
    seen_attrs = class_emb[seen_class_indices].to(device)  # (n_seen, attr_dim)
    attr_dim = class_emb.shape[1]

    # ------------------------------------------------------------------
    # Model & loss
    # ------------------------------------------------------------------
    if args.model == "bayesian":
        model = BayesianHyperbolicZSL(
            visual_dim=2048,
            attr_dim=attr_dim,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            curvature=args.curvature,
        ).to(device)

        criterion = BayesianHyperbolicLoss(
            temperature=args.temperature,
            lambda_kl=args.lambda_kl,
            free_nats=args.free_nats,
        )

        # Separate optimiser groups: Euclidean params + manifold params
        euclidean_params = [p for p in model.parameters()
                           if not isinstance(p, geoopt.ManifoldParameter)]
        manifold_params = [p for p in model.parameters()
                          if isinstance(p, geoopt.ManifoldParameter)]

        optimizer = geoopt.optim.RiemannianAdam(
            [
                {"params": euclidean_params, "lr": args.lr},
                {"params": manifold_params, "lr": args.lr_hyp},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        model = EuclideanBaselineZSL(
            visual_dim=2048,
            attr_dim=attr_dim,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
        ).to(device)

        criterion = EuclideanAlignmentLoss(temperature=args.temperature)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_abl{args.ablation_fraction}_dim{args.embed_dim}"
    csv_path = log_dir / f"train_{tag}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "align_loss", "kl_loss",
                         "epoch_time_s", "curvature"])

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"[INFO] Starting training — {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_align = 0.0
        epoch_kl = 0.0
        n_batches = 0
        t0 = time.time()

        for features, labels, attrs in tqdm(train_loader, desc=f"Epoch {epoch}",
                                            leave=False):
            features = features.to(device)
            # Remap labels to contiguous seen-class indices
            labels_mapped = torch.tensor(
                [seen_map[int(l)] for l in labels], device=device,
            )
            attrs = attrs.to(device)

            optimizer.zero_grad()

            if args.model == "bayesian":
                out = model(features, seen_attrs[labels_mapped])
                class_emb_hyp = model.embed_classes(seen_attrs)  # (n_seen, d)
                losses = criterion(out, class_emb_hyp, labels_mapped)
                loss = losses["total"]
                epoch_align += losses["align"].item()
                epoch_kl += losses["kl"].item()
            else:
                out = model(features, seen_attrs[labels_mapped])
                class_emb_euc = model.embed_classes(seen_attrs)
                loss = criterion(out["visual"], class_emb_euc, labels_mapped)

            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        dt = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_align = epoch_align / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)

        cur_c = float(model.ball.c) if hasattr(model, "ball") else 0.0

        csv_writer.writerow([epoch, f"{avg_loss:.6f}", f"{avg_align:.6f}",
                             f"{avg_kl:.6f}", f"{dt:.1f}", f"{cur_c:.4f}"])
        csv_file.flush()

        print(f"  Epoch {epoch:03d}  loss={avg_loss:.4f}  "
              f"align={avg_align:.4f}  kl={avg_kl:.4f}  "
              f"c={cur_c:.4f}  [{dt:.1f}s]")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = ckpt_dir / f"{tag}_ep{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  → saved {ckpt_path}")

    csv_file.close()
    print(f"[DONE] Logs → {csv_path}")


if __name__ == "__main__":
    main()
