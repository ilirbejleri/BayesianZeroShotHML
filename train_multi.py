#!/usr/bin/env python3
"""
Training loop for the curvature study (Paper 2).

Identical to train.py but parameterised by --dataset instead of hardcoded
AwA2 paths.  All hyperparameters match Paper 1 exactly so curvature
differences are attributable to dataset structure, not training differences.

Example:
    python train_multi.py --dataset cifar100 --ablation_fraction 0.05 --epochs 200 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
from tqdm import tqdm

from data_multi import (
    UnifiedDataset, build_zsl_splits, ablate_training_set,
    load_class_embeddings, build_seen_label_map,
)
from models import BayesianHyperbolicZSL, EuclideanBaselineZSL
from loss import BayesianHyperbolicLoss, EuclideanAlignmentLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Bayesian Hyperbolic ZSL (multi-dataset)")
    p.add_argument("--dataset", type=str, required=True,
                    help="Dataset name (e.g. cifar100, cub200, awa2)")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--ablation_fraction", type=float, default=1.0)
    p.add_argument("--model", choices=["bayesian", "euclidean"], default="bayesian")
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_hyp", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lambda_kl", type=float, default=0.01)
    p.add_argument("--free_nats", type=float, default=0.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=50)
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[INFO] device={device}  dataset={args.dataset}  "
          f"model={args.model}  ablation={args.ablation_fraction}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = UnifiedDataset(args.dataset, args.data_root)
    train_sub, test_sub = build_zsl_splits(dataset)

    if args.ablation_fraction < 1.0:
        train_sub = ablate_training_set(train_sub, args.ablation_fraction, args.seed)

    n_train = len(train_sub)
    effective_bs = min(args.batch_size, n_train)
    if effective_bs < args.batch_size:
        print(f"[INFO] Small ablation: clamping batch_size "
              f"{args.batch_size} -> {effective_bs} (n_train={n_train})")
    train_loader = torch.utils.data.DataLoader(
        train_sub, batch_size=effective_bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=(n_train > args.batch_size),
    )

    class_emb = load_class_embeddings(dataset)
    attr_dim = class_emb.shape[1]
    seen_map = build_seen_label_map(dataset)
    n_seen = len(seen_map)
    seen_class_indices = sorted(seen_map.keys())
    seen_attrs = class_emb[seen_class_indices].to(device)

    n_train = len(train_sub)
    print(f"[INFO] Training samples: {n_train}  seen classes: {n_seen}  attr_dim: {attr_dim}")

    # ------------------------------------------------------------------
    # Model & loss
    # ------------------------------------------------------------------
    if args.model == "bayesian":
        model = BayesianHyperbolicZSL(
            visual_dim=2048, attr_dim=attr_dim,
            hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
            curvature=args.curvature,
        ).to(device)

        criterion = BayesianHyperbolicLoss(
            temperature=args.temperature,
            lambda_kl=args.lambda_kl,
            free_nats=args.free_nats,
        )

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
            visual_dim=2048, attr_dim=attr_dim,
            hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
        ).to(device)

        criterion = EuclideanAlignmentLoss(temperature=args.temperature)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

    # ------------------------------------------------------------------
    # Logging — includes dataset name in filenames
    # ------------------------------------------------------------------
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.dataset}_abl{args.ablation_fraction}_dim{args.embed_dim}"
    csv_path = log_dir / f"train_{tag}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "align_loss", "kl_loss",
                         "epoch_time_s", "curvature"])

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop (identical to Paper 1)
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
            labels_mapped = torch.tensor(
                [seen_map[int(l)] for l in labels], device=device,
            )

            optimizer.zero_grad()

            if args.model == "bayesian":
                out = model(features, seen_attrs[labels_mapped])
                class_emb_hyp = model.embed_classes(seen_attrs)
                losses = criterion(out, class_emb_hyp, labels_mapped)
                loss = losses["total"]
                epoch_align += losses["align"].item()
                epoch_kl += losses["kl"].item()
            else:
                out = model(features, seen_attrs[labels_mapped])
                class_emb_euc = model.embed_classes(seen_attrs)
                loss = criterion(out["visual"], class_emb_euc, labels_mapped)

            loss.backward()
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
