#!/bin/bash
# ============================================================
#  Execute training + eval for multi-dataset curvature study.
#  Called by train_multi_job.sh inside SLURM.
# ============================================================

set -euo pipefail

DATASET="${DATASET:-cifar100}"
ABLATION="${ABLATION:-1.0}"
MODEL="${MODEL:-bayesian}"
DATA_ROOT="${DATA_ROOT:-data}"
EPOCHS="${EPOCHS:-200}"

echo "=== Curvature Study Training ==="
echo "  Dataset:  $DATASET"
echo "  Ablation: $ABLATION"
echo "  Model:    $MODEL"
echo "  Device:   cuda"
echo ""

TAG="${MODEL}_${DATASET}_abl${ABLATION}_dim64"

# Train
python3 train_multi.py \
    --dataset "$DATASET" \
    --data_root "$DATA_ROOT" \
    --ablation_fraction "$ABLATION" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --device cuda

# Evaluate the final checkpoint
CKPT="checkpoints/${TAG}_ep${EPOCHS}.pt"
if [ -f "$CKPT" ]; then
    echo ""
    echo "=== Evaluating $CKPT ==="
    python3 eval_multi.py \
        --dataset "$DATASET" \
        --data_root "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --model "$MODEL" \
        --n_samples 10 \
        --device cuda \
        --output_csv "logs/eval_${TAG}.csv"
else
    echo "[WARN] Checkpoint not found: $CKPT"
fi
