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
SEED="${SEED:-42}"
C0="${C0:-1.0}"

echo "=== Curvature Study Training ==="
echo "  Dataset:  $DATASET"
echo "  Ablation: $ABLATION"
echo "  Model:    $MODEL"
echo "  Seed:     $SEED"
echo "  c0:       $C0"
echo "  Device:   cuda"
echo ""

# Match the suffix logic in train_multi.py: defaults are silent in the tag.
if [ "$SEED" = "42" ]; then
    SEED_SUFFIX=""
else
    SEED_SUFFIX="_seed${SEED}"
fi
if [ "$C0" = "1.0" ]; then
    C0_SUFFIX=""
else
    C0_SUFFIX="_c0${C0}"
fi
TAG="${MODEL}_${DATASET}_abl${ABLATION}_dim64${SEED_SUFFIX}${C0_SUFFIX}"

# Train
python3 train_multi.py \
    --dataset "$DATASET" \
    --data_root "$DATA_ROOT" \
    --ablation_fraction "$ABLATION" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --curvature "$C0" \
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
