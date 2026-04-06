#!/bin/sh
# ============================================================
#  Bayesian Hyperbolic ZSL — training + eval execution script
#  Called by train_job.sh after SLURM env setup.
#
#  Override defaults via --export when submitting:
#    sbatch --export=ALL,ABLATION=0.01 scripts/train_job.sh
# ============================================================

DATA_ROOT="${DATA_ROOT:-../Animals_with_Attributes2}"
ABLATION="${ABLATION:-1.0}"
EPOCHS="${EPOCHS:-200}"
EMBED_DIM="${EMBED_DIM:-64}"
MODEL="${MODEL:-bayesian}"
LR="${LR:-1e-3}"
LAMBDA_KL="${LAMBDA_KL:-0.01}"

echo "============================================"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Model:       $MODEL"
echo "  Ablation:    $ABLATION"
echo "  Epochs:      $EPOCHS"
echo "  Embed dim:   $EMBED_DIM"
echo "============================================"

python3 train.py \
    --data_root "$DATA_ROOT" \
    --model "$MODEL" \
    --ablation_fraction "$ABLATION" \
    --epochs "$EPOCHS" \
    --embed_dim "$EMBED_DIM" \
    --lr "$LR" \
    --lambda_kl "$LAMBDA_KL" \
    --batch_size 128 \
    --device cuda \
    --num_workers 12 \
    --log_dir logs \
    --ckpt_dir checkpoints \
    --save_every 25

echo "[DONE] Training finished — Job $SLURM_JOB_ID"

CKPT=$(ls -t checkpoints/${MODEL}_abl${ABLATION}_dim${EMBED_DIM}_ep*.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    echo "[INFO] Evaluating $CKPT ..."
    python3 eval.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --model "$MODEL" \
        --n_samples 10 \
        --device cuda \
        --output_csv "logs/eval_${MODEL}_abl${ABLATION}_dim${EMBED_DIM}.csv"
fi
