#!/bin/bash
#SBATCH --job-name=feat_extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_feat_%j.out
#SBATCH --qos=coc-ice

# ============================================================
#  Extract ResNet-101 features for a dataset on the cluster.
#
#  Usage:
#    sbatch --export=ALL,DATASET=cifar100 scripts/extract_features_job.sh
# ============================================================

set -euo pipefail

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate bhzsl

DATASET="${DATASET:-cifar100}"

echo "=== Feature Extraction: $DATASET ==="
python3 prepare_dataset.py --dataset "$DATASET" --device cuda --batch_size 256
