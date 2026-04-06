#!/bin/bash
#SBATCH --job-name=curv_study
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --qos=coc-ice

# ============================================================
#  SLURM job for multi-dataset curvature study (Paper 2).
#
#  Expects environment variables:
#    DATASET   — dataset name (cifar100, cub200, etc.)
#    ABLATION  — data fraction (0.01, 0.05, ..., 1.0)
#    MODEL     — model type (bayesian or euclidean)
#
#  Usage:
#    sbatch --export=ALL,DATASET=cifar100,ABLATION=0.01,MODEL=bayesian \
#           scripts/train_multi_job.sh
# ============================================================

set -euo pipefail

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate bhzsl

bash scripts/run_multi_train.sh
