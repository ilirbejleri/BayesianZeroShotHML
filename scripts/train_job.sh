#!/bin/bash
#SBATCH --job-name=bhzsl
#SBATCH --nodes=1 --ntasks-per-node=12 --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --qos=coc-ice
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate bhzsl
bash scripts/run_train.sh
