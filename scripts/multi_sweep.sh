#!/bin/bash
# ============================================================
#  Submit the full curvature study sweep as SLURM jobs.
#
#  Usage:
#    bash scripts/multi_sweep.sh              # All datasets, Bayesian only
#    bash scripts/multi_sweep.sh cifar100     # Single dataset
#    bash scripts/multi_sweep.sh all both     # All datasets + Euclidean
#
#  This submits 6 fractions × N datasets × M models jobs.
# ============================================================

set -euo pipefail

FRACTIONS=(0.01 0.05 0.1 0.25 0.5 1.0)

# Parse arguments
DATASET_ARG="${1:-all}"
MODEL_ARG="${2:-bayesian}"

if [ "$DATASET_ARG" = "all" ]; then
    DATASETS=(cifar100 stanford_cars cub200 sun397 tiered_imagenet inaturalist)
else
    DATASETS=("$DATASET_ARG")
fi

if [ "$MODEL_ARG" = "both" ]; then
    MODELS=(bayesian euclidean)
else
    MODELS=("$MODEL_ARG")
fi

echo "=== Curvature Study Sweep ==="
echo "  Datasets: ${DATASETS[*]}"
echo "  Models:   ${MODELS[*]}"
echo "  Fractions: ${FRACTIONS[*]}"
echo ""

JOB_COUNT=0

for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "--- $model on $ds ---"
        for frac in "${FRACTIONS[@]}"; do
            sbatch --export=ALL,DATASET=$ds,ABLATION=$frac,MODEL=$model \
                   --job-name="curv_${ds}_${model}_${frac}" \
                   scripts/train_multi_job.sh
            echo "  Submitted: $model $ds ablation=$frac"
            JOB_COUNT=$((JOB_COUNT + 1))
        done
        echo ""
    done
done

echo "Total jobs submitted: $JOB_COUNT"
echo "Monitor with: squeue -u \$USER"
