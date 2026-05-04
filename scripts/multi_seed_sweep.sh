#!/bin/bash
# ============================================================
#  Submit multi-seed reruns of the curvature study sweep.
#
#  Existing results in logs/ use seed 42. This script submits
#  ADDITIONAL seeds so that each (dataset, fraction) cell has
#  multiple training runs, enabling cross-seed mean ± std and
#  CI estimation.
#
#  Usage:
#    bash scripts/multi_seed_sweep.sh                     # All datasets, seeds 0,1,2
#    bash scripts/multi_seed_sweep.sh awa2                # Single dataset
#    bash scripts/multi_seed_sweep.sh awa2 "0 1 2 3 4"    # Single dataset, 5 extra seeds
#    bash scripts/multi_seed_sweep.sh all "0 1"           # All datasets, 2 extra seeds
#
#  This submits len(seeds) × 6 fractions × N datasets jobs.
# ============================================================

set -euo pipefail

FRACTIONS=(0.01 0.05 0.1 0.25 0.5 1.0)

DATASET_ARG="${1:-all}"
SEEDS_ARG="${2:-0 1 2}"

if [ "$DATASET_ARG" = "all" ]; then
    DATASETS=(awa2 cifar100 stanford_cars cub200 sun397 tiered_imagenet inaturalist)
else
    DATASETS=("$DATASET_ARG")
fi

read -r -a SEEDS <<< "$SEEDS_ARG"

echo "=== Multi-Seed Curvature Study Sweep ==="
echo "  Datasets: ${DATASETS[*]}"
echo "  Seeds:    ${SEEDS[*]}  (NOTE: seed 42 already in logs/, will be skipped)"
echo "  Fractions: ${FRACTIONS[*]}"
echo ""

JOB_COUNT=0

for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ "$seed" = "42" ]; then
            echo "[skip] seed 42 already exists for $ds"
            continue
        fi
        echo "--- $ds seed=$seed ---"
        for frac in "${FRACTIONS[@]}"; do
            sbatch --export=ALL,DATASET=$ds,ABLATION=$frac,MODEL=bayesian,SEED=$seed \
                   --job-name="seed${seed}_${ds}_${frac}" \
                   scripts/train_multi_job.sh
            echo "  Submitted: $ds ablation=$frac seed=$seed"
            JOB_COUNT=$((JOB_COUNT + 1))
        done
        echo ""
    done
done

echo "Total jobs submitted: $JOB_COUNT"
echo "Monitor with: squeue -u \$USER"
