#!/bin/bash
# ============================================================
#  Initial-curvature (c0) ablation on a Pattern-C dataset.
#
#  Reviewer asked: is Pattern C real, or is it just slow drift
#  from c0 = 1.0? This sweep trains SUN397 (cleanest Pattern-C
#  dataset) at the standard 6 ablation fractions with three
#  different c0 values (0.1, 1.0, 5.0). If converged c_100% is
#  roughly invariant to c0, the patterns are real attractors.
#
#  Usage:
#    bash scripts/c0_ablation_sweep.sh                    # SUN397, c0 ∈ {0.1, 5.0}
#    bash scripts/c0_ablation_sweep.sh sun397 "0.1 5.0"   # explicit
#    bash scripts/c0_ablation_sweep.sh inaturalist "0.1 5.0"
#
#  c0=1.0 is already in logs/ from the main sweep; this script
#  skips it. Submits len(c0_values) × 6 fractions jobs.
# ============================================================

set -euo pipefail

FRACTIONS=(0.01 0.05 0.1 0.25 0.5 1.0)

DATASET_ARG="${1:-sun397}"
C0_ARG="${2:-0.1 5.0}"

read -r -a C0_VALUES <<< "$C0_ARG"

echo "=== c0 Ablation Sweep ==="
echo "  Dataset:   $DATASET_ARG"
echo "  c0 values: ${C0_VALUES[*]}  (NOTE: c0=1.0 already in logs/, will be skipped)"
echo "  Fractions: ${FRACTIONS[*]}"
echo ""

JOB_COUNT=0

for c0 in "${C0_VALUES[@]}"; do
    if [ "$c0" = "1.0" ]; then
        echo "[skip] c0=1.0 already exists for $DATASET_ARG"
        continue
    fi
    echo "--- $DATASET_ARG c0=$c0 ---"
    for frac in "${FRACTIONS[@]}"; do
        sbatch --export=ALL,DATASET=$DATASET_ARG,ABLATION=$frac,MODEL=bayesian,C0=$c0 \
               --job-name="c0${c0}_${DATASET_ARG}_${frac}" \
               scripts/train_multi_job.sh
        echo "  Submitted: $DATASET_ARG ablation=$frac c0=$c0"
        JOB_COUNT=$((JOB_COUNT + 1))
    done
    echo ""
done

echo "Total jobs submitted: $JOB_COUNT"
echo "Monitor with: squeue -u \$USER"
