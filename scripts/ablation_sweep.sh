#!/bin/bash
# ============================================================
#  Submit the full ablation sweep as SLURM job arrays on PACE ICE.
#
#  Usage:  bash scripts/ablation_sweep.sh
#
#  This submits 6 Bayesian + 6 Euclidean = 12 jobs total.
#  Job arrays queue independently, so they can run in parallel
#  across available nodes.
# ============================================================

set -euo pipefail

# Use 0.1/0.5 (not 0.10/0.50) to match Python's float formatting
# in checkpoint filenames, so the eval glob finds them.
FRACTIONS=(0.01 0.05 0.1 0.25 0.5 1.0)

echo "=== Submitting Bayesian Hyperbolic jobs ==="
for frac in "${FRACTIONS[@]}"; do
    sbatch --export=ALL,ABLATION=$frac,MODEL=bayesian \
           --job-name="bhzsl_${frac}" \
           scripts/train_job.sh
    echo "  Submitted: bayesian ablation=$frac"
done

echo ""
echo "=== Submitting Euclidean baseline jobs ==="
for frac in "${FRACTIONS[@]}"; do
    sbatch --export=ALL,ABLATION=$frac,MODEL=euclidean \
           --job-name="euc_${frac}" \
           scripts/train_job.sh
    echo "  Submitted: euclidean ablation=$frac"
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
