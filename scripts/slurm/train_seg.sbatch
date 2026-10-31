#!/bin/bash
# Train the per-pixel CORINE segmenter.  sbatch scripts/slurm/train_seg.sbatch <config.yaml> [seed]
#SBATCH --job-name=earthjev-seg
#SBATCH --account=users
#SBATCH --partition=zwickyB-GPU
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
source env/activate.sh
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD)"
SEED_ARGS=(); [ -n "${2:-}" ] && SEED_ARGS=(--seed "$2")
srun python -m s1geo.train_seg --config "${1:?config}" --resume auto "${SEED_ARGS[@]}"
