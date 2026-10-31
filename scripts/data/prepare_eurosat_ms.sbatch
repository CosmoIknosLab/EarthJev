#!/bin/bash
# Convert EuroSAT-MS into 12-band Zarr with the spatial splits (CPU node).
#   sbatch scripts/data/prepare_eurosat_ms.sbatch
#SBATCH --job-name=earthjev-eurosat-ms
#SBATCH --account=users
#SBATCH --partition=zwickyA-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
source env/activate.sh
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD)"
srun python scripts/data/prepare_eurosat_ms.py --zip data/eurosat/ms/raw/EuroSAT_MS.zip \
  --splits data/eurosat_embed/raw --out data/eurosat/ms/zarr
