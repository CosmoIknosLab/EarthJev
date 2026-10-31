#!/bin/bash
# Stream reBEN S2 + reference maps into compressed Zarr (CPU node; compute nodes have internet).
#   sbatch scripts/data/prepare_reben.sbatch <fraction | ids file> <out dir>
#SBATCH --job-name=earthjev-reben
#SBATCH --account=users
#SBATCH --partition=zwickyA-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
source env/activate.sh
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD)"
SEL="${1:?fraction or ids file}"
if [ -f "${SEL}" ]; then SEL_ARGS=(--ids "${SEL}"); else SEL_ARGS=(--fraction "${SEL}"); fi
srun python scripts/data/prepare_reben.py --metadata data/reben/raw/metadata.parquet \
  --maps data/reben/raw/Reference_Maps.tar.zst "${SEL_ARGS[@]}" --seed 0 --out "${2:?out dir}"
