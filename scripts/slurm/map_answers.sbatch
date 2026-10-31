#!/bin/bash
# "Classify, then compute" baseline: U-Net maps -> answers.
#   sbatch scripts/slurm/map_answers.sbatch <seg run dir> <questions jsonl> <out dir>
#SBATCH --job-name=earthjev-mapans
#SBATCH --account=users
#SBATCH --partition=zwickyB-GPU
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
source env/activate.sh
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD)"
srun python -m s1geo.map_answers --seg-run "${1:?seg run}" --subset data/reben/subset_5pct --questions "${2:?questions}" --out "${3:?out}"
