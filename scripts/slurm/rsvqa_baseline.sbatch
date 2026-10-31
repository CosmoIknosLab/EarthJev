#!/bin/bash
# Traditional VQA baseline (s1geo.rsvqa_baseline): train, eval or bench, on one H100.
#   sbatch scripts/slurm/rsvqa_baseline.sbatch train --train <jsonl> --val <jsonl> --subset data/reben/subset_5pct --out runs/rsvqa_baseline/<name>
#SBATCH --job-name=earthjev-rsvqa
#SBATCH --account=users
#SBATCH --partition=zwickyB-GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
source env/activate.sh
export HF_HUB_OFFLINE=1
export PYTHONPATH="$(pwd)/third_party/OneJev${PYTHONPATH:+:${PYTHONPATH}}"   # qev.calibrate
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD) args=$*"
srun python -m s1geo.rsvqa_baseline "$@"
