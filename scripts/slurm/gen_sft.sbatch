#!/bin/bash
# Generative fine-tuning baseline (s1geo.gen_sft): train, or eval / bench a checkpoint, on one H100.
#   sbatch scripts/slurm/gen_sft.sbatch train --train <jsonl> --media-root data/reben/subset_5pct --lr 5e-6 --out runs/gen_sft/<name>
#   sbatch scripts/slurm/gen_sft.sbatch eval --model runs/gen_sft/<name>/final --questions <jsonl> --media-root ... --out runs/eval/<name>
#SBATCH --job-name=earthjev-gensft
#SBATCH --account=users
#SBATCH --partition=zwickyB-GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
source env/activate.sh
export HF_HUB_OFFLINE=1
export PYTHONPATH="$(pwd)/third_party/OneJev${PYTHONPATH:+:${PYTHONPATH}}"   # qev.calibrate
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD) args=$*"
srun python -m s1geo.gen_sft "$@"
