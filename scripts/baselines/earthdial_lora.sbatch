#!/bin/bash
# EarthDial-4B-RGB + LoRA (scripts/baselines/earthdial_lora.py, recipe of s1geo.gen_sft --lora) on one H100, in
# EarthDial's own environment (env/earthdial_README.md). Runs one earthdial_lora.py command with explicit arguments:
#   sbatch scripts/baselines/earthdial_lora.sbatch py train --train data/earthjev/reben5pct_rgb/train.jsonl \
#       --media-root data/reben/subset_5pct --seed 0 --out runs/gen_sft/<name>
# Evaluation goes through scripts/slurm/eval_sets.sbatch (kind earthdial), batch 16: Phi-3 runs with eager attention
# here (no flash-attn), and the fp32 attention weights of 64 rows x ~1,500 tokens do not fit in 80 GB. Submit from
# the repository root.
#SBATCH --job-name=earthjev-earthdial-lora
#SBATCH --account=users
#SBATCH --partition=zwickyB-GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${EARTHDIAL_ENV:-${HOME}/envs/earthdial}"; set -u
export HF_HOME="${HF_HOME:-${PWD}/data/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ "${1:?usage: earthdial_lora.sbatch py <earthdial_lora.py command and arguments>}" = py ] || { echo "only 'py' mode"; exit 1; }
shift
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD) args=$*"
srun python scripts/baselines/earthdial_lora.py "$@"
