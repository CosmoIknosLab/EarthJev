#!/bin/bash
# OneJev fine-tuning (third_party/OneJev/train/sft.py) with the official recipe.
# Submit from the repository root, one process per GPU of the allocation:
#   sbatch --gres=gpu:h100:2 scripts/slurm/onejev_sft.sbatch configs/onejev/<config>.yaml
#   SFT_MODULE=s1geo.msi_sft sbatch scripts/slurm/onejev_sft.sbatch configs/onejev/msi/<config>.yaml   # 12-band
#SBATCH --job-name=earthjev-sft
#SBATCH --account=users
#SBATCH --partition=zwickyB-GPU
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
source env/activate.sh
# NCCL peer-to-peer between the two GPUs of a node hangs on this cluster (GPUs on different NUMA
# nodes, topology SYS).
export NCCL_P2P_DISABLE=1

CONFIG="$(realpath "${1:?usage: sbatch onejev_sft.sbatch <config.yaml> [extra train.sft flags]}")"
shift
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "host=$(hostname) job=${SLURM_JOB_ID} gpus=${NGPU} config=${CONFIG} module=${SFT_MODULE:-train.sft} earthjev_commit=$(git rev-parse --short HEAD) onejev_commit=$(git -C third_party/OneJev rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

cd third_party/OneJev   # train.sft is imported as a module from the OneJev checkout
export HF_HUB_OFFLINE=1   # models are pre-downloaded into $HF_HOME from the login node
srun torchrun --standalone --nproc-per-node "${NGPU}" -m "${SFT_MODULE:-train.sft}" --config "${CONFIG}" "$@"
