#!/bin/bash
# Pilot for the learning rate of the 12-channel input layer (selection rule fixed beforehand:
# lowest validation_1k NLL).
# Trains on the first 20,000 rows, then scores validation_1k with all bands and with the extra bands set to their mean,
# and prints the per-band weight norm. Submit from the repository root:
#   sbatch scripts/slurm/msi_lr_pilot.sbatch <input lr>
#SBATCH --job-name=msi-lr-pilot
#SBATCH --account=users
#SBATCH --partition=zwickyB-GPU
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
source env/activate.sh
export HF_HUB_OFFLINE=1
export NCCL_P2P_DISABLE=1
LR="${1:?input lr}"
REPO="$(pwd)"
CONFIG="${REPO}/configs/onejev/msi/earthjev_msi_08b_reben5pct.yaml"
NAME="msi_lrpilot_${LR}"
RUN="${REPO}/runs/onejev/${NAME}"
VAL="${REPO}/data/earthjev/reben5pct_msi/validation_1k.jsonl"
echo "host=$(hostname) job=${SLURM_JOB_ID} input_lr=${LR} earthjev_commit=$(git rev-parse --short HEAD)"

cd third_party/OneJev
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
srun torchrun --standalone --nproc-per-node 1 -m s1geo.msi_sft --config "${CONFIG}" --input-lr "${LR}" \
  --name "${NAME}" --limit 20000 --eval-every 100000 --save-every 100000
for MODE in all drop; do
  FLAG=(); [ "${MODE}" = drop ] && FLAG=(--msi-drop-extra-bands)
  srun python -m s1geo.onejev_eval --model "${RUN}/final" --media-root "${REPO}/data/reben/subset_5pct" \
    --eval "val=${VAL}" --out-dir "${REPO}/runs/eval/${NAME}_${MODE}" "${FLAG[@]}"
done
cd "${REPO}"
python scripts/checks/msi_band_weights.py "${RUN}/final"
