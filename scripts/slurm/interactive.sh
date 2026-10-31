#!/bin/bash
# Short interactive session on a GPU node, for debugging and quick tests only
# (ADHOC guide: use batch jobs for anything reproducible or long).
#   bash scripts/slurm/interactive.sh            # 1x L40, 1 hour
#   GPU=h100 TIME=00:30:00 bash scripts/slurm/interactive.sh
# Then, on the node:  source env/activate.sh
set -euo pipefail

GPU="${GPU:-l40}"
TIME="${TIME:-01:00:00}"
CPUS="${CPUS:-4}"
MEM="${MEM:-32G}"
PARTITION="${PARTITION:-zwickyB-GPU}"

sinfo -p "${PARTITION}" -o "%P %G %t %D" | grep -E "PARTITION|gpu:${GPU}"
exec srun --account=users --partition="${PARTITION}" --gres="gpu:${GPU}:1" \
  --ntasks=1 --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${TIME}" --pty bash
