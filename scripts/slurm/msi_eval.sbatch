#!/bin/bash
# Raw option logits of a 12-band EarthJev model on the test sets of the paper, with all bands and
# with the nine non-RGB bands set to their training mean (H3). One output dir per set:
# runs/eval/<tag>_<set>[_drop]/probe.logits.jsonl. Submit from the repository root:
#   sbatch scripts/slurm/msi_eval.sbatch <model dir> <tag>
#   SETS="test_corine=data/earthjev/abl_msi_eurosat/test_corine.jsonl" MEDIA_ROOT=data DROP_SETS="" sbatch ... (ablations)
#SBATCH --job-name=msi-eval
#SBATCH --account=users
#SBATCH --partition=zwickyB-GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
source env/activate.sh
export HF_HUB_OFFLINE=1
REPO="$(pwd)"
MODEL="$(realpath "${1:?model dir}")"
TAG="${2:?tag}"
G="${REPO}/data/earthjev"
MEDIA_ROOT="$(realpath "${MEDIA_ROOT:-${REPO}/data/reben/subset_5pct}")"
SETS="${SETS:-val=${G}/reben5pct_msi/validation.jsonl test=${G}/reben5pct_msi/test.jsonl unseen=${G}/reben5pct_msi/test_unseen.jsonl rsvqa=${G}/rsvqaxben_test_msi/presence.jsonl bentxt=${G}/bentxt_msi/bench.jsonl}"
DROP_SETS="${DROP_SETS-test unseen rsvqa bentxt}"
echo "host=$(hostname) job=${SLURM_JOB_ID} model=${MODEL} tag=${TAG} media_root=${MEDIA_ROOT} earthjev_commit=$(git rev-parse --short HEAD)"

cd third_party/OneJev
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
for spec in ${SETS}; do
  name="${spec%%=*}"; path="${spec#*=}"
  [ -f "${path}" ] || path="${REPO}/${path}"
  srun python -m s1geo.onejev_eval --model "${MODEL}" --media-root "${MEDIA_ROOT}" \
    --eval "probe=${path}" --out-dir "${REPO}/runs/eval/${TAG}_${name}"
  if [[ " ${DROP_SETS} " == *" ${name} "* ]]; then
    srun python -m s1geo.onejev_eval --model "${MODEL}" --media-root "${MEDIA_ROOT}" --msi-drop-extra-bands \
      --eval "probe=${path}" --out-dir "${REPO}/runs/eval/${TAG}_${name}_drop"
  fi
done
cd "${REPO}"
python scripts/checks/msi_band_weights.py "${MODEL}"
