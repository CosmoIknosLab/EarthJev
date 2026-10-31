#!/bin/bash
# Build the s1geo conda environment on project storage.
# Run from the login node (downloads only, no heavy compute).
#   bash env/build_env.sh            # default prefix
#   S1GEO_ENV=/path/to/env bash env/build_env.sh
set -euo pipefail

ENV_PREFIX="${S1GEO_ENV:-${HOME}/envs/earthjev}"
PY_VERSION="3.11"
TORCH_VERSION="2.14.1"
TORCHVISION_VERSION="0.29.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"   # driver 590.x supports CUDA 13.0
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  conda create -y -p "${ENV_PREFIX}" "python=${PY_VERSION}" pip
fi
set +u  # conda activation scripts reference unset variables
conda activate "${ENV_PREFIX}"
set -u

python -m pip install --upgrade pip
python -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX}"
python -m pip install -r "${REPO_DIR}/env/requirements.txt"
git -C "${REPO_DIR}" submodule update --init third_party/OneJev
python -m pip install -e "${REPO_DIR}/third_party/OneJev[train]"   # qev + OneJev training deps
python -m pip install -e "${REPO_DIR}" --no-deps

python -m pip freeze --exclude-editable > "${REPO_DIR}/env/requirements.lock.txt"
python -c "import torch, s1geo; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "Environment ready at ${ENV_PREFIX}"
