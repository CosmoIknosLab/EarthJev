# Source this file to activate the project environment:
#   source env/activate.sh
# Safe under `set -u`: conda activation scripts reference unset variables.
_s1geo_nounset=0; [[ $- == *u* ]] && _s1geo_nounset=1 && set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${S1GEO_ENV:-${HOME}/envs/earthjev}"
export S1GEO_DATA="${S1GEO_DATA:-${PWD}/data}"
# Hugging Face cache on project storage, never in $HOME (ADHOC guide)
export TORCH_HOME="${TORCH_HOME:-${PWD}/data/torch}"
export HF_HOME="${HF_HOME:-${PWD}/data/hf}"
(( _s1geo_nounset )) && set -u
unset _s1geo_nounset
