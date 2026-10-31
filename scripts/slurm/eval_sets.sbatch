#!/bin/bash
# All evaluations of one trained model in one job (the per-user limit is 100 submitted jobs).
#   sbatch scripts/slurm/eval_sets.sbatch <onejev|gensft|earthdial> <model dir> <name> <set=file> [<set=file> ...]
# Each set writes runs/eval/<name>_<set>/:
#   onejev     s1geo.onejev_eval -> probe.logits.jsonl (+ s1geo.metrics)
#   gensft     s1geo.gen_sft eval (greedy -> predictions.jsonl) and, for sets marked with a leading '+',
#              s1geo.gen_sft score (option scoring -> option_scores.jsonl) in the same directory
#   earthdial  scripts/baselines/earthdial_lora.py eval / score, batch 16 (see earthdial_lora.sbatch)
# Example: sbatch scripts/slurm/eval_sets.sbatch gensft runs/gen_sft/msi_gensft_s0/final msi_gensft_s0 \
#            +test=data/earthjev/reben5pct_msi/test.jsonl unseen=data/earthjev/reben5pct_msi/test_unseen.jsonl
#SBATCH --job-name=earthjev-eval
#SBATCH --account=users
#SBATCH --partition=zwickyB-GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
KIND=${1:?kind}; MODEL=$(realpath "${2:?model dir}"); NAME=${3:?name}; shift 3
REPO=$(pwd); S=$REPO/data/reben/subset_5pct
if [ "$KIND" = earthdial ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set +u; conda activate "${EARTHDIAL_ENV:-${HOME}/envs/earthdial}"; set -u
  export HF_HOME=$REPO/data/hf TRANSFORMERS_OFFLINE=1
else
  source env/activate.sh
fi
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/third_party/OneJev${PYTHONPATH:+:${PYTHONPATH}}"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD) kind=$KIND model=$MODEL name=$NAME sets=$*"
for spec in "$@"; do
  score=0; [ "${spec:0:1}" = "+" ] && { score=1; spec=${spec:1}; }
  set_name=${spec%%=*}; file=$(realpath "${spec#*=}"); out=$REPO/runs/eval/${NAME}_${set_name}
  echo "== $set_name $file score=$score $(date +%T)"
  case $KIND in
    onejev)
      (cd third_party/OneJev && srun python -m s1geo.onejev_eval --model "$MODEL" --media-root "$S" --eval "probe=$file" --out-dir "$out")
      python -m s1geo.metrics --test "$out/probe.logits.jsonl" --out "$out/metrics.json" ;;
    gensft)
      srun python -m s1geo.gen_sft eval --model "$MODEL" --questions "$file" --media-root "$S" --out "$out"
      if [ $score = 1 ]; then srun python -m s1geo.gen_sft score --model "$MODEL" --questions "$file" --media-root "$S" --out "$out"; fi ;;
    earthdial)
      srun python scripts/baselines/earthdial_lora.py eval --model "$MODEL" --questions "$file" --media-root "$S" --out "$out" --batch-size 16
      if [ $score = 1 ]; then srun python scripts/baselines/earthdial_lora.py score --model "$MODEL" --questions "$file" --media-root "$S" --out "$out" --batch-size 16; fi ;;
    *) echo "unknown kind $KIND"; exit 1 ;;
  esac
done
echo "done $(date +%T)"
