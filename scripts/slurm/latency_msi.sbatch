#!/bin/bash
# Latency of every model of the EarthJev MSI comparison in ONE job on ONE H100, seed 0.
# Two scenarios for every model, 10 warm-up + 100 test images: one question per request ("single", batch 1) and all the
# questions of an image in one request ("image": EarthJev shares the image prefix; generative models generate the
# answers as one batch; task-specific models run one batched forward). Every timer runs from the decoded image in memory to the answer
# (preprocessing, tokenisation and the model; disk reads excluded). Output: $OUT/<model>.json
#   sbatch --exclude=compute-node72 scripts/slurm/latency_msi.sbatch
#   IMAGES=12 WARMUP=2 OUT=runs/tmp/latency_smoke GENSFT=runs/tmp/gensft_smoke/run/final sbatch ... (quick check)
#SBATCH --job-name=msi-latency
#SBATCH --account=users
#SBATCH --partition=zwickyB-GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
S=data/reben/subset_5pct; G=data/earthjev
M=$G/reben5pct_msi/test.jsonl; R=$G/reben5pct_rgb/test.jsonl; T=$G/reben5pct_en/test.jsonl
OUT=${OUT:-runs/latency/msi}; IMAGES=${IMAGES:-110}; WARMUP=${WARMUP:-10}
GENSFT=${GENSFT:-runs/gen_sft/msi_gensft_s0/final}
N="--images $IMAGES --warmup $WARMUP"
mkdir -p $OUT
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD)"; nvidia-smi --query-gpu=name --format=csv,noheader
source env/activate.sh; export HF_HUB_OFFLINE=1 PYTHONPATH="$(pwd)/third_party/OneJev"
python -m s1geo.bench_latency earthjev --model runs/onejev/earthjev_msi_08b_reben5pct_s0/final --questions $M --media-root $S $N --out $OUT/earthjev_msi.json
python -m s1geo.gen_sft bench --model $GENSFT --questions $M --media-root $S $N --out $OUT/gensft_msi.json
python -m s1geo.gen_sft bench --model runs/gen_sft/msi_lora_qwen3-vl-2b_s0/final --questions $R --media-root $S $N --out $OUT/lora_qwen3vl2b.json
python -m s1geo.gen_sft bench --model runs/gen_sft/msi_lora_internvl3-2b_s0/final --questions $R --media-root $S $N --out $OUT/lora_internvl3_2b.json
for e in resnet50 swin_t mambavision_t; do
  python -m s1geo.rsvqa_baseline bench --run runs/rsvqa_baseline/r2_rsvqa_${e}_s0 --questions $T --subset $S $N --out $OUT/rsvqa_$e.json
done
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${EARTHDIAL_ENV:-${HOME}/envs/earthdial}"; set -u
python scripts/baselines/earthdial_lora.py bench --model runs/gen_sft/msi_lora_earthdial4b_s0/final --questions $R --media-root $S $N --out $OUT/lora_earthdial4b.json
echo done
