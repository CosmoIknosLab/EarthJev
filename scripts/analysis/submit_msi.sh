#!/bin/bash
# EarthJev training and evaluation jobs of the comparison in the paper (pre-specified analysis plan). One protocol for
# every model: the English CORINE questions of the reBEN 5% subset, seeds 0, 1, 2, the same test sets.
# Outputs: runs/<family>/msi_<name>_s<seed>, runs/eval/msi_<name>_s<seed>_{test,unseen,val,rsvqa,bentxt}.
#   bash scripts/analysis/submit_msi.sh earthjev     # EarthJev-MSI (12 bands), eval with and without the extra bands
#   bash scripts/analysis/submit_msi.sh lora         # Qwen3-VL-2B, InternVL3-2B, EarthDial-4B, LoRA, RGB
#   bash scripts/analysis/submit_msi.sh gensft       # same-backbone generative, 12 bands
#   bash scripts/analysis/submit_msi.sh ablations    # one-factor ablations of EarthJev-MSI, seed 0
#   bash scripts/analysis/submit_msi.sh taskspecific # ResNet-50 / Swin-T / MambaVision-T VQA, 12 bands, seeds 0-2
#   bash scripts/analysis/submit_msi.sh rules <full-train-split Zarr run of configs/seg/unet_full_w64d4.yaml>
# The task-specific models and the U-Net + rules reference were trained before this comparison with the same questions
# and splits (analysis plan, section 2); the taskspecific and rules stages reproduce them under the same run names.
# Job ids go to runs/msi_jobs.txt.
# DEP=<job ids, ':'-separated> makes the training jobs of a stage wait for those jobs to succeed.
set -euo pipefail
cd "$(dirname "$0")/../.."
X="--exclude=compute-node72"
S=data/reben/subset_5pct
G=data/earthjev
dep() { echo "--dependency=afterok:$1"; }
PRE="${DEP:+--dependency=afterok:$DEP}"
log() { echo "$*" | tee -a runs/msi_jobs.txt; }

# RGB rows (no date, no country) for the models that take a true-colour image; option scoring ('+') where the plan
# compares calibration (test, validation, transfer tests)
R=$G/reben5pct_rgb
RGB_SETS="+test=$R/test.jsonl unseen=$R/test_unseen.jsonl +val=$R/validation_1k.jsonl +rsvqa=$G/rsvqaxben_test_rgb/presence.jsonl +bentxt=$G/bentxt/bench.jsonl"

earthjev() {
  for seed in 0 1 2; do
    local name=earthjev_msi_08b_reben5pct_s$seed
    local j=$(SFT_MODULE=s1geo.msi_sft sbatch --parsable $X --job-name=msi_earthjev_s$seed --partition=zwickyB-GPU \
          --gres=gpu:h100:2 --time=04:00:00 scripts/slurm/onejev_sft.sbatch configs/onejev/msi/earthjev_msi_08b_reben5pct.yaml \
          --seed $seed --name $name)
    log "$(date +%F) msi_earthjev_s$seed train=$j eval=$(sbatch --parsable $X $(dep $j) --job-name=msi_earthjev_s${seed}_eval \
         scripts/slurm/msi_eval.sbatch runs/onejev/$name/final msi_earthjev_s$seed)"
  done
}

lora() {
  for seed in 0 1 2; do
    local name=msi_lora_earthdial4b_s$seed
    local j=$(sbatch --parsable $X --job-name=$name --time=12:00:00 scripts/baselines/earthdial_lora.sbatch py train \
          --train $R/train.jsonl --media-root $S --seed $seed --out runs/gen_sft/$name)
    log "$(date +%F) $name train=$j eval=$(sbatch --parsable $X $(dep $j) --job-name=${name}_eval \
         scripts/slurm/eval_sets.sbatch earthdial runs/gen_sft/$name/final $name $RGB_SETS)"
  done
  for seed in 0 1 2; do
    for m in "Qwen/Qwen3-VL-2B-Instruct msi_lora_qwen3-vl-2b" "OpenGVLab/InternVL3-2B-hf msi_lora_internvl3-2b"; do
      set -- $m; local model=$1 name=${2}_s$seed
      local j=$(sbatch --parsable $X --job-name=$name scripts/slurm/gen_sft.sbatch train --model $model --lora --lr 1e-4 \
            --seed $seed --train $R/train.jsonl --media-root $S --out runs/gen_sft/$name)
      log "$(date +%F) $name train=$j eval=$(sbatch --parsable $X $(dep $j) --job-name=${name}_eval \
           scripts/slurm/eval_sets.sbatch gensft runs/gen_sft/$name/final $name $RGB_SETS)"
    done
  done
}


taskspecific() {   # RSVQA-style VQA: image encoder on 12 bands + sentence encoder, 20 epochs, epoch chosen on validation
  local D=$G/reben5pct_en
  for m in "resnet50 3e-4" "swin_t 1e-4" "mambavision_t 3e-4"; do
    set -- $m; local enc=$1 lr=$2
    for seed in 0 1 2; do
      local name=r2_rsvqa_${enc}_s${seed}
      local j=$(sbatch --parsable $X $PRE --job-name=$name scripts/slurm/rsvqa_baseline.sbatch train --train $D/train.jsonl \
            --val $D/validation_1k.jsonl --subset $S --out runs/rsvqa_baseline/$name --bands all --encoder $enc --lr $lr \
            --epochs 20 --seed $seed)
      local ev=""
      for t in "test $D/test.jsonl" "unseen $D/test_unseen.jsonl" "val $D/validation_1k.jsonl" \
               "rsvqa $G/rsvqaxben_test/presence.jsonl"; do
        set -- $t
        ev="$ev $(sbatch --parsable $X $(dep $j) --job-name=${name}_$1 scripts/slurm/rsvqa_baseline.sbatch eval \
                  --run runs/rsvqa_baseline/$name --subset $S --questions $2 --out runs/eval/${name}_$1)"
      done
      log "$(date +%F) $name train=$j evals=$ev"
    done
  done
}

rules() {   # reference only: CORINE answers computed with the question generator's rules on the predicted map
  log "$(date +%F) r2_rules $(sbatch --parsable $X $PRE scripts/slurm/map_answers.sbatch ${1:?segmenter run} \
       $G/reben5pct_en/test.jsonl runs/eval/r2_rules_test)"
}



gensft() {   # same backbone, 12 bands, generates the answer; same widened and trained input layer as EarthJev-MSI
  local M=$G/reben5pct_msi
  local sets="+test=$M/test.jsonl unseen=$M/test_unseen.jsonl +val=$M/validation_1k.jsonl +rsvqa=$G/rsvqaxben_test_msi/presence.jsonl +bentxt=$G/bentxt_msi/bench.jsonl"
  for seed in 0 1 2; do
    local name=msi_gensft_s$seed
    local j=$(sbatch --parsable $X $PRE --job-name=$name scripts/slurm/gen_sft.sbatch train --model Qwen/Qwen3.5-0.8B --lr 5e-6 \
          --seed $seed --train $M/train.jsonl --media-root $S --msi-norm $S/msi_norm.json --out runs/gen_sft/$name)
    log "$(date +%F) $name train=$j eval=$(sbatch --parsable $X $(dep $j) --job-name=${name}_eval \
         scripts/slurm/eval_sets.sbatch gensft runs/gen_sft/$name/final $name $sets)"
  done
}

ablations() {   # one factor at a time, seed 0; evaluated on CORINE test, validation_1k and the transfer tests
  local M=$G/reben5pct_msi
  for cfg in configs/onejev/msi/ablations/*.yaml; do
    local a=$(basename $cfg .yaml) name=msi_abl_$(basename $cfg .yaml)
    local j=$(SFT_MODULE=s1geo.msi_sft sbatch --parsable $X $PRE --job-name=$name --partition=zwickyB-GPU --gres=gpu:h100:2 \
          --time=05:00:00 scripts/slurm/onejev_sft.sbatch $cfg)
    local sets="val=$M/validation_1k.jsonl test=$M/test.jsonl rsvqa=$G/rsvqaxben_test_msi/presence.jsonl bentxt=$G/bentxt_msi/bench.jsonl"
    [ $a = meta ] && sets="val=$G/abl_msi_meta/validation_1k.jsonl test=$G/abl_msi_meta/test.jsonl rsvqa=$G/abl_msi_meta_rsvqa/presence.jsonl bentxt=$G/bentxt_msi/bench.jsonl"
    local e=$(SETS="$sets" DROP_SETS="" sbatch --parsable $X $(dep $j) --job-name=${name}_eval scripts/slurm/msi_eval.sbatch \
          runs/onejev/$name/final $name)
    if [ $a = eurosat ]; then   # plus the EuroSAT-MS test, whose rows have media paths relative to data/
      e="$e $(SETS="eurosat=$G/abl_msi_eurosat/test_eurosat.jsonl" \
             MEDIA_ROOT=data DROP_SETS="" sbatch --parsable $X $(dep $j) --job-name=${name}_eval2 scripts/slurm/msi_eval.sbatch \
             runs/onejev/$name/final $name)"
    fi
    log "$(date +%F) $name train=$j eval=$e"
  done
}

"$@"
