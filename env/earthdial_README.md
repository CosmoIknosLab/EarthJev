# Environment for the EarthDial-4B baseline

EarthDial (InternVL2-4B: InternViT-300M + MLP projector + Phi-3-mini) needs transformers 4.37 and its own model code,
which conflict with the project environment, so it runs in a separate conda environment. The LoRA fine-tuning,
evaluation and latency code is `scripts/baselines/earthdial_lora.py` (the recipe of `s1geo.gen_sft --lora`).

## Code (not tracked by git)

```bash
cd third_party
git clone https://github.com/hiyamdebary/EarthDial      # commit 92c1260703ae8b3ddcac17fce6e8234de717308b, MIT
```

The EarthDial package is not installed: `earthdial_lora.py` adds `third_party/EarthDial/src` to `sys.path` (its
`pip install -e .` would pull deepspeed, gradio and bitsandbytes, which are not needed here).

## Weights (Hugging Face cache, `HF_HOME=./data/hf`)

```bash
source env/activate.sh
python -I -c "from huggingface_hub import snapshot_download; snapshot_download('akshaydudhane/EarthDial_4B_RGB')"  # 7.8 GB
```

We used snapshot `ae752c47`. The Hugging Face repo has no model card or license tag; EarthDial's GitHub README points
to it as the official checkpoint. Jobs run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Environment

Python 3.10, torch 2.1.2+cu121, transformers 4.37.2 (EarthDial's pin), peft 0.10.0. No flash-attn (InternViT and
Phi-3 fall back to standard attention). Full package list: `scripts/baselines/requirements_earthdial.lock.txt`.

```bash
conda create -y -p ${HOME}/envs/earthdial python=3.10
P=${HOME}/envs/earthdial/bin/python
$P -m pip install --upgrade pip
$P -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
$P -m pip install "numpy<2" transformers==4.37.2 tokenizers==0.15.1 sentencepiece==0.1.99 timm==0.9.12 \
    einops==0.6.1 einops-exts==0.0.4 peft==0.10.0 accelerate==0.27.2 opencv-python-headless==4.9.0.80 \
    decord==0.6.0 imageio pillow pyyaml packaging protobuf
```

The Slurm scripts activate `${HOME}/envs/earthdial`; set `EARTHDIAL_ENV` to use another path.
