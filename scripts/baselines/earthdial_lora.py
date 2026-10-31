"""EarthDial-4B-RGB fine-tuned with LoRA on our training questions: the recipe of `s1geo.gen_sft train --lora`.

Fine-tuned remote-sensing MLLM baseline, the counterpart of the LoRA runs of InternVL3-2B / Qwen3-VL-2B made with
src/s1geo/gen_sft.py. gen_sft loads models through transformers' AutoModelForImageTextToText, which EarthDial
(InternVL2-4B: InternViT-300M + MLP projector + Phi-3-mini, custom code, transformers 4.37 in its own env) does not
support, so this file repeats gen_sft's train / eval / bench line by line on EarthDial's own model code
(third_party/EarthDial, commit 92c12607). Reused unchanged: `render()` and `parse()` (s1geo.gen_sft), `ece()`
(qev.calibrate), `pick()` (s1geo.bench_latency). Environment: its own conda env, see env/earthdial_README.md (peft 0.10.0).

Same as gen_sft --lora:
- data: rows shuffled with random.Random(seed 0), options shuffled per row with the same rng through render(r, rng);
  prompt = render() text (state JSON without the image key + question + options + "Answer with one of the
  options."), target = the option text as render() returns it (or yes/no), followed by the end-of-turn token;
- loss: token cross-entropy on the target tokens only (sum over the micro-batch's target tokens / (n tokens x
  accumulation steps)); the decoded supervised tokens of the first micro-batch are printed ("target check");
- LoRA r 32, alpha 64, dropout 0.05 on the language model's linear projection layers only; for Phi-3 these are the
  fused `self_attn.qkv_proj`, `self_attn.o_proj`, `mlp.gate_up_proj`, `mlp.down_proj` (the same list EarthDial uses
  in InternVLChatModel.wrap_llm_lora for Phi3ForCausalLM; gen_sft's q/k/v/o/gate/up/down regex names the same
  layers unfused). Vision encoder and MLP projector frozen. LoRA weights in fp32, bf16 base weights, bf16 autocast;
- AdamW lr 1e-4, weight decay 0, 3% linear warm-up + cosine (same LambdaLR), 1 epoch, 64 rows per optimiser step
  (micro-batch 16 x 4), gradient clipping 1.0, non-finite gradient norm -> step skipped and counted, seed 0;
  gradient checkpointing (non-reentrant) on the language model; the adapter is merged at the end;
- eval: greedy, max_new_tokens 24, batch 64 (left padding), stop at end-of-turn, parse(), answer_prob = probability
  of the generated tokens before the stop token, same predictions.jsonl / metrics.json fields;
- bench: 10 warm-up + 100 images from pick(), one question per request and all the questions of an image in one batched
  generation; timed from the decoded image in memory (tiling, normalisation, tokenisation and `generate`), as
  gen_sft.bench and s1geo.bench_latency.

EarthDial-specific (from their code, InternVLChatModel.chat()):
- image: `dynamic_preprocess` (max_num 6, use_thumbnail from the config) + `build_transform(is_train=False, 448,
  'imagenet')`; a 120x120 PNG gives one 448x448 tile = 256 image tokens;
- chat format: phi3-chat conversation template with the model's system message (InternVLChatModel.chat()): "<s>
  <|system|>\\n{system}<|end|><|user|>\\n[s2_rgb_10] <image>\\n{prompt}<|end|><|assistant|>\\n", <image> replaced
  by <img><IMG_CONTEXT>x256</img>; the training text appends "{target}<|end|>" (the phi3-chat MPT separator closes
  the assistant turn); eos for decoding = {<|endoftext|>, <|end|>} (their generation_config also lists </s>, id 2,
  which is added too); sensor tag [s2_rgb_10] as in their BigEarthNet_RGB questions;
- forward: InternVLChatModel.extract_feature (frozen, under no_grad) replaces the <IMG_CONTEXT> embeddings, then the
  Phi-3 decoder; only the last K positions go through lm_head (gen_sft: logits_to_keep=K). Position ids follow
  their Phi3 prepare_inputs_for_generation (attention_mask.cumsum - 1), so training and left-padded batched
  decoding see the same positions.
- supervised tokens = tokens of the full text after the tokenised prompt (the prompt ends in "<|assistant|>\\n",
  so the target tokenises the same way at train and test time); rows where the prompt tokens are not a prefix of
  the full text would fall back to gen_sft's separate count and are counted (`prefix_mismatch` in the log).

    python scripts/baselines/earthdial_lora.py train --train data/earthjev/reben5pct_rgb/train.jsonl \\
        --media-root data/reben/subset_5pct --out runs/gen_sft/<name>
    python scripts/baselines/earthdial_lora.py eval --model runs/gen_sft/<name>/final \\
        --questions data/earthjev/reben5pct_rgb/test.jsonl --media-root data/reben/subset_5pct --out runs/eval/<name>_test
    python scripts/baselines/earthdial_lora.py bench --model runs/gen_sft/<name>/final \\
        --questions data/earthjev/reben5pct_rgb/test.jsonl --media-root data/reben/subset_5pct --out runs/latency/<name>.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "OneJev"))      # qev.calibrate
sys.path.insert(0, str(REPO / "third_party" / "EarthDial" / "src"))

from s1geo.gen_sft import options, parse, render, write_option_scores  # noqa: E402

MODEL_ID = "akshaydudhane/EarthDial_4B_RGB"
TAG = "[s2_rgb_10]"
END = "<|end|>"
IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"
LORA_TARGETS = ["self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"]


def load(model_id: str):
    import torch
    from earthdial.model.internvl_chat import InternVLChatModel
    from transformers import AutoTokenizer

    path = model_id
    if not Path(model_id).exists():
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_id)          # local cache (HF_HUB_OFFLINE=1 in the job)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).cuda()
    model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT)
    return tok, model, torch


class Images:
    """RGB PNG -> [n_tiles, 3, 448, 448] exactly as run_earthdial.py (dynamic_preprocess + imagenet transform)."""

    def __init__(self, model, media_root: str):
        from earthdial.train.dataset import build_transform, dynamic_preprocess

        self.size = model.config.force_image_size or model.config.vision_config.image_size
        self.thumb = model.config.use_thumbnail
        self.dyn = dynamic_preprocess
        self.tf = build_transform(is_train=False, input_size=self.size, normalize_type="imagenet")
        self.root = Path(media_root)

    def open(self, row: dict):
        from PIL import Image

        return Image.open(self.root / row["media"][0]["path"]).convert("RGB")

    def prepare(self, img):
        import torch

        tiles = self.dyn(img, image_size=self.size, use_thumbnail=self.thumb, max_num=6)
        return torch.stack([self.tf(t) for t in tiles])

    def __call__(self, row: dict):
        return self.prepare(self.open(row))


def query(model, prompt: str, n_tiles: int) -> str:
    """InternVLChatModel.chat() prompt: phi3-chat template + system message, <image> expanded to the image tokens."""
    from earthdial.conversation import get_conv_template

    t = get_conv_template(model.template)
    t.system_message = model.system_message
    t.append_message(t.roles[0], f"{TAG} <image>\n{prompt}")
    t.append_message(t.roles[1], None)
    return t.get_prompt().replace("<image>", IMG_START + IMG_CONTEXT * model.num_image_token * n_tiles + IMG_END, 1)


def pad_left(seqs: list[list[int]], pad_id: int):
    import torch

    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        attn[i, L - len(s):] = 1
    pos = attn.cumsum(-1) - 1           # as Phi3ForCausalLM.prepare_inputs_for_generation
    pos.masked_fill_(attn == 0, 1)
    return ids.cuda(), attn.cuda(), pos.cuda()


def embed(model, lm, ids, pixel_values):
    """Text embeddings with the <IMG_CONTEXT> positions replaced by the (frozen) visual tokens, as in forward()."""
    import torch

    with torch.no_grad():
        vit = model.extract_feature(pixel_values)                  # [n_tiles, 256, C]
    emb = lm.get_input_embeddings()(ids).clone()
    sel = ids == model.img_context_token_id
    emb[sel] = vit.reshape(-1, emb.shape[-1]).to(emb.dtype)
    return emb


def eos_ids(tok) -> list[int]:
    ids = {tok.eos_token_id, tok.convert_tokens_to_ids(END), tok.convert_tokens_to_ids("<|endoftext|>"), 2}
    return sorted(t for t in ids if isinstance(t, int) and t >= 0)


def train(args) -> None:
    from peft import LoraConfig, get_peft_model

    rows = [json.loads(l) for l in open(args.train)]
    rng = random.Random(args.seed)
    tok, model, torch = load(args.model)
    torch.manual_seed(args.seed)
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    for p in model.parameters():                                    # vision encoder + mlp1 + LLM base frozen
        p.requires_grad = False
    lm = get_peft_model(model.language_model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    model.language_model = lm
    for p in lm.parameters():                                       # LoRA weights in fp32 (as gen_sft)
        if p.requires_grad:
            p.data = p.data.float()
    lm.print_trainable_parameters()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    phi3 = lm.base_model.model                                      # Phi3ForCausalLM with LoRA layers injected
    phi3.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    phi3.config.use_cache = False
    images = Images(model, args.media_root)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    micro, accum = args.micro_batch, args.rows_per_step // args.micro_batch
    total = math.ceil(len(rows) / args.rows_per_step)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    info = {"trainable_params": n_train, "total_params": n_all, "lora_targets": LORA_TARGETS, "steps": total,
            "warmup_steps": warm, "rows": len(rows), "system_message": model.system_message, "template": model.template}
    (out / "train_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in info.items() if k != "system_message"}), flush=True)
    log = open(out / "log.jsonl", "w")
    model.train()
    model.vision_model.eval()                                       # frozen, run under no_grad
    step, t0, skipped, mism = 0, time.time(), 0, 0
    for start in range(0, len(rows), args.rows_per_step):
        chunk = rows[start:start + args.rows_per_step]
        loss_sum, ntok = 0.0, 0
        for m in range(0, len(chunk), micro):
            batch = chunk[m:m + micro]
            seqs, targets, pix = [], [], []
            for r in batch:
                prompt, target, _ = render(r, rng)
                pv = images(r)
                q = query(model, prompt, pv.shape[0])
                p_ids = tok(q).input_ids
                f_ids = tok(q + target + END).input_ids
                if f_ids[:len(p_ids)] == p_ids:
                    n = len(f_ids) - len(p_ids)
                else:                                               # gen_sft's count, logged
                    mism += 1
                    n = len(tok(target + END, add_special_tokens=False).input_ids)
                seqs.append(f_ids); targets.append(n); pix.append(pv)
            ids, attn, pos = pad_left(seqs, pad_id)
            pixel_values = torch.cat(pix).to(torch.bfloat16).cuda()
            K = max(targets) + 1
            labels = ids[:, -(K - 1):].clone()
            for i, n in enumerate(targets):
                labels[i, :K - 1 - n] = -100
            if step == 0 and m == 0:   # the supervised tokens must be exactly the answer + end-of-turn
                for i in range(min(3, len(batch))):
                    print("target check:", repr(tok.decode(ids[i, -targets[i]:])), "| context:",
                          repr(tok.decode(ids[i, -targets[i] - 6:-targets[i]])), flush=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = embed(model, phi3, ids, pixel_values)
                h = phi3.model(inputs_embeds=emb, attention_mask=attn, position_ids=pos, use_cache=False)[0]
                logits = phi3.lm_head(h[:, -K:]).float()
            lp = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels, ignore_index=-100,
                                                   reduction="sum")
            n = int((labels != -100).sum())
            (lp / (n * accum)).backward()
            loss_sum += float(lp)
            ntok += n
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if torch.isfinite(gnorm):
            opt.step()
        else:   # never update with a non-finite gradient; counted in the log
            skipped += 1
        sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total:
            rec = {"step": step, "of": total, "loss": loss_sum / max(ntok, 1), "lr": sched.get_last_lr()[0],
                   "skipped": skipped, "gnorm": float(gnorm), "prefix_mismatch": mism,
                   "elapsed_min": round((time.time() - t0) / 60, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
    train_min = (time.time() - t0) / 60
    model.language_model = lm.merge_and_unload()
    model.language_model.config.use_cache = True
    model.to(torch.bfloat16)
    model.save_pretrained(out / "final", safe_serialization=True)
    tok.save_pretrained(out / "final")
    info.update({"train_minutes": round(train_min, 1), "skipped_steps": skipped, "prefix_mismatch": mism,
                 "gpu": torch.cuda.get_device_name(0)})
    (out / "train_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(f"saved {out / 'final'} after {train_min:.1f} min, skipped {skipped} steps", flush=True)


def evaluate(args) -> None:
    from qev.calibrate import ece

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    tok, model, torch = load(args.model)
    model.eval()
    images = Images(model, args.media_root)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stop = eos_ids(tok)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "predictions.jsonl", "w")
    conf, ok, n_inv, by = [], [], 0, {}
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        pix = [images(r) for r in batch]
        ids, attn, _ = pad_left([tok(query(model, render(r)[0], pv.shape[0])).input_ids for r, pv in zip(batch, pix)],
                                pad_id)
        with torch.inference_mode():
            g = model.generate(pixel_values=torch.cat(pix).to(torch.bfloat16).cuda(), input_ids=ids,
                               attention_mask=attn, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                               output_scores=True, eos_token_id=stop, pad_token_id=pad_id,
                               return_dict_in_generate=True)
        # generation from inputs_embeds (transformers 4.37): sequences = [BOS] + new tokens, scores = new tokens only
        new = g.sequences[:, -len(g.scores):]
        logp = torch.stack([s.float().log_softmax(-1) for s in g.scores], 1)       # [B, T, V]
        tok_lp = logp.gather(-1, new[:, :logp.shape[1], None]).squeeze(-1)
        for i, r in enumerate(batch):
            seq = new[i].tolist()
            n = next((j for j, t in enumerate(seq) if t in stop), len(seq))
            text = tok.decode(seq[:n], skip_special_tokens=True)
            ans = parse(text, r)
            gold = max(r["target"], key=r["target"].get)
            p = math.exp(float(tok_lp[i, :max(n, 1)].sum()))      # probability of the generated answer sequence
            good = ans == gold
            n_inv += ans is None
            conf.append(p); ok.append(float(good))
            by.setdefault(r.get("task"), []).append(good)
            fh.write(json.dumps({"id": r["id"], "task": r.get("task"), "text": text, "answer": ans, "gold": gold,
                                 "correct": good, "answer_prob": p, "lang": (r.get("meta") or {}).get("lang")},
                                ensure_ascii=False) + "\n")
        if (start // args.batch_size) % 50 == 0:
            print(f"{start + len(batch)}/{len(rows)}", flush=True)
    res = {"n": len(ok), "acc": sum(ok) / len(ok), "invalid": n_inv / len(ok), "ece15_answer_prob": ece(conf, ok),
           "by_task": {k: sum(v) / len(v) for k, v in by.items()}, "model": args.model, "questions": args.questions}
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def bench(args) -> None:
    """Latency, same protocol as gen_sft.bench: warm-up excluded, timed from the decoded image in memory (tiling,
    normalisation, tokenisation and `generate`). "single": one question per request; "image": all the questions of an
    image in one batched generation (left padding, as in eval)."""
    from s1geo.bench_latency import pick, time_scenarios

    groups = pick(Path(args.questions), args.images)
    tok, model, torch = load(args.model)
    model.eval()
    images = Images(model, args.media_root)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stop = eos_ids(tok)

    def run(rows):
        img = images.open(rows[0])   # all rows of a request share the image
        torch.cuda.synchronize(); t = time.perf_counter()
        pix = [images.prepare(img) for _ in rows]
        ids, attn, _ = pad_left([tok(query(model, render(r)[0], pv.shape[0])).input_ids for r, pv in zip(rows, pix)],
                                pad_id)
        with torch.inference_mode():
            model.generate(pixel_values=torch.cat(pix).to(torch.bfloat16).cuda(), input_ids=ids, attention_mask=attn,
                           max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, eos_token_id=stop,
                           pad_token_id=pad_id)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    res = time_scenarios(run, groups, args.warmup)
    res.update({"model": args.model, "gpu": torch.cuda.get_device_name(0), "warmup": args.warmup,
                "questions": args.questions, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def score(args) -> None:
    """Option-scoring read-out, as `s1geo.gen_sft score`: summed log-probability of every option text + end-of-turn
    after the test prompt, with train()'s forward and token count; same option_scores.jsonl."""
    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    tok, model, torch = load(args.model)
    model.eval()
    lm = model.language_model
    images = Images(model, args.media_root)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs = [(i, render(r)[0], t) for i, r in enumerate(rows) for t in options(r)[1]]
    lp_sum, n_tok, mism = [0.0] * len(pairs), [0] * len(pairs), 0
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start:start + args.batch_size]
        seqs, targets, pix = [], [], []
        for i, prompt, t in batch:
            pv = images(rows[i])
            q = query(model, prompt, pv.shape[0])
            p_ids = tok(q).input_ids
            f_ids = tok(q + t + END).input_ids
            if f_ids[:len(p_ids)] == p_ids:
                n = len(f_ids) - len(p_ids)
            else:
                mism += 1
                n = len(tok(t + END, add_special_tokens=False).input_ids)
            seqs.append(f_ids); targets.append(n); pix.append(pv)
        ids, attn, pos = pad_left(seqs, pad_id)
        K = max(targets) + 1
        labels = ids[:, -(K - 1):].clone()
        for j, n in enumerate(targets):
            labels[j, :K - 1 - n] = -100
        if start == 0:
            print("target check:", repr(tok.decode(ids[0, -targets[0]:])), flush=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            emb = embed(model, lm, ids, torch.cat(pix).to(torch.bfloat16).cuda())
            h = lm.model(inputs_embeds=emb, attention_mask=attn, position_ids=pos, use_cache=False)[0]
            logits = lm.lm_head(h[:, -K:]).float()
        tok_lp = -torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels, ignore_index=-100,
                                                    reduction="none")
        for j, n in enumerate(targets):
            lp_sum[start + j], n_tok[start + j] = float(tok_lp[j].sum()), n
        if (start // args.batch_size) % 50 == 0:
            print(f"{start + len(batch)}/{len(pairs)} prefix_mismatch {mism}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_option_scores(out, rows, lp_sum, n_tok, args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", default=MODEL_ID)
    t.add_argument("--train", required=True)
    t.add_argument("--media-root", required=True)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--rows-per-step", type=int, default=64)
    t.add_argument("--micro-batch", type=int, default=16)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lora-r", type=int, default=32)
    t.add_argument("--limit", type=int, default=0, help="first N shuffled rows only (smoke tests)")
    t.add_argument("--out", required=True)
    for name in ("eval", "bench", "score"):
        e = sub.add_parser(name)
        e.add_argument("--model", required=True)
        e.add_argument("--questions", required=True)
        e.add_argument("--media-root", required=True)
        e.add_argument("--batch-size", type=int, default=64)
        e.add_argument("--max-new-tokens", type=int, default=24)
        e.add_argument("--images", type=int, default=110)
        e.add_argument("--warmup", type=int, default=10)
        e.add_argument("--limit", type=int, default=0, help="first N rows only (smoke tests)")
        e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"train": train, "eval": evaluate, "bench": bench, "score": score}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
