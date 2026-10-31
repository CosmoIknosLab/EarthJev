"""Generative fine-tuning baseline: the same VLM, data and inputs as EarthJev, trained to *generate* the answer text.

The controlled counterpart of EarthJev: Qwen3.5-0.8B, full fine-tuning, 1 epoch, 64 rows per optimiser step, option
order permuted at random, the same 12-band image (--msi-norm; same widened and trained input layer as EarthJev, rest of
the vision tower frozen) and the same state JSON. The only difference is the objective and the readout: next-token cross-entropy on
the answer text ("Water bodies", "yes", ...), greedy decoding and parsing at test time, instead of the single-position
letter readout with cross-entropy + Brier.

    python -m s1geo.gen_sft train --train data/earthjev/reben5pct_msi/train.jsonl --media-root data/reben/subset_5pct \
        --msi-norm data/reben/subset_5pct/msi_norm.json --lr 5e-6 --out runs/gen_sft/<name>
    python -m s1geo.gen_sft train --train data/earthjev/reben5pct_rgb/train.jsonl ... \
        --model Qwen/Qwen3-VL-2B-Instruct --lora --lr 1e-4    # MLLM baselines (LoRA, RGB)
    python -m s1geo.gen_sft eval --model runs/gen_sft/<name>/final --questions data/earthjev/reben5pct_msi/test.jsonl \
        --media-root data/reben/subset_5pct --out runs/eval/<name>_test
    python -m s1geo.gen_sft bench --model runs/gen_sft/<name>/final --questions ... --out runs/latency/<name>.json
    python -m s1geo.gen_sft score --model runs/gen_sft/<name>/final --questions ... --media-root ... \
        --out runs/eval/<name>_test      # option-scoring read-out -> option_scores.jsonl

12-band rows (s1geo.msi; media with a Zarr "row"): train with --msi-norm <band statistics JSON>. The vision encoder's
input layer is widened to 12 channels and trained, the rest of the encoder stays frozen (as EarthJev MSI); the band
statistics are saved as <out>/final/msi_norm.json and picked up from the model directory by eval, score and bench.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

SYSTEM = None  # no system prompt: the chat template's default


def render(row: dict, rng: random.Random | None = None) -> tuple[str, str, list[str]]:
    """(prompt text, target text, option keys in the shown order). The state goes in as JSON without the image key."""
    st = {k: v for k, v in row["state"].items() if not (isinstance(v, str) and v.startswith("<image:"))}
    q = row["question"]
    head = f"State (JSON):\n{json.dumps(st, ensure_ascii=False, indent=2)}\n\nQuestion: {q['instructions']}\n"
    gold = max(row["target"], key=row["target"].get)
    if q["type"] == "noul":
        return head + "Answer with yes or no.", gold, ["yes", "no"]
    keys = list(q["criteria"])
    if rng is not None:
        rng.shuffle(keys)
    opts = "\n".join(f"- {str(q['criteria'][k]).rstrip('.')}" for k in keys)
    return head + f"Options:\n{opts}\nAnswer with one of the options.", str(q["criteria"][gold]).rstrip("."), keys


def parse(text: str, row: dict) -> str | None:
    q = row["question"]
    t = text.strip().strip("*").strip()
    if q["type"] == "noul":
        w = t.split()[0].strip(".,!").lower() if t else ""
        return {"yes": "yes", "no": "no", "sì": "yes", "si": "yes"}.get(w)
    norm = lambda s: " ".join(str(s).lower().rstrip(".").split())
    exact = [k for k, d in q["criteria"].items() if norm(d) == norm(t)]
    if len(exact) == 1:
        return exact[0]
    hits = [k for k, d in q["criteria"].items() if norm(d) in norm(t)]
    return hits[0] if len(hits) == 1 else None


def load(model_id: str, dtype, msi_norm: str | None = None):
    """Processor, model on cuda:0 and torch. With band statistics (--msi-norm, or msi_norm.json in the model directory)
    the input layer is widened to 12 channels if needed and the processor reads 12-band patches; the plain processor,
    for saving, is then `proc.base_processor`."""
    import torch
    import transformers

    proc = transformers.AutoProcessor.from_pretrained(model_id)
    model = transformers.AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    norm_path = Path(msi_norm) if msi_norm else Path(model_id) / "msi_norm.json"
    if msi_norm or norm_path.is_file():
        from s1geo import msi

        if msi._patch_embed(model).proj.in_channels == 3:
            msi.widen_input_layer(model)
        base, norm = proc, msi.BandNorm.load(norm_path)
        proc = msi.msi_processor(base, norm)
        proc.base_processor, proc.msi_norm = base, norm
        print(f"12-band input, band statistics {norm_path}", flush=True)
    return proc, model.to("cuda:0"), torch


def image(media_root: str, media: dict):
    """The model input for one media item: a 12-band patch (s1geo.msi) or an RGB image."""
    if "row" in media:
        from s1geo import msi

        return msi.load_patch(media_root, media)
    from PIL import Image

    return Image.open(Path(media_root) / media["path"]).convert("RGB")


def chat(proc, prompt: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def train(args) -> None:
    import torch

    rows = [json.loads(l) for l in open(args.train)]
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    rng.shuffle(rows)
    if args.lora:   # LoRA on the language model's linear layers (OneJev's LoRA defaults), bf16 base weights
        from peft import LoraConfig, get_peft_model
        proc, model, _ = load(args.model, torch.bfloat16)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
        for prm in model.parameters():   # LoRA weights in fp32 (AdamW on bf16 weights diverged for InternVL3)
            if prm.requires_grad:
                prm.data = prm.data.float()
        model.print_trainable_parameters()
    else:
        proc, model, _ = load(args.model, torch.float32, args.msi_norm)   # fp32 master weights, bf16 autocast (as OneJev full FT)
        for n, prm in model.named_parameters():
            prm.requires_grad = not (".visual." in f".{n}." or n.startswith("visual") or ".vision_tower." in f".{n}.")
        if args.msi_norm:   # the widened input layer is the only trained part of the vision encoder
            from s1geo import msi

            for prm in msi._patch_embed(model).parameters():
                prm.requires_grad = True
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    tok = proc.tokenizer
    tok.padding_side = "left"   # answers end every row, so only the last positions need logits
    end = "<|im_end|>"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    micro, accum = args.micro_batch, args.rows_per_step // args.micro_batch
    total = math.ceil(len(rows) / args.rows_per_step)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    log = open(out / "log.jsonl", "w")
    model.train()
    step, t0, skipped = 0, time.time(), 0
    for start in range(0, len(rows), args.rows_per_step):
        chunk = rows[start:start + args.rows_per_step]
        loss_sum, ntok = 0.0, 0
        for m in range(0, len(chunk), micro):
            batch = chunk[m:m + micro]
            texts, images, targets = [], [], []
            for r in batch:
                prompt, target, _ = render(r, rng)
                texts.append(chat(proc, prompt) + target + end)
                targets.append(len(tok(target + end, add_special_tokens=False)["input_ids"]))
                images.append(image(args.media_root, r["media"][0]))
            enc = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda:0")
            K = max(targets) + 1
            labels = enc["input_ids"][:, -(K - 1):].clone()      # tokens predicted by the last K-1 kept positions
            for i, n in enumerate(targets):
                labels[i, :K - 1 - n] = -100
            if step == 0 and m == 0:   # the supervised tokens must be exactly the answer + end-of-turn
                for i in range(min(3, len(batch))):
                    print("target check:", repr(tok.decode(enc["input_ids"][i, -targets[i]:])), "| context:",
                          repr(tok.decode(enc["input_ids"][i, -targets[i] - 6:-targets[i]])), flush=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc, logits_to_keep=K).logits.float()
            lp = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels, ignore_index=-100,
                                                   reduction="sum")
            n = int((labels != -100).sum())
            (lp / (n * accum)).backward()
            loss_sum += float(lp)
            ntok += n
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gnorm):
            opt.step()
        else:   # never update with a non-finite gradient; counted in the log
            skipped += 1
        sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total:
            rec = {"step": step, "of": total, "loss": loss_sum / max(ntok, 1), "lr": sched.get_last_lr()[0], "skipped": skipped,
                   "elapsed_min": round((time.time() - t0) / 60, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
    if args.lora:
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(out / "final", safe_serialization=True)
    getattr(proc, "base_processor", proc).save_pretrained(out / "final")
    if hasattr(proc, "msi_norm"):
        (out / "final" / "msi_norm.json").write_text(proc.msi_norm.to_json())
    print(f"saved {out / 'final'}", flush=True)


def evaluate(args) -> None:
    import torch

    from qev.calibrate import ece

    rows = [json.loads(l) for l in open(args.questions)]
    proc, model, _ = load(args.model, torch.bfloat16)
    model.eval()
    tok = proc.tokenizer
    tok.padding_side = "left"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "predictions.jsonl", "w")
    conf, ok, n_inv, by = [], [], 0, {}
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        texts = [chat(proc, render(r)[0]) for r in batch]
        images = [image(args.media_root, r["media"][0]) for r in batch]
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            g = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, output_scores=True,
                               eos_token_id=[t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")} if isinstance(t, int) and t >= 0],
                               return_dict_in_generate=True)
        new = g.sequences[:, enc["input_ids"].shape[1]:]
        logp = torch.stack([s.float().log_softmax(-1) for s in g.scores], 1)       # [B, T, V]
        tok_lp = logp.gather(-1, new[:, :logp.shape[1], None]).squeeze(-1)
        for i, r in enumerate(batch):
            ids = new[i].tolist()
            n = next((j for j, t in enumerate(ids) if t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))), len(ids))
            text = tok.decode(ids[:n], skip_special_tokens=True)
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


def options(row: dict) -> tuple[list[str], list[str]]:
    """(option keys, answer texts) in the order render() shows them at test time; texts as the training targets."""
    q = row["question"]
    if q["type"] == "noul":
        return ["yes", "no"], ["yes", "no"]
    keys = list(q["criteria"])
    return keys, [str(q["criteria"][k]).rstrip(".") for k in keys]


def score(args) -> None:
    """Option-scoring read-out: log-likelihood of every admissible answer.

    For each question and option, the summed log-probability of the option text + end-of-turn token after the test
    prompt (the tokens the model was trained on). Writes option_scores.jsonl in the probe.logits format: `logits` =
    summed log-likelihood, `logits_norm` = divided by the number of tokens; softmax over options gives the read-out."""
    import torch

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    proc, model, _ = load(args.model, torch.bfloat16)
    model.eval()
    tok = proc.tokenizer
    tok.padding_side = "left"
    end = "<|im_end|>"
    pairs = []          # (row index, prompt, answer text)
    for i, r in enumerate(rows):
        prompt = render(r)[0]
        pairs += [(i, prompt, t) for t in options(r)[1]]
    lp_sum, n_tok = [0.0] * len(pairs), [0] * len(pairs)
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start:start + args.batch_size]
        texts = [chat(proc, p) + t + end for _, p, t in batch]
        targets = [len(tok(t + end, add_special_tokens=False)["input_ids"]) for _, _, t in batch]
        images = [image(args.media_root, rows[i]["media"][0]) for i, _, _ in batch]
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda:0")
        K = max(targets) + 1
        labels = enc["input_ids"][:, -(K - 1):].clone()
        for j, n in enumerate(targets):
            labels[j, :K - 1 - n] = -100
        if start == 0:
            print("target check:", repr(tok.decode(enc["input_ids"][0, -targets[0]:])), flush=True)
        with torch.inference_mode():
            logits = model(**enc, logits_to_keep=K).logits.float()
        tok_lp = -torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels, ignore_index=-100,
                                                    reduction="none")
        for j, n in enumerate(targets):
            lp_sum[start + j], n_tok[start + j] = float(tok_lp[j].sum()), n
        if (start // args.batch_size) % 50 == 0:
            print(f"{start + len(batch)}/{len(pairs)}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_option_scores(out, rows, lp_sum, n_tok, args)


def write_option_scores(out: Path, rows: list[dict], lp_sum: list[float], n_tok: list[int], args) -> None:
    """option_scores.jsonl (one record per question, pairs in options() order) + option_scores_metrics.json."""
    from qev.calibrate import ece

    res, k = {}, 0
    with open(out / "option_scores.jsonl", "w") as fh:
        for r in rows:
            keys = options(r)[0]
            s, n = lp_sum[k:k + len(keys)], n_tok[k:k + len(keys)]
            k += len(keys)
            fh.write(json.dumps({"id": r["id"], "task": r.get("task"), "kind": r["question"]["type"], "labels": keys,
                                 "logits": s, "logits_norm": [a / b for a, b in zip(s, n)], "n_tokens": n,
                                 "target": [float(r["target"].get(c, 0.0)) for c in keys],
                                 "item_id": (r.get("meta") or {}).get("item_id")}) + "\n")
            gold = keys.index(max(r["target"], key=r["target"].get))
            for name, v in (("summed", s), ("length_normalised", [a / b for a, b in zip(s, n)])):
                m = max(v); e = [math.exp(x - m) for x in v]; p = [x / sum(e) for x in e]
                top = max(range(len(p)), key=p.__getitem__)
                res.setdefault(name, {"conf": [], "ok": []})
                res[name]["conf"].append(p[top]); res[name]["ok"].append(float(top == gold))
    metrics = {name: {"acc": sum(d["ok"]) / len(d["ok"]), "ece15": ece(d["conf"], d["ok"])} for name, d in res.items()}
    metrics.update({"n": len(rows), "model": args.model, "questions": args.questions})
    (out / "option_scores_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


def bench(args) -> None:
    """Latency, same protocol as s1geo.bench_latency: warm-up excluded, the time runs from the decoded image in memory to
    the generated answers (preprocessing and tokenisation included). "single": one question per request; "image": all
    the questions of an image in one batched generation (left padding)."""
    import torch

    from s1geo.bench_latency import pick, time_scenarios

    groups = pick(Path(args.questions), args.images)
    proc, model, _ = load(args.model, torch.bfloat16)
    model.eval()
    tok = proc.tokenizer
    tok.padding_side = "left"
    stop = [t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
            if isinstance(t, int) and t >= 0]

    def run(rows):
        img = image(args.media_root, rows[0]["media"][0])   # all rows of a request share the image
        torch.cuda.synchronize(); t = time.perf_counter()
        enc = proc(text=[chat(proc, render(r)[0]) for r in rows], images=[img] * len(rows), return_tensors="pt",
                   padding=True).to("cuda:0")
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=stop)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    res = time_scenarios(run, groups, args.warmup)
    res.update({"model": args.model, "gpu": torch.cuda.get_device_name(0), "warmup": args.warmup,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "questions": str(args.questions)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    t.add_argument("--train", required=True)
    t.add_argument("--media-root", required=True)
    t.add_argument("--lr", type=float, default=5e-6)
    t.add_argument("--rows-per-step", type=int, default=64)
    t.add_argument("--micro-batch", type=int, default=16)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tuning (vision encoder untouched)")
    t.add_argument("--lora-r", type=int, default=32)
    t.add_argument("--msi-norm", default=None, help="12-band rows: band statistics JSON (s1geo.msi.make_norm)")
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
        e.add_argument("--limit", type=int, default=0, help="score: first N questions only (smoke tests)")
        e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"train": train, "eval": evaluate, "bench": bench, "score": score}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
