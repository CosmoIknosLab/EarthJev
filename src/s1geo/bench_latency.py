"""Latency benchmark (RQ3): EarthJev through OneJev's production engine vs generative VLMs.

Same GPU, same images and questions, warm-up excluded, bf16. Two scenarios per model:
  single  one question per request (batch 1)
  image   one image with all its questions: one EarthJev request (prefix shared, branches in one forward),
          or one generation batch with all the questions for a generative VLM
Reports p50 / p95 latency (ms) per request and the mean per question.

    python -m s1geo.bench_latency earthjev --model runs/onejev/earthjev_08b_reben5pct/final \
        --questions data/earthjev/reben5pct/test.jsonl --media-root data/reben/subset_5pct --out runs/latency/earthjev.json
    python -m s1geo.bench_latency generative --model Qwen/Qwen3-VL-8B-Instruct ... --out runs/latency/qwen3vl.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

import numpy as np


def pick(questions: Path, n_images: int, seed: int = 0) -> list[list[dict]]:
    by = collections.defaultdict(list)
    for line in open(questions):
        r = json.loads(line)
        by[r["meta"]["item_id"]].append(r)
    keys = sorted(by)
    random.Random(seed).shuffle(keys)
    return [by[k] for k in keys[:n_images]]


def time_scenarios(run, groups: list[list[dict]], warmup: int) -> dict:
    """Both scenarios for one model: "single" (the first question of each image, one per request) and "image" (all the
    questions of an image in one request). `run(rows)` answers `rows` in one request and returns its latency in ms."""
    for g in groups[:warmup]:
        run(g); run(g[:1])
    out = {}
    for name, sel in (("single", lambda g: [g[0]]), ("image", lambda g: g)):
        ms, nq = [], []
        for g in groups[warmup:]:
            ms.append(run(sel(g))); nq.append(len(sel(g)))
        out[name] = stats(ms, nq)
    return out


def stats(ms: list[float], n_questions: list[int]) -> dict:
    a = np.array(ms)
    return {"requests": len(ms), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "mean_ms_per_question": float(a.sum() / sum(n_questions)),
            "questions_per_request": float(np.mean(n_questions))}


def bench_earthjev(args, groups):
    import torch

    from qev.mm_engine import MMDecisionEngine
    from qev.schema import SystemOneRequest

    engine_cls = MMDecisionEngine
    if (Path(args.model) / "msi_norm.json").is_file():   # 12-band model (s1geo.msi)
        from s1geo.msi_engine import MSIDecisionEngine as engine_cls
        from s1geo.msi_engine import preload

        for g in groups:   # outside the timer, as the other models' images
            preload(args.media_root.resolve(), [g[0]["media"][0]])
    eng = engine_cls(args.model, media_root=str(args.media_root.resolve()), allow_local_paths=True,
                     gpu_preprocess=True, cuda_graphs=not args.no_cuda_graphs)

    def request(rows):
        r0 = rows[0]
        return SystemOneRequest(state=r0["state"], questions={f"q{i}": r["question"] for i, r in enumerate(rows)},
                                media=[dict(r0["media"][0])])

    def run(rows):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        eng.decide(request(rows))
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    for g in groups[: args.warmup]:
        run(g); run(g[:1])
    out = {}
    for name, sel in (("single", lambda g: [g[0]]), ("image", lambda g: g)):
        ms, nq = [], []
        for g in groups[args.warmup:]:
            ms.append(run(sel(g))); nq.append(len(sel(g)))
        out[name] = stats(ms, nq)
    return out


def bench_generative(args, groups):
    import torch
    import transformers
    from PIL import Image

    from s1geo.vlm_baseline import build_prompt

    proc = transformers.AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"
    model = transformers.AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                                      device_map={"": "cuda:0"}).eval()
    kw = {"enable_thinking": False} if "qwen3" in args.model.lower() else {}

    def run(rows):
        texts, images = [], []
        for r in rows:
            prompt, _, _ = build_prompt(r["question"])
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw))
            images.append(Image.open(args.media_root / r["media"][0]["path"]).convert("RGB"))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,  # stop at the end of the turn
                           eos_token_id=[t for t in {proc.tokenizer.eos_token_id, proc.tokenizer.convert_tokens_to_ids("<|im_end|>"), proc.tokenizer.convert_tokens_to_ids("<|endoftext|>")} if isinstance(t, int) and t >= 0])
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    for g in groups[: args.warmup]:
        run(g[:1])
    out = {}
    for name, sel in (("single", lambda g: [g[0]]), ("image", lambda g: g)):
        ms, nq = [], []
        for g in groups[args.warmup:]:
            ms.append(run(sel(g))); nq.append(len(sel(g)))
        out[name] = stats(ms, nq)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["earthjev", "generative"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--media-root", type=Path, required=True)
    ap.add_argument("--images", type=int, default=110, help="images benchmarked (the first --warmup are warm-up)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--no-cuda-graphs", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import torch

    groups = pick(args.questions, args.images)
    res = bench_earthjev(args, groups) if args.mode == "earthjev" else bench_generative(args, groups)
    res.update({"model": args.model, "mode": args.mode, "gpu": torch.cuda.get_device_name(0), "warmup": args.warmup,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                "questions": str(args.questions)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
