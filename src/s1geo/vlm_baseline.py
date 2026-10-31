"""Generative VLM baselines on OneJev-format question rows (our CORINE test and RSVQAxBEN rows).

The model sees the image and a prompt with the question and, for choice questions, the options labelled with
letters; it generates an answer (greedy) that is parsed back to an option. Recorded per row: generated text,
parsed answer (None if unparseable), whether it is correct, and the probabilities of the option letters /
yes-no at the first generated position (for calibration). Wall-clock time per batch is recorded too.

    python -m s1geo.vlm_baseline --model Qwen/Qwen3.5-0.8B --questions data/earthjev/reben5pct/test.jsonl \
        --media-root data/reben/subset_5pct --sample 6000 --out runs/baselines/vlm/qwen35_08b_reben
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_prompt(q: dict) -> tuple[str, list[str], list[str]]:
    """Return (prompt text, option keys, option letters). Noul -> yes/no."""
    if q["type"] == "noul":
        return f"{q['instructions']}\nAnswer with yes or no.", ["yes", "no"], ["yes", "no"]
    if q["type"] == "open_lc":  # RSVQAxBEN land-cover question: free list of class names
        cands = q.get("candidates")
        listing = f"\nPossible classes: {'; '.join(cands)}." if cands else ""
        return (f"{q['instructions']}{listing}\nAnswer only with the land-cover class names present in the image, "
                f"separated by commas, or with None if there are none."), [], []
    keys = list(q["criteria"].keys())
    if q.get("inline_options"):  # options already written in the question text, e.g. "a) X, b) Y" (BigEarthNet.txt)
        return f"{q['instructions']}\nAnswer with the letter of the correct option only.", keys, [k.upper() for k in keys]
    letters = list(LETTERS[: len(keys)])
    opts = "\n".join(f"{l}. {str(q['criteria'][k]).rstrip('.')}" for l, k in zip(letters, keys))
    return (f"{q['instructions']}\nOptions:\n{opts}\nAnswer with the letter of the correct option only.", keys, letters)


def parse_answer(text: str, q: dict, keys: list[str], letters: list[str]) -> str | None:
    t = text.strip()
    if q["type"] == "open_lc":
        from s1geo.rsvqa import parse_classes

        last = [line for line in t.splitlines() if line.strip()]
        found, rest = parse_classes(re.sub(r"[\*\.]", "", last[-1] if last else ""))
        return None if rest else "|".join(sorted(found))
    if q["type"] == "noul":
        m = re.match(r"\W*(yes|no|sì|si)\b", t, re.I)
        return None if not m else ("no" if m.group(1).lower() == "no" else "yes")
    m = re.match(r"\W*([A-Za-z])(?:[\.\):,\s]|$)" if q.get("inline_options") else r"\W*([A-Z])(?:[\.\):\s]|$)", t)
    if m and m.group(1).upper() in letters:
        return keys[letters.index(m.group(1).upper())]
    # Reasoning-style outputs: an explicit "answer is X", or a line holding just the letter.
    for pat in (r"answer(?:\s+is)?\s*[:\-]?\s*\**\(?([A-Z])\)?(?:[\.\):\s\*]|$)", r"(?m)^\W*([A-Z])\W*$"):
        found = [x for x in re.findall(pat, t, re.I if pat.startswith("answer") else 0) if x.upper() in letters]
        if found:
            return keys[letters.index(found[-1].upper())]
    low = t.lower()
    names = {k: {str(q["criteria"][k]).lower().rstrip("."), k.replace("_", " ")} for k in keys}
    # The last bold span that names exactly one option (e.g. "... is **agricultural areas**").
    for span in reversed(re.findall(r"\*\*(.+?)\*\*", low)):
        span = re.sub(r"^[a-z]\.\s*", "", span.strip()).rstrip(".")
        hits = [k for k, ns in names.items() if span in ns]
        if len(hits) == 1:
            return hits[0]
    hits = [k for k, ns in names.items() if any(n in low for n in ns)]
    return hits[0] if len(hits) == 1 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--media-root", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=0, help="random subset of rows (seed 0); 0 = all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch
    import transformers
    from PIL import Image

    rows = [json.loads(l) for l in open(args.questions)]
    if args.sample and args.sample < len(rows):
        rows = random.Random(0).sample(rows, args.sample)
    proc = transformers.AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"
    model = transformers.AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                                      device_map={"": "cuda:0"}).eval()
    tok = proc.tokenizer

    def first_token_ids(words: list[str]) -> list[int]:
        return [tok.encode(w, add_special_tokens=False)[0] for w in words]

    args.out.mkdir(parents=True, exist_ok=True)
    out = open(args.out / "predictions.jsonl", "w")
    t_total, n_done = 0.0, 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        texts, images, meta = [], [], []
        for r in batch:
            prompt, keys, letters = build_prompt(r["question"])
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            kw = {"enable_thinking": False} if "qwen3" in args.model.lower() else {}
            texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw))
            images.append(Image.open(args.media_root / r["media"][0]["path"]).convert("RGB"))
            meta.append((keys, letters))
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda:0")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,  # stop at the end of the turn
                                 eos_token_id=[t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")} if isinstance(t, int) and t >= 0],
                                 output_scores=True, return_dict_in_generate=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        t_total += dt
        first = gen.scores[0].float().softmax(-1).cpu()
        new = gen.sequences[:, enc["input_ids"].shape[1]:]
        for i, r in enumerate(batch):
            keys, letters = meta[i]
            text = tok.decode(new[i], skip_special_tokens=True)
            ans = parse_answer(text, r["question"], keys, letters)
            if r["question"]["type"] == "open_lc":
                p, gold = [], "|".join(sorted(r["target"]["gold"]))
            else:
                ids = first_token_ids(["Yes", "No"] if r["question"]["type"] == "noul" else letters)
                p = first[i, ids]
                p = (p / p.sum()).tolist() if float(p.sum()) > 0 else [1.0 / len(ids)] * len(ids)
                gold = max(r["target"], key=r["target"].get)
            out.write(json.dumps({"id": r["id"], "task": r.get("task"), "kind": r["question"]["type"], "labels": keys,
                                  "first_token_probs": p, "text": text, "answer": ans, "gold": gold,
                                  "correct": ans == gold, "lang": (r.get("meta") or {}).get("lang")}, ensure_ascii=False) + "\n")
        n_done += len(batch)
        if (start // args.batch_size) % 20 == 0:
            print(f"{n_done}/{len(rows)} rows, {t_total / n_done * 1000:.1f} ms/row", flush=True)
    out.close()
    summary = {"model": args.model, "questions": str(args.questions), "rows": n_done, "batch_size": args.batch_size,
               "seconds_generate": round(t_total, 2), "ms_per_row": round(t_total / max(n_done, 1) * 1000, 2),
               "gpu": torch.cuda.get_device_name(0)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
