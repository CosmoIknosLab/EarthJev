"""BigEarthNet.txt binary and multiple-choice questions as OneJev rows.

Writes <out>/:
  bench.jsonl       benchmark split (manually verified), EarthJev format: binary -> noul, mcq -> choice whose
                    criteria are the a) b) c) d) options split out of the question text
  bench_gen.jsonl   same questions for generative baselines: the original question text with its inline options
                    (question.inline_options = True), as in the BigEarthNet.txt paper protocol
  train.jsonl       train split restricted to the patches of our reBEN train subset (for fine-tuning)
  validation_1k.jsonl  questions on 1,000 random images of our validation subset (validation split), for
                    in-training evaluation
The state holds only sensor information and the image: no country and no date, which would give away the
country / season / climate-zone questions.

    python scripts/data/make_bentxt.py --parquet data/bentxt/raw/BigEarthNet.txt.parquet --out data/earthjev/bentxt
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
STATE = {"sensor": "Sentinel-2 L2A, true-colour composite (bands B4, B3, B2)", "ground_sample_distance": "10 m",
         "patch": "120 x 120 pixels (1.2 km x 1.2 km), north up", "image": "<image:1>"}
OPT = re.compile(r"(?:^|,\s*|\s)([a-d])\)\s")


def split_mcq(text: str) -> tuple[str, dict[str, str]]:
    """'Q? a) X, b) Y, c) Z, d) W' -> ('Q?', {'a': 'X', ...}). Options may contain commas themselves."""
    marks = [(m.group(1), m.start(), m.end()) for m in OPT.finditer(text)]
    letters = [m[0] for m in marks]
    if letters != ["a", "b", "c", "d"][: len(letters)] or len(letters) < 2:
        raise ValueError(f"cannot split options: {text!r}")
    stem = text[: marks[0][1]].strip().rstrip(":").strip()
    opts = {}
    for i, (l, _, end) in enumerate(marks):
        stop = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        opts[l] = text[end:stop].strip().rstrip(",").strip()
    return stem, opts


def to_rows(r, media_path: str, gen: bool) -> dict:
    base = {"id": f"bentxt/{r.ID}", "source": "bentxt", "task": f"{r.type}_{r.category}", "split": r.split,
            "license": "cdla-permissive-1.0", "state": STATE,
            "media": [{"type": "image", "path": media_path, "w": 120, "h": 120}],
            "meta": {"item_id": r.patch_id, "type": r.type, "category": r.category}}
    out = str(r.output).strip().lower()
    if r.type == "binary":
        if out not in ("yes", "no"):
            raise ValueError(f"binary answer {out!r}")
        q = {"type": "noul", "instructions": r.input.strip()}
        tgt = {"yes": float(out == "yes"), "no": float(out == "no")}
    else:
        stem, opts = split_mcq(r.input.strip())
        if out not in opts:
            raise ValueError(f"mcq answer {out!r} not in {list(opts)}")
        q = ({"type": "choice", "instructions": r.input.strip(), "criteria": opts, "inline_options": True} if gen
             else {"type": "choice", "instructions": stem, "criteria": opts})
        tgt = {out: 1.0}
    return {**base, "question": q, "target": tgt}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--subset", type=Path, default=REPO / "data/reben/subset_5pct")
    ap.add_argument("--bench-media", default="../bentxt_bench/rgb/test", help="bench image dir relative to --media-root")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    d = pd.read_parquet(args.parquet)
    d = d[d.type.isin(["binary", "mcq"])]
    bench = d[d.split == "bench"]
    with open(args.out / "bench.jsonl", "w") as f1, open(args.out / "bench_gen.jsonl", "w") as f2:
        for r in bench.itertuples():
            path = f"{args.bench_media}/{r.patch_id}.png"
            f1.write(json.dumps(to_rows(r, path, gen=False), ensure_ascii=False) + "\n")
            f2.write(json.dumps(to_rows(r, path, gen=True), ensure_ascii=False) + "\n")
    print("bench", len(bench), bench.groupby("type").size().to_dict(), flush=True)
    for split, ours in (("train", "train"), ("validation", "validation")):
        ids = set(pd.read_parquet(args.subset / f"{ours}_index.parquet").patch_id)
        part = d[(d.split == split) & d.patch_id.isin(ids)]
        if split == "validation":
            keep = set(random.Random(0).sample(sorted(part.patch_id.unique()), 1000))
            part = part[part.patch_id.isin(keep)]
        name = "train.jsonl" if split == "train" else "validation_1k.jsonl"
        with open(args.out / name, "w") as fh:
            for r in part.itertuples():
                fh.write(json.dumps(to_rows(r, f"rgb/{ours}/{r.patch_id}.png", gen=False), ensure_ascii=False) + "\n")
        print(split, len(part), "rows on", part.patch_id.nunique(), "images", flush=True)


if __name__ == "__main__":
    main()
