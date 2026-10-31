"""Patch-id lists for prepare_reben.py --ids: the whole reBEN train split (segmenter of the U-Net + rules reference)
and the 1,065 images of the BigEarthNet.txt benchmark split (the patches of make_bentxt.py's bench.jsonl).

    python scripts/data/make_id_lists.py --metadata data/reben/raw/metadata.parquet \
        --bench data/earthjev/bentxt/bench.jsonl --out-train data/reben/raw/train_split_ids.txt \
        --out-bench data/bentxt/bench_patch_ids.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--out-train", type=Path, required=True)
    ap.add_argument("--out-bench", type=Path, required=True)
    args = ap.parse_args()
    meta = pd.read_parquet(args.metadata)
    train = sorted(meta.loc[meta["split"] == "train", "patch_id"])
    bench = sorted({json.loads(line)["meta"]["item_id"] for line in open(args.bench)})
    args.out_train.write_text("\n".join(train) + "\n")
    args.out_bench.write_text("\n".join(bench) + "\n")
    print(f"train split: {len(train)} patches; BigEarthNet.txt benchmark: {len(bench)} patches")


if __name__ == "__main__":
    main()
