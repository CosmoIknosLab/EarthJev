"""RSVQAxBEN questions whose image is one of our reBEN patches, for one split (train / val / test).

RSVQAxBEN images carry BigEarthNet v1 names ("S2A_MSIL2A_20170613T101031_23_39.tif"); reBEN patch ids add the
processing baseline, orbit and tile ("S2A_MSIL2A_20170613T101031_N9999_R022_T34VER_23_39"). The match uses the
official v1 name of each reBEN patch (column s2v1_name of reBEN's metadata.parquet). It also reports, as a leakage
check, how many RSVQAxBEN images of this split fall in each reBEN split (only those in --subset-split are kept). Output (same layout as data/rsvqaxben/test_on_reben5pct_test.json):
{"img_to_v2": {img_id: patch_id}, "rows": [{"qid", "img_id", "type", "question", "answer"}]}.

    python scripts/data/make_rsvqaxben_subset.py --raw data/rsvqaxben/raw --split train \
        --subset data/reben/subset_5pct --subset-split train --out data/rsvqaxben/train_on_reben5pct_train.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd


def key_v1(name: str) -> tuple[str, str, str]:
    parts = name.removesuffix(".tif").split("_")
    return "_".join(parts[:3]), parts[-2], parts[-1]


def key_v2(pid: str) -> tuple[str, str, str]:
    parts = pid.split("_")
    return "_".join(parts[:3]), parts[-2], parts[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--subset-split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--metadata", type=Path, default=Path("data/reben/raw/metadata.parquet"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    meta = pd.concat([pd.read_parquet(args.metadata),
                      pd.read_parquet(args.metadata.with_name("metadata_for_patches_with_snow_cloud_or_shadow.parquet"))])
    v1_to_v2 = dict(zip(meta["s2v1_name"], meta["patch_id"]))
    v2_split = dict(zip(meta["patch_id"], meta["split"]))
    pids = set(pd.read_parquet(args.subset / f"{args.subset_split}_index.parquet")["patch_id"])
    images = json.loads((args.raw / f"RSVQAxBEN_split_{args.split}_images.json").read_text())["images"]
    img_to_v2, where = {}, collections.Counter()
    for im in images:
        if not im.get("active", True):
            continue
        v2 = v1_to_v2.get(im["original_name"].removesuffix(".tif"))
        where[v2_split.get(v2, "not in reBEN")] += 1
        if v2 in pids:
            img_to_v2[im["id"]] = v2
            # cross-check: same patch row/column (v2 zero-pads them; the datatake timestamp can differ by a second)
            assert tuple(map(int, key_v1(im["original_name"])[1:])) == tuple(map(int, key_v2(v2)[1:])), (im["original_name"], v2)
    print(f"RSVQAxBEN {args.split}: {len(images)} images; reBEN split of these images: {dict(where)}", flush=True)
    print(f"kept {len(img_to_v2)} images of our {args.subset_split} subset", flush=True)
    questions = json.loads((args.raw / f"RSVQAxBEN_split_{args.split}_questions.json").read_text())["questions"]
    answers = {a["id"]: a for a in json.loads((args.raw / f"RSVQAxBEN_split_{args.split}_answers.json").read_text())["answers"]}
    rows = []
    for q in questions:
        if q.get("active", True) and q["img_id"] in img_to_v2:
            a = answers[q["answers_ids"][0]]
            rows.append({"qid": q["id"], "img_id": q["img_id"], "type": q["type"], "question": q["question"],
                         "answer": a["answer"]})
    args.out.write_text(json.dumps({"img_to_v2": {str(k): v for k, v in img_to_v2.items()}, "rows": rows}))
    print(f"{len(rows)} questions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
