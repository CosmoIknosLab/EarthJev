"""Answer reBEN typed questions directly from a CORINE map ("classify, then compute").

Given a level-3 (or level-2 code) mask and a question row from scripts/data/make_reben_questions.py, return
the answer key. Used (1) as a baseline with the U-Net's predicted maps and (2) with the reference mask as a
consistency check of the question generator (must score 100%).

    python -m s1geo.map_answers --seg-run runs/seg/<run> --subset data/reben/subset_5pct \
        --questions data/earthjev/reben5pct/test.jsonl --out runs/eval/reben_map_unet
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import zarr

from s1geo.clc import LEVEL_CODES, NAMES, fractions, to_level

PRESENT_DECISION = 0.025  # predicted share above which a class counts as present (midway between 0 and 5%)
SHARE_KEYS = ["none", "under_10", "10_30", "30_60", "over_60"]
SHARE_EDGES = [0.10, 0.30, 0.60]
QUADRANTS = ["north_west", "north_east", "south_west", "south_east"]


def key(code: int) -> str:
    return NAMES[code].lower().replace(",", "").replace("/", " or ").replace(" ", "_")


def answer(mask: np.ndarray, meta: dict) -> str:
    """`mask` holds CLC level-3 codes (or any finer code whose prefixes give levels 1-2)."""
    q, lvl, cls = meta["qtype"], meta["level"], meta["classes"]
    fr = fractions(mask, lvl)
    if q == "dominant":
        return key(max(fr, key=fr.get)) if fr else key(LEVEL_CODES[lvl][0])
    if q == "presence":
        return "yes" if fr.get(cls[0], 0.0) > PRESENT_DECISION else "no"
    if q == "share":
        f = fr.get(cls[0], 0.0)
        return SHARE_KEYS[0] if f == 0 else SHARE_KEYS[1 + sum(f > e for e in SHARE_EDGES)]
    if q == "location":
        lv = to_level(mask, lvl) == cls[0]
        h, w = lv.shape
        q4 = [lv[:h // 2, :w // 2].sum(), lv[:h // 2, w // 2:].sum(), lv[h // 2:, :w // 2].sum(), lv[h // 2:, w // 2:].sum()]
        return QUADRANTS[int(np.argmax(q4))]
    a, b = cls
    return key(a if fr.get(a, 0.0) >= fr.get(b, 0.0) else b)


@torch.no_grad()
def predict_maps(seg_run: Path, zarr_path: Path) -> np.ndarray:
    """Level-2 CLC codes predicted by the segmenter of `seg_run` (best_model.pt) for every patch."""
    from s1geo.data.reben import RebenSeg, class_codes
    from s1geo.models.unet import UNet

    cfg = yaml.safe_load(open(seg_run / "config.yaml"))
    st = json.loads((seg_run / "band_stats.json").read_text())
    ds = RebenSeg(zarr_path, cfg["level"], np.array(st["mean"]), np.array(st["std"]))
    model = UNet(12, len(class_codes(cfg["level"])), width=cfg["width"], depth=cfg["depth"]).cuda().eval()
    model.load_state_dict(torch.load(seg_run / "best_model.pt", map_location="cuda"))
    codes = torch.tensor(class_codes(cfg["level"]))
    out = []
    for x, _ in torch.utils.data.DataLoader(ds, batch_size=128, num_workers=4):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(codes[model(x.cuda()).argmax(1).cpu()].numpy())
    # level-2 code -> a representative level-3-like code (code * 10 + 1) so that to_level/fractions apply
    pred = np.concatenate(out) * 10 + 1
    if cfg["level"] != 2:
        raise ValueError("predict_maps expects a level-2 segmenter")
    return pred


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seg-run", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    idx = pd.read_parquet(args.subset / f"{args.split}_index.parquet")
    row_of = {pid: i for i, pid in enumerate(idx["patch_id"])}
    ref = zarr.open_group(str(args.subset / f"{args.split}.zarr"), mode="r")["clc"][:]
    pred = predict_maps(args.seg_run, args.subset / f"{args.split}.zarr")
    acc = {"pred": collections.defaultdict(list), "ref": collections.defaultdict(list)}
    rows = [json.loads(l) for l in open(args.questions)]
    args.out.mkdir(parents=True, exist_ok=True)
    fh = open(args.out / "predictions.jsonl", "w")   # per question, U-Net map (no probabilities: answer_prob null)
    for r in rows:
        i = row_of[r["meta"]["item_id"]]
        gold = max(r["target"], key=r["target"].get)
        for name, m in (("pred", pred[i]), ("ref", ref[i])):
            a = answer(m, r["meta"])
            acc[name][r["task"]].append(a == gold)
            acc[name]["all"].append(a == gold)
            if name == "pred":
                fh.write(json.dumps({"id": r["id"], "task": r["task"], "answer": a, "gold": gold,
                                     "correct": a == gold, "answer_prob": None}) + "\n")
    fh.close()
    res = {name: {t: {"n": len(v), "acc": float(np.mean(v))} for t, v in d.items()} for name, d in acc.items()}
    (args.out / "map_answers.json").write_text(json.dumps(res, indent=1))
    for t in sorted(res["pred"], key=lambda t: (t != "all", t)):
        print(f"{t:9s} n {res['pred'][t]['n']:6d}  from U-Net map {res['pred'][t]['acc']:.4f}   "
              f"from reference map {res['ref'][t]['acc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
