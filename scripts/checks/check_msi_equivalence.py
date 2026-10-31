"""Before training, the 12-band model (s1geo.msi: input layer widened, extra bands zero) must answer like the RGB model.

Scores the same reBEN rows with the same base checkpoint: (A) the RGB model on the PNG media; (D) the RGB model on
the B04/B03/B02 channels of the 12-band input (float pixels, no PNG); (B) the widened model on the 12-band input.
B vs D isolates the widening (must agree up to bf16 noise); A vs D is the effect of the PNG's 8-bit rounding on the
pixels (tests/test_msi.py), which the untrained model is sensitive to.
Run from third_party/OneJev on a GPU node:
    python ../../scripts/checks/check_msi_equivalence.py --rows ../../data/earthjev/reben5pct_en/test.jsonl \
        --subset ../../data/reben/subset_5pct --n 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from train import common, mm
    from train.sft import collect_mm, load_eval_rows

    from s1geo import msi

    rows = load_eval_rows(str(args.rows), args.n)
    ids = pd.read_parquet(args.subset / f"{args.split}_index.parquet")["patch_id"].tolist()
    row_of = {pid: i for i, pid in enumerate(ids)}
    msi_rows = []
    for r in rows:
        (m,) = r["media"]
        msi_rows.append({**r, "media": [{"type": "image", "path": f"{args.split}.zarr",
                                         "row": row_of[r["meta"]["item_id"]], "h": m["h"], "w": m["w"]}]})
    mm.set_root(args.subset)

    def score(bundle, rs):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = collect_mm(bundle, rs, 16384, 16384, 64, "cuda:0")
        return {s.idx: (np.array(s.logits), s.target) for s in out}

    norm = msi.BandNorm.load(args.subset / msi.NORM_FILE)
    rgb = common.load_bundle(args.model, device="cuda:0", dtype="bfloat16", multimodal=True)
    a = score(rgb, rows)
    msi.install(norm)
    full_pixels = msi.pixel_inputs

    def rgb_pixels(patches, ip, nm):  # the RGB channels of the 12-band input, for the 3-channel model
        out = full_pixels(patches, ip, nm)
        v = out["pixel_values"].view(-1, 12, ip.temporal_patch_size * ip.patch_size * ip.patch_size)
        return {**out, "pixel_values": v[:, nm.rgb_idx].reshape(v.shape[0], -1)}

    msi.pixel_inputs = rgb_pixels
    d = score(rgb, msi_rows)
    msi.pixel_inputs = full_pixels
    del rgb
    torch.cuda.empty_cache()
    wide = common.load_bundle(args.model, device="cuda:0", dtype="bfloat16", multimodal=True)
    assert msi._patch_embed(wide.model).proj.in_channels == 12
    b = score(wide, msi_rows)

    def probs(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def compare(x, y):
        keys = sorted(set(x) & set(y))
        dp = np.array([np.abs(probs(x[k][0]) - probs(y[k][0])).max() for k in keys])
        return {"scored": len(keys), "max_abs_prob_diff": float(dp.max()), "mean_max_abs_prob_diff": float(dp.mean()),
                "p99_max_abs_prob_diff": float(np.quantile(dp, 0.99)),
                "max_abs_logit_diff": float(max(np.abs(x[k][0] - y[k][0]).max() for k in keys)),
                "argmax_agreement": float(np.mean([x[k][0].argmax() == y[k][0].argmax() for k in keys]))}

    def acc(x):
        return float(np.mean([np.argmax(t) == z.argmax() for z, t in x.values()]))

    res = {"rows": len(rows), "widening_B_vs_D": compare(b, d), "png_rounding_A_vs_D": compare(a, d),
           "A_vs_B": compare(a, b), "accuracy": {"A_rgb_png": acc(a), "D_rgb_float": acc(d), "B_msi": acc(b)}}
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
