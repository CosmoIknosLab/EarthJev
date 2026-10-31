"""Write true-colour PNGs (B04, B03, B02) for every patch of a reBEN Zarr subset, for the VLM.

With --bands / --name, writes another 3-band composite instead (e.g. --bands B12 B08 B04 --name fc_swir for the
false-colour ablation), in <subset>/<name>/.

Fixed linear stretch per band: [p2, p98] of the train split (computed on a sample, stored in
rgb/stretch.json), clipped to 0-255. Output: <subset>/rgb/<split>/<patch_id>.png (120x120, north up).

    python scripts/data/reben_rgb.py --subset data/reben/subset_5pct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image

RGB = [3, 2, 1]  # indices of B04, B03, B02 in the stored band order
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
SPLITS = ("train", "validation", "test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--stretch", type=Path, default=None, help="reuse this stretch.json (e.g. of the training subset)")
    ap.add_argument("--bands", nargs=3, default=["B04", "B03", "B02"])
    ap.add_argument("--name", default="rgb")
    args = ap.parse_args()
    RGB[:] = [BANDS.index(b) for b in args.bands]
    out = args.subset / args.name
    out.mkdir(exist_ok=False)

    if args.stretch:
        st = json.loads(args.stretch.read_text())
        lo, hi = np.array(st["p2"]), np.array(st["p98"])
    else:
        train = zarr.open_group(str(args.subset / "train.zarr"), mode="r")["s2"]
        rows = np.sort(np.random.default_rng(0).choice(train.shape[0], min(args.sample, train.shape[0]), replace=False))
        x = np.stack([train[int(r)][RGB] for r in rows]).astype(np.float64)
        lo = np.percentile(x, 2, axis=(0, 2, 3))
        hi = np.percentile(x, 98, axis=(0, 2, 3))
    (out / "stretch.json").write_text(json.dumps({"bands": args.bands, "p2": lo.tolist(), "p98": hi.tolist()}))
    print("stretch p2", lo.round(1).tolist(), "p98", hi.round(1).tolist(), flush=True)

    for split in [s for s in SPLITS if (args.subset / f"{s}.zarr").exists()]:
        s2 = zarr.open_group(str(args.subset / f"{split}.zarr"), mode="r")["s2"]
        ids = pd.read_parquet(args.subset / f"{split}_index.parquet")["patch_id"].tolist()
        (out / split).mkdir()
        for start in range(0, len(ids), 256):
            block = s2[start:start + 256][:, RGB].astype(np.float64)
            img = np.clip((block - lo[None, :, None, None]) / (hi - lo)[None, :, None, None], 0, 1)
            img = (img * 255).round().astype(np.uint8).transpose(0, 2, 3, 1)
            for pid, im in zip(ids[start:start + 256], img):
                Image.fromarray(im).save(out / split / f"{pid}.png", optimize=True)
        print(f"{split}: {len(ids)} PNGs", flush=True)


if __name__ == "__main__":
    main()
