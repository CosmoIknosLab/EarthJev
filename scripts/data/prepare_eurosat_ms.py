"""Convert EuroSAT-MS (13-band Sentinel-2 L1C GeoTIFFs) into compressed Zarr in the reBEN band layout.

The zip is read in place (no extraction). B10 (cirrus) is dropped and the remaining 12 bands are reordered
to the reBEN order, so both datasets share one 12-band input pipeline. Splits are the EuroSAT-Embed spatial
(longitude) splits.

Band order of the source files. The GeoTIFFs carry no band descriptions; the order was inferred from the
data (600 random images, 2026-10-30) and is re-checked on every run (check_band_order):
  index 9 has mean ~12 (cirrus, B10); indices 0, 8, 9 are the smoothest (60 m: B01, B09, B10); indices 1, 2, 3, 7
  are the sharpest (10 m: B02, B03, B04, B08); index 12 correlates 0.98 with index 7 (B8A vs B08).
  -> B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A

Output (<out>/):
  {train,validation,test}.zarr  group with array
      s2  uint16 (N, 12, 64, 64)  bands B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12 (as reBEN)
  {split}_index.parquet         row -> item_id, eurosat_class (row order = Zarr order)
  summary.json

    python scripts/data/prepare_eurosat_ms.py --zip data/eurosat/ms/raw/EuroSAT_MS.zip \
        --splits data/eurosat_embed/raw --out data/eurosat/ms/zarr
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import zarr

SOURCE_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
TAKE = [SOURCE_BANDS.index(b) for b in BANDS]
SIZE = 64
SHARD = 512  # patches per Zarr shard (one file per shard)
SPLITS = {"train": "eurosat-spatial-train.txt", "validation": "eurosat-spatial-val.txt", "test": "eurosat-spatial-test.txt"}
CHECK_IMAGES = 600


def read_tif(data: bytes) -> np.ndarray:
    with rasterio.MemoryFile(data) as mem, mem.open() as ds:
        a = ds.read()
    if a.shape != (len(SOURCE_BANDS), SIZE, SIZE) or a.dtype != np.uint16:
        raise ValueError(f"unexpected image {a.shape} {a.dtype}")
    return a


def check_band_order(images: list[np.ndarray]) -> dict:
    """Fail unless the statistics that identified the source band order still hold."""
    x = np.concatenate([a.reshape(len(SOURCE_BANDS), -1)[:, ::7] for a in images], 1).astype(np.float64)
    corr = np.corrcoef(x)
    rough = np.mean([np.abs(np.diff(a.astype(np.float32), axis=2)).mean((1, 2)) / (a.std((1, 2)) + 1e-6)
                     for a in images], 0)
    i = SOURCE_BANDS.index
    stats = {"b10_mean": float(x[i("B10")].mean()), "corr_b8a_b08": float(corr[i("B8A"), i("B08")]),
             "roughness": dict(zip(SOURCE_BANDS, np.round(rough, 3).tolist()))}
    smooth = {i("B01"), i("B09"), i("B10")}
    sharp = {i("B02"), i("B03"), i("B04"), i("B08")}
    ok = (stats["b10_mean"] < 100 and stats["corr_b8a_b08"] > 0.9
          and max(rough[k] for k in smooth) < min(rough[k] for k in range(len(SOURCE_BANDS)) if k not in smooth)
          and min(rough[k] for k in sharp) > max(rough[k] for k in range(len(SOURCE_BANDS)) if k not in sharp))
    if not ok:
        raise SystemExit(f"band-order check failed: {json.dumps(stats)}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True, help="directory with the EuroSAT-Embed split files")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    zf = zipfile.ZipFile(args.zip)
    members = {Path(n).stem: n for n in zf.namelist() if n.endswith(".tif")}
    split_names = {s: [Path(l.strip()).stem for l in open(args.splits / f) if l.strip()] for s, f in SPLITS.items()}
    all_names = [n for names in split_names.values() for n in names]
    if len(set(all_names)) != len(all_names):
        raise SystemExit("an image appears in more than one split")
    missing = set(all_names) - set(members)
    if missing:
        raise SystemExit(f"{len(missing)} split images are not in the zip, e.g. {sorted(missing)[:3]}")

    rng = np.random.default_rng(0)
    probe = [read_tif(zf.read(members[n])) for n in rng.choice(sorted(members), CHECK_IMAGES, replace=False)]
    band_check = check_band_order(probe)
    print(f"band-order check passed: {json.dumps(band_check)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=False)
    compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    for split, names in split_names.items():
        root = zarr.open_group(str(args.out / f"{split}.zarr"), mode="w-")
        s2 = root.create_array("s2", shape=(len(names), len(BANDS), SIZE, SIZE), dtype="uint16",
                               chunks=(1, len(BANDS), SIZE, SIZE), shards=(SHARD, len(BANDS), SIZE, SIZE),
                               compressors=compressor)
        root.attrs["bands"] = BANDS
        root.attrs["product"] = "Sentinel-2 L1C (EuroSAT-MS), B10 dropped"
        for start in range(0, len(names), SHARD):
            chunk = names[start:start + SHARD]
            s2[start:start + len(chunk)] = np.stack([read_tif(zf.read(members[n]))[TAKE] for n in chunk])
        pd.DataFrame({"item_id": names, "eurosat_class": [n.rsplit("_", 1)[0] for n in names]}).to_parquet(
            args.out / f"{split}_index.parquet")
        print(f"{split}: {len(names)} images, {time.time() - t0:.0f}s", flush=True)

    summary = {"source": str(args.zip), "splits": "EuroSAT-Embed spatial", "per_split": {s: len(n) for s, n in split_names.items()},
               "source_bands": SOURCE_BANDS, "bands": BANDS, "band_check": band_check, "seconds": round(time.time() - t0)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
