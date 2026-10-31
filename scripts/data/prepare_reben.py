"""Stream BigEarthNet v2.0 (reBEN) Sentinel-2 patches + CORINE reference maps into compressed Zarr.

The 63 GB S2 archive is read as a stream (URL or local file) and never stored: only the selected patches
are decoded and written, so a subset costs only its own size on disk. The MD5 of the compressed stream is
checked against Zenodo when the stream ends.

Selection: rows of metadata.parquet (patches without seasonal snow, cloud or shadow), sampled per
(split, country) with --fraction and --seed; --fraction 1 converts everything.

Output (<out>/):
  {train,validation,test}.zarr  group with arrays
      s2   uint16 (N, 12, 120, 120)  bands B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12, 20 m and 60 m bands
                                     upsampled to 10 m by pixel repetition (values unchanged)
      clc  uint16 (N, 120, 120)      CORINE level-3 codes from the reBEN reference maps
  {split}_index.parquet              row -> patch_id, country, labels (row order = Zarr order)
  summary.json

    python scripts/data/prepare_reben.py --metadata data/reben/raw/metadata.parquet \
        --maps data/reben/raw/Reference_Maps.tar.zst --fraction 0.05 --out data/reben/subset_5pct
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import zarr
import zstandard

S2_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"
S2_MD5_API = "https://zenodo.org/api/records/10891137"
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
SIZE = 120
SHARD = 256  # patches per Zarr shard (one file per shard)
SPLITS = ("train", "validation", "test")


class HashingReader(io.RawIOBase):
    """File-like wrapper that MD5-hashes every byte read through it."""

    def __init__(self, raw):
        self.raw, self.md5, self.n = raw, hashlib.md5(), 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self.raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.md5.update(data)
        self.n += n
        return n


def open_source(src: str):
    return urllib.request.urlopen(src) if src.startswith("http") else open(src, "rb")


def read_band(data: bytes) -> np.ndarray:
    with rasterio.MemoryFile(data) as mem, mem.open() as ds:
        a = ds.read(1)
    f = SIZE // a.shape[0]
    if f * a.shape[0] != SIZE or a.shape[0] != a.shape[1]:
        raise ValueError(f"unexpected band shape {a.shape}")
    return np.repeat(np.repeat(a, f, axis=0), f, axis=1) if f > 1 else a


def select(meta: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1:
        return meta
    parts = [g.sample(frac=fraction, random_state=seed) for _, g in meta.groupby(["split", "country"], sort=True)]
    return pd.concat(parts)


class SplitWriter:
    def __init__(self, path: Path, n: int):
        compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
        self.root = zarr.open_group(str(path), mode="w-")
        self.s2 = self.root.create_array("s2", shape=(n, len(BANDS), SIZE, SIZE), dtype="uint16",
                                         chunks=(1, len(BANDS), SIZE, SIZE), shards=(SHARD, len(BANDS), SIZE, SIZE),
                                         compressors=compressor)
        self.clc = self.root.create_array("clc", shape=(n, SIZE, SIZE), dtype="uint16", chunks=(1, SIZE, SIZE),
                                          shards=(SHARD, SIZE, SIZE), compressors=compressor)
        self.root.attrs["bands"] = BANDS
        self.n, self.rows, self.buf_s2, self.buf_clc, self.ids = n, 0, [], [], []

    def add(self, pid: str, s2: np.ndarray, clc: np.ndarray) -> None:
        self.buf_s2.append(s2); self.buf_clc.append(clc); self.ids.append(pid)
        if len(self.buf_s2) == SHARD:
            self.flush()

    def flush(self) -> None:
        if not self.buf_s2:
            return
        k = len(self.buf_s2)
        self.s2[self.rows:self.rows + k] = np.stack(self.buf_s2)
        self.clc[self.rows:self.rows + k] = np.stack(self.buf_clc)
        self.rows += k
        self.buf_s2, self.buf_clc = [], []


def load_maps(src: str, wanted: set[str]) -> dict[str, np.ndarray]:
    maps = {}
    reader = zstandard.ZstdDecompressor().stream_reader(open_source(src))
    with tarfile.open(fileobj=reader, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith("_reference_map.tif"):
                continue
            pid = Path(m.name).name[: -len("_reference_map.tif")]
            if pid in wanted:
                with rasterio.MemoryFile(tf.extractfile(m).read()) as mem, mem.open() as ds:
                    a = ds.read(1)
                if a.shape != (SIZE, SIZE):
                    raise ValueError(f"{pid}: reference map shape {a.shape}")
                maps[pid] = a
    return maps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--maps", required=True, help="Reference_Maps.tar.zst (path or URL)")
    ap.add_argument("--s2", default=S2_URL, help="BigEarthNet-S2.tar.zst (path or URL)")
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--ids", type=Path, default=None, help="text file with patch ids to select instead of --fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    meta = pd.read_parquet(args.metadata)
    if (args.fraction is None) == (args.ids is None):
        raise SystemExit("give exactly one of --fraction and --ids")
    if args.ids:
        ids = {l.strip() for l in open(args.ids) if l.strip()}
        sel = meta[meta.patch_id.isin(ids)]
        if len(sel) != len(ids):
            raise SystemExit(f"{len(ids) - len(sel)} requested ids are not in {args.metadata}")
    else:
        sel = select(meta, args.fraction, args.seed)
    wanted = set(sel["patch_id"])
    split_of = dict(zip(sel["patch_id"], sel["split"]))
    counts = sel["split"].value_counts().to_dict()
    print(f"selected {len(sel)} of {len(meta)} patches: {counts}", flush=True)

    maps = load_maps(args.maps, wanted)
    print(f"reference maps: {len(maps)}/{len(wanted)} in {time.time() - t0:.0f}s", flush=True)
    missing_maps = wanted - set(maps)
    if missing_maps:
        raise SystemExit(f"{len(missing_maps)} selected patches have no reference map, e.g. {sorted(missing_maps)[:3]}")

    args.out.mkdir(parents=True, exist_ok=False)
    writers = {s: SplitWriter(args.out / f"{s}.zarr", counts.get(s, 0)) for s in SPLITS if counts.get(s, 0)}
    raw = HashingReader(open_source(args.s2))
    reader = zstandard.ZstdDecompressor().stream_reader(io.BufferedReader(raw, buffer_size=1 << 22))
    done, current, bands = 0, None, {}

    def finish(pid):
        nonlocal done
        if pid is None or pid not in wanted:
            return
        if set(bands) != set(BANDS):
            raise ValueError(f"{pid}: bands {sorted(bands)}")
        writers[split_of[pid]].add(pid, np.stack([bands[b] for b in BANDS]), maps[pid])
        done += 1
        if done % 1000 == 0:
            print(f"  {done}/{len(wanted)} patches, {raw.n / 1e9:.1f} GB read, {time.time() - t0:.0f}s", flush=True)

    with tarfile.open(fileobj=reader, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".tif"):
                continue
            pid = Path(m.name).parent.name
            if pid != current:
                finish(current)
                current, bands = pid, {}
            if pid in wanted:
                band = Path(m.name).stem.rsplit("_", 1)[1]
                bands[band] = read_band(tf.extractfile(m).read())
        finish(current)
    while raw.raw.read(1 << 22):  # hash any trailing bytes after the tar end marker
        pass
    md5 = raw.md5.hexdigest()
    for w in writers.values():
        w.flush()
        if w.rows != w.n:
            raise SystemExit(f"wrote {w.rows} rows, expected {w.n}")
    expected_md5 = None
    if args.s2.startswith("http"):
        rec = json.load(urllib.request.urlopen(S2_MD5_API))
        expected_md5 = next(f["checksum"] for f in rec["files"] if f["key"] == "BigEarthNet-S2.tar.zst").split(":")[1]
    by_id = sel.set_index("patch_id")
    for s, w in writers.items():
        idx = by_id.loc[w.ids, ["country", "labels"]].reset_index()
        idx["labels"] = idx["labels"].map(list)
        idx.to_parquet(args.out / f"{s}_index.parquet")
    summary = {"fraction": args.fraction, "ids": str(args.ids) if args.ids else None, "seed": args.seed, "selected": len(sel), "written": done,
               "per_split": {s: w.rows for s, w in writers.items()}, "s2_bytes_read": raw.n, "s2_md5": md5,
               "s2_md5_expected": expected_md5, "md5_ok": (md5 == expected_md5) if expected_md5 else None,
               "bands": BANDS, "seconds": round(time.time() - t0)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if expected_md5 and md5 != expected_md5:
        raise SystemExit("MD5 mismatch on the S2 stream")


if __name__ == "__main__":
    main()
