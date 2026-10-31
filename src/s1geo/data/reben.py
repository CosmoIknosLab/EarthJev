"""BigEarthNet v2.0 (reBEN) Zarr subsets written by scripts/data/prepare_reben.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from s1geo.clc import LEVEL_CODES, to_level


def class_codes(level: int) -> list[int]:
    return LEVEL_CODES[level]


def encode_mask(clc: np.ndarray, level: int) -> np.ndarray:
    """CLC level-3 codes -> class indices 0..K-1 at `level`; invalid / unlabelled pixels -> -1."""
    lv = to_level(clc, level)
    lut = np.full(int(lv.max(initial=0)) + 1, -1, dtype=np.int64)
    for i, c in enumerate(class_codes(level)):
        if c < len(lut):
            lut[c] = i
    return lut[lv]


def band_stats(path: str | Path, n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and std over a random sample of patches (use on the train split only)."""
    s2 = zarr.open_group(str(path), mode="r")["s2"]
    rows = np.sort(np.random.default_rng(seed).choice(s2.shape[0], size=min(n, s2.shape[0]), replace=False))
    x = np.stack([s2[int(r)] for r in rows]).astype(np.float64)
    return x.mean(axis=(0, 2, 3)), x.std(axis=(0, 2, 3))


def tile_of(patch_id: str) -> str:
    """Sentinel-2 tile of a reBEN patch id, e.g. 'S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57' -> 'T33UUP'."""
    return patch_id.split("_")[5]


def fold_of(patch_id: str, n_folds: int) -> int:
    """Deterministic fold by Sentinel-2 tile (neighbouring patches share a fold)."""
    import zlib

    return zlib.crc32(tile_of(patch_id).encode()) % n_folds


class RebenSeg(torch.utils.data.Dataset):
    """Returns (image float32 [12,120,120] normalised, target int64 [120,120] with -1 = ignore).

    `rows` optionally restricts the dataset to those Zarr row indices (e.g. one cross-fitting fold)."""

    def __init__(self, path: str | Path, level: int, mean: np.ndarray, std: np.ndarray, in_memory: bool = True,
                 rows: np.ndarray | None = None):
        g = zarr.open_group(str(path), mode="r")
        self.s2 = g["s2"][:] if in_memory else g["s2"]
        self.clc = g["clc"][:] if in_memory else g["clc"]
        if rows is not None:
            if not in_memory:
                raise ValueError("rows requires in_memory=True")
            self.s2, self.clc = self.s2[rows], self.clc[rows]
        self.level = level
        self.mean = mean.astype(np.float32)[:, None, None]
        self.std = std.astype(np.float32)[:, None, None]

    def __len__(self) -> int:
        return self.s2.shape[0]

    def __getitem__(self, i: int):
        x = (np.asarray(self.s2[i], dtype=np.float32) - self.mean) / self.std
        y = encode_mask(np.asarray(self.clc[i]), self.level)
        return torch.from_numpy(x), torch.from_numpy(y)
