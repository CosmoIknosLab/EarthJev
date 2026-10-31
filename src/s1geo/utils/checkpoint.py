"""Atomic checkpoint save/load for resumable jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

LAST_CKPT = "checkpoints/last.pt"


def save_checkpoint(state: dict[str, Any], run_dir: str | Path) -> Path:
    """Save ``state`` to ``run_dir/checkpoints/last.pt`` atomically.

    The file is written to a temporary path and renamed, so a job killed
    mid-write never leaves a truncated ``last.pt``.
    """
    path = Path(run_dir) / LAST_CKPT
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(run_dir: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
    path = Path(run_dir) / LAST_CKPT
    if not path.is_file():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)
