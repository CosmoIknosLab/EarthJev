"""Run directories: ``runs/<date>_<name>/`` with saved config, git commit and seed."""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .checkpoint import LAST_CKPT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_info(repo_dir: str | Path) -> dict[str, Any]:
    """Return current commit and dirty flag, or ``None`` values outside a git repo."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def find_resumable_run(runs_root: str | Path, name: str) -> Path | None:
    """Most recent ``runs_root/*_<name>`` directory that contains a checkpoint."""
    candidates = sorted(
        p for p in Path(runs_root).glob(f"*_{name}") if (p / LAST_CKPT).is_file()
    )
    return candidates[-1] if candidates else None


def make_run_dir(runs_root: str | Path, name: str, resume: str = "auto") -> tuple[Path, bool]:
    """Create or reuse a run directory.

    ``resume``: ``"auto"`` reuses the latest run with a checkpoint, ``"never"``
    always creates a new one, any other value is taken as an explicit run path.
    Returns ``(run_dir, resumed)``.
    """
    if resume not in ("auto", "never"):
        run_dir = Path(resume)
        if not (run_dir / LAST_CKPT).is_file():
            raise FileNotFoundError(f"no checkpoint to resume in {run_dir}")
        return run_dir, True
    if resume == "auto":
        found = find_resumable_run(runs_root, name)
        if found is not None:
            return found, True

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(runs_root) / f"{stamp}_{name}"
    i = 0
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir, False
        except FileExistsError:
            i += 1
            run_dir = Path(runs_root) / f"{stamp}-{i}_{name}"


def save_run_metadata(run_dir: Path, config: dict[str, Any], seed: int, repo_dir: str | Path) -> None:
    """Write ``config.yaml`` and append an entry to ``meta.jsonl`` (one line per (re)start)."""
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    meta = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "git": git_info(repo_dir),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
    }
    with open(run_dir / "meta.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")
