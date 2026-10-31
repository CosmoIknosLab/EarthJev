"""Per-band weight norm of the 12-channel input layer of an EarthJev MSI checkpoint (s1geo.msi).

The 9 non-RGB bands start at zero; their norm after training, next to that of B04/B03/B02, shows how much the model
learned to use them. Compare with the input statistics: inputs are z-scored (non-RGB) or in [-1, 1] (RGB).

    python scripts/checks/msi_band_weights.py runs/onejev/<run>/final [more checkpoints]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def band_norms(ckpt: Path) -> dict[str, float]:
    for f in sorted(ckpt.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            key = next((k for k in st.keys() if k.endswith("visual.patch_embed.proj.weight")), None)
            if key is None:
                continue
            w = st.get_tensor(key).float()  # (out, 12, 2, 16, 16)
            if w.shape[1] != len(BANDS):
                raise SystemExit(f"{ckpt}: input layer has {w.shape[1]} channels")
            return {b: float(w[:, i].norm()) for i, b in enumerate(BANDS)}
    raise SystemExit(f"{ckpt}: no visual.patch_embed.proj.weight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpts", type=Path, nargs="+")
    args = ap.parse_args()
    for c in args.ckpts:
        n = band_norms(c)
        rgb = sum(n[b] for b in ("B04", "B03", "B02")) / 3
        print(json.dumps({"ckpt": str(c), "norm": {b: round(v, 5) for b, v in n.items()},
                          "relative_to_mean_rgb": {b: round(v / rgb, 4) for b, v in n.items()}}))


if __name__ == "__main__":
    main()
