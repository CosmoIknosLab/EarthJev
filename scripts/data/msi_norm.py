"""Band statistics for the 12-band input (s1geo.msi.make_norm): RGB stretch of the PNG pipeline + per-band mean/std.

    python scripts/data/msi_norm.py --zarr data/reben/subset_5pct/train.zarr \
        --stretch data/reben/subset_5pct/rgb/stretch.json --out data/reben/subset_5pct/msi_norm.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from s1geo.msi import make_norm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zarr", type=Path, required=True, help="training split Zarr")
    ap.add_argument("--stretch", type=Path, required=True, help="stretch.json of the RGB PNGs (reben_rgb.py)")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    norm = make_norm(args.zarr, args.stretch, sample=args.sample)
    args.out.write_text(norm.to_json())
    print(norm.to_json())


if __name__ == "__main__":
    main()
