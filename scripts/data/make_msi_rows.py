"""Turn RGB-PNG question rows into 12-band rows (s1geo.msi): media -> Zarr patch, state -> image + sensor description.

Questions, targets and ids are kept; only `state` and `media` change. The state holds the image and a fixed
description of the sensor; with --meta, the acquisition date and country of the input rows are kept too (ablation).

    python scripts/data/make_msi_rows.py --preset reben --rows data/earthjev/reben5pct_en \
        --zarr data/reben/subset_5pct --out data/earthjev/reben5pct_msi
    python scripts/data/make_msi_rows.py --preset eurosat --rows data/earthjev/eurosat_spatial \
        --zarr data/eurosat/ms/zarr --out data/earthjev/eurosat_ms_spatial

Media paths are relative to the Zarr directory (`media_root` in the training config). Other row files can be
converted with --files NAME=SPLIT (e.g. the transfer tests); --media-prefix is prepended to the Zarr path when the
media root is not the Zarr directory:
    python scripts/data/make_msi_rows.py --preset reben --rows data/earthjev/rsvqaxben_test --zarr data/reben/subset_5pct \
        --files presence=test --out data/earthjev/rsvqaxben_test_msi
    python scripts/data/make_msi_rows.py --preset reben --rows data/earthjev/bentxt --zarr data/reben/bentxt_bench \
        --files bench=test --media-prefix ../bentxt_bench/ --out data/earthjev/bentxt_msi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BANDS_TEXT = "12 bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 (20 m and 60 m bands resampled to 10 m)"
PRESETS = {
    "reben": {"id_column": "patch_id", "size": 120, "product": "Sentinel-2 L2A (surface reflectance)",
              "patch": "120 x 120 pixels (1.2 km x 1.2 km), north up",
              "splits": {"train": "train", "validation": "validation", "validation_1k": "validation",
                         "test": "test", "test_unseen": "test"}},
    "eurosat": {"id_column": "item_id", "size": 64, "product": "Sentinel-2 L1C (top-of-atmosphere reflectance)",
                "patch": "64 x 64 pixels (640 m x 640 m)",
                "splits": {"train": "train", "val": "validation", "val_1k": "validation", "test": "test"}},
}
META_KEYS = ("acquisition_date", "country")


def state_for(preset: dict, old: dict, meta: bool) -> dict:
    state = {"sensor": f"{preset['product']}, multispectral", "bands": BANDS_TEXT,
             "ground_sample_distance": "10 m", "patch": preset["patch"]}
    if meta:
        state.update({k: old[k] for k in META_KEYS if k in old})
    state["image"] = "<image:1>"
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), required=True)
    ap.add_argument("--rows", type=Path, required=True, help="directory of RGB rows ({split}.jsonl)")
    ap.add_argument("--zarr", type=Path, required=True, help="directory with {split}.zarr and {split}_index.parquet")
    ap.add_argument("--meta", action="store_true", help="keep acquisition date and country in the state (ablation)")
    ap.add_argument("--files", nargs="+", default=None, help="NAME=SPLIT pairs instead of the preset's files")
    ap.add_argument("--media-prefix", default="", help="prepended to '<split>.zarr' in the media path")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    preset = PRESETS[args.preset]

    args.out.mkdir(parents=True, exist_ok=False)
    summary = {"preset": args.preset, "rows": str(args.rows), "zarr": str(args.zarr), "meta": args.meta, "files": {}}
    files = dict(f.split("=", 1) for f in args.files) if args.files else preset["splits"]
    summary["media_prefix"] = args.media_prefix
    for name, split in files.items():
        src = args.rows / f"{name}.jsonl"
        if not src.exists():
            continue
        ids = pd.read_parquet(args.zarr / f"{split}_index.parquet")[preset["id_column"]].tolist()
        row_of = {pid: i for i, pid in enumerate(ids)}
        n = 0
        with open(src) as fin, open(args.out / f"{name}.jsonl", "w") as fout:
            for line in fin:
                r = json.loads(line)
                item = r["meta"]["item_id"]
                if item not in row_of:
                    raise SystemExit(f"{src}: {item} not in {split}_index.parquet")
                if len(r.get("media") or []) != 1:
                    raise SystemExit(f"{src}: row {r['id']} has {len(r.get('media') or [])} media items")
                r["state"] = state_for(preset, r["state"], args.meta)
                r["media"] = [{"type": "image", "path": f"{args.media_prefix}{split}.zarr", "row": row_of[item],
                               "h": preset["size"], "w": preset["size"]}]
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        summary["files"][name] = n
        print(f"{name}: {n} rows", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
