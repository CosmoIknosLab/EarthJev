"""Convert EuroSAT RGB into OneJev's unified JSONL rows (state, question, target, media).

    python scripts/data/make_eurosat_onejev.py \
        --images data/eurosat/rgb/EuroSAT_RGB --splits data/eurosat_embed/raw --out data/earthjev

Writes <out>/eurosat_<split_type>/{train,val,test,val_1k}.jsonl for split_type in {spatial, random}, using the
split files released with EuroSAT-Embed (spatial = longitude split, random = standard split). Media paths
are relative to the EuroSAT data root (`--media-root` in the training config: data/eurosat).

Each image gets five questions on the same state, so OneJev can share the state prefix:
- choice `land_cover`: the 10 EuroSAT classes;
- noul `water`, `natural_vegetation`, `agriculture`, `built_up`: class groups. They are deterministic
  functions of the EuroSAT label (no new annotation); they exercise the typed multi-question interface.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CLASSES = {
    "AnnualCrop": ("annual_crop", "Annual crop fields (arable land)."),
    "Forest": ("forest", "Forest."),
    "HerbaceousVegetation": ("herbaceous_vegetation", "Herbaceous vegetation (grassland, shrubland)."),
    "Highway": ("highway", "A highway or major road."),
    "Industrial": ("industrial", "Industrial or commercial buildings."),
    "Pasture": ("pasture", "Pasture (meadows used for grazing)."),
    "PermanentCrop": ("permanent_crop", "Permanent crops (orchards, vineyards, olive groves)."),
    "Residential": ("residential", "Residential buildings."),
    "River": ("river", "A river."),
    "SeaLake": ("sea_lake", "The sea or a lake."),
}

GROUPS = {
    "water": ("The main land cover in the image is water (a river, a lake or the sea).",
              {"River", "SeaLake"}),
    "natural_vegetation": ("The main land cover in the image is natural vegetation (forest, grassland or shrubland).",
                           {"Forest", "HerbaceousVegetation"}),
    "agriculture": ("The main land cover in the image is agricultural land (crop fields, orchards, vineyards or pasture).",
                    {"AnnualCrop", "PermanentCrop", "Pasture"}),
    "built_up": ("The main land cover in the image is built-up or artificial (buildings, industrial areas or a highway).",
                 {"Residential", "Industrial", "Highway"}),
}

STATE = {
    "sensor": "Sentinel-2 MSI, true-colour composite (bands B4, B3, B2)",
    "ground_sample_distance": "10 m",
    "patch": "64 x 64 pixels (640 m x 640 m)",
    "region": "Europe",
    "image": "<image:1>",
}

VAL_SUBSET = 1000  # images in val_1k.jsonl

SPLIT_FILES = {
    "spatial": {"train": "eurosat-spatial-train.txt", "val": "eurosat-spatial-val.txt", "test": "eurosat-spatial-test.txt"},
    "random": {"train": "eurosat-train.txt", "val": "eurosat-val.txt", "test": "eurosat-test.txt"},
}


def land_cover_question() -> dict:
    return {
        "type": "choice",
        "instructions": "Which land-cover class best describes this Sentinel-2 image patch?",
        "criteria": {key: desc for key, desc in CLASSES.values()},
    }


def rows_for_image(cls: str, stem: str, rel_path: str, split: str) -> list[dict]:
    common = {
        "source": "eurosat",
        "split": split,
        "license": "mit",
        "state": STATE,
        "media": [{"type": "image", "path": rel_path, "w": 64, "h": 64}],
        "meta": {"item_id": stem, "eurosat_class": cls, "judge": "dataset_label"},
    }
    rows = [{**common, "id": f"eurosat/{stem}/land_cover", "task": "land_cover",
             "question": land_cover_question(), "target": {CLASSES[cls][0]: 1.0}}]
    for task, (statement, members) in GROUPS.items():
        yes = cls in members
        rows.append({**common, "id": f"eurosat/{stem}/{task}", "task": task,
                     "question": {"type": "noul", "instructions": statement},
                     "target": {"yes": 1.0 if yes else 0.0, "no": 0.0 if yes else 1.0}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True, help="EuroSAT_RGB directory with one folder per class")
    ap.add_argument("--splits", type=Path, required=True, help="directory with the EuroSAT-Embed split files")
    ap.add_argument("--media-root", type=Path, default=None, help="root media paths are relative to (default: parent of --images' parent)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    media_root = args.media_root or args.images.parent.parent

    for split_type, files in SPLIT_FILES.items():
        seen: set[str] = set()
        out_dir = args.out / f"eurosat_{split_type}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, fname in files.items():
            names = [l.strip() for l in open(args.splits / fname) if l.strip()]
            n_rows = 0
            with open(out_dir / f"{split}.jsonl", "w") as fh:
                for name in names:
                    stem = Path(name).stem
                    cls = stem.rsplit("_", 1)[0]
                    if cls not in CLASSES:
                        raise ValueError(f"{fname}: unknown class in {name!r}")
                    if stem in seen:
                        raise ValueError(f"{stem} appears in more than one split of {split_type}")
                    seen.add(stem)
                    img = args.images / cls / f"{stem}.jpg"
                    if not img.is_file():
                        raise FileNotFoundError(img)
                    for row in rows_for_image(cls, stem, str(img.relative_to(media_root)), split):
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_rows += 1
            print(f"{split_type}/{split}: {len(names)} images, {n_rows} rows")
            if split == "val":
                # Fixed random subset of val images for cheap in-training evaluation.
                keep = set(random.Random(0).sample(sorted(Path(n).stem for n in names), VAL_SUBSET))
                with open(out_dir / f"val_{VAL_SUBSET // 1000}k.jsonl", "w") as small, open(out_dir / "val.jsonl") as full:
                    for line in full:
                        if json.loads(line)["meta"]["item_id"] in keep:
                            small.write(line)
        print(f"{split_type}: {len(seen)} distinct images")


if __name__ == "__main__":
    main()
