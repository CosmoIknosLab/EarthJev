"""Variants of EarthJev question files: same questions, targets and ids; only the state or the set of rows changes.

Variants (--variant):
  nometa      drop acquisition_date and country from the state (the RGB rows of the language-model baselines)
  subsample   keep the questions of a random fraction (--fraction) of the patches (training split only; data ablations)

    python scripts/data/make_state_variants.py --src data/earthjev/reben5pct_en --out data/earthjev/reben5pct_rgb --variant nometa
    python scripts/data/make_state_variants.py --src data/earthjev/reben5pct_msi --out data/earthjev/abl_msi_frac25 \
        --variant subsample --fraction 0.25 --files train
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def transform(r: dict, args) -> dict:
    st = dict(r["state"])
    if args.variant == "nometa":
        st.pop("acquisition_date", None)
        st.pop("country", None)
    return {**r, "state": st}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", required=True,
                    choices=["nometa", "subsample"])
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--files", nargs="*", default=None, help="files to convert (default: every *.jsonl in --src)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    files = args.files or sorted(p.stem for p in args.src.glob("*.jsonl"))
    for name in files:
        rows = [json.loads(line) for line in open(args.src / f"{name}.jsonl")]
        if args.variant == "subsample":
            if name == "train":
                pids = sorted({r["meta"]["item_id"] for r in rows})
                keep = set(random.Random(args.seed).sample(pids, round(args.fraction * len(pids))))
                rows = [r for r in rows if r["meta"]["item_id"] in keep]
        else:
            rows = [transform(r, args) for r in rows]
        with open(args.out / f"{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows", flush=True)
    (args.out / "variant.json").write_text(json.dumps(vars(args), default=str, indent=1))


if __name__ == "__main__":
    main()
