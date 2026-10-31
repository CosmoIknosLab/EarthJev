"""Dump raw answer-slot logits of a OneJev-style model on unified JSONL rows.

Same loading, rendering (frozen production prompt, no permutation) and readout as
third_party/OneJev/train/eval_mm.py, but writes the slot logits instead of rounded probabilities,
so temperature fitting and NLL are exact. Metrics are computed offline by s1geo.metrics.

Run from third_party/OneJev with it on PYTHONPATH (it imports OneJev's `train` package):
    python -m s1geo.onejev_eval --model Qwen/Qwen3.5-0.8B --media-root data/eurosat \
        --eval test=data/earthjev/eurosat_spatial/test.jsonl --out-dir runs/eval/<name>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--media-root", required=True)
    ap.add_argument("--eval", dest="evals", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="use at most this many rows per file (0: all)")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--max-rows", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--msi-norm", default=None,
                    help="12-band rows (s1geo.msi): band statistics JSON (default: <model>/msi_norm.json if present)")
    ap.add_argument("--msi-drop-extra-bands", action="store_true",
                    help="ablation: the 9 non-RGB bands set to their training mean (no information)")
    args = ap.parse_args(argv)

    import torch
    from train import common, mm
    from train.sft import collect_mm, load_eval_rows

    mm.set_root(args.media_root)
    norm_path = args.msi_norm or (Path(args.model) / "msi_norm.json")
    if args.msi_norm or Path(norm_path).is_file():
        from s1geo import msi

        norm = msi.BandNorm.load(norm_path)
        norm.drop_extra = args.msi_drop_extra_bands
        msi.install(norm, freeze=False)
        print(f"12-band input, band statistics {norm_path}", flush=True)
    bundle = common.load_bundle(args.model, device=args.device, dtype="bfloat16", multimodal=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "msi_norm": str(norm_path) if args.msi_norm or Path(norm_path).is_file() else None,
               "msi_drop_extra_bands": args.msi_drop_extra_bands, "evals": {}}
    for spec in args.evals:
        name, _, path = spec.partition("=")
        rows = load_eval_rows(path, args.limit)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scored = collect_mm(bundle, rows, args.max_len, args.batch_tokens, args.max_rows, args.device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        with open(out_dir / f"{name}.logits.jsonl", "w") as fh:
            for sc in scored:
                r = rows[sc.idx]
                fh.write(json.dumps({"id": r["id"], "task": r.get("task"), "kind": sc.kind, "labels": sc.labels,
                                     "logits": sc.logits, "target": sc.target,
                                     "item_id": (r.get("meta") or {}).get("item_id")}) + "\n")
        summary["evals"][name] = {"path": path, "rows": len(rows), "scored": len(scored), "seconds": round(elapsed, 2),
                                  "gpu": torch.cuda.get_device_name(0)}
        print(f"{name}: {len(scored)}/{len(rows)} rows scored in {elapsed:.1f}s", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
