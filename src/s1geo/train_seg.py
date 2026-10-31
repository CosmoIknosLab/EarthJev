"""Train the calibrated per-pixel CORINE segmenter of the U-Net + rules reference (s1geo.map_answers).

    python -m s1geo.train_seg --config configs/seg/unet_full_w64d4.yaml --resume auto

Loss: cross-entropy (+ optional Brier) over the CLC classes at `level`, ignoring unlabelled pixels.
After training: temperature fitted on validation pixels; OA, mIoU and pixel ECE (15 bins) on test at the
training level and at level 1 (probabilities summed over child classes, so the two levels are consistent).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from s1geo.clc import LEVEL_CODES
from s1geo.data.reben import RebenSeg, band_stats, class_codes
from s1geo.models.unet import UNet
from s1geo.utils.checkpoint import load_checkpoint, save_checkpoint
from s1geo.utils.runs import make_run_dir, save_run_metadata, set_seed

REPO = Path(__file__).resolve().parents[2]


def seg_loss(logits, y, brier_weight: float):
    ce = F.cross_entropy(logits, y, ignore_index=-1)
    if brier_weight <= 0:
        return ce
    valid = y >= 0
    p = logits.softmax(1).permute(0, 2, 3, 1)[valid]
    onehot = F.one_hot(y[valid], logits.shape[1]).float()
    return ce + brier_weight * ((p - onehot) ** 2).sum(-1).mean()


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    out_l, out_y = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_l.append(model(x.to(device, non_blocking=True)).float().cpu())
        out_y.append(y)
    return torch.cat(out_l), torch.cat(out_y)


def fit_temperature_pixels(logits, y, n: int = 2_000_000, seed: int = 0) -> float:
    lg = logits.permute(0, 2, 3, 1)[y >= 0]
    yy = y[y >= 0]
    if len(yy) > n:
        idx = torch.from_numpy(np.random.default_rng(seed).choice(len(yy), n, replace=False))
        lg, yy = lg[idx], yy[idx]
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def pixel_metrics(probs: torch.Tensor, y: torch.Tensor, k: int, n_bins: int = 15) -> dict:
    """probs [N,K,H,W], y [N,H,W] with -1 ignored."""
    valid = y >= 0
    p = probs.permute(0, 2, 3, 1)[valid]
    yy = y[valid]
    conf, pred = p.max(-1)
    correct = (pred == yy).float()
    cm = torch.bincount(yy * k + pred, minlength=k * k).reshape(k, k).double()
    inter = cm.diag()
    union = cm.sum(0) + cm.sum(1) - inter
    present = cm.sum(1) > 0
    iou = (inter / union.clamp_min(1))[present]
    ece = 0.0
    bins = torch.clamp((conf * n_bins).long(), max=n_bins - 1)
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    nll = -torch.log(p[torch.arange(len(yy)), yy].clamp_min(1e-12)).mean().item()
    return {"pixels": int(valid.sum()), "oa": correct.mean().item(), "miou": iou.mean().item(),
            "iou_per_class": {int(c): float(v) for c, v in zip(present.nonzero().flatten(), iou)},
            "nll": nll, "ece15": ece, "classes_present": int(present.sum())}


def to_level1(probs: torch.Tensor, level: int) -> torch.Tensor:
    if level == 1:
        return probs
    codes = class_codes(level)
    parents = [c // 10 ** (level - 1) for c in codes]
    l1 = LEVEL_CODES[1]
    out = torch.zeros(probs.shape[0], len(l1), *probs.shape[2:])
    for i, par in enumerate(parents):
        out[:, l1.index(par)] += probs[:, i]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["name"] = f"{cfg['name']}_s{args.seed}"
    data = REPO / cfg["data"]
    level, k = cfg["level"], len(class_codes(cfg["level"]))

    run_dir, resumed = make_run_dir(REPO / "runs/seg", cfg["name"], args.resume)
    save_run_metadata(run_dir, cfg, cfg["seed"], REPO)
    set_seed(cfg["seed"])
    stats_path = run_dir / "band_stats.json"
    if stats_path.exists():
        st = json.loads(stats_path.read_text()); mean, std = np.array(st["mean"]), np.array(st["std"])
    else:
        mean, std = band_stats(data / "train.zarr")
        stats_path.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))
    train_rows = None
    if cfg.get("n_folds"):
        # Cross-fitting: train on the tiles outside fold `fold`; scripts predict that fold with this model.
        import pandas as pd

        from s1geo.data.reben import fold_of

        ids = pd.read_parquet(data / "train_index.parquet")["patch_id"]
        train_rows = np.array([i for i, p in enumerate(ids) if fold_of(p, cfg["n_folds"]) != cfg["fold"]])
    ds = {s: RebenSeg(data / f"{s}.zarr", level, mean, std, rows=train_rows if s == "train" else None)
          for s in ("train", "validation", "test")}
    loaders = {s: torch.utils.data.DataLoader(d, batch_size=cfg["batch_size"], shuffle=(s == "train"),
                                              num_workers=4, pin_memory=True, drop_last=(s == "train"))
               for s, d in ds.items()}
    print(f"run_dir={run_dir} resumed={resumed} level={level} K={k} sizes={ {s: len(d) for s, d in ds.items()} }",
          flush=True)

    device = torch.device("cuda")
    model = UNet(12, k, width=cfg["width"], depth=cfg["depth"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total, pct_start=0.05)
    start, best = 0, {"val_miou": -1.0}
    state = load_checkpoint(run_dir) if resumed else None
    if state is not None:
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"]); torch.set_rng_state(state["rng_cpu"])
        start, best = state["epoch"], state["best"]
        print(f"resumed at epoch {start}", flush=True)

    for epoch in range(start, cfg["epochs"]):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if torch.rand(1).item() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1).item() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = seg_loss(model(x).float(), y, cfg["brier_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item() * len(x); n += len(x)
        vl, vy = predict_logits(model, loaders["validation"], device)
        vm = pixel_metrics(vl.softmax(1), vy, k)
        if vm["miou"] > best["val_miou"]:
            best = {"val_miou": vm["miou"], "val_oa": vm["oa"], "epoch": epoch + 1}
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        save_checkpoint({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": opt.state_dict(),
                         "scheduler": sched.state_dict(), "rng_cpu": torch.get_rng_state(), "best": best}, run_dir)
        print(f"epoch {epoch + 1}/{cfg['epochs']} loss {tot / n:.4f} val OA {vm['oa']:.4f} mIoU {vm['miou']:.4f} "
              f"ECE {vm['ece15']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    vl, vy = predict_logits(model, loaders["validation"], device)
    t = fit_temperature_pixels(vl, vy)
    tl, ty = predict_logits(model, loaders["test"], device)
    res = {"config": cfg, "best": best, "temperature": t}
    for name, temp in (("T1", 1.0), ("calibrated", t)):
        probs = (tl / temp).softmax(1)
        res[f"test_level{level}_{name}"] = pixel_metrics(probs, ty, k)
        if level > 1:
            y1 = torch.where(ty >= 0, torch.tensor([LEVEL_CODES[1].index(c // 10 ** (level - 1)) for c in class_codes(level)])[ty.clamp_min(0)], ty)
            res[f"test_level1_{name}"] = pixel_metrics(to_level1(probs, level), y1, len(LEVEL_CODES[1]))
    (run_dir / "results.json").write_text(json.dumps(res, indent=1))
    for key in sorted(k_ for k_ in res if k_.startswith("test_")):
        m = res[key]
        print(f"{key}: OA {m['oa']:.4f} mIoU {m['miou']:.4f} NLL {m['nll']:.4f} ECE {m['ece15']:.4f} "
              f"(classes present {m['classes_present']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
