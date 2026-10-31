"""Traditional VQA baseline on Sentinel-2 bands, in the style of RSVQA (Lobry et al., TGRS 2020).

Image encoder (--encoder, the three families of RSJEV): ResNet-50 (CNN, torchvision IMAGENET1K_V2), Swin-T (Transformer,
torchvision IMAGENET1K_V1) or MambaVision-T (Mamba, nvidia/MambaVision-T-1K), ImageNet-pretrained, whose first convolution
takes the 12 Sentinel-2 bands (RGB filters copied to B04/B03/B02, their mean to the other bands), fine-tuned. Question encoder: a frozen multilingual sentence encoder
(paraphrase-multilingual-MiniLM-L12-v2, mean pooling). Fusion as in RSVQA: both
projected to 1200-d with tanh, element-wise product, MLP, classifier over the answer vocabulary. The softmax is
restricted to the options of each question (the same information EarthJev receives), trained with cross-entropy.
Checkpoint chosen on validation accuracy. Outputs slot logits in the format of s1geo.onejev_eval
(probe.logits.jsonl), so metrics are computed by the same code as for EarthJev.

    python -m s1geo.rsvqa_baseline train --train data/earthjev/reben5pct_en/train.jsonl --val data/earthjev/reben5pct_en/validation_1k.jsonl \
        --subset data/reben/subset_5pct --out runs/rsvqa_baseline/<name> [--bands rgb]
    python -m s1geo.rsvqa_baseline eval --run runs/rsvqa_baseline/<name> --questions <jsonl> --subset ... --out runs/eval/<name>
    python -m s1geo.rsvqa_baseline bench --run runs/rsvqa_baseline/<name> --questions <jsonl> --subset ... --out runs/latency/<name>.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zarr

ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
RGB = [3, 2, 1]


def options(q: dict) -> list[str]:
    return ["yes", "no"] if q["type"] == "noul" else list(q["criteria"])


class TextEncoder:
    """Frozen sentence encoder with an embedding cache (questions are few distinct strings)."""

    def __init__(self, device):
        import transformers
        self.tok = transformers.AutoTokenizer.from_pretrained(ENCODER)
        self.model = transformers.AutoModel.from_pretrained(ENCODER).to(device).eval()
        self.device, self.cache = device, {}

    @torch.no_grad()
    def encode(self, texts: list[str], use_cache: bool = True) -> torch.Tensor:
        if not use_cache:   # latency measurements: every question is encoded, repeated texts included
            enc = self.tok(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"][..., None].float()
            return (h * m).sum(1) / m.sum(1)
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"][..., None].float()
            emb = (h * m).sum(1) / m.sum(1)
            self.cache.update(zip(chunk, emb))
        return torch.stack([self.cache[t] for t in texts])


def adapt_first_conv(net: nn.Module, in_ch: int, rgb_idx: list[int] | None) -> None:
    """Replace the first 3-channel convolution with one for `in_ch` bands: RGB filters copied to B04/B03/B02, their
    mean to the other bands (scaled so the summed response keeps its magnitude)."""
    for parent in net.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                w = child.weight.data
                conv = nn.Conv2d(in_ch, child.out_channels, child.kernel_size, child.stride, child.padding,
                                 bias=child.bias is not None)
                with torch.no_grad():
                    if rgb_idx is None:
                        conv.weight.copy_(w)
                    else:
                        conv.weight.copy_(w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3 / in_ch)
                        for j, b in enumerate(rgb_idx):
                            conv.weight[:, b] = w[:, j] * 3 / in_ch
                    if child.bias is not None:
                        conv.bias.copy_(child.bias)
                setattr(parent, name, conv)
                return
    raise ValueError("no 3-channel convolution found")


def image_encoder(name: str) -> tuple[nn.Module, int, int]:
    """(ImageNet-pretrained encoder returning pooled features, feature size, input size) for the three families
    compared in RSJEV: CNN (ResNet-50), Transformer (Swin-T), Mamba (MambaVision-T)."""
    import torchvision
    if name == "resnet50":
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Identity()
        return net, 2048, 120
    if name == "swin_t":
        net = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
        net.head = nn.Identity()
        return net, 768, 224
    if name == "mambavision_t":
        # The released remote code cannot be loaded by transformers 5, so the plain nn.Module of their
        # modeling_mambavision.py is built from config.json and the weights are loaded directly.
        import importlib
        import sys
        import types

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        d = Path(hf_hub_download("nvidia/MambaVision-T-1K", "config.json")).parent   # works offline on a partial snapshot
        if "mambavision_hf" not in sys.modules:
            pkg = types.ModuleType("mambavision_hf"); pkg.__path__ = [str(d)]; sys.modules["mambavision_hf"] = pkg
        mod = importlib.import_module("mambavision_hf.modeling_mambavision")
        cfg = json.loads((d / "config.json").read_text())
        net = mod.MambaVision(depths=cfg["depths"], num_heads=cfg["num_heads"], window_size=cfg["window_size"],
                              dim=cfg["dim"], in_dim=cfg["in_dim"], mlp_ratio=cfg["mlp_ratio"])
        state = {k.removeprefix("model."): v for k, v in load_file(str(d / "model.safetensors")).items()}
        missing, unexpected = net.load_state_dict(state, strict=False)
        assert not [k for k in missing if not k.startswith("head.")] and not unexpected, (missing, unexpected)

        class MV(nn.Module):   # forward_features returns (pooled features, stage features)
            def __init__(self):
                super().__init__()
                self.m = net

            def forward(self, x):
                return self.m.forward_features(x)[0]
        return MV(), 640, 224
    raise ValueError(name)


class RSVQANet(nn.Module):
    def __init__(self, in_ch: int, n_answers: int, rgb_idx: list[int] | None, text_dim: int = 384, hid: int = 1200,
                 encoder: str = "resnet50"):
        super().__init__()
        net, dim, self.size = image_encoder(encoder)
        adapt_first_conv(net, in_ch, rgb_idx)
        self.image = net
        self.img_proj = nn.Sequential(nn.Linear(dim, hid), nn.Tanh(), nn.Dropout(0.5))
        self.txt_proj = nn.Sequential(nn.Linear(text_dim, hid), nn.Tanh(), nn.Dropout(0.5))
        self.head = nn.Sequential(nn.Linear(hid, 256), nn.Tanh(), nn.Dropout(0.5), nn.Linear(256, n_answers))

    def forward(self, x, t):
        if x.shape[-1] != self.size:   # transformer / Mamba encoders were pretrained at 224 x 224
            x = nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.head(self.img_proj(self.image(x)) * self.txt_proj(t))


class Images:
    """All patches of the needed splits in memory (uint16), standardised on the GPU."""

    def __init__(self, subset: Path, splits: list[str], bands: str, stats: dict, device):
        self.idx, arrays = {}, []
        off = 0
        for s in splits:
            ids = pd.read_parquet(subset / f"{s}_index.parquet")["patch_id"].tolist()
            arrays.append(np.asarray(zarr.open_group(str(subset / f"{s}.zarr"), mode="r")["s2"][:]))
            self.idx.update({p: off + i for i, p in enumerate(ids)})
            off += len(ids)
        self.x = torch.from_numpy(np.concatenate(arrays).astype(np.int32))
        self.sel = RGB if bands == "rgb" else list(range(12))
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)[self.sel, None, None]
        self.std = torch.tensor(stats["std"], dtype=torch.float32, device=device)[self.sel, None, None]
        self.device = device

    def get(self, pids: list[str]) -> torch.Tensor:
        x = self.x[[self.idx[p] for p in pids]][:, self.sel].to(self.device).float()
        return (x - self.mean) / self.std


def batches(rows, answers, size, shuffle, rng):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        b = [rows[j] for j in order[i:i + size]]
        mask = torch.full((len(b), len(answers)), float("-inf"))
        tgt = torch.zeros(len(b), len(answers))
        for k, r in enumerate(b):
            for o in options(r["question"]):
                mask[k, answers[o]] = 0.0
            for o, p in r["target"].items():
                tgt[k, answers[o]] = p
        yield b, mask, tgt


def load_rows(path):
    return [json.loads(l) for l in open(path)]


def split_of(rows):
    return sorted({r["media"][0]["path"].split("/")[1] for r in rows})


def train(args) -> None:
    dev = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tr, va = load_rows(args.train), load_rows(args.val)
    answers = sorted({o for r in tr + va for o in options(r["question"])})
    amap = {a: i for i, a in enumerate(answers)}
    seg = sorted(Path("runs/seg").glob("*_unet_reben5pct_l2"))[0]
    stats = json.loads((seg / "band_stats.json").read_text())   # training-split band statistics
    imgs = Images(Path(args.subset), split_of(tr + va), args.bands, stats, dev)
    txt = TextEncoder(dev)
    txt.encode([r["question"]["instructions"] for r in tr + va])
    model = RSVQANet(3 if args.bands == "rgb" else 12, len(answers), None if args.bands == "rgb" else RGB,
                     encoder=args.encoder).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(tr) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "answers": answers, "band_stats": str(seg)}, indent=1))
    best, log = -1.0, open(out / "log.jsonl", "w")
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for b, mask, tgt in batches(tr, amap, args.batch_size, True, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(imgs.get([r["meta"]["item_id"] for r in b]), txt.encode([r["question"]["instructions"] for r in b]))
            logp = (logits.float() + mask.to(dev)).log_softmax(-1)
            loss = -(tgt.to(dev) * logp.nan_to_num(neginf=0.0)).sum(-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(b)
        acc = accuracy(model, va, amap, imgs, txt)
        rec = {"epoch": ep + 1, "train_loss": tot / len(tr), "val_acc": acc, "min": round((time.time() - t0) / 60, 2)}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out / "best.pt")
    (out / "best.json").write_text(json.dumps({"val_acc": best}))


@torch.no_grad()
def predict(model, rows, amap, imgs, txt, size=256):
    model.eval()
    for b, mask, tgt in batches(rows, amap, size, False, None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(imgs.get([r["meta"]["item_id"] for r in b]), txt.encode([r["question"]["instructions"] for r in b]))
        yield b, logits.float().cpu()


def accuracy(model, rows, amap, imgs, txt) -> float:
    ok = n = 0
    for b, logits in predict(model, rows, amap, imgs, txt):
        for r, z in zip(b, logits):
            opts = options(r["question"])
            pred = opts[int(torch.stack([z[amap[o]] for o in opts]).argmax())]
            ok += pred == max(r["target"], key=r["target"].get); n += 1
    return ok / n


def restore(run: Path, rows, subset: Path, dev):
    cfg = json.loads((run / "config.json").read_text())
    amap = {a: i for i, a in enumerate(cfg["answers"])}
    stats = json.loads((Path(cfg["band_stats"]) / "band_stats.json").read_text())
    imgs = Images(subset, split_of(rows), cfg["bands"], stats, dev)
    model = RSVQANet(3 if cfg["bands"] == "rgb" else 12, len(amap), None if cfg["bands"] == "rgb" else RGB,
                     encoder=cfg.get("encoder", "resnet50")).to(dev)
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    return model.eval(), amap, imgs


def evaluate(args) -> None:
    dev = torch.device("cuda:0")
    rows = load_rows(args.questions)
    model, amap, imgs = restore(Path(args.run), rows, Path(args.subset), dev)
    txt = TextEncoder(dev)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "probe.logits.jsonl", "w") as fh:
        for b, logits in predict(model, rows, amap, imgs, txt):
            for r, z in zip(b, logits):
                opts = options(r["question"])
                fh.write(json.dumps({"id": r["id"], "task": r.get("task"), "kind": r["question"]["type"], "labels": opts,
                                     "logits": [float(z[amap[o]]) for o in opts],
                                     "target": [float(r["target"].get(o, 0.0)) for o in opts],
                                     "item_id": r["meta"]["item_id"]}) + "\n")
    print(f"wrote {out / 'probe.logits.jsonl'}", flush=True)


@torch.no_grad()
def bench(args) -> None:
    """Latency, same protocol as s1geo.bench_latency (warm-up excluded, question encoding included): "single", one
    question per request; "image", all the questions of an image in one forward (the image repeated in the batch)."""
    from s1geo.bench_latency import pick, time_scenarios
    dev = torch.device("cuda:0")
    groups = pick(Path(args.questions), args.images)
    model, amap, imgs = restore(Path(args.run), [r for g in groups for r in g], Path(args.subset), dev)
    txt = TextEncoder(dev)

    def run(rows):
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(imgs.get([r["meta"]["item_id"] for r in rows]),
                  txt.encode([r["question"]["instructions"] for r in rows], use_cache=False))
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    res = time_scenarios(run, groups, args.warmup)
    res.update({"run": args.run, "gpu": torch.cuda.get_device_name(0), "warmup": args.warmup,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True); t.add_argument("--val", required=True)
    t.add_argument("--subset", required=True); t.add_argument("--out", required=True)
    t.add_argument("--bands", choices=["all", "rgb"], default="all")
    t.add_argument("--encoder", choices=["resnet50", "swin_t", "mambavision_t"], default="resnet50")
    t.add_argument("--epochs", type=int, default=12); t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--seed", type=int, default=0)
    for name in ("eval", "bench"):
        e = sub.add_parser(name)
        e.add_argument("--run", required=True); e.add_argument("--questions", required=True)
        e.add_argument("--subset", required=True); e.add_argument("--out", required=True)
        e.add_argument("--images", type=int, default=110); e.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args(argv)
    {"train": train, "eval": evaluate, "bench": bench}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
