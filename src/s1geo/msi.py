"""12-band Sentinel-2 input for Qwen3.5 (EarthJev).

The vision encoder's input layer (a Conv3d over 16x16 pixels x 2 frames) is widened from 3 to 12 channels: the
R, G, B kernels are copied onto B04, B03, B02 and the 9 other bands start at zero, so before training the model
computes exactly what the RGB model computes on the same R, G, B values. Only this layer is trained in the vision
encoder; the rest of the encoder is held as buffers, so OneJev's `--train-vision` trains the input layer alone.

Pixels: media rows reference a Zarr patch, {"type": "image", "path": "<split>.zarr", "row": i, "h": H, "w": W},
relative to the media root; the Zarr holds `s2` uint16 (N, 12, H, W) in BANDS order (scripts/data/prepare_reben.py,
prepare_eurosat_ms.py). Per band, before Qwen's own resize and patchify:
  - B04, B03, B02: the fixed linear stretch of the RGB PNGs (reben_rgb.py, [p2, p98] -> [0, 1], clipped), then after
    the resize the processor's own clamp to [0, 1] and (v - 0.5) / 0.5, as for an RGB image;
  - the other bands: z-score with the reBEN train mean and std.
Statistics live in a JSON file (make_norm) saved with every checkpoint as msi_norm.json.

OneJev's training and evaluation code is used unchanged: `install` replaces train.mm.mm_inputs (MSI media through
the 12-band image processor, anything else through the original path), train.common.load_bundle (widen and freeze
after loading) and train.sft.save_weights (also writes msi_norm.json). With `input_lr`, the optimizer train.sft
builds (torch.optim.AdamW) gets a second parameter group, the input layer, at that learning rate; the scheduler
applies the same warmup and cosine factor to both groups.
"""
from __future__ import annotations

import copy
import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger("s1geo.msi")

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
RGB_BANDS = ("B04", "B03", "B02")  # the base model's input channels R, G, B, in this order
NORM_FILE = "msi_norm.json"


@dataclass
class BandNorm:
    bands: list[str]
    p2: list[float]  # RGB_BANDS stretch
    p98: list[float]
    mean: list[float]  # all bands; used for the non-RGB ones
    std: list[float]
    drop_extra: bool = False  # test-time ablation: non-RGB bands set to their mean (0 after the z-score)

    @classmethod
    def load(cls, path: str | Path) -> "BandNorm":
        d = json.loads(Path(path).read_text())
        norm = cls(bands=d["bands"], p2=d["p2"], p98=d["p98"], mean=d["mean"], std=d["std"])
        if norm.bands != BANDS:
            raise ValueError(f"{path}: bands {norm.bands} != {BANDS}")
        return norm

    def to_json(self) -> str:
        return json.dumps({"bands": self.bands, "rgb_bands": list(RGB_BANDS), "p2": self.p2, "p98": self.p98,
                           "mean": self.mean, "std": self.std}, indent=1)

    @property
    def rgb_idx(self) -> list[int]:
        return [self.bands.index(b) for b in RGB_BANDS]

    def scale(self, x):
        """(N, 12, H, W) float tensor of raw values -> per-band values before resizing."""
        import torch

        out = (x - torch.tensor(self.mean, dtype=x.dtype)[:, None, None]) / torch.tensor(self.std, dtype=x.dtype)[:, None, None]
        idx = self.rgb_idx
        lo = torch.tensor(self.p2, dtype=x.dtype)[:, None, None]
        hi = torch.tensor(self.p98, dtype=x.dtype)[:, None, None]
        if self.drop_extra:
            out.zero_()
        out[:, idx] = ((x[:, idx] - lo) / (hi - lo)).clamp(0, 1)
        return out

    def finish(self, x, image_mean: Sequence[float], image_std: Sequence[float]):
        """Resized values -> model input: RGB bands clamped to [0, 1] and normalised like an RGB image."""
        import torch

        idx = self.rgb_idx
        m = torch.tensor(image_mean, dtype=x.dtype)[:, None, None]
        s = torch.tensor(image_std, dtype=x.dtype)[:, None, None]
        x = x.clone()
        x[:, idx] = (x[:, idx].clamp(0, 1) - m) / s
        return x


def make_norm(zarr_path: str | Path, stretch_json: str | Path, sample: int = 2000, seed: int = 0) -> BandNorm:
    """Per-band mean and std over `sample` random patches of a training split; RGB stretch from the PNG pipeline."""
    import zarr

    s2 = zarr.open_group(str(zarr_path), mode="r")["s2"]
    rows = np.sort(np.random.default_rng(seed).choice(s2.shape[0], min(sample, s2.shape[0]), replace=False))
    x = np.stack([s2[int(r)] for r in rows]).astype(np.float64)
    st = json.loads(Path(stretch_json).read_text())
    if tuple(st["bands"]) != RGB_BANDS:
        raise ValueError(f"{stretch_json}: stretch bands {st['bands']} != {RGB_BANDS}")
    return BandNorm(bands=list(BANDS), p2=list(st["p2"]), p98=list(st["p98"]),
                    mean=x.mean(axis=(0, 2, 3)).tolist(), std=x.std(axis=(0, 2, 3)).tolist())


@functools.lru_cache(maxsize=64)
def _open_s2(path: str):
    import zarr

    g = zarr.open_group(path, mode="r")
    if list(g.attrs.get("bands", [])) != BANDS:
        raise ValueError(f"{path}: bands {g.attrs.get('bands')} != {BANDS}")
    return g["s2"]


def is_msi(media: dict) -> bool:
    return media.get("type") == "image" and "row" in media


def load_patch(root: str | Path, media: dict) -> np.ndarray:
    a = _open_s2(str(Path(root) / media["path"]))[int(media["row"])]
    if a.shape[1:] != (int(media["h"]), int(media["w"])):
        raise ValueError(f"{media}: stored patch is {a.shape}")
    return a


def pixel_inputs(patches: Sequence[np.ndarray], image_processor, norm: BandNorm) -> dict:
    """pixel_values (sum of patches, 12 * 2 * 16 * 16) and image_grid_thw, with Qwen's own resize and patchify."""
    import torch

    ip = image_processor
    values, grids = [], []
    for a in patches:
        x = norm.scale(torch.from_numpy(np.asarray(a, dtype=np.float32))[None])
        x = ip.resize(x, size=ip.size, resample=ip.resample, factor=ip.patch_size * ip.merge_size)
        x = norm.finish(x, ip.image_mean, ip.image_std)
        p, gh, gw = ip.patchify(x, patch_size=ip.patch_size, merge_size=ip.merge_size,
                                temporal_patch_size=ip.temporal_patch_size)
        values.append(p[0])
        grids.append([1, gh, gw])
    return {"pixel_values": torch.cat(values), "image_grid_thw": torch.tensor(grids, dtype=torch.long)}


def rgb_as_patch(image, norm: BandNorm) -> np.ndarray:
    """A 12-band patch from an RGB image (8-bit): R, G, B undo the PNG stretch onto B04, B03, B02, the other bands sit at
    their training mean. For callers that only have RGB pixels (OneJev's engine self-test); not used for real data."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    out = np.array(norm.mean, dtype=np.float32)[:, None, None].repeat(rgb.shape[1], 1).repeat(rgb.shape[2], 2)
    lo, hi = np.array(norm.p2, dtype=np.float32), np.array(norm.p98, dtype=np.float32)
    out[norm.rgb_idx] = rgb * (hi - lo)[:, None, None] + lo[:, None, None]
    return out


class MSIImageProcessor:
    """Stands in for the processor's image processor: takes 12-band patches, returns Qwen's pixel layout."""

    def __init__(self, base, norm: BandNorm):
        self.base, self.norm = base, norm

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, images, **kwargs):
        from transformers.image_processing_utils import BatchFeature

        images = [rgb_as_patch(im, self.norm) if hasattr(im, "convert") else im for im in images]
        return BatchFeature(data=pixel_inputs(images, self.base, self.norm), tensor_type=kwargs.get("return_tensors"))


def msi_processor(processor, norm: BandNorm):
    """Shallow copy of a Qwen3-VL processor whose image processor reads 12-band patches."""
    p = copy.copy(processor)
    p.image_processor = MSIImageProcessor(processor.image_processor, norm)
    return p


def _patch_embed(model):
    mods = [m for n, m in model.named_modules() if n.endswith("visual.patch_embed")]
    if len(mods) != 1:
        raise ValueError(f"expected one visual.patch_embed, found {len(mods)}")
    return mods[0]


def _visual(model):
    mods = [m for n, m in model.named_modules() if n.endswith("visual") and hasattr(m, "patch_embed")]
    if len(mods) != 1:
        raise ValueError(f"expected one vision tower, found {len(mods)}")
    return mods[0]


def widen_input_layer(model, init: str = "zero") -> None:
    """3 -> 12 input channels: RGB kernels copied onto B04, B03, B02, bias unchanged. The other bands start at zero
    (init="zero", the model then equals the RGB model) or at the mean of the R, G, B kernels (init="rgb_mean", ablation)."""
    if init not in ("zero", "rgb_mean"):
        raise ValueError(f"unknown init {init!r}")
    import torch

    pe = _patch_embed(model)
    old = pe.proj
    if old.in_channels != 3:
        raise ValueError(f"input layer has {old.in_channels} channels, expected 3")
    new = torch.nn.Conv3d(len(BANDS), old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                          bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        if init == "rgb_mean":
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).expand_as(new.weight))
        new.weight[:, [BANDS.index(b) for b in RGB_BANDS]] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    pe.proj = new
    pe.in_channels = len(BANDS)
    model.config.vision_config.in_channels = len(BANDS)


def freeze_vision_except_input(model) -> int:
    """Turn every vision-tower parameter outside the input layer into a persistent buffer (same state_dict keys),
    so no optimizer or requires_grad switch can train it. Returns the number of frozen tensors."""
    visual = _visual(model)
    n = 0
    for name, mod in visual.named_modules():
        if name == "patch_embed" or name.startswith("patch_embed."):
            continue
        for pname, prm in list(mod._parameters.items()):
            if prm is None:
                continue
            del mod._parameters[pname]
            mod.register_buffer(pname, prm.detach(), persistent=True)
            n += 1
    return n


def prepare_model(model, init: str = "zero") -> None:
    """Widen a 3-channel base checkpoint (12-channel ones load as such) and freeze the rest of the vision tower."""
    if _patch_embed(model).proj.in_channels == 3:
        widen_input_layer(model, init)
        log.info("input layer widened to %d channels (RGB kernels on %s, other bands: %s)", len(BANDS), RGB_BANDS, init)
    if _patch_embed(model).proj.in_channels != len(BANDS):
        raise ValueError(f"input layer has {_patch_embed(model).proj.in_channels} channels")
    n = freeze_vision_except_input(model)
    log.info("vision tower: %d tensors frozen as buffers, input layer trainable", n)


def install(norm: BandNorm, freeze: bool = True, input_lr: float | None = None, init: str = "zero") -> None:
    """Route MSI media through the 12-band path in OneJev's train package (see module docstring)."""
    import torch
    from train import common, mm

    input_params: set[int] = set()

    orig_mm_inputs = mm.mm_inputs
    orig_load_bundle = common.load_bundle

    def mm_inputs(processor, prompts, medias, device):
        flags = {is_msi(m) for media in medias for m in media}
        if flags != {True}:
            if True in flags:
                raise ValueError("a batch mixes 12-band and other media")
            return orig_mm_inputs(processor, prompts, medias, device)
        if any(m["type"] != "image" for media in medias for m in media):
            raise ValueError("12-band rows carry images only")
        patches = [load_patch(mm.MM_ROOT, m) for media in medias for m in media]
        processor.tokenizer.padding_side = "right"
        out = msi_processor(processor, norm)(text=list(prompts), images=patches, padding=True, return_tensors="pt")
        out = {k: v.to(device) for k, v in out.items()}
        ends = (out["attention_mask"].sum(dim=1) - 1).tolist()
        return out, ends

    def load_bundle(*args, **kwargs):
        bundle = orig_load_bundle(*args, **kwargs)
        if bundle.model is not None and kwargs.get("multimodal"):
            if freeze:
                prepare_model(bundle.model, init)
            elif _patch_embed(bundle.model).proj.in_channels == 3:
                widen_input_layer(bundle.model, init)
            input_params.update(id(p) for p in _patch_embed(bundle.model).parameters())
        return bundle

    mm.mm_inputs = mm_inputs
    common.load_bundle = load_bundle
    if input_lr is not None:
        orig_adamw = torch.optim.AdamW

        def adamw(params, lr, **kwargs):
            params = list(params)
            rest = [p for p in params if id(p) not in input_params]
            inp = [p for p in params if id(p) in input_params]
            if not inp:
                raise ValueError("input layer parameters are not among the optimizer's parameters")
            log.info("optimizer groups: %d tensors at lr %.2e, input layer (%d tensors) at lr %.2e",
                     len(rest), lr, len(inp), input_lr)
            return orig_adamw([{"params": rest}, {"params": inp, "lr": input_lr}], lr=lr, **kwargs)

        torch.optim.AdamW = adamw
    try:
        from train import sft
    except ImportError:
        return
    orig_save = sft.save_weights

    def save_weights(model, bundle, out_dir, *args, **kwargs):
        orig_save(model, bundle, out_dir, *args, **kwargs)
        if Path(out_dir).is_dir():
            (Path(out_dir) / NORM_FILE).write_text(norm.to_json())

    sft.save_weights = save_weights
