import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from s1geo import msi

SUBSET = Path("data/reben/subset_5pct")


class _Embed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.proj = torch.nn.Conv3d(3, 8, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True)


class _Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _Embed()
        self.block = torch.nn.Linear(8, 8)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()
        self.lm = torch.nn.Linear(8, 8)
        self.config = SimpleNamespace(vision_config=SimpleNamespace(in_channels=3))


def test_widened_layer_matches_rgb_layer():
    torch.manual_seed(0)
    model = _Model()
    old = torch.nn.Conv3d(3, 8, kernel_size=(2, 16, 16), stride=(2, 16, 16))
    old.load_state_dict(model.visual.patch_embed.proj.state_dict())
    msi.widen_input_layer(model)
    assert model.config.vision_config.in_channels == 12 and model.visual.patch_embed.in_channels == 12
    x12 = torch.randn(5, 12, 2, 16, 16)
    rgb = x12[:, [msi.BANDS.index(b) for b in msi.RGB_BANDS]]
    torch.testing.assert_close(model.visual.patch_embed.proj(x12), old(rgb))


def test_freeze_keeps_keys_and_trains_input_layer_only():
    model = _Model()
    msi.widen_input_layer(model)
    keys = set(model.state_dict())
    assert msi.freeze_vision_except_input(model) == 2  # block weight and bias
    assert set(model.state_dict()) == keys
    trainable = {n for n, _ in model.named_parameters()}
    assert trainable == {"visual.patch_embed.proj.weight", "visual.patch_embed.proj.bias", "lm.weight", "lm.bias"}


def test_norm_scale_and_finish():
    norm = msi.BandNorm(bands=list(msi.BANDS), p2=[0.0, 0.0, 0.0], p98=[100.0, 100.0, 100.0],
                        mean=[10.0] * 12, std=[2.0] * 12)
    x = torch.full((1, 12, 4, 4), 50.0)
    y = norm.scale(x)
    rgb = norm.rgb_idx
    other = [i for i in range(12) if i not in rgb]
    assert torch.allclose(y[:, rgb], torch.tensor(0.5)) and torch.allclose(y[:, other], torch.tensor(20.0))
    z = norm.finish(y, [0.5] * 3, [0.5] * 3)
    assert torch.allclose(z[:, rgb], torch.tensor(0.0)) and torch.allclose(z[:, other], torch.tensor(20.0))


@pytest.mark.skipif(not (SUBSET / "msi_norm.json").exists(), reason="needs the reBEN 5% subset")
def test_rgb_channels_match_the_png_pipeline():
    """The B04/B03/B02 channels of the 12-band input equal the processor's pixels for the RGB PNG of the same patch,
    up to the PNG's 8-bit rounding."""
    import pandas as pd
    import zarr
    from PIL import Image
    from transformers import AutoProcessor

    try:
        proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5-0.8B processor not in the local cache")
    norm = msi.BandNorm.load(SUBSET / "msi_norm.json")
    ids = pd.read_parquet(SUBSET / "test_index.parquet")["patch_id"].tolist()
    s2 = zarr.open_group(str(SUBSET / "test.zarr"), mode="r")["s2"]
    rows = [0, 7, 123]
    patches = [s2[r] for r in rows]
    pngs = [Image.open(SUBSET / "rgb" / "test" / f"{ids[r]}.png").convert("RGB") for r in rows]
    ours = msi.pixel_inputs(patches, proc.image_processor, norm)
    ref = proc.image_processor(images=pngs, return_tensors="pt")
    assert torch.equal(ours["image_grid_thw"], ref["image_grid_thw"])
    p = proc.image_processor
    ours_rgb = ours["pixel_values"].view(-1, 12, p.temporal_patch_size, p.patch_size, p.patch_size)[:, norm.rgb_idx]
    ref_rgb = ref["pixel_values"].view(-1, 3, p.temporal_patch_size, p.patch_size, p.patch_size)
    diff = (ours_rgb - ref_rgb).abs().flatten()
    # One 8-bit step is 2/255 after (v - 0.5) / 0.5. The PNG rounds the stretch, and torchvision's uint8 bicubic
    # resize rounds and clamps to [0, 255] between its horizontal and vertical passes while the float path does not,
    # so a few pixels next to saturated areas differ by up to ~10 steps (bicubic overshoot).
    step = 2 / 255
    q999 = diff.quantile(0.999).item()
    assert diff.mean().item() < step and q999 < 2 * step and diff.max().item() < 15 * step, \
        (diff.mean().item(), q999, diff.max().item())


def test_drop_extra_bands_sets_them_to_the_mean():
    norm = msi.BandNorm(bands=list(msi.BANDS), p2=[0.0] * 3, p98=[100.0] * 3, mean=[10.0] * 12, std=[2.0] * 12,
                        drop_extra=True)
    y = norm.scale(torch.full((1, 12, 4, 4), 50.0))
    other = [i for i in range(12) if i not in norm.rgb_idx]
    assert torch.all(y[:, other] == 0) and torch.allclose(y[:, norm.rgb_idx], torch.tensor(0.5))


def test_rgb_mean_init():
    torch.manual_seed(0)
    model = _Model()
    w = model.visual.patch_embed.proj.weight.detach().clone()
    msi.widen_input_layer(model, init="rgb_mean")
    new = model.visual.patch_embed.proj.weight
    other = [i for i in range(12) if msi.BANDS[i] not in msi.RGB_BANDS]
    torch.testing.assert_close(new[:, other], w.mean(dim=1, keepdim=True).expand(-1, len(other), -1, -1, -1))
    torch.testing.assert_close(new[:, [msi.BANDS.index(b) for b in msi.RGB_BANDS]], w)
