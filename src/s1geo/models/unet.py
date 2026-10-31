"""Compact U-Net for 120x120 multispectral patches (pure PyTorch, no extra dependencies)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, width: int = 32, depth: int = 3):
        super().__init__()
        chans = [width * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(in_ch, chans[0])] + [_block(chans[i], chans[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_block(2 * chans[i], chans[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chans[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            skip = skips.pop()
            x = up(x)
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]])
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)
