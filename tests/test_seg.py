import numpy as np
import torch

from s1geo.data.reben import encode_mask
from s1geo.models.unet import UNet
from s1geo.train_seg import pixel_metrics, to_level1


def test_unet_shape_120():
    out = UNet(12, 15, width=8, depth=3)(torch.randn(2, 12, 120, 120))
    assert out.shape == (2, 15, 120, 120)


def test_encode_mask_level2():
    m = np.array([[111, 112], [312, 0]], dtype=np.uint16)
    # level-2 codes sorted 11,12,13,14,21,22,23,24,31,...: 11 -> 0, 31 -> 8; invalid -> -1
    assert encode_mask(m, 2).tolist() == [[0, 0], [8, -1]]


def test_level1_aggregation_and_metrics():
    probs = torch.zeros(1, 15, 1, 2)
    probs[0, 0, 0, 0] = 1.0   # class 11 -> level-1 class 1 (index 0)
    probs[0, 8, 0, 1] = 1.0   # class 31 -> level-1 class 3 (index 2)
    p1 = to_level1(probs, 2)
    assert p1[0, 0, 0, 0] == 1 and p1[0, 2, 0, 1] == 1
    m = pixel_metrics(p1, torch.tensor([[[0, 2]]]), 5)
    assert m["oa"] == 1.0 and m["miou"] == 1.0 and m["ece15"] == 0.0
