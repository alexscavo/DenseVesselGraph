"""Checks for the 3D dynamic snake convolution port (4 Nov 2026)."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from models.dscnet3d import DSConv3d  # noqa: E402
import dense_graph_20261030 as D  # noqa: E402


def count(m):
    return sum(p.numel() for p in m.parameters())


def test_snake_centre_pinned_and_cumulative():
    m = DSConv3d(2, 4, kernel_size=9)
    off = torch.rand(1, 9, 2, 2, 2) * 2 - 1
    cum = m.snake(off)
    assert torch.all(cum[:, 4] == 0)
    assert torch.allclose(cum[:, 5], off[:, 5]) and torch.allclose(cum[:, 8], off[:, 5:9].sum(1))
    assert torch.allclose(cum[:, 3], off[:, 3]) and torch.allclose(cum[:, 0], off[:, 0:4].sum(1))


def test_zero_offsets_equal_straight_line_conv():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 8, 9)
    for morph, axis in ((0, 4), (1, 3), (2, 2)):  # W, H, D in (B, C, D, H, W)
        m = DSConv3d(3, 5, kernel_size=5, morph=morph).eval()
        with torch.no_grad():
            m.offset_conv.weight.zero_(); m.offset_conv.bias.zero_()
            y = m(x)
            w = m.combine.weight.view(5, 3, 5)  # (out, C, K): combine input is C-major, then K
            shape = [5, 3, 1, 1, 1]; shape[axis] = 5
            pad = [0] * 6; pad[2 * (4 - axis)] = pad[2 * (4 - axis) + 1] = 2
            ref = F.conv3d(F.pad(x, pad, mode='replicate'), w.view(shape), m.combine.bias)
            ref = F.relu(m.norm(ref))
        assert torch.allclose(y, ref, atol=1e-4), morph


def test_backbones_shapes():
    x = torch.randn(1, 1, 64, 64, 64)
    nets = {b: D.DenseGraph(backbone=b).eval() for b in ('unet', 'dsc3', 'dsc9')}
    for b, n in nets.items():
        with torch.no_grad():
            y = n.forward_image(x)
        assert y.shape == (1, 37, 64, 64, 64) and torch.isfinite(y).all(), b
    print('params', {b: count(n.unet) for b, n in nets.items()})


if __name__ == '__main__':
    test_snake_centre_pinned_and_cumulative()
    test_zero_offsets_equal_straight_line_conv()
    test_backbones_shapes()
    print('ALL TESTS PASSED')
