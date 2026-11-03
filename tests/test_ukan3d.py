"""Shape and size checks for the 3D U-KAN stages (3 Nov 2026)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
from models.ukan3d import KANLinear  # noqa: E402


def count(m):
    return sum(p.numel() for p in m.parameters())


def test_ukan_shapes_and_sizes():
    torch.manual_seed(0)
    nets = {b: D.DenseGraph(backbone=b) for b in ('unet', 'ukan', 'ukan_mlp', 'ukan_add')}
    x = torch.randn(1, 1, 64, 64, 64)
    for b, net in nets.items():
        with torch.no_grad():
            y = net.forward_image(x)
        assert y.shape == (1, 37, 64, 64, 64) and torch.isfinite(y).all(), b
    for name in ('down_3', 'down_4', 'upcat_4', 'upcat_3'):
        print(name, {b: count(getattr(n.unet, name)) for b, n in nets.items()})
    print('totals', {b: count(n.unet) for b, n in nets.items()})


def test_kan_linear_matches_bspline_partition_of_unity():
    layer = KANLinear(4, 2)
    x = torch.rand(10, 4) * 1.8 - 0.9  # inside the grid range
    assert torch.allclose(layer.b_splines(x).sum(-1), torch.ones(10, 4), atol=1e-5)


if __name__ == '__main__':
    test_kan_linear_matches_bspline_partition_of_unity()
    test_ukan_shapes_and_sizes()
    print('ALL TESTS PASSED')
