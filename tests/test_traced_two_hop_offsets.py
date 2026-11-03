"""Unit checks for the opt-in two-hop candidates and neighbourhood offset targets (3 Nov 2026)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_traced_20261030 as T  # noqa: E402


def test_two_hop_pairs():
    # chain 0-1-2 plus a junction 1-3; 0-2 not touching
    contacts = {(0, 1): (1.0, None), (1, 2): (2.0, None), (1, 3): (5.0, None), (0, 3): (9.0, None)}
    hops = T.two_hop_pairs(contacts)
    assert hops[(0, 2)] == 1
    assert hops[(2, 3)] == 1
    assert (0, 3) not in hops  # already touching
    assert (0, 1) not in hops


def test_neighbourhood_offsets_land_on_gt():
    nodes = [torch.tensor([[10.3, 20.6, 30.1], [11.2, 20.9, 30.4], [50.0, 5.0, 0.2]]) / T.SIZE]
    offset, mask = T.neighbourhood_offsets(nodes, 'cpu', 1.75)
    vox = mask[0].nonzero()
    pos = vox.float() + offset[0][:, vox[:, 0], vox[:, 1], vox[:, 2]].T
    gt = nodes[0] * T.SIZE
    d = torch.cdist(pos, gt).min(1).values
    assert d.max() < 1e-4  # voxel + target is exactly a GT node
    # every supervised voxel points to its nearest GT node
    nearest = torch.cdist(vox.float(), gt).argmin(1)
    assert torch.allclose(pos, gt[nearest], atol=1e-4)
    # the rounded GT voxel keeps the original sub-voxel target
    c = gt.round().long()
    assert mask[0, c[:, 0], c[:, 1], c[:, 2]].all()
    # 3x3x3 neighbourhood per isolated node, clipped at the boundary (node 2 sits at W=0)
    assert int(mask.sum()) < 3 * 27 and int(mask.sum()) > 27


def test_contract_chain():
    import numpy as np
    pairs = np.array([[0, 1], [1, 2], [0, 2], [1, 3]])
    row = dict(node_scores=np.array([0.9, 0.5, 0.9, 0.9]), edge_scores=np.array([0.9, 0.9, 0.95, 0.1]),
               pairs=pairs, via=np.array([-1, -1, 1, -1]), pass_scores=np.array([np.nan, np.nan, 0.8, np.nan]))
    plain = T.contract(row, 0.3, 0.5, None)
    assert plain['edge_scores'][2] == 0.0 and plain['edge_scores'][0] == 0.9  # chain never an ordinary edge
    out = T.contract(row, 0.3, 0.5, 0.6)
    assert out['contracted'] == [(0, 1, 2)]
    assert out['node_scores'][1] < 0 and out['edge_scores'][2] == 1.0
    assert out['edge_scores'][0] == 0.0 and out['edge_scores'][1] == 0.0
    assert T.contract(row, 0.3, 0.5, 0.85)['contracted'] == []  # below tau
    busy = dict(row, edge_scores=np.array([0.9, 0.9, 0.95, 0.9]))  # C also connects to node 3: keep C
    assert T.contract(busy, 0.3, 0.5, 0.6)['contracted'] == []
    empty = dict(nodes=np.zeros((1, 3)), node_scores=np.array([0.9]), pairs=np.zeros((0, 2), int), edge_scores=None, via=None)
    assert T.contract(empty, 0.3, 0.5, 0.6)['contracted'] == []  # patch with fewer than two detections


if __name__ == '__main__':
    test_two_hop_pairs()
    test_neighbourhood_offsets_land_on_gt()
    test_contract_chain()
    print('ALL TESTS PASSED')
