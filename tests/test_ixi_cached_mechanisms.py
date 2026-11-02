"""Focused checks: raw-query insertion, ablation semantics, additive loss scale."""
import unittest
import numpy as np
import torch
from scripts.ixi_cached_mechanisms import decode, variant, decisions
from scripts.ixi_query_diagnostics import categorize, edge_accounting
from training.losses.criterion import _ratio_upsample


class MechanismTests(unittest.TestCase):
    def cache(self):
        nodes = np.full((256, 3), .9, dtype=np.float32)
        nodes[0] = [.1, .1, .1]; nodes[1] = [.4, .1, .1]
        logits = np.tile([2., -2.], (256, 1)).astype(np.float32); logits[0] = [-2, 2]
        pairs = torch.combinations(torch.arange(256)).numpy()
        pair_logits = np.tile([2., -2.], (len(pairs), 1)).astype(np.float32)
        pair_logits[0] = [-2, 2]
        return dict(nodes=nodes, boxes=np.c_[nodes, np.ones_like(nodes)*.1], node_logits=logits,
                    original_nodes=np.c_[nodes+.01, np.ones_like(nodes)*.1], original_logits=-logits,
                    node_scores=torch.tensor(logits).softmax(-1)[:, 1].numpy(),
                    tokens=np.arange(258*3, dtype=np.float32).reshape(258, 3), edges=pairs,
                    pair_logits=pair_logits, edge_scores=torch.tensor(pair_logits).softmax(-1)[:, 1].numpy(),
                    edge_selected=pair_logits.argmax(-1) == 1, retained=np.array([0]), metric=np.eye(3, dtype=np.float32)*10)

    def test_raw_oracle_includes_rejected_node_and_existing_relation(self):
        c = self.cache(); gt = c['nodes'][:2]
        mapping = categorize(c['nodes'], gt, c['metric'], np.ones((10, 10, 10), bool), np.arange(256), .01)[0]
        normal = decode(c, None); oracle = decode(c, None, np.array(sorted(mapping)))
        self.assertEqual(len(normal['nodes']), 1); self.assertEqual(len(normal['edges']), 0)
        np.testing.assert_array_equal(oracle['query_ids'], [0, 1]); self.assertEqual(len(oracle['edges']), 1)
        np.testing.assert_array_equal(oracle['nodes'], c['nodes'][:2])
        # A negative relation remains absent even when GT says adjacent.
        c['pair_logits'][0] = [2., -2.]
        self.assertEqual(len(decode(c, None, np.array([0, 1]))['edges']), 0)

    def test_component_selection_and_ties(self):
        c = self.cache()
        coordinates = variant(c, True, False); classification = variant(c, False, True)
        np.testing.assert_array_equal(coordinates['nodes'], c['nodes'])
        np.testing.assert_array_equal(coordinates['node_logits'], c['original_logits'])
        np.testing.assert_array_equal(classification['nodes'], c['original_nodes'][:, :3])
        np.testing.assert_array_equal(classification['node_logits'], c['node_logits'])
        np.testing.assert_array_equal(decisions(np.array([[0., 0.], [-1, 1]])), [False, True])

    def test_independent_auxiliary_does_not_rescale_baseline_gradient(self):
        x = torch.tensor([[0., 1.], [1., 0.], [1., 0.]], requires_grad=True)
        labels = torch.tensor([1, 0, 0])
        logits, target = _ratio_upsample(x, labels, .15, .02)
        base = torch.nn.functional.cross_entropy(logits, target)
        aux_logits = torch.tensor([[1., 0.]], requires_grad=True)
        auxiliary = torch.nn.functional.cross_entropy(aux_logits, torch.tensor([0]))
        before = torch.autograd.grad(base, x, retain_graph=True)[0]
        after = torch.autograd.grad(base+.1*auxiliary, x)[0]
        torch.testing.assert_close(before, after)


if __name__ == '__main__':
    unittest.main()
