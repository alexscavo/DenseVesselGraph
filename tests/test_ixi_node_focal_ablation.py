import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from scripts.ixi_node_focal_ablation import ARMS, configure, order_for
from scripts.evaluate_ixi_node_focal import uniqueness
from training.losses.criterion import GraphCriterion
from models.matcher import build_matcher


class NodeFocalChecks(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('docs/research/evidence/ixi_mechanisms_20261019/D/focal_control/resolved-config.json').read_text())

    def test_only_node_classification_differs(self):
        configs = [configure(self.config, a) for a in ARMS]
        for c in configs:
            self.assertEqual(c['loss']['edge']['classification']['name'], 'cross_entropy')
            self.assertEqual(c['loss']['edge']['balancing']['positive_to_negative_ratio'], .15)
            self.assertFalse(c['loss']['edge']['candidates']['include_unmatched'])
            del c['loss']['node']['classification']
        self.assertEqual(configs[0], configs[1]); self.assertEqual(configs[1], configs[2])
        self.assertEqual(self.config['loss']['edge']['classification']['name'], 'focal')

    def test_actual_node_loss_matches_formulas_and_has_gradients(self):
        logits = torch.tensor([[[1., -1.], [-.5, .5], [.2, -.2]]], requires_grad=True)
        assignments = [(torch.tensor([1]), torch.tensor([0]))]
        target = torch.tensor([[0, 1, 0]])
        for arm in ARMS:
            cfg = configure(self.config, arm)
            criterion = GraphCriterion(cfg, build_matcher(cfg), torch.nn.Linear(1, 2))
            loss = criterion.loss_classification(logits, assignments)
            weights = torch.tensor(cfg['loss']['node']['classification']['class_weights'])
            if arm != 'focal_equal':
                expected = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(), weight=weights)
            else:
                ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='none')
                expected = (((1-ce.neg().exp())**2)*ce).mean()
            torch.testing.assert_close(loss, expected)
            gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all()); self.assertGreater(float(gradient.abs().sum()), 0)

    def test_uniqueness_and_rejected_legitimate_query(self):
        nodes = np.full((256, 3), .99); nodes[:3] = [[.2]*3,[.205]*3,[.7]*3]
        c = dict(nodes=nodes, retained=np.array([0,1]), node_scores=np.full(256,.1))
        c['node_scores'][:3] = [.9,.95,.4]
        gt = dict(nodes=np.array([[.2]*3,[.7]*3]), metric=np.eye(3)*8)
        result, samples = uniqueness(c, gt, np.ones((8,8,8), bool), 1.)
        self.assertEqual(result['raw_node_recall'], 1.)
        self.assertEqual(result['retained_node_recall'], .5)
        self.assertEqual(result['retained_duplicates'], 1)
        self.assertEqual(result['retained_high_confidence_duplicates'], 1)
        self.assertEqual(result['retained_duplicates_per_recovered_gt'], 1.)
        self.assertEqual(result['node_count_bias'], 0)

    def test_common_minibatch_prefix(self):
        np.testing.assert_array_equal(order_for(10,2048),order_for(20,2048)[:10])


if __name__ == '__main__': unittest.main()
