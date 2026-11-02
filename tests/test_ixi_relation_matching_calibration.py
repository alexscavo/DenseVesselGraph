"""Focused checks for actual assignment, physical validity, tied cutoffs and AP."""
import unittest
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from models.matcher import HungarianMatcher
from scripts.ixi_relation_matching_calibration import pair_labels, select_threshold, scores_from_logits, weighted_ap


class MatchingCalibrationTests(unittest.TestCase):
    def test_hungarian_can_choose_far_high_confidence_query(self):
        nodes = torch.tensor([[[0., 0, 0], [.2, 0, 0]]])
        logits = torch.tensor([[[-5., 5.], [5., -5.]]])
        source, target = HungarianMatcher(2., 5.)({'pred_nodes': nodes, 'pred_logits': logits}, {'nodes': [torch.tensor([[.2, 0, 0]])]})[0]
        self.assertEqual(source.tolist(), [0])
        distance_mm = torch.linalg.vector_norm((nodes[0, source]-torch.tensor([[.2, 0, 0]]))*10, dim=-1)
        self.assertFalse(bool((distance_mm <= 1.).any()))
        # Matching consumes the supplied corrected output, not a hidden preliminary coordinate.
        refined = nodes.clone(); refined[0, 0] = torch.tensor([.2, 0, 0])
        source2, _ = HungarianMatcher(2., 5.)({'pred_nodes': refined, 'pred_logits': logits}, {'nodes': [torch.tensor([[.2, 0, 0]])]})[0]
        self.assertEqual(source2.tolist(), [0])
        self.assertEqual(float(torch.linalg.vector_norm(refined[0, source2]-torch.tensor([[.2, 0, 0]]))), 0.)

    def test_ties_and_unattainable_precision(self):
        scores = np.array([.8, .8, .7, .1]); y = np.array([1, 0, 1, 0], bool)
        result = select_threshold(scores, y, 3, .5)
        np.testing.assert_array_equal(scores > result['train_max_f1'], [True, True, True, False])
        np.testing.assert_array_equal(scores > result['train_preserve_precision'], [True, True, True, False])
        result = select_threshold(scores, y, 3, .75)
        self.assertFalse(result['precision_target_attainable'])
        self.assertFalse((scores > result['train_preserve_precision']).any())

    def test_candidate_labels_and_unlocalized_assignment(self):
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        mapping = np.array([0, 1, -1]); gt = np.array([[0, 1]])
        active, conditional, labels = pair_labels(pairs, mapping, gt, np.array([0, 1, 2]))
        np.testing.assert_array_equal(active, [1, 1, 1])
        np.testing.assert_array_equal(conditional, [1, 0, 0])
        np.testing.assert_array_equal(labels, [1, 0, 0])
        _, _, labels = pair_labels(pairs, np.array([0, -1, -1]), gt, np.arange(3))
        self.assertFalse(labels.any())

    def test_score_parity_and_weighted_subject_replication(self):
        logits = np.array([[0., 0.], [-1., 1.], [1., -1.]])
        np.testing.assert_array_equal(scores_from_logits(logits) > .5, logits.argmax(-1) == 1)
        y = np.array([1, 0, 1, 0], bool); scores = np.array([.8, .8, .7, .1]); weights = np.array([2, 1, 0, 3])
        expected = average_precision_score(np.repeat(y, weights), np.repeat(scores, weights))
        self.assertAlmostEqual(weighted_ap(y, scores, weights), expected, places=12)
        self.assertTrue(np.isnan(weighted_ap(np.zeros(4, bool), scores, weights)))


if __name__ == '__main__':
    unittest.main()
