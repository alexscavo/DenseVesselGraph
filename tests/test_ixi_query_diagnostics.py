"""Small crafted checks for P0 identity, matching and deletion accounting."""
import unittest
import numpy as np

from scripts.ixi_query_diagnostics import betti, categorize, edge_accounting, nms


class QueryDiagnosticTests(unittest.TestCase):
    def test_duplicate_reservation_and_oracle(self):
        nodes = np.array([[.1, 0, 0], [.11, 0, 0], [.8, 0, 0]])
        gt = nodes[[0, 2]]
        mask = np.ones((10, 10, 10), dtype=bool)
        mapping, categories, _, _ = categorize(nodes, gt, np.eye(3)*10, mask, np.arange(3), 1.)
        self.assertEqual(mapping, {0: 0, 2: 1})
        self.assertEqual(categories[1], 'duplicate_candidate')
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        args = (pairs, np.ones(3), np.ones(3, bool), mapping, np.array([[0, 1]]), 2)
        baseline = edge_accounting(np.arange(3), *args)[0]
        oracle = edge_accounting(np.array([0, 2]), *args)[0]
        self.assertEqual((baseline['tp'], baseline['MU_fp'], baseline['beta1']), (1, 2, 1))
        self.assertEqual((oracle['tp'], oracle['fp'], oracle['beta1']), (1, 0, 0))
        self.assertEqual(oracle['fn'], baseline['fn'])

    def test_physical_nms_ties_and_no_gt(self):
        points = np.array([[0., 0, 0], [.4, 0, 0], [1.1, 0, 0]])
        np.testing.assert_array_equal(nms(points, np.ones(3), np.array([2, 1, 0]), .5), [0, 2])
        np.testing.assert_array_equal(nms(points, np.array([.7, .9, .6]), np.arange(3), .5), [1, 2])

    def test_empty_and_boundary(self):
        self.assertEqual(betti(0, []), (0, 0))
        self.assertEqual(betti(3, []), (3, 0))
        nodes = np.array([[-.05, 0, 0], [-.06, 0, 0]])
        mapping, categories, _, inside = categorize(nodes, np.empty((0, 3)), np.eye(3)*10,
            np.ones((10, 10, 10), bool), np.arange(2), 1.)
        self.assertEqual(mapping, {})
        np.testing.assert_array_equal(inside, [True, False])
        self.assertEqual(categories[0], 'foreground_supported_extra')
        self.assertEqual(categories[1], 'off_vessel_or_outside')
        row = edge_accounting(np.empty(0, int), np.empty((0, 2), int), np.empty(0),
            np.empty(0, bool), {}, np.empty((0, 2), int), 0)[0]
        self.assertEqual((row['tp'], row['fp'], row['fn']), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
