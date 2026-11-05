"""2D crop targets re-contracted like the 3D ones (5 Nov 2026).

native2d/targets.crop_graph(recontract=True) contracts every degree-2 node of a crop except boundary anchors, then
inserts 'breaking point' anchors only for parallel branches, self-loops and anchor-free rings (3D placement rules of
data/spline_targets.subdivide_loops). Every degree-2 node of a target is then an explicit, flagged breaking point.
"""
import io
import unittest

import numpy as np

from metrics.loop_aware import inserted_from_degree
from native2d.evaluation import positional
from native2d.targets import crop_graph, from_dense_graph, save_target
from training.evaluation.metrics import graph_betti_numbers


def length(graph):
    return sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in graph.paths)


def degree(graph):
    return np.bincount(graph.edges.ravel(), minlength=len(graph.nodes))


def ring(points):
    return [(i, (i + 1) % len(points)) for i in range(len(points))]


def arc_point(path, fraction):
    """Point at an arc-length fraction of a polyline."""
    d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return np.array([np.interp(fraction * d[-1], d, path[:, a]) for a in range(2)])


class RecontractionTests(unittest.TestCase):
    def crops(self, nodes, edges, start, shape):
        graph = from_dense_graph(np.asarray(nodes, float), np.asarray(edges, int), provenance={})
        return crop_graph(graph, start, shape, recontract=False), crop_graph(graph, start, shape)

    def check(self, old, new):
        """Target invariants: simple graph, flags consistent, geometry/length and Betti numbers unchanged."""
        d = degree(new)
        pairs = [tuple(sorted(e)) for e in new.edges.tolist()]
        self.assertTrue(all(a != b for a, b in pairs), 'self-loop left')
        self.assertEqual(len(set(pairs)), len(pairs), 'parallel edge left')
        self.assertEqual(new.inserted.tolist(), (d == 2).tolist(), 'degree-2 nodes must be exactly the inserted ones')
        on_cycle = inserted_from_degree(len(new.nodes), new.edges)
        self.assertTrue(np.all(on_cycle[new.inserted]), 'inserted anchor off a cycle')
        self.assertFalse(np.any(new.inserted & new.boundary))
        for (a, b), p in zip(new.edges, new.paths):
            np.testing.assert_allclose(p[[0, -1]], new.nodes[[a, b]], atol=1e-7)
        self.assertAlmostEqual(length(new), length(old), places=9)
        self.assertEqual(graph_betti_numbers(len(new.nodes), new.edges), graph_betti_numbers(len(old.nodes), old.edges))
        points = np.concatenate(new.paths) if new.paths else np.empty((0, 2))
        for p in np.concatenate(old.paths) if old.paths else []:  # every legacy path point is still on a path
            self.assertLess(np.linalg.norm(points - p, axis=1).min(), 1e-7)

    def test_junction_whose_branch_is_cropped_away_is_contracted(self):
        # J=(9,5) lies on the right crop edge (x=9); its third branch leaves the crop at once.
        nodes = [[2, 2], [5, 3.5], [9, 5], [5, 6.5], [2, 8], [15, 5]]
        old, new = self.crops(nodes, [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)], [0, 0], [10, 10])
        self.assertEqual(int((degree(old) == 2).sum()), 1)  # the legacy target keeps J as a degree-2 node
        self.check(old, new)
        self.assertEqual((len(new.nodes), len(new.edges)), (2, 1))
        self.assertFalse(np.any(degree(new) == 2))
        self.assertFalse(new.inserted.any())

    def test_crop_that_opens_a_loop_inserts_nothing(self):
        # An anchor-free ring crossing x=9: the crop keeps an open arc; the legacy ring anchors became degree 2.
        square = [[2, 2], [8, 2], [14, 2], [14, 8], [8, 8], [2, 8], [2, 5]]
        old, new = self.crops(square, ring(square), [0, 0], [10, 10])
        self.assertGreater(int((degree(old) == 2).sum()), 0)
        self.check(old, new)
        self.assertEqual((len(new.nodes), len(new.edges)), (2, 1))
        self.assertEqual(new.boundary.tolist(), [True, True])
        self.assertFalse(new.inserted.any())

    def test_full_parallel_pair_gets_one_arc_length_midpoint(self):
        # A=(2,5), B=(8,5) with tails leaving the crop; asymmetric upper and lower paths.
        nodes = [[2, 5], [8, 5], [3, 3], [7.5, 2], [5, 8], [-3, 5], [14, 5]]
        edges = [(0, 2), (2, 3), (3, 1), (0, 4), (4, 1), (0, 5), (1, 6)]
        old, new = self.crops(nodes, edges, [0, 0], [10, 10])
        self.check(old, new)
        self.assertEqual(int(new.inserted.sum()), 1)
        self.assertEqual(len(new.nodes), 5)  # A, B, two boundary ends, one breaking point
        mid = new.nodes[new.inserted][0]
        upper, lower = np.array([[2, 5], [3, 3], [7.5, 2], [8, 5]]), np.array([[2, 5], [5, 8], [8, 5]])
        self.assertTrue(any(np.allclose(mid, arc_point(p, .5)) or np.allclose(mid, arc_point(p[::-1], .5)) for p in (upper, lower)))

    def test_self_loop_gets_anchors_at_one_and_two_thirds(self):
        loop = [[5, 5], [8, 4], [8, 8], [5, 8.5]]
        nodes = loop + [[-3, 5], [5, 15]]  # J=(5,5): loop + two tails leaving the crop
        edges = ring(loop) + [(0, 4), (0, 5)]
        old, new = self.crops(nodes, edges, [0, 0], [10, 10])
        self.check(old, new)
        self.assertEqual(int(new.inserted.sum()), 2)
        closed = np.array(loop + [loop[0]], float)
        got = sorted(map(tuple, np.round(new.nodes[new.inserted], 9)))
        forward = sorted(tuple(np.round(arc_point(closed, f), 9)) for f in (1 / 3, 2 / 3))
        backward = sorted(tuple(np.round(arc_point(closed[::-1], f), 9)) for f in (1 / 3, 2 / 3))
        self.assertIn(got, (forward, backward))

    def test_ring_inside_the_crop_gets_three_anchors_on_the_curve(self):
        square = [[2, 2], [5, 2], [7, 2], [7, 7], [2, 7]]
        old, new = self.crops(square, ring(square), [0, 0], [10, 10])
        self.check(old, new)
        self.assertEqual((len(new.nodes), int(new.inserted.sum())), (3, 3))
        for x, y in new.nodes:  # on the square's perimeter, never a centroid
            self.assertTrue(np.isclose(x, 2) or np.isclose(x, 7) or np.isclose(y, 2) or np.isclose(y, 7))

    def test_crop_that_turns_a_parallel_pair_into_a_ring(self):
        # A and B lie on the crop edges and their tails leave the crop: inside, only an anchor-free ring remains.
        nodes = [[0, 5], [9, 5], [3, 2], [6, 2], [4, 8], [-5, 5], [15, 5]]
        edges = [(0, 2), (2, 3), (3, 1), (0, 4), (4, 1), (0, 5), (1, 6)]
        old, new = self.crops(nodes, edges, [0, 0], [10, 10])
        self.check(old, new)
        self.assertEqual((len(new.nodes), int(new.inserted.sum())), (3, 3))
        self.assertLessEqual(len(new.nodes), len(old.nodes))

    def test_loop_free_uncut_graph_is_identical_to_legacy(self):
        nodes = [[1, 1], [4, 4], [8, 2], [4, 8], [6, 6], [-3, 4]]
        edges = [(0, 1), (1, 2), (1, 4), (4, 3), (1, 5)]
        old, new = self.crops(nodes, edges, [0, 0], [10, 10])
        np.testing.assert_array_equal(old.nodes, new.nodes)
        np.testing.assert_array_equal(old.edges, new.edges)
        for a, b in zip(old.paths, new.paths):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(new.inserted.any())
        self.assertEqual(int(new.boundary.sum()), 1)
        legacy, fixed = io.BytesIO(), io.BytesIO()
        save_target(legacy, old, [10, 10]);save_target(fixed, new, [10, 10])
        legacy.seek(0);fixed.seek(0)
        with np.load(legacy) as a, np.load(fixed) as b:
            self.assertEqual(set(b.files) - set(a.files), {'inserted', 'boundary'})
            self.assertNotIn('inserted', a.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(b['boundary'].tolist(), new.boundary.tolist())

    def test_isolated_nodes_are_kept(self):
        nodes = [[2, 2], [5, 3.5], [9, 5], [5, 6.5], [2, 8], [15, 5], [4, 4]]
        graph = from_dense_graph(np.asarray(nodes, float), np.array([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)]), provenance={})
        new = crop_graph(graph, [0, 0], [10, 10])
        self.assertEqual(len(new.nodes), 3)
        self.assertTrue(any(np.allclose(p, [4, 4]) for p in new.nodes))


class ExplicitInsertedFlagTests(unittest.TestCase):
    GT = (np.array([[0, 0], [4, 0], [2, 1], [-1, 0], [5, 0]], float), [(0, 1), (0, 2), (2, 1), (0, 3), (1, 4)])

    def test_explicit_mask_overrides_the_degree_rule(self):
        nodes, edges = self.GT
        pred, pred_edges = nodes[[0, 1, 3, 4]], [(0, 1), (0, 2), (1, 3)]
        flagged = positional(pred, pred_edges, nodes, edges, .5, inserted=[False, False, True, False, False])
        self.assertEqual(flagged, positional(pred, pred_edges, nodes, edges, .5))  # same as the degree fallback here
        plain = positional(pred, pred_edges, nodes, edges, .5, inserted=np.zeros(5, bool))
        self.assertEqual(plain, positional(pred, pred_edges, nodes, edges, .5, loop_aware=False))
        with self.assertRaises(ValueError):
            positional(pred, pred_edges, nodes, edges, .5, inserted=[False])

    def test_evaluate_reads_the_target_mask(self):
        from native2d.evaluation import evaluate
        nodes = [[2, 5], [8, 5], [3, 3], [7.5, 2], [5, 8], [-3, 5], [14, 5]]
        edges = [(0, 2), (2, 3), (3, 1), (0, 4), (4, 1), (0, 5), (1, 6)]
        target = crop_graph(from_dense_graph(np.asarray(nodes, float), np.asarray(edges), provenance={}), [0, 0], [10, 10])
        stream = io.BytesIO();save_target(stream, target, [10, 10]);stream.seek(0)
        with np.load(stream) as z:
            gt = dict(z)
        real = ~gt['inserted']
        remap = -np.ones(len(real), int);remap[real] = np.arange(real.sum())
        # Prediction: the real anchors and the GT edges between them; the breaking point is not predicted.
        prediction = dict(nodes=gt['nodes'][real], edges=np.array([(int(remap[u]), int(remap[v])) for u, v in gt['edges']
                                                                   if real[u] and real[v]]))
        row = evaluate(prediction, gt)
        # The unpredicted breaking point is 'don't care': no node miss, and the mask (not degree) selected it.
        self.assertIn('ignored_nodes', row['2.0'])
        self.assertEqual((row['2.0']['node']['tp'], row['2.0']['node']['fn']), (int(real.sum()), 0))


if __name__ == '__main__':
    unittest.main()
