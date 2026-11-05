"""Breaking-point subdivision of loop branches in the 3D spline targets (5 Nov 2026).

subdivide=True must keep every patch, give a simple graph (one edge per node pair, no self-loop), keep the
centreline geometry (total length), beta_0 and beta_1, and flag exactly the inserted anchors.
"""
import unittest

import numpy as np

from data.spline_targets import contract_spline_graph, crop_dense_paths


def betti(anchor_ids, branches):
    index = {k: i for i, k in enumerate(anchor_ids)}
    parent = list(range(len(index)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for b in branches:
        parent[root(index[b.start])] = root(index[b.end])
    b0 = len({root(i) for i in range(len(index))})
    return b0, len(branches) - len(index) + b0


def length(branches):
    return sum(np.linalg.norm(np.diff(b.polyline, axis=0), axis=1).sum() for b in branches)


def dense_length(nodes, edges):
    edges = np.asarray(edges).reshape(-1, 2)
    return np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1).sum()


def chain(points, start_id):
    """Edges of a path through consecutive new node ids."""
    return [(start_id + i, start_id + i + 1) for i in range(len(points) - 1)]


class LoopSubdivisionTests(unittest.TestCase):
    def check_simple(self, graph, nodes, edges, b0, b1):
        pairs = [tuple(sorted((b.start, b.end))) for b in graph.branches]
        self.assertTrue(all(a != b for a, b in pairs), 'self-loop left')
        self.assertEqual(len(set(pairs)), len(pairs), 'parallel branch left')
        self.assertTrue(all(k >= 0 for k in graph.anchors), 'virtual anchor left')
        self.assertEqual(betti(sorted(graph.anchors), graph.branches), (b0, b1))
        self.assertAlmostEqual(length(graph.branches), dense_length(nodes, edges), places=9)
        for b in graph.branches:  # every branch ends on its anchors
            np.testing.assert_allclose(b.polyline[[0, -1]], [graph.anchors[b.start], graph.anchors[b.end]], atol=1e-9)

    def test_parallel_branches_get_one_midpoint_anchor(self):
        # A=(0,0,0) and B=(4,0,0) joined by an upper and a lower path; each has a tail, so both are junctions.
        nodes = np.array([[0, 0, 0], [4, 0, 0], [1, 1, 0], [2, 1.2, 0], [3, 1, 0], [1, -1, 0], [2, -1.2, 0], [3, -1, 0],
                          [-1, 0, 0], [5, 0, 0]], float)
        edges = [(0, 2), (2, 3), (3, 4), (4, 1), (0, 5), (5, 6), (6, 7), (7, 1), (0, 8), (1, 9)]
        old, reason = contract_spline_graph(nodes, edges)
        self.assertEqual(reason, 'parallel_branches')
        graph, reason = contract_spline_graph(nodes, edges, subdivide=True)
        self.assertIsNone(reason)
        self.assertEqual(len(graph.inserted), 1)
        self.assertEqual(len(graph.anchors), 5)
        self.check_simple(graph, nodes, edges, 1, 1)
        mid = graph.anchors[next(iter(graph.inserted))]
        self.assertAlmostEqual(abs(mid[1]), 1.2, places=6)  # the arc-length midpoint of a symmetric path

    def test_self_loop_on_junction_gets_two_anchors(self):
        loop = [[1, 0, 0], [2, 1, 0], [1, 2, 0], [0, 1, 0]]  # J=(1,0,0) -> ... -> back to J
        nodes = np.array(loop + [[1, -2, 0]], float)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)]
        nodes = np.vstack([nodes, [[3, 0, 0]]])  # second tail: J has degree 4, a junction
        edges.append((0, 5))
        graph, reason = contract_spline_graph(nodes, edges, subdivide=True)
        self.assertIsNone(reason)
        self.assertEqual(len(graph.inserted), 2)
        self.check_simple(graph, nodes, edges, 1, 1)

    def test_ring_without_anchors_gets_three_anchors_on_the_curve(self):
        square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        self.assertEqual(contract_spline_graph(square, edges)[1], 'closed_branch')
        graph, reason = contract_spline_graph(square, edges, subdivide=True)
        self.assertIsNone(reason)
        self.assertEqual(len(graph.inserted), 3)
        self.assertEqual(len(graph.anchors), 3)
        self.check_simple(graph, square, edges, 1, 1)
        for k in graph.anchors:  # on the square's perimeter, not the centroid
            p = graph.anchors[k]
            self.assertTrue(np.isclose(p[0], 0) or np.isclose(p[0], 1) or np.isclose(p[1], 0) or np.isclose(p[1], 1))

    def test_duplicate_dense_records_become_a_subdivided_parallel_pair(self):
        from scripts.audit_synthetic_mri_grid import SourceGraph, CenterlineEdge
        nodes = {0: np.array([1., 1., 1.]), 1: np.array([2., 1., 1.])}
        edge = CenterlineEdge(0, 1, (nodes[0], nodes[1]))
        source = SourceGraph(nodes, [(0, 1), (0, 1)], [edge, edge])
        dense = crop_dense_paths(source, np.array([[0., 3.]] * 3), boundary_policy='last_inside_sample')
        graph, reason = contract_spline_graph(dense.positions, dense.edges, subdivide=True)
        self.assertIsNone(reason)
        # Two points joined twice and nothing else is a ring without junctions: three anchors on the curve.
        self.assertEqual(len(graph.inserted), 3)
        self.check_simple(graph, dense.positions, dense.edges, 1, 1)
        # With a tail on each side both points are junctions: the repeat is a parallel branch, one midpoint anchor.
        nodes = np.vstack([dense.positions, [[0., 1., 1.], [3., 1., 1.]]])
        edges = np.vstack([dense.edges, [[0, len(nodes) - 2], [1, len(nodes) - 1]]])
        graph, reason = contract_spline_graph(nodes, edges, subdivide=True)
        self.assertIsNone(reason)
        self.assertEqual(len(graph.inserted), 1)
        self.check_simple(graph, nodes, edges, 1, 1)

    def test_two_segment_closed_branch_is_kept(self):
        from scripts.audit_synthetic_mri_grid import SourceGraph, CenterlineEdge
        nodes = {0: np.array([1., 1., 1.])}
        edge = CenterlineEdge(0, 0, (np.array([2., 1., 1.]),))
        source = SourceGraph(nodes, [(0, 0)], [edge])
        dense = crop_dense_paths(source, np.array([[0., 3.]] * 3), boundary_policy='last_inside_sample')
        graph, reason = contract_spline_graph(dense.positions, dense.edges, closed_components=dense.closed_components,
                                              subdivide=True)
        self.assertIsNone(reason)
        self.check_simple(graph, dense.positions, dense.edges, 1, 1)

    def test_loop_free_graph_is_unchanged(self):
        nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [3, 0, 0]], float)
        edges = [(0, 1), (1, 2), (2, 3), (2, 4)]
        old, r0 = contract_spline_graph(nodes, edges)
        new, r1 = contract_spline_graph(nodes, edges, subdivide=True)
        self.assertIsNone(r0)
        self.assertIsNone(r1)
        self.assertEqual(len(new.inserted), 0)
        self.assertEqual(sorted(old.anchors), sorted(new.anchors))
        self.assertEqual([(b.start, b.end) for b in old.branches], [(b.start, b.end) for b in new.branches])
        for a, b in zip(old.branches, new.branches):
            np.testing.assert_array_equal(a.polyline, b.polyline)


if __name__ == '__main__':
    unittest.main()


class LoopAwareAccountingTests(unittest.TestCase):
    """Evaluation of subdivided GT: inserted anchors are 'don't care'; edges are scored on the original branches."""

    def account(self, pred_nodes, pred_edges, gt_nodes, gt_edges, inserted):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from ixi_query_diagnostics import edge_accounting
        from diagnose_branch_relations import match_nodes
        pred_nodes, gt_nodes = np.asarray(pred_nodes, float), np.asarray(gt_nodes, float)
        pairs = np.asarray(pred_edges, int).reshape(-1, 2)
        mapping = match_nodes(pred_nodes, gt_nodes, 0.5) if len(pred_nodes) else {}
        row, *_ = edge_accounting(np.arange(len(pred_nodes)), pairs, np.ones(len(pairs)), np.ones(len(pairs), bool),
                                  mapping, np.asarray(gt_edges, int).reshape(-1, 2), len(gt_nodes), inserted)
        return row

    GT = dict(gt_nodes=[[0, 0, 0], [4, 0, 0], [2, 1, 0]], gt_edges=[(0, 1), (0, 2), (2, 1)], inserted=[False, False, True])

    def test_both_parallel_branches_recovered(self):
        row = self.account([[0, 0, 0], [4, 0, 0], [2, 1, 0]], [(0, 1), (0, 2), (2, 1)], **self.GT)
        self.assertEqual((row['tp'], row['fp'], row['fn']), (2, 0, 0))
        self.assertEqual((row['node_tp'], row['node_fp'], row['node_fn'], row['ignored_nodes']), (2, 0, 0, 1))

    def test_one_branch_found_is_one_miss_not_two(self):
        row = self.account([[0, 0, 0], [4, 0, 0]], [(0, 1)], **self.GT)
        self.assertEqual((row['tp'], row['fp'], row['fn']), (1, 0, 1))
        self.assertEqual((row['node_tp'], row['node_fp'], row['node_fn']), (2, 0, 0))

    def test_dangling_edge_to_inserted_anchor_is_false_but_node_ignored(self):
        row = self.account([[0, 0, 0], [4, 0, 0], [2, 1, 0]], [(0, 2)], **self.GT)
        self.assertEqual((row['tp'], row['fp'], row['fn']), (0, 1, 2))
        self.assertEqual((row['node_fp'], row['ignored_nodes']), (0, 1))

    def test_inserted_anchor_need_not_be_predicted(self):
        row = self.account([[0, 0, 0], [4, 0, 0]], [], **self.GT)
        self.assertEqual((row['node_tp'], row['node_fn'], row['fn']), (2, 0, 2))

    def test_self_loop_and_anchor_free_ring(self):
        # junction J=0 with a tail to 1 and a loop J-m1-m2-J (inserted 2, 3); a separate ring of inserted 4, 5, 6
        gt = dict(gt_nodes=[[0, 0, 0], [-2, 0, 0], [1, 1, 0], [-1, 1, 0], [5, 5, 0], [6, 5, 0], [5, 6, 0]],
                  gt_edges=[(0, 1), (0, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 4)],
                  inserted=[False, False, True, True, True, True, True])
        row = self.account([[0, 0, 0], [-2, 0, 0], [1, 1, 0], [-1, 1, 0]], [(0, 1), (0, 2), (2, 3), (3, 0)], **gt)
        self.assertEqual((row['tp'], row['fp'], row['fn'], row['dropped_gt_rings']), (2, 0, 0, 1))
        self.assertEqual((row['node_tp'], row['node_fn'], row['gt_nodes']), (2, 0, 2))

    def test_without_inserted_anchors_the_old_protocol_is_used(self):
        gt = dict(gt_nodes=[[0, 0, 0], [4, 0, 0]], gt_edges=[(0, 1)])
        old = self.account([[0, 0, 0], [4, 0, 0]], [(0, 1)], inserted=None, **gt)
        new = self.account([[0, 0, 0], [4, 0, 0]], [(0, 1)], inserted=[False, False], **gt)
        self.assertEqual(old, new)
        self.assertNotIn('ignored_nodes', new)


class NativeTwoDLoopAwareTests(unittest.TestCase):
    def test_2d_parallel_pair_scored_on_original_branches(self):
        from native2d.evaluation import positional
        gt_nodes, gt_edges = np.array([[0, 0], [4, 0], [2, 1], [-1, 0], [5, 0]], float), [(0, 1), (0, 2), (2, 1), (0, 3), (1, 4)]
        both = positional(gt_nodes, gt_edges, gt_nodes, gt_edges, 0.5)
        self.assertEqual((both['edge']['tp'], both['edge']['fp'], both['edge']['fn']), (4, 0, 0))
        self.assertEqual((both['node']['tp'], both['node']['fp'], both['node']['fn']), (4, 0, 0))
        half = positional(gt_nodes[[0, 1, 3, 4]], [(0, 1), (0, 2), (1, 3)], gt_nodes, gt_edges, 0.5)
        self.assertEqual((half['edge']['tp'], half['edge']['fp'], half['edge']['fn']), (3, 0, 1))

    def test_2d_loop_free_target_is_unchanged(self):
        from native2d.evaluation import positional
        gt_nodes, gt_edges = np.array([[0, 0], [4, 0], [2, 3], [2, 0.]]), [(0, 3), (3, 1), (3, 2)]
        pred, pred_edges = gt_nodes + 0.1, [(0, 3), (3, 1)]
        self.assertEqual(positional(pred, pred_edges, gt_nodes, gt_edges, 0.5),
                         positional(pred, pred_edges, gt_nodes, gt_edges, 0.5, loop_aware=False))


class InsertedFromDegreeTests(unittest.TestCase):
    def test_degree_two_off_a_cycle_is_an_ordinary_node(self):
        from metrics.loop_aware import inserted_from_degree
        # path 0-1-2: node 1 has degree 2 but is not on a cycle (e.g. a junction whose third branch was cropped)
        self.assertEqual(inserted_from_degree(3, [(0, 1), (1, 2)]).tolist(), [False, False, False])
        # parallel pair A=0,B=1 with breaking point 2, tails 3, 4: only node 2 is inserted
        self.assertEqual(inserted_from_degree(5, [(0, 1), (0, 2), (2, 1), (0, 3), (1, 4)]).tolist(),
                         [False, False, True, False, False])
