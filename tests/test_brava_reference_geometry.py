"""Regression for coincident SWC parent/child points in full-volume references."""
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.audit_brava_study import reference_polylines
from metrics.common_vessel_metrics import evaluate_common_vessels

class ReferenceGeometryTests(unittest.TestCase):
    def test_coincident_parent_child_keeps_edge_and_metric(self):
        nodes=np.array([[1.,2.,3.],[1.,2.,3.]])
        edges=np.array([[0,1]])
        source=SimpleNamespace(edge_polylines=(nodes[:1].copy(),))
        curves=reference_polylines(source,nodes,edges)
        np.testing.assert_array_equal(curves[0],nodes)
        self.assertEqual(len(source.edge_polylines[0]),1)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'reference.npz'
            np.savez(p,nodes=nodes,edges=edges,metric=np.eye(3),full_paths=np.concatenate(curves),path_offsets=[0,2])
            result=evaluate_common_vessels(dict(nodes=nodes,edges=edges),p)
        self.assertEqual(result['common_branch_tp'],1)
        self.assertEqual(result['common_branch_fp'],0)
        self.assertEqual(result['common_branch_fn'],0)
        self.assertEqual(result['common_betti0_abs_error'],0)
        self.assertEqual(result['common_betti1_abs_error'],0)

if __name__=='__main__':unittest.main()
