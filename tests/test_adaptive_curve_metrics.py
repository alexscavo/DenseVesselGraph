"""Variable-K controls must retain the same common-v2 geometry definitions."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels

class AdaptiveCurveMetricsTest(unittest.TestCase):
    def test_ragged_controls_count_and_same_graph_metrics(self):
        nodes=np.array([[0.,0.,0.],[5.,0.,0.],[10.,0.,0.]])
        edges=np.array([[0,1],[1,2]])
        controls=[np.linspace(nodes[a],nodes[b],k) for (a,b),k in zip(edges,[4,8])]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'gt.npz'
            np.savez(path,metric=np.eye(3),nodes=nodes,edges=edges,full_paths=nodes[edges].reshape(-1,3),path_offsets=[0,2,4])
            chord=evaluate_common_vessels(dict(nodes=nodes,edges=edges),path)
            result=evaluate_common_vessels(dict(nodes=nodes,edges=edges,spline_controls=controls),path)
            self.assertEqual(result['representation_internal_controls'],8)
            for key in ['common_curve_f1','common_branch_f1']:
                self.assertEqual(aggregate_common_vessels([result])[key],aggregate_common_vessels([chord])[key])
            for key in ['common_betti0_abs_error','common_betti1_abs_error']:
                self.assertEqual(result[key],chord[key])
            fixed=evaluate_common_vessels(dict(nodes=nodes,edges=edges,spline_controls=np.array([np.linspace(nodes[a],nodes[b],8) for a,b in edges])),path)
            self.assertEqual(fixed['representation_internal_controls'],12)

if __name__=='__main__':unittest.main()
