import tempfile
import unittest
from pathlib import Path
import numpy as np
from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels
from data.adaptive_split_targets import split_target

class TargetAndMetricTests(unittest.TestCase):
    def test_polyline_adapter_preserves_graph_and_validates_order(self):
        nodes=np.array([[0.,0.,0.],[5.,0.,0.],[10.,0.,0.]]);edges=np.array([[0,1],[1,2]])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'gt.npz';np.savez(path,metric=np.eye(3),nodes=nodes,edges=edges,full_paths=nodes[edges].reshape(-1,3),path_offsets=[0,2,4])
            a=evaluate_common_vessels(dict(nodes=nodes,edges=edges),path)
            curves=np.array([np.linspace(nodes[left],nodes[right],257) for left,right in edges])
            b=evaluate_common_vessels(dict(nodes=nodes,edges=edges,edge_polylines=curves),path)
            self.assertEqual(aggregate_common_vessels([a])['common_curve_f1'],aggregate_common_vessels([b])['common_curve_f1'])
            self.assertEqual(a['common_branch_tp'],b['common_branch_tp'])
            with self.assertRaises(ValueError):evaluate_common_vessels(dict(nodes=nodes,edges=edges,edge_polylines=curves[:,::-1]),path)
    def test_snapshot_orientation_dependency(self):
        import torch
        from scripts.multidataset_graph_evaluation import snapshot
        orientation=snapshot.__wrapped__.__globals__['canonical_reverse']
        points=torch.tensor([[[.8,.2,.1],[.1,.2,.1]],[[.1,.2,.1],[.8,.2,.1]]])
        self.assertEqual(orientation(points).tolist(),[True,False])

    def test_explicit_split_keeps_anchors_and_branch_loss_mass(self):
        t=np.linspace(0,1,121);path=np.c_[t,.5+.1*np.sin(6*np.pi*t),.5+.05*np.sin(4*np.pi*t)]
        target=dict(nodes=path[[0,-1]],edges=np.array([[0,1]]),full_paths=path,path_offsets=np.array([0,len(path)]),metric=np.eye(3)*40,boundary=np.array([True,True]))
        split=split_target(target,.5)
        self.assertGreater(len(split['edges']),1)
        np.testing.assert_allclose(split['nodes'][:2],target['nodes']);self.assertTrue(split['boundary'][:2].all());self.assertFalse(split['boundary'][2:].any())
        self.assertEqual(len(split['edges'])-len(split['nodes'])+1,0)
        self.assertAlmostEqual(float(split['branch_weights'].sum()),1.,places=6)
        np.testing.assert_allclose(split['controls'][:,[0,-1]],split['nodes'][split['edges']],atol=1e-6)
        np.testing.assert_array_equal(split['parent_branch'],np.zeros(len(split['edges']),int))

if __name__=='__main__':unittest.main()
