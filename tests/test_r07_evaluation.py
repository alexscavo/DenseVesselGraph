import tempfile,unittest
from pathlib import Path
import numpy as np
from scripts.collect_r07 import candidate_context,graph_metrics,aggregate,curve_metrics,paired_curve_ci,calibration


class R07EvaluationChecks(unittest.TestCase):
    def test_float32_calibrated_cutoff_includes_selected_tie_group(self):
        labels=np.array([1,0,1],bool)
        scores=np.array([.8,.8,.6],dtype=np.float32)
        cuts=calibration(labels,scores,gt_edges=2,target_precision=.5)
        for cutoff in cuts.values():
            np.testing.assert_array_equal(scores>cutoff,[True,True,True])

    def test_exact_graph_and_frozen_curve_reconstruction(self):
        nodes=np.array([[0.,0,0],[2.,0,0],[4.,0,0]])
        edges=np.array([[0,1],[1,2]]);pairs=np.array([[0,1],[0,2],[1,2]])
        gt=dict(edges=edges,links={(0,1),(1,2)},geometry=[dict(arc_mm=2.),dict(arc_mm=2.)])
        c=dict(physical=nodes,truth=nodes,gt_edges=edges,scores=np.full(3,.9),pairs=pairs,
               relations=np.array([.8,.2,.7]),metric=np.eye(3),curve_pair_indices=np.arange(3),
               hungarian_source=np.arange(3),hungarian_target=np.arange(3),
               curve_controls=np.array([np.linspace(nodes[a],nodes[b],8) for a,b in pairs]))
        c['context']={2.:candidate_context(c,gt,2.)}
        row,chosen=graph_metrics(c,gt,c['context'][2.],c['relations'],.5,2.)
        self.assertEqual(aggregate([row])['edge_f1'],1.)
        self.assertEqual(row['beta0_error'],0);self.assertEqual(row['beta1_error'],0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gt.npz'
            np.savez(path,nodes=nodes,edges=edges,metric=np.eye(3),full_paths=nodes[edges].reshape(-1,3),path_offsets=[0,2,4])
            c['record']=dict(gt=str(path));curve=curve_metrics(c,chosen)
        curve.update(sample='test',subject='subject')
        ci=paired_curve_ci([curve],[curve])
        self.assertEqual(ci['curve_f1']['delta'],0.)
        self.assertEqual(ci['branch_f1']['ci95'],[0.,0.])

if __name__=='__main__':unittest.main()
