import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from models.node_refinement import LocalNodeRefinement
from scripts.brava_graph_study import configuration,orders
from scripts.evaluate_brava_study import patch_metrics
from data.spline_targets import fit_spline


class BravaStudyChecks(unittest.TestCase):
    def test_configuration_only_changes_refinement(self):
        plan=json.loads(Path('experiments/brava_graph_study/plan.json').read_text())
        configs=[configuration(plan,arm) for arm in ['standard','coordinate_refinement']]
        for cfg in configs:del cfg['model']['node_refinement']['enabled'];del cfg['experiment']['name']
        self.assertEqual(configs[0],configs[1])
        self.assertEqual(configs[0]['training']['input'],'image')
        self.assertFalse(configs[0]['model']['node_refinement']['refine_classification'])

    def test_coordinate_only_and_identity_initialization(self):
        head=LocalNodeRefinement(4,refine_classification=False)
        tokens=torch.zeros(1,2,4);nodes=torch.full((1,2,6),.5);logits=torch.randn(1,2,2);features=torch.randn(1,64,8,8,8)
        refined,classes=head(tokens,nodes,logits,features,(16,16,16))
        torch.testing.assert_close(refined,nodes);torch.testing.assert_close(classes,logits)
        with torch.no_grad():head.network[-1].bias.fill_(1.)
        refined,classes=head(tokens,nodes,logits,features,(16,16,16))
        self.assertFalse(torch.equal(refined,nodes));torch.testing.assert_close(classes,logits)

    def test_same_graph_geometry_uses_manual_paths_without_masks(self):
        nodes=np.array([[.1,.1,.2],[.9,.1,.2]],np.float32);edges=np.array([[0,1]])
        guide=np.array([[.1,.1,.2],[.3,.5,.2],[.7,.5,.2],[.9,.1,.2]])
        controls,curves=fit_spline(guide,8,64)
        gt=dict(nodes=nodes,edges=edges,metric=np.eye(3)*10,full_paths=guide,path_offsets=np.array([0,len(guide)]))
        c=dict(nodes=nodes,edges=edges,candidate_edges=edges,candidate_scores=np.array([.9]),node_scores=np.array([.9,.9]),
            raw_nodes=nodes,raw_scores=np.array([.9,.9]),spline_controls=controls[None])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'gt.npz';np.savez(path,**gt);rows,geometry=patch_metrics(c,gt,path)
        self.assertEqual(len(rows),4)
        for tol in [1,2]:
            a,b=[r for r in rows if r['tolerance_mm']==tol]
            self.assertEqual(a['tp'],b['tp']);self.assertEqual(a['common_branch_tp'],b['common_branch_tp'])
        self.assertTrue(all(r['spline_error_mm']<r['chord_error_mm'] for r in geometry))

    def test_order_covers_all_training_records_and_common_prefix(self):
        np.testing.assert_array_equal(orders(3,16),orders(6,16)[:3])
        self.assertEqual(set(orders(2,16).ravel()),set(range(16)))


if __name__=='__main__':unittest.main()
