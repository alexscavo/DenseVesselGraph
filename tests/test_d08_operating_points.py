"""Focused controls for cache-only D08 calibration and strict threshold ties."""
import unittest
import numpy as np
from scripts.d08_operating_points import match_distances,node_row,edge_labels,edge_row,edge_thresholds,select_node


class D08CalibrationTests(unittest.TestCase):
    def test_positional_match_preserves_maximum_cardinality(self):
        d=np.array([[.9,.9],[0.,1.8]])
        self.assertEqual(len(match_distances(d,1.)),2)

    def test_native_float32_threshold_keeps_whole_tie_group(self):
        scores=np.array([.8,.8,.7,.1],np.float32);y=np.array([1,1,0,0],bool)
        t=edge_thresholds(scores,y,2,1.)
        for policy in ['train_max_f1','train_preserve_precision']:
            np.testing.assert_array_equal(scores>t[policy],[True,True,False,False])

    def test_empty_or_no_positive_population_rejects_all(self):
        t=edge_thresholds(np.array([.6,.8],np.float32),np.array([0,0],bool),7,.1)
        self.assertEqual(t['train_max_f1'],1.)
        self.assertFalse(t['precision_target_attainable'])

    def test_precision_policy_uses_training_constraint(self):
        score=np.array([.9,.8,.7,.6],np.float32);y=np.array([1,0,1,0],bool)
        t=edge_thresholds(score,y,8,.7)
        self.assertEqual(int((score>t['train_preserve_precision']).sum()),1)

    def test_node_threshold_and_duplicate_counts(self):
        gt=dict(nodes=np.array([[0.,0.,0.],[10.,0.,0.]]),edges=np.array([[0,1]]),metric=np.eye(3))
        c=dict(raw_nodes=np.array([[0.,0.,0.],[.5,0.,0.],[10.,0.,0.]]),raw_scores=np.array([.8,.9,.2],np.float32))
        high=node_row(c,gt,.5,1.);low=node_row(c,gt,.1,1.)
        self.assertEqual(high['node_tp'],1);self.assertEqual(high['duplicates'],1)
        self.assertEqual(low['node_tp'],2);self.assertEqual(low['endpoint_recovered'],1)

    def test_node_threshold_conservative_tie_break(self):
        self.assertEqual(select_node([dict(threshold=.2,metrics=dict(node_f1=.5)),dict(threshold=.4,metrics=dict(node_f1=.5))]),.4)

    def test_relation_population_does_not_change_with_edge_threshold(self):
        gt=dict(nodes=np.array([[0.,0.,0.],[10.,0.,0.],[20.,0.,0.]]),edges=np.array([[0,1]]),metric=np.eye(3))
        c=dict(nodes=gt['nodes'].copy(),candidate_edges=np.array([[0,1],[0,2],[1,2]]),candidate_scores=np.array([.8,.6,.2],np.float32))
        _,labels,conditional=edge_labels(c,gt,1.)
        self.assertEqual(len(labels),3);self.assertEqual(int(labels.sum()),1);self.assertTrue(conditional.all())
        high=edge_row(c,gt,1.,.7);low=edge_row(c,gt,1.,.5)
        self.assertEqual(high['tp'],low['tp']);self.assertEqual(high['fp']+1,low['fp'])
        self.assertEqual(high['nodes'],low['nodes'])


if __name__=='__main__':unittest.main()
