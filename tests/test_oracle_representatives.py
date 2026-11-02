import unittest
from unittest.mock import patch
import numpy as np
from scripts.oracle_representatives import single_swap,pair_interventions,support
from scripts.graph_error_counterfactuals import confidence_oracle
from scripts.node_verifier_metrics import association

class RepresentativeTests(unittest.TestCase):
    def test_confidence_oracle_same_coverage_different_choice(self):
        c=dict(physical=np.array([[0.,0,0],[.7,0,0],[4,0,0]]),truth=np.array([[0.,0,0],[4,0,0]]),scores=np.array([.1,.9,.8]))
        n=association(c['physical'],c['truth'],1.);_,h=confidence_oracle(c,1.)
        self.assertEqual(len(n),len(h));self.assertEqual(n,{0:0,2:1});self.assertEqual(h,{1:0,2:1})
    def test_single_swap_preserves_others_and_rejects_reuse(self):
        n={0:2,1:4};self.assertIsNone(single_swap(n,0,4));self.assertEqual(single_swap(n,0,5),{0:5,1:4});self.assertEqual(n,{0:2,1:4})
    def test_pair_interaction_and_collision(self):
        pairs=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
        c=dict(scores=np.ones(4),pairs=pairs,relations=np.array([.1,.2,.8,.7,.2,.9]),truth=np.zeros((2,3)))
        gt=dict(edges=np.array([[0,1]]))
        r=pair_interventions(c,gt,{0:0,1:1},{0:2,1:3},{},2.)[0]
        self.assertAlmostEqual(r['score_interaction'],.9-.7-.8+.1)
        self.assertEqual(r['decision_interaction'],-1)
        r=pair_interventions(c,gt,{0:0,1:1},{0:1,1:0},{},2.)[0]
        self.assertTrue(r['query_collision']);self.assertIsNone(r['decision_interaction']);self.assertEqual(r['nearest_score'],r['both_score'])
    def test_support_inverts_signed_metric(self):
        mask=np.zeros((4,4,4),np.float32);mask[1,2,3]=1;metric=np.diag([-8.,-4.,12.])
        c=dict(physical=np.array([[.25,.5,.75]])@metric.T)
        with patch('scripts.oracle_representatives.foreground',return_value=(mask,np.zeros_like(mask))):
            r=support(c,dict(metric=metric),'unused');self.assertEqual(r['foreground'][0],1.);self.assertEqual(r['soft_support'][0],1.)
if __name__=='__main__':unittest.main()
