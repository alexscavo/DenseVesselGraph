import unittest
import numpy as np
from scripts.analyze_graph_errors import evaluate_mapping,terminal_branches,match_hungarian,feasible_endpoints
class GraphAttribution(unittest.TestCase):
 def fixture(self):
  gt=np.array([[0.,0,0],[4.,0,0],[8.,0,0]])
  x=np.r_[gt,[[.1,0,0],[20.,0,0]]];pairs=np.array([(a,b) for a in range(5) for b in range(a+1,5)])
  chosen={(0,1),(1,2),(0,2),(1,3),(2,4),(3,4)};scores=np.array([.9 if tuple(p) in chosen else .1 for p in pairs])
  c=dict(physical=x,truth=gt,gt_edges=np.array([[0,1],[1,2]]),scores=np.ones(5)*.9,pairs=pairs,relations=scores)
  target=dict(nodes=gt,edges=c['gt_edges']);return c,target
 def test_false_edge_categories_and_cycle_counts(self):
  c,g=self.fixture();r,_=evaluate_mapping(c,g,1.,np.arange(5),{0:0,1:1,2:2})
  self.assertEqual((r['edge_tp'],r['edge_fp'],r['fp_mm'],r['fp_mu'],r['fp_uu']),(2,4,1,2,1))
  self.assertEqual(r['fp_duplicate'],2);self.assertEqual(r['pred_beta1'],2);self.assertEqual(r['nodes_signed_error'],2)
 def test_unsupported_terminal_component(self):
  c,g=self.fixture();term,branches,unsupported=terminal_branches(np.arange(5),np.array([[0,1],[1,2],[3,4]]),{0:0,1:1,2:2},{(0,1),(1,2)})
  self.assertEqual((term,branches,unsupported),(4,2,1))
 def test_hungarian_is_not_a_spatial_oracle(self):
  metric=np.eye(3)*64;nodes=np.array([[.1,0,0],[.9,0,0]])
  c=dict(physical=np.array([[.11,0,0],[.9,0,0],[.3,0,0]])@metric.T,truth=nodes@metric.T,scores=np.array([.01,.99,.99]))
  s,t,_=match_hungarian(c,dict(metric=metric,nodes=nodes),dict(model=dict(matcher=dict(type='hungarian',node_cost=5.,class_cost=2.))))
  self.assertEqual(dict(zip(t,s))[0],2)
 def test_distinct_raw_endpoints_required(self):
  e=np.array([[0,1]])
  self.assertFalse(feasible_endpoints(np.array([[.2,.3]]),e,1)[0])
  self.assertTrue(feasible_endpoints(np.array([[.2,.3],[.5,.6]]),e,1)[0])
 def test_confidence_oracle_preserves_coverage_and_scores(self):
  from scripts.graph_error_counterfactuals import confidence_oracle
  c=dict(physical=np.array([[0.,0,0],[.2,0,0],[4.,0,0]]),truth=np.array([[0.,0,0],[4.,0,0]]),scores=np.array([.1,.9,.8]))
  before=c['scores'].copy();ids,mapping=confidence_oracle(c,1.)
  self.assertEqual(set(ids),{1,2});self.assertEqual(mapping,{1:0,2:1});np.testing.assert_array_equal(before,c['scores'])
if __name__=='__main__':unittest.main()
