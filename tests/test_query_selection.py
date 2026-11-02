import unittest
import numpy as np
from scripts.query_representative_selection import select_queries,count_controls,relation_features,reference_context

class QuerySelectionChecks(unittest.TestCase):
 def fixture(self):
  x=np.array([[0.,0,0],[.4,0,0],[5.,0,0],[10.,0,0],[15.,0,0]]);p=np.array([.9,.8,.85,.75,.2]);pairs=np.stack(np.triu_indices(5,1),1);scores=np.full(len(pairs),.1);return x,p,pairs,scores
 def test_zero_relation_weight_matches_confidence_nms(self):
  x,p,pairs,r=self.fixture();expected=select_queries(x,p,pairs,r,'confidence_nms')
  for method in ['naive_global','guarded_global','local_profile']:
   np.testing.assert_array_equal(select_queries(x,p,pairs,r,method,0.),expected)
 def test_edge_order_symmetry_and_original_arrays_preserved(self):
  x,p,pairs,r=self.fixture();original=[a.copy() for a in [x,p,pairs,r]]
  a=select_queries(x,p,pairs,r,'guarded_global',1.);b=select_queries(x,p,pairs[:,::-1],r,'guarded_global',1.);np.testing.assert_array_equal(a,b)
  for left,right in zip([x,p,pairs,r],original):np.testing.assert_array_equal(left,right)
 def test_short_predicted_connection_is_not_suppressed(self):
  x,p,pairs,r=self.fixture();r[0]=.9
  self.assertNotIn(1,select_queries(x,p,pairs,r,'confidence_nms'));self.assertIn(1,select_queries(x,p,pairs,r,'protected_nms'))
 def test_counts_are_predicted_and_controls_exact(self):
  x,p,pairs,r=self.fixture()
  for method in ['local_profile','naive_global','guarded_global']:
   ids=select_queries(x,p,pairs,r,method,1.);a,b=count_controls(x,p,len(ids));self.assertEqual(len(a),len(ids));self.assertEqual(len(b),len(ids));self.assertEqual(len(set(ids)),len(ids))
 def test_guard_is_bounded_even_for_a_false_clique(self):
  n=10;x=np.stack([np.arange(n)*3.,np.zeros(n),np.zeros(n)],1);p=np.full(n,.9);r=np.full((n,n),.99);np.fill_diagonal(r,0)
  _,_,refs,d=reference_context(x,p,r);naive,guard=relation_features(x,p,r,refs,d)
  self.assertGreater(naive.max(),1.);self.assertLessEqual(guard.max(),1.)
if __name__=='__main__':unittest.main()
