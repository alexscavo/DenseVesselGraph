import unittest
import numpy as np
from scripts.collect_brava_learning_diagnosis import metrics,summarize,paired_difference

class DiagnosticChecks(unittest.TestCase):
 def test_raw_filtering_and_relation_accounting(self):
  c=dict(physical=np.array([[0.,0,0],[4.,0,0],[20.,0,0]]),truth=np.array([[0.,0,0],[4.,0,0]]),scores=np.array([.9,.3,.8]),gt_edges=np.array([[0,1]]),pairs=np.array([[0,1],[0,2],[1,2]]),relations=np.array([.8,.9,.2]),hungarian_distance=np.zeros(2))
  r,_,_=metrics(c,2.);self.assertEqual(r['raw_tp'],2);self.assertEqual(r['node_tp'],1);self.assertEqual(r['false_edges_spatial_unmatched'],1)
  r2,_,_=metrics(c,2.,.2);self.assertEqual(r2['edge_tp'],1);self.assertEqual(r2['eligible'],1)
  r.update(subject='s',tolerance=2.,step='4000');s=summarize([r]);self.assertNotIn('tolerance',s);self.assertEqual(s['raw_recall'],1.)
 def test_paired_identical_control(self):
  rows=[dict(subject=s,node_tp=2,nodes=3,gt_nodes=2,edge_tp=1,edges=2,gt_edges=1,raw_tp=2) for s in ['a','b']]
  r=paired_difference(rows,rows,'node_f1');self.assertEqual(r['delta'],0.);self.assertEqual(r['ci95'],[0.,0.])
if __name__=='__main__':unittest.main()
