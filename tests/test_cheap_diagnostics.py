import unittest
import numpy as np
from scripts.cheap_research_diagnostics import thresholds,paired_ap
class CheapDiagnostics(unittest.TestCase):
 def test_strict_threshold_ties_preserve_pair_population(self):
  score=np.array([.8,.8,.4,.2]);y=np.array([1,0,1,0],bool)
  p=thresholds(score,y,2,.5)
  chosen=score>p['max_f1'];self.assertEqual(chosen.tolist(),[True,True,True,False])
  self.assertEqual((score>p['preserve_precision']).tolist(),[True,True,True,False])
 def test_weighted_ap_matches_sklearn_with_tied_scores(self):
  from sklearn.metrics import average_precision_score
  labels=np.array([1,0,1,0],bool);score=np.array([.7,.7,.4,.1]);subject=np.array([0,0,1,1]);counts=np.array([[1,1],[2,0],[0,2]])
  ap=paired_ap(labels,score,subject,counts,np.arange(2))
  for i,c in enumerate(counts):self.assertAlmostEqual(ap[i],average_precision_score(labels,score,sample_weight=c[subject]))
if __name__=='__main__':unittest.main()
