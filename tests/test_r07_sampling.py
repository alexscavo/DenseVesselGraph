import unittest
import numpy as np
from scripts.r07_sampling import population, redistribution_indices


class R07Checks(unittest.TestCase):
    def test_budget_positives_and_multiplicity(self):
        labels=np.r_[np.ones(3,int),np.zeros(20,int)]
        hard=np.zeros(23,bool);hard[3:5]=True;groups=np.zeros(23,int)
        chosen=redistribution_indices(labels,hard,groups,np.random.default_rng(4))
        np.testing.assert_array_equal(labels[chosen],labels)
        np.testing.assert_array_equal(chosen[:3],np.arange(3))
        self.assertEqual(len(chosen),len(labels));self.assertGreater(hard[chosen].sum(),hard.sum())
        self.assertLessEqual((chosen!=np.arange(len(chosen))).sum(),5)
        self.assertLessEqual(np.bincount(chosen,minlength=23)[hard].max(),4)

    def test_no_reliable_pairs_leaves_control_unchanged(self):
        chosen=redistribution_indices(np.zeros(10),np.zeros(10,bool),np.zeros(10),np.random.default_rng(4))
        np.testing.assert_array_equal(chosen,np.arange(10))

    def test_no_cross_graph_replacement(self):
        labels=np.zeros(40,int);hard=np.zeros(40,bool);hard[[0,20]]=True;groups=np.repeat([0,1],20)
        chosen=redistribution_indices(labels,hard,groups,np.random.default_rng(4))
        np.testing.assert_array_equal(groups[chosen],groups)

    def test_hungarian_assignment_is_not_spatial_reliability(self):
        gt=dict(edges=np.array([[0,1]]))
        c=dict(truth=np.array([[0.,0,0],[4.,0,0],[8.,0,0]]),
               physical=np.array([[0.,0,0],[4.,0,0],[4.1,0,0]]))
        p=population(c,gt,np.arange(3),np.arange(3))
        self.assertFalse(p['reliable'][2]);self.assertFalse(p['hard'].any())
        self.assertGreater(p['potential_conflict'].sum(),0)

    def test_unsorted_assignments_keep_endpoint_correspondence(self):
        c=dict(truth=np.array([[0.,0,0],[3.,0,0],[6.,0,0]]),physical=np.array([[0.,0,0],[3.,0,0],[6.,0,0]]))
        p=population(c,dict(edges=np.array([[0,1]])),np.array([2,0,1]),np.array([2,0,1]))
        self.assertEqual(int(p['labels'].sum()),1)
        self.assertTrue(p['hard'].any());np.testing.assert_array_equal(p['queries'][0],[0,1])

if __name__=='__main__': unittest.main()
