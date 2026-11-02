import unittest
import numpy as np
import torch
from models.node_verifier import NodeVerifier
from scripts.node_verifier_metrics import association, supervision, measure, select, rank_nms


class VerifierTests(unittest.TestCase):
    def test_identity_gradient_permutation_and_ties(self):
        torch.manual_seed(21);torch.set_num_threads(1)
        for context in [False,True]:
            net=NodeVerifier(16,context=context)
            tokens=torch.randn(2,12,16);logits=torch.randn(2,12,2);xyz=torch.rand(2,12,3)
            xyz[:,1:8]=xyz[:,0:1]  # Exact nearest-neighbor boundary ties.
            inputs=[tokens,logits,xyz,xyz*64]
            self.assertTrue(torch.equal(net(*inputs),logits[...,1]-logits[...,0]))
            opt=torch.optim.Adam(net.parameters(),lr=.01)
            for _ in range(3):
                opt.zero_grad();loss=net(*inputs).square().mean();loss.backward();opt.step()
            self.assertGreater(float(net.point[1].weight.grad.abs().sum()),0)
            permutation=torch.randperm(12);out=net(*inputs);reordered=net(*[x[:,permutation] for x in inputs])
            torch.testing.assert_close(reordered,out[:,permutation],atol=2e-6,rtol=2e-6)
            if not context:
                changed=tokens.clone();changed[:,1:]+=100
                torch.testing.assert_close(net(changed,*inputs[1:])[:,0],out[:,0])

    def test_one_to_one_ambiguity_and_candidate_identity(self):
        x=np.array([[0,0,0],[.1,0,0],[1,0,0],[10,0,0],[30,0,0]],float)
        gt=np.array([[0,0,0],[10,0,0]],float)
        y,w=supervision(x,gt)
        np.testing.assert_array_equal(y,[1,-1,0,1,0]);self.assertEqual(w[2],.25)
        self.assertEqual(len(association(x,gt)),2)
        c=dict(physical=x,truth=gt,gt_edges=np.array([[0,1]]),scores=np.array([.1,.2,.9,.9,.1]),pairs=np.array([[0,3],[2,3],[3,4]]),relations=np.array([.8,.1,.9]))
        a=measure(c,[0,3]);b=measure(c,[2,3])
        self.assertEqual(a['edge_tp'],1);self.assertEqual(b['edge_tp'],0)
        self.assertEqual(a['node_tp'],b['node_tp'])

    def test_cached_suppression_matches_direct_selection(self):
        c=dict(physical=np.array([[0.,0,0],[.1,0,0],[2,0,0],[5,0,0]]))
        scores=np.array([.8,.9,.4,.3])
        for radius in [0.,.5,1.,2.]:
            keep,removed=rank_nms(c['physical'],scores,radius)
            for threshold in [.2,.5,.85]:
                np.testing.assert_array_equal(select(c,scores,threshold,radius),keep[scores[keep]>threshold])
                np.testing.assert_array_equal(select(c,scores,threshold,radius,3),np.r_[keep,removed][:3])
        scores[:]=[.95,.1,.8,.2]
        keep,_=rank_nms(c['physical'],scores,.5)
        np.testing.assert_array_equal(select(c,scores,.5,.5),keep[scores[keep]>.5])


if __name__=='__main__':unittest.main()
