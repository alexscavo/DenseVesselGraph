import unittest
import torch
from models.node_refinement import LocalNodeRefinement


class CoordinateOnlyChecks(unittest.TestCase):
    def test_compact_logits_and_permutation(self):
        torch.manual_seed(14)
        head=LocalNodeRefinement(12,channels=2,refine_classification=False,
                                 detach_features=False,compact_output=True)
        self.assertEqual(head.network[-1].out_features,3)
        t=torch.randn(2,7,12);n=torch.rand(2,7,6);l=torch.randn(2,7,2)
        f=torch.randn(2,2,8,8,8,requires_grad=True)
        a,b=head(t,n,l,f,(16,16,16))
        torch.testing.assert_close(a,n,rtol=0,atol=0)
        torch.testing.assert_close(b,l,rtol=0,atol=0)
        with torch.no_grad():head.network[-1].weight.normal_(0,.02)
        a,b=head(t,n,l,f,(16,16,16));order=torch.tensor([4,0,6,3,1,5,2])
        aa,bb=head(t[:,order],n[:,order],l[:,order],f,(16,16,16))
        torch.testing.assert_close(aa,a[:,order]);torch.testing.assert_close(bb,l[:,order],rtol=0,atol=0)
        a.sum().backward();self.assertGreater(float(f.grad.norm()),0)

    def test_legacy_layout_is_preserved(self):
        self.assertEqual(LocalNodeRefinement(12,refine_classification=False).network[-1].out_features,5)
        with self.assertRaises(ValueError):LocalNodeRefinement(12,compact_output=True)


if __name__=='__main__':unittest.main()
