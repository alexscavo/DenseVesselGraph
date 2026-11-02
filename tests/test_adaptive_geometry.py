import unittest
import numpy as np
import torch
from scipy.interpolate import BSpline
from models.adaptive_geometry import PiecewiseSplineHead,AdaptiveSplineBank,piecewise_knots,fit_basis

class GeometryTest(unittest.TestCase):
    def test_piecewise_continuity_and_endpoint_constraints(self):
        rng=np.random.default_rng(19)
        for continuity in ['C0','C1']:
            knots=piecewise_knots(4,continuity);controls=rng.normal(size=(len(knots)-4,3));s=BSpline(knots,controls,3)
            for t in [.25,.5,.75]:
                np.testing.assert_allclose(s(t-1e-10),s(t+1e-10),atol=1e-6)
                if continuity=='C1':np.testing.assert_allclose(s.derivative()(t-1e-10),s.derivative()(t+1e-10),atol=1e-5)
            head=PiecewiseSplineHead(12,16,segments=4,continuity=continuity,local_geometry=dict(enabled=True,channels=64,samples=8))
            e=torch.tensor([[[.1,.2,.3],[.9,.7,.4]]]);c,x=head(torch.zeros(1,12),e,torch.eye(3)*40,local_features=torch.randn(64,8,8,8),input_shape=(16,16,16))
            torch.testing.assert_close(c[:,[0,-1]],e)
            t=torch.linspace(0,1,64)[None,:,None];torch.testing.assert_close(x,(1-t)*e[:,:1]+t*e[:,1:],atol=1e-6,rtol=1e-5)
            x.square().mean().backward();self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters()))
    def test_bank_projections_and_selector_are_separate(self):
        bank=AdaptiveSplineBank(12,16)
        self.assertIs(bank.heads['4'].network[0],bank.heads['24'].network[0])
        e=torch.rand(3,2,3);f=torch.randn(3,12)
        for k in bank.candidates:
            c,x=bank.forward_k(k,f,e,torch.eye(3)*40);self.assertEqual(c.shape,(3,k,3))
            x.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in bank.selector.parameters()))
        bank.zero_grad(set_to_none=True);bank.requires_grad_(False);bank.selector.requires_grad_(True)
        bank.selector(bank.features(f,e,torch.eye(3)*40).detach()).sum().backward()
        self.assertTrue(all(p.grad is None for n,p in bank.named_parameters() if not n.startswith('selector.')))
    def test_piecewise_simplification_uses_predicted_curve_only(self):
        from models.adaptive_geometry import simplify_piecewise_curve
        path=np.c_[np.linspace(0,50,257),np.zeros((257,2))]
        curve,segments,controls=simplify_piecewise_curve(path,'C1',.5)
        self.assertEqual(segments,1);self.assertEqual(controls,8)
        np.testing.assert_allclose(curve[[0,-1]],path[[0,-1]])

    def test_fit_uses_its_own_basis(self):
        t=np.linspace(0,1,100);path=np.c_[40*t,2*np.sin(6*np.pi*t),np.cos(2*np.pi*t)]
        for continuity in ['C0','C1']:
            c,x=fit_basis(path,piecewise_knots(4,continuity));np.testing.assert_allclose(c[[0,-1]],path[[0,-1]])
            self.assertTrue(np.isfinite(x).all())

if __name__=='__main__':unittest.main()
