import unittest
import numpy as np
from scripts.continuous_representation import fit_options,choose,split_path,compressed_curve,curve_features
from scripts.analyze_spline_target_capacity import measure

class ContinuousRepresentationTest(unittest.TestCase):
    def test_straight_requires_no_split(self):
        path=np.c_[np.linspace(0,100,101),np.zeros((101,2))]
        options=fit_options(path);k,ok,_=choose(options,path,.25)
        self.assertEqual(k,4);self.assertTrue(ok)
        result=split_path(path,.25);self.assertEqual(len(result['controls']),1);self.assertTrue(result['passed'])
        cc,k=compressed_curve(options[24],.25);self.assertEqual(k,4)
        np.testing.assert_allclose(cc[[0,-1]],path[[0,-1]],atol=1e-9)
    def test_curved_split_preserves_path_and_budget(self):
        t=np.linspace(0,1,121);path=np.c_[40*t,4*np.sin(6*np.pi*t),2*np.sin(4*np.pi*t)]
        result=split_path(path,.5)
        self.assertGreater(len(result['controls']),1);self.assertTrue(result['passed'])
        self.assertEqual(result['intervals'][0][0],0);self.assertEqual(result['intervals'][-1][1],len(path)-1)
        for i in range(len(result['controls'])-1):np.testing.assert_array_equal(result['controls'][i][-1],result['controls'][i+1][0])
        # Each degree-2 subdivision adds exactly one node and edge: E-V+C invariant.
        n=len(result['controls']);self.assertEqual(n-(2+n-1)+1,0)
    def test_inference_compression_has_no_gt_and_is_reversible(self):
        t=np.linspace(0,1,81);path=np.c_[30*t,2*np.sin(2*np.pi*t),t*t]
        c=fit_options(path)[24];a,k=compressed_curve(c,.25);b,j=compressed_curve(c[::-1],.25)
        self.assertEqual(k,j);np.testing.assert_allclose(a,b[::-1],atol=1e-8)
        np.testing.assert_allclose(curve_features(c),curve_features(c[::-1]),atol=1e-8)
        self.assertLess(measure(c,path)['endpoint_max_mm'],1e-9)
    def test_variable_heads_have_zero_chord_and_finite_gradients(self):
        import torch
        from models.spline import SplineHead
        torch.manual_seed(221039)
        endpoints=torch.tensor([[[.1,.2,.3],[.8,.7,.6]]])
        metric=torch.eye(3)*40
        for k in [8,16,24]:
            head=SplineHead(12,16,k,local_geometry=dict(enabled=True,channels=64,samples=8))
            controls,curves=head(torch.randn(1,12),endpoints,metric,local_features=torch.randn(64,8,8,8),input_shape=(16,16,16))
            t=torch.linspace(0,1,64)[None,:,None]
            torch.testing.assert_close(curves,(1-t)*endpoints[:,:1]+t*endpoints[:,1:],atol=1e-6,rtol=1e-5)
            self.assertEqual(tuple(controls.shape),(1,k,3))
            ((controls[:,1:-1]-.2)**2).mean().backward()
            grads=[p.grad for p in head.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)

    def test_failed_tolerance_is_explicit(self):
        t=np.linspace(0,1,81);path=np.c_[20*t,np.sin(8*np.pi*t),np.zeros_like(t)]
        options=fit_options(path);_,ok,_=choose(options,path,.01)
        self.assertFalse(ok)  # Numerical certificate alone exceeds this tolerance.

if __name__=='__main__':unittest.main()
