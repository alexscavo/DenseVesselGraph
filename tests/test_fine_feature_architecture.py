import unittest
import torch
from models.node_refinement import LocalNodeRefinement
from training.losses.focal import softmax_focal_loss


class FineFeatureChecks(unittest.TestCase):
    def test_fine_warmup_reaches_material_attention(self):
        from tests.test_multiscale_relationformer import tiny
        from models.multiscale_migration import migrate_single_level
        from models.ops.modules import MSDeformAttn
        from scripts.fine_feature_architecture import fine_warmup
        torch.manual_seed(24);torch.set_num_threads(1)
        base=tiny();m=tiny(2);migrate_single_level(m,base.state_dict(),.1)
        layers=[x for x in m.modules() if isinstance(x,MSDeformAttn)]
        for layer in layers:layer.record_level_diagnostics=True
        x=torch.randn(2,1,8,8,8)
        fine_warmup(m,0,500);m(x)
        for layer in layers:self.assertAlmostEqual(float(layer.last_level_mass[1]),1e-4/(1+1e-4),places=7)
        fine_warmup(m,500,500);m(x)
        for layer in layers:
            self.assertEqual(layer.fine_logit_offset,0.)
            self.assertAlmostEqual(float(layer.last_level_mass[1]),.1/1.1,places=6)

    def test_immediate_unweighted_focal_formula(self):
        x=torch.tensor([[.2,1.],[-.5,.3]],requires_grad=True);y=torch.tensor([1,0]);p=x.softmax(-1)[torch.arange(2),y]
        torch.testing.assert_close(softmax_focal_loss(x,y,[1.,1.],2),(-((1-p)**2)*p.log()).mean())

    def test_local_identity_and_live_features(self):
        torch.manual_seed(24)
        h=LocalNodeRefinement(12,channels=2,detach_features=False)
        t=torch.randn(2,5,12,requires_grad=True);n=torch.rand(2,5,6,requires_grad=True);l=torch.randn(2,5,2,requires_grad=True);f=torch.randn(2,2,8,8,8,requires_grad=True)
        a,b=h(t,n,l,f,(16,16,16));torch.testing.assert_close(a,n,rtol=0,atol=0);torch.testing.assert_close(b,l,rtol=0,atol=0)
        # Zero output initialization necessarily delays feature gradients until
        # the output weights update; test the next step, not an impossible step 0.
        opt=torch.optim.SGD(h.parameters(),lr=.1);(a.square().mean()+b.square().mean()).backward();opt.step();opt.zero_grad();f.grad=None
        a,b=h(t,n,l,f,(16,16,16));(a.square().mean()+b.square().mean()).backward()
        self.assertGreater(float(f.grad.norm()),0)
        h.image_mode='zero';a,b=h(t,n,l,f,(16,16,16));aa,bb=h(t,n,l,f*100,(16,16,16))
        torch.testing.assert_close(a,aa,rtol=0,atol=0);torch.testing.assert_close(b,bb,rtol=0,atol=0)

    def test_local_grid_dhw_alignment(self):
        h=LocalNodeRefinement(1,channels=1,detach_features=False,radius_vox=2)
        d,y,x=torch.meshgrid(torch.arange(8.),torch.arange(8.),torch.arange(8.),indexing='ij');f=(100*d+10*y+x)[None,None]
        captured=[];handle=h.network[0].register_forward_pre_hook(lambda m,a:captured.append(a[0].detach()))
        h(torch.zeros(1,1,1),torch.tensor([[[.5,.375,.25,0.,0.,0.]]]),torch.zeros(1,1,2),f,(16,16,16));handle.remove()
        # Input voxel (8,6,4) is feature index (4,3,2), flattened lattice centre13.
        self.assertAlmostEqual(float(captured[0][0,0,1+13]),432.,places=4)


if __name__=='__main__':unittest.main()
