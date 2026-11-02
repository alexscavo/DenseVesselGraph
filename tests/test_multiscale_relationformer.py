import copy
import unittest
import torch
from torch import nn
from models.deformable_transformer import build_def_detr_transformer, DeformableTransformerEncoder
from models.relationformer import RelationFormer
from models.multiscale_migration import migrate_single_level, set_fine_enabled
from models.ops.modules import MSDeformAttn


class TinyEncoder(nn.Module):
    num_features = 24
    def __init__(self):
        super().__init__()
        self.layer3 = nn.Conv3d(1, 12, 3, 2, 1)
        self.layer4 = nn.Conv3d(12, 24, 3, 2, 1)
    def forward(self,x):return self.layer4(self.layer3(x).relu())


def tiny(levels=1):
    cfg={'model':{'num_classes':2,'decoder':dict(hidden_dim=24,attention_heads=6,encoder_layers=2,decoder_layers=2,feedforward_dim=48,dropout=0.,activation='relu',feature_levels=levels,decoder_points=2,encoder_points=2,relation_attention=True,use_cuda_extension=False,object_queries=12,relation_tokens=2,dummy_tokens=0)}}
    return RelationFormer(TinyEncoder(),build_def_detr_transformer(cfg),cfg)


class MultiscaleTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(23);torch.set_num_threads(1)

    def test_migration_parity_and_connected_fine_path(self):
        base=tiny().eval();multi=tiny(2).eval()
        # A trained checkpoint has query-dependent attention/offset matrices.
        # At the library's zero initialization, positional-embedding gradients
        # are expected to be zero until those matrices take their first update.
        with torch.no_grad():
            for m in base.modules():
                if isinstance(m,MSDeformAttn):
                    m.sampling_offsets.weight.normal_(0,.01)
                    m.attention_weights.weight.normal_(0,.01)
        source=copy.deepcopy(base.state_dict())
        migrate_single_level(multi,source)
        for name,module in multi.named_modules():
            if isinstance(module,MSDeformAttn):
                for field,width in [('sampling_offsets',3),('attention_weights',1)]:
                    key=name+'.'+field+'.weight'
                    old=source[key].reshape(6,1,2,width,-1)
                    new=multi.state_dict()[key].reshape(6,2,2,width,-1)
                    torch.testing.assert_close(new[:,0],old[:,0],rtol=0,atol=0)
                    torch.testing.assert_close(new[:,1],torch.zeros_like(old[:,0]) if field=='sampling_offsets' else old[:,0],rtol=0,atol=0)
                bias=module.sampling_offsets.bias.reshape(6,2,2,3)[:,1]
                self.assertLessEqual(float(bias.abs().max()),1.)
        x=torch.randn(2,1,8,12,16)
        with torch.no_grad():
            a=base(x);set_fine_enabled(multi,False);b=multi(x)
            torch.testing.assert_close(a[0],b[0],rtol=2e-5,atol=2e-5)
            for k in ['pred_logits','pred_nodes']:torch.testing.assert_close(a[1][k],b[1][k],rtol=2e-5,atol=2e-5)
            set_fine_enabled(multi,True);near=multi(x)
            self.assertLess(float((a[1]['pred_nodes']-near[1]['pred_nodes']).abs().max()),.001)
        for m in multi.modules():
            if isinstance(m,MSDeformAttn):m.record_level_diagnostics=True
        tokens,pred,_=multi(x);loss=pred['pred_nodes'].square().mean()+pred['pred_logits'].square().mean();loss.backward()
        for name in ['fine_input_proj.weight','encoder.layer3.weight','input_proj.weight','decoder.level_embed']:
            grad=dict(multi.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0,name)
        for name,m in multi.named_modules():
            if isinstance(m,MSDeformAttn):
                torch.testing.assert_close(m.last_level_mass.sum(),torch.tensor(1.))
                self.assertGreater(float(m.last_level_mass[1]),0)
                grad=m.sampling_offsets.weight.grad.reshape(6,2,2,3,-1)[:,1]
                self.assertGreater(float(grad.abs().sum()),0,name)

    def test_rectangular_spatial_reference_points(self):
        shapes=torch.tensor([[2,3,4],[4,6,8]])
        refs=DeformableTransformerEncoder.get_reference_points(shapes,torch.ones(2,2,3),'cpu')
        self.assertEqual(tuple(refs.shape),(2,216,2,3))
        torch.testing.assert_close(refs[0,0,0],torch.tensor([.125,1/6,.25]))
        torch.testing.assert_close(refs[0,23,1],torch.tensor([.875,5/6,.75]))
        torch.testing.assert_close(refs[0,24,1],torch.tensor([.0625,1/12,.125]))
        torch.testing.assert_close(refs[:,:,0],refs[:,:,1])


if __name__=='__main__':unittest.main()
