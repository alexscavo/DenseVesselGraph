import unittest
import torch
from models.node_refinement import LocalNodeRefinement
from scripts.ixi_local_components import expand_queries,classification_only,refinement_spec

class IXILocalComponents(unittest.TestCase):
 def test_coordinate_and_classification_routing(self):
  torch.manual_seed(4);tokens=torch.randn(1,4,12);nodes=torch.rand(1,4,6);logits=torch.randn(1,4,2);features=torch.randn(1,64,32,32,32,requires_grad=True)
  for arm in ['A-xyz-fresh','A-cls','A']:
   spec=refinement_spec(arm);spec.pop('enabled');head=LocalNodeRefinement(12,**spec)
   if arm=='A-cls':head.register_forward_hook(classification_only)
   xyz,cls=head(tokens,nodes,logits,features,(64,64,64));torch.testing.assert_close(xyz,nodes);torch.testing.assert_close(cls,logits)
   with torch.no_grad():head.network[-1].weight.fill_(.002)
   xyz,cls=head(tokens,nodes,logits,features,(64,64,64))
   if arm=='A-cls':torch.testing.assert_close(xyz,nodes,rtol=0,atol=0);self.assertGreater((cls-logits).abs().max(),0)
   if arm.startswith('A-xyz'):torch.testing.assert_close(cls,logits,rtol=0,atol=0);self.assertGreater((xyz-nodes).abs().max(),0)
   objective=cls.sum() if arm=='A-cls' else xyz.sum();g=torch.autograd.grad(objective,features)[0];self.assertGreater(float(g.norm()),0)
 def test_sampling_voxel_grid(self):
  # Feature ramp at early-convolution centers input index=2*i. At input voxel20 all27 sample means equal10.
  head=LocalNodeRefinement(2,radius_vox=4,detach_features=False)
  features=torch.arange(32.).view(1,1,32,1,1).expand(1,64,32,32,32).clone()
  captured=[];hook=head.network[0].register_forward_pre_hook(lambda m,a:captured.append(a[0]))
  head(torch.zeros(1,1,2),torch.full((1,1,6),20/64),torch.zeros(1,1,2),features,(64,64,64));hook.remove()
  self.assertAlmostEqual(float(captured[0][...,2:-3].mean()),10.,places=5)
 def test_query_expansion(self):
  old=torch.arange(14.).reshape(7,2);target=torch.zeros(10,2)
  expanded=expand_queries({'query_embed.weight':old},{'query_embed.weight':target},5,8)['query_embed.weight']
  torch.testing.assert_close(expanded[:5],old[:5]);torch.testing.assert_close(expanded[8:],old[5:])
  again=expand_queries({'query_embed.weight':old},{'query_embed.weight':target},5,8)['query_embed.weight'];torch.testing.assert_close(expanded,again,rtol=0,atol=0)
if __name__=='__main__':unittest.main()
