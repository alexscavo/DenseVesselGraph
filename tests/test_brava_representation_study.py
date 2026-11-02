import unittest
import numpy as np
import torch
from data.brava_representation_study import migrate_geometry
from models.spline import SplineHead
from scripts.brava_representation_study import target_head

class RepresentationChecks(unittest.TestCase):
 def test_real_voxel_cell_boundary_targets(self):
  from scripts.brava_representation_study import read,ROOT,target_task
  p=read(ROOT/'plan.json');sample=p['ids']['train'][0]
  _,variants,audit=target_task((p['dataset'],'train',sample))
  self.assertGreater(audit['outside_sigmoid_nodes'],0)
  self.assertLess(audit['sigmoid_clamp_max_mm'],1.)

 def test_face_vertex_is_not_necessarily_an_inserted_boundary(self):
  from scripts.brava_representation_study import read,ROOT,target_task
  p=read(ROOT/'plan.json');_,_,audit=target_task((p['dataset'],'train','sample_000002_0082'))
  self.assertGreater(audit['outside_sigmoid_non_boundary'],0)
  self.assertLess(audit['sigmoid_clamp_max_mm'],1.)

 def test_residual_migration_and_C1_continuity(self):
  from scipy.interpolate import BSpline
  from models.adaptive_geometry import piecewise_knots
  torch.manual_seed(7)
  spec={'local_geometry':dict(enabled=True,channels=4,samples=8,scale=1.,image_mode='full')}
  old=SplineHead(12,16,controls=8,local_geometry=spec['local_geometry'])
  with torch.no_grad():
   old.network[-1].weight.normal_(0,.01);old.network[-1].bias.normal_(0,.01)
   old.local_geometry.scorer[-1].weight.normal_(0,.01)
  pair=torch.randn(3,12);ends=torch.rand(3,2,3);metric=torch.eye(3)*30;local=torch.randn(4,8,8,8)
  for method in ['K24','piece_C1']:
   head,info=migrate_geometry(old,spec,method)
   a=old(pair,ends,metric,local_features=local,input_shape=(16,16,16))[1]
   b=head(pair,ends,metric,local_features=local,input_shape=(16,16,16))[1]
   self.assertLess(float(((a-b)@metric.T).abs().max()),.005)
   if method=='piece_C1':
    self.assertLess(info['basis_max_error'],1e-12)
    spline=BSpline(piecewise_knots(),np.random.default_rng(1).normal(size=(head.controls,3)),3)
    for join in [.25,.5,.75]:np.testing.assert_allclose(spline(join-1e-9,1),spline(join+1e-9,1),atol=1e-5)

 def test_rotated_extra_head_targets_preserve_correspondence(self):
  from types import SimpleNamespace
  from data.spline_augmentation import rotation_affine,rotate_spline_geometry
  source=dict(nodes=np.array([[.2,.3,.4],[.7,.6,.5]],np.float32),metric=np.diag([30,35,40]).astype(np.float32),controls=np.random.default_rng(4).random((1,24,3)).astype(np.float32),curves=np.random.default_rng(5).random((1,64,3)).astype(np.float32),guides=np.random.default_rng(6).random((1,64,3)).astype(np.float32))
  source['controls'][:,[0,-1]]=source['nodes'][None]
  turns=(1,2,3);a,offset=rotation_affine(turns,(64,64,64),dtype=torch.float32,device='cpu')
  g={k:torch.from_numpy(source[k]) for k in ['controls','curves','guides','metric']};expected=rotate_spline_geometry(g,turns,(64,64,64));expected['branch_weights']=torch.ones(1)
  y={'nodes':[torch.from_numpy(source['nodes'])@a.T+offset],'spline':[expected]}
  ds=SimpleNamespace(caches={'x':{'K24':source}},records=[SimpleNamespace(sample_id='x')])
  actual=target_head(y,'K24',[0],ds)['spline'][0]
  for k in expected:torch.testing.assert_close(actual[k],expected[k],atol=1e-6,rtol=1e-6)

if __name__=='__main__':unittest.main()
