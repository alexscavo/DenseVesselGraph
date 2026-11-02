"""Focused native-2D scientific contracts; no dataset or GPU required."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from native2d.config import load_config
from native2d.targets import Graph2D,from_mask,from_dense_graph,crop_graph,fit_controls,save_target
from native2d.data import transform
from native2d.geometry import LocalCoordinateRefinement2D,SplineHead2D,sample_local
from native2d.evaluation import evaluate,match_nodes,lift
from metrics.branch_connectivity import contract_degree_two


class Native2DTests(unittest.TestCase):
    def test_crop_preserves_outside_crossing_and_distinct_crossings(self):
        nodes=np.array([[-2,4],[12,4],[4,-2],[4,12]],float)
        graph=from_dense_graph(nodes,[[0,1],[2,3]],provenance={})
        crop=crop_graph(graph,[0,0],[10,10])
        self.assertEqual(len(crop.nodes),4);self.assertEqual(len(crop.edges),2)
        self.assertEqual(contract_degree_two(crop.nodes,crop.edges).betti_0,2)
        np.testing.assert_allclose(crop.paths[0],[[0,4],[9,4]])

    def test_mask_junction_and_cycle(self):
        mask=np.zeros((15,15),bool);mask[7,2:13]=True;mask[2:13,7]=True
        graph=from_mask(mask,provenance={})
        self.assertTrue(graph.provenance['requires_crossing_review'])
        cycle=np.array([[0,0],[1,0],[1,1],[0,1]],float)
        graph=from_dense_graph(cycle,[[0,1],[1,2],[2,3],[3,0]],provenance={})
        self.assertEqual(len(graph.nodes),3)
        self.assertEqual(contract_degree_two(graph.nodes,graph.edges).betti_1,1)

    def test_spacing_rotation_and_dense_paths(self):
        image=torch.zeros(3,5,8);image[:,1,6]=1
        target=dict(nodes=np.array([[6/8,1/5],[2/8,3/5]]),
            controls=np.array([[[6/8,1/5],[2/8,3/5]]]),
            full_paths=np.array([[6/8,1/5],[2/8,3/5]]),metric=np.diag([8*.2,5*.4]),shape_hw=np.array([5,8]))
        before=np.diff(target['nodes'],axis=0)@target['metric'].T
        rotated,new=transform(image,target,quarter_turns=1,flip_x=True)
        points=new['nodes']*np.array([rotated.shape[-1],rotated.shape[-2]])
        np.testing.assert_allclose(rotated[:,int(points[0,1]),int(points[0,0])],1)
        np.testing.assert_allclose(np.linalg.norm(np.diff(new['nodes'],axis=0)@new['metric'].T),np.linalg.norm(before))

    def test_local_sampling_pixel_centers_and_zero_refinement(self):
        features=torch.arange(24.).reshape(1,1,4,6)
        point=torch.tensor([[[[4.,2.]]]])
        self.assertEqual(sample_local(features,point).item(),8.)
        refiner=LocalCoordinateRefinement2D(8,channels=1)
        nodes=torch.rand(1,3,4);tokens=torch.rand(1,3,8)
        torch.testing.assert_close(refiner(tokens,nodes,features,[8,12]),nodes,rtol=0,atol=0)

    def test_spline_chord_endpoints_gradients(self):
        head=SplineHead2D(12,16)
        endpoints=torch.tensor([[[.1,.2],[.7,.9]]],requires_grad=True)
        metric=torch.diag(torch.tensor([23.,17.]))
        features=torch.randn(1,12,requires_grad=True)
        controls,curves=head(features.detach(),endpoints.detach(),metric)
        torch.testing.assert_close(controls[:,[0,-1]],endpoints.detach(),rtol=0,atol=0)
        line=(1-torch.linspace(0,1,64)[None,:,None])*endpoints[:,:1]+torch.linspace(0,1,64)[None,:,None]*endpoints[:,1:]
        torch.testing.assert_close(curves,line,rtol=1e-5,atol=1e-6)
        curves.square().mean().backward()
        self.assertIsNone(endpoints.grad);self.assertIsNone(features.grad)
        self.assertTrue(torch.isfinite(head.network[-1].weight.grad).all())
        fitted=fit_controls(np.array([[0,0],[3,2],[7,5]]))
        np.testing.assert_array_equal(fitted[[0,-1]],[[0,0],[7,5]])

    def test_metric_identity_units_and_isometric_branch_counts(self):
        graph=Graph2D(np.array([[1.,1.],[7.,4.]]),np.array([[0,1]]),[np.array([[1.,1.],[4.,8.],[7.,4.]])],{})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'gt.npz';save_target(path,graph,[12,16])
            with np.load(path) as z:target=dict(z)
            prediction=dict(nodes=target['nodes'],edges=target['edges'],edge_polylines=[graph.paths[0]/[16,12]])
            result=evaluate(prediction,target)
        self.assertEqual(result['2.0']['edge']['f1'],1.)
        self.assertEqual(result['common_v2']['common_branch_f1'],1.)
        self.assertEqual(result['common_v2']['common_curve_f1'],1.)
        self.assertEqual(result['units'],'pixel')
        self.assertAlmostEqual(result['node_mAP'],1.)
        a=contract_degree_two(graph.nodes,graph.edges);b=contract_degree_two(lift(graph.nodes),graph.edges)
        self.assertEqual((a.betti_0,a.betti_1),(b.betti_0,b.betti_1))

    def test_maximum_cardinality_positional_matching(self):
        mapping,_=match_nodes([[.9,0],[0,0]],[[0,0],[1.8,0]],1.)
        self.assertEqual(len(mapping),2)

    def test_legacy_3d_factory_branch_is_unchanged(self):
        # No 3D construction/weights are changed by the explicit 2D dispatch.
        import inspect
        import models
        source=inspect.getsource(models.build_model)
        self.assertIn("get('spatial_dims', 3)",source)
        self.assertIn('from .relationformer import build_relationformer',source)


if __name__=='__main__':unittest.main()
