"""Regression from a real clipped AV-WIDE branch with an unsampled tail."""
import unittest
import numpy as np
from scipy.interpolate import BSpline
from native2d.targets import fit_controls,from_dense_graph,crop_graph
from metrics.representation_geometry import point_to_polyline_distance

class StableSplineTargetTests(unittest.TestCase):
    def test_near_boundary_clip_does_not_reuse_outside_anchor(self):
        nodes=np.array([[1399.005,600],[1390,590],[1380,580]],float)
        graph=from_dense_graph(nodes,[[0,1],[1,2]],provenance={})
        patch=crop_graph(graph,[1272,512],[128,128])
        self.assertEqual(float(patch.nodes[:,0].max()),127.)
        for edge,path in zip(patch.edges,patch.paths):
            np.testing.assert_allclose(path[[0,-1]],patch.nodes[edge],atol=1e-9,rtol=0)

    def test_uneven_dense_path_retains_geometry(self):
        path=np.array([[103,90],[100,88],[96,85],[91,83],[88,82],[83,79],
            [79,77],[76,75],[72,73],[69,72],[66,70],[63,68],[60,68],
            [57,68],[54,66],[19.125,61.125]],float)
        controls=fit_controls(path)
        knots=np.r_[np.zeros(4),np.arange(1,5)/5,np.ones(4)]
        curve=BSpline(knots,controls,3)(np.linspace(0,1,257))
        np.testing.assert_array_equal(controls[[0,-1]],path[[0,-1]])
        self.assertLess(float(np.abs(controls).max()),200.)
        self.assertLess(float(point_to_polyline_distance(curve,path).max()),3.)

if __name__=='__main__':unittest.main()
