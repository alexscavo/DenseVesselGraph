import unittest
import numpy as np
from scipy.interpolate import BSpline
from scripts.analyze_spline_target_capacity import measure,spline_polyline,FLATNESS
from metrics.representation_geometry import point_to_polyline_distance

class SplineCapacityChecks(unittest.TestCase):
    def test_straight_identity(self):
        knots=np.r_[np.zeros(4),np.arange(1,5)/5,np.ones(4)]
        greville=np.array([knots[i+1:i+4].mean() for i in range(8)])
        controls=np.column_stack([greville*10,np.zeros((8,2))]);r=measure(controls,[[0,0,0],[10,0,0]])
        self.assertLess(r['hausdorff_sampled_mm'],1e-10)
        self.assertEqual(r['curve_f1_1p0'],1.)
        self.assertLess(abs(r['length_ratio']-1),1e-12)

    def test_shortcut_catches_missing_bend(self):
        controls=np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        r=measure(controls,[[0,0,0],[1.5,3,0],[3,0,0]])
        self.assertGreater(r['dense_to_spline_max_sampled_mm'],2.99)
        self.assertGreater(r['hausdorff_lower_mm'],2.)
        self.assertLess(r['length_ratio'],.5)
        self.assertEqual(r['endpoint_max_mm'],0.)

    def test_overshoot_and_tessellation_bound(self):
        controls=np.array([[0.,0,0],[0,4,0],[3,4,0],[3,0,0]])
        curve=spline_polyline(controls)
        dense=BSpline([0,0,0,0,1,1,1,1],controls,3)(np.linspace(0,1,10001))
        self.assertLessEqual(point_to_polyline_distance(dense,curve).max(),FLATNESS+1e-10)
        r=measure(controls,[[0,0,0],[3,0,0]])
        self.assertGreater(r['spline_to_dense_max_sampled_mm'],2.99)
        self.assertLessEqual(r['hausdorff_lower_mm'],3.)
        self.assertGreaterEqual(r['hausdorff_upper_mm'],3.)

if __name__=='__main__':unittest.main()
