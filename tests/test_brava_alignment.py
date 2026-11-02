"""Focused regressions for the BRAVA per-subject orientation failure."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import tempfile
import json
import numpy as np
from scripts.prepare_brava import alignment,swc_to_voxel,audit


class BravaAlignmentChecks(unittest.TestCase):
    def test_subject_reflection_matches_image_without_changing_trace(self):
        shape=np.array([16]*3);q=np.array([[2.,3.,4.],[3.,5.,6.],[4.,7.,8.]])
        xyz=np.column_stack([q[:,0],q[:,2]-15,15-q[:,1]])*.62
        a=np.zeros((3,7));a[:,2:5]=xyz;original=a.copy()
        image=np.ones(tuple(shape),np.float32);correct=q.copy();correct[:,0]=15-correct[:,0]
        image[tuple(correct.astype(int).T)]=100
        row=dict(subject='reflection-fixture',shape=shape,spacing=[.62]*3,reflect_x=True)
        result=alignment(row,image,np.ones(tuple(shape),bool),a)
        self.assertEqual(result['inside_fraction'],1.)
        np.testing.assert_allclose(swc_to_voxel(xyz,shape,[.62]*3,reflect_x=True),correct)
        np.testing.assert_array_equal(a,original)

    def test_wrong_unrecorded_orientation_is_still_rejected(self):
        shape=np.array([16]*3);q=np.array([[2.,3.,4.],[3.,5.,6.],[4.,7.,8.]])
        a=np.zeros((3,7));a[:,2:5]=np.column_stack([q[:,0],q[:,2]-15,15-q[:,1]])*.62
        image=np.ones(tuple(shape),np.float32);q[:,0]=15-q[:,0];image[tuple(q.astype(int).T)]=100
        with self.assertRaises(AssertionError):alignment(dict(subject='wrong',shape=shape,spacing=[.62]*3),image,np.ones(tuple(shape),bool),a)

    def test_default_mapping_and_native_spacing_remain_unchanged(self):
        xyz=np.array([[6.2,-12.4,18.6]])
        np.testing.assert_allclose(swc_to_voxel(xyz,[100,100,100],[.5625,.5625,.62]),[[10,69,79]])

    def test_audit_reports_all_bad_subjects_before_blocking(self):
        rows=[dict(subject=s,image='image',brain_mask='mask',valid_swcs={'original':'trace'}) for s in ['bad-A','bad-B','good']]
        fake_inventory=dict(swcs={},invalid_swcs={},subjects=rows)
        result=dict(inside_fraction=1.,brain_mask_fraction=1.)
        with tempfile.TemporaryDirectory() as directory,patch('scripts.prepare_brava.inventory',return_value=fake_inventory),patch('scripts.prepare_brava.nib.load',return_value=SimpleNamespace(dataobj=np.ones((4,4,4)))),patch('scripts.prepare_brava.read_swc',return_value=(np.zeros((1,7)),[])),patch('scripts.prepare_brava.alignment',side_effect=[AssertionError('first'),AssertionError('second'),result]) as check:
            with self.assertRaises(AssertionError):audit(Path('source'),Path(directory))
            failures=json.loads((Path(directory)/'audit/failures.json').read_text())
            self.assertEqual([r['subject'] for r in failures],['bad-A','bad-B'])
            self.assertEqual(check.call_count,3)
            self.assertFalse((Path(directory)/'audit/complete.json').exists())


if __name__=='__main__':unittest.main()
