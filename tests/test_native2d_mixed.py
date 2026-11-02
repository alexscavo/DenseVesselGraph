"""Mixed source/target membership, sampling and stage-specific metric checks."""
import unittest
import numpy as np
import torch
from scripts.train_native2d_long import MixedDataset,mixed_exposure,subset,config

class MixedPretrainingTests(unittest.TestCase):
    def test_sampling_contract_and_reproducibility(self):
        a=mixed_exposure(1000,30,3,seed=61)
        np.testing.assert_array_equal(a,mixed_exposure(1000,30,3,seed=61))
        self.assertEqual(a.shape,(3,2000));self.assertGreaterEqual(a.min(),0);self.assertLess(a.max(),1030)
        self.assertLess(abs(float((a<1000).mean())-.5),.02)
        self.assertLess(len(np.unique(a[0])),len(a[0]))

    def test_micron_conversion_and_cache_are_separate(self):
        ds=MixedDataset('reaver','train',augment=True,max_queries=96)
        index=next(i for i,r in enumerate(ds.vessels.rows) if r['nodes']>0)
        ds.vessels.cache[index]=ds.vessels.raw(index)
        original={k:v.copy() for k,v in ds.vessels.cache[index][1].items()}
        for seed in [1,12,29]:
            torch.manual_seed(seed);image,target,row=ds[ds.source_count+index]
            self.assertEqual(image.shape,(3,128,128));self.assertEqual(str(target['units'].item()),'pixel')
            np.testing.assert_allclose(np.linalg.norm(target['metric'],axis=0),[128,128],atol=1e-5)
            for k,v in original.items():np.testing.assert_array_equal(ds.vessels.cache[index][1][k],v)
        self.assertEqual(str(original['units'].item()),'micron')
        self.assertEqual(config('reaver','B0')['training']['native2d_protocol']['units'],'micron')
        self.assertEqual(config('reaver','B0',pretrain=True)['training']['native2d_protocol']['units'],'pixel')

    def test_only_training_membership_is_sampled(self):
        ds=MixedDataset('avwide','train',max_queries=96)
        val=MixedDataset('avwide','validation',max_queries=96)
        self.assertTrue(all(r['split']=='train' for r in ds.rows))
        self.assertFalse({r['subject_id'] for r in ds.rows}&{r['subject_id'] for r in val.rows})
        self.assertEqual(set(config('avwide','B0',pretrain=True)['data']['datasets']),{'plants','avwide'})

    def test_validation_selection_contains_only_target(self):
        ds=MixedDataset('reaver','validation',max_queries=96)
        ids=subset(ds,list(range(ds.source_count,len(ds))))
        self.assertTrue(ids);self.assertTrue(all(i>=ds.source_count for i in ids))
        self.assertEqual({ds.rows[i]['subject_id'] for i in ids},{r['subject_id'] for r in ds.vessels.rows})

if __name__=='__main__':unittest.main()
