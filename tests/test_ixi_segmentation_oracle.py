import copy
from pathlib import Path
import random
import unittest
import torch
from data.loaders.synthetic_mri import SyntheticMRIDataset
from data.loaders.common import SamplePaths
from scripts.ixi_segmentation_oracle import configuration,transfer_preixi,order_for


class MaskInputChecks(unittest.TestCase):
    def dataset(self,arm,**kwargs):
        self.mask=torch.zeros(8,8,8);self.mask[2:5,3:6,1:4]=1
        self.reads=[]
        def read(path):
            self.reads.append(str(path));return self.mask.clone() if str(path)=='mask' else self.mask.clone()*2
        def graph(path):return torch.tensor([[.25,.375,.125],[.5,.625,.375]]),torch.tensor([[0,1]])
        settings=dict(image_size=(8,8,8),volume_reader=read,graph_reader=graph,zoom_range=None)
        settings.update(kwargs)
        return SyntheticMRIDataset([SamplePaths(Path('raw'),Path('mask'),Path('graph'),'one')],input_modality=arm,**settings)

    def test_binary_no_intensity_or_raw_read(self):
        def fail(x):raise AssertionError('Intensity augmentation applied to mask')
        d=self.dataset('vessel_mask',augment=True,intensity_transform=fail,rng=random.Random(7),image_size=(12,12,12))
        b=d[0];self.assertEqual(self.reads,['mask']);self.assertEqual(set(b[0][0].unique().tolist()),{0.,1.})
        torch.testing.assert_close(b[0][0],b[1][0]+.5)

    def test_paired_spatial_transform_and_raw_default(self):
        d=self.dataset('raw_mra',augment=True,rng=random.Random(7),intensity_transform=lambda x:x)
        a=d[0]
        d=self.dataset('vessel_mask',augment=True,rng=random.Random(7));b=d[0]
        torch.testing.assert_close(a[2][0],b[2][0]);torch.testing.assert_close(a[3][0],b[3][0])
        torch.testing.assert_close((a[0][0]+d.foreground_mean)/2,b[0][0])

    def test_nonbinary_rejected(self):
        d=self.dataset('vessel_mask');self.mask[0,0,0]=.2
        with self.assertRaises(ValueError):d[0]

    def test_common_config_and_order(self):
        configs=[configuration({'dataset':'example'},a) for a in ['raw_mra','vessel_mask']]
        for c in configs:del c['data']['input_modality'];del c['experiment']['name']
        self.assertEqual(*configs)
        self.assertTrue((order_for(4,2048)==order_for(8,2048)[:4]).all())

    def test_preixi_transfer_preserves_relation_tokens(self):
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.query_embed=torch.nn.Embedding(258,3);self.node_refinement=torch.nn.Linear(3,3)
        model=Dummy();fresh=model.query_embed.weight.detach().clone();original=torch.randn(122,3)
        cp=dict(net={'query_embed.weight':original},config=dict(model=dict(decoder=dict(object_queries=120,relation_tokens=2)),data=dict(datasets=dict(synthetic_mri=dict(root='synthetic')))))
        transfer_preixi(model,cp)
        torch.testing.assert_close(model.query_embed.weight[:120],original[:120])
        torch.testing.assert_close(model.query_embed.weight[120:256],fresh[120:256])
        torch.testing.assert_close(model.query_embed.weight[256:],original[120:])


if __name__=='__main__':unittest.main()
