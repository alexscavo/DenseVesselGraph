#!/usr/bin/env python3
"""Native2D long recipe with the original GrEVe ImageNet ResNet101 initialization.

The native2D port set init_weights=None, which leaves a random ResNet101 whose
FrozenBatchNorm2d layers are identities (no normalization) and trains it with the
2e-5 backbone LR intended for a pretrained backbone. This arm restores the original
GrEVe choice (ResNet101_Weights + ImageNet mean/std) and changes nothing else:
same crops, D4 augmentation, losses, queries, optimizer, schedule and epochs as
experiments/native2d_mixed_20261028 vessel-only B0/Axyz runs.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from native2d import build_model
from scripts import train_native2d_long as L

NEW_OUT=L.ROOT/'experiments/native2d_imagenet_20261030'
MEAN=torch.tensor([.485,.456,.406]).view(3,1,1);STD=torch.tensor([.229,.224,.225]).view(3,1,1)

base_config=L.config
def config(dataset,variant,*,pretrain=False):
    cfg=base_config(dataset,variant,pretrain=pretrain)
    cfg['model']['encoder']['init_weights']='ResNet101_Weights'
    cfg['experiment']['name']='native2d-imagenet-'+dataset+'-'+variant
    cfg['training']['native2d_protocol']['backbone_initialization']='torchvision ResNet101_Weights.DEFAULT, ImageNet mean/std input'
    return cfg
L.config=config

base_getitem=L.StudyDataset.__getitem__
def getitem(self,index):
    image,target,row=base_getitem(self,index)
    # Base maps uint8 to x/255-0.5 before D4; undo the shift and apply ImageNet statistics.
    return ((image+.5)-MEAN)/STD,target,row
L.StudyDataset.__getitem__=getitem

def run(dataset,variant):
    # Read the passed gates/plan from the original study; write runs to a new folder.
    original=L.OUT;NEW_OUT.mkdir(parents=True,exist_ok=True)
    for name in ['plan.json','gates.json']:(NEW_OUT/name).write_text((original/name).read_text())
    L.OUT=NEW_OUT
    L.train(dataset,variant,'imagenet')
    # Training-set fit: fixed subject-stratified training subset, no augmentation.
    directory=NEW_OUT/'runs'/(dataset+'_imagenet_'+variant)
    cfg=config(dataset,variant);torch.manual_seed(L.SEED);model=build_model(cfg).cuda()
    model.load_state_dict(torch.load(directory/'final.pt',map_location='cpu')['model'])
    ds=L.StudyDataset(L.DATA/dataset/'manifest.json','train',max_queries=read_gate()['common_queries'])
    ids=L.subset(ds);values=L.validation(model,ds,ids,geometry=False)
    L.write(directory/'final-train-subset.json',dict(sample_ids=[ds.rows[i]['sample_id'] for i in ids],summary=L.aggregate(values),samples=values))
    print('TRAIN_SUBSET '+json.dumps(L.aggregate(values)['2.0']),flush=True)

def read_gate():return L.read(NEW_OUT/'gates.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['avwide','reaver','octasynth'],required=True)
    p.add_argument('--variant',choices=['B0','Axyz'],default='B0');a=p.parse_args()
    run(a.dataset,a.variant)
