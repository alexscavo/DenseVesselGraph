"""Original native2D training loop, with only spline prediction/supervision disabled."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from native2d import build_model
from native2d.data import collate
from native2d.losses import GraphCriterion2D
from scripts import train_native2d_long as original

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=original.OUT
OUT=ROOT/'experiments/native2d_straight_20261029'
original_config=original.config

def config(dataset,variant,*,pretrain=False):
    cfg=original_config(dataset,variant,pretrain=pretrain)
    cfg['model']['spline']['enabled']=False
    cfg['loss']['enabled']=[key for key in cfg['loss']['enabled'] if key!='spline']
    cfg['loss']['weights'].pop('spline',None)
    cfg['experiment']['name']='native2d-straight-'+dataset+'-'+variant
    cfg['experiment']['edge_geometry']='straight endpoint chords'
    return cfg

def core_digest(model,shared_only=True):
    """Common detector initialization hash; a straight model has no spline weights."""
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith(('spline_head.','node_refinement.')):continue
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def setup():
    # Reuse the completed study's exact train loop, datasets, sampler, schedule,
    # validation and checkpoint handling. These bindings apply to this process only.
    original.OUT=OUT
    original.config=config
    original.pilot.digest=core_digest

def smoke():
    torch.set_num_threads(2);torch.manual_seed(original.SEED)
    cfg=config('octasynth','B0');model=build_model(cfg).cuda().eval()
    assert model.spline_head is None and 'spline' not in cfg['loss']['enabled']
    initial=core_digest(model)
    torch.manual_seed(original.SEED);reference=build_model(original_config('octasynth','B0')).cpu()
    assert core_digest(reference)==initial;del reference;gc.collect()
    old=original.read(REFERENCE/'gates.json')
    ds=original.StudyDataset(original.DATA/'octasynth/manifest.json','train',max_queries=96)
    names=set(old['datasets']['octasynth']['tiny_samples']);ids=[i for i,row in enumerate(ds.rows) if row['sample_id'] in names]
    assert len(ids)==2
    x,t,g,r=collate([ds[i] for i in ids],'cuda')
    criterion=GraphCriterion2D(cfg,model);optimizer=original.pilot.optimizer(model)
    for _ in range(1200):original.pilot.backward_step(model,criterion,optimizer,x,t)
    result=original.aggregate(original.measure(model,x,t,g,r,False))
    assert result['2.0']['node']['f1']>=.9 and result['2.0']['edge']['f1']>=.8,'Fixed tiny-overfit gate failed; do not extend'
    del model,criterion,optimizer,x,t;torch.cuda.empty_cache();gc.collect()
    for pretrain in [True,False]:
        torch.manual_seed(original.SEED);cfg=config('octasynth','B0',pretrain=pretrain);model=build_model(cfg).cuda().train()
        if pretrain:
            ds=original.MixedDataset('octasynth','train',augment=True,max_queries=96)
            ids=[*np.linspace(0,ds.source_count-1,16,dtype=int),*(ds.source_count+np.linspace(0,len(ds.vessels)-1,16,dtype=int))]
        else:
            ds=original.StudyDataset(original.DATA/'octasynth/manifest.json','train',augment=True,max_queries=96)
            ids=np.linspace(0,len(ds)-1,4,dtype=int)
        x,t,g,r=collate([ds[int(i)] for i in ids],'cuda');criterion=GraphCriterion2D(cfg,model);optimizer=original.pilot.optimizer(model)
        for _ in range(5):
            _,_,loss,_=original.pilot.backward_step(model,criterion,optimizer,x,t)
            assert 'spline' not in loss and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        del model,criterion,optimizer,x,t,ds;torch.cuda.empty_cache();gc.collect()
    gate=dict(old,initialization_sha256=initial,geometry='straight',straight_full_model_smoke=True,straight_tiny_metrics=result,reference=str(REFERENCE))
    original.write(OUT/'gates.json',gate)
    print('STRAIGHT GATES PASSED '+json.dumps(result['2.0']),flush=True)

def train(pretrain):
    gate=original.read(OUT/'gates.json');assert gate['straight_full_model_smoke']
    original.train('octasynth','B0','scratch' if pretrain else 'mixed',pretrain=pretrain)
    name='octasynth_mixedpretrain_50' if pretrain else 'octasynth_mixed_B0'
    actual=original.read(OUT/'runs'/name/'protocol.json');reference=original.read(REFERENCE/'runs'/name/'protocol.json')
    for key in ['exposure_sha256','epochs','batch','updates','sample_exposures','manifest_sha256']:
        assert actual[key]==reference[key],key
    original.write(OUT/('parity-'+name+'.json'),dict(passed=True,reference=str(REFERENCE/'runs'/name),matched_fields=['exposure_sha256','epochs','batch','updates','sample_exposures','manifest_sha256'],geometry='straight; no spline head/loss'))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','pretrain','finetune']);a=p.parse_args();setup()
    try:smoke() if a.stage=='smoke' else train(a.stage=='pretrain')
    except Exception as error:
        original.write(OUT/(a.stage+'-failure.json'),dict(error=repr(error),stopping='stop on error; no automatic extensions or retries'));raise

if __name__=='__main__':main()
