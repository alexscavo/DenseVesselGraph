#!/usr/bin/env python3
"""Fixed-epoch native 2D scratch and mixed plant/vessel pretraining study."""
import argparse
import gc
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path
from zipfile import ZipFile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from PIL import Image
from native2d import build_model
from native2d.config import load_config
from native2d.data import GraphDataset2D,collate,transform
from native2d.losses import GraphCriterion2D
from scripts import train_native2d_study as pilot

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native2d_mixed_20261028'
DATA=Path(json.loads((OUT/'plan.json').read_text())['data_root'])
SEED=364505

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=True)+'\n');tmp.replace(path)

class StudyDataset(GraphDataset2D):
    """Same native augmentation; packed targets and uint8 caching limit Lustre IO."""
    def __init__(self,manifest,split,*,augment=False,max_queries=None,preload=False):
        super().__init__(manifest,split,augment=augment,max_queries=max_queries)
        self.archive=None;self.cache={}
        if preload:
            for i in range(len(self)):
                self.cache[i]=self.raw(i)
                if (i+1)%5000==0:print('CACHE '+str(i+1),flush=True)

    def raw(self,index):
        row=self.rows[index]
        with Image.open(self.root/row['image']) as image:array=np.asarray(image.convert('RGB')).copy()
        assert array.shape==(128,128,3)
        if row.get('target_archive'):
            if self.archive is None:self.archive=ZipFile(self.root/row['target_archive'])
            with np.load(io.BytesIO(self.archive.read(row['target']))) as z:target={k:z[k].copy() for k in z.files}
        else:
            with np.load(self.root/row['target']) as z:target={k:z[k].copy() for k in z.files}
        assert len(target['nodes'])<=self.max_queries and np.isfinite(target['controls']).all()
        return array,target

    def __getitem__(self,index):
        image,target=self.cache[index] if index in self.cache else self.raw(index)
        image=torch.from_numpy((image.astype(np.float32)/255.-.5).transpose(2,0,1).copy())
        if self.augment:image,target=transform(image,target,quarter_turns=int(torch.randint(4,()).item()),flip_x=bool(torch.randint(2,()).item()))
        return image,target,self.rows[index]

class MixedDataset:
    """Pair plants with ONE target cohort; all pretraining geometry is in pixels."""
    def __init__(self,dataset,split,*,augment=False,max_queries=None,preload=False):
        self.plants=StudyDataset(DATA/'plants/manifest.json',split,augment=augment,max_queries=max_queries,preload=preload and split=='train')
        self.vessels=StudyDataset(DATA/dataset/'manifest.json',split,augment=augment,max_queries=max_queries,preload=preload)
        self.rows=self.plants.rows+self.vessels.rows
        self.source_count=len(self.plants)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        image,target,row=self.plants[index] if index<self.source_count else self.vessels[index-self.source_count]
        target={k:v.copy() for k,v in target.items()}
        # D4 rotations/reflections in the metric remain intact. Convert the
        # calibrated vessel frame to pixels without touching the stored arrays.
        if str(target['units'].item())=='micron':
            pitch=np.linalg.norm(target['metric'],axis=0)/np.asarray(target['shape_hw'])[::-1]
            np.testing.assert_allclose(pitch,pitch[0],atol=1e-8,rtol=1e-6)
            target['metric']=target['metric']/pitch[0]
        else:assert str(target['units'].item())=='pixel'
        target['units']=np.array('pixel')
        return image,target,row

def mixed_exposure(source_count,target_count,epochs,seed=SEED):
    # Same probabilities, replacement and twice-source epoch definition as
    # data/loaders/mixed.py's established WeightedRandomSampler protocol.
    rng=np.random.default_rng(seed);shape=(epochs,2*source_count)
    domain=rng.random(shape)<.5
    source=rng.integers(source_count,size=shape,dtype=np.int32)
    target=rng.integers(target_count,size=shape,dtype=np.int32)+source_count
    return np.where(domain,source,target).astype(np.int32)

def config(dataset,variant,*,pretrain=False):
    plan=read(OUT/'plan.json');assert dataset in plan['sources'] and plan['sources'][dataset]['passed']
    if pretrain:
        assert plan['mixed_sampling']['plant_probability']==.5 and plan['mixed_sampling']['epoch_size']=='twice_source' and plan['mixed_sampling']['replacement']
    cfg=load_config(ROOT/'configs/native2d.yaml');cfg['model']['decoder']['object_queries']=plan['common_queries']
    cfg['model']['node_refinement']['enabled']=variant=='Axyz'
    cfg['experiment'].update(name='native2d-long-'+dataset+'-'+variant,seed=SEED)
    batch=32 if pretrain or dataset=='plants' else 4
    cfg['data'].update(input_modality='paired_raw_2d_image',batch_size=batch,image_size=[128,128],manifest=str(DATA/dataset/'manifest.json'),datasets={dataset:dict(manifest=str(DATA/dataset/'manifest.json'))})
    if pretrain:
        cfg['data']['datasets']['plants']=dict(manifest=str(DATA/'plants/manifest.json'))
        cfg['data']['mixed_sampling']=plan['mixed_sampling']
    cfg['training'].update(epochs=50 if pretrain or dataset=='plants' else 100)
    cfg['training']['optimizer'].update(base_lr=2e-4,weight_decay=1e-4)
    cfg['evaluation'].update(node_threshold=.5,edge_threshold=.5)
    cfg['training']['native2d_protocol']=dict(backbone_lr=2e-5,epochs=cfg['training']['epochs'],warmup_updates=100,
        schedule='cosine to10% base LR over the declared full epoch budget',augmentation='deterministic D4 per exposure',
        stopping='exact fixed epochs, stop on nonfinite/error, no automatic extension',test_policy='no test loading or inference',
        finetune_data='vessel cohort only; no plant replay',selection='fixed interim subset edge micro F1 at2 native distance units; latest trained tie',
        source_geometry='supplied sparse plant graph segments; no claim of independent dense manual curves',units='pixel' if pretrain else read(DATA/dataset/'manifest.json')['units'],
        stage='mixed pretraining' if pretrain else 'vessel-only target training',pretraining_geometry='pixel units for both domains; REAVER uses calibrated microns again at finetuning')
    assert cfg['loss']['node']['classification']['name']=='focal' and cfg['loss']['node']['classification']['focal_gamma']==2
    assert cfg['loss']['edge']['balancing']['positive_to_negative_ratio']==.15 and not cfg['loss']['edge']['candidates']['include_unmatched']
    return cfg

def import_shared(model,state):
    missing,extra=model.load_state_dict(state,strict=False)
    assert not extra and all(k.startswith('node_refinement.') for k in missing),(missing,extra)

@torch.no_grad()
def measure(model,x,target,truth,rows,geometry):
    result=pilot.metrics(model,x,target,truth,rows,full_geometry=geometry)
    for r,t in zip(result,truth):r['units']=str(t['units'].item())
    return result

def aggregate(rows):
    result=pilot.aggregate(rows);result['units']=rows[0]['units'];n=sum(r['target_nodes'] for r in rows)
    for t in ['1.0','2.0','4.0']:
        result[t]['raw_node_recall_micro']=sum(round(r['raw_node_recall'][t]*r['target_nodes']) for r in rows)/n if n else None
    positive=[r for r in rows if r['target_edges']>0];result['target_edge_positive_crops']=len(positive)
    if 'spline' in rows[0]:
        for name in ['spline','chord']:
            result[name+'_edge_positive_crop_macro']={k:float(np.mean([r[name][k] for r in positive])) if positive else None for k in ['common_branch_f1','common_curve_f1']}
    return result

def subset(ds,indices=None):
    indices=list(range(len(ds))) if indices is None else indices
    groups=sorted({ds.rows[i]['subject_id'] for i in indices});groups=[groups[i] for i in np.linspace(0,len(groups)-1,min(16,len(groups)),dtype=int)]
    selected=[]
    for parent in groups:
        ids=[i for i in indices if ds.rows[i]['subject_id']==parent]
        selected.extend(np.array(ids)[np.unique(np.linspace(0,len(ids)-1,min(4,len(ids)),dtype=int))].tolist())
    return selected

@torch.no_grad()
def validation(model,ds,ids,geometry):
    model.eval();rows=[]
    for start in range(0,len(ids),4):
        x,t,g,r=collate([ds[i] for i in ids[start:start+4]],'cuda');rows.extend(measure(model,x,t,g,r,geometry))
    return rows

def gates():
    torch.set_num_threads(2);plan=read(OUT/'plan.json');report={};benchmarks={}
    for dataset in ['avwide','reaver','octasynth','plants']:
        if dataset not in plan['sources']:continue
        torch.manual_seed(SEED);cfg=config(dataset,'B0');model=build_model(cfg).cuda().eval();initial=pilot.digest(model)
        ds=StudyDataset(DATA/dataset/'manifest.json','train',max_queries=plan['common_queries'])
        ids=[i for i,r in enumerate(ds.rows) if 4<=r['nodes']<=12 and 2<=r['edges']<=12][:2];assert len(ids)==2
        x,t,g,r=collate([ds[i] for i in ids],'cuda')
        torch.manual_seed(SEED);ref=build_model(config(dataset,'Axyz')).cuda().eval();assert pilot.digest(ref)==initial
        with torch.no_grad():
            a,b=model(x);c,d=ref(x);torch.testing.assert_close(a,c,rtol=0,atol=0);torch.testing.assert_close(b['pred_nodes'],d['pred_nodes'],rtol=0,atol=0);torch.testing.assert_close(b['pred_logits'],d['pred_logits'],rtol=0,atol=0)
        criterion=GraphCriterion2D(config(dataset,'Axyz'),ref);a,b=ref(x);loss=criterion(a,b,t);loss['spline'].backward(retain_graph=True)
        assert all(p.grad is None or not p.grad.any() for k,p in ref.named_parameters() if not k.startswith('spline_head.'))
        ref.zero_grad(set_to_none=True);loss['total'].backward();assert ref.node_refinement.network[-1].weight.grad.abs().sum()>0
        assert all(torch.isfinite(p.grad).all() for p in ref.parameters() if p.grad is not None)
        del ref,criterion,a,b,c,d,loss;torch.cuda.empty_cache()
        criterion=GraphCriterion2D(cfg,model);opt=pilot.optimizer(model);start=time.monotonic()
        for step in range(1200):pilot.backward_step(model,criterion,opt,x,t)
        values=measure(model,x,t,g,r,False);s=aggregate(values);passed=s['2.0']['node']['f1']>=.9 and s['2.0']['edge']['f1']>=.8
        # Strict full-core import from a trained B0 into zero-refinement A-xyz.
        torch.manual_seed(SEED);ref=build_model(config(dataset,'Axyz')).cuda().eval();import_shared(ref,model.state_dict());assert pilot.digest(ref)==pilot.digest(model)
        with torch.no_grad():
            a,b=model(x);c,d=ref(x);torch.testing.assert_close(b['pred_nodes'],d['pred_nodes'],rtol=0,atol=0);torch.testing.assert_close(b['pred_logits'],d['pred_logits'],rtol=0,atol=0)
        report[dataset]=dict(passed=passed,units=s['units'],tiny_updates=1200,tiny_samples=[z['sample_id'] for z in r],tiny_metrics=s,seconds=time.monotonic()-start,initialization_sha256=initial,trained_checkpoint_import_parity=True)
        write(OUT/'gate-progress.json',report);assert passed,dataset+' full-model overfit gate failed; do not extend'
        del ref,a,b,c,d,x,t,g,r,model,criterion,opt;torch.cuda.empty_cache();gc.collect()
        # Representative real batch and production batch size, including loss.
        torch.manual_seed(SEED);model=build_model(cfg).cuda().train();criterion=GraphCriterion2D(cfg,model);opt=pilot.optimizer(model)
        batch=cfg['data']['batch_size'];ids=np.linspace(0,len(ds)-1,batch,dtype=int);x,t,_,_=collate([ds[int(i)] for i in ids],'cuda')
        for _ in range(3):pilot.backward_step(model,criterion,opt,x,t)
        torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for _ in range(12):pilot.backward_step(model,criterion,opt,x,t)
        torch.cuda.synchronize();seconds=(time.monotonic()-start)/12
        benchmarks[dataset]=dict(batch=batch,seconds_per_update=seconds,images_per_second=batch/seconds,peak_bytes=torch.cuda.max_memory_allocated(),estimated_training_hours=math.ceil(len(ds)/batch)*cfg['training']['epochs']*seconds/3600)
        del model,criterion,opt,x,t,ds;torch.cuda.empty_cache();gc.collect()
    for dataset in ['avwide','reaver','octasynth']:
        torch.manual_seed(SEED);cfg=config(dataset,'B0',pretrain=True);model=build_model(cfg).cuda().train();criterion=GraphCriterion2D(cfg,model);opt=pilot.optimizer(model)
        ds=MixedDataset(dataset,'train',max_queries=plan['common_queries'])
        ids=[*np.linspace(0,ds.source_count-1,16,dtype=int),*(ds.source_count+np.linspace(0,len(ds.vessels)-1,16,dtype=int))]
        x,t,truth,rows=collate([ds[int(i)] for i in ids],'cuda')
        assert all(str(g['units'].item())=='pixel' for g in truth)
        for _ in range(3):pilot.backward_step(model,criterion,opt,x,t)
        torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for _ in range(12):pilot.backward_step(model,criterion,opt,x,t)
        torch.cuda.synchronize();seconds=(time.monotonic()-start)/12
        benchmarks['mixed_'+dataset]=dict(batch=32,seconds_per_update=seconds,images_per_second=32/seconds,peak_bytes=torch.cuda.max_memory_allocated(),estimated_training_hours=math.ceil(2*ds.source_count/32)*50*seconds/3600,plant_and_vessel_real_batch=True)
        del model,criterion,opt,x,t,ds;torch.cuda.empty_cache();gc.collect()
    hashes={r['initialization_sha256'] for r in report.values()};assert len(hashes)==1
    write(OUT/'gates.json',dict(passed=True,datasets=report,benchmark=benchmarks,initialization_sha256=next(iter(hashes)),common_queries=plan['common_queries'],no_training_checkpoint_reused=True))
    print('GATES '+json.dumps(benchmarks),flush=True)

def train(dataset,variant,initialization,*,pretrain=False):
    gate=read(OUT/'gates.json');assert gate['passed'] and gate['datasets'][dataset]['passed']
    if pretrain:assert variant=='B0' and initialization=='scratch' and dataset!='plants'
    name=dataset+'_mixedpretrain_50' if pretrain else dataset+'_'+initialization+'_'+variant
    directory=OUT/'runs'/name;directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'latest.pt').exists(),'Fresh run only; preserve existing experiments'
    cfg=config(dataset,variant,pretrain=pretrain);cfg['experiment']['output_dir']=str(directory);cfg['experiment']['initialization']=initialization
    torch.set_num_threads(2);torch.manual_seed(SEED);model=build_model(cfg).cuda()
    if initialization=='mixed':
        p=OUT/'runs'/(dataset+'_mixedpretrain_50')/'final.pt';source=torch.load(p,map_location='cpu')
        assert source['epoch']==50 and source['dataset']==dataset and source['stage']=='mixed_pretraining' and source['completed']
        import_shared(model,source['model']);del source
        cfg['experiment']['initial_checkpoint']=str(p)
    initial=pilot.digest(model)
    if initialization=='scratch':assert initial==gate['initialization_sha256']
    (directory/'resolved-config.yaml').write_text(yaml.safe_dump(cfg))
    if pretrain:
        ds=MixedDataset(dataset,'train',augment=True,max_queries=gate['common_queries'],preload=True)
        val=MixedDataset(dataset,'validation',max_queries=gate['common_queries'],preload=True)
        validation_ids=list(range(val.source_count,len(val)))
    else:
        ds=StudyDataset(DATA/dataset/'manifest.json','train',augment=True,max_queries=gate['common_queries'],preload=True)
        val=StudyDataset(DATA/dataset/'manifest.json','validation',max_queries=gate['common_queries'],preload=True)
        validation_ids=list(range(len(val)))
    selected=subset(val,validation_ids);selected_ids={val.rows[i]['sample_id'] for i in selected}
    epochs=cfg['training']['epochs'];batch=cfg['data']['batch_size']
    rng=np.random.default_rng(SEED)
    exposure=mixed_exposure(ds.source_count,len(ds.vessels),epochs) if pretrain else np.stack([rng.permutation(len(ds)).astype(np.int32) for _ in range(epochs)])
    epoch_draws=exposure.shape[1];per_epoch=math.ceil(epoch_draws/batch);budget=epochs*per_epoch
    exposure_hash=hashlib.sha256(exposure.tobytes()).hexdigest();np.save(directory/'exposure.npy',exposure)
    write(directory/'protocol.json',dict(dataset=dataset,variant=variant,stage='mixed_pretraining' if pretrain else 'vessel_only',initialization=initialization,epochs=epochs,batch=batch,updates=budget,sample_exposures=epochs*epoch_draws,epoch_draws=epoch_draws,train_samples=len(ds),validation_samples=len(validation_ids),validation_interim=selected,units='pixel' if pretrain else gate['datasets'][dataset]['units'],initialization_sha256=initial,exposure_sha256=exposure_hash,manifest_sha256=hashlib.sha256((DATA/dataset/'manifest.json').read_bytes()).hexdigest(),thresholds=dict(node=.5,edge=.5),mixed_sampling=read(OUT/'plan.json')['mixed_sampling'] if pretrain else None,realized_plant_draws_per_epoch=(exposure<ds.source_count).sum(1).tolist() if pretrain else None))
    criterion=GraphCriterion2D(cfg,model);opt=pilot.optimizer(model);best=-1.;curve=[];step=0;start=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();totals={};seen=0
        for offset in range(0,epoch_draws,batch):
            torch.manual_seed(SEED+step);torch.cuda.manual_seed_all(SEED+step)
            x,t,_,_=pilot.batch_for(ds,exposure[epoch-1,offset:offset+batch],step)
            fraction=(step-100)/max(1,budget-100);factor=(step+1)/100 if step<100 else .1+.9*(1+math.cos(math.pi*fraction))/2
            for group,base in zip(opt.param_groups,[2e-5,2e-4]):group['lr']=base*factor
            _,_,loss,norm=pilot.backward_step(model,criterion,opt,x,t);n=len(x);seen+=n
            for k,v in loss.items():totals[k]=totals.get(k,0.)+float(v.detach())*n
            step+=1
        assert seen==epoch_draws
        row=dict(epoch=epoch,step=step,elapsed=time.monotonic()-start,lr=opt.param_groups[-1]['lr'],gradient_norm=norm,losses={k:v/seen for k,v in totals.items()})
        with (directory/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('EPOCH '+json.dumps(row),flush=True)
        write(directory/'status.json',dict(state='training',epoch=epoch,epochs=epochs,step=step,budget_updates=budget,elapsed=time.monotonic()-start))
        if epoch%5 and epoch!=epochs:continue
        ids=validation_ids if epoch==epochs else selected
        values=validation(model,val,ids,geometry=not pretrain);summary=aggregate(values)
        evaluation=dict(epoch=epoch,step=step,summary=summary,samples=values)
        write(directory/f'validation-epoch{epoch:03d}.json',evaluation)
        curve.append(dict(epoch=epoch,step=step,elapsed=time.monotonic()-start,validation=summary));write(directory/'learning-curve.json',curve)
        print('VALIDATION '+json.dumps(curve[-1]),flush=True)
        score=aggregate([r for r in values if r['sample_id'] in selected_ids])['2.0']['edge']['f1']
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),config=cfg,dataset=dataset,variant=variant,stage='mixed_pretraining' if pretrain else 'vessel_only',initialization=initialization,epoch=epoch,epochs=epochs,step=step,budget=budget,completed=epoch==epochs,seed=SEED,initialization_sha256=initial,exposure_sha256=exposure_hash)
        torch.save(state,directory/'latest.pt')
        if score>=best:
            best=score;torch.save(state,directory/'best.pt');write(directory/'best.json',dict(epoch=epoch,score=score))
        if epoch==epochs:
            # Same final bytes, no second full-state GPU serialization.
            import shutil
            shutil.copy2(directory/'latest.pt',directory/'final.pt')
            write(directory/'final-heldout.json',evaluation)
    assert step==budget
    best_meta=read(directory/'best.json')
    if best_meta['epoch']==epochs:
        write(directory/'best-heldout.json',read(directory/'final-heldout.json'))
    else:
        state=torch.load(directory/'best.pt',map_location='cpu');model.load_state_dict(state['model']);del state
        values=validation(model,val,validation_ids,geometry=not pretrain)
        write(directory/'best-heldout.json',dict(epoch=best_meta['epoch'],summary=aggregate(values),samples=values))
    write(directory/'status.json',dict(state='complete',epoch=epochs,epochs=epochs,step=step,budget_updates=budget,elapsed=time.monotonic()-start,initialization_sha256=initial,exposure_sha256=exposure_hash))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gates',action='store_true');p.add_argument('--pretrain-mixed',action='store_true');p.add_argument('--dataset',choices=['avwide','reaver','octasynth']);p.add_argument('--variant',choices=['B0','Axyz'],default='B0');p.add_argument('--initialization',choices=['scratch','mixed'],default='scratch');a=p.parse_args()
    try:gates() if a.gates else train(a.dataset,a.variant,a.initialization,pretrain=a.pretrain_mixed)
    except Exception as e:
        write(OUT/('gates-failure.json' if a.gates else (a.dataset+'_mixedpretrain_50' if a.pretrain_mixed else a.dataset+'_'+a.initialization+'_'+a.variant)+'-failure.json'),dict(error=repr(e),policy='stop; no automatic extension or retry'));raise
