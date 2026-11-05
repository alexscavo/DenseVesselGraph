#!/usr/bin/env python3
"""Bounded original-width B0/A-xyz study. Gates precede both training arms."""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from native2d import build_model
from native2d.config import load_config
from native2d.data import GraphDataset2D,collate
from native2d.losses import GraphCriterion2D
from native2d.evaluation import decode,evaluate,relation_ranking,positional,common_metrics,_boxes
from training.evaluation.metrics import detection_ap_ar,graph_betti_numbers

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('NATIVE2D_STUDY_DIR',str(ROOT/'experiments/native2d_20261028')))
SEED=364505

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=True)+'\n');tmp.replace(path)

def config(arm):
    cfg=load_config(ROOT/'configs/native2d.yaml')
    audit=json.loads((OUT/'av-wide-audit.json').read_text());assert audit['passed']
    cfg['model']['decoder']['object_queries']=audit['queries']
    cfg['model']['node_refinement']['enabled']=arm=='Axyz'
    cfg['experiment'].update(name='native2d-wide-'+arm,seed=SEED)
    cfg['data'].update(batch_size=4,image_size=[128,128],manifest=str(OUT/'manifest.json'))
    cfg['training']['optimizer'].update(base_lr=2e-4,weight_decay=1e-4)
    cfg['training']['native2d_protocol']=dict(backbone_lr=2e-5,batch_size=4,max_updates=6000,
        eval_interval=1000,augmentation='all D4 rotations/flips, deterministic per exposure',
        lr_schedule='linear warmup 100 updates then cosine decay to 10 percent',
        stopping='fixed budget <=6000 updates, <=40 train passes, benchmark-derived <=30min training per arm; no extension',
        selection='best validation micro edge F1 at 2px; report final and best, identical paired selection rule',
        confidence_thresholds=dict(node=.5,edge=.5),pixel_tolerances=[1,2,4],common_v2='1px anchor/curve, .25px samples',
        test_policy='no test loading or evaluation')
    assert cfg['loss']['node']['classification']['name']=='focal'
    assert cfg['loss']['node']['classification']['focal_gamma']==2
    assert cfg['loss']['edge']['balancing']['positive_to_negative_ratio']==.15
    assert not cfg['loss']['edge']['candidates']['include_unmatched']
    return cfg

def optimizer(model):
    backbone=[];other=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(backbone if name.startswith('encoder.0.body.') else other).append(p)
    return torch.optim.AdamW([dict(params=backbone,lr=2e-5),dict(params=other,lr=2e-4)],weight_decay=1e-4)

def digest(model,shared_only=True):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        if shared_only and name.startswith('node_refinement.'):continue
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def batch_for(dataset,indices,step,device='cuda',augment=True):
    # Dataset's exact D4 transform draws only CPU RNG. Regenerate by exposure.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED+100000+step)
        samples=[dataset[int(i)] for i in indices]
    return collate(samples,device)

def backward_step(model,criterion,opt,x,target):
    opt.zero_grad(set_to_none=True)
    tokens,pred=model(x);loss=criterion(tokens,pred,target)
    assert all(torch.isfinite(v).all() for v in loss.values()),'Nonfinite loss: stop'
    loss['total'].backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
    opt.step()
    return tokens,pred,loss,float(norm)

@torch.no_grad()
def metrics(model,x,targets,truth,rows,*,full_geometry=True):
    tokens,pred=model(x);values=[]
    for b,(gt,row) in enumerate(zip(truth,rows)):
        decoded=decode(model,tokens,pred,b,targets['spline'][b]['metric'])
        m=np.asarray(gt['metric']);gn=gt['nodes'];ge=gt['edges']
        inserted=gt.get('inserted')
        result={str(t):positional(decoded['nodes']@m.T,decoded['edges'],gn@m.T,ge,t,inserted=inserted) for t in [1.,2.,4.]}
        raw=pred['pred_nodes'][b,:,:2].cpu().numpy();scores=pred['pred_logits'][b].softmax(-1)[:,1].cpu().numpy()
        raw_boxes=pred['pred_nodes'][b].cpu().numpy();pairs=torch.combinations(torch.arange(len(raw),device=x.device),r=2)
        rel=model.relation_logits(tokens[b],pairs).softmax(-1)[:,1].cpu().numpy()
        node_ap,node_ar=detection_ap_ar(_boxes(raw,sizes=raw_boxes[:,2:]),scores,_boxes(gn),max_detections=len(raw))
        edge_ap,edge_ar=detection_ap_ar(_boxes(raw,pairs.cpu().numpy()),rel,_boxes(gn,ge),max_detections=300)
        p0,p1=graph_betti_numbers(len(decoded['nodes']),decoded['edges']);g0,g1=graph_betti_numbers(len(gn),ge)
        result.update(sample_id=row['sample_id'],source_image=row['subject_id'],units='pixel',node_mAP=node_ap,node_mAR=node_ar,
            edge_mAP=edge_ap,edge_mAR=edge_ar,predicted_nodes=len(decoded['nodes']),target_nodes=len(gn),
            predicted_edges=len(decoded['edges']),target_edges=len(ge),predicted_beta0=p0,target_beta0=g0,predicted_beta1=p1,target_beta1=g1,
            beta0_signed_error=p0-g0,beta1_signed_error=p1-g1,
            raw_node_recall={str(t):positional(raw@m.T,np.empty((0,2),int),gn@m.T,ge,t,inserted=inserted)['node']['recall'] for t in [1.,2.,4.]},
            ranking=relation_ranking(model,tokens,pred,b,gt))
        if full_geometry:
            result['spline']=common_metrics(decoded,gt)
            chord={k:v for k,v in decoded.items() if k!='spline_controls'}
            result['chord']=common_metrics(chord,gt)
            assert result['spline']['common_branch_f1']==result['chord']['common_branch_f1']
        values.append(result)
    return values

def aggregate(rows):
    output=dict(samples=len(rows),units='pixel')
    for t in ['1.0','2.0','4.0']:
        output[t]={}
        for kind in ['node','edge']:
            totals={key:sum(r[t][kind][key] for r in rows) for key in ['tp','fp','fn']}
            tp,fp,fn=(totals[k] for k in ['tp','fp','fn']);totals.update(f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                recall=tp/(tp+fn) if tp+fn else 1.,precision=tp/(tp+fp) if tp+fp else float(not fn))
            output[t][kind]=totals
        output[t]['duplicates']=sum(r[t]['duplicates'] for r in rows)
        eligible=sum(r[t]['gt_edges_both_endpoints_recovered'] for r in rows)
        output[t]['conditional_relation_recall']=output[t]['edge']['tp']/eligible if eligible else None
        output[t]['raw_node_recall_macro']=float(np.mean([r['raw_node_recall'][t] for r in rows]))
    for key in ['node_mAP','node_mAR','edge_mAP','edge_mAR','beta0_signed_error','beta1_signed_error']:
        a=np.array([r[key] for r in rows]);output[key+'_crop_macro']=float(np.nanmean(a)) if np.isfinite(a).any() else None
    for name in ['spline','chord']:
        if name not in rows[0]:continue
        output[name]={k:float(np.mean([r[name][k] for r in rows])) for k in ['common_branch_f1','common_curve_f1']}
    return output

def gates():
    torch.set_num_threads(2)
    torch.manual_seed(SEED);cfg=config('B0')
    ds=GraphDataset2D(OUT/'manifest.json','train',max_queries=cfg['model']['decoder']['object_queries'])
    ids=[i for i,r in enumerate(ds.rows) if 4<=r['nodes']<=10 and 2<=r['edges']<=12][:2]
    assert len(ids)==2
    x,target,gt,rows=collate([ds[i] for i in ids],'cuda')
    model=build_model(cfg).cuda().eval();shared=digest(model)
    torch.manual_seed(SEED);ref=build_model(config('Axyz')).cuda().eval()
    assert digest(ref)==shared,'Shared initialization differs'
    with torch.no_grad():
        t,p=model(x);rt,rp=ref(x)
        torch.testing.assert_close(t,rt,rtol=0,atol=0)
        torch.testing.assert_close(p['pred_logits'],rp['pred_logits'],rtol=0,atol=0)
        torch.testing.assert_close(p['pred_nodes'],rp['pred_nodes'],rtol=0,atol=0)
    del t,p,rt,rp
    criterion=GraphCriterion2D(config('Axyz'),ref)
    t,p=ref(x);loss=criterion(t,p,target);loss['spline'].backward(retain_graph=True)
    assert all(v.grad is None or not v.grad.any() for k,v in ref.named_parameters() if not k.startswith('spline_head.'))
    ref.zero_grad(set_to_none=True);loss['total'].backward()
    assert ref.node_refinement.network[-1].weight.grad.abs().sum()>0
    assert all(torch.isfinite(v.grad).all() for v in ref.parameters() if v.grad is not None)
    del ref,criterion,t,p,loss;torch.cuda.empty_cache()
    # Train the full model on two real 128-square tiles for a fixed 1200 updates.
    # Disable dropout only for this memorization gate. No held-out selection.
    model.eval();criterion=GraphCriterion2D(cfg,model);opt=optimizer(model)
    history=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step in range(1201):
        if step in [0,100,400,800,1200]:
            with torch.no_grad():values=metrics(model,x,target,gt,rows,full_geometry=False)
            item=dict(step=step,elapsed=time.monotonic()-start,metrics=aggregate(values));history.append(item)
            print('FULL_OVERFIT '+json.dumps(item),flush=True)
            write(OUT/'full-overfit-curve.json',history)
        if step==1200:break
        backward_step(model,criterion,opt,x,target)
    final=history[-1]['metrics'];passed=final['2.0']['node']['f1']>=.9 and final['2.0']['edge']['f1']>=.8
    write(OUT/'full-overfit.json',dict(passed=passed,history=history,shape=list(x.shape),parameters=sum(p.numel() for p in model.parameters()),
        sample_ids=[r['sample_id'] for r in rows],updates=1200,peak_bytes=torch.cuda.max_memory_allocated(),shared_initialization_sha256=shared))
    assert passed,'Full original-width memorization gate failed: do not train or extend'
    # Benchmark actual training mode/batch, separately from evaluation and loading.
    del model,criterion,opt;torch.cuda.empty_cache()
    # Time the unchanged native 2D attention core independently, at the real
    # encoder token dimensions (16² + 8² + 4² + 2², 8 heads, width 512).
    from native2d.attention import ms_deform_attn_core_pytorch
    shapes=torch.tensor([[16,16],[8,8],[4,4],[2,2]],device='cuda')
    value=torch.randn(4,340,8,64,device='cuda',requires_grad=True)
    locations=torch.rand(4,340,8,4,4,2,device='cuda',requires_grad=True)
    weights=torch.full((4,340,8,4,4),1/16,device='cuda',requires_grad=True)
    for _ in range(3):
        ms_deform_attn_core_pytorch(value,shapes,locations,weights).square().mean().backward()
        for v in [value,locations,weights]:v.grad=None
    torch.cuda.synchronize();attention_start=time.monotonic()
    for _ in range(20):
        ms_deform_attn_core_pytorch(value,shapes,locations,weights).square().mean().backward()
        for v in [value,locations,weights]:v.grad=None
    torch.cuda.synchronize();attention_seconds=(time.monotonic()-attention_start)/20
    del value,locations,weights
    benchmark=[]
    for arm in ['B0','Axyz']:
        torch.manual_seed(SEED);model=build_model(config(arm)).cuda().train();criterion=GraphCriterion2D(config(arm),model);opt=optimizer(model)
        xb,tb,_,_=collate([ds[i] for i in ids]*2,'cuda')
        for _ in range(3):backward_step(model,criterion,opt,xb,tb)
        torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for _ in range(12):backward_step(model,criterion,opt,xb,tb)
        torch.cuda.synchronize();seconds=(time.monotonic()-start)/12
        benchmark.append(dict(arm=arm,seconds_per_update=seconds,images_per_second=4/seconds,
            peak_bytes=torch.cuda.max_memory_allocated(),batch=4,input_shape=list(xb.shape),attention='existing PyTorch grid_sample implementation'))
        del model,criterion,opt,xb,tb;torch.cuda.empty_cache()
    worst=max(r['seconds_per_update'] for r in benchmark)
    # Equal fixed updates, bounded by data exposure and measured throughput.
    updates=min(6000,40*math.ceil(len(ds)/4),int(1800/worst))
    updates=max(1,(updates//100)*100)
    assert updates>=1000,'Full model too slow for bounded pilot'
    record=dict(passed=True,benchmark=benchmark,attention_core_forward_backward_seconds=attention_seconds,
        attention_core_shape=dict(batch=4,tokens=340,heads=8,head_width=64,levels=4,points=4),
        updates=updates,max_epochs=40,train_walltime_budget_seconds=1800,
        initialization_sha256=shared,full_architecture_optimizer_updates=1200,
        gate='correctness + exact shared initialization + graph_only gradients + 128-square full-model overfit + short GPU throughput')
    write(OUT/'gates.json',record);print('GATES '+json.dumps(record),flush=True)

@torch.no_grad()
def validation(model,ds,indices,*,full_geometry=True):
    result=[];model.eval()
    for start in range(0,len(indices),4):
        x,target,gt,rows=collate([ds[i] for i in indices[start:start+4]],'cuda')
        result.extend(metrics(model,x,target,gt,rows,full_geometry=full_geometry))
    return result

def train(arm):
    gate=json.loads((OUT/'gates.json').read_text());assert gate['passed']
    directory=OUT/arm;directory.mkdir(exist_ok=True)
    assert not (directory/'latest.pt').exists(),'Do not silently resume or overwrite a completed arm'
    cfg=config(arm);(directory/'resolved-config.yaml').write_text(yaml.safe_dump(cfg))
    torch.set_num_threads(2);torch.manual_seed(SEED)
    model=build_model(cfg).cuda();assert digest(model)==gate['initialization_sha256']
    criterion=GraphCriterion2D(cfg,model);opt=optimizer(model)
    ds=GraphDataset2D(OUT/'manifest.json','train',augment=True,max_queries=cfg['model']['decoder']['object_queries'])
    val=GraphDataset2D(OUT/'manifest.json','validation',max_queries=cfg['model']['decoder']['object_queries'])
    # Uniform spatial samples per validation image, fixed before training.
    interim=[]
    for image in sorted({r['subject_id'] for r in val.rows}):
        ids=[i for i,r in enumerate(val.rows) if r['subject_id']==image]
        interim.extend(np.array(ids)[np.linspace(0,len(ids)-1,4,dtype=int)].tolist())
    write(directory/'validation-selection.json',dict(interim=interim,final='every validation tile',no_test=True))
    budget=gate['updates'];rng=np.random.default_rng(SEED);exposure=[]
    while len(exposure)<budget*4:exposure.extend(rng.permutation(len(ds)).tolist())
    schedule=np.array(exposure[:budget*4]).reshape(budget,4)
    schedule_hash=hashlib.sha256(schedule.tobytes()).hexdigest();np.save(directory/'exposure.npy',schedule)
    start=time.monotonic();best=-1.;curve=[];train_loss=[]
    for step in range(budget+1):
        if step==0 or step%1000==0 or step==budget:
            indices=list(range(len(val))) if step==budget else interim
            # Random update-zero graphs can contain thousands of edges. Keep
            # detection/topology here; start costly curve metrics at update1000.
            values=validation(model,val,indices,full_geometry=step>0);summary=aggregate(values)
            write(directory/f'validation-{step:05d}.json',dict(step=step,summary=summary,samples=values))
            item=dict(step=step,elapsed=time.monotonic()-start,validation=summary);curve.append(item)
            write(directory/'learning-curve.json',curve);print('VALIDATION '+json.dumps(item),flush=True)
            # Keep checkpoint selection on the same interim population even at final.
            selected=aggregate([r for r in values if r['sample_id'] in {val.rows[i]['sample_id'] for i in interim}])
            score=selected['2.0']['edge']['f1']
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),config=cfg,step=step,
                exposure_sha256=schedule_hash,initialization_sha256=gate['initialization_sha256'],budget=budget,
                seed=SEED,checkpoint_selection='interim24 validation micro edge F1 @2px, fixed .5 thresholds')
            torch.save(state,directory/'latest.pt')
            # Initialization is a diagnostic, never a trained candidate. Resolve
            # exact ties toward the later trained checkpoint deterministically.
            if step>0 and score>=best:
                best=score;torch.save(state,directory/'best.pt')
                write(directory/'best.json',dict(step=step,score=score))
            write(directory/'status.json',dict(state='complete' if step==budget else 'training',step=step,budget=budget,
                elapsed=time.monotonic()-start,exposure_sha256=schedule_hash,initialization_sha256=gate['initialization_sha256']))
        if step==budget:break
        model.train();torch.manual_seed(SEED+step);torch.cuda.manual_seed_all(SEED+step)
        x,target,_,_=batch_for(ds,schedule[step],step)
        fraction=(step-100)/max(1,budget-100)
        factor=(step+1)/100 if step<100 else .1+.9*(1+math.cos(math.pi*fraction))/2
        for group,base in zip(opt.param_groups,[2e-5,2e-4]):group['lr']=base*factor
        _,_,loss,norm=backward_step(model,criterion,opt,x,target)
        if (step+1)%100==0:
            row=dict(step=step+1,elapsed=time.monotonic()-start,gradient_norm=norm,lr=opt.param_groups[-1]['lr'],
                losses={k:float(v.detach()) for k,v in loss.items()})
            with (directory/'training.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print('TRAIN '+json.dumps(row),flush=True)
    # Evaluate the selected checkpoint on ALL validation tiles, without test use.
    selected=torch.load(directory/'best.pt',map_location='cpu');model.load_state_dict(selected['model'])
    values=validation(model,val,list(range(len(val))))
    write(directory/'best-heldout.json',dict(step=selected['step'],summary=aggregate(values),samples=values))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['gates','B0','Axyz']);args=p.parse_args()
    try:
        gates() if args.mode=='gates' else train(args.mode)
    except Exception as error:
        write(OUT/(args.mode+'-failure.json'),dict(error=repr(error),policy='stop; no automatic retry, budget extension or sweep'))
        raise

if __name__=='__main__':main()
