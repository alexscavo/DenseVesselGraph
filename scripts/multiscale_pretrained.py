"""Matched, bounded pretrained single/two-level RelationFormer experiment."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.multiscale_migration import migrate_single_level, set_fine_enabled
from models.ops.modules import MSDeformAttn
from scripts.node_verifier_metrics import association, measure, aggregate
from scripts.node_verifier_feasibility import write

ROOT=Path('experiments/multiscale_pretrained')
SEED=231123


def seed(step=0):
    random.seed(SEED+step);np.random.seed(SEED+step);torch.manual_seed(SEED+step)


def prepare():
    from models.checkpoint import load_checkpoint_file
    source=json.loads(Path('experiments/multidataset_continuous/plan.json').read_text())
    checkpoint=source['checkpoints']['synthetic_spline'];cp=load_checkpoint_file(checkpoint['path'])
    spec=source['datasets']['SyntheticMRI'];train=set(spec['subjects'][s.split('_')[1]] for s in spec['train_ids']);val=set(spec['subjects'][s.split('_')[1]] for s in spec['val_ids']);assert not train&val
    cfg=copy.deepcopy(cp['config'])
    cfg['loss']['node']['classification'].update(name='focal',class_weights=[1.,1.],focal_gamma=2.)
    cfg['loss']['node']['classification']['curriculum']['enabled']=False
    cfg['loss']['edge']['classification'].update(name='cross_entropy',class_weights=[1.,1.])
    cfg['loss']['edge']['classification']['curriculum']['enabled']=False
    cfg['loss']['edge']['balancing'].update(mode='ratio_upsample',positive_to_negative_ratio=.15,tolerance=.02)
    assert not cfg['loss']['edge']['candidates']['include_unmatched']
    cfg['data']['datasets']={'synthetic_mri':dict(cfg['data']['datasets']['synthetic_mri'],root=spec['root'],train_samples=None,validation_samples=None,patch_selection='all',spline_controls=8,spline_family='bspline')}
    cfg['data'].update(input_modality='raw_mra',batch_size=8,validation_batch_size=1,train_augmentation=True)
    cfg['experiment'].update(seed=SEED,name='multiscale_pretrained')
    cfg['runtime']['workers']=0;cfg['tracking']['enabled']=False;cfg['domain_adaptation']['enabled']=False
    # Retain K8 geometry, its loss weight and the original graph_only route.
    assert cfg['model']['spline']['controls']==8
    assert not cfg['model'].get('node_refinement',{}).get('enabled',False)
    ids=[];seen=set()
    for s in spec['monitor_train']:
        subject=spec['subjects'][s.split('_')[1]]
        with np.load(Path(spec['root'])/'train'/'spline'/(s+'.npz')) as g:
            if 5<=len(g['nodes'])<=25 and len(g['edges'])>=3 and subject not in seen:
                ids.append(s);seen.add(subject)
        if len(ids)==8:break
    assert len(ids)==8
    plan=dict(checkpoint=checkpoint,source_recipe=cp['config']['loss'],source_best_epoch=cp.get('best_epoch'),source_validation_selection='Existing MRI-only spline best: validation Branch F1 then native edge F1; no new test selection',dataset=spec,config=cfg,seed=SEED,updates=2000,batch_size=8,learning_rate=7e-5,schedule='constant; fresh identical AdamW optimizers',checkpoints=[0,250,1000,2000],fine_prior=1e-4,tiny_ids=ids,tiny_updates=512,tiny_batch_size=4,gradient_route='graph_only',max_h100=2)
    write(ROOT/'plan.json',plan);print('prepared',len(spec['train_ids']),len(spec['val_ids']),flush=True)


def setup(arm,augment=True):
    from models import build_model
    from models.checkpoint import load_checkpoint_file
    from models.matcher import build_matcher
    from scripts.train_ixi_transfer import TransferCriterion
    from training.optim import build_optimizer
    from data.loaders import build_data_loaders
    from configs import validate_config
    plan=json.loads((ROOT/'plan.json').read_text());cfg=copy.deepcopy(plan['config']);cfg['model']['decoder']['feature_levels']=1 if arm=='B0' else 2
    cfg['data']['train_augmentation']=augment;validate_config(cfg);seed()
    cp=load_checkpoint_file(plan['checkpoint']['path']);model=build_model(cfg)
    if arm=='B0':model.load_state_dict(cp['net'],strict=True);migration=dict(strict_single_level=True)
    else:migration=migrate_single_level(model,cp['net'],plan['fine_prior'])
    del cp
    model.cuda().train().requires_grad_(True)
    # Retain the existing disabled legacy relation-attention sub-block.
    for layer in model.decoder.decoder.layers:
        if layer.rln_attn:layer.cross_attn2.requires_grad_(False);layer.norm12.requires_grad_(False)
    for m in model.modules():
        if isinstance(m,MSDeformAttn):m.record_level_diagnostics=True
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda();criterion.set_training_progress(1,0.)
    opt=build_optimizer(cfg,model)
    loader,_=build_data_loaders(cfg);dataset=loader.dataset.datasets[0];lookup={r.sample_id:r for r in dataset.records};dataset.records=[lookup[s] for s in plan['dataset']['train_ids']]
    return plan,cfg,model,criterion,opt,dataset,loader.collate_fn,migration


def levels(model):
    return {n:dict(mass=m.last_level_mass.cpu().tolist(),inside_mass=m.last_inside_mass.cpu().tolist()) for n,m in model.named_modules() if isinstance(m,MSDeformAttn) and hasattr(m,'last_level_mass')}


def gradients(model):
    out={}
    for group in ['encoder.','decoder.','class_embed.','coord_embed.','relation_embed.','spline_head.','input_proj.','fine_input_proj.']:
        values=[p.grad for n,p in model.named_parameters() if n.startswith(group) and p.grad is not None]
        if values:out[group]=float(sum(g.detach().square().sum() for g in values).sqrt())
    out['new_attention_rows']={}
    for n,m in model.named_modules():
        if isinstance(m,MSDeformAttn) and m.n_levels==2:
            out['new_attention_rows'][n]={f:float(getattr(m,f).weight.grad.reshape(m.n_heads,2,-1)[:,1].norm()) for f in ['sampling_offsets','attention_weights']}
    return out


def graph_cache(model,criterion,tokens,pred,targets,b=0):
    from training.evaluation.inference import infer_graphs
    raw=infer_graphs(tokens[b:b+1],{k:v[b:b+1] for k,v in pred.items() if k in ['pred_nodes','pred_logits']},model.relation_embed,object_queries=model.obj_token,relation_tokens=criterion.relation_tokens,node_threshold=-1.,edge_threshold=-1.)[0]
    metric=targets['spline'][b]['metric'].cpu().numpy();nodes=pred['pred_nodes'][b,:,:3].detach().cpu().numpy();truth=targets['nodes'][b][:,:3].cpu().numpy()
    c=dict(physical=nodes@metric.T,truth=truth@metric.T,gt_edges=targets['edges'][b].cpu().numpy(),scores=pred['pred_logits'][b].softmax(-1)[:,1].detach().cpu().numpy(),pairs=raw['query_ids'][raw['edges']].numpy(),relations=raw['edge_scores'].numpy())
    assignment=criterion.matcher({k:v[b:b+1] for k,v in pred.items() if k in ['pred_nodes','pred_logits']},{'nodes':[targets['nodes'][b]]})[0]
    s,t=assignment;c['hungarian_distance']=np.linalg.norm(c['physical'][s.numpy()]-c['truth'][t.numpy()],axis=-1)
    return c


def rows_for(c):
    from sklearn.metrics import average_precision_score
    result={};d=np.linalg.norm(c['physical'][:,None]-c['truth'][None],axis=-1)
    for tol in [1.,2.]:
        ids=np.flatnonzero(c['scores']>.5);r=measure(c,ids,tol);raw=association(c['physical'],c['truth'],tol)
        local=association(c['physical'][ids],c['truth'],tol);mapping={int(ids[i]):j for i,j in local.items()}
        r.update(raw_tp=len(raw),hungarian_mm_sum=float(c['hungarian_distance'].sum()),hungarian_n=len(c['hungarian_distance']),nearest_mm_sum=float(d.min(0).sum()),localized_mm_sum=float(sum(d[q,t] for q,t in mapping.items())))
        pairs=c['pairs'];eligible=np.array([int(a) in mapping and int(b) in mapping for a,b in pairs]);links={tuple(sorted(map(int,e))) for e in c['gt_edges']}
        labels=np.array([tuple(sorted((mapping[int(a)],mapping[int(b)]))) in links for a,b in pairs[eligible]])
        r['conditional_ap']=float(average_precision_score(labels,c['relations'][eligible])) if labels.any() else None
        result[str(tol)]=r
    return result


def summarize(rows):
    r=aggregate([{k:v for k,v in row.items() if k!='conditional_ap'} for row in rows]);r.update(raw_recall=r['raw_tp']/r['gt_nodes'],hungarian_mm=r['hungarian_mm_sum']/r['hungarian_n'],nearest_mm=r['nearest_mm_sum']/r['gt_nodes'],localized_mm=r['localized_mm_sum']/r['node_tp'] if r['node_tp'] else None)
    ap=[x['conditional_ap'] for x in rows if x['conditional_ap'] is not None];r['mean_patch_conditional_ap']=float(np.mean(ap)) if ap else None
    return r


@torch.no_grad()
def snapshot(bundle,arm,step,mode='normal',full=False):
    from scripts.multidataset_graph_evaluation import loaders
    from training.engine import _move_graph_batch
    plan,cfg,model,criterion,*_=bundle;model.eval();model.fine_feature_mode='roll' if mode=='roll' else 'normal';set_fine_enabled(model,mode!='blocked')
    spec=copy.deepcopy(plan['dataset']);spec['monitor_train']=spec['monitor_train'][:128 if full else 32];spec['monitor_val']=spec['monitor_val'] if full else spec['monitor_val'][:64]
    legacy=dict(datasets={'SyntheticMRI':spec});out=ROOT/'runs'/arm/f'predictions-{step:04d}-{mode}';out.mkdir(parents=True,exist_ok=True);rows=[];records=[];attention=[]
    for split,names,loader in loaders(cfg,legacy,{'dataset':'SyntheticMRI'},monitor=not full or mode!='normal'):
        if mode!='normal' and split=='train':continue
        for i,batch in enumerate(loader):
            volumes,targets=_move_graph_batch(batch,'cuda','image',True);tokens,pred,_=model(volumes);c=graph_cache(model,criterion,tokens,pred,targets)
            sample=names[i];subject=spec['subjects'][sample.split('_')[1]]
            for tol,row in rows_for(c).items():rows.append(dict(row,sample=sample,subject=subject,split=split,tolerance=tol))
            path=out/(sample+'.npz');np.savez_compressed(path,**{k:v for k,v in c.items() if isinstance(v,np.ndarray)})
            records.append(dict(sample=sample,split=split,subject=subject,cache=str(path),gt=str(Path(spec['root'])/split/'spline'/(sample+'.npz'))))
            attention.append(levels(model))
            if i%128==0:print('snapshot',arm,step,mode,split,i,len(names),flush=True)
    summary={split:{tol:summarize([r for r in rows if r['split']==split and r['tolerance']==tol]) for tol in ['1.0','2.0']} for split in sorted({r['split'] for r in rows})}
    mean_levels={n:{field:np.mean([a[n][field] for a in attention],axis=0).tolist() for field in ['mass','inside_mass']} for n in attention[0]}
    write(out/'rows.json',rows);write(out/'records.json',records);write(out/'summary.json',dict(metrics=summary,attention=mean_levels,step=step,mode=mode))
    model.fine_feature_mode='normal';set_fine_enabled(model,True);return summary


def smoke():
    from training.engine import _move_graph_batch
    from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction,ms_deform_attn_core_pytorch
    # Check the actual compiled multi-level operator against its torch reference,
    # including gradients of values, sampling locations and attention weights.
    seed();shapes=torch.tensor([[2,3,4],[4,6,8]],device='cuda');starts=torch.tensor([0,24],device='cuda')
    value=torch.randn(2,216,6,4,device='cuda',requires_grad=True);loc=torch.rand(2,7,6,2,3,3,device='cuda',requires_grad=True);weight=torch.rand(2,7,6,2,3,device='cuda',requires_grad=True)
    cuda=MSDeformAttnFunction.apply(value,shapes,starts,loc,weight,64);ref=ms_deform_attn_core_pytorch(value,shapes,loc,weight)
    torch.testing.assert_close(cuda,ref,atol=2e-5,rtol=2e-5)
    grad1=torch.autograd.grad(cuda.square().sum(),[value,loc,weight]);grad2=torch.autograd.grad(ref.square().sum(),[value,loc,weight])
    for a,b in zip(grad1,grad2):torch.testing.assert_close(a,b,atol=3e-4,rtol=3e-4)
    b0=setup('B0',False);b1=setup('B1',False);plan=b0[0];lookup={r.sample_id:i for i,r in enumerate(b0[5].records)}
    batch=_move_graph_batch(b0[6]([b0[5][lookup[s]] for s in plan['tiny_ids'][:4]]),'cuda','image',True)
    for b in [b0,b1]:b[2].eval()
    with torch.no_grad():
        a=b0[2](batch[0]);set_fine_enabled(b1[2],False);blocked=b1[2](batch[0]);set_fine_enabled(b1[2],True);normal=b1[2](batch[0])
    parity={}
    for key in ['pred_nodes','pred_logits']:
        strict=float((a[1][key]-blocked[1][key]).abs().max());near=float((a[1][key]-normal[1][key]).abs().max())
        torch.testing.assert_close(a[1][key],blocked[1][key],atol=5e-5,rtol=5e-5)
        assert near<(.001 if key=='pred_nodes' else .02),(key,near)
        parity[key]=dict(blocked_max_error=strict,active_max_error=near)
    audit=dict(cuda_reference_max_error=float((cuda-ref).abs().max()),cuda_gradient_max_errors=[float((a-b).abs().max()) for a,b in zip(grad1,grad2)],parity=parity,coarse_shape=list(a[2].shape),fine_shape=list(b1[2]._multiscale_features.shape),migration=b1[-1],initial_attention=levels(b1[2]),job_id=os.getenv('SLURM_JOB_ID'))
    # Preserve the real-checkpoint parity result before any optimizer updates.
    write(ROOT/'gpu-correctness.json',audit)
    del b0,b1,a,normal,blocked;torch.cuda.empty_cache()
    fits={}
    for arm in ['B0','B1']:
        bundle=setup(arm,False);plan,cfg,model,criterion,opt,dataset,collate,_=bundle;lookup={r.sample_id:i for i,r in enumerate(dataset.records)}
        samples=[dataset[lookup[s]] for s in plan['tiny_ids']];batches=[_move_graph_batch(collate(samples[i:i+4]),'cuda','image',True) for i in [0,4]];history=[]
        for step in range(513):
            if step in [0,128,512]:
                model.eval();rows=[]
                with torch.no_grad():
                    for volumes,targets in batches:
                        t,p,_=model(volumes)
                        for i in range(len(volumes)):rows.append(rows_for(graph_cache(model,criterion,t,p,targets,i))['2.0'])
                history.append(dict(step=step,metrics=summarize(rows),attention=levels(model)));print('tiny',arm,step,history[-1]['metrics'],flush=True)
            if step==512:break
            seed(step);model.train();volumes,targets=batches[step%2];opt.zero_grad(set_to_none=True);t,p,_=model(volumes);losses=criterion(t,p,targets);losses['total'].backward()
            if step==0:
                g=gradients(model);audit[arm+'_gradients']=g;write(ROOT/'gpu-correctness.json',audit)
                assert all(g.get(k,0)>0 for k in ['encoder.','decoder.','class_embed.','coord_embed.','relation_embed.','spline_head.'])
                if arm=='B1':assert g['fine_input_proj.']>0 and all(v['sampling_offsets']>0 and v['attention_weights']>0 for v in g['new_attention_rows'].values())
                audit[arm+'_gradients']=g
            assert torch.isfinite(losses['total']);opt.step()
        fits[arm]=history;write(ROOT/'tiny-fits.json',fits);del bundle,model,opt,criterion;torch.cuda.empty_cache()
    assert all(fits[a][-1]['metrics']['node_recall']>.85 and fits[a][-1]['metrics']['edge_f1']>.65 for a in fits),'Tiny fits failed; debug before pilot'
    audit['completed']=True;write(ROOT/'gpu-correctness.json',audit)


def train(arm):
    from training.engine import _move_graph_batch
    assert json.loads((ROOT/'gpu-correctness.json').read_text())['completed']
    bundle=setup(arm);plan,cfg,model,criterion,opt,dataset,collate,migration=bundle;out=ROOT/'runs'/arm;out.mkdir(parents=True,exist_ok=True)
    write(out/'config.json',cfg);write(out/'migration.json',migration)
    indices=np.random.default_rng(SEED).integers(len(dataset),size=(plan['updates'],8));batch_hash=hashlib.sha256(indices.tobytes()).hexdigest()
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith(('fine_input_proj.','input_proj.'))}
    history=[];start=time.time();snapshot(bundle,arm,0)
    for step,ix in enumerate(indices,1):
        seed(step)
        if dataset.intensity_transform is not None:dataset.intensity_transform.set_random_state(seed=SEED+step)
        volumes,targets=_move_graph_batch(collate([dataset[int(i)] for i in ix]),'cuda','image',True)
        if step==1:write(out/'batch-proof.json',dict(index_sha256=batch_hash,first_volumes_sha256=hashlib.sha256(volumes.cpu().numpy().tobytes()).hexdigest(),first_samples=[dataset.records[int(i)].sample_id for i in ix]))
        model.train();opt.zero_grad(set_to_none=True);tokens,pred,_=model(volumes);losses=criterion(tokens,pred,targets)
        assert torch.isfinite(losses['total']);losses['total'].backward()
        if step in [1,100,250,1000,2000]:
            record=dict(step=step,losses={k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1},gradients=gradients(model),attention=levels(model));history.append(record);write(out/'training.json',history);print('update',arm,step,record['losses']['total'],flush=True)
        opt.step()
        if step in [250,1000,2000]:
            if step==2000:torch.save(dict(net=model.state_dict(),config=cfg,step=step,source=plan['checkpoint'],migration=migration),out/'model-2000.pt')
            snapshot(bundle,arm,step,full=step==2000)
    if arm=='B1':
        snapshot(bundle,arm,2000,'blocked',True);snapshot(bundle,arm,2000,'roll',True)
    deltas={n:float((p.detach().cpu()-initial[n]).norm()/initial[n].norm().clamp_min(1e-12)) for n,p in model.named_parameters() if n in initial}
    write(out/'complete.json',dict(completed=True,updates=2000,batch_sha256=batch_hash,seconds=time.time()-start,projection_relative_change=deltas,job_id=os.getenv('SLURM_JOB_ID'),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','train']);p.add_argument('--arm',choices=['B0','B1']);a=p.parse_args();torch.set_num_threads(1)
    ROOT.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':prepare()
    elif a.mode=='smoke':smoke()
    else:train(a.arm)


if __name__=='__main__':main()
