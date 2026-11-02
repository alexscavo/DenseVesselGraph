"""Bounded BRAVA detector recovery and independent frozen geometry learning.

All supervised inputs are MRA and predicted-query features. Original dense paths
are targets only. This runner never selects a geometry branch using GT at inference.
"""
import argparse,copy,hashlib,json,math,os,random,signal,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from models.adaptive_geometry import AdaptiveSplineBank,PiecewiseSplineHead,CANDIDATES,fit_basis,uniform_knots
from models.spline import SplineHead,canonical_reverse
from scripts.analyze_spline_target_capacity import write
from scripts.ixi_node_focal_ablation import atomic_checkpoint
from scripts.frozen_spline_diagnostic import detector_digest,training_evaluation_loader

SEED=223039
WEIGHT=1.4739989612791453


def seed(n=0):random.seed(SEED+n);np.random.seed(SEED+n);torch.manual_seed(SEED+n)


def configure(plan,arm):
    from models.checkpoint import load_checkpoint_file
    from configs import validate_config
    cp=load_checkpoint_file(plan['checkpoints'][arm['initializer']]['path']) if arm['initializer']!='scratch' else None
    cfg=copy.deepcopy(plan['base_config'])
    if arm['kind']=='geometry':cfg=copy.deepcopy(cp['config'])
    target=copy.deepcopy(cfg['data']['datasets']['synthetic_mri'])
    target.update(root=plan['datasets'][arm['dataset']]['root'],train_samples=None,validation_samples=None,patch_selection='all',spline_controls=8,spline_family='bspline')
    cfg['data']['datasets']={'synthetic_mri':target};cfg['data'].update(batch_size=8 if arm['kind']=='detector' else 4,validation_batch_size=1,input_modality='raw_mra',train_augmentation=arm['kind']=='detector')
    cfg['model']['node_refinement']={'enabled':False}
    cfg['model']['spline'].update(enabled=True,controls=8,samples=64)
    cfg['loss']['weights']['spline']=WEIGHT;cfg['loss']['spline'].update(mode='controls',target='residual',beta_mm=1.)
    cfg['loss']['enabled']=[x for x in cfg['loss']['enabled'] if x!='domain']
    if 'spline' not in cfg['loss']['enabled']:cfg['loss']['enabled'].append('spline')
    cfg['domain_adaptation']['enabled']=False;cfg['tracking']['enabled']=False;cfg['runtime']['workers']=0
    cfg['experiment'].update(seed=SEED,name=arm['name'],geometry_representation=arm['representation'])
    cfg['augmentation']['synthetic_mri']['zoom']['enabled']=False
    if arm['kind']=='detector':cfg['model']['decoder']['object_queries']=256
    cfg['evaluation']['protocol']['max_detections']=cfg['model']['decoder']['object_queries']
    validate_config(cfg)
    return cfg,cp


def geometry_head(old,spec,method):
    kwargs=dict(residual_scale='mm',local_geometry=spec['local_geometry'],samples=64,token_inputs='full')
    pair=old.network[0].in_features-10;hidden=old.network[0].out_features
    if method=='adaptive':head=AdaptiveSplineBank(pair,hidden,**kwargs);heads=list(head.heads.values())
    elif method.startswith('piece'):
        head=PiecewiseSplineHead(pair,hidden,segments=4,continuity=method.split('_')[1],**kwargs);heads=[head]
    else:head=SplineHead(pair,hidden,controls=int(method[1:]),**kwargs);heads=[head]
    omit={'basis','greville','network.4.weight','network.4.bias','local_geometry.scorer.2.weight','local_geometry.scorer.2.bias'}
    state={n:v for n,v in old.state_dict().items() if n not in omit}
    for h in heads:
        result=h.load_state_dict(state,strict=False)
        assert not result.unexpected_keys
        assert set(result.missing_keys)==omit|({'knots'} if hasattr(h,'knots') else set())
    return head


def setup(root,arm_index):
    from models import build_model
    from models.matcher import build_matcher
    from data.loaders import build_data_loaders
    from scripts.train_ixi_transfer import TransferCriterion
    plan=json.loads((root/'plan.json').read_text());arm=plan['arms'][arm_index];cfg,cp=configure(plan,arm);seed()
    model=build_model(cfg)
    query_transfer=None
    if cp is not None:
        state={n:v for n,v in cp['net'].items() if 'domain_discriminator' not in n}
        target=model.state_dict();assert set(state)==set(target)
        for name in state:
            if state[name].shape!=target[name].shape:
                assert name=='query_embed.weight','Unsupported checkpoint mismatch: '+name
                source_q=cp['config']['model']['decoder']['object_queries'];target_q=cfg['model']['decoder']['object_queries'];rel=cfg['model']['decoder']['relation_tokens']
                assert source_q<target_q and state[name].shape[1:]==target[name].shape[1:]
                expanded=target[name].clone();expanded[:source_q]=state[name][:source_q];expanded[target_q:target_q+rel]=state[name][source_q:source_q+rel]
                state[name]=expanded;query_transfer=dict(source_object_queries=source_q,target_object_queries=target_q,policy='copy source object rows and relation rows; retain seeded additional object rows')
        model.load_state_dict(state,strict=True);del cp,state
    if arm['kind']=='geometry':
        model.spline_head=geometry_head(model.spline_head,cfg['model']['spline'],arm['representation'])
        model.requires_grad_(False);model.spline_head.requires_grad_(True)
        if arm['representation']=='adaptive':model.spline_head.selector.requires_grad_(False)
        model.eval()
    else:model.train().requires_grad_(True)
    model=model.cuda()
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head if arm['kind']=='detector' else None,route='graph_only').cuda();criterion.set_training_progress(1,0.)
    loader,_=build_data_loaders(cfg);dataset=loader.dataset.datasets[0];records={r.sample_id:r for r in dataset.records}
    ids=plan['datasets'][arm['dataset']]['train_ids'];dataset.records=[records[s] for s in ids]
    if dataset.intensity_transform is not None:dataset.intensity_transform.set_random_state(seed=SEED)
    optcfg=cfg['training']['optimizer'];groups=[]
    if arm['kind']=='detector':
        groups=[dict(params=[p for n,p in model.named_parameters() if n.startswith('encoder.')],lr=3.5e-5,base_lr=3.5e-5),dict(params=[p for n,p in model.named_parameters() if not n.startswith('encoder.')],lr=7e-5,base_lr=7e-5)]
    else:groups=[dict(params=[p for p in model.parameters() if p.requires_grad],lr=7e-5,base_lr=7e-5)]
    optimizer=torch.optim.AdamW(groups,betas=tuple(optcfg['betas']),eps=optcfg['epsilon'],weight_decay=optcfg['weight_decay'])
    return plan,arm,model,cfg,criterion,optimizer,dataset,loader.collate_fn,query_transfer


def edge_inputs(model,criterion,tokens,pred,targets,b,assignment):
    source,target=assignment;mapping=dict(zip(target.tolist(),source.tolist()))
    selected=[i for i,(a,c) in enumerate(targets['edges'][b].tolist()) if a in mapping and c in mapping]
    if not selected:return None
    pairs=torch.tensor([[mapping[a],mapping[c]] for a,c in targets['edges'][b][selected].tolist()],device=tokens.device)
    metric=targets['spline'][b]['metric'].to(tokens);reverse=canonical_reverse(pred['pred_nodes'][b,pairs,:3]@metric.T);pairs[reverse]=pairs[reverse].flip(1)
    endpoints=pred['pred_nodes'][b,pairs,:3].detach();rel=tokens[b,criterion.object_queries:criterion.object_queries+criterion.relation_tokens]
    features=criterion._relation_features(tokens[b,:criterion.object_queries],rel,pairs).detach()
    local=dict(local_features=pred['spline_local_features'][b].detach(),input_shape=pred['spline_input_shape'])
    return selected,reverse,features,endpoints,metric,local


def get_targets(record,head,memo):
    key=(record.sample_id,tuple(head.knots.cpu().tolist()) if hasattr(head,'knots') else head.controls)
    if key not in memo:
        with np.load(record.graph.parent.parent/'spline'/(record.sample_id+'.npz')) as f:
            metric=f['metric'].astype(float);inv=np.linalg.inv(metric);paths=f['full_paths'].astype(float)@metric.T;controls=[]
            knots=head.knots.cpu().numpy() if hasattr(head,'knots') else uniform_knots(head.controls)
            for a,b in zip(f['path_offsets'][:-1],f['path_offsets'][1:]):controls.append(fit_basis(paths[a:b],knots)[0]@inv.T)
        memo[key]=torch.tensor(np.array(controls).reshape(-1,head.controls,3),dtype=torch.float32)
    return memo[key]


def native_geometry_loss(head,features,endpoints,metric,local,target,reverse):
    control,_=head(features,endpoints,metric,**local);target=target.to(control).clone();target[reverse]=target[reverse].flip(1)
    g=head.greville[None,1:-1,None];delta=(control[:,1:-1]-target[:,1:-1])@metric.T
    endpoint_delta=(endpoints-target[:,[0,-1]])@metric.T
    delta=delta-((1-g)*endpoint_delta[:,:1]+g*endpoint_delta[:,1:])
    return F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=1.,reduction='none').mean((1,2))


def geometry_loss(model,criterion,tokens,pred,targets,assignments,records,memo):
    errors=[];head=model.spline_head
    for b,assignment in enumerate(assignments):
        data=edge_inputs(model,criterion,tokens,pred,targets,b,assignment)
        if data is None:continue
        selected,reverse,features,endpoints,metric,local=data
        heads=list(head.heads.values()) if isinstance(head,AdaptiveSplineBank) else [head]
        contributions=[native_geometry_loss(h,features,endpoints,metric,local,get_targets(records[b],h,memo)[selected],reverse) for h in heads]
        errors.append(torch.stack(contributions).mean(0))
    return torch.cat(errors).mean() if errors else sum(p.sum()*0 for p in head.parameters())


def update(bundle,indices,number,memo):
    from training.engine import _move_graph_batch
    plan,arm,model,cfg,criterion,opt,dataset,collate,_=bundle
    seed(number);volumes,targets=_move_graph_batch(collate([dataset[int(i)] for i in indices]),'cuda','image',True)
    opt.zero_grad(set_to_none=True)
    if arm['kind']=='detector':
        model.train();tokens,pred,_=model(volumes);losses=criterion(tokens,pred,targets);loss=losses['total']
        if number==1:
            shared=[p for n,p in model.named_parameters() if not n.startswith('spline_head.')]
            grads=torch.autograd.grad(losses['spline'],shared,retain_graph=True,allow_unused=True)
            assert all(g is None or not bool(g.abs().sum()) for g in grads),'Unexpected geometry gradient to detector'
    else:
        model.eval();model.spline_head.train()
        with torch.no_grad():tokens,pred,_=model(volumes);assignments=criterion.matcher(pred,targets)
        loss=WEIGHT*geometry_loss(model,criterion,tokens,pred,targets,assignments,[dataset.records[int(i)] for i in indices],memo)
    assert torch.isfinite(loss);loss.backward()
    gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    if arm['kind']=='geometry':assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('spline_head.'))
    opt.step();return float(loss.detach())


def smoke(root):
    plan=json.loads((root/'plan.json').read_text());timing={};hashes={}
    smoke_root=root/('smoke-'+os.environ.get('SLURM_JOB_ID',str(time.time_ns())));smoke_root.mkdir(exist_ok=False)
    probe=copy.deepcopy(plan)
    for spec in probe['datasets'].values():spec['monitor_train']=spec['monitor_train'][:2];spec['val_ids']=spec['monitor_val'][:2]
    write(smoke_root/'plan.json',probe)
    for index,arm in enumerate(plan['arms']):
        # Each independent job has an actual forward/backward check, not just a config check.
        bundle=setup(root,index);model=bundle[2];identity=detector_digest(model);memo={};seconds=[];torch.cuda.reset_peak_memory_stats()
        for n in range(1,7):
            torch.cuda.synchronize();start=time.monotonic();loss=update(bundle,np.arange(4 if arm['kind']=='geometry' else 8)+(n-1)*(4 if arm['kind']=='geometry' else 8),n,memo);torch.cuda.synchronize()
            if n>=3:seconds.append(time.monotonic()-start)
        if arm['kind']=='geometry':
            assert detector_digest(model)==identity
            dataset=arm['dataset'];assert dataset not in hashes or hashes[dataset]==identity;hashes[dataset]=identity
        if arm['name']=='D1_IXI' or arm['dataset']=='IXI':
            from scripts.multidataset_graph_evaluation import snapshot,evaluate
            small=list(bundle);small[0]=probe
            run=smoke_root/'runs'/arm['name'];run.mkdir(parents=True,exist_ok=True);write(run/'config.json',bundle[3])
            snapshot(smoke_root,index,tuple(small),6,smoke_root/'predictions'/arm['name']/'000006')
            evaluate(smoke_root,index)
        timing[arm['name']]=dict(seconds_per_update=float(np.mean(seconds)),peak_gpu_bytes=torch.cuda.max_memory_allocated(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),smoke_loss=loss,query_transfer=bundle[-1])
        del bundle,model;torch.cuda.empty_cache()
    detector_slowest=max(v['seconds_per_update'] for k,v in timing.items() if k.startswith('D'))
    assert max(v['seconds_per_update'] for k,v in timing.items() if not k.startswith('D'))*plan['geometry_updates']<4*3600,'Geometry training exceeds bounded allocation'
    assert detector_slowest*plan['detector_updates']<4*3600,'Detector training exceeds bounded allocation'
    write(root/'smoke.json',dict(completed=True,timings=timing,geometry_detector_hashes=hashes))


def monitor(root,index,bundle,step):
    from scripts.multidataset_graph_evaluation import monitor_model
    return monitor_model(root,index,bundle,step)


def train(root,index):
    plan=json.loads((root/'plan.json').read_text());assert json.loads((root/'smoke.json').read_text())['completed']
    bundle=setup(root,index);_,arm,model,cfg,criterion,opt,dataset,collate,transfer=bundle
    out=root/'runs'/arm['name'];out.mkdir(parents=True,exist_ok=False);write(out/'config.json',cfg)
    count=plan['detector_updates'] if arm['kind']=='detector' else plan['geometry_updates'];batch=cfg['data']['batch_size'];rng=np.random.default_rng(SEED);order=[]
    while len(order)<count*batch:order.extend(rng.permutation(len(dataset)).tolist())
    order=np.array(order[:count*batch]).reshape(count,batch);initial=detector_digest(model);memo={};history=[];start=time.monotonic();stop=[False]
    for sig in [signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,lambda *a:stop.__setitem__(0,True))
    checkpoints=plan['detector_checkpoints'] if arm['kind']=='detector' else plan['geometry_checkpoints']
    if arm['kind']=='detector':monitor(root,index,bundle,0)
    for n,indices in enumerate(order,1):
        scale=min(1.,n/500)*(.3+.7*(1+math.cos(math.pi*n/count))/2)
        for g in opt.param_groups:g['lr']=g['base_lr']*scale
        loss=update(bundle,indices,n,memo)
        if n==1 or n%100==0:
            history.append(dict(step=n,loss=loss,seconds=time.monotonic()-start,learning_rates=[g['lr'] for g in opt.param_groups]));write(out/'history.json',history);print(arm['name'],history[-1],flush=True)
        if n in checkpoints:
            atomic_checkpoint(out/f'model-{n:06d}.pt',model,cfg,opt,n)
            if arm['kind']=='detector':monitor(root,index,bundle,n)
        if stop[0] or time.monotonic()-start>16000:
            atomic_checkpoint(out/'interrupted.pt',model,cfg,opt,n);raise RuntimeError('Interrupted; block incomplete comparison')
    if arm['representation']=='adaptive':
        train_selector(root,index,bundle)
        atomic_checkpoint(out/f'model-{count:06d}.pt',model,cfg,opt,count)
    if arm['kind']=='geometry':assert detector_digest(model)==initial
    write(out/'complete.json',dict(completed=True,updates=count,seconds=time.monotonic()-start,initial_detector_hash=initial,final_detector_hash=detector_digest(model),query_transfer=transfer,order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))


def train_selector(root,index,bundle):
    from data.loaders import build_data_loaders
    from training.engine import _move_graph_batch
    from scripts.continuous_representation import fit_options,choose
    plan,arm,model,cfg,criterion,opt,_,_,_=bundle;head=model.spline_head
    cfg=copy.deepcopy(cfg);cfg['data'].update(train_augmentation=False,batch_size=1)
    full,_=build_data_loaders(cfg);names=plan['datasets'][arm['dataset']]['monitor_train'];loader=training_evaluation_loader(full,cfg,names)
    xs=[];ys=[];unmet=0
    model.eval().requires_grad_(False);model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for i,batch in enumerate(loader):
            volumes,targets=_move_graph_batch(batch,'cuda','image',True);tokens,pred,_=model(volumes);assign=criterion.matcher(pred,targets)[0]
            data=edge_inputs(model,criterion,tokens,pred,targets,0,assign)
            if data is None:continue
            chosen,reverse,features,endpoints,metric,local=data
            xs.append(head.features(features,endpoints,metric).cpu())
            record=loader.dataset.records[i]
            with np.load(record.graph.parent.parent/'spline'/(record.sample_id+'.npz')) as f:
                for j in chosen:
                    a,b=f['path_offsets'][j:j+2];path=f['full_paths'][a:b].astype(float)@f['metric'].T
                    k,ok,_=choose(fit_options(path),path,.5);unmet+=int(not ok);ys.append(CANDIDATES.index(k))
    x=torch.cat(xs).cuda();y=torch.tensor(ys,device='cuda')
    torch.save(dict(features=x.cpu(),labels=y.cpu(),sample_ids=names,training_only=True),root/'runs'/arm['name']/'selector-features.pt')
    head.selector.requires_grad_(True);optimizer=torch.optim.AdamW(head.selector.parameters(),lr=1e-3)
    seed();
    for n in range(2000):
        selected=torch.randint(len(y),(min(256,len(y)),),device='cuda');optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(head.selector(x[selected].detach()),y[selected]);loss.backward();optimizer.step()
    write(root/'runs'/arm['name']/'selector.json',dict(completed=True,training_subjects_only=True,examples=len(y),oracle_tolerance_failures=unmet,updates=2000,loss=float(loss.detach()),feature_source='detached shared spline-trunk embedding',classes=list(CANDIDATES)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','train']);p.add_argument('--root',required=True,type=Path);p.add_argument('--arm',type=int);a=p.parse_args();torch.set_num_threads(1)
    if a.mode=='smoke':smoke(a.root)
    else:train(a.root,a.arm)
