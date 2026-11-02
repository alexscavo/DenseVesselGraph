"""Matched-update frozen-detector geometry heads. No node/relation training."""
import argparse,copy,hashlib,json,sys,time,signal,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.analyze_spline_target_capacity import write
from scripts.brava_graph_study import load_zero,seeded
from scripts.frozen_spline_diagnostic import freeze_detector,detector_digest,training_evaluation_loader
from data.spline_targets import fit_spline
from models.spline import SplineHead,canonical_reverse
from scripts.ixi_node_focal_ablation import atomic_checkpoint

KS=[8,16,24]


def loader_config(cfg):
    value=copy.deepcopy(cfg);value['data']['datasets']['synthetic_mri']['spline_controls']=8
    return value


def setup(root,k):
    from data.loaders import build_data_loaders
    from models.matcher import build_matcher
    from scripts.train_ixi_transfer import TransferCriterion
    plan=json.loads((root/'plan.json').read_text());seeded()
    model,cfg=load_zero(root,'ixi_spline');old=model.spline_head
    assert old.family=='bspline' and old.local_geometry is not None and old.curve_evidence is None
    cfg['model']['spline']['controls']=k;cfg['loss']['weights']['spline']=1.4739989612791453
    cfg['loss']['spline'].update(mode='controls',target='residual')
    cfg['data'].update(train_augmentation=False,batch_size=4,validation_batch_size=1)
    # Loader reads immutable K8 caches; targets are refitted from dense paths below.
    cfg['data']['datasets']['synthetic_mri']['spline_controls']=k
    from configs import validate_config
    validate_config(cfg)
    spec=cfg['model']['spline'];seeded()
    head=SplineHead(old.network[0].in_features-10,old.network[0].out_features,k,spec['samples'],residual_scale=spec['residual_scale'],local_geometry=spec['local_geometry'],token_inputs=spec.get('token_inputs','full')).cuda()
    copied={n:v for n,v in old.state_dict().items() if n not in ['basis','greville','network.4.weight','network.4.bias','local_geometry.scorer.2.weight','local_geometry.scorer.2.bias']}
    missing=head.load_state_dict(copied,strict=False)
    assert set(missing.missing_keys)=={'basis','greville','network.4.weight','network.4.bias','local_geometry.scorer.2.weight','local_geometry.scorer.2.bias'}
    model.spline_head=head;freeze_detector(model)
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=head,route='graph_only').cuda()
    criterion.set_training_progress(1,0.)
    full,_=build_data_loaders(loader_config(cfg));dataset=full.dataset.datasets[0]
    records={r.sample_id:r for r in dataset.records};dataset.records=[records[s] for s in plan['head_ids']['train']]
    optcfg=cfg['training']['optimizer'];opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=7e-5,betas=tuple(optcfg['betas']),eps=optcfg['epsilon'],weight_decay=optcfg['weight_decay'])
    return model,cfg,criterion,opt,dataset,full.collate_fn


def target_cache(record,k,memo):
    key=(record.sample_id,k)
    if key not in memo:
        with np.load(record.graph.parent.parent/'spline'/(record.sample_id+'.npz')) as f:
            m=f['metric'].astype(float);inverse=np.linalg.inv(m);cs=[];xs=[]
            for a,b in zip(f['path_offsets'][:-1],f['path_offsets'][1:]):
                c,x=fit_spline(f['full_paths'][a:b].astype(float)@m.T,k);cs.append(c@inverse.T);xs.append(x@inverse.T)
            memo[key]=(torch.tensor(np.array(cs).reshape(-1,k,3),dtype=torch.float32),torch.tensor(np.array(xs).reshape(-1,64,3),dtype=torch.float32))
    return memo[key]


def step(model,criterion,opt,dataset,collate,indices,k,memo):
    from training.engine import _move_graph_batch
    items=[]
    for i in indices:
        item=list(dataset[int(i)]);geometry=dict(item[4][0]);geometry['controls'],geometry['curves']=target_cache(dataset.records[int(i)],k,memo);item[4]=[geometry];items.append(tuple(item))
    volumes,targets=_move_graph_batch(collate(items),'cuda','image',True)
    model.eval();model.spline_head.train();opt.zero_grad(set_to_none=True)
    with torch.no_grad():tokens,pred,_=model(volumes);assignments=criterion.matcher(pred,targets)
    loss=criterion.loss_spline(tokens,pred,targets,assignments);total=loss['spline']*criterion.weights['spline'];assert torch.isfinite(total)
    total.backward();assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('spline_head.'))
    grads=[p.grad for p in model.spline_head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
    opt.step();return float(total.detach())


def benchmark(root):
    gate=json.loads((root/'results/gt-gate.json').read_text());assert gate['completed'] and gate['larger_fixed_mean_improves']
    times={};identities=[];initial_predictions=[]
    smoke=root/('smoke-'+os.environ.get('SLURM_JOB_ID',str(time.time_ns())));smoke.mkdir(exist_ok=False)
    smoke_plan=json.loads((root/'plan.json').read_text());smoke_plan['head_ids']={s:v[:4] for s,v in smoke_plan['head_ids'].items()};smoke_plan['head_monitor_ids']=smoke_plan['head_ids']['train'][:2]
    write(smoke/'plan.json',smoke_plan)
    from training.engine import _move_graph_batch
    for k in KS:
        model,cfg,criterion,opt,dataset,collate=setup(root,k);identity=detector_digest(model);identities.append(identity)
        with torch.no_grad():
            volumes,_=_move_graph_batch(collate([dataset[0]]),'cuda','image',True);_,pred,_=model(volumes)
            initial_predictions.append((pred['pred_nodes'].cpu(),pred['pred_logits'].cpu()))
            endpoints=pred['pred_nodes'][0,:2,:3].unsqueeze(0);metric=torch.eye(3,device='cuda')*64
            feat=torch.zeros((1,model.spline_head.network[0].in_features-10),device='cuda')
            c,x=model.spline_head(feat,endpoints,metric,local_features=pred['spline_local_features'][0],input_shape=pred['spline_input_shape'])
            t=torch.linspace(0,1,64,device='cuda')[None,:,None];torch.testing.assert_close(x,(1-t)*endpoints[:,:1]+t*endpoints[:,1:],atol=1e-6,rtol=1e-5)
        memo={};timing=[]
        for i in range(12):
            torch.cuda.synchronize();start=time.monotonic();loss=step(model,criterion,opt,dataset,collate,np.arange(i*4,i*4+4)%len(dataset),k,memo);torch.cuda.synchronize()
            if i>=3:timing.append(time.monotonic()-start)
        assert identity==detector_digest(model)
        checkpoint=smoke/'heads'/str(k)/'model-000012.pt';checkpoint.parent.mkdir(parents=True)
        atomic_checkpoint(checkpoint,model,cfg,opt,12)
        times[str(k)]=dict(seconds_per_update=float(np.mean(timing)),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),smoke_loss=loss)
        del model,criterion,opt;torch.cuda.empty_cache()
    assert len(set(identities))==1
    for pred in initial_predictions[1:]:
        for a,b in zip(pred,initial_predictions[0]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    write(smoke/'budget.json',dict(updates=12,detector_sha256=identities[0],timing=times))
    for k in KS:predict(smoke,k)
    from scripts.evaluate_continuous_geometry import collect
    collect(smoke,publish_results=False)
    slow=max(v['seconds_per_update'] for v in times.values());updates=min(8000,int(3600/(1.2*slow))//100*100)
    assert updates>=1000,'Insufficient throughput for a useful bounded head pilot'
    write(root/'budget.json',dict(completed=True,updates=updates,batch_size=4,timing=times,walltime='02:00:00',detector_sha256=identities[0],target_seconds=3600,checkpoints=sorted({1000,updates//2,updates})))


def train(root,k):
    budget=json.loads((root/'budget.json').read_text());model,cfg,criterion,opt,dataset,collate=setup(root,k)
    identity=detector_digest(model);assert identity==budget['detector_sha256']
    out=root/'heads'/str(k);out.mkdir(parents=True,exist_ok=False);memo={};rng=np.random.default_rng(221039);order=[]
    while len(order)<budget['updates']*4:order.extend(rng.permutation(len(dataset)).tolist())
    order=np.array(order[:budget['updates']*4]).reshape(-1,4);rows=[];start=time.monotonic();stop=[False]
    for sig in [signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,lambda *a:stop.__setitem__(0,True))
    for i,indices in enumerate(order,1):
        scale=min(1.,i/100)*max(0.,1-i/(len(order)+1))**.9
        for g in opt.param_groups:g['lr']=7e-5*scale
        loss=step(model,criterion,opt,dataset,collate,indices,k,memo)
        if i%50==0 or i==1:
            rows.append(dict(step=i,loss=loss,seconds=time.monotonic()-start));write(out/'history.json',rows);print(k,rows[-1],flush=True)
        if i in budget['checkpoints']:atomic_checkpoint(out/f'model-{i:06d}.pt',model,cfg,opt,i)
        if stop[0] or time.monotonic()-start>6600:
            atomic_checkpoint(out/'interrupted.pt',model,cfg,opt,i);raise RuntimeError('Interrupted; block unequal-budget comparison')
    assert detector_digest(model)==identity
    write(out/'complete.json',dict(completed=True,updates=len(order),seconds=time.monotonic()-start,detector_sha256=identity,order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))


@torch.no_grad()
def predict(root,k):
    from models.checkpoint import load_checkpoint_file
    from data.loaders import build_data_loaders,build_evaluation_loader
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    model,cfg,criterion,opt,_,_=setup(root,k);del opt
    budget=json.loads((root/'budget.json').read_text());plan=json.loads((root/'plan.json').read_text())
    cp=load_checkpoint_file(root/'heads'/str(k)/f'model-{budget["updates"]:06d}.pt');model.load_state_dict(cp['net'],strict=True);del cp
    model.eval();freeze_detector(model);identity=detector_digest(model)
    assert identity==budget['detector_sha256']
    out=root/'predictions'/str(k);out.mkdir(parents=True,exist_ok=False);records=[]
    for split,names in [('train',plan['head_monitor_ids']),('val',plan['head_ids']['val'])]:
        if split=='train':full,_=build_data_loaders(loader_config(cfg));loader=training_evaluation_loader(full,loader_config(cfg),names)
        else:loader=build_evaluation_loader(loader_config(cfg),dataset_name='synthetic_mri',sample_ids=names)
        for index,batch in enumerate(loader):
            sample=names[index];volumes,targets=_move_graph_batch(batch,'cuda','image',True);tokens,pred,_=model(volumes);metric=targets['spline'][0]['metric'].to(tokens)
            kw=dict(object_queries=model.obj_token,relation_tokens=criterion.relation_tokens,node_threshold=cfg['evaluation'].get('node_threshold'))
            normal=infer_graphs(tokens,pred,model.relation_embed,edge_threshold=cfg['evaluation'].get('edge_threshold'),spline_head=model.spline_head,spline_metrics=[metric],**kw)[0]
            allpairs=infer_graphs(tokens,pred,model.relation_embed,edge_threshold=-1.,**kw)[0]
            c={n:v.cpu().numpy() for n,v in normal.items() if torch.is_tensor(v)}
            c.update(raw_nodes=pred['pred_nodes'][0,:,:3].cpu().numpy(),raw_scores=pred['pred_logits'][0].softmax(-1)[:,1].cpu().numpy(),candidate_edges=allpairs['edges'].numpy(),candidate_scores=allpairs['edge_scores'].numpy())
            source,target=criterion.matcher(pred,targets)[0];mapping=dict(zip(target.tolist(),source.tolist()))
            gt_edges=targets['edges'][0];chosen=[j for j,(a,b) in enumerate(gt_edges.tolist()) if a in mapping and b in mapping]
            pairs=torch.tensor([[mapping[a],mapping[b]] for a,b in gt_edges[chosen].tolist()],device='cuda',dtype=torch.long).reshape(-1,2)
            if len(pairs):
                reverse=canonical_reverse(pred['pred_nodes'][0,pairs,:3]@metric.T);pairs[reverse]=pairs[reverse].flip(1)
                relation=tokens[0,criterion.object_queries:criterion.object_queries+criterion.relation_tokens]
                features=criterion._relation_features(tokens[0,:criterion.object_queries],relation,pairs)
                controls,_=model.spline_head(features,pred['pred_nodes'][0,pairs,:3],metric,local_features=pred['spline_local_features'][0],input_shape=pred['spline_input_shape'])
                c.update(matched_controls=controls.cpu().numpy(),matched_edges=np.array(chosen),matched_reverse=reverse.cpu().numpy())
            else:c.update(matched_controls=np.empty((0,k,3)),matched_edges=np.array([],int),matched_reverse=np.array([],bool))
            cache=out/(sample+'.npz');np.savez_compressed(cache,**c)
            records.append(dict(sample_id=sample,subject=plan['subjects_by_id'][sample],split=split,cache=str(cache),gt=str(Path(plan['dataset'])/split/'spline'/(sample+'.npz'))))
            if index%64==0:print(k,split,index,len(names),flush=True)
    assert detector_digest(model)==identity
    write(out/'records.json',records);write(out/'complete.json',dict(completed=True,detector_sha256=identity,batching=1,records=len(records)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','train','predict']);p.add_argument('--root',type=Path,required=True);p.add_argument('--k',type=int);a=p.parse_args();torch.set_num_threads(1)
    if a.mode=='benchmark':benchmark(a.root)
    elif a.mode=='train':train(a.root,a.k)
    else:predict(a.root,a.k)
