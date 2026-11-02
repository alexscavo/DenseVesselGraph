"""BRAVA Stage 1: scratch paired baselines and immutable zero-shot predictions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.ixi_cached_mechanisms import save,sha
from scripts.ixi_node_focal_ablation import state_digest,atomic_checkpoint

ARMS=['standard','coordinate_refinement']
SEED=211039


def configuration(plan,arm):
    cfg=copy.deepcopy(plan['training_config'])
    cfg['experiment'].update(seed=SEED,name='brava_'+arm)
    cfg['data'].update(input_modality='raw_mra',train_augmentation=True,batch_size=8,validation_batch_size=1)
    target=cfg['data']['datasets']['synthetic_mri']
    target.update(root=plan['dataset'],train_samples=None,validation_samples=None,patch_selection='all',spline_controls=8,spline_family='bspline')
    cfg['data']['datasets']={'synthetic_mri':target}
    cfg['runtime']['workers']=0;cfg['tracking']['enabled']=False
    cfg['domain_adaptation']['enabled']=False
    cfg['training']['input']='image'
    cfg['model']['decoder']['object_queries']=256
    cfg['model']['node_refinement'].update(enabled=arm=='coordinate_refinement',refine_classification=False)
    cfg['model']['spline']['enabled']=True
    cfg['loss']['enabled']=[k for k in cfg['loss']['enabled'] if k!='domain']
    if 'spline' not in cfg['loss']['enabled']:cfg['loss']['enabled'].append('spline')
    cfg['loss']['weights']['spline']=1.4739989612791453
    cfg['loss']['spline']['target']='residual'
    cfg['evaluation']['protocol']['max_detections']=256
    assert cfg['model']['encoder']['input_channels']==1
    assert cfg['loss']['node']['classification']['name']=='weighted_cross_entropy'
    assert cfg['loss']['node']['classification']['class_weights']==[.25,.75]
    assert cfg['loss']['edge']['classification']['name']=='focal'
    assert not cfg['loss']['edge']['candidates']['include_unmatched']
    assert not cfg['augmentation']['synthetic_mri']['zoom']['enabled']
    return cfg


def seeded(number=0):
    random.seed(SEED+number);np.random.seed(SEED+number);torch.manual_seed(SEED+number)


def setup(root,arm):
    from models import build_model
    from models.matcher import build_matcher
    from configs import validate_config
    from data.loaders import build_data_loaders
    from scripts.train_ixi_transfer import TransferCriterion
    p=json.loads((root/'plan.json').read_text());cfg=configuration(p,arm);validate_config(cfg)
    seeded();model=build_model(cfg).cuda().train().requires_grad_(True)
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda()
    criterion.set_training_progress(1,0.)
    loader,_=build_data_loaders(cfg);assert len(loader.dataset.datasets)==1
    dataset=loader.dataset.datasets[0]
    assert [r.sample_id for r in dataset.records]==p['ids']['train']
    if dataset.intensity_transform is not None:dataset.intensity_transform.set_random_state(seed=SEED)
    optcfg=cfg['training']['optimizer']
    opt=torch.optim.AdamW(model.parameters(),lr=7e-5,betas=tuple(optcfg['betas']),eps=optcfg['epsilon'],weight_decay=optcfg['weight_decay'])
    return model,cfg,criterion,opt,dataset,loader.collate_fn


def detector_identity(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('node_refinement.'):continue
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def orders(updates,count):
    rng=np.random.default_rng(SEED);draws=[]
    while len(draws)<updates*8:draws.extend(rng.permutation(count).tolist())
    return np.array(draws[:updates*8]).reshape(updates,8)


def update(model,cfg,criterion,opt,dataset,collate,indices,number):
    from training.engine import _move_graph_batch
    seeded(number);batch=collate([dataset[int(i)] for i in indices])
    volumes,targets=_move_graph_batch(batch,'cuda','image',True)
    model.train();opt.zero_grad(set_to_none=True)
    tokens,predictions,_=model(volumes);losses=criterion(tokens,predictions,targets)
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.spline_head.parameters()), 'No spline training gradient'
    if number==1 and model.node_refinement is not None:
        assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.node_refinement.parameters()), 'No coordinate-refinement gradient'
    opt.step()
    return dict(step=number,**{k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1})


def benchmark(root,smoke=False):
    out=root/('smoke' if smoke else 'benchmark');out.mkdir(parents=True,exist_ok=False)
    states=[];timings={};initial_outputs=[]
    for arm in ARMS:
        model,cfg,criterion,opt,dataset,collate=setup(root,arm);states.append(detector_identity(model))
        from training.engine import _move_graph_batch
        seeded();probe=collate([dataset[i] for i in range(2)]);volumes,_=_move_graph_batch(probe,'cuda','image',True)
        model.eval()
        with torch.no_grad():_,pred,_=model(volumes)
        initial_outputs.append({k:v.detach().cpu().clone() for k,v in pred.items() if k in ['pred_logits','pred_nodes']})
        order=orders(2 if smoke else 35,len(dataset));started=None;history=[];torch.cuda.reset_peak_memory_stats()
        for step,indices in enumerate(order,1):
            if step==4:torch.cuda.synchronize();started=time.monotonic()
            history.append(update(model,cfg,criterion,opt,dataset,collate,indices,step))
        torch.cuda.synchronize()
        timings[arm]=dict(seconds_per_update=(time.monotonic()-started)/32 if started else None,
            parameters=sum(p.numel() for p in model.parameters()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),last=history[-1])
        if smoke:atomic_checkpoint(out/(arm+'.pt'),model,cfg,opt,len(order))
        del model,criterion,opt;torch.cuda.empty_cache()
    assert states[0]==states[1], 'Shared scratch parameters differ'
    for key in initial_outputs[0]:torch.testing.assert_close(initial_outputs[0][key],initial_outputs[1][key],rtol=0,atol=0)
    save(out/'summary.json',dict(completed=True,timings=timings,common_initial_state=states[0],initial_refinement_identity=True))
    if not smoke:
        plan=json.loads((root/'plan.json').read_text());per_epoch=int(np.ceil(len(plan['ids']['train'])/8))
        slowest=max(r['seconds_per_update'] for r in timings.values())
        updates=min(40*per_epoch,max(20*per_epoch,int(10800/(1.2*slowest))//100*100))
        assert updates*slowest*1.2<16200, 'Twenty full-data passes exceed bounded allocation; review before training'
        save(out/'budget.json',dict(completed=True,updates=updates,minimum_full_passes=20,maximum_full_passes=40,
            batch_size=8,seconds_per_update=slowest,planned_training_seconds=updates*slowest,
            checkpoints=sorted({k for k in [1000,5000,10000,20000,updates] if k<=updates}),
            schedule='7e-5; one-epoch linear warmup then polynomial power0.9 to common final step',walltime='06:00:00'))


def train(root,arm):
    budget=json.loads((root/'benchmark/budget.json').read_text())
    model,cfg,criterion,opt,dataset,collate=setup(root,arm)
    out=root/'runs'/arm;out.mkdir(parents=True,exist_ok=False)
    initial=detector_identity(model);order=orders(budget['updates'],len(dataset));stop=[False];started=time.monotonic();history=[]
    for sig in [signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,lambda *a:stop.__setitem__(0,True))
    warmup=int(np.ceil(len(dataset)/8))
    for step,indices in enumerate(order,1):
        scale=step/warmup if step<=warmup else max(0.,1-(step-warmup)/(len(order)-warmup))**.9
        for group in opt.param_groups:group['lr']=7e-5*scale
        row=update(model,cfg,criterion,opt,dataset,collate,indices,step)
        if step==1 or step%50==0:
            history.append(dict(row,lr=7e-5*scale,elapsed_seconds=time.monotonic()-started));save(out/'history.json',history)
            print(arm,history[-1],flush=True)
        if step in budget['checkpoints']:atomic_checkpoint(out/f'model-{step:06d}.pt',model,cfg,opt,step)
        if step%1000==0:atomic_checkpoint(out/'latest.pt',model,cfg,opt,step)
        if stop[0] or time.monotonic()-started>19800:
            atomic_checkpoint(out/'interrupted.pt',model,cfg,opt,step);raise RuntimeError('Incomplete run saved; block evaluation of unequal update budgets')
    save(out/'config.json',cfg)
    save(out/'summary.json',dict(completed=True,arm=arm,updates=len(order),checkpoints=budget['checkpoints'],
        common_initial_state=initial,order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters()),initializer='fresh seed211039, no external data'))


def load_zero(root,name):
    from models import build_model
    from models.checkpoint import load_checkpoint_file
    plan=json.loads((root/'plan.json').read_text());item=plan['checkpoints'][name]
    assert sha(item['path'])==item['sha256']
    cp=load_checkpoint_file(item['path']);cfg=copy.deepcopy(cp['config'])
    target=cfg['data']['datasets']['synthetic_mri'];target.update(root=plan['dataset'],train_samples=None,validation_samples=None,patch_selection='all',spline_controls=8,spline_family='bspline')
    cfg['data']['datasets']={'synthetic_mri':target};cfg['domain_adaptation']['enabled']=False
    cfg['data'].update(input_modality='raw_mra',train_augmentation=False,batch_size=1,validation_batch_size=1)
    cfg['runtime']['workers']=0;cfg['tracking']['enabled']=False
    assert cfg['training']['input']=='image' and cfg['model']['spline']['enabled']
    model=build_model(cfg);model.load_state_dict(cp['net'],strict=True);del cp
    return model.cuda().eval().requires_grad_(False),cfg


def predict_checkpoint(root,name,model,cfg,out,smoke=False,training=False):
    from data.loaders import build_data_loaders,build_evaluation_loader
    from scripts.frozen_spline_diagnostic import training_evaluation_loader
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    plan=json.loads((root/'plan.json').read_text());cfg=copy.deepcopy(cfg)
    cfg['data'].update(train_augmentation=False,batch_size=1,validation_batch_size=1);cfg['runtime']['workers']=0
    out.mkdir(parents=True,exist_ok=False);state=state_digest(model);records=[]
    populations={'val':plan['smoke_ids']['val'] if smoke else plan['ids']['val']}
    if training:populations['train']=plan['smoke_ids']['train'] if smoke else plan['train_monitor_ids']
    with torch.no_grad():
        for split,names in populations.items():
            if split=='val':loader=build_evaluation_loader(cfg,dataset_name='synthetic_mri',sample_ids=names)
            else:
                full,_=build_data_loaders(cfg);loader=training_evaluation_loader(full,cfg,names)
            for index,batch in enumerate(loader):
                sample=names[index];volumes,_=_move_graph_batch(batch,'cuda','image',True)
                tokens,pred,_=model(volumes);metric=batch[4][0]['metric'].to(tokens)
                kw=dict(object_queries=model.obj_token,relation_tokens=cfg['model']['decoder']['relation_tokens'],
                    node_threshold=cfg['evaluation'].get('node_threshold'),node_type_degree_caps=cfg['evaluation'].get('node_type_degree_caps',False),
                    node_type_degree_penalty=cfg['evaluation'].get('node_type_degree_penalty',0.))
                normal=infer_graphs(tokens,pred,model.relation_embed,edge_threshold=cfg['evaluation'].get('edge_threshold'),
                    spline_head=model.spline_head,spline_metrics=[metric],**kw)[0]
                all_pairs=infer_graphs(tokens,pred,model.relation_embed,edge_threshold=-1.,**dict(kw,node_type_degree_caps=False,node_type_degree_penalty=0.))[0]
                np.testing.assert_array_equal(normal['query_ids'],all_pairs['query_ids'])
                c={k:v.numpy() for k,v in normal.items() if torch.is_tensor(v)}
                c.update(raw_nodes=pred['pred_nodes'][0,:,:3].cpu().numpy(),raw_scores=pred['pred_logits'][0].softmax(-1)[:,1].cpu().numpy(),
                    candidate_edges=all_pairs['edges'].numpy(),candidate_scores=all_pairs['edge_scores'].numpy())
                lookup={tuple(sorted(map(int,e))):float(s) for e,s in zip(c['candidate_edges'],c['candidate_scores'])}
                np.testing.assert_allclose([lookup[tuple(sorted(map(int,e)))] for e in c['edges']],c['edge_scores'],rtol=2e-6,atol=2e-6)
                path=out/(sample+'.npz');np.savez_compressed(path,**c)
                records.append(dict(sample_id=sample,split=split,subject=plan['subjects_by_id'][sample],cache=str(path),
                    gt=str(Path(plan['dataset'])/split/'spline'/(sample+'.npz'))))
                if (index+1)%128==0:print(name,split,index+1,'/',len(names),flush=True)
    assert state_digest(model)==state, 'Frozen prediction changed model state'
    save(out/'config.json',cfg);save(out/'records.json',records)
    save(out/'summary.json',dict(completed=True,name=name,records=len(records),frozen_state=True,batching=1,
        object_queries=model.obj_token,edge_threshold=cfg['evaluation'].get('edge_threshold'),node_threshold=cfg['evaluation'].get('node_threshold')))


def predictions(root,name,smoke=False):
    from models import build_model
    from models.checkpoint import load_checkpoint_file
    if name not in ARMS:
        model,cfg=load_zero(root,name);predict_checkpoint(root,name,model,cfg,root/('smoke-predictions' if smoke else 'predictions')/name/'zero',smoke)
    else:
        summary=dict(checkpoints=[2]) if smoke else json.loads((root/'runs'/name/'summary.json').read_text())
        for step in summary['checkpoints']:
            path=root/'smoke'/(name+'.pt') if smoke else root/'runs'/name/f'model-{step:06d}.pt'
            cp=load_checkpoint_file(path);cfg=cp['config'];model=build_model(cfg).cuda();model.load_state_dict(cp['net'],strict=True);del cp
            model.eval().requires_grad_(False)
            predict_checkpoint(root,name,model,cfg,root/('smoke-predictions' if smoke else 'predictions')/name/f'{step:06d}',smoke,True)
            del model;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['smoke','benchmark','train','predict'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--arm');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    if a.mode in ['smoke','benchmark']:benchmark(a.root,a.mode=='smoke')
    elif a.mode=='train':train(a.root,a.arm)
    else:predictions(a.root,a.arm,a.smoke)
