"""One scheduled scratch baseline and gated, matched fine-feature pilots."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.multiscale_pretrained import graph_cache, rows_for, summarize, levels, gradients
from scripts.node_verifier_metrics import discrimination
from scripts.node_verifier_feasibility import write
from models.multiscale_migration import migrate_single_level, set_fine_enabled

ROOT=Path('experiments/fine_feature_architecture')
SEED=364505
ARMS=['B0','A','A-control','C']


def read(p):return json.loads(Path(p).read_text())
def seed(step=0):
    random.seed(SEED+step);np.random.seed(SEED+step);torch.manual_seed(SEED+step)


def prepare():
    from configs import validate_config
    old=read('experiments/multiscale_pretrained/plan.json');cfg=copy.deepcopy(old['config']);spec=copy.deepcopy(old['dataset'])
    spec['train_ids']=sorted(p.stem for p in (Path(spec['root'])/'train'/'spline').glob('*.npz'))
    assert len(spec['train_ids'])==25000 and len(spec['val_ids'])==480
    subjects=lambda names:{spec['subjects'][s.split('_')[1]] for s in names}
    assert not subjects(spec['train_ids'])&subjects(spec['val_ids'])
    cfg['data'].update(batch_size=32,validation_batch_size=1,train_augmentation=True)
    cfg['runtime']['workers']=4;cfg['experiment'].update(seed=SEED,name='fine_feature_scratch')
    cfg['training'].update(epochs=100,warmup_epochs=1)
    cfg['training']['optimizer']['base_lr']=7e-5
    cfg['model']['decoder']['feature_levels']=1;cfg['model']['node_refinement']={'enabled':False}
    # Deliberately retain the known immediate node-focal / edge-CE recipe.
    assert cfg['loss']['node']['classification']['name']=='focal'
    assert cfg['loss']['node']['classification']['class_weights']==[1.,1.]
    assert cfg['loss']['node']['classification']['focal_gamma']==2
    assert not cfg['loss']['node']['classification']['curriculum']['enabled']
    assert cfg['loss']['edge']['classification']['name']=='cross_entropy'
    assert not cfg['loss']['edge']['candidates']['include_unmatched']
    assert cfg['loss']['edge']['balancing']['mode']=='ratio_upsample'
    assert cfg['model']['spline']['controls']==8
    validate_config(cfg)
    plan=dict(config=cfg,dataset=spec,seed=SEED,baseline_epochs=100,baseline_updates=100*math.ceil(25000/32),
        baseline_initializer='random; configuration lineage only, no trained tensor loaded',gradient_route='graph_only',
        tiny_ids=old['tiny_ids'],tiny_updates=1500,baseline_batch=32,pilot_updates=3000,pilot_batch=32,pilot_peak_lr=7e-6,
        pilot_warmup=100,pilot_floor_lr=1e-6,selection=['validation edge_f1 at 2 mm','validation node_f1 at 1 mm'],
        quality_gate=dict(raw_recall=.82,node_f1=.70,edge_f1=.40,conditional_recall=.70,hungarian_mm_max=1.8),
        fine_stage='layer2',fine_prior=.1,fine_warmup=500,fine_initial_prior=1e-4,historical=old['checkpoint'],max_concurrent_h100=2)
    write(ROOT/'plan.json',plan);print('prepared',len(spec['train_ids']),len(spec['val_ids']),plan['baseline_updates'],flush=True)


def worker_init(worker_id):
    from data.loaders.common import seed_data_worker
    seed_data_worker(worker_id)
    def initialize(ds):
        if getattr(ds,'intensity_transform',None) is not None:ds.intensity_transform.set_random_state(seed=torch.initial_seed()%(2**32))
        for sub in getattr(ds,'datasets',[]):initialize(sub)
    initialize(torch.utils.data.get_worker_info().dataset)


def setup(arm='baseline',source=None,augment=True):
    from configs import validate_config
    from models import build_model
    from models.matcher import build_matcher
    from models.checkpoint import load_checkpoint_file
    from models.ops.modules import MSDeformAttn
    from scripts.train_ixi_transfer import TransferCriterion
    from data.loaders import build_data_loaders
    from training.optim import build_optimizer
    plan=read(ROOT/'plan.json');cfg=copy.deepcopy(plan['config'])
    cfg['data']['train_augmentation']=augment;cfg['experiment']['name']='fine_feature_'+arm
    if arm in ['A','A-control']:
        cfg['model']['node_refinement']=dict(enabled=True,image_mode='full' if arm=='A' else 'zero',radius_vox=4.,refine_classification=True,detach_features=False)
    if arm=='C':cfg['model']['decoder'].update(feature_levels=2,fine_feature_stage=plan['fine_stage'])
    if arm in ARMS:cfg['training']['optimizer']['base_lr']=plan['pilot_peak_lr']
    validate_config(cfg);seed();model=build_model(cfg);migration={'random_initialization':True}
    if source is not None:
        cp=load_checkpoint_file(source)
        own_baseline=Path(source).resolve()==(ROOT/'runs'/'baseline'/'best.pt').resolve()
        if own_baseline:
            assert cp['config']['loss']==cfg['loss'],'Pilot would change the baseline objective'
            assert cp['config']['model']['matcher']==cfg['model']['matcher']
            assert cp['config']['model']['decoder']['object_queries']==cfg['model']['decoder']['object_queries']
            assert cp['config']['data']['datasets']==cfg['data']['datasets']
            assert cp['config']['augmentation']==cfg['augmentation']
        if arm=='C':migration=migrate_single_level(model,cp['net'],plan['fine_prior'])
        else:
            report=model.load_state_dict(cp['net'],strict=False)
            assert not report.unexpected_keys
            assert all(n.startswith('node_refinement.') for n in report.missing_keys)
            if arm not in ['A','A-control']:assert not report.missing_keys
            migration=dict(source=str(source),new_tensors=report.missing_keys)
        migration['source_objective_verified']=own_baseline
        del cp
    model.cuda().requires_grad_(True)
    for layer in model.decoder.decoder.layers:
        if layer.rln_attn:layer.cross_attn2.requires_grad_(False);layer.norm12.requires_grad_(False)
    for m in model.modules():
        if isinstance(m,MSDeformAttn):m.record_level_diagnostics=True
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda()
    criterion.set_training_progress(1,0.)
    opt=build_optimizer(cfg,model);loader,_=build_data_loaders(cfg);loader.worker_init_fn=worker_init
    assert [r.sample_id for r in loader.dataset.datasets[0].records]==plan['dataset']['train_ids']
    return dict(plan=plan,cfg=cfg,model=model,criterion=criterion,opt=opt,loader=loader,migration=migration)


def local_intervention(model,mode):
    """Alter only A's image inputs, leaving the spline feature map unchanged."""
    if model.node_refinement is None or mode not in ['zero','roll']:return None
    def hook(module,args):
        args=list(args);f=args[3]
        args[3]=torch.zeros_like(f) if mode=='zero' else torch.roll(f,(4,4,4),(-3,-2,-1))
        return tuple(args)
    return model.node_refinement.register_forward_pre_hook(hook)


def fine_warmup(model,step,length):
    from models.ops.modules import MSDeformAttn
    offset=math.log(1e-4/.1)*max(0.,1-step/length)
    for m in model.modules():
        if isinstance(m,MSDeformAttn):m.fine_logit_offset=offset


@torch.no_grad()
def evaluate(bundle,arm,label,small=False,intervention='normal',save=True,cache_archive=False):
    import io
    import zipfile
    from scripts.multidataset_graph_evaluation import loaders
    from training.engine import _move_graph_batch
    from scripts.train_spline_weight_pilot import monitoring_rng
    model=bundle['model'];criterion=bundle['criterion'];plan=bundle['plan'];model.eval()
    model.fine_feature_mode='roll' if arm=='C' and intervention=='roll' else 'normal'
    set_fine_enabled(model,intervention!='blocked');hook=local_intervention(model,intervention)
    spec=copy.deepcopy(plan['dataset']);spec['monitor_train']=spec['monitor_train'][:32 if small else 128]
    if small:spec['monitor_val']=spec['monitor_val'][:64]
    out=ROOT/'runs'/arm/label;out.mkdir(parents=True,exist_ok=True);rows=[];records=[];cs={'train':[],'val':[]};attention=[]
    archive=zipfile.ZipFile(out/'caches.zip','w',compression=zipfile.ZIP_STORED) if save and cache_archive else None
    try:
        with monitoring_rng():
            for split,names,loader in loaders(bundle['cfg'],dict(datasets={'SyntheticMRI':spec}),dict(dataset='SyntheticMRI'),monitor=small or intervention!='normal'):
                if intervention!='normal' and split=='train':continue
                for i,batch in enumerate(loader):
                    volumes,targets=_move_graph_batch(batch,'cuda','image',True);tokens,pred,_=model(volumes)
                    c=graph_cache(model,criterion,tokens,pred,targets);sample=names[i];subject=spec['subjects'][sample.split('_')[1]]
                    for tol,r in rows_for(c).items():rows.append(dict(r,sample=sample,subject=subject,split=split,tolerance=tol))
                    cs[split].append(c);attention.append(levels(model));path=out/(sample+'.npz')
                    if save:
                        target=io.BytesIO() if archive is not None else path
                        np.savez_compressed(target,**{k:v for k,v in c.items() if isinstance(v,np.ndarray)})
                        if archive is not None:archive.writestr(path.name,target.getvalue())
                    records.append(dict(sample=sample,subject=subject,split=split,cache=str(path)))
        result=dict(metrics={s:{t:summarize([r for r in rows if r['split']==s and r['tolerance']==t]) for t in ['1.0','2.0']} for s in cs if cs[s]},
            discrimination={s:discrimination(c,[v['scores'] for v in c]) for s,c in cs.items() if c},
            attention={n:{field:np.mean([a[n][field] for a in attention],axis=0).tolist() for field in ['mass','inside_mass']} for n in attention[0]})
        write(out/'summary.json',result);write(out/'rows.json',rows);write(out/'records.json',records)
        return result
    finally:
        if archive is not None:archive.close()
        if hook is not None:hook.remove()
        model.fine_feature_mode='normal';set_fine_enabled(model,True)


def diagnostic(model,losses,step):
    g=gradients(model)
    for prefix in ['node_refinement.']:
        values=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        if values:g[prefix]=float(sum(v.square().sum() for v in values).sqrt())
    assert all(g.get(k,0)>0 for k in ['encoder.','decoder.','class_embed.','coord_embed.','relation_embed.','spline_head.'])
    return dict(step=step,losses={k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1},gradients=g,attention=levels(model))


def smoke():
    from training.engine import _move_graph_batch
    b=setup(augment=False);model=b['model'];dataset=b['loader'].dataset.datasets[0];lookup={r.sample_id:i for i,r in enumerate(dataset.records)}
    samples=[dataset[lookup[s]] for s in b['plan']['tiny_ids']]
    volumes,targets=_move_graph_batch(b['loader'].collate_fn(samples),'cuda','image',True)
    # A true random initialization; no saved net was loaded by setup.
    history=[];audit=dict(job_id=os.getenv('SLURM_JOB_ID'),random_initialization=True,completed=False)
    for step in range(b['plan']['tiny_updates']+1):
        if step in [0,250,750,1500]:
            model.eval()
            with torch.no_grad():
                t,p,_=model(volumes);rr=[rows_for(graph_cache(model,b['criterion'],t,p,targets,i))['2.0'] for i in range(len(volumes))]
            history.append(dict(step=step,metrics=summarize(rr)));write(ROOT/'tiny-fit.json',history);print('tiny',history[-1],flush=True)
        if step==b['plan']['tiny_updates']:break
        seed(step);model.train();b['opt'].zero_grad(set_to_none=True)
        for group in b['opt'].param_groups:group['lr']=7e-5*min(1.,(step+1)/20)
        t,p,_=model(volumes);loss=b['criterion'](t,p,targets);assert torch.isfinite(loss['total']);loss['total'].backward()
        if step in [0,250]:audit[str(step)]=diagnostic(model,loss,step);write(ROOT/'smoke.json',audit)
        b['opt'].step()
    last=history[-1]['metrics'];assert last['node_f1']>=.95 and last['edge_f1']>=.90,'Random-model tiny-set learnability failed'
    # Full batch forward/backward verifies memory before releasing the long run.
    large=b['loader'].collate_fn([dataset[i] for i in range(32)]);v,y=_move_graph_batch(large,'cuda','image',True)
    model.train();b['opt'].zero_grad(set_to_none=True);t,p,_=model(v);loss=b['criterion'](t,p,y);loss['total'].backward();audit['batch32']=diagnostic(model,loss,1501)
    audit.update(completed=True,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30);write(ROOT/'smoke.json',audit)


def baseline(until_epoch):
    from training.optim import build_scheduler
    from training.engine import train_step
    from models.checkpoint import load_checkpoint_file
    from scripts.extend_scratch_geometry import atomic_checkpoint,capture_rng,restore_rng
    assert read(ROOT/'smoke.json')['completed']
    b=setup();model=b['model'];loader=b['loader'];cfg=b['cfg'];out=ROOT/'runs'/'baseline';out.mkdir(parents=True,exist_ok=True)
    write(out/'config.json',cfg);scheduler=build_scheduler(cfg,b['opt'],len(loader));epoch=step=0;best=(-1.,-1.);best_epoch=0
    if (out/'latest.pt').exists():
        cp=load_checkpoint_file(out/'latest.pt');model.load_state_dict(cp['net'],strict=True);b['opt'].load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);restore_rng(cp['rng'],loader)
        epoch=cp['epoch'];step=cp['steps'];best=tuple(cp['best']);best_epoch=cp['best_epoch'];del cp
        assert scheduler.last_epoch==step
    start=time.time()
    def payload():return dict(net=model.state_dict(),config=cfg,epoch=epoch,steps=step,optimizer=b['opt'].state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(loader),best=best,best_epoch=best_epoch,plan=b['plan'])
    for epoch in range(epoch+1,min(until_epoch,100)+1):
        loader.generator.manual_seed(SEED+epoch);total=0.;n=0;started=time.time()
        for batch in loader:
            step+=1;seed(step)
            losses=train_step(model,b['criterion'],b['opt'],scheduler,batch,cfg,'cuda',epoch=epoch,iteration=step,total_iterations=b['plan']['baseline_updates'])
            total+=float(losses['total'].detach());n+=1
            if step==1 or step%5000==0:
                record=diagnostic(model,losses,step);record.update(epoch=epoch,lr=scheduler.get_last_lr())
                with (out/'gradients.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if step%100==0:write(out/'progress.json',dict(epoch=epoch,step=step,lr=scheduler.get_last_lr(),loss=float(losses['total']),job_id=os.getenv('SLURM_JOB_ID')))
        row=dict(epoch=epoch,step=step,mean_train_loss=total/n,lr=scheduler.get_last_lr(),epoch_seconds=time.time()-started,job_id=os.getenv('SLURM_JOB_ID'))
        if epoch%5==0:
            metrics=evaluate(b,'baseline',f'epoch-{epoch:03d}',save=False);row['evaluation']=metrics
            score=(metrics['metrics']['val']['2.0']['edge_f1'],metrics['metrics']['val']['1.0']['node_f1'])
            if score>best:best=score;best_epoch=epoch;atomic_checkpoint(out/'best.pt',payload())
        row.update(best_epoch=best_epoch,best=best)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        atomic_checkpoint(out/'latest.pt',payload());print('epoch',json.dumps(row),flush=True)
    write(out/f'segment-{until_epoch}.json',dict(epoch=epoch,step=step,best_epoch=best_epoch,best=best,seconds=time.time()-start,job_id=os.getenv('SLURM_JOB_ID')))
    if epoch==100:
        cp=load_checkpoint_file(out/'best.pt');model.load_state_dict(cp['net'],strict=True);del cp
        result=evaluate(b,'baseline','selected');s=result['metrics']['val']['2.0'];thresholds=b['plan']['quality_gate']
        checks={k:bool(s[k]>=v) for k,v in thresholds.items() if k!='hungarian_mm_max'};checks['hungarian_mm']=bool(s['hungarian_mm']<=thresholds['hungarian_mm_max'])
        gate=dict(passed=all(checks.values()),checks=checks,metrics=s,selected_epoch=best_epoch,selected_checkpoint=str(out/'best.pt'),job_id=os.getenv('SLURM_JOB_ID'))
        write(ROOT/'baseline-gate.json',gate)
        reconstruction(b,'baseline','reconstruction')
        assert gate['passed'],'Baseline failed quality gate; dependent pilots must stop'


@torch.no_grad()
def reconstruction(bundle,arm,label,intervention='normal'):
    from data.loaders import build_evaluation_loader
    from scripts.train_spline_weight_pilot import geometry,monitoring_rng
    model=bundle['model'];cfg=copy.deepcopy(bundle['cfg']);cfg['data']['validation_batch_size']=4
    ids=bundle['plan']['dataset']['val_ids'] if intervention=='normal' else bundle['plan']['dataset']['monitor_val']
    loader=build_evaluation_loader(cfg,dataset_name='synthetic_mri',sample_ids=ids)
    model.fine_feature_mode='roll' if arm=='C' and intervention=='roll' else 'normal';set_fine_enabled(model,intervention!='blocked')
    hook=local_intervention(model,intervention);out=ROOT/'runs'/arm/label;out.mkdir(parents=True,exist_ok=True)
    try:
        with monitoring_rng():r=geometry(model,loader,cfg,out,0,None,0,True)
        write(out/'summary.json',r);return r
    finally:
        if hook is not None:hook.remove()
        model.fine_feature_mode='normal';set_fine_enabled(model,True)


def architecture_smoke(historical=False):
    from training.engine import _move_graph_batch
    from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction,ms_deform_attn_core_pytorch
    plan=read(ROOT/'plan.json')
    if not historical:assert read(ROOT/'baseline-gate.json')['passed']
    source=plan['historical']['path'] if historical else str(ROOT/'runs'/'baseline'/'best.pt')
    audit=dict(completed=False,source=source,job_id=os.getenv('SLURM_JOB_ID'),arms={})
    destination=ROOT/('architecture-preflight.json' if historical else 'architecture-gate.json')
    seed();shapes=torch.tensor([[2,2,2],[8,8,8]],device='cuda');starts=torch.tensor([0,8],device='cuda')
    v=torch.randn(1,520,6,4,device='cuda',requires_grad=True);loc=torch.rand(1,11,6,2,4,3,device='cuda',requires_grad=True);w=torch.rand(1,11,6,2,4,device='cuda',requires_grad=True)
    a=MSDeformAttnFunction.apply(v,shapes,starts,loc,w,64);bb=ms_deform_attn_core_pytorch(v,shapes,loc,w)
    torch.testing.assert_close(a,bb,atol=2e-5,rtol=2e-5)
    ga=torch.autograd.grad(a.square().sum(),[v,loc,w]);gb=torch.autograd.grad(bb.square().sum(),[v,loc,w])
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=5e-4,rtol=5e-4)
    audit['cuda_max_error']=float((a-bb).abs().max());audit['cuda_gradient_errors']=[float((x-y).abs().max()) for x,y in zip(ga,gb)]
    base=setup('B0',source,False);ds=base['loader'].dataset.datasets[0];lookup={r.sample_id:i for i,r in enumerate(ds.records)}
    samples=[ds[lookup[s]] for s in plan['tiny_ids']];volumes,targets=_move_graph_batch(base['loader'].collate_fn(samples),'cuda','image',True)
    base['model'].eval()
    with torch.no_grad():_,original,_=base['model'](volumes);original={k:v.detach().clone() for k,v in original.items() if k in ['pred_nodes','pred_logits']}
    del base;torch.cuda.empty_cache()
    for arm in ARMS:
        b=setup(arm,source,False);model=b['model'];model.eval()
        if arm=='C':fine_warmup(model,0,32 if historical else 64)
        with torch.no_grad():
            if arm=='C':set_fine_enabled(model,False)
            _,same,_=model(volumes)
            for key in original:torch.testing.assert_close(same[key],original[key],atol=5e-5,rtol=5e-5)
            set_fine_enabled(model,True);_,active,coarse= model(volumes)
        info=dict(migration=b['migration'],initial_max_delta={k:float((active[k]-original[k]).abs().max()) for k in original},initial_attention=levels(model))
        if arm=='C':
            assert tuple(model._multiscale_features.shape[1:])==(512,8,8,8)
            info['feature_shape']=list(model._multiscale_features.shape)
            assert all(9e-5<x['mass'][1]<1.1e-4 for x in info['initial_attention'].values())
            info['spatial_feature_std_over_rms']={name:float(f.flatten(2).std(-1).mean()/f.square().mean().sqrt().clamp_min(1e-12)) for name,f in [('fine',model._multiscale_features),('coarse_projected',coarse)]}
        # Independent train-only learnability check; discard these updates.
        for step in range(64 if historical else 256):
            seed(step);model.train();b['opt'].zero_grad(set_to_none=True)
            if arm=='C':fine_warmup(model,step+1,32 if historical else 64)
            for g in b['opt'].param_groups:g['lr']=7e-5
            t,p,_=model(volumes);loss=b['criterion'](t,p,targets);assert torch.isfinite(loss['total']);loss['total'].backward()
            if step in [0,1]:
                info['gradient_'+str(step)]=diagnostic(model,loss,step)
                if arm in ['A','A-control']:assert info['gradient_'+str(step)]['gradients']['node_refinement.']>0
                if arm=='C':
                    assert info['gradient_'+str(step)]['gradients']['fine_input_proj.']>0
                    assert all(x['sampling_offsets']>0 and x['attention_weights']>0 for x in info['gradient_'+str(step)]['gradients']['new_attention_rows'].values())
            b['opt'].step()
        model.eval()
        with torch.no_grad():
            t,p,_=model(volumes);info['tiny_metrics']=summarize([rows_for(graph_cache(model,b['criterion'],t,p,targets,i))['2.0'] for i in range(8)])
            if arm=='C':
                model.fine_feature_mode='roll';_,shifted,_=model(volumes);model.fine_feature_mode='normal'
                set_fine_enabled(model,False);_,blocked,_=model(volumes);set_fine_enabled(model,True)
                info['post_fit_fine_interventions']={mode:{k:float((other[k]-p[k]).abs().mean()) for k in original} for mode,other in [('roll',shifted),('blocked',blocked)]}
        if arm=='A':
            feat=model._spline_local_features.detach().requires_grad_(True)
            xyz,logits=model.node_refinement(t[:,:model.obj_token].detach(),p['pred_nodes'].detach(),p['pred_logits'].detach(),feat,(64,64,64))
            g=torch.autograd.grad(xyz.sum()+logits.sum(),feat)[0];info['isolated_local_feature_gradient']=float(g.norm());assert g.norm()>0
        if not historical:assert info['tiny_metrics']['node_f1']>=.95 and info['tiny_metrics']['edge_f1']>=.90
        # A/C need a full pilot batch memory/gradient smoke too.
        if arm in ['A','C']:
            vv=volumes.repeat(4,1,1,1,1);yy={k:v.repeat(4) if torch.is_tensor(v) else v*4 for k,v in targets.items()}
            model.train();b['opt'].zero_grad(set_to_none=True);t,p,_=model(vv);loss=b['criterion'](t,p,yy);loss['total'].backward();info['batch32_gradient']=diagnostic(model,loss,step+1)
        info['peak_gpu_gib']=torch.cuda.max_memory_allocated()/2**30;audit['arms'][arm]=info;write(destination,audit)
        del b,model,t,p,loss,active,same;torch.cuda.empty_cache()
    audit['completed']=True;write(destination,audit)


def pilot(arm):
    from training.engine import _move_graph_batch
    from scripts.extend_scratch_geometry import atomic_checkpoint
    assert read(ROOT/'baseline-gate.json')['passed'] and read(ROOT/'architecture-gate.json')['completed']
    source=ROOT/'runs'/'baseline'/'best.pt';b=setup(arm,source);model=b['model'];plan=b['plan'];out=ROOT/'runs'/arm;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'complete.json').exists(),'Pilot already completed; preserve it'
    write(out/'config.json',b['cfg']);write(out/'migration.json',b['migration'])
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith(('fine_input_proj.','node_refinement.'))}
    if arm=='C':fine_warmup(model,0,plan['fine_warmup'])
    evaluate(b,arm,'step-0000',small=True);step=epoch=0;digest=hashlib.sha256();start=time.time();history=[]
    while step<plan['pilot_updates']:
        epoch+=1;b['loader'].generator.manual_seed(SEED+epoch)
        for batch in b['loader']:
            if step>=plan['pilot_updates']:break
            step+=1;seed(step)
            volumes,targets=_move_graph_batch(batch,'cuda','image',True)
            digest.update(volumes.cpu().numpy().tobytes())
            for nodes in targets['nodes']:digest.update(nodes.cpu().numpy().tobytes())
            lr=plan['pilot_peak_lr']*step/plan['pilot_warmup'] if step<=plan['pilot_warmup'] else plan['pilot_floor_lr']+(plan['pilot_peak_lr']-plan['pilot_floor_lr'])*(1-(step-plan['pilot_warmup'])/(plan['pilot_updates']-plan['pilot_warmup']))**.9
            for group in b['opt'].param_groups:group['lr']=lr
            if arm=='C':fine_warmup(model,step,plan['fine_warmup'])
            model.train();b['opt'].zero_grad(set_to_none=True);t,p,_=model(volumes);loss=b['criterion'](t,p,targets);assert torch.isfinite(loss['total']);loss['total'].backward()
            if step in [1,100,500,1000,2000,3000]:
                row=diagnostic(model,loss,step);row['lr']=lr;history.append(row);write(out/'training.json',history);print('pilot',arm,step,row['losses']['total'],flush=True)
            b['opt'].step()
            if step in [1000,2000]:evaluate(b,arm,f'step-{step:04d}',small=True)
    atomic_checkpoint(out/'model-3000.pt',dict(net=model.state_dict(),config=b['cfg'],steps=step,source=str(source),migration=b['migration'],optimizer=b['opt'].state_dict()))
    evaluate(b,arm,'final');reconstruction(b,arm,'reconstruction')
    for mode in (['zero','roll'] if arm=='A' else ['blocked','roll'] if arm=='C' else []):
        evaluate(b,arm,'final-'+mode,intervention=mode);reconstruction(b,arm,'reconstruction-'+mode,intervention=mode)
    deltas={n:float((p.detach().cpu()-initial[n]).norm()/initial[n].norm().clamp_min(1e-8)) for n,p in model.named_parameters() if n in initial}
    write(out/'complete.json',dict(completed=True,steps=step,seconds=time.time()-start,job_id=os.getenv('SLURM_JOB_ID'),batch_sha256=digest.hexdigest(),new_parameter_relative_change=deltas,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))


def historical():
    plan=read(ROOT/'plan.json');b=setup('historical',plan['historical']['path'],False)
    evaluate(b,'historical','selected');reconstruction(b,'historical','reconstruction')
    write(ROOT/'historical-complete.json',dict(completed=True,job_id=os.getenv('SLURM_JOB_ID')))


def baseline_reconstruction():
    b=setup('baseline',ROOT/'runs'/'baseline'/'best.pt',False);reconstruction(b,'baseline','reconstruction')
    write(ROOT/'baseline-reconstruction-complete.json',dict(completed=True,job_id=os.getenv('SLURM_JOB_ID')))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','baseline','preflight','architecture-smoke','pilot','historical','baseline-reconstruction']);p.add_argument('--until-epoch',type=int,default=100);p.add_argument('--arm',default='B0');a=p.parse_args();torch.set_num_threads(1)
    ROOT.mkdir(parents=True,exist_ok=True)
    # Preserve the exact source files for future dependent jobs in this dirty
    # research checkout, without modifying or committing unrelated work.
    if os.getenv('SLURM_JOB_ID'):
        import zipfile
        archive=ROOT/('source-'+os.environ['SLURM_JOB_ID']+'.zip')
        if not archive.exists():
            paths=[p for folder in ['models','training','data','configs'] for p in Path(folder).rglob('*.py')]
            paths += [Path('scripts')/name for name in ['fine_feature_architecture.py','multiscale_pretrained.py','node_verifier_metrics.py','node_verifier_feasibility.py','multidataset_graph_evaluation.py','train_ixi_transfer.py','controlled_spline_unfreeze.py','train_spline_weight_pilot.py','extend_scratch_geometry.py','frozen_spline_diagnostic.py']]
            paths += [ROOT/'plan.json',Path('cluster/jean_zay/fine_feature_architecture.slurm')]
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for path in paths:z.write(path,str(path))
    if a.mode=='prepare':prepare()
    elif a.mode=='smoke':smoke()
    elif a.mode=='baseline':baseline(a.until_epoch)
    elif a.mode=='preflight':architecture_smoke(True)
    elif a.mode=='architecture-smoke':architecture_smoke()
    elif a.mode=='historical':historical()
    elif a.mode=='baseline-reconstruction':baseline_reconstruction()
    else:pilot(ARMS[int(a.arm)] if a.arm.isdigit() else a.arm)


if __name__=='__main__':main()
