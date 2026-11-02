"""Bounded two-seed local-refinement continuation, reusing established evaluation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts import fine_feature_architecture as prior
from scripts.node_verifier_feasibility import write
from scripts.multiscale_pretrained import graph_cache,rows_for,summarize

ROOT=Path('experiments/local_refinement_long')
SOURCE=Path('experiments/fine_feature_architecture/runs/baseline/best.pt')
SOURCE_SHA='d4bc78ad3eb7c1c29c9be8ec3e2927e88f2f45e6f34a3cb8be9ade5fd9d2431c'
ARMS=['B0','A','A-xyz','A-control']
SEEDS=[364505,92741]


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare():
    assert not (ROOT/'plan.json').exists(),'Preserve existing protocol'
    p=read('experiments/fine_feature_architecture/plan.json')
    p.update(seeds=SEEDS,arms=ARMS,source=str(SOURCE),source_sha256=SOURCE_SHA,updates=12000,
             checkpoints=[3000,6000,9000,12000],warmup=100,peak_lr=7e-6,floor_lr=1e-6,
             primary_selection='fixed final update 12000',secondary_selection=['full validation edge_f1 2mm','full validation node_f1 1mm'],
             max_h100=2,precision_targets=[.5,.7,.8,.9],bootstrap_draws=2000,bootstrap_seed=251026)
    assert sha(SOURCE)==SOURCE_SHA
    write(ROOT/'plan.json',p)


def setup(arm,seed,augment=True):
    from models import build_model
    from models.checkpoint import load_checkpoint_file
    from models.matcher import build_matcher
    from models.ops.modules import MSDeformAttn
    from scripts.train_ixi_transfer import TransferCriterion
    from training.optim import build_optimizer
    from data.loaders import build_data_loaders
    from configs import validate_config
    p=read(ROOT/'plan.json');cfg=copy.deepcopy(p['config']);prior.SEED=seed;prior.ROOT=ROOT/str(seed)
    cfg['experiment'].update(seed=seed,name=f'local_long_{arm}_{seed}')
    cfg['training']['optimizer']['base_lr']=p['peak_lr'];cfg['data']['train_augmentation']=augment
    cfg['model']['node_refinement']={'enabled':False} if arm=='B0' else dict(enabled=True,image_mode='zero' if arm=='A-control' else 'full',radius_vox=4.,refine_classification=arm!='A-xyz',compact_output=arm=='A-xyz',detach_features=False)
    validate_config(cfg);prior.seed();model=build_model(cfg);cp=load_checkpoint_file(SOURCE)
    for key in ['loss','augmentation']:assert cfg[key]==cp['config'][key]
    assert cfg['model']['matcher']==cp['config']['model']['matcher']
    assert cfg['model']['decoder']==cp['config']['model']['decoder']
    assert cfg['model']['spline']==cp['config']['model']['spline']
    assert cfg['data']['datasets']==cp['config']['data']['datasets']
    assert cfg['loss']['node']['classification']['name']=='focal' and cfg['loss']['node']['classification']['focal_gamma']==2
    assert cfg['loss']['edge']['classification']['name']=='cross_entropy'
    loaded=model.load_state_dict(cp['net'],strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('node_refinement.') for k in loaded.missing_keys)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in cp['net'].items())
    del cp
    model.cuda().requires_grad_(True)
    for layer in model.decoder.decoder.layers:
        if layer.rln_attn:layer.cross_attn2.requires_grad_(False);layer.norm12.requires_grad_(False)
    for m in model.modules():
        if isinstance(m,MSDeformAttn):m.record_level_diagnostics=True
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda();criterion.set_training_progress(1,0.)
    loader,_=build_data_loaders(cfg);loader.worker_init_fn=prior.worker_init
    assert [r.sample_id for r in loader.dataset.datasets[0].records]==p['dataset']['train_ids']
    return dict(plan=p,cfg=cfg,model=model,criterion=criterion,opt=build_optimizer(cfg,model),loader=loader)


def hash_tree(h,x):
    """Hash complete images/targets, with structure, shape and dtype boundaries."""
    if torch.is_tensor(x):
        a=x.detach().cpu().contiguous().numpy();h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
    elif isinstance(x,dict):
        for k in sorted(x):h.update(k.encode());hash_tree(h,x[k])
    elif isinstance(x,(list,tuple)):
        h.update(str(len(x)).encode())
        for v in x:hash_tree(h,v)
    else:h.update(json.dumps(x,sort_keys=True).encode())


def evaluate(bundle,arm,label,**kwargs):
    return prior.evaluate(bundle,arm,label,cache_archive=True,**kwargs)


def correction_intervention(model,mode):
    if model.node_refinement is None or mode not in ['no-confidence','no-coordinate']:return None
    def hook(m,args,result):return (result[0],args[2]) if mode=='no-confidence' else (args[1],result[1])
    return model.node_refinement.register_forward_hook(hook)


def smoke():
    from training.engine import _move_graph_batch
    audit=dict(completed=False,job_id=os.getenv('SLURM_JOB_ID'),arms={});reference=None
    for arm in ARMS:
        b=setup(arm,SEEDS[0],False);model=b['model'];ds=b['loader'].dataset.datasets[0];lookup={r.sample_id:i for i,r in enumerate(ds.records)}
        v,y=_move_graph_batch(b['loader'].collate_fn([ds[lookup[s]] for s in b['plan']['tiny_ids']]),'cuda','image',True)
        model.eval()
        with torch.no_grad():
            t,p,_=model(v);current={k:p[k].clone() for k in ['pred_nodes','pred_logits']}
            if reference is None:reference=current
            else:
                for k in current:torch.testing.assert_close(current[k],reference[k],rtol=0,atol=0)
        info=dict(initial_prediction_parity=True,parameters=sum(p.numel() for p in model.parameters()),refinement_parameters=0 if model.node_refinement is None else sum(p.numel() for p in model.node_refinement.parameters()))
        for step in range(256):
            prior.seed(step);model.train();b['opt'].zero_grad(set_to_none=True)
            for group in b['opt'].param_groups:group['lr']=7e-5
            t,p,_=model(v);loss=b['criterion'](t,p,y);assert torch.isfinite(loss['total']);loss['total'].backward()
            if step in [0,1,255]:info['gradient_'+str(step)]=prior.diagnostic(model,loss,step)
            b['opt'].step()
        model.eval()
        with torch.no_grad():
            t,p,_=model(v);info['tiny_metrics']=summarize([rows_for(graph_cache(model,b['criterion'],t,p,y,i))['2.0'] for i in range(8)])
            if arm=='A-xyz':torch.testing.assert_close(p['pred_logits'],model.class_embed(t[:,:model.obj_token]),rtol=0,atol=0)
        assert info['tiny_metrics']['node_f1']>=.95 and info['tiny_metrics']['edge_f1']>=.90
        if arm in ['A','A-xyz']:
            feat=model._spline_local_features.detach().requires_grad_(True)
            xyz,logits=model.node_refinement(t[:,:model.obj_token].detach(),p['pred_nodes'].detach(),p['pred_logits'].detach(),feat,(64,64,64))
            grad=torch.autograd.grad(xyz.sum()+logits.sum(),feat)[0];info['isolated_image_gradient']=float(grad.norm());assert grad.norm()>0
        model.train();b['opt'].zero_grad(set_to_none=True);v,y=_move_graph_batch(b['loader'].collate_fn([ds[i] for i in range(32)]),'cuda','image',True)
        hash_tree(hashlib.sha256(),v);hash_tree(hashlib.sha256(),y)
        t,p,_=model(v);loss=b['criterion'](t,p,y);loss['total'].backward();info['batch32']=prior.diagnostic(model,loss,256)
        info['peak_gpu_gib']=torch.cuda.max_memory_allocated()/2**30;audit['arms'][arm]=info;write(ROOT/'smoke.json',audit)
        del b,model,t,p,loss,v,y;torch.cuda.empty_cache()
    audit['completed']=True;write(ROOT/'smoke.json',audit)


def train(arm,seed,resume_step=0):
    from training.engine import _move_graph_batch
    from scripts.extend_scratch_geometry import atomic_checkpoint
    from models.checkpoint import load_checkpoint_file
    assert read(ROOT/'smoke.json')['completed'];assert sha(SOURCE)==SOURCE_SHA
    b=setup(arm,seed);model=b['model'];plan=b['plan'];out=prior.ROOT/'runs'/arm;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'complete.json').exists(),'Preserve completed runs'
    step=epoch=0;digest=hashlib.sha256();history=[];evaluations={};best=(-1.,-1.);best_step=0;start=time.time();old_history=[]
    if resume_step:
        assert resume_step==9000 and seed==364505 and arm in ['A-xyz','A-control']
        cp=load_checkpoint_file(out/f'model-{resume_step}.pt')
        assert cp['steps']==resume_step and cp['config']==b['cfg']
        model.load_state_dict(cp['net'],strict=True);b['opt'].load_state_dict(cp['optimizer']);del cp
        old_history=read(out/'training.json');write(out/'training-interrupted.json',old_history)
        history=[x for x in old_history if x['step']<=resume_step]
        for saved_step in plan['checkpoints']:
            if saved_step>resume_step:continue
            folder=out/f'step-{saved_step}'
            if (folder/'summary.json').exists():result=read(folder/'summary.json')
            else:
                assert saved_step==resume_step
                if (folder/'caches.zip').exists():(folder/'caches.zip').rename(folder/'interrupted-caches.zip')
                result=evaluate(b,arm,f'step-{saved_step}')
            score=(result['metrics']['val']['2.0']['edge_f1'],result['metrics']['val']['1.0']['node_f1'])
            if score>best:best=score;best_step=saved_step
        write(out/'recovery.json',dict(resume_step=resume_step,checkpoint_sha256=sha(out/f'model-{resume_step}.pt'),job_id=os.getenv('SLURM_JOB_ID'),reason='Project inode quota; replay data only to reconstruct full batch hash'))
    else:
        assert not (out/'training.json').exists(),'Preserve previous run; investigate before restarting'
        write(out/'config.json',b['cfg']);evaluate(b,arm,'initial',small=True)
    while step<plan['updates']:
        epoch+=1;b['loader'].generator.manual_seed(seed+epoch)
        for batch in b['loader']:
            if step>=plan['updates']:break
            step+=1;prior.seed(step);v,y=_move_graph_batch(batch,'cpu' if step<=resume_step else 'cuda','image',True)
            hash_tree(digest,v);hash_tree(digest,y)
            if step<=resume_step:
                if step%1000==0:print('data replay',arm,seed,step,flush=True)
                if step==resume_step:
                    reference=read(ROOT/'recovery-reference-9000.json')
                    assert reference['step']==resume_step and digest.hexdigest()==reference['sha256'],'Resumed data stream differs'
                    write(out/'replay-proof.json',dict(step=step,sha256=digest.hexdigest(),matched=True))
                continue
            lr=plan['peak_lr']*step/plan['warmup'] if step<=plan['warmup'] else plan['floor_lr']+(plan['peak_lr']-plan['floor_lr'])*(1-(step-plan['warmup'])/(plan['updates']-plan['warmup']))**.9
            for g in b['opt'].param_groups:g['lr']=lr
            model.train();b['opt'].zero_grad(set_to_none=True);t,p,_=model(v);loss=b['criterion'](t,p,y);assert torch.isfinite(loss['total']);loss['total'].backward()
            if step in [1,100,500] or step%1000==0:
                row=prior.diagnostic(model,loss,step);row['lr']=lr;history.append(row);write(out/'training.json',history)
                original=next((x for x in old_history if x['step']==step),None)
                if original is not None:
                    delta=abs(row['losses']['total']-original['losses']['total'])
                    with (out/'recovery-loss-parity.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,total_loss_absolute_difference=delta))+'\n')
                print(arm,seed,step,float(loss['total']),flush=True)
            b['opt'].step()
            if step%100==0:write(out/'progress.json',dict(step=step,epoch=epoch,seconds=time.time()-start,loss=float(loss['total']),lr=lr))
            if step in plan['checkpoints']:
                payload=dict(net=model.state_dict(),config=b['cfg'],steps=step,seed=seed,source=str(SOURCE),source_sha256=SOURCE_SHA,optimizer=b['opt'].state_dict())
                atomic_checkpoint(out/f'model-{step}.pt',payload)
                result=evaluate(b,arm,f'step-{step}');evaluations[str(step)]=result
                score=(result['metrics']['val']['2.0']['edge_f1'],result['metrics']['val']['1.0']['node_f1'])
                if score>best:best=score;best_step=step
                write(out/'selection.json',dict(primary_step=plan['updates'],secondary_best_step=best_step,secondary_best_score=best))
                write(out/'batch-proof.json',dict(step=step,sha256=digest.hexdigest(),scope='all images and complete targets'))
    if resume_step:assert digest.hexdigest()==read(ROOT/'364505/runs/B0/complete.json')['batch_sha256'],'Full resumed batch sequence differs'
    prior.reconstruction(b,arm,'reconstruction-final')
    if arm in ['A','A-xyz']:
        modes=['zero','roll']+(['no-confidence','no-coordinate'] if arm=='A' else [])
        for mode in modes:
            hook=correction_intervention(model,mode)
            try:
                evaluate(b,arm,'final-'+mode,intervention=mode)
                prior.reconstruction(b,arm,'reconstruction-'+mode,intervention=mode)
            finally:
                if hook is not None:hook.remove()
    if best_step!=plan['updates']:
        cp=load_checkpoint_file(out/f'model-{best_step}.pt');model.load_state_dict(cp['net']);del cp
        prior.reconstruction(b,arm,'reconstruction-selected')
    write(out/'complete.json',dict(completed=True,arm=arm,seed=seed,steps=step,selected_step=best_step,job_id=os.getenv('SLURM_JOB_ID'),array_job_id=os.getenv('SLURM_ARRAY_JOB_ID'),array_task_id=os.getenv('SLURM_ARRAY_TASK_ID'),batch_sha256=digest.hexdigest(),seconds=time.time()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))


def archive():
    path=ROOT/f"source-{os.getenv('SLURM_JOB_ID','local')}.zip"
    if path.exists():return
    paths=[p for folder in ['models','training','data','configs','scripts'] for p in Path(folder).rglob('*.py')]
    paths += [ROOT/'plan.json',ROOT/'execution_plan.md',Path('cluster/jean_zay/local_refinement_long.slurm')]
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);p.add_argument('--resume-step',type=int,default=0);a=p.parse_args();torch.set_num_threads(1)
    if a.mode=='prepare':prepare();return
    archive()
    if a.mode=='smoke':smoke()
    else:train(ARMS[a.index%4],SEEDS[a.index//4],a.resume_step)


if __name__=='__main__':main()
