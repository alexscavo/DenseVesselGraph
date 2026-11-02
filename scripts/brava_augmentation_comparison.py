"""Matched original-K8 augmentation experiment; imports the frozen diagnosis code."""
import argparse
import copy
import hashlib
import io
import json
import os
import pickle
import random
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIAG = REPO.parent / 'gnbm-brava-learning-diagnosis-20261026'
sys.path.insert(0, str(DIAG))
import numpy as np
import torch
from scripts.brava_learning_audit import PREVIOUS, read, write, sha
from scripts.brava_representation_study import dataset, batch, seed, SEED
from scripts.brava_learning_diagnosis import bn_state, restore_bn, weight_hash
from scripts.multiscale_pretrained import graph_cache

OUT = REPO / 'experiments/brava_augmentation_20261026'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    old = read(PREVIOUS / 'plan.json')
    diagnostic = read(DIAG / 'experiments/brava_learning_diagnosis_20261026/plan.json')
    assert len(old['ids']['train']) == 1248 and len(old['ids']['val']) == 256
    assert diagnostic['populations']['validation'] == old['ids']['val']
    assert sha(old['source']) == old['source_sha256']
    assert sha(Path(old['dataset']) / 'manifest.json') == old['manifest_sha256']
    assert old['seed'] == SEED == 261039
    assert old['config']['loss']['node']['classification']['focal_gamma'] == 2
    assert old['config']['loss']['edge']['classification']['name'] == 'cross_entropy'
    plan = dict(old, populations={k: diagnostic['populations'][k] for k in
                ['representative_train', 'validation', 'tiny_reference']},
                arms=['on', 'off'], reused_control=False,
                control_decision='Fresh K8-only pair: historical control had auxiliary K24/C1 heads; exact whole-run parity not established.',
                augmentation_off_rng='Draw and discard the existing augmentation, then return the clean item. This preserves RNG state for stochastic loss sampling.',
                evaluation='Saved BN; clean inputs; batch eight; unchanged physical metrics and 0.5 cutoffs.',
                scientific_scope='One seed, fixed 1k/2k/4k checkpoints; no test access; annotation-relative errors; brain masks unused as vessel labels.')
    write(OUT / 'plan.json', plan)
    sources = {str(f.relative_to(DIAG)): sha(f) for folder in ['scripts','data','models','training','configs','metrics']
               for f in (DIAG / folder).rglob('*') if f.is_file() and f.suffix in ['.py','.yaml']}
    write(OUT / 'frozen-source-sha256.json', sources)
    with zipfile.ZipFile(OUT / 'frozen-source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sources: z.write(DIAG / name, name)
    order_hash = hashlib.sha256()
    for step in range(1,4001):
        epoch, offset = divmod(step-1,156)
        ix = np.random.default_rng(SEED+epoch).permutation(1248)[offset*8:offset*8+8]
        order_hash.update(ix.astype('<i8').tobytes())
    write(OUT / 'preflight.json', dict(passed=True,source_sha256=old['source_sha256'],
          targets_sha256=sha(PREVIOUS/'targets.pkl'), source_plan_sha256=sha(PREVIOUS/'plan.json'),
          diagnostic_plan_sha256=sha(DIAG/'experiments/brava_learning_diagnosis_20261026/plan.json'),
          patch_order_sha256=order_hash.hexdigest(), training_patches=1248,validation_patches=256,
          train_subjects=39,validation_subjects=8, no_auxiliary_heads=True))


class CleanWithMatchedRNG:
    """Keep loss sampling identical while removing both actual augmentations."""
    def __init__(self, augmented): self.ds = augmented
    def __len__(self): return len(self.ds)
    def __getattr__(self, name): return getattr(self.ds,name)
    def __getitem__(self,index):
        self.ds[index]
        self.ds.augment = False
        try: return self.ds[index]
        finally: self.ds.augment = True


def setup(arm):
    from models import build_model
    from models.matcher import build_matcher
    from scripts.train_ixi_transfer import TransferCriterion
    p=read(OUT/'plan.json'); cfg=copy.deepcopy(p['config']);cfg['data']['train_augmentation']=arm=='on'
    seed(); cp=torch.load(p['source'],map_location='cpu',mmap=True)
    assert cp['config']['loss']==cfg['loss'] and cp['config']['model']['matcher']==cfg['model']['matcher']
    model=build_model(cfg);model.load_state_dict(cp['net'],strict=True);del cp
    model.cuda().requires_grad_(True)
    # These unused relation cross-attention parameters are disabled in the established recipe.
    for layer in model.decoder.decoder.layers:
        if layer.rln_attn: layer.cross_attn2.requires_grad_(False);layer.norm12.requires_grad_(False)
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,
                                spline_head=model.spline_head,route='graph_only').cuda()
    criterion.set_training_progress(1,0.)
    oc=cfg['training']['optimizer'];groups=[]
    for encoder,lr in [(True,p['backbone_lr']),(False,p['peak_lr'])]:
        groups.append(dict(params=[v for n,v in model.named_parameters() if v.requires_grad and n.startswith('encoder.')==encoder],lr=lr,base_lr=lr))
    opt=torch.optim.AdamW(groups,betas=tuple(oc['betas']),eps=oc['epsilon'],weight_decay=oc['weight_decay'])
    with (PREVIOUS/'targets.pkl').open('rb') as f: cache=pickle.load(f)
    cache={s:{'K8':v['K8']} for s,v in cache.items()}
    ds=dataset(cfg,cache,p['ids']['train'],'K8','train',True)
    if arm=='off':ds=CleanWithMatchedRNG(ds)
    return p,cfg,model,criterion,opt,cache,ds


def indices(step):
    epoch,offset=divmod(step-1,156)
    return np.random.default_rng(SEED+epoch).permutation(1248)[offset*8:offset*8+8].tolist()


def gradient_gate(model,loss):
    shared=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('spline_head.')]
    grads=torch.autograd.grad(loss['spline'],shared,retain_graph=True,allow_unused=True)
    assert all(g is None or not bool(g.abs().sum()) for g in grads)
    return dict(geometry_to_detector_zero=True)


@torch.no_grad()
def evaluate(model,criterion,cfg,cache,p,arm,step):
    model.eval();folder=OUT/'predictions'/arm/str(step);folder.mkdir(parents=True,exist_ok=True)
    for name,ids in p['populations'].items():
        ds=dataset(cfg,cache,ids,'K8','val' if name=='validation' else 'train',False);records=[]
        with zipfile.ZipFile(folder/(name+'.zip'),'w',zipfile.ZIP_STORED) as z:
            for start in range(0,len(ds),8):
                x,y=batch(ds,list(range(start,min(start+8,len(ds)))));seed(100000+start)
                tokens,pred,_=model(x);assignment=criterion.matcher(pred,y)
                loss=criterion(tokens,pred,y)
                scalar={k:float(v) for k,v in loss.items() if torch.is_tensor(v) and v.numel()==1}
                for j,s in enumerate(ids[start:start+8]):
                    c=graph_cache(model,criterion,tokens,pred,y,j);a,b=assignment[j]
                    c.update(hungarian_source=a.numpy(),hungarian_target=b.numpy())
                    buf=io.BytesIO();np.savez_compressed(buf,**c);z.writestr(s+'.npz',buf.getvalue())
                records.append(dict(batch_start=start,batch_size=len(x),losses=scalar,
                                    image_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()))
        write(folder/(name+'-losses.json'),records)
        if step==4000:
            # Saved BN must retain its predictions when changing inference batching.
            bx,by=batch(ds,list(range(8)));bt,bpred,_=model(bx)
            errors=[]
            for i in range(8):
                sx,sy=batch(ds,[i]);st,spred,_=model(sx)
                a=graph_cache(model,criterion,bt,bpred,by,i);b=graph_cache(model,criterion,st,spred,sy)
                error={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['physical','scores','relations']}
                assert all(np.isfinite(v) for v in error.values()),error
                np.testing.assert_array_equal(a['pairs'],b['pairs'])
                errors.append(dict(sample=ids[i],max_errors=error,
                  node_threshold_crossings=int(((a['scores']>.5)!=(b['scores']>.5)).sum()),
                  relation_threshold_crossings=int(((a['relations']>.5)!=(b['relations']>.5)).sum()),
                  within_original_strict_tolerance=error['physical']<1e-3 and error['scores']<1e-4 and error['relations']<1e-4))
            write(folder/(name+'-batch-parity.json'),dict(completed=True,finite=True,batch_sizes=[1,8],
                 scope='First eight fixed patches per population; measured drift, no change to primary batch-eight evaluation.',rows=errors))
    print('evaluated',arm,step,flush=True)


def smoke():
    p,cfg,model,criterion,opt,cache,ds=setup('on');initial=weight_hash(model);ix=indices(1)
    seed(1);x,y=batch(ds,ix);rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
    clean=CleanWithMatchedRNG(ds);seed(1);cx,cy=batch(clean,ix)
    assert rng[0]==random.getstate() and np.array_equal(rng[1][1],np.random.get_state()[1])
    assert torch.equal(rng[2],torch.get_rng_state()) and torch.equal(rng[3],torch.cuda.get_rng_state())
    plain=dataset(cfg,cache,p['ids']['train'],'K8','train',False);px,py=batch(plain,ix)
    torch.testing.assert_close(cx,px,rtol=0,atol=0)
    for a,b in zip(cy['nodes'],py['nodes']):torch.testing.assert_close(a,b,rtol=0,atol=0)
    initial_state={n:v.detach().clone() for n,v in model.state_dict().items()};checks={}
    for arm in ['on','off']:
        model.load_state_dict(initial_state);d=ds if arm=='on' else clean
        for step in [1,2]:
            seed(step);bx,by=batch(d,indices(step));model.train();opt.zero_grad(set_to_none=True)
            t,pred,_=model(bx);loss=criterion(t,pred,by);assert torch.isfinite(loss['total'])
            if step==1:checks[arm]=gradient_gate(model,loss)
            loss['total'].backward();assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            opt.step()
        evaluate(model,criterion,cfg,cache,dict(p,populations={k:v[:8] for k,v in p['populations'].items()}),'smoke_'+arm,2)
    write(OUT/'smoke.json',dict(passed=True,gradient_gates=checks,clean_input_exact=True,
          post_loader_rng_exact=True,initial_parameter_hash=initial,updates_per_arm=2,no_auxiliary_heads=True))


def train(arm):
    assert read(OUT/'smoke.json')['passed']
    for name,digest in read(OUT/'frozen-source-sha256.json').items():assert sha(DIAG/name)==digest,name
    p,cfg,model,criterion,opt,cache,ds=setup(arm);folder=OUT/'runs'/arm;folder.mkdir(parents=True,exist_ok=False)
    write(folder/'config.json',cfg);history=[];ih=hashlib.sha256();oh=hashlib.sha256();start=time.time()
    write(folder/'initial.json',dict(parameter_hash=weight_hash(model),source_sha256=p['source_sha256']))
    for step in range(1,4001):
        seed(step);ix=indices(step);oh.update(np.asarray(ix,dtype='<i8').tobytes());model.train();x,y=batch(ds,ix)
        ih.update(x.cpu().numpy().tobytes());opt.zero_grad(set_to_none=True)
        scale=min(1.,step/p['warmup'])*(p['lr_floor_ratio']+(1-p['lr_floor_ratio'])*.5*(1+np.cos(np.pi*max(0,step-p['warmup'])/(4000-p['warmup']))))
        for g in opt.param_groups:g['lr']=g['base_lr']*scale
        t,pred,_=model(x);loss=criterion(t,pred,y);assert torch.isfinite(loss['total'])
        if step==1:write(folder/'gradient-check.json',gradient_gate(model,loss))
        loss['total'].backward()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        if step==1 or step%10==0:
            history.append(dict(step=step,lr=[g['lr'] for g in opt.param_groups],
              losses={k:float(v.detach()) for k,v in loss.items() if torch.is_tensor(v) and v.numel()==1}))
        opt.step()
        if step%100==0:
            write(folder/'training.json',history);print('progress',arm,step,'seconds',time.time()-start,flush=True)
        if step in [1000,2000,4000]:
            path=folder/f'model-{step:06d}.pt'
            torch.save(dict(net=model.state_dict(),optimizer=opt.state_dict(),config=cfg,step=step,
                           source=p['source'],arm=arm),path)
            evaluate(model,criterion,cfg,cache,p,arm,step)
            write(folder/f'checkpoint-{step}.json',dict(path=str(path),sha256=sha(path),step=step))
    assert oh.hexdigest()==read(OUT/'preflight.json')['patch_order_sha256']
    write(folder/'complete.json',dict(completed=True,steps=4000,seconds=time.time()-start,
          image_stream_sha256=ih.hexdigest(),patch_order_sha256=oh.hexdigest(),job_id=os.getenv('SLURM_JOB_ID'),
          peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))


def recover_evaluation():
    """Resume only final inference after a checker failure; never update weights."""
    from scripts.brava_learning_diagnosis import build
    p=read(OUT/'plan.json')
    with (PREVIOUS/'targets.pkl').open('rb') as f:cache=pickle.load(f)
    cache={s:{'K8':v['K8']} for s,v in cache.items()}
    for arm in p['arms']:
        folder=OUT/'runs'/arm;path=folder/'model-004000.pt'
        cp=torch.load(path,map_location='cpu',mmap=True);assert cp['step']==4000 and cp['arm']==arm;del cp
        history=read(folder/'training.json');assert history[-1]['step']==4000
        model,cfg,criterion=build(path);before=weight_hash(model)
        evaluate(model,criterion,cfg,cache,p,arm,4000);assert weight_hash(model)==before
        write(folder/'checkpoint-4000.json',dict(path=str(path),sha256=sha(path),step=4000))
        prior=read(folder/'complete.json') if (folder/'complete.json').exists() else {}
        write(folder/'complete.json',dict(prior,completed=True,steps=4000,
          patch_order_sha256=read(OUT/'preflight.json')['patch_order_sha256'],
          patch_order_audit_source='Deterministically reconstructed from frozen runner and plan; terminal evaluator failure preempted the original completion write.',
          job_id='1612527_'+str(p['arms'].index(arm)),evaluation_recovered=True,
          recovery_job_id=os.getenv('SLURM_JOB_ID'),recovery_optimizer_updates=0,
          training_completion_evidence='Saved step-4000 net/optimizer plus training.json step 4000; final evaluator checker failed after checkpoint save.',
          image_stream_audit='Complete input hash unavailable: original process stopped at final evaluator before writing it.' if not prior else prior.get('image_stream_sha256')))
        del model,criterion;torch.cuda.empty_cache()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','smoke','train','recover-evaluation']);ap.add_argument('--arm',choices=['on','off'])
    a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='smoke':smoke()
    elif a.mode=='recover-evaluation':recover_evaluation()
    else:train(a.arm or ['on','off'][int(os.environ['SLURM_ARRAY_TASK_ID'])])
