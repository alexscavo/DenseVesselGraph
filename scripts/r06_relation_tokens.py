"""R06: frozen modern A-xyz, original relation CE, independent token/CNN residuals."""
import argparse,copy,hashlib,io,json,os,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from models.relation_token_refinement import RelationTokenResidual,sample_local_cnn
from training.losses.criterion import GraphCriterion
from scripts.analyze_graph_errors import read,write,sha,geometry
ORIGINAL=ROOT.parent/'Graph-Native-Betti-Matching-Representation'
OUT=ROOT/'experiments/r06_relation_tokens_20261027'
SEEDS=[364505,92741]


def prepare():
    assert not (OUT/'plan.json').exists()
    identities=read(ROOT.parent/'gnbm-graph-errors-20261026/experiments/oracle_representatives_20261026/input-identities.json')
    source_plan=read(ORIGINAL/'experiments/local_refinement_long/plan.json');root=Path(source_plan['dataset']['root'])
    p=dict(id='R06',seeds=SEEDS,steps=2000,checkpoints=[0,100,300,1000,2000],primary_step=2000,
      train_patches=128,train_subjects=32,val_patches=480,val_subjects=13,batch_size=8,
      lr=7e-6,warmup=20,weight_decay=1e-4,edge_weight=6.,projection=64,
      inference_batch=1,node_threshold=.5,edge_threshold=.5,inputs=identities,configs={},records={},
      arms=['R0','R1','R2','R3_head'],r2_ablation='Per-patch fixed query permutation of local CNN vectors; coordinates/tokens unchanged.',
      population='Existing optimized-on N06/N08 128-patch monitor is the residual training set, not the 25000-patch detector population. Fixed 480-patch development validation, no test access.',
      schedule='Fixed 2000 updates; early checkpoints descriptive, no validation-selected winner or extension.',
      supervision='Original matched-pair CE, all negatives under cap9999, ratio upsampling0.15; identical orientation/balancing seeds. No R07 mining.',
      decision_gate='N08 completed negative selection result does not establish a token information deficit. Normal conditional AP is already ~0.978; residual FP and near-nonadjacent confusion leave a small testable target. Only a bounded frozen comparison is justified, not a large architecture campaign.',
      confirmation_gate='R2 must improve final conditional/all-retained AP and graph F1 against R0/R1 in both seeds, show loss of benefit with image shuffling, and avoid worsened cycles or early-only gains. Subject intervals must support the improvement; otherwise no unfrozen confirmation.',
      bootstrap_seed=271026,bootstrap_draws=1000,max_study_h100=1,max_global_h100=2)
    for seed in SEEDS:
        folder=ORIGINAL/f'experiments/local_refinement_long/{seed}/runs/A-xyz'
        cfg=read(folder/'config.json');e=cfg['loss']['edge']
        assert e['classification']['name']=='cross_entropy' and e['classification']['class_weights']==[1.,1.]
        assert e['balancing']==dict(mode='ratio_upsample',positive_to_negative_ratio=.15,tolerance=.02)
        assert not e['candidates']['include_unmatched'] and not cfg['loss']['degree']['enabled']
        assert cfg['model']['decoder']['hidden_dim']==552 and cfg['model']['decoder']['object_queries']==120
        identity=next(v for v in identities if v['seed']==seed)
        assert sha(folder/'model-12000.pt')==identity['checkpoint_identity']['sha256']
        records=[dict(r,gt=str(root/r['split']/'spline'/(r['sample']+'.npz'))) for r in read(folder/'step-12000/records.json')]
        assert len(records)==608 and sum(r['split']=='train' for r in records)==128
        assert len({r['subject'] for r in records if r['split']=='train'})==32
        assert not {r['subject'] for r in records if r['split']=='train'}&{r['subject'] for r in records if r['split']=='val'}
        p['records'][str(seed)]=records;p['configs'][str(seed)]=cfg
    assert [r['sample'] for r in p['records'][str(SEEDS[0])]]==[r['sample'] for r in p['records'][str(SEEDS[1])]]
    write(OUT/'plan.json',p)


def score(head,tokens,pairs):
    rel=tokens[120:122].reshape(1,-1).expand(len(pairs),-1)
    a=torch.cat([tokens[pairs[:,0]],tokens[pairs[:,1]],rel],-1)
    b=torch.cat([tokens[pairs[:,1]],tokens[pairs[:,0]],rel],-1)
    return (.5*(head(a)+head(b))).softmax(-1)[:,1]


def refine(module,tokens,local,shuffle=False):
    if module is None:return tokens
    if shuffle:
        permutation=torch.as_tensor(np.random.default_rng(271026).permutation(120),device=local.device)
        local=local[...,permutation,:]
    objects=module(tokens[...,:120,:],local)
    return torch.cat([objects,tokens[...,120:,:]],dim=-2)


def extract(seed,smoke=False):
    import nibabel as nib
    from models import build_model
    from models.matcher import build_matcher
    p=read(OUT/'plan.json');cfg=copy.deepcopy(p['configs'][str(seed)])
    folder=ORIGINAL/f'experiments/local_refinement_long/{seed}/runs/A-xyz'
    cp=torch.load(folder/'model-12000.pt',map_location='cpu',mmap=True)
    model=build_model(cfg).eval().requires_grad_(False);model.load_state_dict(cp['net'],strict=True);del cp;model.cuda()
    matcher=build_matcher(cfg);dest=OUT/'large'/str(seed);dest.mkdir(parents=True,exist_ok=True)
    archive=dest/('smoke-features.zip' if smoke else 'features.zip');assert not archive.exists()
    records=p['records'][str(seed)]
    if smoke:records=[r for split in ['train','val'] for r in [v for v in records if v['split']==split][:4]]
    state_digest=hashlib.sha256()
    for name,value in model.state_dict().items():state_digest.update(name.encode());state_digest.update(value.cpu().numpy().tobytes())
    before=state_digest.hexdigest();torch.save(model.relation_embed.state_dict(),dest/'initial-relation.pt');audit=[]
    with zipfile.ZipFile(folder/'step-12000/caches.zip') as old,zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
        with torch.inference_mode():
            for i,r in enumerate(records):
                with np.load(io.BytesIO(old.read(r['sample']+'.npz'))) as f:c={k:f[k] for k in f.files}
                gt=geometry(r['gt']);raw=Path(r['gt']).parent.parent/'raw'/(r['sample']+'_data.nii.gz')
                image=np.asarray(nib.load(raw).dataobj,dtype=np.float32)-cfg['data']['datasets']['synthetic_mri']['foreground_mean']
                tokens,pred,_=model(torch.from_numpy(image.copy())[None,None].cuda());pairs=torch.as_tensor(c['pairs'],device='cuda')
                prob=score(model.relation_embed,tokens[0],pairs).cpu().numpy()
                err=dict(coordinate_mm=float(np.abs(pred['pred_nodes'][0,:,:3].cpu().numpy()@gt['metric'].T-c['physical']).max()),
                         confidence=float(np.abs(pred['pred_logits'][0].softmax(-1)[:,1].cpu().numpy()-c['scores']).max()),
                         relations=float(np.abs(prob-c['relations']).max()))
                assert err['coordinate_mm']<.001 and err['confidence']<1e-5 and err['relations']<1e-5,err
                hs,ht=matcher({k:pred[k] for k in ['pred_nodes','pred_logits']},{'nodes':[torch.as_tensor(gt['nodes'],device='cuda')]})[0]
                hs,ht=hs.cpu().numpy(),ht.cpu().numpy()
                np.testing.assert_allclose(np.linalg.norm(c['physical'][hs]-c['truth'][ht],axis=-1),c['hungarian_distance'],atol=2e-4,rtol=0)
                local=sample_local_cnn(model._spline_local_features,pred['pred_nodes'],image.shape)
                extra=dict(tokens=tokens[0].cpu().numpy(),local=local[0].cpu().numpy(),
                           logits=pred['pred_logits'][0].cpu().numpy(),boxes=pred['pred_nodes'][0].cpu().numpy(),
                           hungarian_source=hs,hungarian_target=ht)
                buf=io.BytesIO();np.savez_compressed(buf,**c,**extra);z.writestr(r['sample']+'.npz',buf.getvalue())
                audit.append(dict(sample=r['sample'],**err));model._spline_local_features=None
                if i%64==0:print('extracted',seed,i,len(records),flush=True)
    state_digest=hashlib.sha256()
    for name,value in model.state_dict().items():state_digest.update(name.encode());state_digest.update(value.cpu().numpy().tobytes())
    assert state_digest.hexdigest()==before
    write(dest/('smoke-feature-audit.json' if smoke else 'feature-audit.json'),dict(completed=True,records=audit,patches=len(audit),
          frozen_state_sha256=before,frozen_parameters_and_bn_unchanged=True,inference_batch=1,features_sha256=sha(archive)))
    del model;torch.cuda.empty_cache()


def load_features(seed,smoke=False):
    data=[];p=read(OUT/'plan.json');records={r['sample']:r for r in p['records'][str(seed)]}
    with zipfile.ZipFile(OUT/'large'/str(seed)/('smoke-features.zip' if smoke else 'features.zip')) as z:
        for name in z.namelist():
            with np.load(io.BytesIO(z.read(name))) as f:c={k:f[k] for k in f.files}
            c['record']=records[Path(name).stem];data.append(c)
    return data


class AuditedCriterion(GraphCriterion):
    """Bookkeeping only: inherited sampler, balancing and CE remain unchanged."""
    def _matched_edge_pool(self,tokens,target_edges,assignments,pair_contexts=None):
        result=super()._matched_edge_pool(tokens,target_edges,assignments,pair_contexts)
        if pair_contexts is not None:
            for b,pairs in pair_contexts:
                self.digest.update(np.asarray([b],dtype='<i8').tobytes());self.digest.update(pairs.cpu().numpy().tobytes())
        return result
    def _classification_loss(self,logits,labels,config):
        self.digest.update(labels.cpu().numpy().tobytes());self.pair_slots+=len(labels);self.positive_slots+=int((labels==1).sum())
        return super()._classification_loss(logits,labels,config)


def build_head(seed,trainable=False):
    from models.relationformer import MLP
    head=MLP(2208,552,2,3).cuda()
    head.load_state_dict(torch.load(OUT/'large'/str(seed)/'initial-relation.pt',map_location='cuda'))
    return head.eval().requires_grad_(trainable)


def evaluate(head,module,data,path,shuffle=False):
    path.parent.mkdir(parents=True,exist_ok=True);deltas=[]
    with torch.inference_mode(),zipfile.ZipFile(path,'w',zipfile.ZIP_STORED) as z:
        for c in data:
            tokens=torch.as_tensor(c['tokens'],device='cuda');local=torch.as_tensor(c['local'],device='cuda')
            refined=refine(module,tokens,local,shuffle);pairs=torch.as_tensor(c['pairs'],device='cuda')
            assert torch.equal(refined[120:],tokens[120:])
            scores=score(head,refined,pairs).cpu().numpy()
            delta=refined[:120]-tokens[:120];deltas.append(float(delta.square().mean().sqrt()))
            buf=io.BytesIO();np.savez_compressed(buf,relations=scores);z.writestr(c['record']['sample']+'.npz',buf.getvalue())
    write(path.with_suffix('.json'),dict(mean_token_delta_rms=np.mean(deltas),node_coords_logits_and_spline_inputs_unchanged=True,shared_relation_tokens_unchanged=True,shuffled=shuffle))


def train(seed,smoke=False):
    from models.matcher import build_matcher
    p=read(OUT/'plan.json');cfg=p['configs'][str(seed)];data=load_features(seed,smoke)
    training=[c for c in data if c['record']['split']=='train'];steps=3 if smoke else p['steps'];batch=2 if smoke else p['batch_size']
    order=np.random.default_rng(271026).integers(len(training),size=(steps,batch));complete=[]
    base=OUT/'large'/str(seed)/('smoke' if smoke else 'runs')
    original=build_head(seed);evaluate(original,None,data,base/'R0/scores-0000.zip');del original
    for arm in ['R1','R2','R3_head']:
        torch.manual_seed(seed);head=build_head(seed,arm=='R3_head')
        torch.manual_seed(seed);module=RelationTokenResidual(arm=='R2',projection=p['projection']).cuda() if arm!='R3_head' else None
        parameters=list(module.parameters()) if module is not None else list(head.parameters())
        opt=torch.optim.AdamW(parameters,lr=p['lr'],weight_decay=p['weight_decay'],betas=(.9,.999),eps=1e-8)
        criterion=AuditedCriterion(cfg,build_matcher(cfg),head).cuda();criterion.digest=hashlib.sha256();criterion.pair_slots=0;criterion.positive_slots=0
        frozen_before={k:v.clone() for k,v in head.state_dict().items()} if module is not None else None
        folder=base/arm;folder.mkdir(parents=True,exist_ok=False);history=[];start=time.monotonic()
        evaluate(head,module,data,folder/'scores-0000.zip')
        for step,ix in enumerate(order,1):
            selected=[training[i] for i in ix];torch.manual_seed(seed+step)
            tokens=torch.as_tensor(np.stack([c['tokens'] for c in selected]),device='cuda')
            local=torch.as_tensor(np.stack([c['local'] for c in selected]),device='cuda')
            assignments=[(torch.as_tensor(c['hungarian_source'],device='cuda'),torch.as_tensor(c['hungarian_target'],device='cuda')) for c in selected]
            edges=[torch.as_tensor(c['gt_edges'],device='cuda') for c in selected]
            modified=refine(module,tokens,local);loss,_=criterion.loss_edges(modified,None,edges,assignments);loss=loss*p['edge_weight']
            opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward()
            assert any(v.grad is not None and bool(v.grad.abs().sum()) for v in parameters)
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in parameters)
            for g in opt.param_groups:g['lr']=p['lr']*min(1.,step/p['warmup'])
            opt.step()
            if step==1 or step%10==0:history.append(dict(step=step,ce=float(loss.detach())/p['edge_weight'],seconds=time.monotonic()-start))
            if step in p['checkpoints'] or step==steps:
                path=folder/f'module-{step:04d}.pt';torch.save(dict(state=module.state_dict() if module is not None else head.state_dict(),arm=arm,seed=seed,step=step),path)
                evaluate(head,module,data,folder/f'scores-{step:04d}.zip')
                if arm=='R2':evaluate(head,module,data,folder/f'scores-{step:04d}-shuffled.zip',True)
                write(folder/'history.json',history);print('trained',seed,arm,step,time.monotonic()-start,flush=True)
        if frozen_before is not None:assert all(torch.equal(v,head.state_dict()[k]) for k,v in frozen_before.items())
        record=dict(completed=True,arm=arm,seed=seed,steps=steps,parameters=sum(v.numel() for v in parameters),
                    pair_sampling_sha256=criterion.digest.hexdigest(),pair_slots=criterion.pair_slots,positive_slots=criterion.positive_slots,
                    patch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),seconds=time.monotonic()-start,
                    original_relation_head_frozen=module is not None,detector_frozen_by_immutable_features=True)
        write(folder/'complete.json',record);complete.append(record)
        del head,module,opt,criterion;torch.cuda.empty_cache()
    assert len({r['pair_sampling_sha256'] for r in complete})==1
    assert complete[0]['parameters']==complete[1]['parameters']
    write(base/'complete.json',dict(completed=True,seed=seed,steps=steps,arms=complete))


def smoke():
    from models.matcher import build_matcher
    from tests.test_relation_token_refinement import test_identity_and_parameter_matched_active_control,test_local_sampler_uses_established_feature_centres
    test_identity_and_parameter_matched_active_control();test_local_sampler_uses_established_feature_centres()
    for seed in SEEDS:
        extract(seed,True);data=load_features(seed,True);head=build_head(seed);p=read(OUT/'plan.json')
        c=data[0];tokens=torch.as_tensor(c['tokens'],device='cuda')[None];local=torch.as_tensor(c['local'],device='cuda')[None]
        module=RelationTokenResidual(True).cuda();modified=refine(module,tokens,local)
        assert torch.equal(modified,tokens)
        criterion=GraphCriterion(p['configs'][str(seed)],build_matcher(p['configs'][str(seed)]),head).cuda()
        edges=[torch.as_tensor(c['gt_edges'],device='cuda')];assignment=[(torch.as_tensor(c['hungarian_source'],device='cuda'),torch.as_tensor(c['hungarian_target'],device='cuda'))]
        torch.manual_seed(seed);a,_=criterion.loss_edges(tokens,None,edges,assignment)
        torch.manual_seed(seed);b,_=criterion.loss_edges(modified,None,edges,assignment)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        del head,module,criterion;train(seed,True)
    write(OUT/'smoke.json',dict(passed=True,zero_initial_loss_parity=True,parameter_match=True,finite_backward=True,
      detector_logits_coords_and_bn_frozen=True,shared_relation_tokens_unchanged=True,original_sampling_digest_equal=True,
      local_sampler_alignment=True,token_only_image_invariance=True,both_seeds=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','smoke','run']);ap.add_argument('--seed',type=int,default=SEEDS[0]);a=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.mode=='prepare':prepare()
    elif a.mode=='smoke':smoke()
    else:
        assert read(OUT/'smoke.json')['passed'];extract(a.seed);train(a.seed)
