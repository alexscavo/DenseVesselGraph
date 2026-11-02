"""Bounded R07 frozen-token relation-only continuation; no detector updates."""
import os, sys, io, json, time, copy, argparse, zipfile, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/r07_matched_supervision_20261027'
# This is the exact source/runtime which passed N06's batch-one replay.
sys.path[:0] = [str(OUT/'frozen-source'), str(OUT/'runtime'), str(ROOT)]
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# Pin the model package before legacy audit helpers prepend ROOT to sys.path.
# Otherwise GraphCriterion's matcher import can cache ROOT's older five-output
# RelationFormer, ignoring this checkpoint's compact coordinate-only head.
import models as frozen_models
assert Path(frozen_models.__file__).resolve().is_relative_to((OUT/'frozen-source').resolve()), 'R07 must load its frozen model package'
from scripts.analyze_graph_errors import read, write, sha, geometry
from scripts.r07_sampling import population, redistribution_indices
from training.losses.criterion import GraphCriterion, _ratio_upsample

ORIGINAL = ROOT.parent/'Graph-Native-Betti-Matching-Representation'
SEEDS = [364505, 92741]


def verify_code():
    identity=read(OUT/'code-identities.json')
    for name,digest in identity['own_and_reused_code'].items():assert sha(ROOT/name)==digest,('Changed experiment code',name)
    for name,digest in identity['frozen_source'].items():assert sha(OUT/'frozen-source'/name)==digest,('Changed model source',name)


def prepare():
    assert not (OUT/'plan.json').exists(), 'Preserve existing protocol'
    identities = read(ROOT.parent/'gnbm-graph-errors-20261026/experiments/oracle_representatives_20261026/input-identities.json')
    plan = dict(id='R07', datasets=['SyntheticMRI'], seeds=SEEDS, steps=1000,
                checkpoints=[0,100,300,1000], batch_size=8, lr=7e-6, warmup=20,
                weight_decay=1e-4, edge_weight=6., primary_step=1000,
                train_patches=128, val_patches=480, train_subjects=32,val_subjects=13,
                inference_batch=1, node_threshold=.5, edge_threshold=.5,
                reliable_distance_mm=2., nearest_margin_mm=.5, nearby_mm=5.,
                max_replaced_negative_fraction=.25, max_hard_multiplicity_ratio=4,
                bootstrap_seed=271026, graph_bootstrap=1000, ap_bootstrap=500,
                no_unmatched_mining=True, no_unfrozen_confirmation=True,
                population_note='Same clean, frozen train-monitor/validation populations as N06/N08; bounded pilot, not full source training exposure.',
                inputs=identities, records={}, configs={})
    source_plan=read(ORIGINAL/'experiments/local_refinement_long/plan.json')
    root=Path(source_plan['dataset']['root'])
    for seed in SEEDS:
        folder=ORIGINAL/f'experiments/local_refinement_long/{seed}/runs/A-xyz'
        cfg=read(folder/'config.json'); spec=cfg['loss']['edge']
        assert spec['classification']['name']=='cross_entropy' and spec['classification']['class_weights']==[1.,1.]
        assert spec['balancing']==dict(mode='ratio_upsample',positive_to_negative_ratio=.15,tolerance=.02)
        assert spec['candidates']['max_per_graph']==9999 and spec['candidates']['positive_cap'] is None
        assert not spec['candidates']['include_unmatched'] and not cfg['loss']['degree']['enabled']
        records=read(folder/'step-12000/records.json')
        records=[dict(r,gt=str(root/r['split']/'spline'/(r['sample']+'.npz'))) for r in records]
        assert len(records)==608 and sum(r['split']=='train' for r in records)==128
        assert not {r['subject'] for r in records if r['split']=='train'} & {r['subject'] for r in records if r['split']=='val'}
        plan['records'][str(seed)]=records;plan['configs'][str(seed)]=cfg
    write(OUT/'plan.json',plan)


def score(head,tokens,pairs):
    relation=tokens[120:122].reshape(1,-1).expand(len(pairs),-1)
    forward=torch.cat([tokens[pairs[:,0]],tokens[pairs[:,1]],relation],-1)
    reverse=torch.cat([tokens[pairs[:,1]],tokens[pairs[:,0]],relation],-1)
    return (.5*(head(forward)+head(reverse))).softmax(-1)[:,1]


def extract(seed, smoke=False):
    import nibabel as nib
    from models import build_model
    from models.matcher import build_matcher
    from models.spline import canonical_reverse
    p=read(OUT/'plan.json');cfg=copy.deepcopy(p['configs'][str(seed)])
    folder=ORIGINAL/f'experiments/local_refinement_long/{seed}/runs/A-xyz'
    identity=next(v for v in p['inputs'] if v['seed']==seed)
    assert sha(folder/'model-12000.pt')==identity['checkpoint_identity']['sha256']
    assert sha(folder/'step-12000/caches.zip')==identity['prediction_archive_sha256']
    cp=torch.load(folder/'model-12000.pt',map_location='cpu',mmap=True)
    model=build_model(cfg).eval().requires_grad_(False);model.load_state_dict(cp['net'],strict=True);model.cuda();del cp
    matcher=build_matcher(cfg);audit=[]
    records=p['records'][str(seed)]
    if smoke:
        audit_rows=read(OUT/'supervision-audit.json')['per_patch']
        difficult={r['sample'] for r in audit_rows if r['seed']==seed and r['split']=='train' and r['reliable_near_negatives']>0 and r['negatives']>=4}
        train_records=[r for r in records if r['split']=='train']
        chosen=[r for r in train_records if r['sample'] in difficult][:2]
        chosen += [r for r in train_records if r['sample'] not in {v['sample'] for v in chosen}][:4-len(chosen)]
        records=chosen+[r for r in records if r['split']=='val'][:4]
    dest=OUT/'large'/str(seed);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'initial-relation.pt').exists():
        initial=torch.load(dest/'initial-relation.pt',map_location='cpu')
        for k,v in model.relation_embed.state_dict().items():assert torch.equal(v.cpu(),initial[k])
    else:torch.save(model.relation_embed.state_dict(),dest/'initial-relation.pt')
    archive=dest/('smoke-features.zip' if smoke else 'features.zip')
    assert not archive.exists(), 'Preserve feature caches'
    reuse=zipfile.ZipFile(dest/'smoke-features.zip') if not smoke and (dest/'smoke-features.zip').exists() else None
    reusable=set(reuse.namelist()) if reuse is not None else set()
    if reusable:assert read(dest/'smoke-feature-audit.json')['checkpoint_sha256']==identity['checkpoint_identity']['sha256']
    with zipfile.ZipFile(folder/'step-12000/caches.zip') as old, zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as cache:
        for i,r in enumerate(records):
            if r['sample']+'.npz' in reusable:
                cache.writestr(r['sample']+'.npz',reuse.read(r['sample']+'.npz'))
                audit.append(dict(sample=r['sample'],reused_gpu_smoke_features=True));continue
            with np.load(io.BytesIO(old.read(r['sample']+'.npz'))) as f:c={k:f[k] for k in f.files}
            gt=geometry(r['gt']);raw=Path(r['gt']).parent.parent/'raw'/(r['sample']+'_data.nii.gz')
            np.testing.assert_allclose(gt['truth'],c['truth'],atol=2e-5,rtol=0)
            np.testing.assert_array_equal(gt['edges'],c['gt_edges'])
            image=np.asarray(nib.load(raw).dataobj,dtype=np.float32)-cfg['data']['datasets']['synthetic_mri']['foreground_mean']
            with torch.inference_mode():
                tokens,pred,_=model(torch.from_numpy(image.copy())[None,None].cuda())
                pairs=torch.as_tensor(c['pairs'],device='cuda')
                probabilities=score(model.relation_embed,tokens[0],pairs).cpu().numpy()
                err=dict(coordinate_mm=float(np.abs(pred['pred_nodes'][0,:,:3].cpu().numpy()@gt['metric'].T-c['physical']).max()),
                         confidence=float(np.abs(pred['pred_logits'][0].softmax(-1)[:,1].cpu().numpy()-c['scores']).max()),
                         relations=float(np.abs(probabilities-c['relations']).max()))
                assert err['coordinate_mm']<.001 and err['confidence']<1e-5 and err['relations']<1e-5,err
                assignment=matcher({k:pred[k] for k in ['pred_nodes','pred_logits']},
                                   {'nodes':[torch.as_tensor(gt['nodes'],device='cuda')]})[0]
                hs,ht=[v.cpu().numpy() for v in assignment]
                np.testing.assert_allclose(np.linalg.norm(c['physical'][hs]-c['truth'][ht],axis=-1),c['hungarian_distance'],atol=2e-4,rtol=0)
                retained=(c['scores'][c['pairs']]>.5).all(1)
                curve_pairs=pairs[torch.as_tensor(retained,device='cuda')].clone()
                endpoints=pred['pred_nodes'][0,curve_pairs,:3]
                metric=torch.as_tensor(gt['metric'],device='cuda',dtype=tokens.dtype)
                reverse=canonical_reverse(endpoints@metric.T)
                curve_pairs[reverse]=curve_pairs[reverse][:,[1,0]]
                features=torch.cat([tokens[0,curve_pairs[:,0]],tokens[0,curve_pairs[:,1]],tokens[0,120:122].reshape(1,-1).expand(len(curve_pairs),-1)],-1)
                if len(curve_pairs):
                    controls,_=model.spline_head(features,pred['pred_nodes'][0,curve_pairs,:3],metric,
                        local_features=model._spline_local_features[0],input_shape=(64,64,64))
                    controls[reverse]=controls[reverse].flip(1)
                else:controls=tokens.new_empty((0,8,3))
                np.testing.assert_allclose(controls[:,[0,-1]].cpu().numpy()@gt['metric'].T,
                    c['physical'][c['pairs'][retained]],atol=.001,rtol=0)
                extra=dict(tokens=tokens[0].cpu().numpy(),boxes=pred['pred_nodes'][0].cpu().numpy(),
                           logits=pred['pred_logits'][0].cpu().numpy(),normalized_truth=gt['nodes'],metric=gt['metric'],
                           hungarian_source=hs,hungarian_target=ht,
                           curve_pair_indices=np.flatnonzero(retained),curve_controls=controls.cpu().numpy())
            buf=io.BytesIO();np.savez_compressed(buf,**c,**extra);cache.writestr(r['sample']+'.npz',buf.getvalue())
            audit.append(dict(sample=r['sample'],**err));model._spline_local_features=None
            if i%64==0:print('extracted',seed,i,len(records),flush=True)
    write(dest/('smoke-feature-audit.json' if smoke else 'feature-audit.json'),dict(completed=True,patches=len(audit),records=audit,
          checkpoint_sha256=identity['checkpoint_identity']['sha256'],inference_batch=1,features_sha256=sha(archive)))
    if reuse is not None:reuse.close()
    del model;torch.cuda.empty_cache()


def load_features(seed,smoke=False):
    result=[];p=read(OUT/'plan.json');records={r['sample']:r for r in p['records'][str(seed)]}
    with zipfile.ZipFile(OUT/'large'/str(seed)/('smoke-features.zip' if smoke else 'features.zip')) as z:
        for name in z.namelist():
            with np.load(io.BytesIO(z.read(name))) as f:c={k:f[k] for k in f.files}
            c['record']=records[name.removesuffix('.npz')]
            gt=geometry(c['record']['gt'])
            c['population']=population(c,gt,c['hungarian_source'],c['hungarian_target'])
            result.append(c)
    return result


def balanced_features(criterion, items, step_seed):
    """Call the original sampler, retaining exact orientation and balancing."""
    torch.manual_seed(step_seed)
    device=next(criterion.relation_embed.parameters()).device
    tokens=torch.stack([torch.as_tensor(c['tokens'],device=device) for c in items])
    assignments=[(torch.as_tensor(c['hungarian_source'],device=device),torch.as_tensor(c['hungarian_target'],device=device)) for c in items]
    edges=[torch.as_tensor(c['gt_edges'],device=device) for c in items]
    contexts=[];features,labels,_=criterion._matched_edge_pool(tokens,edges,assignments,contexts)
    feature=torch.cat(features);label=torch.cat(labels)
    hard=[];graphs=[]
    for b,pairs in contexts:
        pop=items[b]['population'];hard_pairs={tuple(sorted(e)) for e in pop['queries'][pop['hard']].tolist()}
        hard.extend(tuple(sorted(e)) in hard_pairs for e in pairs.tolist());graphs.extend([b]*len(pairs))
    ids,_label=_ratio_upsample(torch.arange(len(label),device=device)[:,None],label,
                             criterion.balancing['positive_to_negative_ratio'],criterion.balancing['tolerance'])
    indices=ids.flatten().long();idx=indices.cpu().numpy()
    return feature[indices],_label,np.asarray(hard,bool)[idx],np.asarray(graphs,int)[idx]


def build_head(seed,device):
    from models.relationformer import MLP
    head=MLP(2208,552,2,3).to(device)
    head.load_state_dict(torch.load(OUT/'large'/str(seed)/'initial-relation.pt',map_location=device))
    return head


def evaluate_head(head,data,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(),zipfile.ZipFile(dest,'w',zipfile.ZIP_STORED) as z:
        for c in data:
            tokens=torch.as_tensor(c['tokens'],device='cuda');pairs=torch.as_tensor(c['pairs'],device='cuda')
            scores=score(head,tokens,pairs).cpu().numpy()
            buf=io.BytesIO();np.savez_compressed(buf,relations=scores)
            z.writestr(c['record']['sample']+'.npz',buf.getvalue())


def train(seed,smoke=False):
    from models.matcher import build_matcher
    p=read(OUT/'plan.json');cfg=p['configs'][str(seed)]
    data=load_features(seed,smoke);training=[c for c in data if c['record']['split']=='train']
    steps=3 if smoke else p['steps'];batch=2 if smoke else p['batch_size']
    order=np.random.default_rng(seed).integers(len(training),size=(steps,batch))
    if smoke:order[0]=[0,1]
    base=OUT/'large'/str(seed)/('smoke' if smoke else 'runs')
    for arm in ['S0','S1']:
        head=build_head(seed,'cuda').train();criterion=GraphCriterion(cfg,build_matcher(cfg),head).cuda()
        optimizer=torch.optim.AdamW(head.parameters(),lr=p['lr'],weight_decay=p['weight_decay'],betas=(.9,.999),eps=1e-8)
        folder=base/arm;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'complete.json').exists()
        initial=copy.deepcopy(head.state_dict());history=[];start=time.monotonic();budget_digest=hashlib.sha256()
        exposure=dict(pairs=0,positive_slots=0,negative_slots=0,reliable_hard_before=0,reliable_hard_after=0,replaced_slots=0)
        if not smoke:evaluate_head(head,data,folder/'scores-0000.zip')
        for step,ix in enumerate(order,1):
            selected=[training[i] for i in ix]
            feature,label,hard,graphs=balanced_features(criterion,selected,seed+step)
            index=np.arange(len(label)) if arm=='S0' else redistribution_indices(label.cpu().numpy(),hard,graphs,np.random.default_rng(seed+step))
            positives=int((label==1).sum());replaced=int((index!=np.arange(len(index))).sum())
            budget_digest.update(np.asarray([len(label),positives],dtype='<i8').tobytes())
            exposure['pairs']+=len(label);exposure['positive_slots']+=positives;exposure['negative_slots']+=len(label)-positives
            exposure['reliable_hard_before']+=int(hard.sum());exposure['reliable_hard_after']+=int(hard[index].sum());exposure['replaced_slots']+=replaced
            tensor_index=torch.as_tensor(index,device='cuda');loss=F.cross_entropy(head(feature[tensor_index]),label)*p['edge_weight']
            optimizer.zero_grad(set_to_none=True);assert torch.isfinite(loss)
            loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())
            for group in optimizer.param_groups:group['lr']=p['lr']*min(1.,step/p['warmup'])
            optimizer.step()
            if step==1 or step%10==0:
                history.append(dict(step=step,loss=float(loss.detach()),pairs=len(label),positives=int((label==1).sum()),
                    hard_before=int(hard.sum()),hard_after=int(hard[index].sum()),replaced=int((index!=np.arange(len(index))).sum()),
                    seconds=time.monotonic()-start))
            if step in p['checkpoints'] or step==steps:
                torch.save(dict(state=head.state_dict(),seed=seed,arm=arm,step=step),folder/f'relation-{step:04d}.pt')
                evaluate_head(head,data,folder/f'scores-{step:04d}.zip')
                write(folder/'history.json',history);print('trained',seed,arm,step,time.monotonic()-start,flush=True)
        assert any(not torch.equal(initial[k],v) for k,v in head.state_dict().items())
        write(folder/'complete.json',dict(completed=True,seed=seed,arm=arm,updates=steps,
              order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),trainable='original relation_embed only',
              effective_budget_sha256=budget_digest.hexdigest(),exposure=exposure,
              seconds=time.monotonic()-start,parameters=sum(v.numel() for v in head.parameters())))
        del head,criterion,optimizer;torch.cuda.empty_cache()


def smoke():
    seed=SEEDS[0];extract(seed,True)
    from models.matcher import build_matcher
    p=read(OUT/'plan.json');data=load_features(seed,True);head=build_head(seed,'cuda')
    criterion=GraphCriterion(p['configs'][str(seed)],build_matcher(p['configs'][str(seed)]),head).cuda()
    items=[c for c in data if c['record']['split']=='train'][:2]
    features,labels,hard,graphs=balanced_features(criterion,items,seed)
    manual=F.cross_entropy(head(features),labels)
    torch.manual_seed(seed)
    tokens=torch.as_tensor(np.stack([c['tokens'] for c in items]),device='cuda')
    assignments=[(torch.as_tensor(c['hungarian_source'],device='cuda'),torch.as_tensor(c['hungarian_target'],device='cuda')) for c in items]
    edges=[torch.as_tensor(c['gt_edges'],device='cuda') for c in items]
    original,_=criterion.loss_edges(tokens,None,edges,assignments)
    loss_error=float((manual-original).abs().detach())
    torch.testing.assert_close(manual,original,rtol=1e-6,atol=1e-7)
    g1=torch.autograd.grad(manual,tuple(head.parameters()),retain_graph=True)
    g2=torch.autograd.grad(original,tuple(head.parameters()))
    gradient_error=max(float((a-b).abs().max()) for a,b in zip(g1,g2))
    for a,b in zip(g1,g2):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    del criterion,head;train(seed,True)
    from scripts.collect_r07 import smoke as collector_smoke
    collector_smoke()
    assert read(OUT/'large'/str(seed)/'smoke/S1/complete.json')['exposure']['replaced_slots']>0
    write(OUT/'gpu-smoke.json',dict(passed=True,baseline_loss_and_gradient_parity=True,
          loss_absolute_error=loss_error,gradient_max_absolute_error=gradient_error,
          note='Pair membership/labels/balance are exact; duplicating frozen features before the MLP can cause float-roundoff versus duplicating logits afterward.',
          parity_patches=8,finite_gradients=True,detector_frozen_by_cached_features=True,
          no_new_detector_losses=True,training_steps=3))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','smoke','run']);parser.add_argument('--seed',type=int,default=SEEDS[0]);a=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.stage=='prepare':prepare()
    elif a.stage=='smoke':verify_code();smoke()
    else:
        verify_code();assert read(OUT/'gpu-smoke.json')['passed'];extract(a.seed);train(a.seed)
