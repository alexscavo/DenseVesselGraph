"""Small frozen-candidate verifier study. No changes to the joint detector loss."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.node_verifier_metrics import association, supervision, select, measure, aggregate, discrimination

ROOT=Path('experiments/node_verifier_feasibility')
SOURCE=Path('experiments/multidataset_continuous/plan.json')
ARMS={'IXI':(3,'IXI_K8','020000'),'SyntheticMRI':(9,'SyntheticMRI_K8','020000'),'BRAVA':(0,'D1_IXI','030000')}
SEED=231023


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def records(dataset, small=True):
    plan=json.loads(SOURCE.read_text());_,arm,step=ARMS[dataset]
    rows=json.loads((Path(plan['output_root'])/'predictions'/arm/step/'records.json').read_text())
    if small:
        spec=plan['datasets'][dataset];ids=set(spec['monitor_train']+spec['monitor_val'])
        rows=[r for r in rows if r['sample_id'] in ids]
    tr={r['subject'] for r in rows if r['split']=='train'};va={r['subject'] for r in rows if r['split']=='val'}
    assert not tr&va
    return rows


def load_record(r, features=False, dataset=None):
    path=ROOT/'cache'/dataset/(r['sample_id']+'.npz') if features else Path(r['cache'])
    with np.load(path) as p, np.load(r['gt']) as g:
        c=dict(record=r,physical=p['raw_nodes']@g['metric'].T,truth=g['nodes']@g['metric'].T,gt_edges=g['edges'],scores=p['raw_scores'])
        if 'raw_pairs' in p:c.update(pairs=p['raw_pairs'],relations=p['raw_relation_scores'])
        if features:c.update(tokens=p['tokens'],logits=p['logits'],coordinates=p['raw_nodes'])
    return c


def baseline():
    result={}
    for dataset in ARMS:
        rows=records(dataset,False);out={};detail=[]
        for split in ['train','val']:
            cs=[load_record(r) for r in rows if r['split']==split]
            out[split]={'patches':len(cs),'subjects':len({c['record']['subject'] for c in cs}),'discrimination':discrimination(cs,[c['scores'] for c in cs])}
            for tol in [1.,2.]:
                raw=[measure(c,np.arange(len(c['scores'])),tol,False) for c in cs]
                filtered=[measure(c,np.flatnonzero(c['scores']>.5),tol,False) for c in cs]
                oracle=[measure(c,list(association(c['physical'],c['truth'],tol)),tol) for c in cs]
                out[split][str(tol)]=dict(raw=aggregate(raw),filtered=aggregate(filtered),oracle_selection=aggregate(oracle),valid_proposals_rejected=sum(len(set(association(c['physical'],c['truth'],tol))-set(np.flatnonzero(c['scores']>.5))) for c in cs))
                for c,a,b in zip(cs,raw,filtered):detail.append(dict(sample_id=c['record']['sample_id'],subject=c['record']['subject'],split=split,tolerance=tol,raw=a,filtered=b))
            print(dataset,split,out[split]['2.0'],flush=True)
        result[dataset]=out;write(ROOT/'baseline.json',result);write(ROOT/f'baseline-{dataset}-patches.json',detail)


def brava_policy():
    """Cache-only V1 diagnosis even when the verifier transfer gate is closed."""
    cs=[load_record(r) for r in records('BRAVA')]
    train=[c for c in cs if c['record']['split']=='train'];val=[c for c in cs if c['record']['split']=='val']
    policy=calibrate(train,[c['scores'] for c in train]);result=dict(policy=policy,validation_subjects=len({c['record']['subject'] for c in val}),results={})
    details={}
    for name,p in [('V0',dict(threshold=.5,radius=0)),('V1',policy)]:
        result['results'][name]={};details[name]={}
        for mode in ['calibrated','matched_count']:
            for tol in [1.,2.]:
                rr=[]
                for c in val:
                    ids=select(c,c['scores'],p['threshold'],p['radius'],int((c['scores']>.5).sum()) if mode=='matched_count' else None)
                    row=measure(c,ids,tol);row.update(subject=c['record']['subject'],sample=c['record']['sample_id']);rr.append(row)
                key=f'{mode}_{tol}';details[name][key]=rr;result['results'][name][key]=aggregate(rr)
    result['paired']={key:paired(details['V1'][key],details['V0'][key]) for key in details['V1']}
    write(ROOT/'results'/'BRAVA'/'baseline-policy.json',result);write(ROOT/'results'/'BRAVA'/'baseline-policy-patches.json',details)
    print('BRAVA policy',result,flush=True)


def extract(dataset):
    import torch
    from scripts.multidataset_graph_training import setup
    from scripts.multidataset_graph_evaluation import loaders
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    from models.checkpoint import load_checkpoint_file
    plan=json.loads(SOURCE.read_text());index,arm,step=ARMS[dataset]
    bundle=setup(Path(plan['output_root']),index);_,spec,model,cfg,criterion,*_=bundle
    if dataset=='BRAVA':
        checkpoint=Path(plan['output_root'])/'runs'/arm/f'model-{step}.pt'
        model.load_state_dict(load_checkpoint_file(checkpoint)['net'],strict=True)
    else:checkpoint=Path(plan['checkpoints'][spec['initializer']]['path'])
    model.eval().requires_grad_(False);out=ROOT/'cache'/dataset;out.mkdir(parents=True,exist_ok=True)
    lookup={r['sample_id']:r for r in records(dataset)};max_coord=0.;max_score=0.;max_rel=0.
    with torch.no_grad():
        for split,names,loader in loaders(cfg,plan,spec,True):
            for n,batch in enumerate(loader):
                sample=names[n];path=out/(sample+'.npz')
                if path.exists():continue
                volumes,targets=_move_graph_batch(batch,'cuda','image',True)
                tokens,pred,_=model(volumes)
                raw=infer_graphs(tokens,pred,model.relation_embed,object_queries=model.obj_token,relation_tokens=criterion.relation_tokens,node_threshold=-1.,edge_threshold=-1.)[0]
                nodes=pred['pred_nodes'][0,:,:3].cpu().numpy();scores=pred['pred_logits'][0].softmax(-1)[:,1].cpu().numpy()
                pairs=raw['query_ids'][raw['edges']].numpy();relations=raw['edge_scores'].numpy()
                with np.load(lookup[sample]['cache']) as old:
                    dc=float(np.max(abs(nodes-old['raw_nodes'])));ds=float(np.max(abs(scores-old['raw_scores'])))
                    max_coord=max(max_coord,dc);max_score=max(max_score,ds)
                    assert dc<2e-5 and ds<2e-4,(dataset,sample,dc,ds)
                    matrix=np.zeros((len(scores),len(scores)));matrix[pairs[:,0],pairs[:,1]]=relations;matrix+=matrix.T
                    qp=old['query_ids'][old['candidate_edges']]
                    dr=float(np.max(abs(matrix[qp[:,0],qp[:,1]]-old['candidate_scores']))) if len(qp) else 0.
                    max_rel=max(max_rel,dr);assert dr<2e-4,(sample,dr)
                np.savez_compressed(path,raw_nodes=nodes,raw_scores=scores,raw_pairs=pairs,raw_relation_scores=relations,tokens=tokens[0,:model.obj_token].cpu().numpy(),logits=pred['pred_logits'][0].cpu().numpy())
                if n%32==0:print('cache',dataset,split,n,len(names),flush=True)
    write(out/'complete.json',dict(checkpoint=str(checkpoint),records=list(lookup.values()),max_coordinate_difference=max_coord,max_score_difference=max_score,max_relation_difference=max_rel,stage1_frozen=True,inference_batch_size=1,job_id=os.getenv('SLURM_JOB_ID')))


def tensor_batch(cs,device):
    import torch
    return [torch.tensor(np.stack([c[k] for c in cs]),device=device,dtype=torch.float32) for k in ['tokens','logits','coordinates','physical']]


def fit(dataset,device='cpu'):
    import torch
    from models.node_verifier import NodeVerifier
    torch.set_num_threads(1)
    cs=[load_record(r,True,dataset) for r in records(dataset) if r['split']=='train']
    x=tensor_batch(cs,device);ys=[supervision(c['physical'],c['truth']) for c in cs]
    y=torch.tensor(np.stack([a for a,b in ys]),device=device);w=torch.tensor(np.stack([b for a,b in ys]),device=device)
    results={}
    for context in [False,True]:
        name='V3' if context else 'V2';torch.manual_seed(SEED);np.random.seed(SEED)
        net=NodeVerifier(x[0].shape[-1],context=context).to(device)
        # Fit one reliably labeled patch first; same inputs and loss as pilot.
        clean=int(np.argmax([(a==1).sum() for a,b in ys]));assert (y[clean]==1).sum()>=3
        tiny=copy.deepcopy(net);opt=torch.optim.AdamW(tiny.parameters(),lr=.003,weight_decay=0.)
        losses=[]
        for step in range(600):
            opt.zero_grad();z=tiny(*[v[clean:clean+1] for v in x]);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,y[clean:clean+1].clamp_min(0),reduction='none')*w[clean:clean+1]).sum()/w[clean].sum();loss.backward();opt.step();losses.append(float(loss.detach()))
        with torch.no_grad():
            z=tiny(*[v[clean:clean+1] for v in x]);mask=w[clean]>0;acc=float(((z[0,mask]>0)==(y[clean,mask]>0)).float().mean())
        tiny_result=dict(sample=cs[clean]['record']['sample_id'],initial_loss=losses[0],final_loss=losses[-1],accuracy=acc,updates=600)
        write(ROOT/'runs'/dataset/f'{name}-overfit.json',tiny_result)
        assert losses[-1]<.08 and acc>.97,('Tiny clean set failed',dataset,name,tiny_result)
        # Identical seeded sample order, update count and optimizer for both arms.
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);history=[];start=time.time()
        for step in range(1,2001):
            ids=np.random.choice(len(cs),8,replace=False);opt.zero_grad();z=net(*[v[ids] for v in x]);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,y[ids].clamp_min(0),reduction='none')*w[ids]).sum()/w[ids].sum()
            assert torch.isfinite(loss);loss.backward()
            if step==1:assert net.output[-1].weight.grad.abs().sum()>0
            opt.step()
            if step in [100,500,1000,2000]:
                history.append(dict(step=step,loss=float(loss.detach())));print(dataset,name,history[-1],flush=True)
                path=ROOT/'runs'/dataset/f'{name}-{step}.pt';path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(state=net.state_dict(),context=context,token_width=x[0].shape[-1],step=step,seed=SEED),path)
        with torch.no_grad():
            permutation=torch.randperm(x[0].shape[1],device=device)
            expected=net(*[v[:2] for v in x])[:,permutation]
            actual=net(*[v[:2,permutation] for v in x])
            error=float((expected-actual).abs().max())
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
        results[name]=dict(overfit=tiny_result,history=history,seconds=time.time()-start,parameters=sum(p.numel() for p in net.parameters()),updates=2000,trained_real_feature_permutation_error=error)
    write(ROOT/'runs'/dataset/'training.json',results)


def calibrate(cs,scores):
    best=(-1,0,.5)
    for radius in [0.,.5,1.,2.]:
        for threshold in np.arange(.05,1.,.05):
            s=aggregate([measure(c,select(c,p,threshold,radius),graph=False) for c,p in zip(cs,scores)])['node_f1']
            if s>best[0]:best=(s,radius,float(threshold))
    return dict(radius=best[1],threshold=best[2],training_node_f1=best[0])


def paired(a,b,seed=SEED):
    subjects=sorted({r['subject'] for r in a});rng=np.random.default_rng(seed);samples=[]
    aa={s:[r for r in a if r['subject']==s] for s in subjects};bb={s:[r for r in b if r['subject']==s] for s in subjects}
    keys=['node_recall','node_f1','duplicate_rate','edge_f1','beta0_mae','beta1_mae']
    for _ in range(1000):
        ss=rng.choice(subjects,len(subjects),replace=True);ar=aggregate([r for s in ss for r in aa[s]]);br=aggregate([r for s in ss for r in bb[s]])
        samples.append([ar.get(k,0)-br.get(k,0) for k in keys])
    lo,hi=np.percentile(samples,[2.5,97.5],axis=0)
    return {k:[float(l),float(h)] for k,l,h in zip(keys,lo,hi)}


def evaluate(dataset,transfer=None):
    import torch
    from models.node_verifier import NodeVerifier
    torch.set_num_threads(1);cs=[load_record(r,True,dataset) for r in records(dataset)]
    masks={s:np.array([c['record']['split']==s for c in cs]) for s in ['train','val']}
    scores={'V0':[c['scores'] for c in cs],'V1':[c['scores'] for c in cs]}
    # Predeclared final checkpoint: validation is never used for early selection.
    for name in ['V2','V3']:
        cp=torch.load(ROOT/'runs'/(transfer or dataset)/f'{name}-2000.pt',map_location='cpu',weights_only=False)
        net=NodeVerifier(cp['token_width'],context=cp['context']);net.load_state_dict(cp['state']);net.eval()
        with torch.no_grad():scores[name]=[net(*tensor_batch([c],'cpu')).sigmoid()[0].numpy() for c in cs]
    result={'dataset':dataset,'transfer':transfer,'policies':{},'results':{},'paired':{}};details={}
    for name,ps in scores.items():
        train=[c for c,m in zip(cs,masks['train']) if m];tp=[p for p,m in zip(ps,masks['train']) if m]
        policy=dict(threshold=.5,radius=0) if name=='V0' else calibrate(train,tp)
        result['policies'][name]=policy;result['results'][name]={};details[name]={}
        for split,mask in masks.items():
            cohort=[c for c,m in zip(cs,mask) if m];pp=[p for p,m in zip(ps,mask) if m]
            result['results'][name][split]={'discrimination':discrimination(cohort,pp)};details[name][split]={}
            for mode in ['calibrated','matched_count']:
                for tol in [1.,2.]:
                    rr=[]
                    for c,p in zip(cohort,pp):
                        ids=select(c,p,policy['threshold'],policy['radius'],int((c['scores']>.5).sum()) if mode=='matched_count' else None)
                        row=measure(c,ids,tol);row.update(subject=c['record']['subject'],sample=c['record']['sample_id']);rr.append(row)
                    key=f'{mode}_{tol}';result['results'][name][split][key]=aggregate(rr);details[name][split][key]=rr
        print('evaluated',dataset,name,policy,result['results'][name]['val']['matched_count_2.0']['node_f1'],flush=True)
    # Matched precision is a descriptive PR-envelope comparison, not a deployed
    # validation-tuned policy. All thresholds and attained precision are retained.
    cohort=[c for c,m in zip(cs,masks['val']) if m]
    target=result['results']['V0']['val']['calibrated_2.0']['node_precision']
    for name,ps in scores.items():
        pp=[p for p,m in zip(ps,masks['val']) if m];radius=result['policies'][name]['radius'];curve=[]
        for threshold in np.linspace(0,1,201):
            rr=[measure(c,select(c,p,threshold,radius),graph=False) for c,p in zip(cohort,pp)]
            curve.append(dict(threshold=float(threshold),**aggregate(rr)))
        eligible=[r for r in curve if r['node_precision']>=target]
        chosen=max(eligible,key=lambda r:r['node_recall']) if eligible else None
        result['results'][name]['val']['precision_envelope']=dict(target=target,selected=chosen,descriptive_only=True)
        write(ROOT/'results'/dataset/f'{name}-pr.json',curve)
        print('precision envelope',dataset,name,chosen,flush=True)
        if chosen:
            rr=[]
            for c,p in zip(cohort,pp):
                row=measure(c,select(c,p,chosen['threshold'],radius));row.update(subject=c['record']['subject'],sample=c['record']['sample_id']);rr.append(row)
            details[name]['val']['precision_envelope_2.0']=rr
            result['results'][name]['val']['precision_envelope']['graph_metrics']=aggregate(rr)
    for control in ['V0','V1','V2']:
        result['paired']['V3-'+control]={key:paired(details['V3']['val'][key],details[control]['val'][key]) for key in details['V3']['val']}
    # Candidate selection oracle; relation scores remain unchanged, so this is
    # a node recovery bound and an illustrative graph result, not an edge oracle.
    result['oracle']={str(tol):aggregate([measure(c,list(association(c['physical'],c['truth'],tol)),tol) for c in cohort]) for tol in [1.,2.]}
    result['learning_curve']=[]
    for name in ['V2','V3']:
        for step in [100,500,1000,2000]:
            cp=torch.load(ROOT/'runs'/(transfer or dataset)/f'{name}-{step}.pt',map_location='cpu',weights_only=False)
            net=NodeVerifier(cp['token_width'],context=cp['context']);net.load_state_dict(cp['state']);net.eval()
            for split,mask in masks.items():
                cohort=[c for c,m in zip(cs,mask) if m]
                with torch.no_grad():pp=[net(*tensor_batch([c],'cpu')).sigmoid()[0].numpy() for c in cohort]
                metrics=aggregate([measure(c,select(c,p,count=int((c['scores']>.5).sum())),graph=False) for c,p in zip(cohort,pp)])
                result['learning_curve'].append(dict(arm=name,step=step,split=split,**metrics,**discrimination(cohort,pp)))
    write(ROOT/'results'/dataset/'summary.json',result);write(ROOT/'results'/dataset/'patches.json',details)
    print(dataset,{n:r['val']['matched_count_2.0']['node_f1'] for n,r in result['results'].items()},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline','brava-policy','extract','fit','evaluate']);p.add_argument('--dataset',choices=list(ARMS));p.add_argument('--device',default='cpu');p.add_argument('--transfer');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if a.mode=='baseline':baseline()
    elif a.mode=='brava-policy':brava_policy()
    elif a.mode=='extract':extract(a.dataset)
    elif a.mode=='fit':fit(a.dataset,a.device)
    else:evaluate(a.dataset,a.transfer)


if __name__=='__main__':main()
