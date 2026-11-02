"""Eight patch CPU replay; frozen-head class-direction interventions, no optimization."""
import os,sys,json,time,io,zipfile,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/oracle_representatives_20261026'
sys.path.insert(0,str(OUT/'replay-runtime'));sys.path.insert(0,str(OUT/'replay-source'));sys.path.insert(1,str(ROOT))
import torch
import numpy as np
import nibabel as nib
from models import build_model
from scripts.analyze_graph_errors import read,write,csvwrite,geometry,match_hungarian
from scripts.graph_error_counterfactuals import confidence_oracle
from scripts.node_verifier_metrics import association

def score(head,left,right,relation):
    rel=relation.reshape(1,-1).expand(len(left),-1)
    return (.5*(head(torch.cat([left,right,rel],-1))+head(torch.cat([right,left,rel],-1)))).softmax(-1)[:,1]

def main(device='cpu'):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    selection=read(OUT/'smoke-embedding-replay-selection.json');rows=[];pairrows=[];audits=[]
    original=ROOT.parent/'Graph-Native-Betti-Matching-Representation'
    for seed in [364505,92741]:
        folder=original/f'experiments/local_refinement_long/{seed}/runs/A-xyz'
        cp=torch.load(folder/'model-12000.pt',map_location='cpu',mmap=True)
        cfg=read(folder/'config.json');cfg['model']['decoder']['use_cuda_extension']=device=='cuda'
        model=build_model(cfg).eval().requires_grad_(False);model.load_state_dict(cp['net'],strict=True);model.to(device);del cp
        z=zipfile.ZipFile(folder/'step-12000/caches.zip');u=(model.class_embed.weight[1]-model.class_embed.weight[0]);u=u/u.norm()
        for entry in [v for v in selection if v['seed']==seed]:
            r=entry['record'];gt=geometry(r['gt']);raw=Path(r['gt']).parent.parent/'raw'/(r['sample']+'_data.nii.gz')
            if not raw.exists():
                candidates=list(raw.parent.glob(r['sample']+'*.nii.gz'));assert len(candidates)==1;raw=candidates[0]
            image=np.asarray(nib.load(raw).dataobj,dtype=np.float32)-cfg['data']['datasets']['synthetic_mri']['foreground_mean']
            with np.load(io.BytesIO(z.read(r['sample']+'.npz'))) as f:c={k:f[k] for k in f.files}
            start=time.monotonic()
            with torch.inference_mode():
                tokens,pred,_=model(torch.from_numpy(image.copy())[None,None].to(device));t=tokens[0,:120];rel=tokens[0,120:122]
                pairs=torch.from_numpy(c['pairs']).to(device);prob=score(model.relation_embed,t[pairs[:,0]],t[pairs[:,1]],rel).cpu().numpy()
                errors=dict(coordinates_mm=float(np.abs(pred['pred_nodes'][0,:,:3].cpu().numpy()@gt['metric'].T-c['physical']).max()),confidence=float(np.abs(pred['pred_logits'][0].softmax(-1)[:,1].cpu().numpy()-c['scores']).max()),relation=float(np.abs(prob-c['relations']).max()))
                passed=errors['coordinates_mm']<.001 and errors['confidence']<.0001 and errors['relation']<.0001
                audits.append(dict(seed=seed,**r,seconds=time.monotonic()-start,parity=errors,passed=passed,token_shape=list(tokens.shape),inference_batch=1,device=device))
                write(OUT/'embedding-replay-audit.json',audits)
                print('replay',seed,r['sample'],audits[-1]['seconds'],errors,flush=True)
                if not passed:continue
                nearest=association(c['physical'],c['truth'],2.);_,conf=confidence_oracle(c,2.)
                n={g:q for q,g in nearest.items()};h={g:q for q,g in conf.items()};common=sorted(set(n)&set(h));stable=[g for g in common if n[g]==h[g]]
                np.savez_compressed(OUT/f'embeddings-{seed}-{r["sample"]}.npz',tokens=tokens.cpu().numpy(),query_ids=np.arange(120),physical=c['physical'],scores=c['scores'])
                adj=np.zeros((len(gt['nodes']),)*2,bool);a,b=gt['edges'].T;adj[a,b]=adj[b,a]=True
                for g in common:
                    qn,qc=n[g],h[g]
                    if qn==qc:continue
                    tn,tc=t[qn],t[qc];delta=tc-tn;along=delta.dot(u)*u
                    rows.append(dict(seed=seed,split=r['split'],sample=r['sample'],subject=r['subject'],gt_vertex=g,nearest_query=qn,confidence_query=qc,cosine=float(torch.nn.functional.cosine_similarity(tn,tc,dim=0)),nearest_norm=float(tn.norm()),confidence_norm=float(tc.norm()),class_axis_delta_energy_fraction=float(along.square().sum()/delta.square().sum()),class_logit_delta=float((model.class_embed(tc)[1]-model.class_embed(tc)[0])-(model.class_embed(tn)[1]-model.class_embed(tn)[0]))))
                    if not stable:continue
                    neighbors=torch.stack([t[n[v]] for v in stable]);states={'nearest':tn,'confidence':tc,'class_axis_only':tn+along,'orthogonal_only':tc-along}
                    values={name:score(model.relation_embed,x[None].expand(len(stable),-1),neighbors,rel).cpu().numpy() for name,x in states.items()}
                    for i,v in enumerate(stable):
                        pairrows.append(dict(seed=seed,split=r['split'],sample=r['sample'],subject=r['subject'],gt_vertex=g,neighbor=v,label=int(adj[g,v]),**{k+'_score':float(p[i]) for k,p in values.items()},nearest_neighbor_cosine=float(torch.nn.functional.cosine_similarity(tn,t[n[v]],dim=0)),confidence_neighbor_cosine=float(torch.nn.functional.cosine_similarity(tc,t[n[v]],dim=0))))
            # Do not retain the early CNN maps across patches.
            model._spline_local_features=None
        del model
    csvwrite(OUT/'embedding-alternatives.csv.gz',rows);csvwrite(OUT/'embedding-intervention-pairs.csv.gz',pairrows)
    write(OUT/'embedding-replay-summary.json',dict(patches=len(audits),passed=sum(r['passed'] for r in audits),representatives=len(rows),pairs=len(pairrows),audits=audits))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=p.parse_args();main(args.device)
