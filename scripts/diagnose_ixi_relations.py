#!/usr/bin/env python3
"""Frozen matched-token and GT-endpoint image relation probes."""
import argparse,copy,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml
from sklearn.metrics import average_precision_score
from data.loaders import build_data_loaders,build_evaluation_loader
from models import build_model
from models.checkpoint import load_checkpoint_file
from scripts.frozen_spline_diagnostic import training_evaluation_loader
from scripts.diagnose_ixi_features_nodes import choose
from scripts.diagnose_branch_relations import match_nodes
from scripts.controlled_spline_unfreeze import frozen_state_digest
from scripts.train_spline_weight_pilot import seed
from training.engine import _move_graph_batch

ROOT=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_node_refinement_20261018/local_image')


def pair_image_features(feature,nodes,pairs,shape):
    """16 chord samples, seven-point local cross stencil, shared endpoint reversal."""
    outputs=[];offsets=feature.new_tensor([[0,0,0],[2,0,0],[-2,0,0],[0,2,0],[0,-2,0],[0,0,2],[0,0,-2]])
    for pair in pairs.split(128):
        ends=nodes[pair]*feature.new_tensor(shape)
        t=torch.linspace(0,1,16,device=feature.device)[None,:,None]
        points=ends[:,0,None]*(1-t)+ends[:,1,None]*t
        points=points[:,:,None,:]+offsets[None,None]
        grid=2*(points/2+.5)/feature.new_tensor(feature.shape[-3:])-1
        grid=grid[...,[2,1,0]].reshape(1,len(pair),16*7,1,3)
        values=F.grid_sample(feature,grid,align_corners=False,padding_mode='border')[0,:,:, :,0]
        values=values.permute(1,2,0).reshape(len(pair),16,7,-1).mean(2)
        values=.5*(values+values.flip(1))
        outputs.append(values.flatten(1).cpu())
    return torch.cat(outputs) if outputs else torch.empty((0,16*feature.shape[1]))


def scores(y,p):
    y=np.asarray(y,bool);p=np.asarray(p);hit=p>.5
    tp=int((hit&y).sum());fp=int((hit&~y).sum());fn=int((~hit&y).sum())
    return dict(pairs=len(y),positives=int(y.sum()),prevalence=float(y.mean()) if len(y) else None,
        ap=float(average_precision_score(y,p)) if y.any() else None,
        precision_at_half=tp/(tp+fp) if tp+fp else None,recall_at_half=tp/(tp+fn) if tp+fn else None,
        false_positive_rate_at_half=fp/int((~y).sum()) if (~y).any() else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);seed(191029);torch.set_num_threads(1)
    adapter=load_checkpoint_file(ROOT/'trained-heads.pt');cfg=adapter['config'];cfg['data'].update(train_augmentation=False,batch_size=1,validation_batch_size=1);cfg['runtime']['workers']=0
    cp=load_checkpoint_file(adapter['provenance']['initial_checkpoint']);model=build_model(cfg).cuda();base=model.load_state_dict(cp['net'],strict=False);assert not base.unexpected_keys and all(k.startswith('node_refinement.') for k in base.missing_keys)
    report=model.load_state_dict(adapter['parameters'],strict=False);assert not report.unexpected_keys
    assert model.spline_head.curve_evidence is None
    assert not cfg['evaluation'].get('node_type_degree_caps',False) and not cfg['evaluation'].get('node_type_degree_penalty',0)
    model.eval();model.requires_grad_(False);before=frozen_state_digest(model)
    root=Path(cfg['data']['datasets']['synthetic_mri']['root']);train,_=build_data_loaders(cfg)
    ids={s:choose(root,s,4 if a.smoke else n) for s,n in [('train',256),('val',128)]}
    assert not {x.rsplit('_',1)[0] for x in ids['train']}&{x.rsplit('_',1)[0] for x in ids['val']}
    loaders={'train':training_evaluation_loader(train,cfg,ids['train']),'val':build_evaluation_loader(cfg,dataset_name='synthetic_mri',sample_ids=ids['val'])}
    captured={};handle=model.encoder.layer0.relu1.register_forward_hook(lambda m,i,o:captured.update(early=o.detach()))
    data={};patches=[];excluded=0
    with torch.no_grad():
      for split,loader in loaders.items():
        chunks=[]
        for index,batch in enumerate(loader):
            name=loader.dataset.records[index].sample_id;vol,_=_move_graph_batch(batch,'cuda',cfg['training']['input'],True)
            tokens,pred,_=model(vol)
            with np.load(root/split/'spline'/f'{name}.npz') as c:
                gt=c['nodes'];metric=c['metric'];edges=c['edges'];paths=c['full_paths'];offset=c['path_offsets']
            if len(gt)<2:excluded+=1;continue
            pairs=torch.combinations(torch.arange(len(gt),device='cuda'),r=2);pairs_np=pairs.cpu().numpy()
            lengths={tuple(sorted(map(int,e))):float(np.linalg.norm(np.diff(paths[lo:hi]@metric.T,axis=0),axis=1).sum()) for e,lo,hi in zip(edges,offset[:-1],offset[1:]) if e[0]!=e[1]}
            labels=np.array([tuple(e) in lengths for e in pairs_np]);arc=np.array([lengths.get(tuple(e),np.nan) for e in pairs_np])
            coords=torch.tensor(gt,device='cuda');ends=coords[pairs]
            chord=np.linalg.norm((gt[pairs_np[:,0]]-gt[pairs_np[:,1]])@metric.T,axis=1)
            geom=torch.cat([ends.mean(1),(ends[:,0]-ends[:,1]).abs(),torch.tensor(chord,device='cuda')[:,None]],1).float().cpu()
            image=pair_image_features(captured['early'],coords,pairs,vol.shape[-3:])
            threshold=cfg['evaluation'].get('node_threshold');valid=pred['pred_logits'][0].argmax(-1)==1 if threshold is None else pred['pred_logits'][0].softmax(-1)[:,1]>threshold
            selected=torch.where(valid)[0].cpu().numpy();px=pred['pred_nodes'][0,selected,:3].cpu().numpy()
            mapping=match_nodes(px@metric.T,gt@metric.T,2.);inverse={j:int(selected[i]) for i,j in mapping.items()}
            keep=np.array([int(i) in inverse and int(j) in inverse for i,j in pairs_np]);ki=np.flatnonzero(keep)
            q=model.obj_token;rel=tokens[0,q:q+cfg['model']['decoder']['relation_tokens']].flatten()
            qp=torch.tensor([[inverse[int(i)],inverse[int(j)]] for i,j in pairs_np[keep]],device='cuda',dtype=torch.long).reshape(-1,2)
            forward=torch.cat([tokens[0,qp[:,0]],tokens[0,qp[:,1]],rel.expand(len(qp),-1)],1)
            reverse=torch.cat([tokens[0,qp[:,1]],tokens[0,qp[:,0]],rel.expand(len(qp),-1)],1)
            native=(.5*(model.relation_embed(forward)+model.relation_embed(reverse))).softmax(-1)[:,1].cpu()
            chunks.append(dict(image=image,geom=geom,y=torch.tensor(labels,dtype=torch.float32),keep=torch.tensor(keep),forward=forward.cpu(),reverse=reverse.cpu(),native=native,arc=arc,chord=chord))
            patches.append(dict(split=split,sample_id=name,pairs=len(pairs),matched_pairs=int(keep.sum()),positives=int(labels.sum()),matched_positives=int(labels[keep].sum())))
            if (index+1)%32==0:print(split,index+1,flush=True)
        assert chunks
        data[split]={k:torch.cat([v[k] for v in chunks]) for k in ('image','geom','y','keep','forward','reverse','native')}
        data[split].update(arc=np.concatenate([v['arc'] for v in chunks]),chord=np.concatenate([v['chord'] for v in chunks]))
    handle.remove();assert frozen_state_digest(model)==before
    original=copy.deepcopy(model.relation_embed).cuda();original.requires_grad_(True);del model,cp,adapter
    history={};probabilities={'native':{s:d['native'].numpy() for s,d in data.items()}}
    weights={};steps=10 if a.smoke else 1000
    for arm in ('token_refit','geometry_only','image_geometry'):
        seed(191029)
        if arm=='token_refit':
            head=original;X=data['train']['forward'].cuda();R=data['train']['reverse'].cuda();y=data['train']['y'][data['train']['keep']].cuda()
            mean=std=None
        else:
            allx=torch.cat([data['train']['geom'],data['train']['image']],1).cuda();mean=allx.mean(0);std=allx.std(0).clamp_min(1e-5);X=(allx-mean)/std
            if arm=='geometry_only':X[:,7:]=0
            y=data['train']['y'].cuda();head=nn.Sequential(nn.Linear(X.shape[1],128),nn.ReLU(),nn.Linear(128,2)).cuda()
        positive=torch.where(y==1)[0];negative=torch.where(y==0)[0];assert len(positive) and len(negative)
        optim=torch.optim.AdamW(head.parameters(),lr=7e-5 if arm=='token_refit' else 1e-4,weight_decay=1e-4);log=[]
        for step in range(1,steps+1):
            ix=torch.cat([positive[torch.randint(len(positive),(256,),device='cuda')],negative[torch.randint(len(negative),(256,),device='cuda')]])
            logits=.5*(head(X[ix])+head(R[ix])) if arm=='token_refit' else head(X[ix]);loss=F.cross_entropy(logits,y[ix].long());assert torch.isfinite(loss)
            optim.zero_grad();loss.backward();optim.step()
            if step==1 or step%100==0:log.append(dict(step=step,balanced_bce=float(loss.detach())))
        history[arm]=log;probabilities[arm]={}
        for split,d in data.items():
            pp=[]
            with torch.no_grad():
                size=len(d['forward']) if arm=='token_refit' else len(d['y'])
                for start in range(0,size,2048):
                    if arm=='token_refit':logits=.5*(head(d['forward'][start:start+2048].cuda())+head(d['reverse'][start:start+2048].cuda()))
                    else:
                        z=(torch.cat([d['geom'][start:start+2048],d['image'][start:start+2048]],1).cuda()-mean)/std
                        if arm=='geometry_only':z[:,7:]=0
                        logits=head(z)
                    pp.append(logits.softmax(-1)[:,1].cpu())
            probabilities[arm][split]=torch.cat(pp).numpy()
        weights[arm]=dict(state={k:v.cpu() for k,v in head.state_dict().items()},mean=mean.cpu() if mean is not None else None,std=std.cpu() if std is not None else None)
        print('fitted',arm,flush=True)
    # Length bins use sampled training positives only; negatives use chord distance.
    cuts=np.quantile(data['train']['arc'][np.isfinite(data['train']['arc'])],[.25,.5,.75])
    results={};rows=[]
    for split,d in data.items():
      for arm,predprob in probabilities.items():
        populations=['matched'] if arm in ('native','token_refit') else ['all_gt','matched']
        for pop in populations:
            keep=d['keep'].numpy() if pop=='matched' else np.ones(len(d['y']),bool)
            yy=d['y'].numpy()[keep];pr=predprob[split] if arm in ('native','token_refit') else predprob[split][keep]
            row=dict(split=split,arm=arm,population=pop,**scores(yy,pr));rows.append(row)
            arc=d['arc'][keep];chord=d['chord'][keep];bins=[]
            for i in range(4):
                select=(yy==1)&(np.searchsorted(cuts,arc,side='right')==i)
                bins.append(dict(bin=i,positives=int(select.sum()),recall_at_half=float(np.mean(pr[select]>.5)) if select.any() else None))
            near=(yy==0)&(chord<=5.)
            results[f'{split}/{arm}/{pop}']=dict(row,positive_arc_bins=bins,near_negative_pairs=int(near.sum()),near_negative_fpr_at_half=float(np.mean(pr[near]>.5)) if near.any() else None)
    summary=dict(parent=str(ROOT),seed=191029,smoke=a.smoke,steps=steps,ids=ids,excluded_fewer_than_two_nodes=excluded,gt_arc_quartiles_mm=cuts.tolist(),frozen_state_unchanged=True,results=results,
        protocol='2mm positional matching; original native symmetrized logits; train-only balanced pair fitting; natural candidate prevalence evaluation; chord samples at GT endpoints for image probe; geometry inputs midpoint, absolute separation, physical chord length')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(a.output/'fit-history.json').write_text(json.dumps(history,indent=2)+'\n')
    for name,rr in [('comparison.csv',rows),('patch-counts.csv',patches)]:
        with (a.output/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    np.savez_compressed(a.output/'pair-scores.npz',**{f'{s}_{k}':v.numpy() if torch.is_tensor(v) else v for s,d in data.items() for k,v in d.items() if k in ('y','keep','arc','chord')},**{f'{s}_{arm}_prob':v for arm,dd in probabilities.items() for s,v in dd.items()})
    torch.save(weights,a.output/'probe-heads.pt')
    lines=['# IXI relation isolation','','|Split|Arm|Pairs|AP|Precision @.5|Recall @.5|','|---|---|---|---:|---:|---:|']
    for r in rows:lines.append('|'+ '|'.join(str(r[k]) for k in ['split','arm','population','ap','precision_at_half','recall_at_half'])+'|')
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=0;examples=[]
    for row in [r for r in patches if r['split']=='val'][:2]:
        name=row['sample_id'];count=row['pairs'];y=data['val']['y'][start:start+count].numpy()
        chord=data['val']['chord'][start:start+count];positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
        selected=([int(positive[0])] if len(positive) else [])+([int(negative[np.argmin(chord[negative])])] if len(negative) else [])
        raw=nib.load(root/'val/raw'/f'{name}_data.nii.gz').get_fdata(dtype=np.float32)
        with np.load(root/'val/spline'/f'{name}.npz') as c:gt=c['nodes']*np.array(raw.shape);paths=c['full_paths']*np.array(raw.shape);off=c['path_offsets']
        pairs=np.array([(i,j) for i in range(len(gt)) for j in range(i+1,len(gt))])
        fig,axes=plt.subplots(len(selected),3,figsize=(12,4*len(selected)),squeeze=False)
        for r,index in enumerate(selected):
            ends=gt[pairs[index]];label='GT adjacent' if y[index] else 'GT nonadjacent (nearest pair)'
            pi=float(probabilities['image_geometry']['val'][start+index]);pg=float(probabilities['geometry_only']['val'][start+index])
            examples.append(dict(sample_id=name,pair_index=index,gt_adjacent=bool(y[index]),chord_mm=float(chord[index]),image_score=pi,geometry_score=pg))
            for axis,ax in enumerate(axes[r]):
                dims=[i for i in range(3) if i!=axis];ax.imshow(raw.max(axis=axis).T,origin='lower',cmap='gray')
                for lo,hi in zip(off[:-1],off[1:]):ax.plot(paths[lo:hi,dims[0]],paths[lo:hi,dims[1]],color='lime',alpha=.5,linewidth=.6)
                ax.plot(ends[:,dims[0]],ends[:,dims[1]],'o--',color='cyan' if y[index] else 'magenta',linewidth=2)
                ax.set_title(f'{label}; axis {axis}\nimage {pi:.3f}, geometry {pg:.3f}')
        fig.suptitle(name+' | green: reference paths; dashed: sampled chord; projections can overlap')
        fig.tight_layout();fig.savefig(a.output/(name+'-pairs.png'),dpi=120);plt.close(fig)
        lines+=['',f'![Fixed pair examples]({name}-pairs.png)'];start+=count
    (a.output/'visual-examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    (a.output/'results.md').write_text('\n'.join(lines)+'\n');print('COMPLETE',a.output,flush=True)

if __name__=='__main__':main()
