#!/usr/bin/env python3
"""Frozen IXI foreground probes and node-error accounting; no detector updates."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import build_model
from models.checkpoint import load_checkpoint_file
from data.loaders import build_data_loaders, build_evaluation_loader
from scripts.frozen_spline_diagnostic import training_evaluation_loader
from scripts.diagnose_branch_relations import match_nodes
from training.engine import _move_graph_batch

RUN=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_transfer_20261015/runs/ixi_delayed_spline')
SEED=181027

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def choose(root,split,n):
    groups={}
    for p in sorted((root/split/'spline').glob('*.npz')): groups.setdefault(p.stem.rsplit('_',1)[0],[]).append(p.stem)
    rank=lambda s:hashlib.sha256((str(SEED)+s).encode()).hexdigest()
    groups={s:sorted(v,key=rank) for s,v in groups.items()}
    order=sorted(groups,key=rank);ids=[];depth=0
    while len(ids)<n:
        layer=[groups[s][depth] for s in order if len(groups[s])>depth]
        if not layer:break
        ids.extend(layer[:n-len(ids)]);depth+=1
    return ids

def node_rows(pred,all_queries,cache,split,name):
    gt=cache['nodes']@cache['metric'].T;px=pred@cache['metric'].T;qx=all_queries@cache['metric'].T
    degree=np.bincount(cache['edges'].ravel(),minlength=len(gt))
    types=np.where(cache['boundary'],'boundary',np.where(degree>=3,'junction',np.where(degree==1,'termination','other')))
    rows=[]
    for tol in (1.,2.):
        mapping=match_nodes(px,gt,tol);inverse={j:i for i,j in mapping.items()}
        duplicates={i for i in range(len(px)) if i not in mapping and any(np.linalg.norm(px[i]-gt[j])<=tol for j in inverse)}
        left_p=[i for i in range(len(px)) if i not in mapping and i not in duplicates];left_g=[i for i in range(len(gt)) if i not in inverse]
        extra=match_nodes(px[left_p],gt[left_g],2*tol)
        displaced={left_g[j]:left_p[i] for i,j in extra.items()}
        used=set(mapping)|set(displaced.values())
        for j,g in enumerate(gt):
            state='recovered' if j in inverse else 'displaced' if j in displaced else 'missing'
            idx=inverse.get(j,displaced.get(j))
            rows.append(dict(split=split,sample_id=name,tolerance_mm=tol,population='GT',index=j,type=str(types[j]),outcome=state,
                error_mm=float(np.linalg.norm(px[idx]-g)) if idx is not None else None,
                nearest_any_query_mm=float(np.linalg.norm(qx-g,axis=1).min()),
                D=float(cache['nodes'][j,0]*64),H=float(cache['nodes'][j,1]*64),W=float(cache['nodes'][j,2]*64)))
        for i in range(len(px)):
            if i in used:continue
            d=np.linalg.norm(gt-px[i],axis=1);j=int(d.argmin())
            state='duplicate_candidate' if i in duplicates else 'unmatched_other'
            rows.append(dict(split=split,sample_id=name,tolerance_mm=tol,population='prediction',index=i,type=str(types[j]),outcome=state,
                error_mm=float(d[j]),nearest_any_query_mm=None,D=float(pred[i,0]*64),H=float(pred[i,1]*64),W=float(pred[i,2]*64)))
    return rows

def sample_features(features,vox,shape):
    # Same interpolation convention for all stages; CNN lattice alignment is approximate.
    grid=2*(vox.float()+.5)/torch.tensor(shape,device=vox.device)-1
    grid=grid[:,[2,1,0]].reshape(1,-1,1,1,3)
    return {k:F.grid_sample(v,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,:,0,0].T.cpu() for k,v in features.items()}

def metrics(y,prob):
    y=np.asarray(y).astype(bool);prob=np.asarray(prob);hit=prob>=.5
    tp=int((hit&y).sum());fp=int((hit&~y).sum());fn=int((~hit&y).sum())
    return dict(n=len(y),foreground=int(y.sum()),prevalence=float(y.mean()),
                average_precision=float(average_precision_score(y,prob)) if y.any() else None,
                roc_auc=float(roc_auc_score(y,prob)) if y.any() and (~y).any() else None,
                dice_at_half=2*tp/max(2*tp+fp+fn,1))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(1)
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((RUN/'resolved-config.yaml').read_text());cfg['data'].update(train_augmentation=False,batch_size=1,validation_batch_size=1);cfg['runtime']['workers']=0
    root=Path(cfg['data']['datasets']['synthetic_mri']['root'])
    ids={s:choose(root,s,n) for s,n in [('train',4 if a.smoke else 128),('val',4 if a.smoke else 64)]}
    assert not {x.rsplit('_',1)[0] for x in ids['train']}&{x.rsplit('_',1)[0] for x in ids['val']}
    train,_=build_data_loaders(cfg)
    loaders={'train':training_evaluation_loader(train,cfg,ids['train']),
             'val':build_evaluation_loader(cfg,dataset_name='synthetic_mri',sample_ids=ids['val'])}
    checkpoint=load_checkpoint_file(RUN/'best.pt');checkpoint_step=int(checkpoint['steps']);model=build_model(cfg).cuda();model.load_state_dict(checkpoint['net'],strict=True)
    model.eval();model.requires_grad_(False)
    buffers={k:v.clone() for k,v in model.named_buffers()}
    versions={k:v._version for k,v in model.named_parameters()}
    captured={}
    def hook(name):return lambda m,args,out:captured.__setitem__(name,out.detach())
    handles=[model.encoder.layer0.relu1.register_forward_hook(hook('early_cnn')),model.decoder.encoder.register_forward_hook(hook('encoder'))]
    fit={};evaluation={};all_nodes=[];examples=[];feature_shapes={};rng=np.random.default_rng(SEED)
    for split,loader in loaders.items():
        evaluation[split]=[]
        for index,batch in enumerate(loader):
            name=loader.dataset.records[index].sample_id
            with torch.no_grad():
                volumes,_=_move_graph_batch(batch,'cuda',cfg['training']['input'],True)
                tokens,predictions,cnn=model(volumes)
                enc=captured['encoder'].transpose(1,2).reshape_as(cnn)
                features=dict(early_cnn=captured['early_cnn'],projected_cnn=cnn,encoder=enc,raw_intensity=volumes)
                feature_shapes={k:list(v.shape[1:]) for k,v in features.items()}
                mask=(batch[1][0,0]>0).cpu().numpy();shape=mask.shape
                assert tuple(shape)==tuple(volumes.shape[2:])
                natural=rng.choice(mask.size,min(2048,mask.size),replace=False)
                vox=np.stack(np.unravel_index(natural,shape),axis=1)
                ev=sample_features(features,torch.tensor(vox,device='cuda'),shape)
                labels=torch.tensor(mask.ravel()[natural],dtype=torch.float32)
                evaluation[split].append(dict(id=name,x=ev,y=labels))
                if split=='train':
                    fg=np.flatnonzero(mask.ravel());bg=np.flatnonzero(~mask.ravel());assert len(fg) and len(bg)
                    ix=np.concatenate([rng.choice(fg,256,replace=len(fg)<256),rng.choice(bg,256,replace=len(bg)<256)])
                    vx=np.stack(np.unravel_index(ix,shape),axis=1)
                    x=sample_features(features,torch.tensor(vx,device='cuda'),shape)
                    for k,v in x.items():fit.setdefault(k,[]).append(v)
                threshold=cfg['evaluation'].get('node_threshold')
                valid=predictions['pred_logits'][0].argmax(-1)==1 if threshold is None else predictions['pred_logits'][0].softmax(-1)[:,1]>threshold
                queries=predictions['pred_nodes'][0,:,:3].cpu().numpy();nodes=queries[valid.cpu().numpy()]
                with np.load(root/split/'spline'/f'{name}.npz') as c:cache={k:c[k] for k in ('nodes','metric','edges','boundary')}
                all_nodes.extend(node_rows(nodes,queries,cache,split,name))
                if index<2:
                    examples.append(dict(split=split,id=name,raw=volumes[0,0].cpu(),mask=mask,
                                         features={k:v.cpu() for k,v in features.items()},gt=cache['nodes']*shape,pred=nodes*shape))
            if (index+1)%16==0:print(json.dumps(dict(stage='features',split=split,patches=index+1)),flush=True)
    for h in handles:h.remove()
    assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
    assert all(v.grad is None and v._version==versions[k] for k,v in model.named_parameters())
    del model,checkpoint,buffers
    torch.cuda.empty_cache()
    fit={k:torch.cat(v) for k,v in fit.items()}
    y=torch.tensor(np.tile(np.r_[np.ones(256),np.zeros(256)],len(ids['train'])),dtype=torch.float32)
    result={};histories={};probe_states={};per_patch=[]
    for key in [*fit,'encoder_shuffled_labels']:
        source='encoder' if key=='encoder_shuffled_labels' else key
        X=fit[source].cuda();mean=X.mean(0);std=X.std(0).clamp_min(1e-5);X=(X-mean)/std
        yy=y[torch.randperm(len(y))] if key.endswith('shuffled_labels') else y
        yy=yy.cuda();torch.manual_seed(SEED)
        head=torch.nn.Linear(X.shape[1],1).cuda();optimizer=torch.optim.Adam(head.parameters(),lr=.01,weight_decay=1e-4)
        history=[]
        for step in range(1,(20 if a.smoke else 1000)+1):
            ix=torch.randint(len(X),(2048,),device='cuda')
            loss=F.binary_cross_entropy_with_logits(head(X[ix]).squeeze(-1),yy[ix])
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();optimizer.step()
            if step==1 or step%100==0:history.append(dict(step=step,balanced_fit_bce=float(loss.detach())))
        result[key]={};histories[key]=history
        for split,items in evaluation.items():
            ys=[];ps=[]
            for item in items:
                with torch.no_grad():prob=head((item['x'][source].cuda()-mean)/std).sigmoid().flatten().cpu().numpy()
                target=item['y'].numpy();ys.extend(target);ps.extend(prob)
                per_patch.append(dict(probe=key,split=split,sample_id=item['id'],**metrics(target,prob)))
            result[key][split]=metrics(ys,ps)
        probe_states[key]=dict(weight=head.weight.detach().cpu(),bias=head.bias.detach().cpu(),mean=mean.cpu(),std=std.cpu())
        print(json.dumps(dict(stage='probe',name=key,results=result[key])),flush=True)
    # Save tiny readouts, not image feature caches or detector copies.
    torch.save(probe_states,a.output/'probes.pt')
    for example in examples:
        fig,axes=plt.subplots(2,4,figsize=(14,7));axis=0
        k=int(np.argmax(example['mask'].sum(axis=(1,2))))
        raw=example['raw'].numpy()[k].T;mask=example['mask'][k].T
        axes[0,0].imshow(raw,cmap='gray',origin='lower');axes[0,0].set_title('Raw')
        axes[0,1].imshow(mask,origin='lower',vmin=0,vmax=1);axes[0,1].set_title('Mask reference')
        axes[0,2].imshow(raw,cmap='gray',origin='lower');axes[0,2].set_title('GT green / predicted magenta nodes')
        for pts,color in [(example['gt'],'lime'),(example['pred'],'magenta')]:
            q=pts[np.abs(pts[:,0]-k)<=1.5];axes[0,2].scatter(q[:,1],q[:,2],s=35,c=color,marker='x')
        axes[0,3].axis('off')
        for ax,key in zip(axes[1],fit):
            st=probe_states[key];f=example['features'][key]
            logits=F.conv3d((f-st['mean'][None,:,None,None,None])/st['std'][None,:,None,None,None],st['weight'][:,:,None,None,None],st['bias'])
            prob=F.interpolate(logits,size=example['mask'].shape,mode='trilinear',align_corners=False).sigmoid()[0,0,k].numpy().T
            ax.imshow(prob,origin='lower',vmin=0,vmax=1,cmap='magma');ax.set_title(key)
        fig.suptitle(example['split']+' '+example['id']+f' axis0 slice{k}; probe scores uncalibrated (balanced fitting)')
        fig.tight_layout();fig.savefig(a.output/(example['split']+'-'+example['id']+'.png'),dpi=120);plt.close(fig)
    for filename,rows in [('node-errors.csv',all_nodes),('probe-per-patch.csv',per_patch)]:
        with (a.output/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    node_summary={}
    for split in loaders:
        node_summary[split]={}
        for tol in (1.,2.):
            selected=[r for r in all_nodes if r['split']==split and r['tolerance_mm']==tol]
            groups={}
            for kind in ('all','boundary','junction','termination','other'):
                rr=[r for r in selected if r['population']=='GT' and (kind=='all' or r['type']==kind)]
                groups[kind]=dict(total=len(rr),**{s:sum(r['outcome']==s for r in rr) for s in ('recovered','displaced','missing')},
                    unrecovered_with_any_query_nearby=sum(r['outcome']!='recovered' and r['nearest_any_query_mm']<=tol for r in rr))
            groups['unmatched_predictions']={s:sum(r['population']=='prediction' and r['outcome']==s for r in selected) for s in ('duplicate_candidate','unmatched_other')}
            node_summary[split][str(tol)]=groups
    summary=dict(checkpoint=str(RUN/'best.pt'),checkpoint_step=checkpoint_step,
                 seed=SEED,smoke=a.smoke,ids=ids,feature_shapes=feature_shapes,probes=result,node_errors=node_summary,
                 protocol=dict(fit='512 balanced voxels/patch; fixed 1000 Adam steps, lr .01, weight decay 1e-4; 20 steps smoke',
                     evaluation='2048 uniform voxels without replacement per patch; natural foreground prevalence; same voxels for each probe',
                     node_matching='position-only maximum cardinality at 1/2mm; reserve duplicate candidates near recovered nodes, then residual matching out to 2x tolerance labels displaced; unmatched remainder missing',
                     boundary_priority=True,detector_frozen=True,bn_buffers_unchanged=True,
                     caveats=['Probe failure is not proof of absent information; linear readout and spatial resolution limit recovery.',
                              'Early CNN differs in resolution; projected CNN versus encoder uses identical spatial grids.',
                              'Feature interpolation uses normalized dense-grid convention; actual CNN receptive field phase is approximate.',
                              'Duplicate candidates are proximity-defined, not proven redundant anatomical objects.',
                              'An unselected query near GT does not establish a correct object; it flags confidence/query-assignment ambiguity.',
                              'Voxel observations and patches are correlated; no significance claim or test-set use.',
                              'Boundary nodes override graph degree for mutually exclusive GT categories.']))
    save(a.output/'summary.json',summary);save(a.output/'probe-history.json',histories)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    keys=list(result);x=np.arange(len(keys))
    for i,split in enumerate(loaders):axes[0].bar(x+i*.35,[result[k][split]['average_precision'] for k in keys],.35,label=split)
    axes[0].set_xticks(x+.175,keys,rotation=25,ha='right');axes[0].set_ylabel('Foreground average precision');axes[0].legend()
    for i,split in enumerate(loaders):
        vals=node_summary[split]['2.0']['all'];bottom=0
        for state,color in [('recovered','green'),('displaced','orange'),('missing','red')]:
            v=vals[state]/vals['total'];axes[1].bar(i,v,bottom=bottom,color=color,label=state if i==0 else None);bottom+=v
    axes[1].set_xticks([0,1],['train','validation']);axes[1].set_title('GT node outcomes at 2 mm');axes[1].legend()
    fig.tight_layout();fig.savefig(a.output/'overview.png',dpi=130);plt.close(fig)
    report=['# IXI frozen-feature and node diagnostics', '',
            'Checkpoint: '+summary['checkpoint']+'; step '+str(checkpoint_step),
            'Training patches: '+str(len(ids['train']))+'; validation patches: '+str(len(ids['val'])),
            '', '## Foreground probes', '',
            '| Probe | Train AP | Validation AP | Validation foreground prevalence |',
            '|---|---:|---:|---:|']
    for key,rr in result.items():
        report.append(f"|{key}|{rr['train']['average_precision']:.4f}|{rr['val']['average_precision']:.4f}|{rr['val']['prevalence']:.4f}|")
    report.extend(['', '## Node outcomes at 2 mm', '', '| Split/type | GT nodes | Recovered | Displaced (2–4 mm) | Missing |', '|---|---:|---:|---:|---:|'])
    for split in loaders:
        for kind,rr in node_summary[split]['2.0'].items():
            if kind=='unmatched_predictions':continue
            report.append(f"|{split}/{kind}|{rr['total']}|{rr['recovered']}|{rr['displaced']}|{rr['missing']}|")
    report.extend(['', '## Interpretation limits', '',
        'Inspect probe-history.json before interpreting a weak probe: incomplete readout optimization can mimic absent information.',
        'A projected-CNN/encoder gap localizes a loss of linear accessibility on a common spatial grid, not necessarily complete information destruction.',
        'Natural-prevalence AP is primary. Scores and Dice at 0.5 are not calibrated after balanced foreground/background fitting.',
        'Train/validation gaps are descriptive; patch populations and difficulty differ. See per-patch CSV and subtype counts.',
        'Missing/displaced/duplicate categories use the recorded tolerance rules and do not establish anatomical truth.',
        '', '![Overview](overview.png)', ''])
    report.extend('!['+ex['id']+']('+ex['split']+'-'+ex['id']+'.png)' for ex in examples)
    (a.output/'results.md').write_text('\n'.join(report)+'\n')
    print('COMPLETE '+str(a.output),flush=True)

if __name__=='__main__':main()
