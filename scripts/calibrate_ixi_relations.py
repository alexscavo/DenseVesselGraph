#!/usr/bin/env python3
"""Train-only relation threshold selection, fixed-node validation/topology audit."""
import argparse,json,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models import build_model
from models.checkpoint import load_checkpoint_file
from data.loaders import build_data_loaders,build_evaluation_loader
from scripts.frozen_spline_diagnostic import training_evaluation_loader
from scripts.controlled_spline_unfreeze import frozen_state_digest
from scripts.diagnose_branch_relations import match_nodes
from training.engine import _move_graph_batch
from training.evaluation.inference import infer_graphs

PARENT=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_node_refinement_20261018/local_image')

def betti(n,edges):
    parent=list(range(n));components=n
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in edges:
        a,b=root(int(a)),root(int(b))
        if a!=b:parent[a]=b;components-=1
    return components,len(edges)-n+components

def metrics(rows,t):
    tp=fp=total=0;b0=[];b1=[];matched_positive=0
    for r in rows:
        selected=r['score']>t;tp+=int(r['positive'][selected].sum());fp+=int((~r['positive'][selected]).sum());total+=r['target_edges'];matched_positive+=r['recoverable']
        p=betti(r['nodes'],r['pairs'][selected]);g=r['gt_betti'];b0.append(abs(p[0]-g[0]));b1.append(abs(p[1]-g[1]))
    return dict(threshold=float(t),tp=tp,fp=fp,fn=total-tp,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/total if total else 0.,f1=2*tp/(tp+fp+total) if total else 0.,conditional_recall=tp/matched_positive if matched_positive else 0.,beta0_mae=float(np.mean(b0)),beta1_mae=float(np.mean(b1)))

def thresholds(rows):
    p=np.concatenate([r['score'] for r in rows]);y=np.concatenate([r['positive'] for r in rows]);order=np.argsort(-p,kind='stable');p=p[order];y=y[order]
    end=np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1];tp=np.cumsum(y)[end];n=end+1;total=sum(r['target_edges'] for r in rows)
    f1=2*tp/(n+total);precision=tp/n
    # Include complete equal-score groups using the same strict > comparator as inference.
    cut=np.nextafter(p[end].astype(np.float64),-np.inf)
    best=int(np.argmax(f1));base=metrics(rows,.5);eligible=np.flatnonzero(precision>=base['precision'])
    conservative=int(eligible[np.argmax(tp[eligible])]) if len(eligible) else None
    return {'baseline':.5,'train_max_f1':float(cut[best]),'train_preserve_precision':float(cut[conservative]) if conservative is not None else .5}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(191030);torch.set_num_threads(1)
    adapter=load_checkpoint_file(PARENT/'trained-heads.pt');cfg=adapter['config'];cfg['data'].update(train_augmentation=False,batch_size=1,validation_batch_size=1);cfg['runtime']['workers']=0
    checkpoint=load_checkpoint_file(adapter['provenance']['initial_checkpoint']);model=build_model(cfg).cuda();r=model.load_state_dict(checkpoint['net'],strict=False);assert not r.unexpected_keys and all(k.startswith('node_refinement.') for k in r.missing_keys)
    model.load_state_dict(adapter['parameters'],strict=False);model.eval();model.requires_grad_(False);digest=frozen_state_digest(model)
    ids=json.loads(Path('experiments/ixi_relation_isolation/full-1278148/summary.json').read_text())['ids']
    if a.smoke:ids={k:v[:4] for k,v in ids.items()}
    assert not {x.rsplit('_',1)[0] for x in ids['train']}&{x.rsplit('_',1)[0] for x in ids['val']}
    root=Path(cfg['data']['datasets']['synthetic_mri']['root']);full,_=build_data_loaders(cfg)
    loaders={'train':training_evaluation_loader(full,cfg,ids['train']),'val':build_evaluation_loader(cfg,dataset_name='synthetic_mri',sample_ids=ids['val'])}
    data={};unsupported=0
    with torch.no_grad():
      for split,loader in loaders.items():
        data[split]=[]
        for i,batch in enumerate(loader):
            name=loader.dataset.records[i].sample_id;vol,_=_move_graph_batch(batch,'cuda',cfg['training']['input'],True);tokens,pred,_=model(vol)
            # Obtain every pair; this checkpoint uses plain symmetric relation logits.
            assert model.spline_head.curve_evidence is None
            graphs=infer_graphs(tokens,pred,model.relation_embed,object_queries=model.obj_token,relation_tokens=cfg['model']['decoder']['relation_tokens'],node_threshold=cfg['evaluation'].get('node_threshold'),edge_threshold=-1.)
            g=graphs[0];nodes=g['nodes'].numpy();pairs=g['edges'].numpy();score=g['edge_scores'].numpy().astype(np.float64)
            with np.load(root/split/'spline'/f'{name}.npz') as c:gt=c['nodes'];metric=c['metric'];edges=c['edges']
            links={tuple(sorted(map(int,e))) for e in edges if e[0]!=e[1]};unsupported+=len(edges)-len(links)
            assert len(links)==len(edges),'Closed/parallel GT needs explicit scoring adaptation'
            mapping=match_nodes(nodes@metric.T,gt@metric.T,2.);recovered=set(mapping.values())
            labels=np.array([a in mapping and b in mapping and tuple(sorted((mapping[a],mapping[b]))) in links for a,b in pairs],bool)
            data[split].append(dict(id=name,nodes=len(nodes),pairs=pairs,score=score,positive=labels,target_edges=len(links),recoverable=sum(a in recovered and b in recovered for a,b in links),gt_betti=betti(len(gt),edges)))
        print('scored',split,len(data[split]),flush=True)
    assert digest==frozen_state_digest(model)
    selected=thresholds(data['train'])
    rows=[dict(split=split,policy=policy,**metrics(rr,t)) for split,rr in data.items() for policy,t in selected.items()]
    per_patch=[dict(split=s,policy=policy,sample_id=r['id'],**metrics([r],t)) for s,rr in data.items() for policy,t in selected.items() for r in rr]
    for name,rr in [('comparison.csv',rows),('per-patch.csv',per_patch)]:
        with (a.output/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    summary=dict(parent=str(PARENT),ids=ids,smoke=a.smoke,thresholds=selected,results=rows,unsupported_targets=unsupported,
        method='All decoded-node pairs, 2mm positional node matching; unmatched-endpoint edges count as FP, all GT edges in recall denominator. Thresholds selected only on training patches. Betti counts on full decoded graphs including isolated nodes.',
        caveat='Betti count errors do not identify anatomical correctness; errors can cancel. Positional edge F1 is diagnostic, not the native benchmark matching. Train calibration is in-sample for the detector; validation remains a diagnostic set.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Cache pair graphs for transparent reinspection without another model forward.
    serial={s:[dict(r,pairs=r['pairs'].tolist(),score=r['score'].tolist(),positive=r['positive'].tolist()) for r in rr] for s,rr in data.items()}
    import gzip
    with gzip.open(a.output/'pair-graphs.json.gz','wt') as f:json.dump(serial,f)
    lines=['# IXI train-calibrated relation thresholds','','|Split|Policy|Threshold|Precision|Recall|F1|β0 MAE|β1 MAE|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+r['split']+'|'+r['policy']+'|'+'|'.join(f'{r[k]:.4f}' for k in ['threshold','precision','recall','f1','beta0_mae','beta1_mae'])+'|')
    (a.output/'results.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
