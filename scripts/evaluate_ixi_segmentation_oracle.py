"""CPU native/physical metrics and training-calibrated mask-input comparison."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from scripts.ixi_cached_mechanisms import save,table,measure,summarize_run
from scripts.ixi_segmentation_oracle import ARMS
from scripts.ixi_relation_matching_calibration import Geometry,select_threshold
from scripts.diagnose_branch_relations import match_nodes
from scripts.diagnose_ixi_queries import graph


def spline_head(root):
    from models.spline import SplineHead
    from models.checkpoint import load_checkpoint_file
    plan=json.loads((root/'plan.json').read_text());cp=load_checkpoint_file(plan['initializer'])
    cfg=cp['config'];s=cfg['model']['spline'];w=cfg['model']['decoder']['hidden_dim'];r=cfg['model']['decoder']['relation_tokens']
    head=SplineHead(w*(2+r),w,s.get('controls',8),s.get('samples',64),s.get('residual_scale','mm'),
        family=s.get('family','bspline'),token_inputs=s.get('token_inputs','full'),local_geometry=s.get('local_geometry'))
    head.load_state_dict({k.removeprefix('spline_head.'):v for k,v in cp['net'].items() if k.startswith('spline_head.')},strict=True)
    return head.eval().requires_grad_(False)


def native_prediction(c,g):
    q=g['query_ids'];lookup={tuple(sorted(e)):i for i,e in enumerate(c['edges'])}
    pairs=q[g['edges']]
    return dict(g,boxes=c['boxes'][q],node_scores=c['node_scores'][q],
        edge_scores=np.array([c['edge_scores'][lookup[tuple(sorted(e))]] for e in pairs]))


def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.integer,np.floating)):return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def evaluate(root,arm,smoke=False):
    from scripts.evaluate_ixi_node_focal import evaluate as common_evaluate
    from training.evaluation.metrics import evaluate_graph,summarize_metrics
    # Existing uniqueness/common-v2 evaluator, unmodified, on this new root only.
    common_evaluate(root,arm,smoke)
    predictions=root/('smoke-predictions' if smoke else 'predictions')/arm
    destination=root/('smoke-evaluation' if smoke else 'evaluation')/arm
    head=None
    for folder in sorted(predictions.iterdir()):
        records=json.loads((folder/'records.json').read_text());cfg=json.loads((folder/'config.json').read_text())
        summary=json.loads((folder/'summary.json').read_text());out=destination/folder.name
        protocol=cfg['evaluation']['protocol'];rows=[];states=[];localization=[]
        for record in records:
            with np.load(record['cache']) as f:c={k:f[k] for k in f.files}
            with np.load(record['gt']) as f:gt={k:f[k] for k in f.files}
            g=graph(c,c['retained']);g['query_ids']=c['retained'];prediction=native_prediction(c,g)
            transform=np.eye(4);transform[:3,:3]=gt['metric']
            scores,ns,es=evaluate_graph(prediction,gt['nodes'],gt['edges'],protocol=protocol,world_transform=transform,return_detection_states=True)
            key={k:record[k] for k in ['split','sample_id','subject','site']}
            rows.append(dict(**key,**scores));states.append(dict(**key,node=ns,edge=es))
            for tol in [1,2]:
                q=c['retained'];mapping=match_nodes(c['nodes'][q]@gt['metric'].T,gt['nodes']@gt['metric'].T,tol)
                error=[float(np.linalg.norm((c['nodes'][q[i]]-gt['nodes'][j])@gt['metric'].T)) for i,j in mapping.items()]
                localization.append(dict(**key,tolerance_mm=tol,matched=len(error),
                    localization_mean_mm=float(np.mean(error)) if error else None,localization_p95_mm=float(np.quantile(error,.95)) if error else None))
        table(out/'native-per-patch.csv',safe(rows));save(out/'native-per-patch.json',safe(rows));save(out/'detection-states.json',safe(states))
        table(out/'localization.csv',localization)
        native=[]
        for split in ['train','val']:
            selected=[i for i,r in enumerate(records) if r['split']==split]
            for subject in [None]+sorted({records[i]['subject'] for i in selected}):
                ids=[i for i in selected if subject is None or records[i]['subject']==subject]
                result=summarize_metrics([{k:v for k,v in rows[i].items() if k not in ['split','sample_id','subject','site']} for i in ids],folds=protocol.get('folds',5),
                    node_detection_states=[states[i]['node'] for i in ids],edge_detection_states=[states[i]['edge'] for i in ids],
                    iou_thresholds=protocol.get('iou_thresholds'))
                native.append(safe(dict(split=split,subject=subject,**result)))
        save(out/'native-summary.json',native)
        if not summary['final']:continue
        if head is None:head=spline_head(root)
        # Both thresholds are selected exclusively on training predictions, for
        # each physical tolerance. Validation PR is descriptive, not calibration.
        thresholds={};pr_rows=[]
        for tol in [1,2]:
            pooled={s:[[],[],0] for s in ['train','val']}
            for record in records:
                with np.load(record['cache']) as c,np.load(record['gt']) as gt:
                    q=c['retained'];mapping={int(q[i]):j for i,j in match_nodes(c['nodes'][q]@gt['metric'].T,gt['nodes']@gt['metric'].T,tol).items()}
                    pairs=c['edges'];active=np.isin(pairs,q).all(1);links={tuple(sorted(e)) for e in gt['edges']}
                    labels=np.array([a in mapping and b in mapping and tuple(sorted((mapping[a],mapping[b]))) in links for a,b in pairs])
                    target=pooled[record['split']];target[0].append(c['edge_scores'][active]);target[1].append(labels[active]);target[2]+=len(gt['edges'])
            train_scores=np.concatenate(pooled['train'][0]);train_y=np.concatenate(pooled['train'][1]);selected=train_scores>.5
            precision=float(train_y[selected].mean()) if selected.any() else 0.
            thresholds[str(tol)]=dict(fixed_0p5=.5,**select_threshold(train_scores,train_y,pooled['train'][2],precision))
            for split in ['train','val']:
                scores=np.concatenate(pooled[split][0]);labels=np.concatenate(pooled[split][1])
                if labels.any():
                    pp,rr,tt=precision_recall_curve(labels,scores);np.savez_compressed(out/f'{split}-relation-PR-{tol}.npz',precision=pp,recall=rr,threshold=tt)
                pr_rows.append(dict(split=split,tolerance_mm=tol,candidates=len(scores),positives=int(labels.sum()),
                    positive_prevalence=float(labels.mean()) if len(labels) else None))
        save(out/'thresholds.json',thresholds);table(out/'relation-populations.csv',pr_rows)
        calibrated=[];lengths=[]
        for record in records:
            with np.load(record['cache']) as f:c={k:f[k] for k in f.files}
            with np.load(record['gt']) as f:gt={k:f[k] for k in f.files}
            feature=torch.tensor(np.load(record['features']));geometry=Geometry(c,head,feature)
            mask=np.asarray(nib.load(Path(record['gt']).parent.parent/'seg'/(record['sample_id']+'_seg.nii.gz')).dataobj)>.5
            active=np.isin(c['edges'],c['retained']).all(1)
            for tol in [1,2]:
                threshold=thresholds[str(tol)]['train_max_f1'];keep=active&(c['edge_scores']>threshold)
                g=geometry.graph(c['retained'],keep);row,ll=measure(dict(c,edge_selected=keep),g,Path(record['gt']),gt,mask,tol)
                key={k:record[k] for k in ['split','sample_id','subject','site']}
                key.update(tolerance_mm=tol,policy='train_max_f1',threshold=threshold)
                calibrated.append(dict(**key,**row));lengths.extend(dict(**key,**r) for r in ll)
        cal=out/'calibration';cal.mkdir()
        summarize_run(cal,calibrated,lengths,dict(arm=arm,step=summary['step'],calibration='training only'))
        print(arm,folder.name,'native and calibrated evaluation complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);evaluate(a.root,a.arm,a.smoke)
