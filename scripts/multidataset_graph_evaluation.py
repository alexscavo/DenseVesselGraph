"""Prediction caches, fixed-population monitors and CPU evaluation for staged study."""
import argparse,copy,csv,gzip,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.interpolate import BSpline
from sklearn.metrics import average_precision_score,confusion_matrix,roc_auc_score
from scripts.analyze_spline_target_capacity import write
from scripts.multidataset_graph_training import setup,edge_inputs,SEED
from models.adaptive_geometry import AdaptiveSplineBank,uniform_knots,CANDIDATES
from models.spline import canonical_reverse
from scripts.diagnose_branch_relations import match_nodes
from scripts.ixi_cached_mechanisms import edge_accounting
from scripts.ixi_query_diagnostics import summarize
from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels
from metrics.representation_geometry import point_to_polyline_distance,sample_polyline_by_arclength


def loaders(cfg,plan,arm,monitor=False):
    from data.loaders import build_data_loaders,build_evaluation_loader
    from scripts.frozen_spline_diagnostic import training_evaluation_loader
    cfg=copy.deepcopy(cfg);cfg['data'].update(train_augmentation=False,batch_size=1,validation_batch_size=1)
    spec=plan['datasets'][arm['dataset']]
    for split in ['train','val']:
        names=spec['monitor_train'] if split=='train' else spec['monitor_val'] if monitor else spec['val_ids']
        if split=='train':full,_=build_data_loaders(cfg);loader=training_evaluation_loader(full,cfg,names)
        else:loader=build_evaluation_loader(cfg,dataset_name='synthetic_mri',sample_ids=names)
        yield split,names,loader


def accounting(c,gt,tolerance=2.,threshold=.5):
    metric=gt['metric'];mapping=match_nodes(c['nodes']@metric.T,gt['nodes']@metric.T,tolerance)
    chosen=c['candidate_scores']>threshold
    row,_,labels,_,_=edge_accounting(np.arange(len(c['nodes'])),c['candidate_edges'],c['candidate_scores'],chosen,mapping,gt['edges'],len(gt['nodes']))
    distance=np.linalg.norm((c['nodes'][:,None]-gt['nodes'][None])@metric.T,axis=-1);recovered=set(mapping.values())
    duplicates=[i for i in range(len(c['nodes'])) if i not in mapping and any(distance[i,j]<=tolerance for j in recovered)]
    conditional=np.array([a in mapping and b in mapping for a,b in c['candidate_edges']],bool)
    row.update(duplicates=len(duplicates),high_confidence_duplicates=sum(c['node_scores'][duplicates]>=.9),
        raw_matched=len(match_nodes(c['raw_nodes']@metric.T,gt['nodes']@metric.T,tolerance)),
        retained_relation_ap=float(average_precision_score(labels,c['candidate_scores'])) if labels.any() else None,
        conditional_relation_ap=float(average_precision_score(labels[conditional],c['candidate_scores'][conditional])) if labels[conditional].any() else None,
        localized_sum_mm=float(sum(distance[i,j] for i,j in mapping.items())))
    if mapping and duplicates:
        correct=list(mapping);prob=np.r_[c['node_scores'][correct],c['node_scores'][duplicates]]
        row['correct_vs_duplicate_confidence_auc']=float(roc_auc_score(np.r_[np.ones(len(correct)),np.zeros(len(duplicates))],prob))
    else:row['correct_vs_duplicate_confidence_auc']=None
    row['mean_correct_confidence']=float(np.mean(c['node_scores'][list(mapping)])) if mapping else None
    row['mean_duplicate_confidence']=float(np.mean(c['node_scores'][duplicates])) if duplicates else None
    return row,labels


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


@torch.no_grad()
def monitor_model(root,index,bundle,step):
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    plan,arm,model,cfg,criterion,_,_,_,_=bundle;model.eval();rows=[]
    for split,names,loader in loaders(cfg,plan,arm,True):
        for n,batch in enumerate(loader):
            volumes,targets=_move_graph_batch(batch,'cuda','image',True);tokens,pred,_=model(volumes)
            graph=infer_graphs(tokens,pred,model.relation_embed,object_queries=model.obj_token,relation_tokens=criterion.relation_tokens,node_threshold=cfg['evaluation'].get('node_threshold'),edge_threshold=-1.)[0]
            c={k:v.numpy() for k,v in graph.items() if torch.is_tensor(v)};c['candidate_edges']=c['edges'];c['candidate_scores']=c['edge_scores'];c['raw_nodes']=pred['pred_nodes'][0,:,:3].cpu().numpy()
            gt=dict(nodes=targets['nodes'][0].cpu().numpy(),edges=targets['edges'][0].cpu().numpy(),metric=targets['spline'][0]['metric'].cpu().numpy())
            source,target=criterion.matcher(pred,targets)[0];d=np.linalg.norm((c['raw_nodes'][source]-gt['nodes'][target])@gt['metric'].T,axis=-1)
            for tol in [1.,2.]:
                r,_=accounting(c,gt,tol);rows.append(dict(split=split,sample_id=names[n],subject=plan['datasets'][arm['dataset']]['subjects'][names[n].split('_')[1]],tolerance_mm=tol,hungarian_mean_mm=float(d.mean()) if len(d) else None,hungarian_fraction_localized=float((d<=tol).mean()) if len(d) else None,**r))
    summary=[]
    for split in ['train','val']:
        for tol in [1.,2.]:
            r=[v for v in rows if v['split']==split and v['tolerance_mm']==tol]
            z=summarize(r);z.update(split=split,tolerance_mm=tol,raw_query_recall=sum(v['raw_matched'] for v in r)/sum(v['gt_nodes'] for v in r),mean_hungarian_distance_mm=float(np.mean([v['hungarian_mean_mm'] for v in r if v['hungarian_mean_mm'] is not None])))
            summary.append(z)
    out=root/'runs'/arm['name']/'monitors';out.mkdir(parents=True,exist_ok=True);write(out/f'{step:06d}.json',clean(dict(step=step,summary=summary,rows=rows)));return summary


def curve_outputs(head,features,endpoints,metric,local):
    """GT-independent neural selection; evaluates the chosen native basis."""
    if isinstance(head,AdaptiveSplineBank):choices=head.choose(features,endpoints,metric)
    else:choices=torch.zeros(len(endpoints),dtype=torch.long,device=endpoints.device)
    curves=np.empty((len(endpoints),257,3),dtype=np.float32);counts=np.empty(len(endpoints),np.int64)
    for label in choices.unique().tolist():
        selected=(choices==label).nonzero().flatten();h=head.heads[str(CANDIDATES[label])] if isinstance(head,AdaptiveSplineBank) else head
        control,_=h(features[selected],endpoints[selected],metric,**local)
        knots=h.knots.cpu().numpy() if hasattr(h,'knots') else uniform_knots(h.controls)
        basis=BSpline(knots,np.eye(h.controls),3)(np.linspace(0,1,257))
        curves[selected.cpu().numpy()]=np.einsum('sk,nkc->nsc',basis,control.cpu().numpy())
        counts[selected.cpu().numpy()]=h.controls
    return curves,counts


@torch.no_grad()
def snapshot(root,index,bundle,step,out,monitor=False):
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    plan,arm,model,cfg,criterion,_,_,_,_=bundle;model.eval();out.mkdir(parents=True,exist_ok=False);records=[]
    for split,names,loader in loaders(cfg,plan,arm,monitor):
        for n,batch in enumerate(loader):
            volumes,targets=_move_graph_batch(batch,'cuda','image',True);tokens,pred,_=model(volumes);metric=targets['spline'][0]['metric'].to(tokens)
            kw=dict(object_queries=model.obj_token,relation_tokens=criterion.relation_tokens,node_threshold=cfg['evaluation'].get('node_threshold'))
            allretained=infer_graphs(tokens,pred,model.relation_embed,edge_threshold=-1.,**kw)[0]
            c={k:v.numpy() for k,v in allretained.items() if torch.is_tensor(v)};pairs=allretained['edges'].to(tokens.device);query=allretained['query_ids'].to(tokens.device)
            curves=[];counts=[]
            for first in range(0,len(pairs),256):
                pp=pairs[first:first+256].clone();rev=canonical_reverse(pred['pred_nodes'][0,query[pp],:3]@metric.T);pp[rev]=pp[rev].flip(1)
                qp=query[pp];relation=tokens[0,criterion.object_queries:criterion.object_queries+criterion.relation_tokens];features=criterion._relation_features(tokens[0,:criterion.object_queries],relation,qp)
                local=dict(local_features=pred['spline_local_features'][0],input_shape=pred['spline_input_shape'])
                x,k=curve_outputs(model.spline_head,features,pred['pred_nodes'][0,qp,:3],metric,local);x[rev.cpu().numpy()]=x[rev.cpu().numpy(),::-1];curves.extend(x);counts.extend(k)
            c.update(candidate_edges=c.pop('edges'),candidate_scores=c.pop('edge_scores'),candidate_curves=np.array(curves,dtype=np.float32).reshape(-1,257,3),control_counts=np.array(counts),raw_nodes=pred['pred_nodes'][0,:,:3].cpu().numpy(),raw_scores=pred['pred_logits'][0].softmax(-1)[:,1].cpu().numpy())
            selected=c['candidate_scores']>.5;c['edges']=c['candidate_edges'][selected];c['edge_scores']=c['candidate_scores'][selected]
            source,target=criterion.matcher(pred,targets)[0];c['hungarian_source']=source.numpy();c['hungarian_target']=target.numpy()
            data=edge_inputs(model,criterion,tokens,pred,targets,0,(source,target))
            if data is not None:
                chosen,reverse,features,endpoints,metric,local=data;x,k=curve_outputs(model.spline_head,features,endpoints,metric,local)
                c.update(matched_curves=x,matched_counts=k,matched_edges=np.array(chosen),matched_reverse=reverse.cpu().numpy())
                if isinstance(model.spline_head,AdaptiveSplineBank):
                    options=[]
                    for kk in CANDIDATES:
                        h=model.spline_head.heads[str(kk)];cc,_=h(features,endpoints,metric,**local);basis=BSpline(uniform_knots(kk),np.eye(kk),3)(np.linspace(0,1,257));options.append(np.einsum('sk,nkc->nsc',basis,cc.cpu().numpy()))
                    c['matched_bank_curves']=np.stack(options,axis=1)
            else:c.update(matched_curves=np.empty((0,257,3)),matched_counts=np.array([],int),matched_edges=np.array([],int),matched_reverse=np.array([],bool))
            # Raw-query relation populations for the recovery diagnosis, including rejected queries.
            if arm['kind']=='detector':
                raw=infer_graphs(tokens,pred,model.relation_embed,edge_threshold=-1.,**dict(kw,node_threshold=-1.))[0]
                c.update(raw_pairs=raw['edges'].numpy(),raw_relation_scores=raw['edge_scores'].numpy())
            sample=names[n];cache=out/(sample+'.npz');np.savez_compressed(cache,**c)
            spec=plan['datasets'][arm['dataset']];records.append(dict(sample_id=sample,subject=spec['subjects'][sample.split('_')[1]],split=split,cache=str(cache),gt=str(Path(spec['root'])/split/'spline'/(sample+'.npz'))))
            if n%64==0:print(arm['name'],step,split,n,len(names),flush=True)
    write(out/'records.json',records);write(out/'complete.json',dict(completed=True,batching=1,step=step,records=len(records)))


def predict(root,index):
    from models.checkpoint import load_checkpoint_file
    bundle=setup(root,index);plan,arm,model,cfg,criterion,_,_,_,_=bundle
    assert json.loads((root/'runs'/arm['name']/'complete.json').read_text())['completed']
    if arm['kind']=='detector':
        monitors=[]
        for file in (root/'runs'/arm['name']/'monitors').glob('*.json'):
            item=json.loads(file.read_text())
            if item['step']>0:
                row=next(x for x in item['summary'] if x['split']=='val' and x['tolerance_mm']==2.)
                monitors.append((row['f1'] or 0,item['step']))
        best=max(monitors)[1];steps=sorted({best,plan['detector_updates']})
        write(root/'runs'/arm['name']/'checkpoint-selection.json',dict(validation_monitor_only=True,monitor_validation_patches=len(plan['datasets'][arm['dataset']]['monitor_val']),selected_step=best,primary_final_step=plan['detector_updates']))
    else:steps=[plan['geometry_updates']]
    for step in steps:
        cp=load_checkpoint_file(root/'runs'/arm['name']/f'model-{step:06d}.pt');model.load_state_dict(cp['net'],strict=True);del cp
        snapshot(root,index,bundle,step,root/'predictions'/arm['name']/f'{step:06d}')


def distance(curve,path):
    a=point_to_polyline_distance(sample_polyline_by_arclength(curve,.1),path);b=point_to_polyline_distance(sample_polyline_by_arclength(path,.1),curve)
    return float((a.mean()+b.mean())/2),float(max(a.max(),b.max()))


def accounting_arrays(record):
    with np.load(record['cache']) as f:c={k:f[k] for k in ['nodes','node_scores','candidate_edges','candidate_scores','raw_nodes']}
    with np.load(record['gt']) as f:gt={k:f[k] for k in ['nodes','edges','metric']}
    return c,gt


def evaluate(root,index):
    from scripts.continuous_representation import fit_options,choose
    from training.evaluation.metrics import evaluate_graph,summarize_metrics
    plan=json.loads((root/'plan.json').read_text());arm=plan['arms'][index];out=root/'evaluation'/arm['name'];out.mkdir(parents=True,exist_ok=False)
    for folder in sorted((root/'predictions'/arm['name']).iterdir()):
        records=json.loads((folder/'records.json').read_text());thresholds=[.05+i*.025 for i in range(37)];sums={t:np.zeros(3) for t in thresholds};prepared=[];geometry=[];selector_truth=[];selector_pred=[];population=[]
        for record in records:
            c,gt=accounting_arrays(record);row,labels=accounting(c,gt,2.)
            if record['split']=='train':
                for t in thresholds:
                    chosen=c['candidate_scores']>t;tp=int((chosen&labels.astype(bool)).sum());fp=int((chosen&~labels.astype(bool)).sum());fn=len(gt['edges'])-tp;sums[t]+=np.array([tp,fp,fn])
            prepared.append(record)
        threshold=max(thresholds,key=lambda t:(2*sums[t][0]/max(1,2*sums[t][0]+sums[t][1]+sums[t][2]),t))
        rows=[];native=[];node_states=[];edge_states=[]
        for record in prepared:
            with np.load(record['cache']) as f:c={k:f[k] for k in f.files}
            with np.load(record['gt']) as f:gt={k:f[k] for k in f.files}
            meta={key:record[key] for key in ['split','subject','sample_id']};metric=gt['metric']
            for tol in [1.,2.]:
                for policy,t in [('default',.5),('train_max_f1',threshold)]:
                    row,_=accounting(c,gt,tol,t);selected=c['candidate_scores']>t
                    g=dict(nodes=c['nodes'],edges=c['candidate_edges'][selected],edge_polylines=c['candidate_curves'][selected])
                    common=evaluate_common_vessels(g,record['gt']);chord=evaluate_common_vessels(dict(nodes=g['nodes'],edges=g['edges']),record['gt'])
                    extra={}
                    if arm['representation'].startswith('piece') and tol==2. and policy=='default':
                        from models.adaptive_geometry import simplify_piecewise_curve
                        simple=[];segments=[]
                        for curve in c['candidate_curves'][selected]:
                            converted,number,_=simplify_piecewise_curve(curve@metric.T,arm['representation'].split('_')[1]);simple.append(converted@np.linalg.inv(metric).T);segments.append(number)
                        sm=evaluate_common_vessels(dict(nodes=g['nodes'],edges=g['edges'],edge_polylines=simple),record['gt'])
                        extra=dict(simplified_curve_f1=aggregate_common_vessels([sm])['common_curve_f1'],simplified_mean_segments=float(np.mean(segments)) if segments else 0.)
                    rows.append(dict(meta,tolerance_mm=tol,policy=policy,threshold=t,controls=int(c['control_counts'][selected].sum()),chord_curve_f1=aggregate_common_vessels([chord])['common_curve_f1'],**extra,**row,**common))
            transform=np.eye(4);transform[:3,:3]=gt['metric']
            # Native detection/SMD diagnostic preserves its established convention.
            cfg=json.loads((root/'runs'/arm['name']/'config.json').read_text())
            result,nstate,estate=evaluate_graph(c,gt['nodes'],gt['edges'],protocol=cfg['evaluation']['protocol'],world_transform=transform,return_detection_states=True)
            native.append(dict(meta,**result));node_states.append(nstate);edge_states.append(estate)
            for number,(curve,edge,reverse) in enumerate(zip(c['matched_curves'],c['matched_edges'],c['matched_reverse'])):
                a,b=gt['path_offsets'][edge:edge+2];path=gt['full_paths'][a:b].astype(float)@metric.T
                if reverse:path=path[::-1]
                physical=curve.astype(float)@metric.T;t=np.linspace(0,1,len(curve))[:,None];neutral=physical-((1-t)*physical[0]+t*physical[-1])+((1-t)*path[0]+t*path[-1])
                mean,maximum=distance(physical,path);nm,_=distance(neutral,path);endpoint=float(np.linalg.norm(physical[[0,-1]]-path[[0,-1]],axis=1).max())
                length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum());chord=float(np.linalg.norm(path[-1]-path[0]))
                original_fit=BSpline(uniform_knots(8),gt['controls'][edge].astype(float)@metric.T,3)(np.linspace(0,1,257));fit_error=distance(original_fit,path)
                tangents=np.diff(path,axis=0);norms=np.linalg.norm(tangents,axis=1);tangents=tangents[norms>1e-10]/norms[norms>1e-10,None]
                turn=float(np.arccos(np.clip((tangents[:-1]*tangents[1:]).sum(1),-1,1)).sum()) if len(tangents)>1 else 0.
                geometry.append(dict(meta,edge=int(edge),mean_mm=mean,max_mm=maximum,neutral_mean_mm=nm,endpoint_max_mm=endpoint,controls=int(c['matched_counts'][number]),length_mm=length,chord_mm=chord,tortuosity=length/max(chord,1e-10),turn_radians=turn,K8_fit_mean_mm=fit_error[0],K8_fit_sampled_max_mm=fit_error[1]))
                if arm['representation']=='adaptive':
                    k,ok,_=choose(fit_options(path),path,.5);
                    oracle_curve=c['matched_bank_curves'][number,CANDIDATES.index(k)].astype(float)@metric.T
                    oracle_mean,oracle_max=distance(oracle_curve,path)
                    geometry[-1].update(oracle_selected_K=k,oracle_tolerance_passed=ok,oracle_head_mean_mm=oracle_mean,oracle_head_max_mm=oracle_max)
                    if record['split']=='val':selector_truth.append(k);selector_pred.append(int(c['matched_counts'][number]))
            if arm['kind']=='detector':
                hs=set(c['hungarian_source'].tolist());sp=match_nodes(c['raw_nodes']@metric.T,gt['nodes']@metric.T,2.);gtset={tuple(sorted(map(int,e))) for e in gt['edges']}
                for kind in ['hungarian_matched_matched','hungarian_unmatched_involving','spatial_matched_matched','spatial_unmatched_involving']:
                    pairs=c['raw_pairs'];scores=c['raw_relation_scores'];subset=np.array([(a in hs and b in hs) if kind.startswith('hungarian') else (int(a) in sp and int(b) in sp) for a,b in pairs])
                    if 'unmatched_involving' in kind:subset=~subset
                    truth=np.array([int(a) in sp and int(b) in sp and tuple(sorted((sp[int(a)],sp[int(b)]))) in gtset for a,b in pairs]);predicted=scores>.5
                    population.append(dict(meta,population=kind,candidates=int(subset.sum()),positive_pairs=int((subset&truth).sum()),predicted_edges=int((subset&predicted).sum()),false_edges=int((subset&predicted&~truth).sum())))
        import pandas as pd
        target=out/folder.name;target.mkdir()
        df=pd.DataFrame(rows);df.to_csv(target/'patches.csv.gz',index=False);pd.DataFrame(geometry).to_csv(target/'geometry.csv.gz',index=False);pd.DataFrame(population).to_csv(target/'relation-populations.csv.gz',index=False)
        if selector_truth:write(target/'selector-confusion.json',dict(labels=list(CANDIDATES),matrix=confusion_matrix(selector_truth,selector_pred,labels=list(CANDIDATES)).tolist(),accuracy=float(np.mean(np.array(selector_truth)==selector_pred)),note='Validation subjects only; classifier trained separately on training subjects.'))
        summaries=[]
        for split in ['train','val']:
            for tol in [1.,2.]:
                for policy in ['default','train_max_f1']:
                    rr=[r for r in rows if r['split']==split and r['tolerance_mm']==tol and r['policy']==policy];base=summarize(rr);base.update(aggregate_common_vessels([{k:v for k,v in r.items() if k.startswith('common_')} for r in rr]));base.update(split=split,tolerance_mm=tol,policy=policy,threshold=.5 if policy=='default' else threshold,raw_query_recall=sum(r['raw_matched'] for r in rr)/sum(r['gt_nodes'] for r in rr));summaries.append(base)
        subject_rows=[]
        for split in ['train','val']:
            for subject in sorted({r['subject'] for r in rows if r['split']==split}):
                rr=[r for r in rows if r['split']==split and r['subject']==subject and r['policy']=='default' and r['tolerance_mm']==2.]
                z=summarize(rr);z.update(aggregate_common_vessels([{k:v for k,v in r.items() if k.startswith('common_')} for r in rr]));subject_rows.append(dict(split=split,subject=subject,**z))
        write(target/'per-subject.json',clean(subject_rows))
        for summary in summaries:
            ys=[];ss=[];conditional_y=[];conditional_s=[]
            for record in prepared:
                if record['split']!=summary['split']:continue
                c,gt=accounting_arrays(record)
                _,labels=accounting(c,gt,summary['tolerance_mm']);mapping=match_nodes(c['nodes']@gt['metric'].T,gt['nodes']@gt['metric'].T,summary['tolerance_mm'])
                eligible=np.array([int(a) in mapping and int(b) in mapping for a,b in c['candidate_edges']],bool)
                ys.extend(labels);ss.extend(c['candidate_scores']);conditional_y.extend(labels[eligible]);conditional_s.extend(c['candidate_scores'][eligible])
            summary['retained_relation_ap']=float(average_precision_score(ys,ss)) if any(ys) else None
            summary['conditional_relation_ap']=float(average_precision_score(conditional_y,conditional_s)) if any(conditional_y) else None
        write(target/'summary.json',clean(summaries))
        native_summaries=[]
        for split in ['train','val']:
            indices=[i for i,r in enumerate(native) if r['split']==split]
            result=summarize_metrics([{k:v for k,v in native[i].items() if k not in ['split','subject','sample_id']} for i in indices],node_detection_states=[node_states[i] for i in indices],edge_detection_states=[edge_states[i] for i in indices],iou_thresholds=cfg['evaluation']['protocol'].get('iou_thresholds'))
            native_summaries.append(dict(split=split,**result))
        write(target/'native.json',clean(native_summaries))
    write(out/'complete.json',dict(completed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['predict','evaluate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',type=int,required=True);a=p.parse_args();torch.set_num_threads(1)
    if a.mode=='predict':predict(a.root,a.arm)
    else:evaluate(a.root,a.arm)
