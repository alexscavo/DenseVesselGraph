"""Annotation-relative BRAVA metrics. Brain masks are never vessel labels."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.interpolate import BSpline
from sklearn.metrics import average_precision_score
from scripts.ixi_cached_mechanisms import save,table
from scripts.ixi_query_diagnostics import edge_accounting,summarize
from scripts.diagnose_branch_relations import match_nodes
from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels
from metrics.representation_geometry import point_to_polyline_distance,sample_polyline_by_arclength


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,(np.integer,np.floating)):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def aggregate(rows):
    result=summarize(rows)
    result.update(aggregate_common_vessels([{k:v for k,v in r.items() if k.startswith('common_')} for r in rows]))
    matched=sum(r['node_tp'] for r in rows)
    result.update(duplicates_per_recovered_gt=sum(r['duplicates'] for r in rows)/matched if matched else None,
        duplicates_per_patch=float(np.mean([r['duplicates'] for r in rows])),
        high_confidence_duplicates=sum(r['high_confidence_duplicates'] for r in rows),
        node_count_bias=float(np.mean([r['nodes']-r['gt_nodes'] for r in rows])),
        node_count_mae=float(np.mean([abs(r['nodes']-r['gt_nodes']) for r in rows])),
        localization_mean_mm=sum(r['localization_error_sum_mm'] for r in rows)/matched if matched else None,
        raw_query_recall=sum(r['raw_matched'] for r in rows)/sum(r['gt_nodes'] for r in rows),
        mean_nodes=float(np.mean([r['nodes'] for r in rows])),mean_edges=float(np.mean([r['edges'] for r in rows])))
    return result


def physical_curves(c,metric):
    controls=c['spline_controls'];k=controls.shape[1]
    knots=np.r_[np.zeros(4),np.arange(1,k-3)/(k-3),np.ones(4)]
    return [BSpline(knots,control,3)(np.linspace(0,1,257))@metric.T for control in controls]


def curve_error(curve,guide):
    a=point_to_polyline_distance(sample_polyline_by_arclength(curve,.25),guide)
    b=point_to_polyline_distance(sample_polyline_by_arclength(guide,.25),curve)
    return float((a.mean()+b.mean())/2)


def patch_metrics(c,gt,path):
    metric=gt['metric'];nodes=c['nodes'];pairs=c['candidate_edges']
    chosen_set={tuple(sorted(map(int,e))) for e in c['edges']}
    chosen=np.array([tuple(map(int,e)) in chosen_set for e in pairs],bool)
    chord=dict(nodes=nodes,edges=c['edges'])
    curved=dict(chord,spline_controls=c['spline_controls'],spline_family='bspline')
    commons={name:evaluate_common_vessels(g,path) for name,g in [('chord',chord),('spline',curved)]}
    # Removing curve controls must preserve all graph counts/branch connectivity.
    for key in ['common_branch_tp','common_branch_fp','common_branch_fn','common_betti0_abs_error','common_betti1_abs_error']:
        assert commons['chord'][key]==commons['spline'][key],key
    curves=physical_curves(c,metric);lengths=[];rows=[]
    references={tuple(sorted(map(int,e))):i for i,e in enumerate(gt['edges'])}
    for tol in [1,2]:
        mapping=match_nodes(nodes@metric.T,gt['nodes']@metric.T,tol)
        row,_,labels,types,_=edge_accounting(np.arange(len(nodes)),pairs,c['candidate_scores'],chosen,mapping,gt['edges'],len(gt['nodes']))
        recovered=set(mapping.values())
        distance=np.linalg.norm((nodes[:,None]-gt['nodes'][None])@metric.T,axis=-1)
        duplicates=[i for i in range(len(nodes)) if i not in mapping and any(distance[i,j]<=tol for j in recovered)]
        errors=[distance[i,j] for i,j in mapping.items()]
        raw=match_nodes(c['raw_nodes']@metric.T,gt['nodes']@metric.T,tol)
        conditional=np.array([a in mapping and b in mapping for a,b in pairs],bool)
        for population,eligible in [('retained',np.ones(len(pairs),bool)),('conditional',conditional)]:
            row[population+'_relation_ap']=float(average_precision_score(labels[eligible],c['candidate_scores'][eligible])) if labels[eligible].any() else None
            row[population+'_candidates']=int(eligible.sum());row[population+'_positive_pairs']=int(labels[eligible].sum())
        row.update(duplicates=len(duplicates),high_confidence_duplicates=int((c['node_scores'][duplicates]>=.9).sum()),
            localization_error_sum_mm=float(sum(errors)),localization_p95_mm=float(np.quantile(errors,.95)) if errors else None,raw_matched=len(raw))
        for mode in ['chord','spline']:rows.append(dict(row,tolerance_mm=tol,geometry=mode,**commons[mode]))
        for i,(a,b) in enumerate(c['edges']):
            if int(a) not in mapping or int(b) not in mapping:continue
            key=tuple(sorted((mapping[int(a)],mapping[int(b)])))
            if key not in references:continue
            edge=references[key];lo,hi=gt['path_offsets'][edge:edge+2];guide=gt['full_paths'][lo:hi]@metric.T
            lengths.append(dict(tolerance_mm=tol,gt_edge=int(edge),predicted_edge=i,
                chord_error_mm=curve_error(nodes[[a,b]]@metric.T,guide),spline_error_mm=curve_error(curves[i],guide),
                gt_arc_mm=float(np.linalg.norm(np.diff(guide,axis=0),axis=1).sum())))
    return rows,lengths


def draw_example(path,c,gt,sample):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metric=gt['metric'];nodes=c['nodes']@metric.T;fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d')
    for lo,hi in zip(gt['path_offsets'][:-1],gt['path_offsets'][1:]):ax.plot(*(gt['full_paths'][lo:hi]@metric.T).T,color='black',alpha=.5,linewidth=1)
    mapping=match_nodes(nodes,gt['nodes']@metric.T,2.);truth={tuple(sorted(map(int,e))) for e in gt['edges']}
    for edge,curve in zip(c['edges'],physical_curves(c,metric)):
        good=all(int(i) in mapping for i in edge) and tuple(sorted(mapping[int(i)] for i in edge)) in truth
        ax.plot(*curve.T,color='green' if good else 'red',alpha=.7)
        ax.plot(*nodes[edge].T,'--',color='gray',alpha=.4,linewidth=.6)
    if len(nodes):ax.scatter(*nodes.T,color='blue',s=6)
    ax.set(title=sample+'\nBlack: annotated paths; green/red: reference-matched/unmatched curves; dashed: same-graph chords',xlabel='x mm',ylabel='y mm',zlabel='z mm')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def evaluate(root,arm,smoke=False):
    from training.evaluation.metrics import evaluate_graph,summarize_metrics
    source=root/('smoke-predictions' if smoke else 'predictions')/arm
    for folder in sorted(source.iterdir()):
        assert json.loads((folder/'summary.json').read_text())['completed']
        out=root/('smoke-evaluation' if smoke else 'evaluation')/arm/folder.name;out.mkdir(parents=True,exist_ok=False)
        records=json.loads((folder/'records.json').read_text());cfg=json.loads((folder/'config.json').read_text())
        rows=[];geometry=[];native=[];ns=[];es=[];visual_done=set()
        visual_subjects=list(dict.fromkeys(r['subject'] for r in records if r['split']=='val'))[:3]
        for number,r in enumerate(records,1):
            with np.load(r['cache']) as f:c={k:f[k] for k in f.files}
            with np.load(r['gt']) as f:gt={k:f[k] for k in f.files}
            rr,ll=patch_metrics(c,gt,Path(r['gt']));key={k:r[k] for k in ['split','subject','sample_id']}
            rows.extend(dict(key,**x) for x in rr);geometry.extend(dict(key,**x) for x in ll)
            transform=np.eye(4);transform[:3,:3]=gt['metric']
            result,nstate,estate=evaluate_graph(c,gt['nodes'],gt['edges'],protocol=cfg['evaluation']['protocol'],world_transform=transform,return_detection_states=True)
            native.append(dict(key,**result));ns.append(nstate);es.append(estate)
            if r['split']=='val' and r['subject'] in visual_subjects and r['subject'] not in visual_done:
                draw_example(out/(r['sample_id']+'-geometry.png'),c,gt,r['sample_id']);visual_done.add(r['subject'])
            if number%128==0:print(arm,folder.name,'CPU',number,'/',len(records),flush=True)
        save(out/'per-patch.json',clean(rows));table(out/'per-patch.csv',clean(rows));table(out/'same-connection-geometry.csv',geometry)
        comparison=[];subject_rows=[];native_summary=[];conditional_geometry=[]
        for split in sorted({r['split'] for r in rows}):
            for tol in [1,2]:
                for mode in ['chord','spline']:
                    rr=[r for r in rows if r['split']==split and r['tolerance_mm']==tol and r['geometry']==mode]
                    comparison.append(dict(split=split,tolerance_mm=tol,geometry=mode,**aggregate(rr)))
                    for subject in sorted({r['subject'] for r in rr}):
                        subject_rows.append(dict(split=split,subject=subject,tolerance_mm=tol,geometry=mode,**aggregate([r for r in rr if r['subject']==subject])))
            for subject in [None]+sorted({r['subject'] for r in records if r['split']==split}):
                selected=[i for i,r in enumerate(records) if r['split']==split and (subject is None or r['subject']==subject)]
                n=summarize_metrics([{k:v for k,v in native[i].items() if k not in ['split','subject','sample_id']} for i in selected],
                    folds=cfg['evaluation']['protocol'].get('folds',5),node_detection_states=[ns[i] for i in selected],
                    edge_detection_states=[es[i] for i in selected],iou_thresholds=cfg['evaluation']['protocol'].get('iou_thresholds'))
                native_summary.append(dict(split=split,subject=subject,**n))
            for tol in [1,2]:
                for subject in [None]+sorted({r['subject'] for r in records if r['split']==split}):
                    rr=[r for r in geometry if r['split']==split and r['tolerance_mm']==tol and (subject is None or r['subject']==subject)]
                    conditional_geometry.append(dict(split=split,tolerance_mm=tol,subject=subject,connections=len(rr),
                        **{k:float(np.mean([r[k] for r in rr])) if rr else None for k in ['chord_error_mm','spline_error_mm']}))
        table(out/'comparison.csv',clean(comparison));table(out/'per-subject.csv',clean(subject_rows));save(out/'native-summary.json',clean(native_summary))
        save(out/'summary.json',clean(dict(completed=True,arm=arm,checkpoint=folder.name,comparison=comparison,
            native=native_summary,annotation_relative=True,brain_mask_used_for_vessel_metrics=False,
            recovered_connections=len(geometry),same_connection_geometry=conditional_geometry,same_graph_verified=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);evaluate(a.root,a.arm,a.smoke)
