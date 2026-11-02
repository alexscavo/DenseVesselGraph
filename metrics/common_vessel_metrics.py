"""Representation-independent topology and shape against shared uncertain guides."""
from collections import defaultdict
import numpy as np
from scipy.interpolate import BSpline
from .branch_connectivity import contract_degree_two, match_anchors
from .representation_geometry import point_to_polyline_distance, sample_polyline_by_arclength, tortuosity, directed_curve_distances


def evaluate_common_vessels(prediction, cache_path):
    with np.load(cache_path) as target:
        metric=target['metric'].astype(float)
        gt_nodes=target['nodes'].astype(float)@metric.T
        gt_edges=target['edges'].astype(int).reshape(-1,2)
        offsets=target['path_offsets'];paths=target['full_paths'].astype(float)@metric.T
        gt_curves=[paths[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    nodes=np.asarray(prediction['nodes'],dtype=float).reshape(-1,3)
    edges=np.asarray(prediction['edges'],dtype=int).reshape(-1,2)
    if 'edge_polylines' in prediction:
        curves=[np.asarray(c,dtype=float)@metric.T for c in prediction['edge_polylines']]
        if len(curves)!=len(edges):raise ValueError('One polyline is required per graph edge')
        for c,e in zip(curves,edges):
            if len(c)<2 or not np.isfinite(c).all() or not np.allclose(c[[0,-1]],nodes[e]@metric.T,atol=1e-4):
                raise ValueError('Curve endpoints must agree with graph edge ordering')
    elif 'spline_controls' in prediction:
        curves=[]
        for controls in prediction['spline_controls']:
            controls=np.asarray(controls,dtype=float); k=len(controls)
            knots=np.r_[np.zeros(4),np.arange(1,k-3)/(k-3),np.ones(4)]
            if prediction.get('spline_family', 'bspline') == 'catmull_rom':
                from scripts.catmull_rom import catmull_rom
                curves.append(catmull_rom(controls @ metric.T, np.linspace(0,1,257)))
            else:
                curves.append(BSpline(knots,controls,3)(np.linspace(0,1,257))@metric.T)
    else:
        curves=[nodes[edge]@metric.T for edge in edges]
    predicted=contract_degree_two(nodes@metric.T,edges,edge_polylines=curves)
    reference=contract_degree_two(gt_nodes,gt_edges,edge_polylines=gt_curves)
    mapping=match_anchors(predicted,reference,1.0)
    anchor_mapping=match_anchors(predicted,reference,1.0,include_isolated=True)
    pred_curves=[b.polyline for b in predicted.branches]
    target_curves=[b.polyline for b in reference.branches]
    pred_sample_count=sum(len(sample_polyline_by_arclength(c,.25)) for c in pred_curves)
    target_sample_count=sum(len(sample_polyline_by_arclength(c,.25)) for c in target_curves)
    if pred_curves and target_curves:
        pred_dist=directed_curve_distances(pred_curves,target_curves,.25)
        gt_dist=directed_curve_distances(target_curves,pred_curves,.25)
        pred_supported=int((pred_dist<=1.).sum()); gt_supported=int((gt_dist<=1.).sum())
    else:
        pred_supported=gt_supported=0
    available=defaultdict(list)
    for branch in reference.branches:available[tuple(sorted((branch.start,branch.end)))].append(branch)
    errors=[];matched=0
    for branch in predicted.branches:
        if branch.start not in mapping or branch.end not in mapping:continue
        key=tuple(sorted((mapping[branch.start],mapping[branch.end])))
        if not available[key]:continue
        target=available[key].pop(0); matched+=1
        # Deterministic topology matching; never choose a branch by shape error.
        curve,guide=branch.polyline,target.polyline
        forward=point_to_polyline_distance(sample_polyline_by_arclength(curve,.25),guide)
        backward=point_to_polyline_distance(sample_polyline_by_arclength(guide,.25),curve)
        errors.append(((forward.mean()+backward.mean())/2,abs(tortuosity(curve)-tortuosity(guide))))
    degrees=np.bincount(edges.ravel(),minlength=len(nodes))
    roles=predicted.roles
    result={'common_anchor_tp':len(anchor_mapping),
            'common_anchor_fp':sum(role!='cycle' for role in roles.values())-len(anchor_mapping),
            'common_anchor_fn':len(reference.anchors)-len(anchor_mapping),
            'common_curve_pred_samples':pred_sample_count,'common_curve_gt_samples':target_sample_count,
            'common_curve_supported_pred_samples':pred_supported,'common_curve_supported_gt_samples':gt_supported,
            'common_branch_tp':matched,'common_branch_fp':len(predicted.branches)-matched,'common_branch_fn':len(reference.branches)-matched,
            'common_predicted_branches':len(predicted.branches),'common_target_branches':len(reference.branches),
            'common_predicted_anchors':sum(role!='cycle' for role in roles.values()),'common_target_anchors':len(reference.anchors),
            'common_predicted_junctions':sum(role=='junction' for role in roles.values()),'common_target_junctions':sum(role=='junction' for role in reference.roles.values()),
            'common_predicted_endpoints':sum(role=='endpoint' for role in roles.values()),'common_target_endpoints':sum(role=='endpoint' for role in reference.roles.values()),
            'common_predicted_isolates':sum(role=='isolated' for role in roles.values()),
            'native_predicted_nodes':len(nodes),'native_predicted_degree2':int((degrees==2).sum()),
            'representation_internal_controls':sum(len(controls)-2 for controls in prediction['spline_controls']) if len(edges) and 'spline_controls' in prediction else 0,
            'common_betti0_abs_error':abs(predicted.betti_0-reference.betti_0),'common_betti1_abs_error':abs(predicted.betti_1-reference.betti_1),
            'common_anchor_count_abs_error':abs(sum(role!='cycle' for role in roles.values())-len(reference.anchors))}
    for i,key in enumerate(('common_guide_distance_mm','common_tortuosity_error')):
        values=[row[i] for row in errors];result[key+'_sum']=float(sum(values));result[key]=float(np.mean(values)) if values else None
    return result


def aggregate_common_vessels(rows):
    tp,fp,fn=(sum(row[k] for row in rows) for k in ('common_branch_tp','common_branch_fp','common_branch_fn'))
    output={'protocol_version':'shared-guides-v2','anchor_threshold_mm':1.,'curve_threshold_mm':1.,'sample_spacing_mm':.25,'common_branch_tp_total':tp,'common_branch_fp_total':fp,'common_branch_fn_total':fn,
            'common_branch_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'common_branch_coverage':tp/(tp+fn) if tp+fn else None,'samples':len(rows)}
    atp,afp,afn=(sum(row[k] for row in rows) for k in ('common_anchor_tp','common_anchor_fp','common_anchor_fn'))
    output['common_anchor_f1']=2*atp/(2*atp+afp+afn) if 2*atp+afp+afn else 1.
    npred=sum(row['common_curve_pred_samples'] for row in rows); ngt=sum(row['common_curve_gt_samples'] for row in rows)
    precision=sum(row['common_curve_supported_pred_samples'] for row in rows)/npred if npred else 1.
    recall=sum(row['common_curve_supported_gt_samples'] for row in rows)/ngt if ngt else 1.
    output.update(common_curve_precision=precision,common_curve_recall=recall,common_curve_f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    for key in ('common_guide_distance_mm','common_tortuosity_error'):
        output[key]=sum(row[key+'_sum'] for row in rows)/tp if tp else None
    for key in rows[0]:
        if key not in ('sample_id','common_guide_distance_mm','common_tortuosity_error') and not key.endswith('_sum'):
            output[key+'_mean']=float(np.mean([row[key] for row in rows]))
    return output
