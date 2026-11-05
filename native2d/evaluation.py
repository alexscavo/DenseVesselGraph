"""2D detection and physical graph metrics with explicit units.

The common-v2 evaluator is reused by an isometric zero-plane embedding. This
changes no Euclidean distances, connectivity or curve sampling. Detection boxes
have identical unit depth, so their 3D IoU equals ordinary 2D area IoU exactly.
The network, targets, losses and sampling remain natively two-dimensional.
"""
from io import BytesIO
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score
from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels
from training.evaluation.metrics import detection_ap_ar,street_mover_distance,graph_betti_numbers


def lift(x):
    a=np.asarray(x,float)
    return np.concatenate((a,np.zeros((*a.shape[:-1],1))),-1)


def match_nodes(pred,gt,tolerance):
    pred=np.asarray(pred).reshape(-1,2);gt=np.asarray(gt).reshape(-1,2)
    if not len(pred) or not len(gt):return {},np.empty(0)
    d=cdist(pred,gt);penalty=(min(len(pred),len(gt))+1)*(tolerance+1)
    rows,cols=linear_sum_assignment(np.where(d<=tolerance,d,penalty))
    eligible=d[rows,cols]<=tolerance
    return dict(zip(rows[eligible].tolist(),cols[eligible].tolist())),d[rows[eligible],cols[eligible]]


def counts(tp,npred,ngt):
    fp=npred-tp;fn=ngt-tp
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=tp/npred if npred else float(not ngt),
        recall=tp/ngt if ngt else float(not npred),f1=2*tp/(npred+ngt) if npred+ngt else 1.)


def positional(pred_nodes,pred_edges,gt_nodes,gt_edges,tolerance,loop_aware=True):
    """Node/edge counts at a tolerance. loop_aware (default since 5 Nov 2026): degree-2 GT nodes lying on a cycle are the
    inserted breaking points of loop branches (native2d/targets._simple_graph) and are scored as in metrics/loop_aware.py;
    a target without loops gives exactly the plain counts."""
    mapping,distance=match_nodes(pred_nodes,gt_nodes,tolerance)
    if loop_aware:
        from metrics.loop_aware import inserted_from_degree,loop_aware_counts
        inserted=inserted_from_degree(len(np.asarray(gt_nodes).reshape(-1,2)),gt_edges)
        if inserted.any():
            c=loop_aware_counts(len(np.asarray(pred_nodes).reshape(-1,2)),pred_edges,mapping,gt_edges,len(inserted),inserted)
            near=(cdist(np.asarray(pred_nodes).reshape(-1,2),np.asarray(gt_nodes).reshape(-1,2)).min(1)<=tolerance).sum() if len(gt_nodes) and len(pred_nodes) else 0
            return dict(node=counts(c['node_tp'],c['pred_nodes'],c['gt_nodes']),edge=counts(c['tp'],c['pred_edges'],c['gt_edges']),
                gt_edges_both_endpoints_recovered=int(c['recoverable']),conditional_relation_recall=c['tp']/c['recoverable'] if c['recoverable'] else None,
                false_edges_touching_annotation_unmatched=int(sum(v[1] for k,v in c['kinds'].items() if k!='MM')),duplicates=int(near-len(mapping)),
                localization_mean=float(distance.mean()) if len(distance) else None,ignored_nodes=c['ignored'])
    reference={tuple(sorted(e)) for e in np.asarray(gt_edges).reshape(-1,2).tolist()}
    true=0;unmatched=0
    for a,b in np.asarray(pred_edges).reshape(-1,2).tolist():
        if a not in mapping or b not in mapping:unmatched+=1
        elif tuple(sorted((mapping[a],mapping[b]))) in reference:true+=1
    recovered=set(mapping.values());eligible=sum(a in recovered and b in recovered for a,b in reference)
    near=(cdist(np.asarray(pred_nodes).reshape(-1,2),np.asarray(gt_nodes).reshape(-1,2)).min(1)<=tolerance).sum() if len(gt_nodes) and len(pred_nodes) else 0
    return dict(node=counts(len(mapping),len(pred_nodes),len(gt_nodes)),edge=counts(true,len(pred_edges),len(gt_edges)),
        gt_edges_both_endpoints_recovered=int(eligible),conditional_relation_recall=true/eligible if eligible else None,
        false_edges_touching_annotation_unmatched=int(unmatched),duplicates=int(near-len(mapping)),
        localization_mean=float(distance.mean()) if len(distance) else None)


def _boxes(nodes,edges=None,size=.2,sizes=None):
    xy=np.asarray(nodes,float).reshape(-1,2)
    if edges is None:
        width=np.asarray(sizes) if sizes is not None else size
        low=xy-width/2;high=xy+width/2
    else:
        endpoints=xy[np.asarray(edges,int).reshape(-1,2)]
        low=endpoints.min(1)-.1;high=endpoints.max(1)+.1
    boxes=np.concatenate((np.clip(low,0,1),np.clip(high,0,1),np.zeros((len(low),1)),np.ones((len(low),1))),1)
    return boxes


def common_metrics(prediction,target):
    # Common-v2 fixed 1-unit anchor/coverage and .25-unit sampling conventions.
    # Units remain pixel unless a scan-specific physical calibration is known.
    metric=np.asarray(target['metric']);metric3=np.eye(3);metric3[:2,:2]=metric
    stream=BytesIO()
    np.savez(stream,metric=metric3,nodes=lift(target['nodes']),edges=target['edges'],
        full_paths=lift(target['full_paths']),path_offsets=target['path_offsets'])
    stream.seek(0)
    pred=dict(nodes=lift(prediction['nodes']),edges=prediction['edges'])
    for key in ('edge_polylines','spline_controls'):
        if key in prediction:pred[key]=[lift(p) for p in prediction[key]]
    row=evaluate_common_vessels(pred,stream)
    result=aggregate_common_vessels([row])
    units=str(np.asarray(target['units']).item())
    if units!='mm':
        result={k.replace('_mm','_distance'):v for k,v in result.items()}
    return {**result,'distance_units':units}


def evaluate(prediction,target,tolerances=(1.,2.)):
    pn=np.asarray(prediction['nodes']).reshape(-1,2);pe=np.asarray(prediction['edges'],int).reshape(-1,2)
    gn=np.asarray(target['nodes']);ge=np.asarray(target['edges'],int).reshape(-1,2);metric=np.asarray(target['metric'])
    result={str(t):positional(pn@metric.T,pe,gn@metric.T,ge,t) for t in tolerances}
    p0,p1=graph_betti_numbers(len(pn),pe);g0,g1=graph_betti_numbers(len(gn),ge)
    result.update(units=str(np.asarray(target['units']).item()),predicted_nodes=len(pn),target_nodes=len(gn),
        predicted_edges=len(pe),target_edges=len(ge),node_count_bias=len(pn)-len(gn),edge_count_bias=len(pe)-len(ge),
        predicted_beta0=p0,target_beta0=g0,predicted_beta1=p1,target_beta1=g1,
        beta0_signed_error=p0-g0,beta1_signed_error=p1-g1)
    widths=np.asarray(prediction['node_boxes'])[:,2:] if 'node_boxes' in prediction else None
    result['node_mAP'],result['node_mAR']=detection_ap_ar(_boxes(pn,sizes=widths),prediction.get('node_scores',np.ones(len(pn))),_boxes(gn))
    result['edge_mAP'],result['edge_mAR']=detection_ap_ar(_boxes(pn,pe),prediction.get('edge_scores',np.ones(len(pe))),_boxes(gn,ge))
    # Retain original normalized-coordinate straight-edge SMD, explicitly separate
    # from physical dense-reference Curve F1 and squared-distance physical SMD.
    result['smd_native_normalized']=street_mover_distance(lift(gn),ge,lift(pn),pe)
    result['smd_physical_squared']=street_mover_distance(lift(gn@metric.T),ge,lift(pn@metric.T),pe)
    result['common_v2']=common_metrics(prediction,target)
    return result


@torch.no_grad()
def decode(model,tokens,predictions,batch,metric,*,node_threshold=.5,edge_threshold=.5):
    confidence=predictions['pred_logits'][batch].softmax(-1)[:,1]
    selected=torch.where(confidence>=node_threshold)[0]
    local_pairs=torch.combinations(torch.arange(len(selected),device=tokens.device),r=2)
    pairs=selected[local_pairs]
    score=model.relation_logits(tokens[batch],pairs).softmax(-1)[:,1] if len(pairs) else tokens.new_empty(0)
    keep=score>=edge_threshold
    result=dict(nodes=predictions['pred_nodes'][batch,selected,:2].cpu().numpy(),
        node_boxes=predictions['pred_nodes'][batch,selected,:4].cpu().numpy(),
        query_ids=selected.cpu().numpy(),edges=local_pairs[keep].cpu().numpy(),
        node_scores=confidence[selected].cpu().numpy(),edge_scores=score[keep].cpu().numpy())
    if model.spline_head is not None and keep.any():
        controls,_=model.curves(tokens[batch],predictions['pred_nodes'][batch],pairs[keep],metric)
        result['spline_controls']=controls.cpu().numpy()
    return result


def relation_ranking(model,tokens,predictions,batch,target,tolerance=2.,node_threshold=.5):
    selected=torch.where(predictions['pred_logits'][batch].softmax(-1)[:,1]>=node_threshold)[0]
    pairs=torch.combinations(torch.arange(len(selected),device=tokens.device),r=2)
    scores=model.relation_logits(tokens[batch],selected[pairs]).softmax(-1)[:,1].detach().cpu().numpy()
    coordinates=predictions['pred_nodes'][batch,selected,:2].detach().cpu().numpy()
    m=np.asarray(target['metric']);mapping,_=match_nodes(coordinates@m.T,np.asarray(target['nodes'])@m.T,tolerance)
    edges={tuple(sorted(x)) for x in target['edges'].tolist()};labels=[];conditional=[]
    for a,b in pairs.cpu().tolist():
        ok=a in mapping and b in mapping;conditional.append(ok)
        labels.append(int(ok and tuple(sorted((mapping[a],mapping[b]))) in edges))
    labels=np.asarray(labels);conditional=np.asarray(conditional,bool)
    result={}
    for name,mask in [('all_retained',np.ones(len(labels),bool)),('conditional',conditional)]:
        y=labels[mask];s=scores[mask]
        result[name]=dict(candidate_count=len(y),positives=int(y.sum()),prevalence=float(y.mean()) if len(y) else None,
            ap=float(average_precision_score(y,s)) if y.sum() else None)
    return result
