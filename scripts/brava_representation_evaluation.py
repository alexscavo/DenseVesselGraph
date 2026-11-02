"""Same anatomical reference for changed graphs; oracle geometry stays diagnostic."""
import io,json,zipfile,copy
from pathlib import Path
import numpy as np
import torch
from scripts.brava_representation_study import ROOT,read,dataset,batch
from scripts.multiscale_pretrained import graph_cache,rows_for,summarize
from scripts.multidataset_graph_evaluation import curve_outputs,distance,clean
from scripts.multidataset_graph_training import edge_inputs
from scripts.node_verifier_metrics import association
from scripts.analyze_spline_target_capacity import write
from models.spline import canonical_reverse
from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels


@torch.no_grad()
def snapshot(bundle,step,tiny=False):
 model=bundle['model'];model.eval();criterion=bundle['criterion'];p=bundle['plan'];folder=ROOT/('tiny-predictions' if tiny else 'predictions')/bundle['group'];folder.mkdir(parents=True,exist_ok=True)
 names_by_split={'train':p['tiny_ids']} if tiny else {'train':p['monitor_train'],'val':p['ids']['val']}
 records=[];parity={};path=folder/f'{step:06d}.zip';assert not path.exists()
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
  for split,names in names_by_split.items():
   ds=dataset(bundle['cfg'],bundle['cache'],names,bundle['arm'],split,False)
   for i,s in enumerate(names):
    x,y=batch(ds,[i]);tokens,pred,_=model(x);c=graph_cache(model,criterion,tokens,pred,y)
    ids=np.flatnonzero(c['scores']>.5);idx={int(q):j for j,q in enumerate(ids)}
    kept=np.array([int(a) in idx and int(b) in idx and score>.5 for (a,b),score in zip(c['pairs'],c['relations'])]);qp=torch.tensor(c['pairs'][kept],device=tokens.device,dtype=torch.long)
    metric=y['spline'][0]['metric'].to(tokens);local=dict(local_features=pred['spline_local_features'][0],input_shape=pred['spline_input_shape'])
    normalized=pred['pred_nodes'][0,:,:3]
    c.update(nodes=normalized[ids].cpu().numpy(),query_ids=ids,edges=np.array([[idx[int(a)],idx[int(b)]] for a,b in c['pairs'][kept]],dtype=np.int64).reshape(-1,2))
    gt=bundle['cache'][s][bundle['arm']];original=bundle['cache'][s]['K8']
    c.update(truth_original=original['nodes']@original['metric'].T,gt_boundary=gt['boundary'],original_boundary=original['boundary'])
    assignment=criterion.matcher(pred,y)[0];data=edge_inputs(model,criterion,tokens,pred,y,0,assignment)
    c.update(hungarian_source=assignment[0].numpy(),hungarian_target=assignment[1].numpy())
    heads=bundle['heads']
    for name,head in heads.items():
     arm=name if bundle['group']=='original' else 'split_K8';out=dict(c)
     ep=normalized[qp];rev=canonical_reverse(ep@metric.T);oriented=qp.clone();oriented[rev]=oriented[rev].flip(1)
     rel=tokens[0,criterion.object_queries:criterion.object_queries+criterion.relation_tokens];features=criterion._relation_features(tokens[0,:criterion.object_queries],rel,oriented)
     curves,_=curve_outputs(head,features,normalized[oriented],metric,local);curves[rev.cpu().numpy()]=curves[rev.cpu().numpy(),::-1]
     out['curves']=curves
     if data is not None:
      selected,reverse,features,endpoints,metric,local=data
      curves,_=curve_outputs(head,features,endpoints,metric,local)
      gt_ends=y['nodes'][0][y['edges'][0][selected]].clone();gt_ends[reverse]=gt_ends[reverse].flip(1)
      oracle,_=curve_outputs(head,features,gt_ends,metric,local)
      curves[reverse.cpu().numpy()]=curves[reverse.cpu().numpy(),::-1];oracle[reverse.cpu().numpy()]=oracle[reverse.cpu().numpy(),::-1]
      out.update(matched_curves=curves,oracle_endpoint_curves=oracle,matched_edges=np.array(selected),parent_branch=gt['parent_branch'][selected])
     else:out.update(matched_curves=np.empty((0,257,3)),oracle_endpoint_curves=np.empty((0,257,3)),matched_edges=np.array([],int),parent_branch=np.array([],int))
     member=f'{arm}/{s}.npz';buffer=io.BytesIO();np.savez_compressed(buffer,**out);archive.writestr(member,buffer.getvalue())
     records.append(dict(member=member,sample=s,subject=p['subjects'][s.split('_')[1]],split=split,arm=arm,gt=str(Path(p['dataset'])/split/'spline'/(s+'.npz'))))
    if i%64==0:print('snapshot',bundle['group'],step,split,i,len(names),flush=True)
 write(folder/f'{step:06d}-records.json',records)


def evaluate(group,step,tiny=False):
 p=read(ROOT/'plan.json');folder=ROOT/('tiny-predictions' if tiny else 'predictions')/group;records=read(folder/f'{step:06d}-records.json');rows=[];geometry=[]
 audits={x['sample']:x for x in read(ROOT/'target-audit.json')['rows']}
 with zipfile.ZipFile(folder/f'{step:06d}.zip') as archive:
  for i,rec in enumerate(records):
   with np.load(io.BytesIO(archive.read(rec['member']))) as f:c={k:f[k] for k in f.files}
   meta={k:rec[k] for k in ['arm','split','sample','subject']}
   with np.load(rec['gt']) as f:gt={k:f[k] for k in f.files}
   prediction=dict(nodes=c['nodes'],edges=c['edges'],edge_polylines=c['curves'])
   common=evaluate_common_vessels(prediction,rec['gt']);chord=evaluate_common_vessels(dict(nodes=c['nodes'],edges=c['edges']),rec['gt'])
   for tol,r in rows_for(c).items():
    metric_keys=list(r)
    ids=np.flatnonzero(c['scores']>.5);t=float(tol);raw=association(c['physical'],c['truth_original'],t);ret=association(c['physical'][ids],c['truth_original'],t)
    border=c.get('original_boundary',gt['boundary']).astype(bool);strata={label+'_original_anchor_tp':sum(bool(border[j])==flag for j in ret.values()) for label,flag in [('border',True),('interior',False)]};strata.update(border_original_anchors=int(border.sum()),interior_original_anchors=int((~border).sum()))
    rows.append(dict(meta,tolerance=tol,**r,**common,**strata,chord_common_curve_f1=aggregate_common_vessels([chord])['common_curve_f1'],raw_original_anchor_tp=len(raw),filtered_original_anchor_tp=len(ret),original_anchors=len(c['truth_original'])))
   # Join learned subedges in original-parent order before comparing geometry.
   for parent in np.unique(c['parent_branch']):
    selection=np.flatnonzero(c['parent_branch']==parent);order=selection[np.argsort(c['matched_edges'][selection])]
    a,b=gt['path_offsets'][parent:parent+2];path=gt['full_paths'][a:b].astype(float)@gt['metric'].T
    def concatenate(key):return np.concatenate([c[key][j] if k==0 else c[key][j][1:] for k,j in enumerate(order)]).astype(float)@gt['metric'].T
    curve=concatenate('matched_curves');oracle=concatenate('oracle_endpoint_curves');endpoint=float(np.linalg.norm(curve[[0,-1]]-path[[0,-1]],axis=1).max())
    from data.spline_targets import resample
    curve=resample(curve,257);oracle=resample(oracle,257);t=np.linspace(0,1,len(curve))[:,None]
    neutral=curve-((1-t)*curve[0]+t*curve[-1])+((1-t)*path[0]+t*path[-1])
    error=distance(curve,path);nm=distance(neutral,path);om=distance(oracle,path);reference=audits[rec['sample']]['oracle'][parent]
    geometry.append(dict(meta,parent=int(parent),subedges=len(order),endpoint_max_mm=endpoint,mean_mm=error[0],max_mm=error[1],neutral_mean_mm=nm[0],neutral_max_mm=nm[1],oracle_endpoint_mean_mm=om[0],oracle_endpoint_max_mm=om[1],difficult=reference['K8_sampled_max_mm']>1.,**reference))
   if i%128==0:print('evaluate',group,step,i,len(records),flush=True)
 summary={}
 for arm in sorted({r['arm'] for r in rows}):
  summary[arm]={}
  for split in ['train','val']:
   rr=[r for r in rows if r['arm']==arm and r['split']==split and r['tolerance']=='2.0'];gg=[r for r in geometry if r['arm']==arm and r['split']==split]
   if not rr:continue
   physical={tol:summarize([{k:r[k] for k in metric_keys} for r in rows if r['arm']==arm and r['split']==split and r['tolerance']==tol]) for tol in ['1.0','2.0']}
   cm=aggregate_common_vessels([{k:v for k,v in r.items() if k.startswith('common_')} for r in rr])
   gs={}
   for label,subset in [('all',gg),('difficult',[r for r in gg if r['difficult']]),('localized',[r for r in gg if r['endpoint_max_mm']<=2]),('difficult_localized',[r for r in gg if r['difficult'] and r['endpoint_max_mm']<=2])]:
    gs[label]=dict(branches=len(subset),**{k:float(np.mean([r[k] for r in subset])) if subset else None for k in ['mean_mm','max_mm','neutral_mean_mm','oracle_endpoint_mean_mm','oracle_endpoint_max_mm']})
   summary[arm][split]=dict(physical=physical,common=cm,geometry=gs,original_anchor_recall={tol:{'raw':sum(r['raw_original_anchor_tp'] for r in rows if r['arm']==arm and r['split']==split and r['tolerance']==tol)/sum(r['original_anchors'] for r in rr),'filtered':sum(r['filtered_original_anchor_tp'] for r in rows if r['arm']==arm and r['split']==split and r['tolerance']==tol)/sum(r['original_anchors'] for r in rr)} for tol in ['1.0','2.0']})
 out=ROOT/('tiny-evaluation' if tiny else 'evaluation')/group/f'{step:06d}';out.mkdir(parents=True,exist_ok=True);write(out/'rows.json',clean(rows));write(out/'geometry.json',clean(geometry));write(out/'summary.json',clean(summary));print('evaluated',out,flush=True)
