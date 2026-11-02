"""CPU-only fixed-cache graph, relation-population and dense-path attribution."""
import argparse,copy,csv,functools,gzip,hashlib,io,json,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components,shortest_path
from sklearn.metrics import average_precision_score
from scripts.node_verifier_metrics import association
from metrics.representation_geometry import sample_polyline_by_arclength,point_to_polyline_distance

ORIGINAL=Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph')
OUT=Path('experiments/graph_error_attribution_20261026')
SEEDS=[364505,92741]
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 def clean(v):
  if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [clean(x) for x in v]
  if isinstance(v,np.ndarray):return clean(v.tolist())
  if isinstance(v,np.generic):return clean(v.item())
  if isinstance(v,float) and not np.isfinite(v):return None
  return v
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvwrite(path,rows):
 fields=sorted(set().union(*(r.keys() for r in rows)))
 with (gzip.open(path,'wt',newline='') if str(path).endswith('.gz') else Path(path).open('w',newline='')) as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def topo(n,edges):
 if not n:return 0,0
 edges=np.asarray(edges,int).reshape(-1,2)
 mat=csr_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n))
 b0=connected_components(mat,directed=False,return_labels=False)
 return int(b0),int(len(edges)-n+b0)
def terminal_branches(ids,edges,mapping,positive_edges):
 adj={int(q):[] for q in ids};seen=set();count=unsupported=0
 for a,b in edges:adj[int(a)].append(int(b));adj[int(b)].append(int(a))
 terminals=[a for a in adj if len(adj[a])==1]
 for start in terminals:
  first=tuple(sorted((start,adj[start][0])))
  if first in seen:continue
  previous=None;node=start;path=[];nodes=[start]
  while True:
   options=[b for b in adj[node] if b!=previous]
   if not options:break
   nxt=options[0];edge=tuple(sorted((node,nxt)));seen.add(edge);path.append(edge);nodes.append(nxt)
   previous,node=node,nxt
   if len(adj[node])!=2:break
  count+=1
  if any(q not in mapping for q in [nodes[0],nodes[-1]] if len(adj[q])==1) and not any(e in positive_edges for e in path):unsupported+=1
 return len(terminals),count,unsupported

def match_hungarian(c,gt,config):
 """Actual normalized-L1 minus object-probability cost; never spatial matching."""
 xyz=np.asarray(c['physical']@np.linalg.inv(gt['metric'].T),dtype=np.float32);truth=gt['nodes'].astype(np.float32)
 spec=config['model']['matcher'];assert spec['type']=='hungarian'
 cost=spec['node_cost']*np.abs(xyz[:,None]-truth[None]).sum(-1)-spec['class_cost']*c['scores'][:,None]
 source,target=linear_sum_assignment(cost)
 distance=np.linalg.norm(c['physical'][source]-c['truth'][target],axis=-1)
 error=float(np.max(np.abs(distance-c['hungarian_distance']))) if 'hungarian_distance' in c else None
 return source,target,error

@functools.lru_cache(maxsize=2048)
def geometry(path):
 with np.load(path) as f:gt={k:f[k] for k in ['nodes','edges','metric','full_paths','path_offsets']}
 metric=gt['metric'];truth=gt['nodes']@metric.T;paths=gt['full_paths']@metric.T;offsets=gt['path_offsets'];geometries=[];segments=[]
 links={tuple(sorted(map(int,e))) for e in gt['edges']};assert len(links)==len(gt['edges']),'Parallel targets require separate scoring'
 for i,(a,b) in enumerate(gt['edges']):
  path=paths[offsets[i]:offsets[i+1]];assert np.allclose(path[[0,-1]],truth[[a,b]],atol=2e-4)
  delta=np.diff(path,axis=0);length=float(np.linalg.norm(delta,axis=1).sum());chord=float(np.linalg.norm(truth[a]-truth[b]));sample=sample_polyline_by_arclength(path,.5)
  direction=np.diff(sample,axis=0);norm=np.linalg.norm(direction,axis=1);direction=direction[norm>1e-8]/norm[norm>1e-8,None]
  angles=np.arccos(np.clip((direction[:-1]*direction[1:]).sum(-1),-1,1)) if len(direction)>1 else np.array([])
  turning=float(angles.sum());geometries.append(dict(edge=i,a=int(a),b=int(b),arc_mm=length,chord_mm=chord,tortuosity=length/chord if chord>1e-8 else None,turning_rad=turning,curvature_per_mm=turning/length if length else 0.))
  segments.append(path)
 gt.update(truth=truth,geometry=geometries,paths=segments,links=links)
 n=len(truth);e=gt['edges'];mat=csr_matrix((np.ones(len(e)*2),(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(n,n))
 gt['hops']=shortest_path(mat,directed=False,unweighted=True)
 return gt

def cases():
 result=[];p=read(ORIGINAL/'experiments/local_refinement_long/plan.json');root=Path(p['dataset']['root'])
 for seed in SEEDS:
  for arm in ['B0','A-xyz']:
   folder=ORIGINAL/f'experiments/local_refinement_long/{seed}/runs/{arm}';done=read(folder/'complete.json');assert done['completed']
   for phase,step in [('final',12000)]+([('selected',done['selected_step'])] if arm=='A-xyz' and done['selected_step']!=12000 else []):
    directory=folder/f'step-{step}';records=read(directory/'records.json');records=[dict(r,gt=str(root/r['split']/'spline'/(r['sample']+'.npz'))) for r in records]
    result.append(dict(name=f'SyntheticMRI/{seed}/{arm}/{phase}',dataset='SyntheticMRI',arm=arm,seed=seed,phase=phase,step=step,config=read(folder/'config.json'),records=records,archive=str(directory/'caches.zip'),checkpoint=str(folder/f'model-{step}.pt')))
 root=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_query_diagnostics_20261019')
 for arm,job in [('parent',1291324),('baseline',1291328),('local_image',1291329)]:
  directory=root/f'pilot-{job}-{arm}';m=read(directory/'manifest.json');records=[]
  for split,names in m['ids'].items():
   for name in names:
    source=read(directory/split/(name+'-source.json'));records.append(dict(sample=name,subject=source['subject'],split=split,gt=source['gt'],gt_sha256=source['gt_sha256'],cache=str(directory/split/(name+'.npz'))))
  result.append(dict(name='IXI-legacy/'+arm,dataset='IXI-legacy',arm=arm,seed=181028,phase='legacy',step=1000 if arm!='parent' else 28170,config=m['config'],records=records,manifest=m))
 # Same local-refinement checkpoint, disable only its confidence residual offline.
 local=copy.deepcopy(result[-1]);local.update(name='IXI-legacy/coordinate-only-ablation',arm='coordinate-only-ablation',phase='inference component ablation');result.append(local)
 return result

def load_case(case):
 archive=zipfile.ZipFile(case['archive']) if 'archive' in case else None
 try:
  for record in case['records']:
   if archive:
    with np.load(io.BytesIO(archive.read(record['sample']+'.npz'))) as f:c={k:f[k] for k in f.files}
   else:
    with np.load(record['cache']) as f:
     c=dict(physical=f['nodes']@f['metric'].T,truth=f['gt_nodes']@f['metric'].T,gt_edges=f['gt_edges'],scores=f['node_scores'],pairs=f['query_ids'][f['edges']],relations=f['edge_scores'])
     if case['arm']=='coordinate-only-ablation':
      logits=f['original_logits'];exp=np.exp(logits-logits.max(-1,keepdims=True));c['scores']=exp[:,1]/exp.sum(-1)
   yield record,c
 finally:
  if archive:archive.close()

def make_cuts(allcases):
 result={}
 for dataset in sorted({c['dataset'] for c in allcases}):
  reference=next(c for c in allcases if c['dataset']==dataset);rows=[]
  for rec in reference['records']:
   if rec['split']=='train':rows.extend(geometry(rec['gt'])['geometry'])
  result[dataset]={field:np.quantile([r[field] for r in rows if r[field] is not None],[.25,.5,.75]).tolist() for field in ['arc_mm','chord_mm','tortuosity','turning_rad','curvature_per_mm']}
  result[dataset]['training_patches']=sum(r['split']=='train' for r in reference['records']);result[dataset]['training_edges']=len(rows)
 return result

def evaluate_mapping(c,gt,tol,ids,mapping,distance=None):
 ids=np.asarray(sorted(ids),int);distance=np.linalg.norm(c['physical'][:,None]-c['truth'][None],axis=-1) if distance is None else distance
 q=len(c['scores']);selected=np.zeros(q,bool);selected[ids]=True;target=np.full(q,-1,int)
 for a,b in mapping.items():target[a]=b
 a,b=c['pairs'].T;active=selected[a]&selected[b];ga,gb=target[a],target[b];mm=(ga>=0)&(gb>=0)
 adjacency=np.zeros((len(c['truth']),len(c['truth'])),bool);e=gt['edges'];adjacency[e[:,0],e[:,1]]=adjacency[e[:,1],e[:,0]]=True
 labels=mm&adjacency[np.maximum(ga,0),np.maximum(gb,0)];chosen=active&(c['relations']>.5);tp=int((chosen&labels).sum());fp=int((chosen&~labels).sum())
 recovered=set(mapping.values());eligible=sum(int(x) in recovered and int(y) in recovered for x,y in gt['edges'])
 duplicate=np.zeros(q,bool)
 if recovered:
  near=(distance[:,list(recovered)]<=tol).any(1);duplicate=selected&near&(target<0)
 local={int(x):i for i,x in enumerate(ids)};decoded=c['pairs'][chosen];local_edges=np.array([[local[int(x)],local[int(y)]] for x,y in decoded],int).reshape(-1,2)
 pred0,pred1=topo(len(ids),local_edges);gt0,gt1=topo(len(gt['nodes']),gt['edges']);links={tuple(sorted(map(int,p))) for p in c['pairs'][chosen&labels]}
 terminals,branches,unsupported=terminal_branches(ids,decoded,mapping,links);gt_degree=np.bincount(gt['edges'].ravel(),minlength=len(gt['nodes']))
 normal=dict(nodes=len(ids),gt_nodes=len(gt['nodes']),node_tp=len(mapping),edges=int(chosen.sum()),gt_edges=len(gt['edges']),edge_tp=tp,edge_fp=fp,edge_fn=len(gt['edges'])-tp,eligible_edges=eligible,duplicates=int(duplicate.sum()),high_conf_duplicates=int((duplicate&(c['scores']>=.9)).sum()),pred_beta0=pred0,gt_beta0=gt0,pred_beta1=pred1,gt_beta1=gt1,terminal_nodes=terminals,gt_terminal_nodes=int((gt_degree==1).sum()),terminal_branches=branches,unsupported_terminal_branches=unsupported,
   fp_mm=int((chosen&~labels&mm).sum()),fp_mu=int((chosen&~labels&((ga>=0)^(gb>=0))).sum()),fp_uu=int((chosen&~labels&(ga<0)&(gb<0)).sum()),fp_duplicate=int((chosen&~labels&(duplicate[a]|duplicate[b])).sum()),matched_mm_negative_candidates=int((active&mm&~labels).sum()),matched_mm_true_candidates=int((active&labels).sum()),unmatched_candidates=int((active&~mm).sum()),patches=1)
 for key,pred,truth in [('nodes',len(ids),len(gt['nodes'])),('edges',int(chosen.sum()),len(gt['edges'])),('beta0',pred0,gt0),('beta1',pred1,gt1),('terminals',terminals,int((gt_degree==1).sum()))]:normal[key+'_signed_error']=pred-truth;normal[key+'_absolute_error']=abs(pred-truth)
 return normal,dict(mapping=mapping,target=target,active=active,mm=mm,labels=labels,chosen=chosen,duplicate=duplicate,distance=distance,eligible_scores=c['relations'][active&mm],eligible_labels=labels[active&mm])

def safe_selection(c,gt,tol,condition,hungarian=None):
 if condition=='normal':ids=np.flatnonzero(c['scores']>.5);local=association(c['physical'][ids],c['truth'],tol);mapping={int(ids[a]):b for a,b in local.items()}
 elif condition=='delete_unmatched':
  ids0=np.flatnonzero(c['scores']>.5);local=association(c['physical'][ids0],c['truth'],tol);mapping={int(ids0[a]):b for a,b in local.items()};ids=np.array(list(mapping))
 elif condition=='raw_spatial_oracle':mapping=association(c['physical'],c['truth'],tol);ids=np.array(list(mapping))
 elif condition=='hungarian_selected':
  ids=hungarian[0];local=association(c['physical'][ids],c['truth'],tol);mapping={int(ids[a]):b for a,b in local.items()}
 return evaluate_mapping(c,gt,tol,ids,mapping)

def feasible_endpoints(distance,edges,tol):
 candidates=distance<=tol;result=[]
 for a,b in edges:
  aa=np.flatnonzero(candidates[:,a]);bb=np.flatnonzero(candidates[:,b]);result.append(len(aa)>0 and len(bb)>0 and (len(aa)>1 or len(bb)>1 or aa[0]!=bb[0]))
 return np.array(result,bool)

def analyze_one(case,cuts,limit=None):
 rows=[];positives=[];false=[];population=[];distance_rows=[];counter=[];ap={};replay=[]
 case_meta={k:case[k] for k in ['name','dataset','arm','seed','phase','step']};cut=cuts[case['dataset']]
 if limit:case=dict(case,records=[r for split in ['train','val'] for r in [x for x in case['records'] if x['split']==split][:2]])
 for number,(record,c) in enumerate(load_case(case)):
  gt=geometry(record['gt']);assert np.array_equal(gt['edges'],c['gt_edges']);assert np.allclose(gt['truth'],c['truth'],atol=2e-5)
  q=len(c['scores']);assert len(c['pairs'])==q*(q-1)//2 and len(np.unique(c['pairs'],axis=0))==len(c['pairs'])
  hs,ht,error=match_hungarian(c,gt,case['config']);replay.append(error);hmap=np.full(q,-1,int);hmap[hs]=ht
  # All assigned GT edges are eligible supervised positives; cap only negatives.
  hg=c['pairs'];hmm=(hmap[hg]>=0).all(1);hl=np.zeros(len(hg),bool)
  adjacency=np.zeros((len(gt['nodes']),)*2,bool);ee=gt['edges'];adjacency[ee[:,0],ee[:,1]]=adjacency[ee[:,1],ee[:,0]]=True
  hl[hmm]=adjacency[hmap[hg[hmm,0]],hmap[hg[hmm,1]]]
  extras=dict(sample=record['sample'],subject=record['subject'],split=record['split'],**case_meta)
  for tol in [1.,2.]:
   summary,flags=safe_selection(c,gt,tol,'normal');summary.update(extras,tolerance=tol)
   raw=association(c['physical'],c['truth'],tol);summary['raw_node_tp']=len(raw);rows.append(summary)
   target=flags['target'];pair=c['pairs'];a,b=pair.T;mm=flags['mm'];chosen=flags['chosen'];labels=flags['labels'];dist=flags['distance'];dup=flags['duplicate'];mapping=flags['mapping'];active=flags['active']
   pair_distance=np.linalg.norm(c['physical'][a]-c['physical'][b],axis=-1);fp=chosen&~labels
   hops=np.full(len(pair),np.nan);hops[mm]=gt['hops'][target[a[mm]],target[b[mm]]]
   shortcut=fp&mm&np.isfinite(hops)&(hops>=2);summary['fp_gt_shortcut']=int(shortcut.sum());summary['fp_cross_component']=int((fp&mm&~np.isfinite(hops)).sum());summary['fp_hungarian_unmatched']=int((fp&~hmm).sum())
   inverse={v:k for k,v in mapping.items()};scores=np.full((q,q),np.nan);scores[a,b]=scores[b,a]=c['relations'];rawpossible=feasible_endpoints(dist,gt['edges'],tol);retpossible=feasible_endpoints(dist[c['scores']>.5],gt['edges'],tol)
   for i,g in enumerate(gt['geometry']):
    x,y=g['a'],g['b'];eligible=x in inverse and y in inverse;score=float(scores[inverse[x],inverse[y]]) if eligible else None;correct=eligible and score>.5
    cause='correct' if correct else 'localized_relation_miss' if eligible else 'no_two_raw_queries' if not rawpossible[i] else 'filtering_loss' if not retpossible[i] else 'retained_assignment_ambiguity'
    row=dict(**extras,tolerance=tol,**g,endpoints_recovered=bool(eligible),raw_pair_feasible=bool(rawpossible[i]),retained_pair_feasible=bool(retpossible[i]),correct=bool(correct),score=score,cause=cause)
    for field in ['arc_mm','chord_mm','tortuosity','turning_rad','curvature_per_mm']:row[field+'_quartile']=int(np.searchsorted(cut[field],g[field],side='right')) if g[field] is not None else -1
    positives.append(row);summary[cause]=summary.get(cause,0)+1
   for i in np.flatnonzero(fp):
    x,y=pair[i];kind='MM' if mm[i] else 'MU' if (target[x]>=0)^(target[y]>=0) else 'UU'
    false.append(dict(**extras,tolerance=tol,q1=int(x),q2=int(y),endpoint_distance_mm=float(pair_distance[i]),score=float(c['relations'][i]),category=kind,duplicate=bool(dup[x] or dup[y]),hungarian_unmatched=bool(not hmm[i]),gt_hops=float(hops[i]) if np.isfinite(hops[i]) else None,gt_shortcut=bool(shortcut[i])))
   for condition in ['normal','delete_unmatched','raw_spatial_oracle','hungarian_selected']:
    r,z=(summary,flags) if condition=='normal' else safe_selection(c,gt,tol,condition,(hs,ht))
    counter.append(dict(r,**extras,tolerance=tol,condition=condition,raw_node_tp=len(raw)));key=(record['split'],tol,condition);ap.setdefault(key,[[],[]]);ap[key][0].append(z['eligible_labels']);ap[key][1].append(z['eligible_scores'])
   # Operational reliable labels are independent of Hungarian assignment.
   near=dist<=tol;unique=near.sum(1)==1;nearest=dist.argmin(1);both_unique=unique[a]&unique[b]&(nearest[a]!=nearest[b]);sp_positive=both_unique&adjacency[nearest[a],nearest[b]];sp_negative=both_unique&~adjacency[nearest[a],nearest[b]];ambiguous=~both_unique
   for pop,mask in [('all_raw',np.ones(len(pair),bool)),('hungarian_pair_pool',hmm),('inference_candidates',active),('predicted_edges',chosen)]:
    population.append(dict(**extras,tolerance=tol,population=pop,pairs=int(mask.sum()),supervised_positive_labels=int((mask&hl).sum()),supervised_negative_labels=int((mask&hmm&~hl).sum()),hungarian_unmatched=int((mask&~hmm).sum()),unambiguous_spatial_positive=int((mask&sp_positive).sum()),unambiguous_topology_negative=int((mask&sp_negative).sum()),spatial_ambiguous=int((mask&ambiguous).sum()),hungarian_unmatched_spatial_positive=int((mask&~hmm&sp_positive).sum()),hungarian_negative_spatial_positive=int((mask&hmm&~hl&sp_positive).sum()),assigned_positive_spatial_negative=int((mask&hl&sp_negative).sum())))
   bins=np.digitize(pair_distance,[5,10,20])
   for pop,mask in [('all_retained',active),('matched_nonadjacent',active&mm&~labels),('unmatched_involving',active&~mm)]:
    for k in range(4):
     m=mask&(bins==k);positive=int((m&labels).sum());negative=int((m&~labels).sum());negative_chosen=int((m&~labels&chosen).sum())
     distance_rows.append(dict(**extras,tolerance=tol,population=pop,distance_bin=k,candidates=int(m.sum()),positives=positive,negatives=negative,false_edges=negative_chosen,true_edges=int((m&labels&chosen).sum())))
  if number%128==0:print(case['name'],number,len(case['records']),flush=True)
 pooled=[]
 for (split,tol,condition),(ys,ps) in ap.items():
  y=np.concatenate(ys);p=np.concatenate(ps);pooled.append(dict(case=case['name'],split=split,tolerance=tol,condition=condition,candidates=len(y),positives=int(y.sum()),conditional_ap=float(average_precision_score(y,p)) if y.any() else None))
 return dict(rows=rows,positives=positives,false_edges=false,populations=population,distance=distance_rows,counterfactuals=counter,ranking=pooled,hungarian_replay_max_mm=max([x for x in replay if x is not None],default=None),hungarian_replay_failures=sum(x is not None and x>.001 for x in replay))

def aggregate(rows):
 keys={k for r in rows for k,v in r.items() if isinstance(v,(int,float,np.number)) and k not in ['tolerance','seed','step']}
 additive={k:sum(r.get(k,0) for r in rows) for k in keys}
 n=additive['patches'];div=lambda a,b:a/b if b else 0.
 additive.update(node_precision=div(additive['node_tp'],additive['nodes']),node_recall=div(additive['node_tp'],additive['gt_nodes']),raw_node_recall=div(additive.get('raw_node_tp',0),additive['gt_nodes']),node_f1=div(2*additive['node_tp'],additive['nodes']+additive['gt_nodes']),edge_precision=div(additive['edge_tp'],additive['edges']),edge_recall=div(additive['edge_tp'],additive['gt_edges']),edge_f1=div(2*additive['edge_tp'],additive['edges']+additive['gt_edges']),conditional_recall=div(additive['edge_tp'],additive['eligible_edges']),false_edge_unmatched_fraction=div(additive['fp_mu']+additive['fp_uu'],additive['edge_fp']),false_edge_duplicate_fraction=div(additive['fp_duplicate'],additive['edge_fp']))
 for field in ['nodes','edges','beta0','beta1','terminals']:
  for kind in ['signed','absolute']:additive[field+'_'+kind+'_error_mean']=additive[field+'_'+kind+'_error']/n
 return additive

def bootstrap_geometry(rows,field):
 """Long/high-complexity minus short/low, paired whole subjects (one seed)."""
 subjects=sorted({r['subject'] for r in rows});matrix=[]
 for s in subjects:
  rr=[r for r in rows if r['subject']==s]
  matrix.append([v for k in [0,3] for v in [sum(r['correct'] for r in rr if r[field+'_quartile']==k),sum(r['endpoints_recovered'] for r in rr if r[field+'_quartile']==k)]])
 matrix=np.asarray(matrix,float);rng=np.random.default_rng(261026);sample=matrix[rng.integers(len(subjects),size=(2000,len(subjects)))].sum(1);valid=(sample[:,1]>0)&(sample[:,3]>0);delta=sample[valid,2]/sample[valid,3]-sample[valid,0]/sample[valid,1];total=matrix.sum(0)
 return dict(subjects=len(subjects),low_correct=int(total[0]),low_eligible=int(total[1]),high_correct=int(total[2]),high_eligible=int(total[3]),delta=float(total[2]/total[3]-total[0]/total[1]) if total[1]*total[3] else None,ci95=np.quantile(delta,[.025,.975]).tolist() if len(delta) else None,valid_draws=int(valid.sum()))

def summaries(results,cuts):
 aggregate_rows=[];strata=[];intervals=[];population=[];distance=[];counter=[];ranking=[]
 for name,data in results.items():
  for split in ['train','val']:
   for tol in [1.,2.]:
    rr=[r for r in data['rows'] if r['split']==split and r['tolerance']==tol]
    if not rr:continue
    aggregate_rows.append(dict(case=name,split=split,tolerance=tol,**aggregate(rr)))
    pp=[r for r in data['positives'] if r['split']==split and r['tolerance']==tol]
    for field in ['arc_mm','chord_mm','tortuosity','turning_rad','curvature_per_mm']:
     for k in range(4):
      records=[r for r in pp if r[field+'_quartile']==k];n=len(records);eligible=sum(r['endpoints_recovered'] for r in records);tp=sum(r['correct'] for r in records)
      strata.append(dict(case=name,split=split,tolerance=tol,geometry=field,quartile=k,gt_edges=n,endpoints_recovered=eligible,true_edges=tp,endpoint_recovery=eligible/n if n else None,full_gt_recall=tp/n if n else None,conditional_recall=tp/eligible if eligible else None))
     if split=='val':intervals.append(dict(case=name,tolerance=tol,geometry=field,**bootstrap_geometry(pp,field)))
    for pop in ['all_raw','hungarian_pair_pool','inference_candidates','predicted_edges']:
     records=[r for r in data['populations'] if r['split']==split and r['tolerance']==tol and r['population']==pop];keys=[k for k,v in records[0].items() if isinstance(v,(int,float)) and k not in ['seed','step','tolerance']]
     population.append(dict(case=name,split=split,tolerance=tol,population=pop,**{k:sum(r[k] for r in records) for k in keys}))
    for condition in ['normal','delete_unmatched','raw_spatial_oracle','hungarian_selected']:
     records=[r for r in data['counterfactuals'] if r['split']==split and r['tolerance']==tol and r['condition']==condition];counter.append(dict(case=name,split=split,tolerance=tol,condition=condition,**aggregate(records)))
    for pop in ['all_retained','matched_nonadjacent','unmatched_involving']:
     for k in range(4):
      records=[r for r in data['distance'] if r['split']==split and r['tolerance']==tol and r['population']==pop and r['distance_bin']==k]
      counts={key:sum(r[key] for r in records) for key in ['candidates','positives','negatives','false_edges','true_edges']};distance.append(dict(case=name,split=split,tolerance=tol,population=pop,distance_bin=k,**counts,false_positive_rate=counts['false_edges']/counts['negatives'] if counts['negatives'] else None))
  ranking.extend(data['ranking'])
 return dict(aggregate=aggregate_rows,geometry_strata=strata,geometry_intervals=intervals,populations=population,distance=distance,counterfactuals=counter,ranking=ranking,cuts=cuts,replay={k:dict(max_mm=v['hungarian_replay_max_mm'],failures=v['hungarian_replay_failures']) for k,v in results.items()})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 cc=cases();cuts=make_cuts(cc);write(OUT/'plan.json',dict(cpu_only=True,cases=[{k:v for k,v in c.items() if k not in ['records','config','manifest']} for c in cc],cuts=cuts,thresholds=[.5,.5],tolerances_mm=[1.,2.],curvature_sampling_mm=.5,brava='excluded: incomplete annotations and concurrently owned study',report='docs/research/metrics/6-ixi-query-attribution.md'))
 results={}
 for case in cc:
  result=analyze_one(case,cuts,limit=3 if args.smoke else None);results[case['name']]=result
  dest=OUT/('smoke' if args.smoke else 'cases')/case['name'].replace('/','_');dest.mkdir(parents=True,exist_ok=True)
  for key in ['rows','positives','false_edges','populations','distance','counterfactuals']:csvwrite(dest/(key+'.csv.gz'),result[key])
  write(dest/'ranking.json',result['ranking']);print('done',case['name'],flush=True)
 output=summaries(results,cuts);write(OUT/('smoke-summary.json' if args.smoke else 'summary.json'),output)
 if not args.smoke:
  for key in ['aggregate','geometry_strata','populations','distance','counterfactuals','ranking']:csvwrite(OUT/(key+'.csv'),output[key])
 print('CPU attribution completed',flush=True)
if __name__=='__main__':main()
