"""CPU-only collection from isolated IXI cached outputs. No inference/training."""
import argparse,contextlib,fcntl,io,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.ixi_local_components import META,ARMS,ORIGINAL,plan,read,write
from scripts.node_verifier_metrics import select,measure,aggregate
from scripts.report_multiscale_pretrained import paired,selected,stats
from metrics.common_vessel_metrics import aggregate_common_vessels

def load_cache(folder,sample):
 with zipfile.ZipFile(folder/'caches.zip') as z:
  with np.load(io.BytesIO(z.read(sample+'.npz'))) as a:return {k:a[k] for k in a.files}
def rows(folder):return read(folder/'rows.json')
def snapshot_caches(folder,monitor=None):
 for rec in read(folder/'records.json'):
  if monitor is not None and rec['sample'] not in monitor:continue
  yield rec,load_cache(folder,rec['sample'])
def operating(root,arm,final,basecounts):
 folder=root/'runs'/arm/f'step-{final}';p=plan();monitor=set(p['dataset']['monitor_train'][:128]+p['dataset']['monitor_val'])
 output={};matchedrows={}
 for tol in [1.,2.]:
  key=str(tol);curve={'train':[],'val':[]};cache=list(snapshot_caches(folder,monitor));thresholds=np.unique(np.r_[np.linspace(.05,.95,19),.5]).tolist()
  memo={}
  for phase in ['train','val']:
   for t in thresholds:
    rr=[]
    for rec,c in cache:
     if rec['split']!=phase:continue
     ids=np.sort(select(c,c['scores'],threshold=t));k=(rec['sample'],len(ids))
     if k not in memo:memo[k]=measure(c,ids,tol)
     rr.append(memo[k])
    curve[phase].append(dict(threshold=t,**aggregate(rr)))
  rr=[];original={r['sample']:r for r in selected(rows(folder),tol=key)}
  for rec,c in snapshot_caches(folder):
   if rec['split']!='val':continue
   r=measure(c,np.sort(select(c,c['scores'],count=basecounts[rec['sample']])),tol)
   for k in ['raw_tp','hungarian_mm_sum','hungarian_n','nearest_mm_sum']:r[k]=original[rec['sample']][k]
   rr.append(dict(r,sample=rec['sample'],subject=rec['subject'],split='val',tolerance=key))
  calibrated={};trainbest=max(curve['train'],key=lambda x:(x['node_f1'],-x['threshold']))
  calibrated['max_train_f1']=dict(train=trainbest,val=next(x for x in curve['val'] if x['threshold']==trainbest['threshold']))
  for target in [.5,.7,.8,.9]:
   eligible=[x for x in curve['train'] if x['nodes'] and x['node_precision']>=target]
   chosen=max(eligible,key=lambda x:(x['node_recall'],-x['threshold'])) if eligible else None
   calibrated[str(target)]=None if chosen is None else dict(train=chosen,val=next(x for x in curve['val'] if x['threshold']==chosen['threshold']))
  output[key]=dict(curves=curve,calibrated=calibrated,matched_count=stats(rr));matchedrows[key]=rr
 write(root/'runs'/arm/'operating.json',output);write(root/'runs'/arm/'matched-count-rows.json',matchedrows)
 return output,matchedrows

def endpoints(folder):
 largest=0.;total=0
 for rec in read(folder/'validation-000000/predictions.json'):
  nodes=np.asarray(rec['nodes_dhw']);edges=np.asarray(rec['edges'],int).reshape(-1,2)
  assert len(set(rec['query_ids']))==len(nodes)
  if not len(edges):continue
  assert edges.min()>=0 and edges.max()<len(nodes)
  for field in ['spline_controls','spline_curves']:
   geo=np.asarray(rec[field]);largest=max(largest,float(np.max(np.abs(geo[:,[0,-1]]-nodes[edges]))))
  total+=len(edges)
 assert largest<1e-6
 return dict(max_normalized_endpoint_error=largest,edges=total)
def common_pair(a,b):
 subjects=sorted({r['subject'] for r in a});aa={r['sample_id']:r for r in a};bb={r['sample_id']:r for r in b};assert aa.keys()==bb.keys()
 def group(rows):return [{k:sum(r[k] for r in rows.values() if r['subject']==s) for k,v in next(iter(rows.values())).items() if isinstance(v,(float,int)) and k not in ['common_guide_distance_mm','common_tortuosity_error']} for s in subjects]
 ga,gb=group(aa),group(bb);sa,sb=aggregate_common_vessels(ga),aggregate_common_vessels(gb);fields=['common_branch_f1','common_curve_f1'];values=[];rng=np.random.default_rng(261026)
 for idx in rng.integers(len(subjects),size=(2000,len(subjects))):
  x=aggregate_common_vessels([ga[i] for i in idx]);y=aggregate_common_vessels([gb[i] for i in idx]);values.append([y[k]-x[k] for k in fields])
 values=np.asarray(values)
 return {k:dict(delta=sb[k]-sa[k],ci95=np.quantile(values[:,j],[.025,.975]).tolist()) for j,k in enumerate(fields)}
def publish(section):
 """Read latest report under lock; replace only this study's bounded section."""
 doc=ORIGINAL/'docs/research/metrics/3-ixi-node-refinement-pilot.md';start='<!-- BEGIN IXI LOCAL COMPONENTS 20261026 -->';end='<!-- END IXI LOCAL COMPONENTS 20261026 -->'
 with (META/'report-append.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);text=doc.read_text();block=start+'\n'+section+'\n'+end
  if start in text:
   assert text.count(start)==text.count(end)==1
   a=text.index(start);b=text.index(end)+len(end);updated=text[:a]+block+text[b:]
  else:updated=text.rstrip()+'\n\n'+block+'\n'
  # Avoid clobbering a concurrent writer that does not use this lock.
  assert doc.read_text()==text,'Report changed concurrently; results retained, rerun collector for append'
  with doc.open('w') as f:f.write(updated)

def collect():
 p=plan();root=Path(p['output_root']);gate=read(root/'smoke.json');final=gate['updates'];result=dict(plan=p,gate=gate,runs={},paired={},operating={},interventions={},zero_shot={});allrows={};common={};matched={};proofs=[]
 for arm in ARMS:
  path=root/'runs'/arm;done=read(path/'complete.json');assert done['completed'] and done['steps']==final;proofs.append(done['batch_sha256'])
  folder=path/f'step-{final}';allrows[arm]=rows(folder);result['runs'][arm]=dict(complete=done,summary=read(folder/'summary.json'),reconstruction=read(path/f'step-{final}-reconstruction/summary.json'),endpoints=endpoints(path/f'step-{final}-reconstruction'))
  common[arm]=[dict(r,subject=p['dataset']['subjects'][r['sample_id'].split('_')[1]]) for r in read(path/f'step-{final}-reconstruction/validation-000000/common-v2-per-patch.json')]
 assert len(set(proofs))==1,'Training data/augmentation stream mismatch'
 counts={rec['sample']:int((c['scores']>.5).sum()) for rec,c in snapshot_caches(root/'runs/B0'/f'step-{final}') if rec['split']=='val'}
 for arm in ARMS:result['operating'][arm],matched[arm]=operating(root,arm,final,counts)
 comparisons=[('B0',a) for a in ARMS if a!='B0']+[('A-xyz-fresh','A-xyz-transfer'),('A-xyz-fresh','A'),('A-cls','A'),('A-control','A')]
 for a,b in comparisons:
  result['paired'][b+'-minus-'+a]=dict(physical={tol:paired(selected(allrows[a],tol=tol),selected(allrows[b],tol=tol)) for tol in ['1.0','2.0']},matched_count={tol:paired(matched[a][tol],matched[b][tol]) for tol in ['1.0','2.0']},common=common_pair(common[a],common[b]))
 for source in ['B0','A-xyz','A']:
  path=root/'runs'/('zero-'+source);assert read(path/'complete.json')['completed'];result['zero_shot'][source]={f.name:read(f/'summary.json') for f in path.iterdir() if f.is_dir() and (f/'summary.json').exists()}
 for arm in ARMS[1:]:
  result['interventions'][arm]={}
  path=root/'runs'/arm
  reference=root/'runs'/arm/f'step-{final}'
  for f in sorted(path.glob('final-*')):
   if not (f/'rows.json').exists():continue
   rr=rows(f);ids={r['sample'] for r in rr};aa=[r for r in allrows[arm] if r['sample'] in ids];changes=[]
   for rec,c in snapshot_caches(f):
    original=load_cache(reference,rec['sample']);assert np.array_equal(original['pairs'],c['pairs']);assert np.max(np.abs(original['relations']-c['relations']))==0
    changes.append(dict(displacement_mm=float(np.linalg.norm(c['physical']-original['physical'],axis=-1).mean()),confidence_change=float(np.abs(c['scores']-original['scores']).mean())))
    if arm.startswith('A-xyz'):assert np.array_equal(c['scores'],original['scores'])
    if arm=='A-cls':assert np.array_equal(c['physical'],original['physical'])
   result['interventions'][arm][f.name]=dict(paired={tol:paired(selected(aa,tol=tol),selected(rr,tol=tol)) for tol in ['1.0','2.0']},prediction_changes={k:float(np.mean([v[k] for v in changes])) for k in changes[0]})
 # Operational recommendation requires paired full-graph and matched-count support, not just a point leader.
 support={}
 for a in ARMS[1:]:
  z=result['paired'][a+'-minus-B0'];support[a]=dict(edge_ci_positive=z['physical']['2.0']['metrics']['edge_f1']['ci95'][0]>0,matched_count_edge_ci_positive=z['matched_count']['2.0']['metrics']['edge_f1']['ci95'][0]>0,curve_nonnegative=z['common']['common_curve_f1']['delta']>=0)
 result['decision_support']=support
 result['learning_trajectory']={}
 for arm in ARMS:
  result['learning_trajectory'][arm]={}
  for step in gate['checkpoints']:
   rr=rows(root/'runs'/arm/f'step-{step}');ids=set(p['dataset']['monitor_val'])
   result['learning_trajectory'][arm][str(step)]={tol:stats([r for r in selected(rr,tol=tol) if r['sample'] in ids]) for tol in ['1.0','2.0']}
 write(root/'analysis.json',result)
 section='### Completed IXI component results —26 October2026\n\nCompleted bounded fixed-budget development comparison. Full reproducible numeric results: `'+str(root/'analysis.json')+'`. One fine-tuning seed; uncertainty resamples validation subjects, not source initializations.\n\n'
 section+='| Arm | Node F1 1 mm | Raw /filtered recall 2 mm | Node F1 2 mm | Edge F1 2 mm | Branch F1 | Curve F1 |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for a in ARMS:
  v=result['runs'][a]['summary']['metrics']['val'];g=result['runs'][a]['reconstruction'];section+=f"| {a} | {v['1.0']['node_f1']:.5f} | {v['2.0']['raw_recall']:.5f} /{v['2.0']['node_recall']:.5f} | {v['2.0']['node_f1']:.5f} | {v['2.0']['edge_f1']:.5f} | {g['common_branch_f1']:.5f} | {g['common_curve_f1']:.5f} |\n"
 section+='\n| Comparison | Edge F1 2-mm delta [95% subject interval] | Curve F1 delta [95% subject interval] |\n|---|---:|---:|\n'
 for name,r in result['paired'].items():
  def cell(x):return f"{x['delta']:+.5f} [{x['ci95'][0]:+.5f}, {x['ci95'][1]:+.5f}]"
  section+=f"| {name} | {cell(r['physical']['2.0']['metrics']['edge_f1'])} | {cell(r['common']['common_curve_f1'])} |\n"
 best=max(ARMS,key=lambda a:result['runs'][a]['summary']['metrics']['val']['2.0']['edge_f1'])
 section+=f'\nThe descriptive final edge-F1 leader is **{best}**. This does not alone select the method: consult matched-count/calibrated policies, confidence ranking, duplicate/topology errors, Stage1 ablations and paired intervals in `analysis.json`. Longer training is not automatically authorized. Attribution: fresh xyz versus B0 tests localization; fresh cls versus B0 tests classification; combined versus each single component tests interaction; transferred versus fresh xyz isolates head transfer on an identical shared initializer. Native zero-shot checkpoints have co-adapted trunks and are a separate comparison.\n'
 supported=[a for a,v in support.items() if all(v.values())]
 section+='\nPaired edge improvement at both default and matched counts, without a negative Curve-F1 point difference, is supported for: **'+(', '.join(supported) if supported else 'none of the tested refinement arms')+'**. These are pilot criteria, not proof of topology improvement. Classification is only an added supported benefit if its fresh arm beats B0 and combined A improves over coordinate-only under the same controls. Transfer is assessed separately by transferred-minus-fresh xyz. Learning trajectories on the same128 validation patches are saved; any longer-training decision must consider plateau, training/validation gap and topology rather than select a favorable checkpoint after the fact. No extension is automatically submitted.\n'
 section+='\nAll arms share original IXI targets/splits, node-focal gamma2 plus edge-CE ratio0.15, K8 geometry and a fully trainable backbone/transformer/prediction stack. Fine-tuning uses a shared 256-query expansion; zero-shot retains native120. All reconstruction and raw-query evaluation uses batch1 consistently; cross-batch drift and same-batch repeat checks are recorded by the smoke gate. Graph metrics retain their original physical/common-v2 definitions. No test data, new representation or edge focal was introduced. Exact protocol and submissions remain in the worktree experiment directory.\n'
 (root/'results.md').write_text(section);publish(section);write(root/'collection-complete.json',dict(completed=True,analysis=str(root/'analysis.json')))
 print(section,flush=True)
if __name__=='__main__':collect()
