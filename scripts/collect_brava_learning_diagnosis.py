"""CPU-only paired K8 diagnostics; append to existing BRAVA report and roadmap."""
import csv,gzip,hashlib,io,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score
from scripts.brava_learning_audit import OUT,LARGE,PREVIOUS,read,write
from scripts.node_verifier_metrics import association,measure,aggregate,discrimination

def load(path):
 with zipfile.ZipFile(path) as z:
  for name in z.namelist():
   with np.load(io.BytesIO(z.read(name))) as f:yield Path(name).stem,{k:f[k] for k in f.files}

def metrics(c,tol,threshold=.5):
 ids=np.flatnonzero(c['scores']>threshold);r=measure(c,ids,tol);raw=association(c['physical'],c['truth'],tol)
 m=association(c['physical'][ids],c['truth'],tol);mp={int(ids[q]):g for q,g in m.items()};q=len(c['scores']);target=np.full(q,-1,int)
 for a,b in mp.items():target[a]=b
 adj=np.zeros((len(c['truth']),)*2,bool);a,b=c['gt_edges'].T;adj[a,b]=adj[b,a]=True
 a,b=c['pairs'].T;active=np.isin(a,ids)&np.isin(b,ids);both=(target[a]>=0)&(target[b]>=0);labels=both&adj[np.maximum(target[a],0),np.maximum(target[b],0)];chosen=active&(c['relations']>.5)
 r.update(raw_tp=len(raw),hungarian_sum_mm=float(c['hungarian_distance'].sum()),hungarian_count=len(c['hungarian_distance']),hungarian_within1=int((c['hungarian_distance']<=1).sum()),hungarian_within2=int((c['hungarian_distance']<=2).sum()),false_edges=int((chosen&~labels).sum()),false_edges_spatial_unmatched=int((chosen&~both).sum()),localization_sum_mm=float(sum(np.linalg.norm(c['physical'][a]-c['truth'][b]) for a,b in mp.items())))
 if 'hungarian_source' in c:
  assigned=np.zeros(q,bool);assigned[c['hungarian_source']]=True;hp=assigned[a]&assigned[b]
  r.update(training_pool_pairs=int(hp.sum()),training_pool_positive_pairs=len(c['gt_edges']),training_pool_negative_pairs=int(hp.sum())-len(c['gt_edges']),retained_candidate_pairs=int(active.sum()),retained_hungarian_unmatched_pairs=int((active&~hp).sum()),predicted_hungarian_unmatched_edges=int((chosen&~hp).sum()))
 return r,(labels[active&both],c['relations'][active&both]),(labels[active],c['relations'][active])

def summarize(rows):
 s=aggregate([{k:v for k,v in r.items() if k not in ['tolerance','step','condition','population','subject','sample']} for r in rows]);s.update(raw_recall=s['raw_tp']/s['gt_nodes'],hungarian_mm=s['hungarian_sum_mm']/s['hungarian_count'],hungarian_within2=s['hungarian_within2']/s['hungarian_count'],localization_mm=s['localization_sum_mm']/s['node_tp'] if s['node_tp'] else None,unmatched_false_edge_fraction=s['false_edges_spatial_unmatched']/s['false_edges'] if s['false_edges'] else None);return s

def paired_difference(left,right,field):
 subjects=sorted({r['subject'] for r in left});values=[]
 for rows in [left,right]:
  mat=[]
  for s in subjects:
   rr=[r for r in rows if r['subject']==s]
   if field=='node_f1':mat.append([sum(r['node_tp'] for r in rr)*2,sum(r['nodes']+r['gt_nodes'] for r in rr)])
   elif field=='edge_f1':mat.append([sum(r['edge_tp'] for r in rr)*2,sum(r['edges']+r['gt_edges'] for r in rr)])
   else:mat.append([sum(r['raw_tp'] for r in rr),sum(r['gt_nodes'] for r in rr)])
  values.append(mat)
 v=np.array(values,float);rng=np.random.default_rng(261026);idx=rng.integers(len(subjects),size=(1000,len(subjects)));d=v[:,idx].sum(2);metric=d[:,:,0]/np.maximum(d[:,:,1],1);delta=metric[1]-metric[0]
 return dict(delta=float(v[1,:,0].sum()/v[1,:,1].sum()-v[0,:,0].sum()/v[0,:,1].sum()),ci95=np.quantile(delta,[.025,.975]).tolist(),subjects=len(subjects),draws=1000)

def main():
 complete=read(OUT/'complete.json');assert complete['completed'];p=read(OUT/'plan.json');rows=[];summaries=[];assignments={};allc={};ranking=[];batchrecords=[]
 for path in sorted(LARGE.glob('*/*/*.zip')):
  step,condition,pop=path.parent.parent.name,path.parent.name,path.stem;caches=[];meta=dict(step=step,condition=condition,population=pop)
  for sample,c in load(path):
   subject=p['subjects'][sample.split('_')[1]];caches.append(c);hm=dict(zip(c['hungarian_target'].tolist(),c['hungarian_source'].tolist()));assignments[(step,condition,pop,sample)]=hm
   for tol in [1.,2.]:
    r,cond,ret=metrics(c,tol);r.update(meta,sample=sample,subject=subject,tolerance=tol);rows.append(r)
    for kind,(label,score) in [('conditional',cond),('all_retained',ret)]:ranking.append(dict(meta,sample=sample,subject=subject,tolerance=tol,kind=kind,labels=label,scores=score))
  allc[(step,condition,pop)]=caches
  for tol in [1.,2.]:
   rr=[r for r in rows if all(r[k]==v for k,v in meta.items()) and r['tolerance']==tol];result=dict(meta,tolerance=tol,**summarize(rr))
   for kind in ['conditional','all_retained']:
    rrk=[r for r in ranking if all(r[k]==v for k,v in meta.items()) and r['tolerance']==tol and r['kind']==kind];y=np.concatenate([r['labels'] for r in rrk]);score=np.concatenate([r['scores'] for r in rrk]);result[kind+'_ap']=float(average_precision_score(y,score)) if y.any() else None;result[kind+'_candidates']=len(y);result[kind+'_positives']=int(y.sum())
   summaries.append(result)
  batchrecords.extend(dict(meta,**r) for r in read(path.with_name(pop+'-records.json')))
 confidence=[dict(step=k[0],condition=k[1],population=k[2],**discrimination(c,[v['scores'] for v in c])) for k,c in allc.items()]
 diffs=[]
 for population in ['representative_train','validation','tiny_reference']:
  for condition in ['saved_b1','batch_bn_b8','recal_train_b8','recal_reverse_b8']:
   a=[r for r in rows if r['step']=='4000' and r['condition']=='saved_b8' and r['population']==population and r['tolerance']==2];b=[r for r in rows if r['step']=='4000' and r['condition']==condition and r['population']==population and r['tolerance']==2]
   if a and b:diffs.append(dict(population=population,condition=condition,**{f:paired_difference(a,b,f) for f in ['node_f1','edge_f1','raw_recall']}))
 churn=[]
 for population in ['representative_train','validation']:
  for a,b in [('1000','2000'),('2000','4000')]:
   values=[]
   for sample in p['populations'][population]:
    x=assignments[(a,'saved_b8',population,sample)];y=assignments[(b,'saved_b8',population,sample)];values.extend(x[g]!=y[g] for g in x)
   churn.append(dict(population=population,from_step=a,to_step=b,churn=float(np.mean(values)),gt_assignments=len(values)))
  for condition in ['saved_b1','batch_bn_b8','recal_train_b8']:
   values=[]
   for sample in p['populations'][population]:
    x=assignments[('4000','saved_b8',population,sample)];y=assignments[('4000',condition,population,sample)];values.extend(x[g]!=y[g] for g in x)
   churn.append(dict(population=population,condition=condition,compared_to='saved_b8',churn=float(np.mean(values)),gt_assignments=len(values)))
 # Train-only coarse confidence operating point; relation cutoff stays .5.
 calibration=[]
 for condition in ['saved_b8','batch_bn_b8','recal_train_b8']:
  train=allc[('4000',condition,'representative_train')];val=allc[('4000',condition,'validation')];sweep=[]
  for threshold in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:sweep.append(dict(threshold=threshold,**summarize([metrics(c,2.,threshold)[0] for c in train])))
  chosen=max(sweep,key=lambda r:(r['node_f1'],r['threshold']));calibration.append(dict(condition=condition,policy='training_grid_max_node_f1',threshold=chosen['threshold'],training=chosen,validation=summarize([metrics(c,2.,chosen['threshold'])[0] for c in val]),sweep=sweep))
 result=dict(completed=True,protocol=p,summaries=summaries,confidence=confidence,differences=diffs,assignment_churn=churn,calibration=calibration,smoke=read(OUT/'smoke.json'),inference=complete)
 write(OUT/'analysis.json',result);write(OUT/'per-patch.json',rows);write(OUT/'losses-by-batch.json',batchrecords)
 # Per-subject results retain the bootstrap unit and complexity counts.
 per_subject=[]
 for meta in summaries:
  selected=[r for r in rows if all(r[k]==meta[k] for k in ['step','condition','population','tolerance'])]
  for subject in sorted({r['subject'] for r in selected}):per_subject.append(dict(**{k:meta[k] for k in ['step','condition','population','tolerance']},subject=subject,**summarize([r for r in selected if r['subject']==subject])))
 write(OUT/'per-subject.json',per_subject)
 publish(result)
 print('BRAVA learning diagnostics collected',flush=True)

def publish(s):
 # Separate experiment evidence from a causal claim about training.
 def get(pop,condition='saved_b8',step='4000',tol=2.):return next(r for r in s['summaries'] if r['population']==pop and r['condition']==condition and r['step']==step and r['tolerance']==tol)
 train=get('representative_train');val=get('validation');bn=get('validation','recal_train_b8');batch=get('validation','batch_bn_b8')
 meaningful_bn=max(bn['node_f1'],batch['node_f1'])-val['node_f1']>.1
 fitting_failure=train['node_f1']<.4 and train['edge_f1']<.2
 hypothesis='representative cohort fitting remains weak' if fitting_failure else 'training-to-validation transfer is the larger measured gap'
 if meaningful_bn:hypothesis+='; normalization/batch state also has a substantial measured effect'
 rows=['## Larger-cohort K8 learning diagnosis — 26 October 2026','',
 '**Scope:** existing original K8 checkpoints only, no representation/geometry change and no optimizer updates. All precision/false-edge statements are relative to incomplete SWC annotations. The `seg/` loader slot contains brain masks and is never vessel supervision or a vascular support oracle.','',
 f'Isolated code and compact results: `{OUT}`. Large prediction caches and BN-only states: `{LARGE}`. The source checkpoint, 1,248-patch/39-subject training population, 256-patch/eight-subject validation population, graph targets and 0.5 node/edge cutoffs remain unchanged.','',
 '### Representative populations and controls','',
 'The new training monitor contains 160 patches: four node/edge-complexity-stratified patches per subject, plus four high-complexity stress examples from distinct subjects. It covers all 39 training subjects and the full observed node-count range. The old 39-patch monitor and the exact 16-patch learnability subset remain separate references. Training-only BN calibration uses 312 patches (eight complexity-spread patches per training subject), 39 batches of eight, augmentation off. BN running means/variances are reset and cumulatively estimated; forward/reversed ordering are separate controls. This is standard layerwise minibatch-statistic recalibration, not an exact jointly pooled population variance.','',
 '| Population | Patches / subjects | Mean nodes | Mean edges | Boundary-anchor fraction |','|---|---:|---:|---:|---:|']
 for name,r in s['protocol']['statistics'].items():rows.append(f'| {name} | {r["patches"]}/{r["subjects"]} | {r["nodes_mean"]:.2f} | {r["edges_mean"]:.2f} | {r["border_fraction"]:.3f} |')
 rows+=['','Saved BN is evaluated at batch one and eight; training batch statistics are evaluated at the same batch eight and identical sample compositions. A smoke comparison checks actual `model.train()` against the isolated BN-only mode. Weights and non-BN buffers are hash-checked; temporary BN changes are restored. Calibration sees training images only, never validation data. The same 256 validation patches are retained throughout.','',
 '### Measured localization, selection and connectivity','',
 '| Checkpoint | Population / condition | Raw R2 | Filtered R2 | Node F1 | Edge F1 | Conditional relation recall / AP | Hungarian mm | β₀ / β₁ MAE |','|---|---|---:|---:|---:|---:|---|---:|---|']
 for step in ['1000','2000','4000']:
  for pop in ['representative_train','validation']:
   for cond in (['saved_b8','saved_b1','batch_bn_b8','recal_train_b8','recal_reverse_b8'] if step=='4000' else ['saved_b8']):
    r=get(pop,cond,step);ap='NA' if r['conditional_ap'] is None else f'{r["conditional_ap"]:.4f}'
    rows.append(f'| {step} | {pop} / {cond} | {r["raw_recall"]:.4f} | {r["node_recall"]:.4f} | {r["node_f1"]:.4f} | {r["edge_f1"]:.4f} | {r["conditional_recall"]:.4f} / {ap} | {r["hungarian_mm"]:.3f} | {r["beta0_mae"]:.3f} / {r["beta1_mae"]:.3f} |')
 rows+=['','The same queries, GT targets and original relation head are used within each checkpoint. BN/mode interventions change predictions through the unchanged model; they are not cached-score threshold manipulations. Training and validation 1-mm results, node precision, duplicates, all-retained AP and candidate prevalence are preserved in `analysis.json`.','',
 '| Tiny-reference checkpoint / condition | Raw R2 | Filtered R2 | Node F1 | Edge F1 |','|---|---:|---:|---:|---:|']
 for step in ['tiny512','4000']:
  for cond in ['saved_b1','saved_b8','batch_bn_b8']:
   r=get('tiny_reference',cond,step);rows.append(f'| {step} / {cond} | {r["raw_recall"]:.4f} | {r["node_recall"]:.4f} | {r["node_f1"]:.4f} | {r["edge_f1"]:.4f} |')
 rows+=['','### Confidence, assignment and supervision','',
 '| Population / condition | Correct-vs-far / duplicate confidence AUC | Mean correct / duplicate / unmatched confidence | Reliable Brier / ECE |','|---|---|---|---|']
 for r in s['confidence']:
  if r['step']!='4000' or r['population'] not in ['representative_train','validation'] or r['condition'] not in ['saved_b8','batch_bn_b8','recal_train_b8']:continue
  fmt=lambda v:'NA' if v is None else f'{v:.4f}'
  rows.append(f'| {r["population"]} / {r["condition"]} | {fmt(r["auc_correct_false"])} / {fmt(r["auc_correct_duplicate"])} | {fmt(r["correct_mean_confidence"])} / {fmt(r["duplicate_mean_confidence"])} / {fmt(r["false_mean_confidence"])} | {fmt(r["brier"])} / {fmt(r["ece"])} |')
 rows+=['','Confidence categories use the existing spatial canonical/reliable-subset rules, not anatomical truth. A coarse predeclared node-threshold grid {0.1,…,0.9} is selected solely on representative training node F1; validation receives that threshold unchanged, edge threshold stays 0.5. This is exploratory operating-point calibration, not a new detector.','',
 '| Condition | Training-selected node cutoff | Training node F1 | Validation node F1 / Edge F1 / recall |','|---|---:|---:|---|']
 for r in s['calibration']:rows.append(f'| {r["condition"]} | {r["threshold"]:.1f} | {r["training"]["node_f1"]:.4f} | {r["validation"]["node_f1"]:.4f} / {r["validation"]["edge_f1"]:.4f} / {r["validation"]["node_recall"]:.4f} |')
 rows+=['','Exact Hungarian query IDs and targets are cached. `assignment_churn` reports long-interval 1,000→2,000→4,000 changes and same-checkpoint batching/BN changes; these are not per-update instability estimates. The unchanged matcher uses classification cost 2 and normalized-coordinate L1 cost 5. Matched queries are not necessarily physically localized. The established relation sampler preserves all GT-positive edges, uses matched-query negatives with its existing cap and ratio-upsample balancing, and excludes unmatched-query mining. Final snapshot candidate counts are not a transcript of the stochastic pairs used at every training update.','',
 '### Augmentation and optimization checks','',
 f'The actual loader passed {s["smoke"]["augmentation"]["checks"]} checks over four real patches and all 64 quarter-turn compositions: image/node transformation, graph connectivity, endpoint correspondence and physical pair-distance preservation. Intensity noise is independent of target geometry. Actual noisy/rotated predictions on a fixed 32-training-patch probe are saved under `aug_saved_b8` / `aug_batch_bn_b8`, with clean counterparts and input hashes. Correct alignment does not prove augmentation is easy to optimize or innocuous for invariance learning.','',
    'Original loss/gradient snapshots are copied into `training-history.json`: seven sampled minibatches, not a densely measured convergence curve. Fixed-checkpoint graph losses on representative train/validation batches are retained in `losses-by-batch.json`; stochastic endpoint orientation and positive balancing are seeded. Tiny training had 256 sample draws per patch and no augmentation, versus about 25.64 draws per patch and augmentation in the larger run; its schedule also differed. This is a substantial comparison confound, not proof that duration alone explains failure. Spline objectives used the previously validated zero-geometry-to-detector `graph_only` route; K24/C1 therefore cannot explain detector gradients in this reference. Annotation/image audit is in `annotation-audit.json` and `image-audit.json`; exact repeated inputs with different graph targets are counted, but lack of exact duplicates cannot rule out missing annotations or overlapping-trace contradictions.','',
 '### Interpretation and next decision','',
 f'**Main measured diagnosis: {hypothesis}.** Representative training node/edge F1 at saved-BN batch eight are {train["node_f1"]:.4f}/{train["edge_f1"]:.4f}, versus {val["node_f1"]:.4f}/{val["edge_f1"]:.4f} on validation. Raw train/validation recall is {train["raw_recall"]:.4f}/{val["raw_recall"]:.4f}; confidence filtering adds another loss. Train-only BN recalibration yields validation {bn["node_f1"]:.4f}/{bn["edge_f1"]:.4f}, while current-batch BN yields {batch["node_f1"]:.4f}/{batch["edge_f1"]:.4f}. These interventions distinguish saved statistics from batch-conditioned behavior; none demonstrates that BRAVA annotations are complete.','',
 'Do not label failure as pure held-out-subject overfitting when representative optimized-on patches are also poor. Conversely, training-batch improvements alone do not establish that recalibration repairs generalization. No single cause is declared from tiny-set success or a decreasing aggregate loss.','',
 ('**Recommended next single training comparison, not launched:** repeat original K8 with the same source checkpoint, 1,248 patches, patch order, seed and fixed 4,000 updates, changing only spatial/intensity augmentation off versus the unchanged augmentation-on control. This is justified only after reviewing the table above: tiny learnability and larger training differ in both population and augmentation; the frozen probes test alignment/robustness but cannot identify augmentation’s optimization effect. Keep losses, optimizer, LR, BN training policy and evaluation identical. Treat it as a learning-curve diagnostic, not a geometry comparison.' if fitting_failure and not meaningful_bn else '**Training decision:** first review the normalization and representative train/validation comparisons above. No additional training was submitted. A training change is justified only if the measured inference/BN interventions leave a specific unresolved optimization question; do not add a geometry sweep.'),'',
 'Original K8 remains the reference. K24, fixed C1, adaptive piecewise geometry and degree-2 subdivision are not promoted while graph detection remains weak.','',
 'Whole-subject paired intervals (1,000 draws, seed 261026) for mode/BN differences are in `analysis.json`; per-patch and per-subject tables preserve denominators and topology errors. These quantify development-subject uncertainty, not training-seed robustness. No validation threshold tuning, test inspection, optimizer update or new pretraining was performed.','',
 'Print completed results from the parent `projects` folder:','',
 '```bash','python3 gnbm-brava-learning-diagnosis-20261026/scripts/show_brava_learning_diagnosis.py','```','']
 audit=read(OUT/'publication-audit.json');publish_block(audit['report'],'BRAVA LEARNING DIAGNOSIS 20261026','\n'.join(rows),'report')
 roadmap=f'''## BRAVA detector diagnosis update — 26 October 2026

Completed the original-K8 representative learning/BN/augmentation diagnostic; see [BRAVA report](../docs/research/BRAVA_experiment_plan.md). New monitor: 160 training patches/39 subjects with complexity strata and 256 validation patches/eight subjects, plus the unchanged tiny reference. Measured finding: **{hypothesis}**. Saved-BN batch-eight train/validation node F1: {train['node_f1']:.4f}/{val['node_f1']:.4f}; validation edge F1 {val['edge_f1']:.4f}. Train-only recalibration node/edge F1: {bn['node_f1']:.4f}/{bn['edge_f1']:.4f}. No new training was launched. Keep K8 as the reference; geometry-capacity and subdivision claims remain gated on functional detection. Follow the report’s single-variable next-experiment recommendation; do not infer anatomical absence from incomplete SWC annotations or use brain masks as vessel labels.
'''
 publish_block(audit['roadmap'],'BRAVA LEARNING PRIORITY 20261026',roadmap,'roadmap')

def publish_block(audit,marker,section,name):
 path=Path(audit['path']);old=path.read_text();start='<!-- BEGIN '+marker+' -->';end='<!-- END '+marker+' -->';block=start+'\n'+section+'\n'+end
 if hashlib.sha256(old.encode()).hexdigest()!=audit['sha256'] or start in old:
  write(OUT/(name+'-integration-pending.json'),dict(target=str(path),reason='Concurrent change; preserve report and stage appendix',section=block));return
 assert path.read_text()==old;path.write_text(old.rstrip()+'\n\n'+block+'\n');write(OUT/(name+'-publication.json'),dict(target=str(path),completed=True,after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

if __name__=='__main__':main()
