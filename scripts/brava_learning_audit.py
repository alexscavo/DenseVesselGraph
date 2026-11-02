"""BRAVA K8 monitoring/cohort audit; no geometry targets or training changes."""
import collections,csv,hashlib,io,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
ORIGIN=Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/Graph-Native-Betti-Matching-Representation-BRAVA-20261026')
PREVIOUS=ORIGIN/'experiments/brava_representation_20261026'
OUT=Path(__file__).resolve().parents[1]/'experiments/brava_learning_diagnosis_20261026'
LARGE=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/brava_learning_diagnosis_20261026')
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
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def prepare():
 p=read(PREVIOUS/'plan.json');audit=read(PREVIOUS/'target-audit.json')['rows'];lookup={r['sample']:r for r in audit};subject=lambda s:p['subjects'][s.split('_')[1]]
 monitors=[];calibration=[];strata=[]
 for subj in p['subject_groups']['train']:
  ids=sorted([s for s in p['ids']['train'] if subject(s)==subj],key=lambda s:(lookup[s]['nodes'],lookup[s]['edges'],hashlib.sha256(s.encode()).hexdigest()))
  monitors.extend(ids[i] for i in [3,11,19,27]);calibration.extend(ids[i] for i in [1,5,9,13,17,21,25,29])
  strata.extend(dict(sample=s,subject=subj,complexity_quartile=i//8,**{k:lookup[s][k] for k in ['nodes','edges','border_nodes','components']}) for i,s in enumerate(ids))
 # Round-robin subject/complexity order; add four fixed patches to get full batches of eight.
 extra=[];stress_subjects=set()
 for s in sorted(set(p['ids']['train'])-set(monitors),key=lambda s:(-lookup[s]['nodes'],-lookup[s]['edges'],hashlib.sha256(s.encode()).hexdigest())):
  if subject(s) in stress_subjects:continue
  extra.append(s);stress_subjects.add(subject(s))
  if len(extra)==4:break
 monitors.extend(extra)
 order=lambda s:hashlib.sha256(('monitor-order:'+s).encode()).hexdigest()
 monitors=sorted(monitors,key=order);calibration=sorted(calibration,key=order)
 assert len(monitors)==160 and len(calibration)==312 and len(set(monitors))==160
 assert len({subject(s) for s in monitors})==39 and not {subject(s) for s in monitors}&{subject(s) for s in p['ids']['val']}
 def stats(ids):
  return dict(patches=len(ids),subjects=len({subject(s) for s in ids}),nodes_mean=np.mean([lookup[s]['nodes'] for s in ids]),nodes_quantiles=np.quantile([lookup[s]['nodes'] for s in ids],[0,.25,.5,.75,1]),edges_mean=np.mean([lookup[s]['edges'] for s in ids]),border_fraction=sum(lookup[s]['border_nodes'] for s in ids)/sum(lookup[s]['nodes'] for s in ids))
 populations=dict(representative_train=monitors,validation=p['ids']['val'],tiny_reference=p['tiny_ids'],old_monitor=p['monitor_train'])
 plan=dict(source_plan=str(PREVIOUS/'plan.json'),source_plan_sha256=sha(PREVIOUS/'plan.json'),dataset=p['dataset'],subjects=p['subjects'],populations=populations,stress_ids=extra,calibration_ids=calibration,statistics={k:stats(v) for k,v in dict(populations,training_population=p['ids']['train']).items()},seed=261026,steps=[1000,2000,4000],batch_size=8,calibration_batch_size=8,calibration_batches=39,thresholds=[.5,.5],tolerances_mm=[1,2],checkpoint=str(PREVIOUS/'runs/original/model-004000.pt'),tiny_checkpoint=str(PREVIOUS/'tiny/original/model-000512.pt'),large_outputs=str(LARGE),hypotheses=['representative training failure','batch-statistics/mode effects','confidence-selection failure','augmentation/annotation consistency'],no_training=True)
 write(OUT/'plan.json',plan);write(OUT/'complexity-strata.json',strata)
 final=read(PREVIOUS/'evaluation/original/004000/summary.json')['K8']['val']['physical']['2.0'];assert abs(final['node_f1']-.11830477908025248)<1e-9
 write(OUT/'symptom-replay.json',dict(node_f1=final['node_f1'],edge_f1=final['edge_f1'],functional_detector_gate=False,tiny_node_f1=read(PREVIOUS/'tiny-evaluation/original/000512/summary.json')['K8']['train']['physical']['2.0']['node_f1']))
 print('Cohort prepared:',plan['statistics'],flush=True)

def cache_audit():
 """Reuse validation/training old-monitor caches and image metadata, no inference."""
 import nibabel as nib
 p=read(OUT/'plan.json');source=read(PREVIOUS/'plan.json');root=Path(p['dataset']);lookup={r['sample_id']:r for r in csv.DictReader((root/'patch_index.csv').open())};rows=[];images={};summary=[]
 for split,ids in source['ids'].items():
  for i,s in enumerate(ids):
   rr=lookup[s];path=root/split/'raw'/(s+'_data.nii.gz')
   if not path.exists():path=next(path.parent.glob(s+'*.nii.gz'))
   a=np.asarray(nib.load(path).dataobj,dtype=np.float32);key=hashlib.sha256(a.tobytes()).hexdigest();images.setdefault(key,[]).append(s)
   rows.append(dict(sample=s,subject=p['subjects'][s.split('_')[1]],split=split,node_count=int(rr['anchor_count']),edge_count=int(rr['branch_count']),image_mean=float(a.mean()),image_std=float(a.std()),image_p01=float(np.quantile(a,.01)),image_p50=float(np.quantile(a,.5)),image_p99=float(np.quantile(a,.99)),nonzero_fraction=float((a!=0).mean()),sha256=key))
 groups=[v for v in images.values() if len(v)>1]
 # Identical array inputs with inconsistent node/edge targets are direct contradictions.
 identical=[]
 for ids in groups:
  targets=[]
  for s in ids:
   split=lookup[s]['split']
   with np.load(root/split/'spline'/(s+'.npz')) as g:targets.append(hashlib.sha256(g['nodes'].tobytes()+g['edges'].tobytes()).hexdigest())
  identical.append(dict(samples=ids,target_hashes=targets,conflicting_targets=len(set(targets))>1))
 write(OUT/'image-audit.json',dict(rows=rows,identical_inputs=identical,brain_masks_used_as_vessel_support=False))
 history=read(PREVIOUS/'runs/original/training.json');write(OUT/'training-history.json',history)
 write(OUT/'annotation-audit.json',dict(segmentation_role='brain mask only; never vascular supervision or foreground oracle',incomplete_centerlines=True,identical_image_groups=len(groups),conflicting_identical_input_groups=sum(r['conflicting_targets'] for r in identical),note='No exact duplicate does not rule out partial-annotation conflicts across overlapping crops/trace variants.'))
 print('Image/annotation audit completed',len(rows),len(groups),flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','images']);a=ap.parse_args()
 (prepare if a.mode=='prepare' else cache_audit)()
