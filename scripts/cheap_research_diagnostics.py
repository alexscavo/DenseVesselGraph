"""Independent cache-only relation learning-curve and train-only BN diagnostics."""
import argparse,copy,csv,hashlib,io,json,os,re,sys,tarfile,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
META=Path('experiments/cheap_diagnostics_20261026')
ORIGINAL=Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/Graph-Native-Betti-Matching-Representation')
RELATION=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_image_relation_learning_20261020-113444')
BN_ROOT=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/cheap_diagnostics_20261026')
ARMS=['token_geometry','endpoints','chord','corridor']
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def table(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def probability(m):
 import numpy as np
 return 1/(1+np.exp(-np.clip(m,-80,80)))
def thresholds(scores,y,total,target):
 """Existing strict-greater/tied-group policy; no validation tuning."""
 import numpy as np
 if not len(scores):return {'max_f1':1.,'preserve_precision':1.}
 order=np.argsort(-scores,kind='stable');p=scores[order];y=y[order]
 end=np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1];tp=np.cumsum(y)[end];count=end+1;cut=np.nextafter(p[end].astype(float),-np.inf)
 best=int(np.argmax(2*tp/(count+total)));possible=np.flatnonzero((tp/count>=target)&(tp>0))
 conservative=int(possible[np.argmax(tp[possible])]) if len(possible) else None
 return dict(max_f1=float(cut[best]),preserve_precision=float(cut[conservative]) if conservative is not None else 1.)
def audit():
 # Distinguish executed calibration from normal training BN / mode probes / filenames.
 archive=Path('/lustre/fsstor/projects/rech/vnc/upz25mj/archives/2026-10-06/job-runs-2026-08.tar.gz');rows=[];count=0
 with tarfile.open(archive) as t:
  for m in t.getmembers():
   if not m.isfile() or '/logs/eval' not in m.name or not m.name.endswith('.out'):continue
   count+=1;raw=t.extractfile(m).read();s=raw.decode(errors='replace')
   if 'BN calibration (' not in s:continue
   progress=re.findall(r'BN calibration \((\d+) batches\):[^\r\n]*?\|\s*(\d+)/(\d+)',s)
   rows.append(dict(log=m.name,sha256=hashlib.sha256(raw).hexdigest(),requested=max((int(v[0]) for v in progress),default=None),observed_progress=max((int(v[1]) for v in progress),default=None),loader_batches=max((int(v[2]) for v in progress),default=None),saved_results='[OK] Saved' in s,calibration_source='validation loader, legacy implementation'))
 current=[]
 for path in [ORIGINAL/'experiments/local_refinement_long/plan.json',ORIGINAL/'experiments/fine_feature_architecture/plan.json',ORIGINAL/'experiments/multiscale_pretrained/plan.json']:
  p=read(path);cfg=p.get('config',{});current.append(dict(plan=str(path),bn_calibration_batches=cfg.get('evaluation',{}).get('bn_calibration_batches'),sha256=sha(path)))
 write(META/'bn-provenance-audit.json',dict(archive=str(archive),evaluations_scanned=count,enabled_legacy=rows,current_plans=current,notes=['Legacy calibration used validation images, not training-only.','Job762225 compared saved versus batch statistics, not post-hoc train-only recalibration.','BRAVA tiny-set train/eval BN comparison was not recalibration.','Current training-mode BN updates do not imply evaluation recalibration.']))
 print('BN audit:',len(rows),'legacy calibration logs of',count,'evaluations',flush=True)

def paired_ap(labels,score,subject,multiplicities,lookup):
 import numpy as np
 order=np.argsort(-score,kind='stable');y=labels[order];p=score[order];ids=lookup[subject[order]]
 end=np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1];out=[]
 for counts in multiplicities:
  w=counts[ids];tp=np.cumsum(w*y)[end];n=np.cumsum(w)[end]
  pr=np.divide(tp,n,out=np.zeros(len(tp),float),where=n>0)
  out.append(float(np.sum(np.diff(np.r_[0,tp])*pr)/tp[-1]) if tp[-1] else float('nan'))
 return np.asarray(out)

def relation():
 import numpy as np
 from sklearn.metrics import average_precision_score
 cache=RELATION/'cache';records=read(cache/'records.json')
 with np.load(cache/'common-pairs.npz') as f:c={k:f[k] for k in f.files}
 offsets=c['offsets'];sizes=np.diff(offsets);split=np.repeat([r['split'] for r in records],sizes)
 subjects,indices=np.unique([r['subject'] for r in records],return_inverse=True);pair_subject=np.repeat(indices,sizes)
 train=split=='train';val=split=='val';vsubjects=np.unique(pair_subject[val]);lookup=np.full(len(subjects),-1);lookup[vsubjects]=np.arange(len(vsubjects))
 assert not set(pair_subject[train])&set(pair_subject[val])
 rng=np.random.default_rng(191032);draws=1000;mult=np.array([np.bincount(x,minlength=len(vsubjects)) for x in rng.integers(len(vsubjects),size=(draws,len(vsubjects)))])
 candidates={'original':c['original_margin']};identities=[]
 # No architecture, loss, optimization or GPU inference is repeated.
 for arm in ARMS:
  run=RELATION/'runs'/arm;summary=read(run/'summary.json')
  assert summary['completed'] and summary['frozen_inputs_unchanged']
  for step in summary['checkpoints']:
   path=run/f'scores-{step:06d}.npz'
   with np.load(path) as f:
    np.testing.assert_array_equal(f['offsets'],offsets);np.testing.assert_array_equal(f['names'],[r['sample_id'] for r in records]);np.testing.assert_array_equal(f['splits'],[r['split'] for r in records]);np.testing.assert_array_equal(f['original_margin'],c['original_margin']);candidates[f'{arm}/{step}']=f['margin'].copy()
   identities.append(dict(path=str(path),sha256=sha(path)))
 rows=[];cis=[];graphs=[];probs={k:probability(v) for k,v in candidates.items()};gt={s:sum(len(r['gt_arc_lengths']) for r in records if r['split']==s) for s in ['train','val']}
 for tol in [1,2]:
  y=c[f'labels_{tol}'];conditional=c[f'conditional_{tol}'];boots={}
  for name,margin in candidates.items():
   for population,eligible in [('conditional',conditional),('all_retained',np.ones(len(y),bool))]:
    for phase in ['train','val']:
     mask=eligible&(split==phase);target=y[mask]
     rows.append(dict(scorer=name,tolerance_mm=tol,population=population,split=phase,candidates=int(mask.sum()),positives=int(target.sum()),prevalence=float(target.mean()),ap=float(average_precision_score(target,margin[mask]))))
   mask=val&conditional;boots[name]=paired_ap(y[mask],margin[mask],pair_subject[mask],mult,lookup)
   # Fixed decision and training-calibrated policies use all retained pairs.
   target=float(y[train&(probs['original']>.5)].sum()/max(1,(train&(probs['original']>.5)).sum()))
   policies={'fixed_0p5':.5,**thresholds(probs[name][train],y[train],gt['train'],target)}
   for policy,t in policies.items():
    for phase in ['train','val']:
     mask=split==phase;chosen=mask&(probs[name]>t);tp=int(y[chosen].sum());n=int(chosen.sum());total=gt[phase];eligible=int(y[mask].sum())
     graphs.append(dict(scorer=name,tolerance_mm=tol,policy=policy,threshold=t,split=phase,tp=tp,fp=n-tp,fn=total-tp,precision=tp/n if n else 0.,recall=tp/total,edge_f1=2*tp/(n+total),conditional_recall=tp/eligible if eligible else 0.,gt_edges=total,eligible_edges=eligible))
   print('relation cached scores',tol,name,flush=True)
  for name,values in boots.items():
   arm,_,step=name.partition('/');references=['original']+([f'token_geometry/{step}'] if step and arm!='token_geometry' else [])
   for reference in references:
    delta=values-boots[reference];lo,hi=np.quantile(delta,[.025,.975]);point=lambda n:next(r['ap'] for r in rows if r['scorer']==n and r['tolerance_mm']==tol and r['population']=='conditional' and r['split']=='val')
    cis.append(dict(scorer=name,reference=reference,tolerance_mm=tol,subjects=len(vsubjects),draws=draws,delta_ap=point(name)-point(reference),ci95_low=float(lo),ci95_high=float(hi)))
 table(META/'relation-learning-curves.csv',rows);table(META/'relation-paired-ap.csv',cis);table(META/'relation-operating-points.csv',graphs)
 write(META/'relation-analysis.json',dict(completed=True,source=str(RELATION),records=len(records),train_patches=int(sum(r['split']=='train' for r in records)),val_patches=int(sum(r['split']=='val' for r in records)),validation_subjects=len(vsubjects),score_identities=identities,ranking=rows,paired_intervals=cis,operating_points=graphs,limitations=['Same previously inspected development subjects; no new generalization set.','Early checkpoints were saved beforehand, but interpretation is retrospective; no validation-selected deployable checkpoint claim.','Subject intervals omit seed/multiple-comparison uncertainty.','Pair identities, labels, offsets and original margins verified exactly.']))


def baseline_cache(record):
 import numpy as np
 with np.load(record['cache']) as f,np.load(record['gt']) as gt:
  x=f['raw_nodes']@gt['metric'].T;y=gt['nodes']@gt['metric'].T
  # Geometry caches retain all raw nodes/confidences and every ORIGINAL
  # retained pair. Those scores exactly suffice at the preserved .5 node policy;
  # no missing rejected-query relation is needed for the saved-BN graph.
  pairs=f['raw_pairs'] if 'raw_pairs' in f else f['query_ids'][f['candidate_edges']]
  scores=f['raw_relation_scores'] if 'raw_relation_scores' in f else f['candidate_scores']
  return dict(physical=x,truth=y,gt_edges=gt['edges'],scores=f['raw_scores'],pairs=pairs,relations=scores,hungarian_distance=np.linalg.norm(x[f['hungarian_source']]-y[f['hungarian_target']],axis=-1))

def calibrate_training_only(model,loader,cfg,device,batches):
 """Reuse existing BN updater with an explicit augmentation-free TRAIN loader."""
 from training.evaluation import calibrate_batch_norm
 return calibrate_batch_norm(model,loader,cfg,device,batches=batches)

def bn(dataset):
 import numpy as np,torch
 from models import build_model
 from models.checkpoint import load_checkpoint_file
 from models.matcher import build_matcher
 from scripts.train_ixi_transfer import TransferCriterion
 from scripts.multidataset_graph_evaluation import loaders
 from training.engine import _move_graph_batch
 from scripts.multiscale_pretrained import graph_cache,rows_for,summarize
 from data.loaders import build_data_loaders
 from scripts.frozen_spline_diagnostic import training_evaluation_loader
 assert not (BN_ROOT/dataset/'complete.json').exists(),'Preserve completed BN diagnostics'
 p=read('experiments/multidataset_continuous/plan.json');source=p['checkpoints']['ixi_spline' if dataset=='IXI' else 'synthetic_spline'];assert sha(source['path'])==source['sha256']
 spec=p['datasets'][dataset];cp=load_checkpoint_file(source['path']);cfg=copy.deepcopy(cp['config']);cfg['data']['datasets']['synthetic_mri'].update(root=spec['root'],train_samples=None,validation_samples=None,patch_selection='all');cfg['data'].update(batch_size=8,validation_batch_size=1,train_augmentation=False);cfg['runtime']['workers']=0;cfg['tracking']['enabled']=False
 model=build_model(cfg).cuda().requires_grad_(False);model.load_state_dict(cp['net'],strict=True);del cp
 criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda()
 from scripts.node_verifier_feasibility import records
 record_map={r['sample_id']:r for r in records(dataset,True)}
 for name in spec['monitor_train']+spec['monitor_val']:assert name in record_map
 assert {record_map[s]['subject'] for s in spec['monitor_train']}.isdisjoint({record_map[s]['subject'] for s in spec['monitor_val']})
 baseline={name:baseline_cache(record_map[name]) for name in spec['monitor_val']};rows=[]
 for name,c in baseline.items():
  for tol,r in rows_for(c).items():rows.append(dict(r,sample=name,subject=record_map[name]['subject'],split='val',tolerance=tol,condition='saved'))
 model.eval();parity=[]
 for split,names,loader in loaders(cfg,p,dict(dataset=dataset),True):
  if split!='val':continue
  for n,batch in enumerate(loader):
   if n>=3:break
   v,t=_move_graph_batch(batch,'cuda','image',True)
   with torch.no_grad():tokens,pred,_=model(v);fresh=graph_cache(model,criterion,tokens,pred,t)
   cached=baseline[names[n]]
   score_matrix=np.full((len(fresh['scores']),len(fresh['scores'])),np.nan)
   score_matrix[fresh['pairs'][:,0],fresh['pairs'][:,1]]=fresh['relations'];score_matrix[fresh['pairs'][:,1],fresh['pairs'][:,0]]=fresh['relations']
   retrieved=score_matrix[cached['pairs'][:,0],cached['pairs'][:,1]];assert np.isfinite(retrieved).all()
   error=dict(position_mm=float(np.abs(fresh['physical']-cached['physical']).max()),confidence=float(np.abs(fresh['scores']-cached['scores']).max()),relation=float(np.abs(retrieved-cached['relations']).max(initial=0)))
   assert error['position_mm']<.003 and error['confidence']<.0002 and error['relation']<3e-6,error
   parity.append(error)
 # Named-parameter checksum excludes BN running buffers by design; also check all non-BN buffers.
 def digest(values):
  h=hashlib.sha256()
  for name,v in values:h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 weight_before=digest(model.named_parameters());bn_buffers={n for prefix,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for n in [prefix+'.running_mean',prefix+'.running_var',prefix+'.num_batches_tracked']}
 other_before=digest((n,v) for n,v in model.named_buffers() if n not in bn_buffers)
 saved={n:v.clone() for n,v in model.named_buffers() if n in bn_buffers}
 train,_=build_data_loaders(cfg);calcfg=copy.deepcopy(cfg);calcfg['data']['batch_size']=8;calcfg['data']['validation_batch_size']=8
 cal=training_evaluation_loader(train,calcfg,spec['monitor_train']);assert len(cal.dataset)==256 and cal.batch_size==8
 assert not cal.dataset.augment
 assert [r.sample_id for r in cal.dataset.records]==spec['monitor_train']
 torch.manual_seed(261026);consumed=calibrate_training_only(model,cal,cfg,'cuda',32)
 assert consumed==32 and weight_before==digest(model.named_parameters())
 assert other_before==digest((n,v) for n,v in model.named_buffers() if n not in bn_buffers)
 assert not model.training
 out=BN_ROOT/dataset;out.mkdir(parents=True,exist_ok=True)
 torch.save({n:v.detach().cpu() for n,v in model.named_buffers() if n in bn_buffers},out/'recalibrated-bn-only.pt')
 changes={n:float((v-saved[n]).float().square().mean().sqrt()) for n,v in model.named_buffers() if n in bn_buffers}
 with zipfile.ZipFile(out/'recalibrated-caches.zip','w',compression=zipfile.ZIP_STORED) as archive:
  for split,names,loader in loaders(cfg,p,dict(dataset=dataset),True):
   if split!='val':continue
   for i,batch in enumerate(loader):
    v,t=_move_graph_batch(batch,'cuda','image',True)
    with torch.no_grad():tokens,pred,_=model(v);c=graph_cache(model,criterion,tokens,pred,t)
    for tol,r in rows_for(c).items():rows.append(dict(r,sample=names[i],subject=record_map[names[i]]['subject'],split='val',tolerance=tol,condition='train_recalibrated'))
    stream=io.BytesIO();np.savez_compressed(stream,**{k:v for k,v in c.items() if isinstance(v,np.ndarray)});archive.writestr(names[i]+'.npz',stream.getvalue())
 write(out/'rows.json',rows);write(out/'complete.json',dict(completed=True,dataset=dataset,checkpoint=source,checkpoint_bn_recalibration_setting=cfg['evaluation'].get('bn_calibration_batches'),calibration_split='train',calibration_samples=spec['monitor_train'],evaluation_samples=spec['monitor_val'],calibration_batches=consumed,calibration_batch_size=8,inference_batch_size=1,weights_unchanged=True,other_buffers_unchanged=True,baseline_cache_parity=parity,bn_changes=changes,job_id=os.getenv('SLURM_JOB_ID'),summary={condition:{tol:summarize([r for r in rows if r['condition']==condition and r['tolerance']==tol]) for tol in ['1.0','2.0']} for condition in ['saved','train_recalibrated']}))
 print('BN comparison completed',dataset,flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','relation','bn']);parser.add_argument('--dataset',choices=['IXI','SyntheticMRI']);args=parser.parse_args();META.mkdir(parents=True,exist_ok=True)
 if args.mode=='audit':audit()
 elif args.mode=='relation':relation()
 else:
  import torch
  torch.set_num_threads(1);bn(args.dataset)
if __name__=='__main__':main()
