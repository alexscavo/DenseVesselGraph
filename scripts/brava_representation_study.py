"""Bounded fully trainable BRAVA graph/geometry study in an isolated worktree."""
import argparse,copy,hashlib,io,json,os,pickle,random,time,zipfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.analyze_spline_target_capacity import write
from scripts.multiscale_pretrained import graph_cache,rows_for,summarize
from data.brava_representation_study import RepresentationDataset,migrate_geometry

ROOT=Path('experiments/brava_representation_20261026')
ORIGIN=Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph')
SOURCE=ORIGIN/'experiments/fine_feature_architecture/runs/baseline/best.pt'
SEED=261039
ARMS=['K8','K24','C1','split_K8']

def read(p):return json.loads(Path(p).read_text())
def seed(n=0):random.seed(SEED+n);np.random.seed(SEED+n);torch.manual_seed(SEED+n)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def prepare():
 assert not (ROOT/'plan.json').exists()
 old=read('experiments/brava_graph_study/plan.json');base=read('experiments/fine_feature_architecture/plan.json')['config']
 subjects={sample.split('_')[1]:subject for sample,subject in old['subjects_by_id'].items()};groups={};chosen={}
 for split in ['train','val']:
  grouped={}
  for s in old['ids'][split]:grouped.setdefault(subjects[s.split('_')[1]],[]).append(s)
  groups[split]=sorted(grouped)
  chosen[split]=sorted(s for subject,ids in grouped.items() for s in sorted(ids,key=lambda x:hashlib.sha256(f'{SEED}:{x}'.encode()).hexdigest())[:32])
 assert not set(groups['train'])&set(groups['val'])
 base['data']['datasets']={'synthetic_mri':dict(base['data']['datasets']['synthetic_mri'],root=old['dataset'],train_samples=None,validation_samples=None,patch_selection='all')}
 base['data'].update(batch_size=8,validation_batch_size=1,train_augmentation=True)
 base['runtime']['workers']=0;base['tracking']['enabled']=False;base['experiment'].update(seed=SEED,name='brava_representation')
 assert base['loss']['node']['classification']['name']=='focal' and base['loss']['node']['classification']['focal_gamma']==2
 assert base['loss']['edge']['classification']['name']=='cross_entropy'
 plan=dict(config=base,source=str(SOURCE),source_sha256='d4bc78ad3eb7c1c29c9be8ec3e2927e88f2f45e6f34a3cb8be9ade5fd9d2431c',source_selected_epoch=95,
  source_plan='experiments/brava_graph_study/plan.json',dataset=old['dataset'],manifest_sha256=old['manifest_sha256'],subjects=subjects,ids=chosen,subject_groups=groups,
  seed=SEED,updates=4000,checkpoints=[0,1000,2000,4000],batch_size=8,peak_lr=7e-5,backbone_lr=3.5e-5,warmup=100,lr_floor_ratio=.3,
  tiny_updates=512,tiny_subjects=16,split_tolerance_mm=.5,query_count=120,
  primary_selection='fixed 4000 updates, same population',secondary_selection='max validation common Branch F1 then Curve F1 then earlier checkpoint',
  geometry_route='graph_only; simultaneous independently optimized K8/K24/fixed-4-section C1 heads on evolving original-graph detector; no global gradient clipping',
  geometry_weighting='each original branch total weight one; subedge weights 1/number_of_subedges; unchanged graph losses',
  population='hash-selected up to 32 patches per subject, 39 train/8 val; no test targets; balanced bounded pilot, not full-population estimate',
  adaptive_K='deferred to avoid unrelated selector sweep',max_study_h100=1,max_global_h100=2)
 assert sha(Path(old['dataset'])/'manifest.json')==old['manifest_sha256']
 write(ROOT/'plan.json',plan);print('plan',len(chosen['train']),len(chosen['val']),flush=True)

def components(nodes,edges):
 from scipy.sparse import coo_matrix
 from scipy.sparse.csgraph import connected_components
 e=np.asarray(edges).reshape(-1,2);mat=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(nodes),len(nodes)))
 n=int(connected_components(mat,directed=False,return_labels=False));return n,len(e)-len(nodes)+n

def target_task(task):
 from data.adaptive_split_targets import split_target
 from models.adaptive_geometry import fit_basis,piecewise_knots,uniform_knots
 from scipy.interpolate import BSpline
 from metrics.representation_geometry import point_to_polyline_distance
 dataset,split,sample=task
 with np.load(Path(dataset)/split/'spline'/(sample+'.npz')) as f:gt={k:f[k] for k in f.files}
 n=len(gt['nodes']);e=len(gt['edges']);metric=gt['metric'].astype(float);inv=np.linalg.inv(metric)
 from data.augmentations import assert_valid_coordinates
 assert np.isfinite(gt['full_paths']).all()
 assert_valid_coordinates(torch.from_numpy(gt['nodes']), (64,64,64), allow_patch_faces=True)
 outside=((gt['nodes'] < -1e-6)|(gt['nodes'] > 1+1e-6)).any(1)
 # boundary denotes inserted clipping anchors, not every point beyond voxel centres.
 # An original SWC vertex can lie inside a half-voxel cell with boundary=False.
 base=dict(gt,branch_weights=np.ones(e,np.float32),parent_branch=np.arange(e));variants={'K8':base};oracle=[]
 for name,knots in [('K24',uniform_knots(24)),('C1',piecewise_knots(4,'C1'))]:
  controls=[];curves=[]
  for a,b in zip(gt['path_offsets'][:-1],gt['path_offsets'][1:]):
   c,x=fit_basis(gt['full_paths'][a:b].astype(float)@metric.T,knots);controls.append(c@inv.T);curves.append(x@inv.T)
  variants[name]=dict(base,controls=np.asarray(controls,np.float32).reshape(e,len(knots)-4,3),curves=np.asarray(curves,np.float32).reshape(e,64,3))
 variants['split_K8']=split_target(gt,.5)
 sp=variants['split_K8'];np.testing.assert_allclose(sp['nodes'][:n],gt['nodes'],atol=1e-6);np.testing.assert_array_equal(sp['boundary'][:n],gt['boundary'])
 assert components(gt['nodes'],gt['edges'])==components(sp['nodes'],sp['edges'])
 degree=np.bincount(sp['edges'].ravel(),minlength=len(sp['nodes']));assert np.all(degree[n:]==2)
 for j,(a,b) in enumerate(zip(gt['path_offsets'][:-1],gt['path_offsets'][1:])):
  parts=np.flatnonzero(sp['parent_branch']==j);assert abs(float(sp['branch_weights'][parts].sum())-1)<1e-5
  combined=[]
  for i,k in enumerate(parts):
   lo,hi=sp['path_offsets'][k:k+2];combined.append(sp['full_paths'][lo:hi] if i==0 else sp['full_paths'][lo+1:hi])
  np.testing.assert_allclose(np.concatenate(combined),gt['full_paths'][a:b],atol=1e-6)
  path=gt['full_paths'][a:b].astype(float)@metric.T;row={'edge':j,'length_mm':float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())}
  for name in ['K8','K24','C1']:
   knots=piecewise_knots(4,'C1') if name=='C1' else uniform_knots(variants[name]['controls'].shape[1])
   x=BSpline(knots,variants[name]['controls'][j].astype(float)@metric.T,3)(np.linspace(0,1,257))
   row[name+'_sampled_max_mm']=float(max(point_to_polyline_distance(path,x).max(),point_to_polyline_distance(x,path).max()))
  oracle.append(row)
 for v in variants.values():
  assert len(v['nodes'])<=120,('query capacity exceeded',sample,len(v['nodes']))
  np.testing.assert_allclose(v['controls'][:,[0,-1]],v['nodes'][v['edges']],atol=2e-6)
 distance=np.linalg.norm((sp['nodes'][:,None]-sp['nodes'][None])@metric.T,axis=-1);np.fill_diagonal(distance,np.inf)
 info=dict(sample=sample,split=split,nodes=n,split_nodes=len(sp['nodes']),edges=e,split_edges=len(sp['edges']),border_nodes=int(gt['boundary'].sum()),outside_sigmoid_nodes=int(outside.sum()),outside_sigmoid_non_boundary=int((outside & ~gt['boundary']).sum()),sigmoid_clamp_max_mm=float(np.linalg.norm((gt['nodes']-np.clip(gt['nodes'],0,1))@metric.T,axis=1).max()),components=components(gt['nodes'],gt['edges'])[0],cycles=components(gt['nodes'],gt['edges'])[1],unresolved_branches=sp['unresolved_branches'].tolist(),min_node_separation_mm=float(distance.min()) if len(distance)>1 else None,oracle=oracle)
 return sample,variants,info

def targets(limit=None):
 import multiprocessing as mp
 p=read(ROOT/'plan.json');tasks=[(p['dataset'],split,s) for split in ['train','val'] for s in p['ids'][split]]
 if limit:tasks=tasks[:limit]
 cache={};rows=[];start=time.time()
 with mp.get_context('spawn').Pool(4) as pool:
  for i,(s,v,r) in enumerate(pool.imap(target_task,tasks,chunksize=4)):
   cache[s]=v;rows.append(r)
   if i%64==0:print('targets',i,'/',len(tasks),'seconds',time.time()-start,flush=True)
 suffix='-probe' if limit else ''
 with (ROOT/('targets'+suffix+'.pkl')).open('wb') as f:pickle.dump(cache,f,protocol=4)
 write(ROOT/('target-audit'+suffix+'.json'),dict(completed=True,rows=rows,seconds=time.time()-start,source_manifest_sha256=sha(Path(p['dataset'])/'manifest.json')))
 if not limit:
  candidates={}
  for r in rows:
   if r['split']=='train' and 5<=r['nodes']<=25 and r['edges']>=3:
    subject=p['subjects'][r['sample'].split('_')[1]];candidates.setdefault(subject,[]).append(r['sample'])
  subjects=sorted(candidates);selected=[subjects[i] for i in np.linspace(0,len(subjects)-1,16).astype(int)]
  p['tiny_ids']=[sorted(candidates[s])[0] for s in selected];p['tiny_subject_names']=selected
  p['monitor_train']=sorted(s for subject in p['subject_groups']['train'] for s in p['ids']['train'] if p['subjects'][s.split('_')[1]]==subject) # trimmed below per subject
  p['monitor_train']=[next(s for s in p['ids']['train'] if p['subjects'][s.split('_')[1]]==subject) for subject in p['subject_groups']['train']]
  write(ROOT/'plan.json',p)
 print('targets complete',len(cache),flush=True)

def dataset(cfg,cache,ids,arm,split,augment):
 from data.loaders.synthetic_mri import build_synthetic_mri_dataset
 base=build_synthetic_mri_dataset(cfg['data']['datasets']['synthetic_mri']['root'],split=split,augment=False,spline_controls=8,zoom_range=None)
 lookup={r.sample_id:r for r in base.records};base.records=[lookup[s] for s in ids]
 return RepresentationDataset(base,cache,arm,augment)

def setup(group,augment=True):
 from models import build_model
 from models.checkpoint import load_checkpoint_file
 from models.matcher import build_matcher
 from scripts.train_ixi_transfer import TransferCriterion
 p=read(ROOT/'plan.json');cfg=copy.deepcopy(p['config']);seed();cp=load_checkpoint_file(p['source'])
 assert cp['config']['loss']==cfg['loss'];assert cp['config']['model']['matcher']==cfg['model']['matcher'];assert cp['config']['model']['decoder']['object_queries']==120
 model=build_model(cfg);model.load_state_dict(cp['net'],strict=True);del cp
 heads={'K8':model.spline_head};migration={}
 if group=='original':
  for name,method in [('K24','K24'),('C1','piece_C1')]:heads[name],migration[name]=migrate_geometry(model.spline_head,cfg['model']['spline'],method)
 model.cuda().requires_grad_(True)
 for head in heads.values():head.cuda().requires_grad_(True)
 for layer in model.decoder.decoder.layers:
  if layer.rln_attn:layer.cross_attn2.requires_grad_(False);layer.norm12.requires_grad_(False)
 criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda();criterion.set_training_progress(1,0.)
 oc=cfg['training']['optimizer'];groups=[dict(params=[v for n,v in model.named_parameters() if v.requires_grad and n.startswith('encoder.')],base_lr=p['backbone_lr']),dict(params=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('encoder.')],base_lr=p['peak_lr'])]
 for name in ['K24','C1']:
  if name in heads:groups.append(dict(params=list(heads[name].parameters()),base_lr=p['peak_lr']))
 for g in groups:g['lr']=g['base_lr']
 opt=torch.optim.AdamW(groups,betas=tuple(oc['betas']),eps=oc['epsilon'],weight_decay=oc['weight_decay'])
 with (ROOT/'targets.pkl').open('rb') as f:cache=pickle.load(f)
 arm='K8' if group=='original' else 'split_K8';ds=dataset(cfg,cache,p['ids']['train'],arm,'train',augment)
 return dict(plan=p,cfg=cfg,model=model,heads=heads,criterion=criterion,opt=opt,cache=cache,dataset=ds,group=group,migration=migration,arm=arm)

def target_head(targets,head_name,indices,ds):
 # For extra heads, graph/augmentation are identical. Recover rotation from metrics
 # and one anchor rather than drawing any new augmentation or image noise.
 from data.spline_augmentation import rotate_spline_geometry
 result=dict(targets);result['spline']=[]
 for b,i in enumerate(indices):
  g=targets['spline'][b];source=ds.caches[ds.records[i].sample_id][head_name]
  metric=torch.from_numpy(source['metric']).to(g['metric']);a=torch.linalg.solve(g['metric'],metric)
  base_nodes=torch.from_numpy(source['nodes']).to(targets['nodes'][b]);offset=targets['nodes'][b][0,:3]-base_nodes[0]@a.T
  geom={k:torch.from_numpy(source[k]).to(g['controls'])@a.T+offset for k in ['controls','curves','guides']}
  geom.update(metric=g['metric'],branch_weights=g['branch_weights']);result['spline'].append(geom)
 return result

def losses(bundle,volumes,targets,indices):
 model,criterion=bundle['model'],bundle['criterion'];criterion.spline_head=bundle['heads']['K8'];tokens,pred,_=model(volumes);loss=criterion(tokens,pred,targets)
 total=loss['total'];assignments=criterion.matcher(pred,targets);extra={}
 for name in ['K24','C1']:
  if name in bundle['heads']:
   criterion.spline_head=bundle['heads'][name];y=target_head(targets,name,indices,bundle['dataset']);v=criterion.loss_spline(tokens,pred,y,assignments);extra[name]=v['spline'];total=total+bundle['cfg']['loss']['weights']['spline']*v['spline']
 criterion.spline_head=bundle['heads']['K8'];return total,loss,extra,tokens,pred,assignments

def batch(ds,indices):
 from data.loaders.common import image_graph_collate
 from training.engine import _move_graph_batch
 return _move_graph_batch(image_graph_collate([ds[int(i)] for i in indices]),'cuda','image',True)

def save(bundle,folder,step):
 folder.mkdir(parents=True,exist_ok=True)
 torch.save(dict(net=bundle['model'].state_dict(),geometry_heads={k:h.state_dict() for k,h in bundle['heads'].items()},optimizer=bundle['opt'].state_dict(),config=bundle['cfg'],step=step,group=bundle['group'],source=str(SOURCE)),folder/f'model-{step:06d}.pt')

@torch.no_grad()
def monitor(bundle,step,ids=None,train_mode=False):
 model=bundle['model'];model.train(train_mode);rows=[];assign=[];prior=bundle.setdefault('previous_assignments',{});churn=[];buffers={n:v.clone() for n,v in model.named_buffers()}
 p=bundle['plan'];splits={'train':ids or p['monitor_train']} if ids else {'train':p['monitor_train'],'val':p['ids']['val']}
 for split,names in splits.items():
  ds=dataset(bundle['cfg'],bundle['cache'],names,bundle['arm'],split,False)
  for i,s in enumerate(names):
   x,y=batch(ds,[i]);t,pred,_=model(x);c=graph_cache(model,bundle['criterion'],t,pred,y);a,b=bundle['criterion'].matcher(pred,y)[0]
   mapping=dict(zip(b.tolist(),a.tolist()));key=(split,s,train_mode)
   if key in prior:churn.extend(mapping[k]!=v for k,v in prior[key].items())
   prior[key]=mapping
   probs=c['scores'];assigned=set(a.tolist());assign.append(dict(split=split,sample=s,assigned=float(probs[a.numpy()].mean()),unassigned=float(np.mean([v for k,v in enumerate(probs) if k not in assigned])),hungarian_within_1mm=float((c['hungarian_distance']<=1).mean()),hungarian_within_2mm=float((c['hungarian_distance']<=2).mean())))
   for tol,r in rows_for(c).items():rows.append(dict(r,split=split,sample=s,subject=bundle['plan']['subjects'][s.split('_')[1]],tolerance=tol))
 for n,v in model.named_buffers():v.copy_(buffers[n])
 return dict(step=step,train_mode=train_mode,metrics={s:{tol:summarize([r for r in rows if r['split']==s and r['tolerance']==tol]) for tol in ['1.0','2.0']} for s in splits},rows=rows,assignment=assign,churn=float(np.mean(churn)) if churn else None)

def gradient_check(bundle,total,loss,extra,tokens,pred):
 model=bundle['model'];shared=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('spline_head.')]
 for name,gloss in [('K8',loss['spline'])]+list(extra.items()):
  gs=torch.autograd.grad(gloss,shared,retain_graph=True,allow_unused=True);assert all(g is None or not bool(g.abs().sum()) for g in gs),name
  hs=torch.autograd.grad(gloss,list(bundle['heads'][name].parameters()),retain_graph=True,allow_unused=True);assert sum(float(g.abs().sum()) for g in hs if g is not None)>0
 # Compare exact derivatives before the custom deformable-attention backward,
 # whose grad_value accumulation uses CUDA atomicAdd and is not bitwise repeatable.
 interface=[tokens,pred['pred_nodes'],pred['pred_logits']]
 one=torch.autograd.grad(loss['total'],interface,retain_graph=True,allow_unused=True)
 many=torch.autograd.grad(total,interface,retain_graph=True,allow_unused=True)
 for a,b in zip(one,many):
  assert (a is None)==(b is None)
  if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
 one=torch.autograd.grad(loss['total'],shared,retain_graph=True,allow_unused=True)
 repeat=torch.autograd.grad(loss['total'],shared,retain_graph=True,allow_unused=True)
 many=torch.autograd.grad(total,shared,retain_graph=True,allow_unused=True)
 def difference(a,b):
  assert all((x is None)==(y is None) for x,y in zip(a,b))
  values=[(x,y) for x,y in zip(a,b) if x is not None]
  norm=sum(float(x.square().sum()) for x,y in values)**.5
  return dict(relative_l2=sum(float((x-y).square().sum()) for x,y in values)**.5/max(norm,1e-12),max_abs=max(float((x-y).abs().max()) for x,y in values))
 noise=difference(one,repeat);delta=difference(one,many)
 checks=dict(geometry_to_detector_zero=True,multiplex_interface_gradient_exact=True,all_heads_trainable=True,repeated_detector_backward=noise,multiplex_detector_backward=delta)
 print('gradient checks',checks,flush=True)
 assert delta['relative_l2']<=max(3*noise['relative_l2'],1e-7),checks
 return checks


def train(group,tiny=False):
 b=setup(group,not tiny);p=b['plan'];model=b['model'];ds=b['dataset'];folder=ROOT/('tiny' if tiny else 'runs')/group;folder.mkdir(parents=True,exist_ok=False);write(folder/'config.json',b['cfg']);write(folder/'migration.json',b['migration'])
 if tiny:
  lookup={r.sample_id:i for i,r in enumerate(ds.records)};indices=[lookup[s] for s in p['tiny_ids']];ds.augment=False
 else:indices=list(range(len(ds)))
 observations=[];history=[];start=time.time();updates=p['tiny_updates'] if tiny else p['updates'];image_hash=hashlib.sha256();bn_start={n:v.clone() for n,v in model.named_buffers() if n.endswith(('running_mean','running_var'))}
 observations.append(monitor(b,0,p['tiny_ids'] if tiny else None));write(folder/'monitors.json',observations)
 for step in range(1,updates+1):
  seed(step);epoch=(step-1)//max(1,len(indices)//p['batch_size']);offset=((step-1)%max(1,len(indices)//p['batch_size']))*p['batch_size']
  order=np.random.default_rng(SEED+epoch).permutation(indices);ix=order[offset:offset+p['batch_size']].tolist();model.train();x,y=batch(ds,ix);image_hash.update(x.cpu().numpy().tobytes());b['opt'].zero_grad(set_to_none=True)
  scale=min(1.,step/20) if tiny else min(1.,step/p['warmup'])*(p['lr_floor_ratio']+(1-p['lr_floor_ratio'])*.5*(1+np.cos(np.pi*max(0,step-p['warmup'])/max(1,updates-p['warmup']))))
  for g in b['opt'].param_groups:g['lr']=g['base_lr']*scale
  total,loss,extra,tokens,pred,assignment=losses(b,x,y,ix);assert torch.isfinite(total)
  if step==1:
   checks=gradient_check(b,total,loss,extra,tokens,pred)
   from scripts.multidataset_graph_training import edge_inputs
   with torch.no_grad():
    edge_data=edge_inputs(model,b['criterion'],tokens,pred,y,0,assignment[0])
    if edge_data is not None:
     _,_,feat,ends,metric,local=edge_data;reference=b['heads']['K8'](feat,ends,metric,**local)[1];checks['initial_curve_migration_max_mm']={}
     for name,head in b['heads'].items():
      value=float(((head(feat,ends,metric,**local)[1]-reference)@metric.T).abs().max());checks['initial_curve_migration_max_mm'][name]=value;assert value<.05
   write(folder/'gradient-check.json',checks)
  total.backward();assert all(torch.isfinite(v.grad).all() for g in b['opt'].param_groups for v in g['params'] if v.grad is not None)
  if step in [1,64,256,512,1000,2000,4000]:
   norms={prefix:float(sum(v.grad.square().sum() for n,v in model.named_parameters() if n.startswith(prefix) and v.grad is not None).sqrt()) for prefix in ['encoder.','decoder.','coord_embed.','class_embed.','relation_embed.']};assert all(v>0 for v in norms.values())
   history.append(dict(step=step,losses={k:float(v.detach()) for k,v in loss.items() if torch.is_tensor(v) and v.numel()==1},extra={k:float(v.detach()) for k,v in extra.items()},gradient_norms=norms));write(folder/'training.json',history)
  b['opt'].step()
  checks=[64,256,512] if tiny else [1000,2000,4000]
  if step in checks:
   result=monitor(b,step,p['tiny_ids'] if tiny else None);observations.append(result);write(folder/'monitors.json',observations);print(group,'tiny' if tiny else 'pilot',step,result['metrics'],flush=True)
   if not tiny:
    save(b,folder,step)
    from scripts.brava_representation_evaluation import snapshot
    snapshot(b,step)
  if step%100==0:print('progress',group,step,'seconds',time.time()-start,flush=True)
 if not tiny:
  for name,head in b['heads'].items():
   torch.save(dict(state_dict=head.state_dict(),representation=name if group=='original' else 'split_K8',controls=head.controls,shared_checkpoint=str(folder/f'model-{updates:06d}.pt'),local_geometry=b['cfg']['model']['spline']['local_geometry']),folder/f'geometry-{name}-final.pt')
 if tiny:
  observations.append(monitor(b,updates,p['tiny_ids'],True));write(folder/'monitors.json',observations);save(b,folder,updates)
  final=observations[-2]['metrics']['train']['2.0'];passed=final['node_f1']>=.8 and final['edge_f1']>=.65
 else:passed=True
 bn=float(np.mean([float((dict(model.named_buffers())[n]-v).square().mean().sqrt()) for n,v in bn_start.items()]))
 write(folder/'complete.json',dict(completed=True,gate_passed=passed,steps=updates,seconds=time.time()-start,image_stream_sha256=image_hash.hexdigest(),bn_running_rms_change=bn,job_id=os.getenv('SLURM_JOB_ID'),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))
 if tiny:
  assert passed,'Diverse-subject tiny-set learnability failed; diagnose before pilots'
  from scripts.brava_representation_evaluation import snapshot,evaluate
  snapshot(b,updates,tiny=True);evaluate(group,updates,tiny=True)


def main():
 if os.getenv('SLURM_JOB_ID'):
  source_archive=ROOT/('source-'+os.environ['SLURM_JOB_ID']+'.zip')
  if not source_archive.exists():
   with zipfile.ZipFile(source_archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for directory in ['scripts','models','data','training','metrics','configs','tests','cluster']:
     for source in Path(directory).rglob('*'):
      if source.is_file() and source.suffix in ['.py','.yaml','.json','.sh','.slurm','.cpp','.h','.cu'] and '__pycache__' not in source.parts:z.write(source)
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','targets','tiny','train','evaluate']);ap.add_argument('--group',choices=['original','split'],default='original');ap.add_argument('--step',type=int,default=4000);ap.add_argument('--limit',type=int);a=ap.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
 if a.mode=='prepare':prepare()
 elif a.mode=='targets':targets(a.limit)
 elif a.mode=='evaluate':
  from scripts.brava_representation_evaluation import evaluate
  evaluate(a.group,a.step)
 else:
  if a.mode=='train':
   for group in ['original','split']:assert read(ROOT/'tiny'/group/'complete.json')['gate_passed']
  train(a.group,a.mode=='tiny')
if __name__=='__main__':main()
