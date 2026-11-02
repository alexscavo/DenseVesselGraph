"""Isolated IXI local-CNN coordinate/classification study; reuse established metrics."""
import argparse,copy,hashlib,io,json,math,os,random,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
META=Path('experiments/ixi_local_components_20261026')
ORIGINAL=Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/Graph-Native-Betti-Matching-Representation')
SEED=261026
ARMS=['B0','A-xyz-fresh','A-xyz-transfer','A-cls','A','A-control']
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def plan():return read(META/'plan.json')
def seed(n=0):
 import numpy as np,torch
 random.seed(SEED+n);np.random.seed(SEED+n);torch.manual_seed(SEED+n)
def refinement_spec(arm):
 if arm=='B0':return dict(enabled=False)
 return dict(enabled=True,image_mode='zero' if arm=='A-control' else 'full',radius_vox=4.,
             refine_classification=not arm.startswith('A-xyz'),compact_output=arm.startswith('A-xyz'),detach_features=False)
def classification_only(module,args,result):return args[1],result[1]
def expand_queries(state,target,oldq,newq,relations=2):
 """Preserve object/relation/dummy rows; seed new objects identically in every arm."""
 import torch
 state=dict(state);old=state['query_embed.weight'];new=target['query_embed.weight'].clone()
 gen=torch.Generator().manual_seed(SEED+991)
 new[:newq]=torch.randn(newq,new.shape[1],generator=gen,dtype=new.dtype)
 new[:oldq]=old[:oldq];new[newq:]=old[oldq:]
 assert new.shape[0]-newq==old.shape[0]-oldq
 state['query_embed.weight']=new;return state

def prepare():
 assert not (META/'plan.json').exists(),'Existing plan is immutable'
 old=read(ORIGINAL/'experiments/local_refinement_long/plan.json');multi=read(ORIGINAL/'experiments/multidataset_continuous/plan.json')
 spec=copy.deepcopy(multi['datasets']['IXI']);src=ORIGINAL/'experiments/local_refinement_long/364505/runs'
 sources={a:str(src/a/'model-12000.pt') for a in ['B0','A','A-xyz']}
 for x in sources.values():assert Path(x).is_file(),x
 cfg=copy.deepcopy(old['config']);d=cfg['data']['datasets']['synthetic_mri'];d.update(root=spec['root'],train_samples=None,validation_samples=None,patch_selection='all')
 cfg['data'].update(batch_size=32,validation_batch_size=1,train_augmentation=True)
 cfg['runtime']['workers']=4;cfg['tracking']['enabled']=False
 cfg['model']['decoder']['object_queries']=256;cfg['evaluation']['protocol']['max_detections']=256
 cfg['experiment'].update(seed=SEED,name='ixi_local_components')
 cfg['training']['optimizer']['base_lr']=7e-6
 cfg['model']['node_refinement']={'enabled':False}
 assert cfg['loss']['node']['classification']['name']=='focal' and cfg['loss']['edge']['classification']['name']=='cross_entropy'
 subjects=lambda ids:{spec['subjects'][s.split('_')[1]] for s in ids}
 assert not subjects(spec['train_ids'])&subjects(spec['val_ids'])
 p=dict(config=cfg,dataset=spec,arms=ARMS,seed=SEED,sources=sources,source_sha256={a:sha(v) for a,v in sources.items()},
        output_root='/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_local_components_20261026',
        updates_cap=6000,minimum_updates=1500,training_seconds_cap=3600,checkpoints_requested=[1500,3000,6000],
        warmup=100,peak_lr=7e-6,floor_lr=1e-6,validation_batch=1,train_monitor=128,validation_monitor=128,
        detector_queries=256,zero_shot_queries=120,max_concurrent_h100_this_batch=1,max_global_h100=2,
        primary='matched final budget; no outcome-dependent extension',secondary='validation edge F1 2mm, tie node F1 1mm',
        classification_init='fresh zero residual on shared B0; no classification-only source exists',
        transfer_init='only A-xyz refinement parameters transplanted into exact common B0 trunk; native A-xyz separately zero-shot',
        sample_sha256={k:hashlib.sha256('\n'.join(spec[k]).encode()).hexdigest() for k in ['train_ids','val_ids','monitor_train','monitor_val']})
 write(META/'plan.json',p)
 print('Prepared',len(spec['train_ids']),len(spec['val_ids']),len(subjects(spec['train_ids'])),len(subjects(spec['val_ids'])),flush=True)

def setup(arm,native=None,augment=True):
 import torch
 from models import build_model
 from models.checkpoint import load_checkpoint_file
 from models.matcher import build_matcher
 from scripts.train_ixi_transfer import TransferCriterion
 from data.loaders import build_data_loaders
 from training.optim import build_optimizer
 from scripts.fine_feature_architecture import worker_init
 from models.ops.modules import MSDeformAttn
 from configs import validate_config
 p=plan();seed();key=native or 'B0';cp=load_checkpoint_file(p['sources'][key]);assert sha(p['sources'][key])==p['source_sha256'][key]
 cfg=copy.deepcopy(p['config']);cfg['data']['train_augmentation']=augment
 assert cfg['loss']==cp['config']['loss'] and cfg['model']['matcher']==cp['config']['model']['matcher']
 assert cfg['model']['spline']==cp['config']['model']['spline']
 if native:
  cfg['model']=copy.deepcopy(cp['config']['model']);cfg['evaluation']['protocol']['max_detections']=120
  spec=cfg['model'].get('node_refinement',{})
  if spec.get('enabled'):spec['detach_features']=False
 else:cfg['model']['node_refinement']=refinement_spec(arm)
 cfg['experiment']['name']='ixi_local_'+arm
 validate_config(cfg);model=build_model(cfg)
 state=cp['net'] if native else expand_queries(cp['net'],model.state_dict(),120,256)
 loaded=model.load_state_dict(state,strict=False)
 assert not loaded.unexpected_keys and all(n.startswith('node_refinement.') for n in loaded.missing_keys)
 for k,v in state.items():assert torch.equal(v,model.state_dict()[k]),k
 if arm=='A-xyz-transfer' and not native:
  donor=load_checkpoint_file(p['sources']['A-xyz']);model.node_refinement.load_state_dict({k.removeprefix('node_refinement.'):v for k,v in donor['net'].items() if k.startswith('node_refinement.')},strict=True);del donor
 if arm=='A-cls':model.node_refinement.register_forward_hook(classification_only)
 model.cuda().requires_grad_(True)
 for layer in model.decoder.decoder.layers:
  if layer.rln_attn:layer.cross_attn2.requires_grad_(False);layer.norm12.requires_grad_(False)
 for m in model.modules():
  if isinstance(m,MSDeformAttn):m.record_level_diagnostics=True
 criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda();criterion.set_training_progress(1,0.)
 loader,_=build_data_loaders(cfg);ds=loader.dataset.datasets[0];lookup={r.sample_id:r for r in ds.records};ds.records=[lookup[s] for s in p['dataset']['train_ids']];loader.worker_init_fn=worker_init
 loader.dataset.cumulative_sizes=torch.utils.data.ConcatDataset.cumsum(loader.dataset.datasets)
 assert len(ds)==len(loader.dataset)==4096
 return dict(plan=p,cfg=cfg,model=model,criterion=criterion,loader=loader,opt=build_optimizer(cfg,model))

def context(b):
 from scripts import fine_feature_architecture as prior
 prior.ROOT=Path(b['plan']['output_root']);prior.SEED=SEED
 return prior

def intervention(model,mode):
 if model.node_refinement is None:return None
 if mode=='stage1':return model.node_refinement.register_forward_hook(lambda m,a,r:(a[1],a[2]))
 if mode=='no-confidence':return model.node_refinement.register_forward_hook(lambda m,a,r:(r[0],a[2]))
 if mode=='no-coordinate':return model.node_refinement.register_forward_hook(lambda m,a,r:(a[1],r[1]))
 return None

def evaluate(b,arm,label,small=False,mode='normal',reconstruction=False):
 """Reuse batch-one raw-query evaluation and archived caches; Stage1 reuses forward outputs."""
 prior=context(b);hook=intervention(b['model'],mode)
 try:
  result=prior.evaluate(b,arm,label,small=small,intervention=mode,cache_archive=True)
  if reconstruction:prior.reconstruction(b,arm,label+'-reconstruction',intervention=mode)
  return result
 finally:
  if hook is not None:hook.remove()

def shared_digest(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():
  if not k.startswith('node_refinement.'):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def smoke():
 import numpy as np,torch
 from training.engine import _move_graph_batch
 from scripts.local_refinement_long import hash_tree
 p=plan();root=Path(p['output_root']);audit=dict(completed=False,arms={});common=None;times=[]
 metrics=[];seen=set();target_nodes=[]
 for sample in p['dataset']['train_ids']+p['dataset']['val_ids']:
  subject=p['dataset']['subjects'][sample.split('_')[1]]
  if subject in seen:continue
  seen.add(subject);split='train' if sample in p['dataset']['train_ids'] else 'val'
  with np.load(Path(p['dataset']['root'])/split/'spline'/(sample+'.npz')) as gt:
   assert np.isfinite(gt['metric']).all() and abs(np.linalg.det(gt['metric']))>0
   assert gt['controls'].shape[1:]==(8,3)
   assert np.allclose(gt['controls'][:,0],gt['nodes'][gt['edges'][:,0]],atol=2e-5)
   assert np.allclose(gt['controls'][:,-1],gt['nodes'][gt['edges'][:,1]],atol=2e-5)
   metrics.append((np.linalg.norm(gt['metric'],axis=0)/64).tolist());target_nodes.append(len(gt['nodes']))
 audit['dataset_audit']=dict(subjects=len(seen),physical_voxel_mm_min=np.min(metrics,axis=0).tolist(),physical_voxel_mm_max=np.max(metrics,axis=0).tolist(),checked_nodes_max=max(target_nodes))
 # Verify physical metric and spline cache/loader endpoint alignment on real IXI patches.
 for arm in ARMS:
  b=setup(arm,augment=False);m=b['model'];ds=b['loader'].dataset.datasets[0]
  m.eval();v,y=_move_graph_batch(b['loader'].collate_fn([ds[i] for i in range(4)]),'cuda','image',True)
  with torch.no_grad():t,o,_=m(v)
  if arm=='B0':
   with torch.no_grad():
    single=torch.cat([m(v[i:i+1])[1]['pred_nodes'] for i in range(4)])
   drift=float((single-o['pred_nodes']).abs().max());audit['batch1_vs4_max_normalized_coordinate_drift']=drift
   # Different batch sizes have known decoder numerical sensitivity. Never mix
   # them in scientific comparisons; graph reconstruction now also uses batch1.
   assert np.isfinite(drift)
   with torch.no_grad():repeat=m(v[:1])[1]['pred_nodes']
   same_batch_drift=float((repeat-single[:1]).abs().max())
   audit['same_batch1_repeat_max_normalized_coordinate_drift']=same_batch_drift
   audit['evaluation_batch_size']=1
   write(root/'smoke.json',audit)
   assert same_batch_drift<1e-6,('Same-batch inference is unstable',same_batch_drift)
  if common is None:common=shared_digest(m)
  assert shared_digest(m)==common,'Shared initializer mismatch'
  assert o['pred_nodes'].shape[1]==256
  base=m.coord_embed(t[:,:m.obj_token]).sigmoid();logits=m.class_embed(t[:,:m.obj_token])
  if arm in ['B0','A-cls','A-xyz-fresh','A','A-control']:torch.testing.assert_close(o['pred_nodes'],base,rtol=0,atol=0)
  if arm in ['B0','A-xyz-fresh','A-xyz-transfer','A','A-control']:torch.testing.assert_close(o['pred_logits'],logits,rtol=0,atol=0)
  if arm=='A-cls':torch.testing.assert_close(o['pred_nodes'],base,rtol=0,atol=0)
  spacing=[]
  for gt in y['spline']:
   metric=gt['metric'];spacing.append((torch.linalg.vector_norm(metric,dim=0)/64).cpu().tolist())
   torch.testing.assert_close(gt['controls'][:,0],y['nodes'][len(spacing)-1][y['edges'][len(spacing)-1][:,0]],atol=2e-5,rtol=0)
  # Real batch32 forward/backward, finite gradients into all active original modules.
  del t,o,v,y
  v,y=_move_graph_batch(b['loader'].collate_fn([ds[i] for i in range(32)]),'cuda','image',True)
  seconds=[];g=None;m.train()
  for step in range(8):
   seed(step);torch.cuda.synchronize();start=time.monotonic();b['opt'].zero_grad(set_to_none=True)
   t,o,_=m(v);loss=b['criterion'](t,o,y);assert torch.isfinite(loss['total']);loss['total'].backward()
   if step in [0,7]:g=context(b).diagnostic(m,loss,step)
   b['opt'].step();torch.cuda.synchronize();seconds.append(time.monotonic()-start)
  times.append(max(seconds[2:]));audit['arms'][arm]=dict(seconds_per_update=seconds,gradients=g,spacing_vox_mm=spacing,shared_initializer_sha256=common,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
  if arm!='B0':
   f=m._spline_local_features.detach().requires_grad_(True)
   tokens=t[:,:256].detach();nodes=m.coord_embed(tokens).sigmoid().detach();scores=m.class_embed(tokens).detach()
   out=m.node_refinement(tokens,nodes,scores,f,(64,64,64));objective=(out[1][...,1]-out[1][...,0]).sum() if arm=='A-cls' else out[0].sum()
   grad=torch.autograd.grad(objective,f,allow_unused=True)[0]
   if arm=='A-control':assert grad is None or grad.norm()==0
   else:assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
   audit['arms'][arm]['image_gradient']=0 if grad is None else float(grad.norm())
  if arm=='B0':
   tiny=copy.deepcopy(b['plan']);tiny['dataset']['monitor_train']=tiny['dataset']['train_ids'][:2];tiny['dataset']['monitor_val']=tiny['dataset']['val_ids'][:2];tiny['dataset']['val_ids']=tiny['dataset']['val_ids'][:2]
   probe=dict(b,plan=tiny);evaluate(probe,'smoke-B0','tiny',small=True,reconstruction=True)
   from scripts.report_ixi_local_components import endpoints
   audit['evaluation_smoke']=endpoints(root/'runs/smoke-B0/tiny-reconstruction')
  del b,m,t,o,loss,v,y;torch.cuda.empty_cache();write(root/'smoke.json',audit)
 budget=min(p['updates_cap'],int(p['training_seconds_cap']/max(times)/1.15)//500*500)
 assert budget>=p['minimum_updates'],('Throughput too low',budget)
 checkpoints=sorted(set([x for x in p['checkpoints_requested'] if x<budget]+[budget]))
 audit.update(completed=True,updates=budget,checkpoints=checkpoints,worst_bench_seconds=max(times),job_id=os.getenv('SLURM_JOB_ID'))
 write(root/'smoke.json',audit);print('GPU gate passed',budget,flush=True)

def zero_shot():
 import torch
 for source in ['B0','A-xyz','A']:
  b=setup('zero-'+source,native=source,augment=False)
  evaluate(b,'zero-'+source,'native',reconstruction=True)
  if source!='B0':
   evaluate(b,'zero-'+source,'stage1',mode='stage1',reconstruction=True)
   for mode in ['zero','roll']:evaluate(b,'zero-'+source,mode,small=True,mode=mode)
  write(Path(b['plan']['output_root'])/'runs'/('zero-'+source)/'complete.json',dict(completed=True,source=source,queries=120))
  del b;torch.cuda.empty_cache()

def train(index):
 import torch
 from training.engine import _move_graph_batch
 from scripts.local_refinement_long import hash_tree
 from scripts.extend_scratch_geometry import atomic_checkpoint
 p=plan();root=Path(p['output_root']);gate=read(root/'smoke.json');assert gate['completed'];arm=ARMS[index]
 out=root/'runs'/arm;assert not (out/'config.json').exists(),'Preserve previous attempt'
 b=setup(arm);m=b['model'];write(out/'config.json',b['cfg']);evaluate(b,arm,'initial',small=True)
 history=[];h=hashlib.sha256();start=time.time();step=epoch=0;best=(-1.,-1.);best_step=0
 while step<gate['updates']:
  epoch+=1;b['loader'].generator.manual_seed(SEED+epoch)
  for batch in b['loader']:
   if step>=gate['updates']:break
   step+=1;seed(step);v,y=_move_graph_batch(batch,'cuda','image',True);hash_tree(h,v);hash_tree(h,y)
   lr=p['peak_lr']*step/p['warmup'] if step<=p['warmup'] else p['floor_lr']+(p['peak_lr']-p['floor_lr'])*(1-(step-p['warmup'])/(gate['updates']-p['warmup']))**.9
   for g in b['opt'].param_groups:g['lr']=lr
   m.train();b['opt'].zero_grad(set_to_none=True);t,o,_=m(v);loss=b['criterion'](t,o,y);assert torch.isfinite(loss['total']);loss['total'].backward()
   if step in [1,100,500] or step%500==0:
    row=context(b).diagnostic(m,loss,step);row.update(lr=lr,seconds=time.time()-start);history.append(row);write(out/'training.json',history);print(arm,step,float(loss['total']),flush=True)
   torch.nn.utils.clip_grad_norm_(m.parameters(),5.);b['opt'].step()
   if step%100==0:write(out/'progress.json',dict(step=step,updates=gate['updates'],seconds=time.time()-start))
   if step in gate['checkpoints']:
    atomic_checkpoint(out/f'model-{step}.pt',dict(net=m.state_dict(),config=b['cfg'],steps=step,source=p['sources']['B0'],seed=SEED,optimizer=b['opt'].state_dict()))
    final=step==gate['updates'];monitor_bundle=b if final else dict(b,plan=dict(p,dataset=dict(p['dataset'],val_ids=p['dataset']['monitor_val'])))
    result=evaluate(monitor_bundle,arm,f'step-{step}',small=False,reconstruction=final)
    score=(result['metrics']['val']['2.0']['edge_f1'],result['metrics']['val']['1.0']['node_f1'])
    # Selection uses a common 128-patch monitor at every step (full final stored separately).
    if final:
     rr=json.loads((out/f'step-{step}'/'rows.json').read_text());ids=set(p['dataset']['monitor_val'])
     from scripts.multiscale_pretrained import summarize
     score=tuple(summarize([r for r in rr if r['split']=='val' and r['sample'] in ids and r['tolerance']==tol])[field] for tol,field in [('2.0','edge_f1'),('1.0','node_f1')])
    if score>best:best=score;best_step=step
    write(out/'batch-proof.json',dict(step=step,sha256=h.hexdigest()))
 if arm!='B0':
  for mode in ['zero','roll','stage1']+(['no-confidence','no-coordinate'] if arm=='A' else []):evaluate(b,arm,'final-'+mode,small=True,mode=mode,reconstruction=mode=='stage1')
 write(out/'complete.json',dict(completed=True,steps=step,seconds=time.time()-start,primary_step=step,secondary_step=best_step,secondary_score=best,batch_sha256=h.hexdigest(),job_id=os.getenv('SLURM_JOB_ID'),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','smoke','zero','train']);parser.add_argument('--index',type=int,default=0);a=parser.parse_args()
 if a.mode=='prepare':prepare();return
 import torch
 torch.set_num_threads(1)
 Path(plan()['output_root']).mkdir(parents=True,exist_ok=True)
 {'smoke':smoke,'zero':zero_shot,'train':lambda:train(a.index)}[a.mode]()
if __name__=='__main__':main()
