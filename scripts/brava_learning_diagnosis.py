"""Frozen K8 checkpoint interventions: matched batching, BN and actual augmentation."""
import copy,hashlib,io,pickle,random,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.brava_learning_audit import OUT,LARGE,PREVIOUS,read,write,sha
from scripts.brava_representation_study import dataset,batch
from scripts.multiscale_pretrained import graph_cache

def seed(n=0):random.seed(261026+n);np.random.seed(261026+n);torch.manual_seed(261026+n)
def bn_state(model):return {n:v.clone() for n,v in model.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))}
def restore_bn(model,state):
 for n,v in model.named_buffers():
  if n in state:v.copy_(state[n])
def bn_mode(model,training):
 model.eval()
 for layer in model.modules():
  if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.train(training)
def weight_hash(model):
 h=hashlib.sha256()
 for n,p in list(model.named_parameters())+[(n,v) for n,v in model.named_buffers() if not n.endswith(('running_mean','running_var','num_batches_tracked'))]:h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def build(path):
 from models import build_model
 from models.matcher import build_matcher
 from scripts.train_ixi_transfer import TransferCriterion
 cp=torch.load(path,map_location='cpu',mmap=True);cfg=cp['config'];model=build_model(cfg);model.load_state_dict(cp['net'],strict=True);model.cuda().eval().requires_grad_(False);del cp
 criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda();criterion.set_training_progress(1,0.)
 return model,cfg,criterion

def datasets(cfg,cache,p):return {name:dataset(cfg,cache,ids,'K8','val' if name=='validation' else 'train',False) for name,ids in p['populations'].items() if name!='old_monitor'}

@torch.no_grad()
def predict(model,criterion,ds,condition,step,batch_size=8,training_bn=False,augment=False,include_losses=True,full_train=False):
 bn_mode(model,training_bn);saved=bn_state(model);oldaug=ds.augment;ds.augment=augment
 if full_train:model.train()
 if augment and ds.intensity is None:
  from data.loaders.synthetic_mri import _build_training_intensity_transform
  ds.intensity=_build_training_intensity_transform(.35,.015,None)
 rows=[];out=LARGE/f'{step}/{condition}';out.mkdir(parents=True,exist_ok=True);name=getattr(ds,'diagnostic_name','population')
 with zipfile.ZipFile(out/(name+'.zip'),'w',compression=zipfile.ZIP_STORED) as archive:
  for start in range(0,len(ds),batch_size):
   ix=list(range(start,min(start+batch_size,len(ds))));seed(start);x,y=batch(ds,ix);image_sha=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest();tokens,pred,_=model(x)
   assignment=criterion.matcher(pred,y)
   losses=None
   if include_losses:
    seed(start);v=criterion(tokens,pred,y);losses={k:float(t) for k,t in v.items() if torch.is_tensor(t) and t.numel()==1}
   for j,i in enumerate(ix):
    c=graph_cache(model,criterion,tokens,pred,y,j);a,b=assignment[j];c.update(hungarian_source=a.numpy(),hungarian_target=b.numpy())
    sample=ds.records[i].sample_id;buffer=io.BytesIO();np.savez_compressed(buffer,**c);archive.writestr(sample+'.npz',buffer.getvalue())
    rows.append(dict(sample=sample,batch_start=start,batch_size=len(ix),image_sha256=image_sha,losses=losses))
 bn=model.named_buffers();restore_bn(model,saved);ds.augment=oldaug;model.eval()
 write(out/(name+'-records.json'),rows);print('predicted',step,condition,name,len(ds),flush=True)

@torch.no_grad()
def calibrate(model,ds,reverse=False):
 bn_mode(model,True);modules=[]
 for m in model.modules():
  if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
   modules.append((m,m.momentum));m.reset_running_stats();m.momentum=None
 order=list(range(len(ds)))
 if reverse:order=order[::-1]
 assert len(order)==312 and not ds.augment
 for i in range(0,len(order),8):
  x,_=batch(ds,order[i:i+8]);model(x)
 for m,momentum in modules:m.momentum=momentum
 model.eval();return bn_state(model)

def augmentation_audit(ds):
 """Exercise the actual study loader with all quarter-turn combinations."""
 from unittest.mock import patch
 from data.augmentations import rotate_volume,rotate_coordinates
 from data.spline_augmentation import rotate_spline_geometry
 class IdentityIntensity:
  def set_random_state(self,seed):pass
  def __call__(self,x):return x
 oldaug,oldint=ds.augment,ds.intensity;ds.augment=False;records=[]
 for index in [0,len(ds)//3,2*len(ds)//3,len(ds)-1]:
  base=ds[index];ds.augment=True;ds.intensity=IdentityIntensity()
  for a in range(4):
   for b in range(4):
    for c in range(4):
     turns=(a,b,c)
     with patch('data.brava_representation_study.random.randint',side_effect=list(turns)):actual=ds[index]
     expected_image=rotate_volume(base[0][0],turns);expected_nodes=rotate_coordinates(base[2][0],turns,(64,64,64),allow_patch_faces=True)
     torch.testing.assert_close(actual[0][0],expected_image,rtol=0,atol=0);torch.testing.assert_close(actual[2][0],expected_nodes,rtol=0,atol=0);torch.testing.assert_close(actual[3][0],base[3][0],rtol=0,atol=0)
     g=actual[4][0];edge=actual[3][0];torch.testing.assert_close(g['controls'][:,0],actual[2][0][edge[:,0]],atol=2e-6,rtol=0);torch.testing.assert_close(g['controls'][:,-1],actual[2][0][edge[:,1]],atol=2e-6,rtol=0)
     # Direct float64 pair differences avoid float32 GEMM cancellation on cdist diagonals.
     before=torch.pdist(base[2][0].double()@base[4][0]['metric'].double().T);after=torch.pdist(actual[2][0].double()@g['metric'].double().T)
     torch.testing.assert_close(before,after,atol=1e-4,rtol=0)
     records.append(dict(sample=ds.records[index].sample_id,turns=turns,passed=True))
  ds.augment=False
 ds.augment=oldaug;ds.intensity=oldint
 return dict(checks=len(records),passed=True,noise_test='geometry-independent Gaussian noise; actual noisy inference measured separately',brain_mask_role='legacy unused image-input slot; no vessel support or supervision')

@torch.no_grad()
def smoke(model,cfg,criterion,cache,p):
 ds=dataset(cfg,cache,p['populations']['representative_train'],'K8','train',False)
 audit=augmentation_audit(ds);prior=zipfile.ZipFile(PREVIOUS/'predictions/original/004000.zip');parity=[]
 val=dataset(cfg,cache,p['populations']['validation'][:3],'K8','val',False)
 for i,s in enumerate(p['populations']['validation'][:3]):
  model.eval();x,y=batch(val,[i]);t,pred,_=model(x);c=graph_cache(model,criterion,t,pred,y)
  with np.load(io.BytesIO(prior.read('K8/'+s+'.npz'))) as f:
   errs={k:float(np.max(np.abs(c[k]-f[k]))) for k in ['physical','scores','relations']};assert errs['physical']<1e-4 and errs['scores']<1e-5 and errs['relations']<1e-5,errs
   parity.append(dict(sample=s,max_errors=errs))
 before=weight_hash(model);saved=bn_state(model);x,y=batch(ds,list(range(8)));seed();model.train();_,actual,_=model(x);restore_bn(model,saved)
 seed();bn_mode(model,True);_,onlybn,_=model(x)
 mode_error={k:float((actual[k]-onlybn[k]).abs().max()) for k in ['pred_nodes','pred_logits']}
 restore_bn(model,saved);assert before==weight_hash(model)
 write(OUT/'smoke.json',dict(passed=True,augmentation=audit,cache_parity=parity,weights_unchanged=True,other_buffers_unchanged=True,bn_layers=len(saved)//3,actual_train_vs_bn_only_max_error=mode_error,dropouts=[m.p for m in model.modules() if isinstance(m,torch.nn.Dropout)]))
 print('smoke passed',flush=True)

def main():
 p=read(OUT/'plan.json');p['populations']['augmentation_probe']=p['populations']['representative_train'][:32]
 with (PREVIOUS/'targets.pkl').open('rb') as f:cache=pickle.load(f)
 cache={s:{'K8':v['K8']} for s,v in cache.items()}
 LARGE.mkdir(parents=True,exist_ok=True);start=time.time();model,cfg,criterion=build(p['checkpoint']);smoke(model,cfg,criterion,cache,p);before=weight_hash(model);saved=bn_state(model)
 dsets=datasets(cfg,cache,p)
 for name,ds in dsets.items():ds.diagnostic_name=name
 # Same saved checkpoint and sample order; isolate batch-size drift from BN mode.
 for name,ds in dsets.items():
  if name=='augmentation_probe':continue
  predict(model,criterion,ds,'saved_b1',4000,1,include_losses=False)
  predict(model,criterion,ds,'saved_b8',4000,8)
  predict(model,criterion,ds,'batch_bn_b8',4000,8,True)
  if max(read(OUT/'smoke.json')['actual_train_vs_bn_only_max_error'].values())>=1e-6:predict(model,criterion,ds,'actual_train_b8',4000,8,True,full_train=True)
 for bn_training in [False,True]:predict(model,criterion,dsets['augmentation_probe'],'aug_batch_bn_b8' if bn_training else 'aug_saved_b8',4000,8,bn_training,True)
 # Evaluate clean versions of exactly the augmentation probe too.
 predict(model,criterion,dsets['augmentation_probe'],'saved_b8',4000,8)
 predict(model,criterion,dsets['augmentation_probe'],'batch_bn_b8',4000,8,True)
 cal=dataset(cfg,cache,p['calibration_ids'],'K8','train',False);stats=[]
 for reverse in [False,True]:
  restore_bn(model,saved);state=calibrate(model,cal,reverse);condition='recal_reverse_b8' if reverse else 'recal_train_b8'
  torch.save({n:v.cpu() for n,v in state.items()},LARGE/(condition+'-bn-only.pt'))
  stats.append(dict(condition=condition,layers={n:float((v-saved[n]).float().square().mean().sqrt()) for n,v in state.items() if 'num_batches' not in n}))
  for name in ['representative_train','validation','tiny_reference']:predict(model,criterion,dsets[name],condition,4000,8)
 restore_bn(model,saved);assert before==weight_hash(model)
 write(OUT/'bn-statistics.json',stats);del dsets,model,criterion;torch.cuda.empty_cache()
 # Fixed-update learning curves on the same representative train/validation cohorts.
 for step in [1000,2000]:
  model,cfg,criterion=build(PREVIOUS/f'runs/original/model-{step:06d}.pt');ds=datasets(cfg,cache,p)
  for name in ['representative_train','validation']:
   ds[name].diagnostic_name=name;predict(model,criterion,ds[name],'saved_b8',step,8)
  del model,criterion,ds;torch.cuda.empty_cache()
 # Positive learnability reference under both matched batch sizes and BN modes.
 model,cfg,criterion=build(p['tiny_checkpoint']);ds=dataset(cfg,cache,p['populations']['tiny_reference'],'K8','train',False);ds.diagnostic_name='tiny_reference'
 for condition,bsize,bn in [('saved_b1',1,False),('saved_b8',8,False),('batch_bn_b8',8,True)]:predict(model,criterion,ds,condition,'tiny512',bsize,bn)
 write(OUT/'complete.json',dict(completed=True,seconds=time.time()-start,job_id=__import__('os').getenv('SLURM_JOB_ID'),training=False,weights_unchanged=True,calibration_training_only=True,calibration_patches=312,batch8_comparisons=True,large_outputs=str(LARGE),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))
 print('BRAVA K8 diagnosis inference complete',flush=True)
if __name__=='__main__':main()
