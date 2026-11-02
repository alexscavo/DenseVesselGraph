"""Paired raw-MRA versus supplied vessel-label input; no graph rasterization."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from scipy.spatial import cKDTree
from scripts.ixi_cached_mechanisms import manifest,save,table,sha
from scripts.ixi_node_focal_ablation import state_digest,update as full_network_update,atomic_checkpoint

ARMS=['raw_mra','vessel_mask']
SEED=201021


def configuration(plan,arm):
    cfg=copy.deepcopy(manifest()['config'])
    cfg['data'].update(input_modality=arm,train_augmentation=True,batch_size=8,validation_batch_size=1)
    cfg['data']['datasets']['synthetic_mri']['root']=plan['dataset']
    cfg['runtime']['workers']=0;cfg['tracking']['enabled']=False
    cfg['training']['input']='image';cfg['experiment'].update(seed=SEED,name='ixi_input_'+arm)
    assert cfg['model']['decoder']['object_queries']==256 and cfg['model']['encoder']['input_channels']==1
    assert cfg['model']['node_refinement']['enabled']
    assert cfg['loss']['node']['classification']['name']=='weighted_cross_entropy'
    assert cfg['loss']['node']['classification']['class_weights']==[.25,.75]
    assert cfg['loss']['edge']['classification']['name']=='focal'
    assert cfg['loss']['edge']['balancing']['mode']=='none'
    assert not cfg['loss']['edge']['candidates']['include_unmatched']
    assert cfg['loss']['weights']['spline']==0.
    assert not cfg['augmentation']['synthetic_mri']['zoom']['enabled']
    return cfg


def transfer_preixi(model,checkpoint):
    state=dict(checkpoint['net']);old=checkpoint['config']['model']['decoder']['object_queries']
    assert checkpoint['config']['model']['decoder']['relation_tokens']==2
    assert set(checkpoint['config']['data']['datasets'])=={'synthetic_mri'}
    assert 'ixi' not in checkpoint['config']['data']['datasets']['synthetic_mri']['root'].lower()
    original=state['query_embed.weight'];expanded=model.query_embed.weight.detach().clone()
    assert original.shape[0]==old+2 and expanded.shape[0]==258
    expanded[:old]=original[:old];expanded[256:]=original[old:old+2]
    state['query_embed.weight']=expanded
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys,result.unexpected_keys
    assert result.missing_keys and all(k.startswith('node_refinement.') for k in result.missing_keys),result.missing_keys
    return dict(source_object_queries=old,target_object_queries=256,new_object_queries=256-old,
        missing_fresh_parameters=result.missing_keys,relation_tokens_preserved=True)


def setup(root,arm):
    from models import build_model
    from models.checkpoint import load_checkpoint_file
    from models.matcher import build_matcher
    from configs import validate_config
    from data.loaders import build_data_loaders
    from scripts.train_ixi_transfer import TransferCriterion
    plan=json.loads((root/'plan.json').read_text());ids=json.loads((root/'audit/manifest.json').read_text())
    assert sha(plan['initializer'])==plan['initializer_sha256']
    cfg=configuration(plan,arm);validate_config(cfg);torch.manual_seed(SEED)
    model=build_model(cfg);cp=load_checkpoint_file(plan['initializer']);transfer=transfer_preixi(model,cp);del cp
    model.cuda().requires_grad_(True).train();model.spline_head.requires_grad_(False)
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda()
    criterion.set_training_progress(1,0.)
    full,_=build_data_loaders(cfg);assert len(full.dataset.datasets)==1
    dataset=copy.copy(full.dataset.datasets[0]);lookup={r.sample_id:r for r in dataset.records}
    dataset.records=[lookup[n] for n in ids['ids']['train']]
    # MONAI has a private RNG; explicitly seed raw-image noise. Spatial rotations
    # use per-update Python seeds in the shared update helper, identical in arms.
    if dataset.intensity_transform is not None and hasattr(dataset.intensity_transform,'set_random_state'):
        dataset.intensity_transform.set_random_state(seed=SEED)
    settings=cfg['training']['optimizer']
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=7e-5,
        weight_decay=settings['weight_decay'],eps=settings['epsilon'],betas=tuple(settings['betas']))
    return model,cfg,criterion,opt,dataset,full.collate_fn,transfer


def order_for(updates,count):
    return np.random.default_rng(SEED).integers(count,size=(updates,8))


def update(model,cfg,criterion,opt,dataset,collate,indices,number,diagnostic=False):
    def paired_collate(samples):
        batch=collate(samples)
        # Raw-only intensity augmentation must not change the model/pair-sampler
        # RNG stream compared with the binary-input arm.
        torch.manual_seed(SEED+number)
        return batch
    return full_network_update(model,cfg,criterion,opt,dataset,paired_collate,indices,number,diagnostic)


def audit(root,limit=None):
    plan=json.loads((root/'plan.json').read_text());dataset=Path(plan['dataset'])
    out=root/('local-audit' if limit else 'audit');out.mkdir(parents=True,exist_ok=False)
    assert sha(dataset/'manifest.json')==plan['dataset_manifest_sha256']
    generation=json.loads((dataset/'generation_config.json').read_text());source=Path(generation['source_root'])
    sources={r['patient_id']:r for r in csv.DictReader((source/'source_manifest.csv').open())}
    index={r['sample_id']:r for r in csv.DictReader((dataset/'patch_index.csv').open())}
    official=json.loads((dataset/'manifest.json').read_text())
    ids={s:(names[:limit] if limit else names) for s,names in plan['ids'].items()}
    rows=[];excluded=[];valid={s:[] for s in ids};input_records=[]
    for split,names in ids.items():
        assert set(names)<=set(official['train_ids' if split=='train' else 'validation_ids'])
        groups={}
        for name in names:groups.setdefault(index[name]['patient_id'],[]).append(name)
        for patient,selected in groups.items():
            info=sources[patient];source_mask=nib.load(info['label']);parent=np.asarray(source_mask.dataobj)
            source_image=nib.load(info['image']);np.testing.assert_allclose(source_image.affine,source_mask.affine,atol=1e-4)
            assert source_image.shape==source_mask.shape
            marker=json.loads(Path(info['completion_marker']).read_text())
            assert marker['label']==info['label'] and marker['image']==info['image']
            assert marker['configuration']['centerline_backend']=='vedo_original'
            assert Path(info['label']).parent.name=='labelsTr', 'Must use supplied labels, not graph-derived raster masks'
            for name in selected:
                record=index[name];start=np.array([int(record['start_'+a]) for a in 'dhw'])
                image_path=dataset/split/'raw'/(name+'_data.nii.gz');mask_path=dataset/split/'seg'/(name+'_seg.nii.gz');gt_path=dataset/split/'spline'/(name+'.npz')
                raw_nii=nib.load(image_path);mask_nii=nib.load(mask_path)
                np.testing.assert_allclose(raw_nii.affine,mask_nii.affine,atol=1e-5)
                raw=np.asarray(raw_nii.dataobj);m=np.asarray(mask_nii.dataobj)
                assert raw.shape==m.shape==(64,64,64) and np.isfinite(raw).all()
                np.testing.assert_allclose([raw.mean(),raw.std()],
                    [float(record['image_mean']),float(record['image_std'])],atol=1e-6,rtol=1e-5)
                assert np.isfinite(m).all() and np.isin(m,[0,1]).all(), ('Nonbinary mask',name)
                np.testing.assert_array_equal(m,parent[tuple(slice(int(s),int(s+64)) for s in start)])
                with np.load(gt_path) as gt:
                    expected=source_mask.affine[:3,:3]@np.diag([64]*3)
                    np.testing.assert_allclose(gt['metric'],expected,atol=1e-5,rtol=1e-5)
                    assert len(gt['nodes'])<=256 and len({tuple(sorted(e)) for e in gt['edges']})==len(gt['edges'])
                    points=np.concatenate([gt['nodes'],gt['full_paths']]);metric=gt['metric']
                    distance=cKDTree(np.argwhere(m)/64@metric.T).query(points@metric.T)[0] if m.any() else np.full(len(points),np.inf)
                    maximum=float(distance.max(initial=0));reason='empty_vessel_mask' if not m.any() else 'GT_more_than_1mm_from_mask' if maximum>1.+1e-4 else None
                    support=dict(sample_id=name,subject=info['subject'],patient_id=patient,split=split,foreground_voxels=int(m.sum()),
                        maximum_reference_to_foreground_mm=maximum if np.isfinite(maximum) else None,exact_source_crop=True,excluded_reason=reason)
                    rows.append(support)
                    if reason:excluded.append(support);continue
                    valid[split].append(name)
                    input_records.append(dict(split=split,sample_id=name,subject=info['subject'],patient_id=patient,
                        image=str(image_path),mask=str(mask_path),gt=str(gt_path),image_sha256=sha(image_path),mask_sha256=sha(mask_path),gt_sha256=sha(gt_path),
                        full_image=info['image'],full_mask=info['label'],completion_marker=info['completion_marker'],
                        world_metric=metric.tolist(),site=plan['sites'][name]))
                    if len([r for r in input_records if r['split']==split])<=2:
                        from scripts.audit_ixi_supervision import write_view
                        write_view(out,name,raw,m>0,gt['full_paths']*64,np.empty((0,3)),gt['nodes']*64,np.empty((0,3)))
            print('audited',split,info['subject'],len(rows),flush=True)
    # Preserve original manifest order, independent of per-parent audit grouping.
    valid={s:[n for n in ids[s] if n in set(valid[s])] for s in ids}
    subjects={s:{r['subject'] for r in input_records if r['split']==s} for s in ids}
    assert not subjects['train']&subjects['val']
    evaluation_ids=dict(train=[n for n in plan['previous_ids']['train'] if n in valid['train']],val=valid['val'])
    save(out/'manifest.json',dict(completed=True,ids=valid,evaluation_ids=evaluation_ids,counts={s:len(v) for s,v in valid.items()},
        subject_counts={s:len(v) for s,v in subjects.items()},excluded=excluded,
        provenance_category=2,provenance='Provided IXI labelsTr vessel masks used upstream to extract graph references; annotator/segmentation algorithm unknown. Not expert-GT or independent-label evaluation.',
        guide_support_gate_mm=1.,initializer=plan['initializer'],initializer_sha256=plan['initializer_sha256']))
    save(out/'inputs.json',input_records);table(out/'alignment.csv',rows)
    if not limit:
        assert len(valid['train'])>=1900 and len(valid['val'])>=450, 'Too many unsupported references: review provenance/alignment before training'
    print({s:len(v) for s,v in valid.items()},'subjects',{s:len(v) for s,v in subjects.items()},'excluded',len(excluded))


def verify_batches(dataset,collate,indices):
    from scripts.ixi_node_focal_ablation import seeded
    seeded(1);batch=collate([dataset[int(i)] for i in indices])
    if dataset.input_modality=='vessel_mask':assert torch.isin(batch[0],torch.tensor([0.,1.])).all()
    return dict(input_min=float(batch[0].min()),input_max=float(batch[0].max()),input_shape=list(batch[0].shape))


def benchmark(root,smoke=False):
    out=root/('smoke' if smoke else 'benchmark');out.mkdir(parents=True,exist_ok=False)
    timings={};states=[];names=[];batch_graph_hash=[]
    for arm in ARMS:
        model,cfg,criterion,opt,dataset,collate,transfer=setup(root,arm)
        states.append(state_digest(model));names.append([n for n,p in model.named_parameters() if p.requires_grad])
        order=order_for(2 if smoke else 35,len(dataset));batch=verify_batches(dataset,collate,order[0])
        from scripts.ixi_node_focal_ablation import seeded
        seeded(1);paired=collate([dataset[int(i)] for i in order[0]])
        batch_graph_hash.append(hashlib.sha256(b''.join(n.numpy().tobytes() for n in paired[2])).hexdigest())
        history=[];started=None;torch.cuda.reset_peak_memory_stats()
        for number,indices in enumerate(order,1):
            if number==4:torch.cuda.synchronize();started=time.monotonic()
            history.append(update(model,cfg,criterion,opt,dataset,collate,indices,number,diagnostic=number==1))
        torch.cuda.synchronize()
        timings[arm]=dict(seconds_per_update=(time.monotonic()-started)/32 if started else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            input=batch,transfer=transfer,last=history[-1])
        save(out/(arm+'-config.json'),cfg)
        if smoke:atomic_checkpoint(out/(arm+'.pt'),model,cfg,opt,len(order))
        del model,criterion,opt;torch.cuda.empty_cache()
    assert len(set(states))==1 and len(set(batch_graph_hash))==1 and names[0]==names[1]
    save(out/'summary.json',dict(completed=True,smoke=smoke,timings=timings,initial_model_sha256=states[0],trainable_names=names[0],paired_augmented_graph_sha256=batch_graph_hash[0]))
    if not smoke:
        slowest=max(v['seconds_per_update'] for v in timings.values());updates=min(30000,int(4500/(1.2*slowest))//100*100)
        assert updates>=2000,'Insufficient common update budget; do not interpret a tiny pilot as a modality comparison'
        checkpoints=sorted({s for s in [1000,2000,4000,updates] if s<=updates})
        save(out/'budget.json',dict(completed=True,final_updates=updates,checkpoints=checkpoints,target_training_seconds=4500,
            safety_factor=1.2,slowest_seconds_per_update=slowest,batch_size=8))


def train(root,arm):
    budget=json.loads((root/'benchmark/budget.json').read_text());assert budget['completed']
    model,cfg,criterion,opt,dataset,collate,transfer=setup(root,arm)
    out=root/'runs'/arm;out.mkdir(parents=True,exist_ok=False);save(out/'resolved-config.json',cfg)
    initial=state_digest(model);order=order_for(budget['final_updates'],len(dataset));history=[];started=time.monotonic();stop=[False]
    for sig in [signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,lambda *unused:stop.__setitem__(0,True))
    for number,indices in enumerate(order,1):
        row=update(model,cfg,criterion,opt,dataset,collate,indices,number,diagnostic=number==1)
        if number==1 or number%50==0 or number==len(order):
            history.append(dict(elapsed_seconds=time.monotonic()-started,**row));save(out/'history.json',history);print(arm,history[-1],flush=True)
        if number in budget['checkpoints']:atomic_checkpoint(out/f'model-{number:06d}.pt',model,cfg,opt,number)
        if number%500==0:atomic_checkpoint(out/'latest.pt',model,cfg,opt,number)
        if stop[0] or time.monotonic()-started>6300:
            atomic_checkpoint(out/'interrupted.pt',model,cfg,opt,number);raise RuntimeError('Incomplete training; recovery saved and dependent evaluation blocked')
    save(out/'summary.json',dict(completed=True,arm=arm,updates=len(order),checkpoints=budget['checkpoints'],
        initial_model_sha256=initial,transfer=transfer,trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
        batch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),elapsed_seconds=time.monotonic()-started,
        optimizer_restart=True,scheduler='constant7e-5',initializer=json.loads((root/'plan.json').read_text())['initializer']))


def predict(root,arm,smoke=False):
    from models import build_model
    from models.checkpoint import load_checkpoint_file
    from data.loaders import build_data_loaders,build_evaluation_loader
    from scripts.frozen_spline_diagnostic import training_evaluation_loader
    from scripts.diagnose_ixi_queries import cache_patch
    m=json.loads((root/'audit/manifest.json').read_text());plan=json.loads((root/'plan.json').read_text())
    summary=dict(checkpoints=[2],updates=2) if smoke else json.loads((root/'runs'/arm/'summary.json').read_text())
    for number in summary['checkpoints']:
        path=root/'smoke'/(arm+'.pt') if smoke else root/'runs'/arm/f'model-{number:06d}.pt'
        cp=load_checkpoint_file(path);cfg=copy.deepcopy(cp['config']);cfg['data'].update(train_augmentation=False,batch_size=1,validation_batch_size=1)
        model=build_model(cfg).cuda();model.load_state_dict(cp['net'],strict=True);del cp;model.eval().requires_grad_(False)
        frozen=state_digest(model);out=root/('smoke-predictions' if smoke else 'predictions')/arm/f'{number:06d}';out.mkdir(parents=True,exist_ok=False)
        captured={};hook=model.encoder.layer0.relu1.register_forward_hook(lambda module,inputs,value:captured.update(early=value.detach()))
        full,_=build_data_loaders(cfg);records=[]
        for split,names in m['evaluation_ids'].items():
            if smoke:names=names[:2]
            loader=training_evaluation_loader(full,cfg,names) if split=='train' else build_evaluation_loader(cfg,dataset_name='synthetic_mri',sample_ids=names)
            for index,batch in enumerate(loader):
                name=names[index]
                if arm=='vessel_mask':assert torch.isin(batch[0],torch.tensor([0.,1.])).all()
                c=cache_patch(model,cfg,batch);gt=Path(plan['dataset'])/split/'spline'/(name+'.npz')
                with np.load(gt) as f:c.update(gt_nodes=f['nodes'],gt_edges=f['edges'],metric=f['metric'])
                cache=out/(name+'.npz');np.savez_compressed(cache,**c)
                features=None
                if number==summary['updates']:
                    features=out/(name+'-early.npy');np.save(features,captured['early'][0].cpu().numpy())
                records.append(dict(split=split,sample_id=name,subject=name.rsplit('_',1)[0],site=plan['sites'][name],
                    gt=str(gt),cache=str(cache),features=str(features) if features else None))
                if (index+1)%64==0:print(arm,number,split,index+1,'/',len(names),flush=True)
        hook.remove();assert state_digest(model)==frozen
        save(out/'records.json',records);save(out/'config.json',cfg);save(out/'summary.json',dict(completed=True,arm=arm,step=number,records=len(records),batching=1,final=number==summary['updates'],frozen_evaluation_state=True))
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['audit','smoke','benchmark','train','predict'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=ARMS);p.add_argument('--smoke',action='store_true');p.add_argument('--limit',type=int)
    a=p.parse_args();torch.set_num_threads(1)
    if a.mode=='audit':audit(a.root,a.limit)
    elif a.mode in ['smoke','benchmark']:benchmark(a.root,a.mode=='smoke')
    elif a.mode=='train':train(a.root,a.arm)
    else:predict(a.root,a.arm,a.smoke)
