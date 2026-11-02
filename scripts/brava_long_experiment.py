"""Controlled full-cohort BRAVA scratch/pretrained study, with unchanged K8 model."""
import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import signal
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAIN = ROOT.parent/'Graph-Native-Betti-Matching-Representation'
OUT = ROOT/'experiments/brava_long_20261027'
SEED = 271039
ARMS = ['scratch', 'synthetic']
MILESTONES = [25, 50, 100, 150, 200, 300]


def read(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def seed(number):
    import torch
    random.seed(number)
    np.random.seed(number % 2**32)
    torch.manual_seed(number)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(number)


def epoch_order(epoch, count=8768):
    return np.random.default_rng(SEED+int(epoch)).permutation(count)


def sample_seed(epoch, sample):
    payload = f'{SEED}:{epoch}:{sample}'.encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), 'little')


def lr_scale(step, steps_per_epoch, epochs=300, warmup_epochs=5):
    warmup = warmup_epochs*steps_per_epoch
    total = epochs*steps_per_epoch
    if step <= warmup:
        return .01+.99*step/warmup
    progress = (step-warmup)/(total-warmup)
    floor = 1./70.  # 7e-5 -> 1e-6; encoder receives the same scale at half LR.
    return floor+(1-floor)*.5*(1+math.cos(math.pi*min(1., progress)))


def summarize_updates(rows):
    """Aggregate losses shared by all updates; diagnostics may occur only once."""
    keys=set.intersection(*(set(row) for row in rows)) - {'step'}
    return {k:float(np.mean([r[k] for r in rows])) for k in sorted(keys)
            if all(isinstance(r[k],(int,float)) for r in rows)}


def expand_queries(source, target, source_queries, target_queries=256, relations=2):
    """Preserve every source tensor; move relation rows past new object queries."""
    assert set(source) == set(target), (set(source)-set(target), set(target)-set(source))
    result = dict(source)
    changed = []
    for name, value in source.items():
        if value.shape == target[name].shape:
            continue
        assert name == 'query_embed.weight', f'Unsupported source architecture mismatch: {name}'
        assert 0 < source_queries < target_queries
        assert value.shape[0] == source_queries+relations
        assert target[name].shape[0] == target_queries+relations
        assert value.shape[1:] == target[name].shape[1:]
        expanded = target[name].clone()
        expanded[:source_queries] = value[:source_queries]
        expanded[target_queries:target_queries+relations] = value[source_queries:source_queries+relations]
        result[name] = expanded
        changed.append(name)
    assert changed == ['query_embed.weight'] if source_queries != target_queries else not changed
    return result


def prepare():
    assert not (OUT/'plan.json').exists(), 'Do not overwrite an existing protocol'
    old = read(MAIN/'experiments/brava_graph_study/plan.json')
    d07 = read(MAIN/'experiments/brava_augmentation_20261026/plan.json')
    source_study = read(MAIN/'experiments/local_refinement_long/analysis.json')
    candidates = []
    for run_seed, arms in source_study['runs'].items():
        for step, record in arms['B0']['trajectory'].items():
            path = MAIN/f'experiments/local_refinement_long/{run_seed}/runs/B0/model-{int(step)}.pt'
            if path.exists():
                candidates.append(dict(path=str(path), seed=int(run_seed), step=int(step),
                    source_validation_edge_f1_2mm=record['metrics']['val']['2.0']['edge_f1']))
    assert candidates
    selected = max(candidates, key=lambda x: (x['source_validation_edge_f1_2mm'], x['step']))
    selected['sha256'] = sha(selected['path'])
    cfg = copy.deepcopy(d07['config'])
    cfg['model']['decoder']['object_queries'] = 256
    cfg['model']['node_refinement'] = {'enabled': False}
    cfg['model']['spline'].update(enabled=True, controls=8, samples=64)
    cfg['model']['node_type'] = {'enabled': False}
    cfg['data'].update(batch_size=16, validation_batch_size=8, train_augmentation=True,
                       input_modality='raw_mra')
    cfg['data']['datasets'] = {'synthetic_mri': dict(cfg['data']['datasets']['synthetic_mri'],
        root=old['dataset'], train_samples=None, validation_samples=None, patch_selection='all')}
    cfg['training']['epochs'] = 300
    cfg['training']['optimizer']['base_lr'] = 7e-5
    cfg['runtime'].update(workers=4, distributed=False)
    cfg['tracking']['enabled'] = False
    cfg['evaluation'].update(node_threshold=.5, edge_threshold=.5, node_type_degree_caps=False,
                              node_type_degree_penalty=0.)
    cfg['evaluation']['protocol']['max_detections'] = 256
    cfg['experiment'].update(seed=SEED, name='brava_long_20261027')
    assert cfg['loss']['node']['classification']['name'] == 'focal'
    assert cfg['loss']['node']['classification']['class_weights'] == [1., 1.]
    assert cfg['loss']['node']['classification']['focal_gamma'] == 2
    assert cfg['loss']['edge']['classification']['name'] == 'cross_entropy'
    assert cfg['loss']['edge']['classification']['class_weights'] == [1., 1.]
    assert cfg['loss']['edge']['balancing']['positive_to_negative_ratio'] == .15
    assert not cfg['loss']['edge']['candidates']['include_unmatched']
    assert not cfg['augmentation']['synthetic_mri']['zoom']['enabled']
    ids = {s: old['ids'][s] for s in ['train', 'val']}
    subjects = {s: old['subjects_by_id'][s] for names in ids.values() for s in names}
    assert len(ids['train']) == 8768 and len(ids['val']) == 1796
    groups = {s: sorted({subjects[x] for x in ids[s]}) for s in ids}
    assert len(groups['train']) == 39 and len(groups['val']) == 8
    assert not set(groups['train']) & set(groups['val'])
    plan = dict(experiment='BRAVA full-cohort long training', registry_id='D08', config=cfg,
        dataset=old['dataset'], manifest_sha256=old['manifest_sha256'], ids=ids, subjects=subjects,
        subject_groups=groups, seed=SEED, arms=ARMS, epochs=300, milestones=MILESTONES,
        train_monitor=d07['populations']['representative_train'], val_monitor=d07['ids']['val'],
        monitor_epochs=[0, 1, 5, 10, 15, 20]+list(range(25, 301, 5)),
        source=selected, source_candidates=candidates,
        source_selection='Highest SyntheticMRI validation edge F1 at 2mm among documented compatible B0 checkpoints. Baseline parent scored 0.5881, below selected B0. A/A-xyz/control refiners are incompatible with the requested original architecture and are excluded.',
        batch_candidates=[16, 32], workers=4, eval_batch_size=8,
        batch_selection='Among sizes passing finite gradient/resume/route checks with peak reserved memory below 85% VRAM, maximize the slower arm end-to-end patches/second. Select 32 if its throughput is at least 95% of 16; no validation-based selection.',
        optimizer=dict(name='AdamW', lr=7e-5, backbone_lr=3.5e-5, weight_decay=1e-4,
                       betas=[.9,.999], eps=1e-8, warmup_epochs=5, floor_lr=1e-6,
                       schedule='Linear 1%-100% warmup, then cosine over the entire 300 epochs; never reset at job boundaries'),
        augmentation='Existing quarter-turn rotations in DH/DW/HW plus Gaussian noise p=.35, std uniformly 0-.015, no zoom. Per-(epoch,sample) seeds are independent of arm/workers/resume.',
        gradient_route='graph_only', checkpoint_policy='Immutable model checkpoints at 25/50/100/150/200/300; resumable latest with optimizer and RNG every 5 epochs and at chunk/signal boundaries',
        training_chunks='25 full epochs per job, persistent optimizer/LR/global step; no validation early stopping',
        evaluation='Saved BN; same clean fixed train160/val256 monitors; full val1796 at milestones; unchanged thresholds .5 and positional 1/2mm plus common-v2 Branch/Curve',
        precision='float32; no AMP or gradient accumulation', max_global_h100=2,
        limitations='One target training seed and eight reused development subjects. Annotation-relative graph errors; no vessel masks. Initialization comparison is conditional on this shared schedule. No test images/targets accessed.',
        large_root='/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/brava_long_20261027')
    write(OUT/'plan.json', plan)
    print(json.dumps(dict(source=selected, train=len(ids['train']), val=len(ids['val']), groups=groups), indent=2))


def configuration(plan, batch_size, arm):
    cfg = copy.deepcopy(plan['config'])
    cfg['data']['batch_size'] = batch_size
    cfg['experiment']['name'] = f'brava_long_{arm}'
    return cfg


def base_dataset(plan, split, names, augment):
    from data.loaders.common import SamplePaths
    from data.loaders.synthetic_mri import SyntheticMRIDataset
    assert split in ['train', 'val']
    assert set(names) <= set(plan['ids'][split])
    root = Path(plan['dataset'])/split
    records = [SamplePaths(root/'raw'/f'{s}_data.nii.gz', root/'seg'/f'{s}_seg.nii.gz',
                            root/'vtp'/f'{s}_graph.vtp', s) for s in names]
    mean = plan['config']['data']['datasets']['synthetic_mri']['foreground_mean']
    return SyntheticMRIDataset(records, augment=augment, zoom_range=None, spline_controls=8,
        foreground_mean=mean, gaussian_noise_probability=.35, gaussian_noise_max_std=.015,
        input_modality='raw_mra')


class SeededDataset:
    """Use the unchanged loader with deterministic augmentation per occurrence."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        epoch, index = item
        number = sample_seed(epoch, self.base.records[index].sample_id)
        self.base.rng = random.Random(number)
        if self.base.intensity_transform is not None:
            self.base.intensity_transform.set_random_state(seed=number ^ 0x36B714F)
        return self.base[index]


class EpochBatches:
    def __init__(self, batch_size, count=8768):
        assert count % batch_size == 0
        self.batch_size, self.count, self.epoch = batch_size, count, 1

    def __len__(self):
        return self.count//self.batch_size

    def __iter__(self):
        order = epoch_order(self.epoch, self.count)
        for start in range(0, self.count, self.batch_size):
            yield [(self.epoch, int(i)) for i in order[start:start+self.batch_size]]


def batch_digest(batch):
    import torch
    result = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            result.update(str((array.shape, array.dtype)).encode())
            result.update(array.tobytes())
        elif isinstance(value, dict):
            for k in sorted(value):
                result.update(str(k).encode()); visit(value[k])
        elif isinstance(value, (tuple, list)):
            for v in value:
                visit(v)
        else:
            result.update(str(value).encode())
    visit(batch)
    return result.digest()


def setup(plan, arm, batch_size, device='cuda', initialize=True):
    import torch
    from configs import validate_config
    from models import build_model
    from models.matcher import build_matcher
    from scripts.train_ixi_transfer import TransferCriterion
    cfg = configuration(plan, batch_size, arm)
    validate_config(cfg)
    seed(SEED)
    model = build_model(cfg)
    assert model.node_refinement is None and model.spline_head.controls == 8
    migration = None
    if arm == 'synthetic' and initialize:
        cp = torch.load(plan['source']['path'], map_location='cpu', mmap=True)
        source_cfg = cp['config']
        assert not source_cfg['model']['node_refinement'].get('enabled', False)
        for name in ['encoder', 'spline', 'matcher']:
            assert source_cfg['model'][name] == cfg['model'][name], name
        assert source_cfg['model']['decoder']['relation_tokens'] == 2
        assert source_cfg['model']['decoder']['dummy_tokens'] == 0
        state = expand_queries(cp['net'], model.state_dict(), source_cfg['model']['decoder']['object_queries'])
        model.load_state_dict(state, strict=True)
        migration = dict(source_queries=source_cfg['model']['decoder']['object_queries'], target_queries=256,
            source_relation_rows=[120,121], target_relation_rows=[256,257],
            policy='Copy all source object rows, move shared relation rows, keep extra object rows from the identical seeded scratch model; strict load of every other tensor.')
        del cp, state
    model.requires_grad_(True)
    # Preserve the established recipe's inactive relation-attention parameters.
    for layer in model.decoder.decoder.layers:
        if layer.rln_attn:
            layer.cross_attn2.requires_grad_(False); layer.norm12.requires_grad_(False)
    model.to(device)
    criterion = TransferCriterion(cfg, build_matcher(cfg), model.relation_embed,
        spline_head=model.spline_head, route='graph_only').to(device)
    criterion.set_training_progress(1, 0.)
    settings = plan['optimizer']
    groups = [dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')],
                   lr=settings['backbone_lr'], base_lr=settings['backbone_lr']),
              dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')],
                   lr=settings['lr'], base_lr=settings['lr'])]
    optimizer = torch.optim.AdamW(groups, betas=tuple(settings['betas']), eps=settings['eps'], weight_decay=settings['weight_decay'])
    return cfg, model, criterion, optimizer, migration


def save_checkpoint(path, plan, cfg, model, opt, epoch, step, **extra):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    payload = dict(net=model.state_dict(), config=cfg, epoch=epoch, step=step,
        optimizer=None if opt is None else opt.state_dict(), plan_sha256=sha(OUT/'plan.json'),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), **extra)
    torch.save(payload, temp); temp.replace(path)


def cpu_preflight():
    import torch
    from data.augmentations import assert_valid_coordinates
    plan = read(OUT/'plan.json')
    assert sha(Path(plan['dataset'])/'manifest.json') == plan['manifest_sha256']
    assert sha(plan['source']['path']) == plan['source']['sha256']
    audit = []
    for split in ['train', 'val']:
        for i, sample in enumerate(plan['ids'][split]):
            root = Path(plan['dataset'])/split
            for directory, suffix in [('raw','_data.nii.gz'),('seg','_seg.nii.gz'),('vtp','_graph.vtp')]:
                assert (root/directory/(sample+suffix)).is_file()
            with np.load(root/'spline'/f'{sample}.npz') as gt:
                nodes, edges = gt['nodes'], gt['edges']
                assert 0 < len(nodes) <= 256 and len(edges) > 0
                assert edges.min() >= 0 and edges.max() < len(nodes)
                assert gt['controls'].shape == (len(edges),8,3)
                assert gt['curves'].shape == (len(edges),64,3)
                for name in ['nodes','controls','curves','guides','metric','full_paths']:
                    assert np.isfinite(gt[name]).all(), (sample,name)
                assert abs(np.linalg.det(gt['metric'])) > 1e-6
                np.testing.assert_allclose(gt['controls'][:,[0,-1]],nodes[edges],atol=2e-6)
                assert_valid_coordinates(torch.from_numpy(nodes),(64,64,64),allow_patch_faces=True)
                audit.append(dict(split=split,sample=sample,nodes=len(nodes),edges=len(edges),
                                  target_sha256=sha(root/'spline'/f'{sample}.npz')))
            if i % 512 == 0:
                print('target audit',split,i,len(plan['ids'][split]),flush=True)
    cfg, scratch, _, _, _ = setup(plan,'scratch',16,'cpu')
    fresh = scratch.query_embed.weight.detach().clone()
    del scratch
    _, pretrained, _, _, migration = setup(plan,'synthetic',16,'cpu')
    torch.testing.assert_close(fresh[120:256],pretrained.query_embed.weight[120:256],rtol=0,atol=0)
    source = torch.load(plan['source']['path'],map_location='cpu',mmap=True)['net']
    for name,value in source.items():
        current = pretrained.state_dict()[name]
        if name == 'query_embed.weight':
            torch.testing.assert_close(current[:120],value[:120],rtol=0,atol=0)
            torch.testing.assert_close(current[256:258],value[120:122],rtol=0,atol=0)
        else:
            torch.testing.assert_close(current,value,rtol=0,atol=0)
    write(OUT/'target-audit.json',audit)
    write(OUT/'cpu-preflight.json',dict(passed=True,train_patches=8768,val_patches=1796,
        max_nodes=max(r['nodes'] for r in audit),query_expansion=migration,
        every_other_source_tensor_exact=True,extra_rows_match_scratch=True,test_data_accessed=False,
        source_sha256=plan['source']['sha256'],manifest_sha256=plan['manifest_sha256'],
        target_audit_sha256=sha(OUT/'target-audit.json')))
    print('CPU preflight passed',flush=True)


def update(model, criterion, opt, batch, step, steps_per_epoch, diagnostic=False):
    import torch
    from training.engine import _move_graph_batch
    seed(SEED+step)
    volumes, targets = _move_graph_batch(batch, 'cuda', 'image', True)
    scale = lr_scale(step, steps_per_epoch)
    for group in opt.param_groups:
        group['lr'] = group['base_lr']*scale
    model.train(); opt.zero_grad(set_to_none=True)
    tokens, predictions, _ = model(volumes)
    losses = criterion(tokens, predictions, targets)
    assert torch.isfinite(losses['total']), f'Nonfinite loss at {step}'
    result = {}
    if diagnostic:
        shared = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('spline_head.')]
        grads = torch.autograd.grad(losses['spline'],shared,retain_graph=True,allow_unused=True)
        assert all(g is None or not bool(g.abs().sum()) for g in grads), 'Spline gradient entered detector'
        result['geometry_to_detector_zero'] = True
    losses['total'].backward()
    if diagnostic:
        for prefix in ['encoder.','decoder.','class_embed.','coord_embed.','relation_embed.','spline_head.']:
            values = [p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            assert values and all(torch.isfinite(v).all() for v in values), prefix
            norm = float(sum(v.square().sum() for v in values).sqrt())
            assert norm > 0, prefix
            result[prefix+'gradient_norm'] = norm
    opt.step()
    result.update({k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1})
    result.update(step=step,lr=7e-5*scale)
    return result


def gpu_smoke():
    import torch
    from torch.utils.data import DataLoader
    from data.loaders.common import image_graph_collate
    assert read(OUT/'cpu-preflight.json')['passed']
    plan = read(OUT/'plan.json'); rows=[]; input_hashes={}
    for size in plan['batch_candidates']:
        for arm in ARMS:
            print('smoke/benchmark',arm,'batch',size,flush=True)
            cfg,model,criterion,opt,_ = setup(plan,arm,size)
            ds = SeededDataset(base_dataset(plan,'train',plan['ids']['train'],True))
            sampler = EpochBatches(size)
            loader = DataLoader(ds,batch_sampler=sampler,num_workers=4,collate_fn=image_graph_collate,
                                pin_memory=True,persistent_workers=False,prefetch_factor=2)
            torch.cuda.reset_peak_memory_stats()
            timings=[]; first_hash=None; last=None
            iterator=iter(loader)
            for number in range(1,16):
                torch.cuda.synchronize(); start=time.perf_counter()
                batch=next(iterator)
                digest=batch_digest(batch).hex()
                if first_hash is None:first_hash=digest
                last=update(model,criterion,opt,batch,number,len(sampler),diagnostic=number==1)
                torch.cuda.synchronize()
                if number>3:timings.append(time.perf_counter()-start)
            key=str(size)
            if key in input_hashes:assert input_hashes[key]==first_hash,'Arm-dependent input augmentation'
            input_hashes[key]=first_hash
            folder=OUT/'large'/'smoke'/f'{arm}-b{size}';folder.mkdir(parents=True,exist_ok=True)
            save_checkpoint(folder/'resume.pt',plan,cfg,model,opt,0,15,arm=arm,batch_size=size)
            cp=torch.load(folder/'resume.pt',map_location='cpu',mmap=True)
            reference={k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(cp['net'],strict=True);opt.load_state_dict(cp['optimizer'])
            assert cp['step']==15 and all(torch.equal(reference[k],v) for k,v in model.state_dict().items())
            assert all(int(v['step'])==15 for v in opt.state.values() if 'step' in v)
            seconds=float(np.median(timings))
            rows.append(dict(arm=arm,batch_size=size,seconds_per_update=seconds,patches_per_second=size/seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                first_batch_sha256=first_hash,finite_training=True,strict_resume=True,last=last))
            write(OUT/'benchmark-progress.json',rows)
            del cp,reference,iterator,loader,ds,model,criterion,opt,batch;torch.cuda.empty_cache()
    valid={size:all(r['peak_reserved_bytes']<.85*r['total_memory_bytes'] for r in rows if r['batch_size']==size)
           for size in plan['batch_candidates']}
    speed={size:min(r['patches_per_second'] for r in rows if r['batch_size']==size) for size in plan['batch_candidates']}
    assert valid[16] or valid[32], 'Neither shared batch size has sufficient memory headroom'
    chosen=32 if valid[32] and (not valid[16] or speed[32]>=.95*speed[16]) else 16
    result=dict(passed=True,batch_size=chosen,workers=4,rows=rows,slowest_patches_per_second=speed[chosen],
        updates_per_epoch=8768//chosen,total_updates=300*(8768//chosen),
        estimated_training_gpu_hours_per_arm=8768*300/speed[chosen]/3600,
        source_sha256=plan['source']['sha256'],equal_augmented_inputs=True,benchmark_updates_discarded=True,
        job_id=os.environ.get('SLURM_JOB_ID'))
    # Exercise the exact prediction/CPU metric path before authorizing long jobs.
    cfg,model,criterion,opt,_=setup(plan,'synthetic',chosen)
    from scripts.brava_long_evaluation import snapshot
    snapshot(plan,cfg,model,criterion,'synthetic',-1,smoke=True)
    write(OUT/'gpu-smoke.json',result)
    write(OUT/'batch-selection.json',result)
    print(json.dumps(result,indent=2),flush=True)


def train_chunk(arm, end_epoch):
    import torch
    from torch.utils.data import DataLoader
    from data.loaders.common import image_graph_collate
    from scripts.brava_long_evaluation import snapshot
    assert read(OUT/'cpu-preflight.json')['passed'] and read(OUT/'gpu-smoke.json')['passed']
    for name,value in read(OUT/'frozen-source-sha256.json').items():
        assert sha(ROOT/name)==value, f'Frozen source changed: {name}'
    plan=read(OUT/'plan.json'); selection=read(OUT/'batch-selection.json'); size=selection['batch_size']
    run=OUT/'large'/'runs'/arm;run.mkdir(parents=True,exist_ok=True)
    latest=run/'latest.pt'; initialize=not latest.exists()
    cfg,model,criterion,opt,migration=setup(plan,arm,size,initialize=initialize)
    epoch=0;step=0;history=[]
    if not initialize:
        cp=torch.load(latest,map_location='cpu',mmap=True)
        assert cp['plan_sha256']==sha(OUT/'plan.json') and cp['arm']==arm and cp['batch_size']==size
        assert not cp.get('interrupted',False), 'Recover interrupted updates explicitly; do not skip an epoch'
        model.load_state_dict(cp['net'],strict=True);opt.load_state_dict(cp['optimizer'])
        epoch,step=cp['epoch'],cp['step'];del cp
        history=read(run/'history.json')
    assert epoch==end_epoch-25, (arm,epoch,end_epoch)
    assert step==epoch*selection['updates_per_epoch']
    if initialize:
        write(run/'config.json',cfg)
        write(run/'initialization.json',dict(arm=arm,source=plan['source'] if arm=='synthetic' else None,
              migration=migration,seed=SEED,extra_query_sha256=hashlib.sha256(model.query_embed.weight[120:256].detach().cpu().numpy().tobytes()).hexdigest()))
        snapshot(plan,cfg,model,criterion,arm,0)
    dataset=SeededDataset(base_dataset(plan,'train',plan['ids']['train'],True))
    sampler=EpochBatches(size)
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=4,collate_fn=image_graph_collate,
                      pin_memory=True,persistent_workers=True,prefetch_factor=2)
    stop=[False]
    for sig in [signal.SIGUSR1,signal.SIGTERM]:
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    start=time.monotonic();torch.cuda.reset_peak_memory_stats();epoch_records=[]
    for epoch in range(epoch+1,end_epoch+1):
        sampler.epoch=epoch; chain=bytes(32); losses=[]; epoch_start=time.monotonic()
        for offset,batch in enumerate(loader):
            chain=hashlib.sha256(chain+batch_digest(batch)).digest()
            step+=1
            row=update(model,criterion,opt,batch,step,len(sampler),diagnostic=step==1)
            losses.append(row)
            if step%50==0 or offset==0:
                print(arm,'epoch',epoch,'batch',offset+1,'/',len(sampler),'step',step,
                      'loss',round(row['total'],5),'lr',row['lr'],flush=True)
            if stop[0]:
                save_checkpoint(run/f'interrupted-{os.environ.get("SLURM_JOB_ID","local")}.pt',
                    plan,cfg,model,opt,epoch-1,step,arm=arm,batch_size=size,interrupted=True,
                    incomplete_epoch=epoch,completed_batches=offset+1,input_chain=chain.hex())
                raise RuntimeError('Interrupted partial epoch saved; block dependent jobs and recover explicitly')
        record=dict(epoch=epoch,step=step,patches=len(dataset),updates=len(sampler),
            patch_order_sha256=hashlib.sha256(epoch_order(epoch).astype('<i8').tobytes()).hexdigest(),
            full_augmented_batch_chain_sha256=chain.hex(),epoch_seconds=time.monotonic()-epoch_start,
            losses=summarize_updates(losses))
        history.append(record);epoch_records.append(record);write(run/'history.json',history)
        if epoch in MILESTONES:
            path=run/f'model-epoch-{epoch:03d}.pt'
            assert not path.exists(), 'Never overwrite an immutable milestone'
            save_checkpoint(path,plan,cfg,model,None,epoch,step,arm=arm,batch_size=size)
            write(run/f'model-epoch-{epoch:03d}.json',dict(path=str(path),sha256=sha(path),epoch=epoch,step=step))
        if epoch in plan['monitor_epochs']:
            snapshot(plan,cfg,model,criterion,arm,epoch)
        if epoch%5==0 or epoch==end_epoch:
            save_checkpoint(latest,plan,cfg,model,opt,epoch,step,arm=arm,batch_size=size,interrupted=False)
    elapsed=time.monotonic()-start
    write(run/f'chunk-{end_epoch:03d}.json',dict(completed=True,arm=arm,end_epoch=end_epoch,step=step,
        gpu_job_id=os.environ.get('SLURM_JOB_ID'),array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
        seconds=elapsed,gpu_hours=elapsed/3600,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),epoch_records=epoch_records))
    if end_epoch==300:
        write(run/'complete.json',dict(completed=True,arm=arm,epochs=300,updates=step,
                                      source=plan['source'] if arm=='synthetic' else None))
    print('Completed chunk',arm,end_epoch,elapsed,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','preflight','smoke','train'])
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--end-epoch',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='preflight':cpu_preflight()
    elif args.mode=='smoke':gpu_smoke()
    else:train_chunk(args.arm,args.end_epoch)
