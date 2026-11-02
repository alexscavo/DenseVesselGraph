"""Literature-scale RelationFormer B0 control on IXI (D08 recipe, synthetic init).

One arm: the D08 `synthetic` arm (SyntheticMRI B0 model-12000, 120->256 query
expansion) trained on the full IXI train split (10,000 patches) for 100 epochs
at batch 32, with every other D08 setting unchanged (K8 cubic B-spline
`graph_only`, node focal gamma 2, matched-edge CE ratio .15, AdamW 7e-5 /
encoder 3.5e-5, wd 1e-4, 5-epoch warmup then cosine to 1e-6 over the whole
budget, quarter turns + Gaussian noise, no zoom, saved-BN evaluation).

Evaluation populations are those of the competing dense model
(experiments/multidataset_continuous/plan.json -> datasets.IXI): val_ids (512)
and monitor_train (256). Thresholds: fixed .5/.5 and a train-monitor-selected
pair (node by node F1 at 2 mm, then edge by edge F1 at 2 mm). Validation is
never used for any selection.

Modes: prepare | preflight | cpu-forward | stage | smoke | train | evaluate |
collect | submit-smoke | submit-train.
"""
import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]  # the repository or its frozen copy
sys.path.insert(0, str(CODE))
REPO = Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/Graph-Native-Betti-Matching-Representation')
OUT = REPO/'experiments/ixi_long_20261030'
LARGE = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_long_20261030')
D08 = Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/GNBM/gnbm-brava-long-20261027/experiments/brava_long_20261027')
IXI = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI/transfer_splines_k8_20261015')
EVAL_PLAN = REPO/'experiments/multidataset_continuous/plan.json'
SLURM = 'cluster/jean_zay/ixi_long_gpu_20261030.slurm'
SEED = 271039            # D08 seed: identical extra query rows / per-step seeding
EPOCHS = 100
BATCH = 32
WARMUP_EPOCHS = 5
MILESTONES = [25, 50, 75, 100]   # immutable model checkpoints
EVAL_EPOCHS = [0, 25, 50, 100]   # full protocol on train-monitor 256 + val 512
GRID = [round(float(x), 2) for x in np.arange(.05, .951, .05)]
CANDIDATE_NODE_THRESHOLD = GRID[0]
FIXED = 'fixed_0p5'
SELECTED = 'train_selected'
POPULATIONS = [('train_monitor', 'train'), ('val', 'val')]


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


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix+k+'.'))
        else:
            out[prefix+k] = v
    return out


# ---------------------------------------------------------------- protocol
def seed(number):
    import torch
    random.seed(number)
    np.random.seed(number % 2**32)
    torch.manual_seed(number)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(number)


def epoch_order(epoch, count):
    return np.random.default_rng(SEED+int(epoch)).permutation(count)


def sample_seed(epoch, sample):
    payload = f'{SEED}:{epoch}:{sample}'.encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), 'little')


def lr_scale(step, steps_per_epoch, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS):
    """D08 schedule: linear 1%->100% warmup, then cosine over the whole budget to 1e-6."""
    warmup = warmup_epochs*steps_per_epoch
    total = epochs*steps_per_epoch
    if step <= warmup:
        return .01+.99*step/warmup
    progress = (step-warmup)/(total-warmup)
    floor = 1./70.  # 7e-5 -> 1e-6; encoder receives the same scale at half LR.
    return floor+(1-floor)*.5*(1+math.cos(math.pi*min(1., progress)))


def summarize_updates(rows):
    keys = set.intersection(*(set(row) for row in rows)) - {'step'}
    return {k: float(np.mean([r[k] for r in rows])) for k in sorted(keys)
            if all(isinstance(r[k], (int, float)) for r in rows)}


def expand_queries(source, target, source_queries, target_queries=256, relations=2):
    """D08: preserve every source tensor; move relation rows past new object queries."""
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


class EpochBatches:
    """Seeded permutation per epoch; full batches only (10,000 % 32 = 16 patches dropped per epoch)."""
    def __init__(self, batch_size, count):
        self.batch_size, self.count, self.epoch = batch_size, count, 1

    def __len__(self):
        return self.count//self.batch_size

    def __iter__(self):
        order = epoch_order(self.epoch, self.count)
        for start in range(0, len(self)*self.batch_size, self.batch_size):
            yield [(self.epoch, int(i)) for i in order[start:start+self.batch_size]]


class SeededDataset:
    """Unchanged loader with deterministic augmentation per (epoch, sample)."""
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


# ------------------------------------------------------------------ prepare
def prepare():
    assert not (OUT/'plan.json').exists(), 'Do not overwrite an existing protocol'
    d08 = read(D08/'plan.json')
    d08_effective = read(D08/'large/runs/synthetic/config.json')
    evaluation = read(EVAL_PLAN)['datasets']['IXI']
    manifest = read(IXI/'manifest.json')
    assert evaluation['root'] == str(IXI)
    assert sha(IXI/'manifest.json') == evaluation['manifest_sha256']
    train, val = sorted(manifest['train_ids']), sorted(manifest['validation_ids'])
    assert len(train) == len(set(train)) == 10000 and len(val) == len(set(val)) == 2649
    for split, names in [('train', train), ('val', val)]:
        on_disk = sorted(p.name[:-len('_data.nii.gz')] for p in (IXI/split/'raw').iterdir())
        assert on_disk == names, split
    monitor, val_eval = evaluation['monitor_train'], evaluation['val_ids']
    assert len(monitor) == len(set(monitor)) == 256 and set(monitor) <= set(train)
    assert len(val_eval) == len(set(val_eval)) == 512 and set(val_eval) <= set(val)
    patient = lambda s: s.split('_')[1]
    assert not {patient(s) for s in train} & {patient(s) for s in val}
    cfg = copy.deepcopy(d08_effective)
    cfg['data']['datasets']['synthetic_mri']['root'] = str(IXI)
    cfg['data']['batch_size'] = BATCH
    cfg['training']['epochs'] = EPOCHS
    cfg['experiment']['name'] = 'ixi_long_20261030_synthetic'
    source = dict(d08['source'])
    assert Path(source['path']).is_file()
    optimizer = dict(d08['optimizer'], schedule='Linear 1%-100% warmup over 5 epochs, then cosine to 1e-6 over the entire 100 epochs (31,200 updates); never reset at job boundaries')
    updates_per_epoch = 10000//BATCH
    differences = {k: dict(d08=v, ixi=flatten(cfg).get(k)) for k, v in flatten(d08_effective).items()
                   if flatten(cfg).get(k) != v}
    plan = dict(experiment='IXI literature-scale RelationFormer B0 control (D08 recipe, synthetic init)',
        config=cfg, dataset=str(IXI), manifest_sha256=evaluation['manifest_sha256'],
        ids=dict(train=train, val=val), populations=dict(train_monitor=monitor, val=val_eval),
        evaluation_plan=str(EVAL_PLAN), evaluation_plan_sha256=sha(EVAL_PLAN),
        d08_record=str(D08), d08_plan_sha256=sha(D08/'plan.json'),
        d08_effective_config_sha256=sha(D08/'large/runs/synthetic/config.json'),
        config_differences_vs_d08_effective=differences,
        seed=SEED, arm='synthetic', epochs=EPOCHS, batch_size=BATCH, updates_per_epoch=updates_per_epoch,
        total_updates=EPOCHS*updates_per_epoch, dropped_patches_per_epoch=10000 % BATCH,
        milestones=MILESTONES, evaluation_epochs=EVAL_EPOCHS, source=source,
        optimizer=optimizer, augmentation=d08['augmentation'], gradient_route='graph_only',
        thresholds=dict(fixed=[.5, .5], grid=GRID, candidate_node_threshold=CANDIDATE_NODE_THRESHOLD,
            comparison='strict > (as infer_graphs)', selection='train_monitor (256 training patches) only: node threshold maximizing node F1 at 2 mm, then edge threshold maximizing edge F1 at 2 mm given that node threshold; ties -> higher threshold. Applied unchanged to validation.'),
        metrics='Physical node/edge P/R/F1 at 2 mm (and 1 mm) via diagnose_branch_relations.match_nodes + ixi_query_diagnostics.edge_accounting; common-v2 Branch/Curve F1 via metrics/common_vessel_metrics; node AP (physical 2 mm, all 256 proposals), edge ranking AP (D08 all-retained/conditional), RelationFormer box mAP (training/evaluation/metrics).',
        checkpoint_policy='Immutable model checkpoints at 25/50/75/100; resumable latest (model+optimizer+RNG) after every epoch',
        precision='float32; no AMP or gradient accumulation', large_root=str(LARGE),
        limitations='One training seed. Annotation-relative graph errors. No test images/targets accessed.')
    write(OUT/'plan.json', plan)
    write(OUT/'config-differences-vs-d08.json', differences)
    print(json.dumps(dict(train=len(train), val_eval=len(val_eval), monitor=len(monitor),
                          updates=plan['total_updates'], differences=differences), indent=2))


# -------------------------------------------------------------------- model
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


def setup(plan, device='cuda', initialize=True, cpu_ops=False):
    import torch
    from configs import validate_config
    from models import build_model
    from models.matcher import build_matcher
    from scripts.train_ixi_transfer import TransferCriterion
    cfg = copy.deepcopy(plan['config'])
    if cpu_ops:  # login-node test only: PyTorch deformable attention instead of the CUDA op
        cfg['model']['decoder']['use_cuda_extension'] = False
    validate_config(cfg)
    seed(SEED)
    model = build_model(cfg)
    assert model.node_refinement is None and model.spline_head.controls == 8
    migration = None
    if initialize:
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
            source_relation_rows=[120, 121], target_relation_rows=[256, 257],
            policy='Copy all source object rows, move shared relation rows, keep extra object rows from the identical seeded scratch model; strict load of every other tensor.')
        del cp, state
    model.requires_grad_(True)
    for layer in model.decoder.decoder.layers:  # D08: preserve inactive relation-attention parameters
        if layer.rln_attn:
            layer.cross_attn2.requires_grad_(False); layer.norm12.requires_grad_(False)
    model.to(device)
    criterion = TransferCriterion(cfg, build_matcher(cfg), model.relation_embed,
        spline_head=model.spline_head, route='graph_only').to(device)
    criterion.set_training_progress(1, 0.)
    settings = plan['optimizer']
    groups = [dict(params=[p for n, p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')],
                   lr=settings['backbone_lr'], base_lr=settings['backbone_lr']),
              dict(params=[p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')],
                   lr=settings['lr'], base_lr=settings['lr'])]
    optimizer = torch.optim.AdamW(groups, betas=tuple(settings['betas']), eps=settings['eps'],
                                  weight_decay=settings['weight_decay'])
    return cfg, model, criterion, optimizer, migration


def extra_query_sha(model):
    return hashlib.sha256(model.query_embed.weight[120:256].detach().cpu().numpy().tobytes()).hexdigest()


def save_checkpoint(path, cfg, model, opt, epoch, step, **extra):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    payload = dict(net=model.state_dict(), config=cfg, epoch=epoch, step=step,
        optimizer=None if opt is None else opt.state_dict(), plan_sha256=sha(OUT/'plan.json'),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), **extra)
    torch.save(payload, temp); temp.replace(path)


def update(model, criterion, opt, batch, step, steps_per_epoch, epochs, warmup_epochs, base_lr,
           diagnostic=False, device='cuda'):
    import torch
    from training.engine import _move_graph_batch
    seed(SEED+step)
    volumes, targets = _move_graph_batch(batch, device, 'image', True)
    scale = lr_scale(step, steps_per_epoch, epochs, warmup_epochs)
    for group in opt.param_groups:
        group['lr'] = group['base_lr']*scale
    model.train(); opt.zero_grad(set_to_none=True)
    tokens, predictions, _ = model(volumes)
    losses = criterion(tokens, predictions, targets)
    assert torch.isfinite(losses['total']), f'Nonfinite loss at {step}'
    result = {}
    if diagnostic:
        shared = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('spline_head.')]
        grads = torch.autograd.grad(losses['spline'], shared, retain_graph=True, allow_unused=True)
        assert all(g is None or not bool(g.abs().sum()) for g in grads), 'Spline gradient entered detector'
        result['geometry_to_detector_zero'] = True
    losses['total'].backward()
    if diagnostic:
        for prefix in ['encoder.', 'decoder.', 'class_embed.', 'coord_embed.', 'relation_embed.', 'spline_head.']:
            values = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            assert values and all(torch.isfinite(v).all() for v in values), prefix
            norm = float(sum(v.square().sum() for v in values).sqrt())
            assert norm > 0, prefix
            result[prefix+'gradient_norm'] = norm
    opt.step()
    result.update({k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1})
    result.update(step=step, lr=base_lr*scale)
    return result


# ------------------------------------------------------------- evaluation
def load_gt(path):
    with np.load(path) as z:
        return dict(nodes=z['nodes'], edges=z['edges'].reshape(-1, 2), metric=z['metric'])


def infer(plan, model, criterion, split, names, policies, device='cuda', workers=2):
    """Clean saved-BN inference. Candidates: all pairs among nodes > CANDIDATE_NODE_THRESHOLD."""
    import torch
    from torch.utils.data import DataLoader
    from data.loaders.common import image_graph_collate
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    ds = base_dataset(plan, split, names, False)
    loader = DataLoader(ds, batch_size=8, num_workers=workers, collate_fn=image_graph_collate,
                        shuffle=False, pin_memory=device == 'cuda')
    kwargs = dict(object_queries=256, relation_tokens=2, node_type_degree_caps=False, node_type_degree_penalty=0.)
    caches, losses = [], []
    model.eval()
    with torch.no_grad():
        for offset, batch in enumerate(loader):
            volumes, targets = _move_graph_batch(batch, device, 'image', True)
            tokens, predictions, _ = model(volumes)
            seed(SEED+1000000+offset)
            loss = criterion(tokens, predictions, targets)
            scalar = {k: float(v) for k, v in loss.items() if torch.is_tensor(v) and v.numel() == 1}
            assert all(np.isfinite(v) for v in scalar.values())
            losses.append(dict(count=len(volumes), **scalar))
            probabilities = predictions['pred_logits'].softmax(-1)[..., 1]  # identical op to infer_graphs
            candidate = infer_graphs(tokens, predictions, model.relation_embed,
                                     node_threshold=CANDIDATE_NODE_THRESHOLD, edge_threshold=-1., **kwargs)
            metrics = [g['metric'] for g in targets['spline']]
            decoded = {name: infer_graphs(tokens, predictions, model.relation_embed, node_threshold=tn,
                                          edge_threshold=te, spline_head=model.spline_head,
                                          spline_metrics=metrics, **kwargs)
                       for name, (tn, te) in policies.items()}
            for j, c in enumerate(candidate):
                sample = names[offset*8+j]
                cache = dict(raw_nodes=predictions['pred_nodes'][j, :, :3].cpu().numpy(),
                             raw_scores=probabilities[j].cpu().numpy(),
                             candidate_query_ids=c['query_ids'].numpy(),
                             candidate_edges=c['edges'].numpy(), candidate_scores=c['edge_scores'].numpy())
                for name, graphs in decoded.items():
                    g = graphs[j]
                    for key in ['query_ids', 'nodes', 'boxes', 'node_scores', 'edges', 'edge_scores', 'spline_controls']:
                        cache[f'{name}__{key}'] = g[key].numpy()
                caches.append((sample, cache))
            if offset % 16 == 0:
                print('infer', split, offset*8, '/', len(names), flush=True)
    loss = {k: sum(r[k]*r['count'] for r in losses)/sum(r['count'] for r in losses)
            for k in losses[0] if k != 'count'}
    return caches, loss


def subset(cache, node_threshold):
    """Candidate graph restricted to nodes > node_threshold (strict, float32 as infer_graphs)."""
    qids = np.flatnonzero(cache['raw_scores'] > np.float32(node_threshold))
    cand = cache['candidate_query_ids']
    assert np.isin(qids, cand).all(), 'Node threshold below candidate threshold'
    keep = np.isin(cand, qids)
    remap = -np.ones(len(cand), dtype=int); remap[keep] = np.arange(keep.sum())
    pairs = cache['candidate_edges'].reshape(-1, 2)
    valid = keep[pairs[:, 0]] & keep[pairs[:, 1]] if len(pairs) else np.zeros(0, bool)
    local = remap[pairs[valid]].reshape(-1, 2)
    return qids, local, cache['candidate_scores'][valid]


def node_curve_task(item):
    from scripts.diagnose_branch_relations import match_nodes
    cache, gt_path = item
    gt = load_gt(gt_path); m = gt['metric']; truth = gt['nodes']@m.T
    out = {}
    for t in GRID:
        ids = np.flatnonzero(cache['raw_scores'] > np.float32(t))
        out[t] = (len(match_nodes(cache['raw_nodes'][ids]@m.T, truth, 2.)), len(ids), len(gt['nodes']))
    return out


def edge_curve_task(item):
    from scripts.diagnose_branch_relations import match_nodes
    from scripts.ixi_query_diagnostics import edge_accounting
    cache, gt_path, tn = item
    gt = load_gt(gt_path); m = gt['metric']
    qids, pairs, scores = subset(cache, tn)
    mapping = match_nodes(cache['raw_nodes'][qids]@m.T, gt['nodes']@m.T, 2.)
    _, _, labels, _, _ = edge_accounting(np.arange(len(qids)), pairs, scores, np.ones(len(pairs), bool),
                                         mapping, gt['edges'], len(gt['nodes']))
    return scores, labels, len({tuple(sorted(map(int, e))) for e in gt['edges']})


def select_thresholds(pool, caches, gt_paths):
    """Training-monitor-only selection: node F1 first, then edge F1 at that node threshold."""
    counts = pool.map(node_curve_task, [(c, g) for (_, c), g in zip(caches, gt_paths)], chunksize=4)
    node_curve = []
    for t in GRID:
        tp, n, g = (sum(x[t][i] for x in counts) for i in range(3))
        node_curve.append(dict(threshold=t, node_tp=tp, nodes=n, gt_nodes=g, node_f1=2*tp/(n+g) if n+g else 1.,
                               node_precision=tp/n if n else None, node_recall=tp/g if g else None))
    tn = max(node_curve, key=lambda r: (r['node_f1'], r['threshold']))['threshold']
    parts = pool.map(edge_curve_task, [(c, g, tn) for (_, c), g in zip(caches, gt_paths)], chunksize=4)
    scores = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.float32)
    labels = np.concatenate([p[1] for p in parts]).astype(bool) if parts else np.zeros(0, bool)
    gt_edges = sum(p[2] for p in parts)
    edge_curve = []
    for t in GRID:
        chosen = scores > np.float32(t); tp = int(labels[chosen].sum()); n = int(chosen.sum())
        edge_curve.append(dict(threshold=t, tp=tp, edges=n, gt_edges=gt_edges, edge_f1=2*tp/(n+gt_edges) if n+gt_edges else 1.,
                               precision=tp/n if n else None, recall=tp/gt_edges if gt_edges else None))
    te = max(edge_curve, key=lambda r: (r['edge_f1'], r['threshold']))['threshold']
    return dict(node=tn, edge=te, node_curve=node_curve, edge_curve=edge_curve,
                selection_population='train_monitor', validation_used=False)


def node_ap_state(raw_nodes, raw_scores, gt, tolerance):
    """Score-ordered greedy matching of all proposals to unmatched GT within tolerance (COCO style)."""
    order = np.argsort(-raw_scores, kind='stable')
    d = np.linalg.norm((raw_nodes[order][:, None]-gt['nodes'][None])@gt['metric'].T, axis=-1)
    matched = np.zeros(len(gt['nodes']), bool); tp = np.zeros(len(order), bool)
    for i in range(len(order)):
        c = np.flatnonzero(~matched & (d[i] <= tolerance))
        if len(c):
            j = c[np.argmin(d[i, c])]; matched[j] = True; tp[i] = True
    return raw_scores[order], tp, len(gt['nodes'])


def patch_task(item):
    """All per-patch metrics for every decoded policy (runs in a CPU worker)."""
    from scripts.diagnose_branch_relations import match_nodes
    from scripts.ixi_query_diagnostics import edge_accounting, betti
    from metrics.common_vessel_metrics import evaluate_common_vessels
    from training.evaluation.metrics import graph_detection_states
    sample, cache, gt_path, policies, protocol = item
    gt = load_gt(gt_path); m = gt['metric']
    result = dict(sample=sample, node_ap={}, policies={})
    for tol in [1., 2.]:
        result['node_ap'][str(tol)] = node_ap_state(cache['raw_nodes'], cache['raw_scores'], gt, tol)
    for name, (tn, te) in policies.items():
        qids, pairs, scores = subset(cache, tn)
        decoded_q = cache[f'{name}__query_ids']
        np.testing.assert_array_equal(decoded_q, qids)  # bitwise-identical node decoding
        edges = cache[f'{name}__edges'].reshape(-1, 2)
        decoded = {tuple(sorted(map(int, e))) for e in edges}
        selected = np.array([tuple(sorted(map(int, p))) in decoded for p in pairs], dtype=bool)
        assert int(selected.sum()) == len(decoded) == len(edges)
        parity = int((selected != (scores > np.float32(te))).sum())
        nodes = cache['raw_nodes'][qids]
        np.testing.assert_allclose(nodes, cache[f'{name}__nodes'], rtol=0, atol=0)
        out = dict(threshold_parity_mismatches=parity, metrics={})
        for tol in [1., 2.]:
            mapping = match_nodes(nodes@m.T, gt['nodes']@m.T, tol)
            row, chosen, labels, _, _ = edge_accounting(np.arange(len(qids)), pairs, scores, selected,
                                                        mapping, gt['edges'], len(gt['nodes']))
            dist = np.linalg.norm((nodes[:, None]-gt['nodes'][None])@m.T, axis=-1)
            recovered = set(mapping.values())
            duplicates = [i for i in range(len(nodes)) if i not in mapping and any(dist[i, j] <= tol for j in recovered)]
            errors = np.array([dist[i, j] for i, j in mapping.items()])
            conditional = np.array([int(a) in mapping and int(b) in mapping for a, b in pairs], dtype=bool)
            g0, g1 = betti(len(gt['nodes']), gt['edges'])
            row.update(localization_sum_mm=float(errors.sum()), localization_count=len(errors),
                       duplicates=len(duplicates), node_count_bias=len(nodes)-len(gt['nodes']),
                       edge_count_bias=row['edges']-len(gt['edges']), beta0_bias=row['beta0']-g0,
                       beta1_bias=row['beta1']-g1)
            row.pop('degree_histogram', None)
            out['metrics'][str(tol)] = dict(row=row, all_retained=(scores, labels),
                                            conditional=(scores[conditional], labels[conditional]))
        prediction = dict(nodes=nodes, edges=edges, boxes=cache[f'{name}__boxes'],
                          node_scores=cache[f'{name}__node_scores'], edge_scores=cache[f'{name}__edge_scores'])
        out['box_states'] = graph_detection_states(prediction, gt['nodes'], gt['edges'], protocol=protocol)
        out['common'] = evaluate_common_vessels(dict(nodes=nodes, edges=edges,
                                                     spline_controls=cache[f'{name}__spline_controls']), gt_path)
        result['policies'][name] = out
    return result


def aggregate_rows(rows):
    from scripts.ixi_query_diagnostics import summarize
    result = summarize(rows)
    ratio = lambda a, b: a/b if b else None
    result.update(localization_mean_mm=ratio(sum(r['localization_sum_mm'] for r in rows), sum(r['localization_count'] for r in rows)),
        duplicates_per_patch=float(np.mean([r['duplicates'] for r in rows])),
        node_count_bias_per_patch=float(np.mean([r['node_count_bias'] for r in rows])),
        edge_count_bias_per_patch=float(np.mean([r['edge_count_bias'] for r in rows])),
        component_bias_per_patch=float(np.mean([r['beta0_bias'] for r in rows])),
        cycle_bias_per_patch=float(np.mean([r['beta1_bias'] for r in rows])))
    return result


def ranking_ap(parts):
    """D08 edge ranking AP over retained candidate pairs (sklearn AP, positives among candidates)."""
    from sklearn.metrics import average_precision_score
    scores = np.concatenate([p[0] for p in parts]); labels = np.concatenate([p[1] for p in parts]).astype(bool)
    return dict(ap=float(average_precision_score(labels, scores)) if labels.any() else None,
                candidates=len(scores), positives=int(labels.sum()),
                prevalence=float(labels.mean()) if len(labels) else None)


def node_ap(states):
    scores = np.concatenate([s[0] for s in states]); tp = np.concatenate([s[1] for s in states])
    gt = sum(s[2] for s in states)
    order = np.argsort(-scores, kind='stable'); tp = tp[order]
    cumulative = np.cumsum(tp); precision = cumulative/np.arange(1, len(tp)+1)
    return dict(node_ap=float(precision[tp].sum()/gt) if gt else None,
                max_recall=float(cumulative[-1]/gt) if gt and len(tp) else None,
                proposals=len(tp), gt_nodes=gt)


def population_summary(results, policies, protocol):
    from metrics.common_vessel_metrics import aggregate_common_vessels
    from training.evaluation.metrics import aggregate_detection_ap_ar
    summary = dict(patches=len(results), node_ap={t: node_ap([r['node_ap'][t] for r in results]) for t in ['1.0', '2.0']},
                   policies={})
    for name, (tn, te) in policies.items():
        entry = dict(node_threshold=tn, edge_threshold=te,
                     threshold_parity_mismatches=sum(r['policies'][name]['threshold_parity_mismatches'] for r in results))
        for tol in ['1.0', '2.0']:
            parts = [r['policies'][name]['metrics'][tol] for r in results]
            entry[tol] = dict(aggregate_rows([p['row'] for p in parts]),
                              edge_ap_all_retained=ranking_ap([p['all_retained'] for p in parts]),
                              edge_ap_conditional=ranking_ap([p['conditional'] for p in parts]))
        entry['common_v2'] = aggregate_common_vessels([r['policies'][name]['common'] for r in results])
        iou = protocol['iou_thresholds']
        for i, kind in enumerate(['node', 'edge']):
            ap, ar = aggregate_detection_ap_ar([r['policies'][name]['box_states'][i] for r in results], iou_thresholds=iou)
            entry[f'relationformer_box_{kind}_mAP'] = ap; entry[f'relationformer_box_{kind}_mAR'] = ar
        summary['policies'][name] = entry
    return summary


def save_caches(path, caches):
    path = Path(path); temp = path.with_name(path.name+'.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED) as archive:
        for sample, cache in caches:
            buffer = io.BytesIO(); np.savez_compressed(buffer, **cache)
            archive.writestr(sample+'.npz', buffer.getvalue())
    temp.replace(path)


def per_patch_rows(results):
    rows = []
    for r in results:
        for name, p in r['policies'].items():
            for tol, part in p['metrics'].items():
                rows.append(dict(sample=r['sample'], policy=name, tolerance_mm=float(tol), **part['row']))
            rows.append(dict(sample=r['sample'], policy=name, tolerance_mm=None, kind='common_v2', **p['common']))
    return rows


def snapshot(plan, cfg, model, criterion, epoch, folder, populations=None, device='cuda', workers=2, cpu_workers=4):
    """Full protocol: train-monitor selection, then fixed and selected decoding on both populations."""
    import torch
    folder = Path(folder)
    if (folder/'summary.json').exists():
        result = read(folder/'summary.json'); assert result['completed'] and result['epoch'] == epoch
        return result
    folder.mkdir(parents=True, exist_ok=True)
    populations = populations or plan['populations']
    started = time.monotonic()
    buffers = {n: v.detach().clone() for n, v in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
    protocol = cfg['evaluation']['protocol']
    gt_path = lambda split, s: str(Path(plan['dataset'])/split/'spline'/f'{s}.npz')
    result = dict(completed=True, epoch=epoch, populations={k: len(v) for k, v in populations.items()}, metrics={}, losses={})
    with mp.get_context('fork').Pool(cpu_workers) as pool:
        names = populations['train_monitor']
        caches, _ = infer(plan, model, criterion, 'train', names, {}, device, workers)
        thresholds = select_thresholds(pool, caches, [gt_path('train', s) for s in names])
        del caches
        policies = {FIXED: (.5, .5), SELECTED: (thresholds['node'], thresholds['edge'])}
        result['thresholds'] = thresholds
        print('selected thresholds', epoch, thresholds['node'], thresholds['edge'], flush=True)
        for population, split in POPULATIONS:
            names = populations[population]
            caches, loss = infer(plan, model, criterion, split, names, policies, device, workers)
            result['losses'][population] = loss
            save_caches(folder/f'{population}.zip', caches)
            items = [(s, c, gt_path(split, s), policies, protocol) for s, c in caches]
            results = []
            for i, r in enumerate(pool.imap(patch_task, items, chunksize=4), 1):
                results.append(r)
                if i % 128 == 0:
                    print('metrics', population, i, '/', len(items), flush=True)
            result['metrics'][population] = population_summary(results, policies, protocol)
            with gzip.open(folder/f'{population}-per-patch.json.gz', 'wt') as stream:
                json.dump(clean(per_patch_rows(results)), stream)
            del caches, items, results
    assert all(torch.equal(buffers[n], dict(model.named_buffers())[n]) for n in buffers), 'Evaluation altered BN'
    result.update(seconds=time.monotonic()-started, bn_buffers_unchanged=True, evaluation_batch_size=8,
                  test_data_accessed=False, validation_used_for_selection=False)
    write(folder/'summary.json', result)
    return result


# ---------------------------------------------------------- freeze & stage
def verify_code():
    """The running code must be the frozen staged copy recorded in the experiment record."""
    record = read(OUT/'frozen-source.json')
    assert CODE == Path(record['code_root']), (CODE, record['code_root'])
    for name, value in record['sha256'].items():
        assert sha(CODE/name) == value, f'Frozen source changed: {name}'
    return record


def stage(tag):
    """Copy the importable code packages to SCRATCH; jobs run from this immutable snapshot."""
    target = LARGE/'code'/tag
    assert not target.exists(), 'Never overwrite a staged code snapshot; use a new tag'
    names = []
    for package in ['boxes', 'configs', 'data', 'models', 'metrics', 'native2d', 'training', 'utils', 'scripts']:
        for path in sorted((REPO/package).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py', '.yaml', '.yml', '.json'}:
                names.append(str(path.relative_to(REPO)))
    for name in names:
        (target/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/name, target/name)
    hashes = {name: sha(target/name) for name in names}
    previous = OUT/'frozen-source.json'
    if previous.exists():
        old = read(previous); write(OUT/f'frozen-source-{old["tag"]}.json', old)
    write(OUT/'frozen-source.json', dict(tag=tag, code_root=str(target), sha256=hashes,
          script_sha256=hashes['scripts/ixi_long_experiment_20261030.py'], staged_at=time.strftime('%Y-%m-%dT%H:%M:%S')))
    with zipfile.ZipFile(OUT/f'frozen-source-{tag}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(target/name, name)
    print('staged', len(names), 'files to', target)


# ---------------------------------------------------------------- preflight
def protocol_checks():
    import torch
    source = {'query_embed.weight': torch.arange(122*4).reshape(122, 4).float(), 'classifier': torch.tensor([2., 3.])}
    target = {'query_embed.weight': torch.full((258, 4), -17.), 'classifier': torch.zeros(2)}
    result = expand_queries(source, target, 120)
    torch.testing.assert_close(result['query_embed.weight'][:120], source['query_embed.weight'][:120])
    torch.testing.assert_close(result['query_embed.weight'][256:258], source['query_embed.weight'][120:122])
    torch.testing.assert_close(result['query_embed.weight'][120:256], target['query_embed.weight'][120:256])
    sampler = EpochBatches(BATCH, 10000); sampler.epoch = 13
    items = [p[1] for b in sampler for p in b]
    assert len(sampler) == 312 and len(items) == len(set(items)) == 9984
    np.testing.assert_array_equal(items, epoch_order(13, 10000)[:9984])
    assert all(len(b) == BATCH for b in sampler)
    steps = 312
    assert abs(lr_scale(5*steps, steps)-1.) < 1e-12 and abs(lr_scale(1, steps)-(.01+.99/1560)) < 1e-12
    assert abs(7e-5*lr_scale(100*steps, steps)-1e-6) < 1e-15
    assert lr_scale(50*steps, steps) < lr_scale(25*steps, steps) < lr_scale(6*steps, steps)
    return dict(updates_per_epoch=312, total_updates=31200, warmup_updates=1560, dropped_per_epoch=16)


def cpu_preflight():
    """Login-node checks: hashes, target audit, protocol, exact source transfer (CPU, no data workers)."""
    import torch
    from data.augmentations import assert_valid_coordinates
    plan = read(OUT/'plan.json')
    assert sha(Path(plan['dataset'])/'manifest.json') == plan['manifest_sha256']
    assert sha(EVAL_PLAN) == plan['evaluation_plan_sha256'] and sha(D08/'plan.json') == plan['d08_plan_sha256']
    assert sha(plan['source']['path']) == plan['source']['sha256']
    checks = protocol_checks()
    audit = []
    evaluated = set(plan['populations']['val'])
    for split, names in [('train', plan['ids']['train']), ('val', [s for s in plan['ids']['val'] if s in evaluated])]:
        root = Path(plan['dataset'])/split
        for i, sample in enumerate(names):
            for directory, suffix in [('raw', '_data.nii.gz'), ('seg', '_seg.nii.gz'), ('vtp', '_graph.vtp')]:
                assert (root/directory/(sample+suffix)).is_file(), (sample, directory)
            with np.load(root/'spline'/f'{sample}.npz') as gt:
                nodes, edges = gt['nodes'], gt['edges']
                assert 0 < len(nodes) <= 256 and len(edges) > 0, (sample, len(nodes), len(edges))
                assert edges.min() >= 0 and edges.max() < len(nodes)
                links = {tuple(sorted(map(int, e))) for e in edges}
                assert len(links) == len(edges) and all(a != b for a, b in links), (sample, 'parallel/self edges')
                assert gt['controls'].shape == (len(edges), 8, 3) and gt['curves'].shape == (len(edges), 64, 3)
                for name in ['nodes', 'controls', 'curves', 'guides', 'metric', 'full_paths']:
                    assert np.isfinite(gt[name]).all(), (sample, name)
                assert abs(np.linalg.det(gt['metric'])) > 1e-6
                np.testing.assert_allclose(gt['controls'][:, [0, -1]], nodes[edges], atol=2e-6)
                assert_valid_coordinates(torch.from_numpy(nodes), (64, 64, 64), allow_patch_faces=True)
                audit.append(dict(split=split, sample=sample, nodes=len(nodes), edges=len(edges)))
            if i % 2000 == 0:
                print('target audit', split, i, len(names), flush=True)
    _, scratch, _, _, _ = setup(plan, 'cpu', initialize=False, cpu_ops=True)
    fresh = scratch.query_embed.weight.detach().clone(); del scratch
    _, model, _, _, migration = setup(plan, 'cpu', cpu_ops=True)
    torch.testing.assert_close(fresh[120:256], model.query_embed.weight[120:256], rtol=0, atol=0)
    source = torch.load(plan['source']['path'], map_location='cpu', mmap=True)['net']
    state = model.state_dict()
    for name, value in source.items():
        if name == 'query_embed.weight':
            torch.testing.assert_close(state[name][:120], value[:120], rtol=0, atol=0)
            torch.testing.assert_close(state[name][256:258], value[120:122], rtol=0, atol=0)
        else:
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)
    d08_init = read(D08/'large/runs/synthetic/initialization.json')
    extra = extra_query_sha(model)
    write(OUT/'target-audit.json', audit)
    write(OUT/'cpu-preflight.json', dict(passed=True, protocol=checks, audited_train=len(plan['ids']['train']),
        audited_val=len(evaluated), max_train_nodes=max(r['nodes'] for r in audit if r['split'] == 'train'),
        max_val_nodes=max(r['nodes'] for r in audit if r['split'] == 'val'), query_expansion=migration,
        every_other_source_tensor_exact=True, extra_rows_match_scratch=True,
        extra_query_sha256=extra, identical_to_d08_synthetic_extra_rows=extra == d08_init['extra_query_sha256'],
        source_sha256=plan['source']['sha256'], test_data_accessed=False,
        target_audit_sha256=sha(OUT/'target-audit.json')))
    assert extra == d08_init['extra_query_sha256'], 'Initialization differs from D08 synthetic arm'
    print('CPU preflight passed', flush=True)


def cpu_forward():
    """One batch-1 CPU forward + loss + inference + metrics on one train-monitor and one val patch."""
    import torch
    plan = read(OUT/'plan.json')
    torch.set_num_threads(2)
    cfg, model, criterion, _, _ = setup(plan, 'cpu', cpu_ops=True)
    populations = dict(train_monitor=plan['populations']['train_monitor'][:1], val=plan['populations']['val'][:1])
    folder = LARGE/'checks'/f'cpu-forward-{int(time.time())}'
    result = snapshot(plan, cfg, model, criterion, -1, folder, populations, device='cpu', workers=0, cpu_workers=1)
    write(OUT/'cpu-forward.json', dict(passed=True, folder=str(folder), thresholds=[result['thresholds']['node'], result['thresholds']['edge']],
        losses=result['losses'], val_fixed=result['metrics']['val']['policies'][FIXED]['2.0'],
        val_common=result['metrics']['val']['policies'][FIXED]['common_v2']))
    print('CPU forward passed', json.dumps(result['losses']), flush=True)


# ------------------------------------------------------------------ training
def train(end_epoch, run=None, ids=None, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, smoke=False):
    import torch
    from torch.utils.data import DataLoader
    from data.loaders.common import image_graph_collate
    plan = read(OUT/'plan.json')
    run = Path(run or LARGE/'runs'/'synthetic'); run.mkdir(parents=True, exist_ok=True)
    ids = ids or plan['ids']['train']
    ids_digest = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    latest = run/'latest.pt'; initialize = not latest.exists()
    cfg, model, criterion, opt, migration = setup(plan, initialize=initialize)
    sampler = EpochBatches(BATCH, len(ids)); steps_per_epoch = len(sampler)
    epoch, step, history = 0, 0, []
    if not initialize:
        cp = torch.load(latest, map_location='cpu', mmap=True)
        assert cp['plan_sha256'] == sha(OUT/'plan.json') and cp['batch_size'] == BATCH and cp['ids_sha256'] == ids_digest
        model.load_state_dict(cp['net'], strict=True); opt.load_state_dict(cp['optimizer'])
        epoch, step = cp['epoch'], cp['step']
        assert all(int(v['step']) == step for v in opt.state.values() if 'step' in v)
        del cp
        history = [h for h in read(run/'history.json') if h['epoch'] <= epoch]
        print('resumed', run, 'epoch', epoch, 'step', step, flush=True)
    assert step == epoch*steps_per_epoch and epoch < end_epoch <= epochs, (epoch, step, end_epoch)
    if initialize:
        write(run/'config.json', cfg)
        write(run/'initialization.json', dict(arm='synthetic', source=plan['source'], migration=migration,
              seed=SEED, extra_query_sha256=extra_query_sha(model)))
    if not smoke:
        for e in EVAL_EPOCHS:  # (re)run a missing evaluation of the state we start from
            if e == epoch:
                snapshot(plan, cfg, model, criterion, e, LARGE/'predictions'/f'epoch-{e:03d}'); collect()
    dataset = SeededDataset(base_dataset(plan, 'train', ids, True))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, collate_fn=image_graph_collate,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)
    stop = [False]
    for sig in [signal.SIGUSR1, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    start = time.monotonic(); torch.cuda.reset_peak_memory_stats(); records = []
    for epoch in range(epoch+1, end_epoch+1):
        sampler.epoch = epoch; chain = bytes(32); losses = []; epoch_start = time.monotonic()
        for offset, batch in enumerate(loader):
            chain = hashlib.sha256(chain+batch_digest(batch)).digest()
            step += 1
            row = update(model, criterion, opt, batch, step, steps_per_epoch, epochs, warmup_epochs,
                         plan['optimizer']['lr'], diagnostic=step == 1)
            losses.append(row)
            if step % 50 == 0 or offset == 0:
                print('epoch', epoch, 'batch', offset+1, '/', steps_per_epoch, 'step', step,
                      'loss', round(row['total'], 5), 'lr', row['lr'], flush=True)
            if stop[0]:
                raise RuntimeError(f'Signal received in epoch {epoch}; resume from latest.pt (epoch {epoch-1})')
        record = dict(epoch=epoch, step=step, patches=steps_per_epoch*BATCH, updates=steps_per_epoch,
            patch_order_sha256=hashlib.sha256(epoch_order(epoch, len(ids)).astype('<i8').tobytes()).hexdigest(),
            full_augmented_batch_chain_sha256=chain.hex(), epoch_seconds=time.monotonic()-epoch_start,
            losses=summarize_updates(losses), lr_end=losses[-1]['lr'])
        history.append(record); records.append(record); write(run/'history.json', history)
        if epoch in MILESTONES and not smoke:
            path = run/f'model-epoch-{epoch:03d}.pt'
            if path.exists():  # replay after a crash between milestone and latest: keep both
                path.rename(path.with_name(path.name+f'.superseded-{int(time.time())}'))
            save_checkpoint(path, cfg, model, None, epoch, step, batch_size=BATCH)
            write(run/f'model-epoch-{epoch:03d}.json', dict(path=str(path), sha256=sha(path), epoch=epoch, step=step))
        save_checkpoint(latest, cfg, model, opt, epoch, step, batch_size=BATCH, ids_sha256=ids_digest)
        if epoch in EVAL_EPOCHS and not smoke:
            snapshot(plan, cfg, model, criterion, epoch, LARGE/'predictions'/f'epoch-{epoch:03d}'); collect()
    elapsed = time.monotonic()-start
    write(run/f'chunk-{end_epoch:03d}.json', dict(completed=True, end_epoch=end_epoch, step=step,
        job_id=os.environ.get('SLURM_JOB_ID'), seconds=elapsed, gpu_hours=elapsed/3600,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        epoch_records=records))
    if end_epoch == epochs and not smoke:
        write(run/'complete.json', dict(completed=True, epochs=epochs, updates=step, source=plan['source']))
        collect()
    print('Completed', run, end_epoch, round(elapsed), 's', flush=True)
    return history


def gpu_smoke():
    import torch
    from torch.utils.data import DataLoader
    from data.loaders.common import image_graph_collate
    assert read(OUT/'cpu-preflight.json')['passed']
    frozen = verify_code()
    plan = read(OUT/'plan.json'); started = time.monotonic()
    root = LARGE/'smoke'/os.environ.get('SLURM_JOB_ID', 'local')
    assert not root.exists(); root.mkdir(parents=True)
    # 1) Benchmark: production schedule, full train set, 4 workers, batch 32.
    cfg, model, criterion, opt, _ = setup(plan)
    ids = plan['ids']['train']; sampler = EpochBatches(BATCH, len(ids))
    loader = DataLoader(SeededDataset(base_dataset(plan, 'train', ids, True)), batch_sampler=sampler, num_workers=4,
                        collate_fn=image_graph_collate, pin_memory=True, persistent_workers=False, prefetch_factor=2)
    torch.cuda.reset_peak_memory_stats(); timings = []; rows = []
    iterator = iter(loader); updates = 200
    for number in range(1, updates+1):
        torch.cuda.synchronize(); tic = time.perf_counter()
        batch = next(iterator)
        row = update(model, criterion, opt, batch, number, len(sampler), EPOCHS, WARMUP_EPOCHS,
                     plan['optimizer']['lr'], diagnostic=number == 1)
        torch.cuda.synchronize()
        if number > 10: timings.append(time.perf_counter()-tic)
        rows.append(row)
        if number % 50 == 0: print('benchmark', number, round(row['total'], 5), flush=True)
    seconds = float(np.mean(timings))
    memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    assert memory['peak_reserved_bytes'] < .85*memory['total_memory_bytes'], 'Batch 32 lacks memory headroom'
    del iterator, loader, model, criterion, opt, batch; torch.cuda.empty_cache()
    # 2) Resume check through the production train() path on a tiny epoch definition.
    tiny = ids[:320]; run = root/'resume'
    first = train(2, run=run, ids=tiny, epochs=4, warmup_epochs=1, smoke=True)
    cp = torch.load(run/'latest.pt', map_location='cpu', mmap=True)
    assert cp['epoch'] == 2 and cp['step'] == 20; saved = {k: v.clone() for k, v in cp['net'].items()}; del cp
    _, probe, _, _, _ = setup(plan, initialize=False)
    probe.load_state_dict(saved, strict=True)
    assert all(torch.equal(saved[k], v.cpu()) for k, v in probe.state_dict().items()); del probe, saved
    second = train(4, run=run, ids=tiny, epochs=4, warmup_epochs=1, smoke=True)
    assert [h['epoch'] for h in second] == [1, 2, 3, 4] and second[-1]['step'] == 40
    assert second[:2] == first[:2]
    assert abs(second[2]['lr_end']-7e-5*lr_scale(30, 10, 4, 1)) < 1e-15
    # 3) Full evaluation protocol on 16 train-monitor + 16 val patches.
    cfg, model, criterion, _, _ = setup(plan)
    tic = time.monotonic()
    populations = dict(train_monitor=plan['populations']['train_monitor'][:16], val=plan['populations']['val'][:16])
    evaluation = snapshot(plan, cfg, model, criterion, -1, root/'evaluation', populations)
    eval_seconds = time.monotonic()-tic
    # Estimate: training at measured speed (+5% epoch/checkpoint overhead), four full snapshots.
    training_hours = plan['total_updates']*seconds*1.05/3600
    eval_hours = len(EVAL_EPOCHS)*eval_seconds*(768+256)/(32+16)/3600  # monitor inferred twice
    d08_rate = 8.019722222222223/82200  # D08 synthetic arm allocated GPU-h per update incl. evaluation
    estimate = training_hours+eval_hours+.15
    walltime = max(60, math.ceil(1.3*estimate*60/10)*10)
    result = dict(passed=True, job_id=os.environ.get('SLURM_JOB_ID'), code=frozen['code_root'],
        batch_size=BATCH, workers=4, benchmark_updates=updates, seconds_per_update=seconds,
        patches_per_second=BATCH/seconds, **memory, first_update=rows[0], last_update=rows[-1],
        resume=dict(passed=True, epochs=[h['epoch'] for h in second], steps=second[-1]['step'],
                    first_chunk_history_preserved=True, optimizer_step_restored=True),
        evaluation=dict(passed=True, seconds=eval_seconds, thresholds=[evaluation['thresholds']['node'], evaluation['thresholds']['edge']],
                        val_fixed_2mm={k: evaluation['metrics']['val']['policies'][FIXED]['2.0'][k] for k in ['node_f1', 'f1']},
                        val_common_fixed={k: evaluation['metrics']['val']['policies'][FIXED]['common_v2'][k] for k in ['common_branch_f1', 'common_curve_f1']},
                        node_ap_2mm=evaluation['metrics']['val']['node_ap']['2.0']['node_ap'],
                        parity_mismatches={p: evaluation['metrics'][g]['policies'][p]['threshold_parity_mismatches'] for g in ['train_monitor', 'val'] for p in [FIXED, SELECTED]}),
        estimate=dict(training_gpu_hours=training_hours, evaluation_gpu_hours=eval_hours, total_gpu_hours=estimate,
                      d08_rate_projection_gpu_hours=d08_rate*plan['total_updates'], walltime_minutes=walltime,
                      single_job=estimate <= 18),
        smoke_seconds=time.monotonic()-started)
    write(OUT/'gpu-smoke.json', result)
    print(json.dumps(result, indent=2), flush=True)


def evaluate_epoch(epoch):
    import torch
    verify_code(); plan = read(OUT/'plan.json')
    cfg, model, criterion, _, _ = setup(plan, initialize=epoch == 0)
    if epoch:
        cp = torch.load(LARGE/'runs'/'synthetic'/f'model-epoch-{epoch:03d}.pt', map_location='cpu', mmap=True)
        model.load_state_dict(cp['net'], strict=True); assert cp['epoch'] == epoch; del cp
    snapshot(plan, cfg, model, criterion, epoch, LARGE/'predictions'/f'epoch-{epoch:03d}')
    collect()


# ------------------------------------------------------------------- collect
def compact(summary):
    out = dict(epoch=summary['epoch'], thresholds=dict(node=summary['thresholds']['node'], edge=summary['thresholds']['edge']),
               losses=summary['losses'], seconds=summary['seconds'], populations={})
    for population, metrics in summary['metrics'].items():
        entry = dict(node_ap_2mm=metrics['node_ap']['2.0'], node_ap_1mm=metrics['node_ap']['1.0'])
        for name, p in metrics['policies'].items():
            m2 = p['2.0']
            entry[name] = dict(node_threshold=p['node_threshold'], edge_threshold=p['edge_threshold'],
                node_precision=m2['node_precision'], node_recall=m2['node_recall'], node_f1=m2['node_f1'],
                edge_precision=m2['precision'], edge_recall=m2['recall'], edge_f1=m2['f1'],
                conditional_recall=m2['conditional_recall'], beta0_mae=m2['beta0_mae'], beta1_mae=m2['beta1_mae'],
                localization_mean_mm=m2['localization_mean_mm'], duplicates_per_patch=m2['duplicates_per_patch'],
                node_count_bias_per_patch=m2['node_count_bias_per_patch'],
                edge_ap_all_retained=m2['edge_ap_all_retained']['ap'], edge_ap_conditional=m2['edge_ap_conditional']['ap'],
                node_f1_1mm=p['1.0']['node_f1'], edge_f1_1mm=p['1.0']['f1'],
                common_branch_f1=p['common_v2']['common_branch_f1'], common_curve_f1=p['common_v2']['common_curve_f1'],
                common_curve_precision=p['common_v2']['common_curve_precision'], common_curve_recall=p['common_v2']['common_curve_recall'],
                relationformer_box_node_mAP=p['relationformer_box_node_mAP'], relationformer_box_edge_mAP=p['relationformer_box_edge_mAP'],
                threshold_parity_mismatches=p['threshold_parity_mismatches'])
        out['populations'][population] = entry
    return out


def collect():
    results = []
    for path in sorted((LARGE/'predictions').glob('epoch-*/summary.json')):
        summary = read(path)
        if summary['epoch'] < 0: continue
        item = compact(summary); results.append(item)
        write(OUT/'analysis'/f'epoch-{summary["epoch"]:03d}.json', dict(item, full=summary['metrics'],
              threshold_curves=dict(node=summary['thresholds']['node_curve'], edge=summary['thresholds']['edge_curve'])))
    run = LARGE/'runs'/'synthetic'
    history = read(run/'history.json') if (run/'history.json').exists() else []
    chunks = [read(p) for p in sorted(run.glob('chunk-*.json'))]
    status = dict(epochs_completed=history[-1]['epoch'] if history else 0, updates=history[-1]['step'] if history else 0,
                  evaluated_epochs=[r['epoch'] for r in results], training_gpu_hours=sum(c['gpu_hours'] for c in chunks),
                  complete=(run/'complete.json').exists(),
                  train_loss_by_epoch={h['epoch']: h['losses'].get('total') for h in history})
    write(OUT/'status.json', status)
    lines = ['# IXI long RelationFormer B0 control (D08 recipe, synthetic init) — 30 Oct 2026', '',
             f'Epochs completed: {status["epochs_completed"]}/{EPOCHS}; updates {status["updates"]}/31200. Evaluated epochs: {status["evaluated_epochs"]}.', '',
             'Val = multidataset_continuous IXI val_ids (512); train monitor = monitor_train (256). Selected thresholds come from the train monitor only.', '',
             '| Epoch | Population | Policy (node/edge thr) | Node P / R / F1 @2mm | Edge P / R / F1 @2mm | Branch F1 | Curve F1 | Node AP | Edge AP (all) |',
             '|---:|---|---|---|---|---:|---:|---:|---:|']
    f = lambda x: 'NA' if x is None else f'{x:.4f}'
    for r in results:
        for population in ['train_monitor', 'val']:
            e = r['populations'][population]
            for name in [FIXED, SELECTED]:
                p = e[name]
                lines.append(f'|{r["epoch"]}|{population}|{name} ({p["node_threshold"]}/{p["edge_threshold"]})|'
                             f'{f(p["node_precision"])} / {f(p["node_recall"])} / {f(p["node_f1"])}|'
                             f'{f(p["edge_precision"])} / {f(p["edge_recall"])} / {f(p["edge_f1"])}|'
                             f'{f(p["common_branch_f1"])}|{f(p["common_curve_f1"])}|{f(e["node_ap_2mm"]["node_ap"])}|{f(p["edge_ap_all_retained"])}|')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    final = [r for r in results if r['epoch'] == EPOCHS]
    if final and status['complete']:
        write(OUT/'final.json', dict(final[0], status=status, trajectory=results,
              protocol=dict(train_patches=10000, epochs=EPOCHS, batch=BATCH, updates=31200,
                            validation_used_for_selection=False)))
    if (OUT/'submission.json').exists():
        jobs = [j for j in read(OUT/'submission.json').get('jobs', {}).values() if j]
        if jobs:
            try:
                text = subprocess.check_output(['sacct', '-j', ','.join(jobs), '-X', '--format=JobID,JobName%24,State,ExitCode,Start,End,Elapsed', '--parsable2'], text=True)
                (OUT/'slurm-accounting.psv').write_text(text)
            except (OSError, subprocess.CalledProcessError):
                pass
    return results


# ------------------------------------------------------------------- submit
def my_gpu_jobs():
    text = subprocess.check_output(['squeue', '-h', '-u', os.environ.get('USER', 'upz25mj'), '-o', '%i|%j|%b'], text=True)
    return [line.split('|')[0] for line in text.splitlines() if line.split('|')[1].startswith('ixi-long') and 'gpu' in line.split('|')[2]]


def submit(mode, *args, options=()):
    record = read(OUT/'submission.json') if (OUT/'submission.json').exists() else dict(jobs={})
    frozen = read(OUT/'frozen-source.json')
    active = my_gpu_jobs(); assert len(active) < 2, f'Already two of this study\'s GPU jobs queued: {active}'
    command = ['sbatch', '--parsable', f'--export=ALL,IXI_LONG_CODE={frozen["code_root"]}', *options, SLURM, mode, *map(str, args)]
    job = subprocess.check_output(command, text=True, cwd=REPO).strip().split(';')[0]
    assert job.isdigit(), job
    record['jobs'][f'{mode}{"-"+"-".join(map(str, args)) if args else ""}-{job}'] = job
    record.setdefault('history', []).append(dict(job=job, mode=mode, args=list(map(str, args)), options=list(options),
        code=frozen['code_root'], code_tag=frozen['tag'], submitted=time.strftime('%Y-%m-%dT%H:%M:%S')))
    write(OUT/'submission.json', record)
    print(job)
    return job


def submit_train():
    smoke = read(OUT/'gpu-smoke.json'); assert smoke['passed']
    accounting = subprocess.check_output(['sacct', '-j', smoke['job_id'], '-X', '-n', '-P', '-o', 'State,ExitCode'], text=True).strip()
    assert accounting.startswith('COMPLETED|0:0'), accounting
    assert smoke['code'] == read(OUT/'frozen-source.json')['code_root'], 'Code restaged after the GPU gate'
    for name, value in read(OUT/'frozen-source.json')['sha256'].items():
        assert sha(Path(smoke['code'])/name) == value, name
    assert smoke['estimate']['single_job'], 'Estimate exceeds one job: chunking required'
    minutes = smoke['estimate']['walltime_minutes']; assert minutes <= 20*60
    wall = f'{minutes//60:02d}:{minutes%60:02d}:00'
    return submit('train', EPOCHS, options=[f'--time={wall}', '--job-name=ixi-long-train'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'preflight', 'cpu-forward', 'stage', 'smoke', 'train',
                                         'evaluate', 'collect', 'submit-smoke', 'submit-train'])
    parser.add_argument('--end-epoch', type=int, default=EPOCHS)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--tag')
    args = parser.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'preflight': cpu_preflight()
    elif args.mode == 'cpu-forward': cpu_forward()
    elif args.mode == 'stage': stage(args.tag)
    elif args.mode == 'smoke': gpu_smoke()
    elif args.mode == 'train':
        verify_code(); assert read(OUT/'gpu-smoke.json')['passed']; train(args.end_epoch)
    elif args.mode == 'evaluate': evaluate_epoch(args.epoch)
    elif args.mode == 'collect': collect()
    elif args.mode == 'submit-smoke':
        submit('smoke', options=['--time=00:15:00', '--qos=qos_gpu_h100-dev', '--job-name=ixi-long-smoke'])
    else: submit_train()
