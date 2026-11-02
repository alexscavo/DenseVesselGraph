#!/usr/bin/env python3
"""Heatmap-anchored RelationFormer arm of the 30 Oct 2026 three-way comparison.

RelationFormer's transformer, relation head, box/class heads and K8 graph_only
spline head are kept; the 256 object queries are anchored at the top-256 peaks of
the frozen WS2 node-heatmap probe (applied to the same augmented input), and the
Hungarian assignment is replaced by a physical 2 mm one-to-one assignment of
*anchors* to GT nodes (models.anchor_matcher). Recipe follows the completed D08
BRAVA run (AdamW 7e-5 / encoder 3.5e-5, 5-epoch warmup + cosine to 1e-6,
quarter-turn + noise augmentation, float32, SyntheticMRI B0 initialization) for
about 15,000 updates per dataset. Evaluation mirrors scripts/dense_graph_20261030
(match_nodes / edge_accounting / common-v2; thresholds selected on monitor_train
only), with RelationFormer's own inference path (training.evaluation.inference).

Modes: prepare | smoke | train --dataset {IXI,BRAVA} | evaluate --dataset ...
"""
import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

OUT = ROOT / 'experiments/anchored_rf_20261030'
LARGE = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/anchored_rf_20261030')
D08_PLAN = Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/'
                'experiments/brava_long_20261027/plan.json')
MULTI_PLAN = ROOT / 'experiments/multidataset_continuous/plan.json'
PROBE_ROOT = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/node_heatmap_probe_20261030')
SEED = 271039  # D08 seed: same initialization RNG and epoch order generator
DATASETS = ('IXI', 'BRAVA')
TARGET_UPDATES = 15000
OBJECT_QUERIES, RELATION_TOKENS = 256, 2
NODE_THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.71, 0.05)]  # as dense_graph_20261030
EDGE_THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.91, 0.05)]
NODE_FLOOR, EDGE_FLOOR = min(NODE_THRESHOLDS), min(EDGE_THRESHOLDS)
TOLERANCE_MM = 2.0
EVAL_BATCH = 8


# ----------------------------------------------------------------------------- utilities
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
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(clean(value), indent=1, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def seed(number):
    import torch
    random.seed(number)
    np.random.seed(number % 2 ** 32)
    torch.manual_seed(number)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(number)


def epoch_order(epoch, count):
    return np.random.default_rng(SEED + int(epoch)).permutation(count)


def sample_seed(epoch, sample):
    payload = f'{SEED}:{epoch}:{sample}'.encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), 'little')


def lr_scale(step, steps_per_epoch, epochs, warmup_epochs=5):
    """D08 schedule: linear 1%->100% warmup over 5 epochs, cosine to 1e-6/7e-5 at the end."""
    warmup = warmup_epochs * steps_per_epoch
    total = epochs * steps_per_epoch
    if step <= warmup:
        return .01 + .99 * step / warmup
    progress = (step - warmup) / (total - warmup)
    floor = 1. / 70.
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min(1., progress)))


def padding_seed(dataset, sample_id):
    from models.anchored_relationformer import proposal_seed
    return proposal_seed(f'anchored-rf-20261030:{dataset}:{sample_id}')


# ----------------------------------------------------------------------------- plan
def prepare():
    assert not (OUT / 'plan.json').exists(), 'Do not overwrite an existing protocol'
    d08 = read(D08_PLAN)
    multi = read(MULTI_PLAN)
    cfg = copy.deepcopy(d08['config'])
    assert cfg['model']['decoder']['object_queries'] == OBJECT_QUERIES
    assert cfg['model']['decoder']['relation_tokens'] == RELATION_TOKENS
    assert cfg['loss']['node']['classification']['name'] == 'focal'
    assert cfg['loss']['node']['classification']['class_weights'] == [1., 1.]
    assert cfg['loss']['edge']['classification']['name'] == 'cross_entropy'
    assert cfg['loss']['edge']['balancing']['positive_to_negative_ratio'] == .15
    assert cfg['model']['spline']['controls'] == 8 and cfg['model']['spline']['local_geometry']['enabled']
    cfg['model']['matcher'] = dict(type='anchor_physical', tolerance_mm=TOLERANCE_MM,
                                   note='Replaces Hungarian: one-to-one Hungarian on anchor positions vs GT in mm, '
                                        'pairs <= 2 mm accepted; other queries no-object')
    cfg['model']['anchored'] = dict(enabled=True, residual_voxels=3.0, proposer='WS2 BasicUNet node heatmap probe',
                                    peaks='3^3 max-pool local maxima of sigmoid(ch0), top-256 by score',
                                    padding='score 0, uniform in-patch positions, RNG seeded per sample')
    cfg['experiment'].update(seed=SEED, name='anchored_rf_20261030')
    datasets = {}
    for name in DATASETS:
        entry = multi['datasets'][name]
        probe = PROBE_ROOT / name / 'model.pt'
        datasets[name] = dict(root=entry['root'], manifest_sha256=entry['manifest_sha256'],
                              train_ids=entry['train_ids'], val_ids=entry['val_ids'],
                              monitor_train=entry['monitor_train'],
                              probe_checkpoint=str(probe), probe_sha256=sha(probe),
                              probe_final=read(PROBE_ROOT / name / 'final.json')['result'])
        assert set(entry['monitor_train']) <= set(entry['train_ids'])
        assert not set(entry['train_ids']) & set(entry['val_ids'])
    plan = dict(
        experiment='Heatmap-anchored RelationFormer (WS arm of 3-way 3D method comparison)', date='2026-10-30',
        config=cfg, datasets=datasets, seed=SEED, source=d08['source'],
        source_policy='SyntheticMRI B0 (D08 synthetic arm source). All tensors loaded strictly by name/shape, '
                      'except query_embed: only its 2 relation rows [120:122] are kept (object rows are replaced '
                      'by anchors). New modules (anchor_pos, content_embed, score_proj, offset_embed) are '
                      'freshly initialized; offset_embed last layer zero so nodes start at anchors.',
        optimizer=dict(name='AdamW', lr=7e-5, backbone_lr=3.5e-5, weight_decay=1e-4, betas=[.9, .999], eps=1e-8,
                       warmup_epochs=5, floor_lr=1e-6,
                       schedule='D08: linear 1%-100% warmup over 5 epochs, then cosine over all epochs'),
        target_updates=TARGET_UPDATES,
        budget_rule='Whole epochs: epochs = round(15000 / (train patches // batch)); exact count recorded',
        batch_candidates=[16, 32],
        batch_selection='D08 rule: 32 if peak reserved memory < 85% and throughput >= 95% of batch 16',
        augmentation=d08['augmentation'], gradient_route='graph_only', precision='float32 detector; '
        'proposer bf16 autocast exactly as the probe evaluation (frozen, eval mode, no_grad)',
        evaluation=dict(
            tolerance_mm=TOLERANCE_MM, node_thresholds=NODE_THRESHOLDS, edge_thresholds=EDGE_THRESHOLDS,
            inference='training.evaluation.inference.infer_graphs: node prob > t, all retained undirected pairs, '
                      'orientation logits averaged before softmax, edge prob > t, K8 spline controls',
            metrics='scripts.diagnose_branch_relations.match_nodes + scripts.ixi_query_diagnostics.edge_accounting '
                    '(2 mm) and metrics.common_vessel_metrics.evaluate_common_vessels (common-v2)',
            thresholds='(a) fixed 0.5/0.5; (b) node then edge thresholds selected on monitor_train only '
                       '(dense_graph_20261030.select), applied unchanged to val_ids',
            anchor_upper_bound='fraction of GT nodes with an anchor within 2 mm (one-to-one match_nodes and any)'),
        large_root=str(LARGE), max_queued_jobs=2, test_data_accessed=False)
    write(OUT / 'plan.json', plan)
    print('wrote', OUT / 'plan.json')


def configuration(plan, dataset, batch_size):
    cfg = copy.deepcopy(plan['config'])
    cfg['data']['batch_size'] = batch_size
    cfg['data']['datasets']['synthetic_mri']['root'] = plan['datasets'][dataset]['root']
    cfg['experiment']['name'] = f'anchored_rf_{dataset}'
    return cfg


# ----------------------------------------------------------------------------- data
def records(plan, dataset, ids, split):
    from data.loaders.discovery import discover_synthetic_mri
    root = Path(plan['datasets'][dataset]['root'])
    by_id = {r.sample_id: r for r in discover_synthetic_mri(root, split)}
    missing = [i for i in ids if i not in by_id]
    assert not missing, (dataset, split, missing[:5])
    return [by_id[i] for i in ids]


def base_dataset(plan, dataset, ids, split, augment):
    from data.loaders.synthetic_mri import SyntheticMRIDataset
    mean = plan['config']['data']['datasets']['synthetic_mri']['foreground_mean']
    return SyntheticMRIDataset(records(plan, dataset, ids, split), augment=augment, zoom_range=None,
                               spline_controls=8, foreground_mean=mean, gaussian_noise_probability=.35,
                               gaussian_noise_max_std=.015, input_modality='raw_mra')


class SeededDataset:
    """D08: unchanged loader with deterministic augmentation per (epoch, sample)."""

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
        return self.base[index], self.base.records[index].sample_id


class IndexedDataset:
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return self.base[index], self.base.records[index].sample_id


def collate(items):
    from data.loaders.common import image_graph_collate
    return image_graph_collate([item[0] for item in items]), [item[1] for item in items]


class EpochBatches:
    def __init__(self, batch_size, count):
        self.batch_size, self.count, self.epoch, self.start = batch_size, count - count % batch_size, 1, 0

    def __len__(self):
        return self.count // self.batch_size

    def __iter__(self):
        order = epoch_order(self.epoch, self.count)
        for start in range(self.start * self.batch_size, self.count, self.batch_size):
            yield [(self.epoch, int(i)) for i in order[start:start + self.batch_size]]


class FixedBatches:
    """Memorization smoke: the same patches every update, no augmentation."""

    def __init__(self, count, updates):
        self.count, self.updates = count, updates

    def __len__(self):
        return self.updates

    def __iter__(self):
        for _ in range(self.updates):
            yield list(range(self.count))


# ----------------------------------------------------------------------------- model
def load_source(model, plan):
    import torch
    checkpoint = torch.load(plan['source']['path'], map_location='cpu', mmap=True, weights_only=False)
    source_cfg = checkpoint['config']
    cfg = plan['config']
    for name in ['encoder', 'spline']:
        assert source_cfg['model'][name] == cfg['model'][name], name
    for key, value in source_cfg['model']['decoder'].items():
        if key not in ('object_queries', 'use_cuda_extension'):  # runtime kernel choice only
            assert value == cfg['model']['decoder'][key], key
    assert not source_cfg['model']['node_refinement'].get('enabled', False)
    source_queries = int(source_cfg['model']['decoder']['object_queries'])
    target = model.state_dict()
    state, skipped = {}, []
    for name, value in checkpoint['net'].items():
        if name == 'query_embed.weight':
            assert value.shape[0] == source_queries + RELATION_TOKENS
            state[name] = value[source_queries:source_queries + RELATION_TOKENS].clone()
            continue
        assert name in target and target[name].shape == value.shape, name
        state[name] = value
    report = model.load_state_dict(state, strict=False)
    assert not report.unexpected_keys, report.unexpected_keys
    new = sorted({k.split('.')[0] for k in report.missing_keys})
    assert set(new) == {'anchor_pos', 'content_embed', 'score_proj', 'offset_embed', 'proposer'}, new
    del checkpoint
    return dict(source=plan['source']['path'], source_queries=source_queries, relation_rows_loaded=[
        source_queries, source_queries + RELATION_TOKENS], skipped=skipped, fresh_modules=new,
        loaded_tensors=len(state), missing_tensors=len(report.missing_keys))


def setup(plan, dataset, batch_size, device='cuda', initialize=True):
    import torch
    from configs import validate_config
    from models.anchor_matcher import AnchorMatcher
    from models.anchored_relationformer import build_anchored_relationformer
    from scripts.train_ixi_transfer import TransferCriterion
    cfg = configuration(plan, dataset, batch_size)
    # The schema only knows hungarian/fgw; validate everything else with D08's (unused) matcher entry.
    check = copy.deepcopy(cfg)
    check['model']['matcher'] = dict(type='hungarian', class_cost=2.0, node_cost=5.0)
    validate_config(check)
    seed(SEED)
    model = build_anchored_relationformer(cfg, plan['datasets'][dataset]['probe_checkpoint'])
    migration = load_source(model, plan) if initialize else None
    assert not any(p.requires_grad for p in model.proposer.parameters())
    model.to(device)
    criterion = TransferCriterion(cfg, AnchorMatcher(TOLERANCE_MM), model.relation_embed,
                                  spline_head=model.spline_head, route='graph_only').to(device)
    criterion.set_training_progress(1, 0.)
    settings = plan['optimizer']
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert not any(n.startswith('proposer.') for n, _ in named)
    groups = [dict(params=[p for n, p in named if n.startswith('encoder.')],
                   lr=settings['backbone_lr'], base_lr=settings['backbone_lr']),
              dict(params=[p for n, p in named if not n.startswith('encoder.')],
                   lr=settings['lr'], base_lr=settings['lr'])]
    optimizer = torch.optim.AdamW(groups, betas=tuple(settings['betas']), eps=settings['eps'],
                                  weight_decay=settings['weight_decay'])
    return cfg, model, criterion, optimizer, migration


def update(model, criterion, opt, batch, step, scale, dataset, diagnostic=False):
    import torch
    from training.engine import _move_graph_batch
    graph_batch, ids = batch
    seed(SEED + step)
    volumes, targets = _move_graph_batch(graph_batch, 'cuda', 'image', True)
    for group in opt.param_groups:
        group['lr'] = group['base_lr'] * scale
    model.train()
    opt.zero_grad(set_to_none=True)
    tokens, predictions, _ = model(volumes, [padding_seed(dataset, i) for i in ids])
    losses = criterion(tokens, predictions, targets)
    assert torch.isfinite(losses['total']), f'Nonfinite loss at {step}'
    result = {}
    if diagnostic:
        assert not model.proposer.training
        shared = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('spline_head.')]
        grads = torch.autograd.grad(losses['spline'], shared, retain_graph=True, allow_unused=True)
        assert all(g is None or not bool(g.abs().sum()) for g in grads), 'Spline gradient entered detector'
        result['geometry_to_detector_zero'] = True
    losses['total'].backward()
    if diagnostic:
        for prefix in ['encoder.', 'decoder.', 'class_embed.', 'coord_embed.', 'relation_embed.', 'spline_head.',
                       'anchor_pos.', 'content_embed', 'score_proj.', 'offset_embed.', 'query_embed.']:
            values = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            assert values and all(torch.isfinite(v).all() for v in values), prefix
            norm = float(sum(v.square().sum() for v in values).sqrt())
            assert norm > 0, prefix
            result[prefix + 'gradient_norm'] = norm
        assert all(p.grad is None and not p.requires_grad for p in model.proposer.parameters())
        result['proposer_gradients_none'] = True
    opt.step()
    assignments = [len(s) for s, _ in criterion.matcher(predictions, targets)]
    gt = [len(n) for n in targets['nodes']]
    result.update({k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1})
    result.update(step=step, lr=7e-5 * scale, matched_fraction=sum(assignments) / max(1, sum(gt)),
                  real_peaks=float(predictions['anchor_peak_counts'].float().mean()))
    return result


def save_checkpoint(path, model, opt, **extra):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    payload = dict(net={k: v for k, v in model.state_dict().items() if not k.startswith('proposer.')},
                   optimizer=None if opt is None else opt.state_dict(), plan_sha256=sha(OUT / 'plan.json'),
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), **extra)
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(model, path, opt=None):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    report = model.load_state_dict(checkpoint['net'], strict=False)
    assert not report.unexpected_keys and all(k.startswith('proposer.') for k in report.missing_keys)
    if opt is not None:
        opt.load_state_dict(checkpoint['optimizer'])
    return checkpoint


# ----------------------------------------------------------------------------- evaluation
def predict(model, plan, dataset, ids, split, workers, batch=EVAL_BATCH, keep_ids=None):
    """Per patch: retained nodes (p>0.1), all retained-pair edge probs, spline controls for p_edge>0.1."""
    import torch
    from torch.utils.data import DataLoader
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    base = base_dataset(plan, dataset, ids, split, False)
    loader = DataLoader(IndexedDataset(base), batch_size=batch, num_workers=workers, collate_fn=collate,
                        shuffle=False, pin_memory=True)
    was_training = model.training
    model.eval()
    buffers = {n: v.detach().clone() for n, v in model.named_buffers() if 'running_' in n or 'num_batches' in n}
    rows = []
    kwargs = dict(object_queries=OBJECT_QUERIES, relation_tokens=RELATION_TOKENS, node_threshold=NODE_FLOOR,
                  node_type_degree_caps=False, node_type_degree_penalty=0.)
    by_id = {r.sample_id: r for r in base.records}
    with torch.no_grad():
        for graph_batch, sample_ids in loader:
            volumes, targets = _move_graph_batch(graph_batch, 'cuda', 'image', True)
            tokens, predictions, _ = model(volumes, [padding_seed(dataset, i) for i in sample_ids])
            metrics = [g['metric'] for g in targets['spline']]
            candidates = infer_graphs(tokens, predictions, model.relation_embed, edge_threshold=-1., **kwargs)
            geometry = infer_graphs(tokens, predictions, model.relation_embed, edge_threshold=EDGE_FLOOR,
                                    spline_head=model.spline_head, spline_metrics=metrics, **kwargs)
            for j, sample in enumerate(sample_ids):
                c, g = candidates[j], geometry[j]
                assert torch.equal(c['query_ids'], g['query_ids'])
                record = by_id[sample]
                rows.append(dict(
                    sample_id=sample, split=split,
                    cache=str(record.graph.parent.parent / 'spline' / f'{sample}.npz'),
                    query_ids=c['query_ids'].numpy(), nodes=c['nodes'].numpy(), node_scores=c['node_scores'].numpy(),
                    pairs=c['edges'].numpy().reshape(-1, 2), edge_scores=c['edge_scores'].numpy(),
                    spline_edges=g['edges'].numpy().reshape(-1, 2), spline_scores=g['edge_scores'].numpy(),
                    spline_controls=g['spline_controls'].numpy(),
                    anchors=predictions['anchors'][j].cpu().numpy(),
                    anchor_scores=predictions['anchor_scores'][j].cpu().numpy(),
                    anchor_peak_count=int(predictions['anchor_peak_counts'][j]),
                    raw_nodes=predictions['pred_nodes'][j, :, :3].cpu().numpy(),
                    raw_scores=predictions['pred_logits'][j].softmax(-1)[:, 1].cpu().numpy()))
    assert all(torch.equal(buffers[n], dict(model.named_buffers())[n]) for n in buffers), 'Evaluation altered BN'
    model.train(was_training)
    return rows


def graph_at(row, node_threshold, edge_threshold):
    """Decoded graph at (node, edge) thresholds from a cached row (same decisions as infer_graphs)."""
    keep = np.nonzero(row['node_scores'] > node_threshold)[0]
    remap = {int(k): i for i, k in enumerate(keep)}
    pairs = row['pairs']
    valid = np.isin(pairs, keep).all(1) if len(pairs) else np.zeros(0, bool)
    sub, scores = pairs[valid], row['edge_scores'][valid]
    local = np.array([[remap[int(a)], remap[int(b)]] for a, b in sub], int).reshape(-1, 2)
    return keep, sub, scores, local


def evaluate(rows, node_threshold, edge_threshold, geometry=True):
    """dense_graph_20261030.evaluate with RelationFormer decoding (strict '>' thresholds, spline controls)."""
    from diagnose_branch_relations import match_nodes
    from ixi_query_diagnostics import edge_accounting
    from metrics.common_vessel_metrics import evaluate_common_vessels, aggregate_common_vessels
    totals = dict(node_tp=0, node_fp=0, node_fn=0, tp=0, fp=0, fn=0, missing_endpoint_edges=0, MM_fp=0, MU_fp=0,
                  UU_fp=0, beta0_error=0, beta1_error=0)
    common = []
    for r in rows:
        with np.load(r['cache']) as z:
            gt_nodes, gt_edges, metric = z['nodes'], z['edges'].reshape(-1, 2), z['metric']
        keep, sub, scores, local = graph_at(r, node_threshold, edge_threshold)
        nodes = r['nodes'][keep]
        mapping = match_nodes(nodes @ metric.T, gt_nodes @ metric.T, TOLERANCE_MM)
        row, chosen, *_ = edge_accounting(np.arange(len(nodes)), local, scores, scores > edge_threshold, mapping,
                                          gt_edges, len(gt_nodes))
        for k in totals:
            totals[k] += row[k]
        if geometry:
            lookup = {tuple(sorted(map(int, e))): (tuple(map(int, e)), c)
                      for e, c in zip(r['spline_edges'], r['spline_controls'])}
            remap = {int(k): i for i, k in enumerate(keep)}
            edges, controls = [], []
            for a, b in sub[chosen]:
                (u, v), c = lookup[tuple(sorted((int(a), int(b))))]
                edges.append((remap[u], remap[v]))
                controls.append(c)
            prediction = dict(nodes=nodes, edges=np.array(edges, int).reshape(-1, 2),
                              spline_controls=np.array(controls).reshape(-1, 8, 3), spline_family='bspline')
            common.append(evaluate_common_vessels(prediction, r['cache']))
    f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0  # noqa: E731
    result = dict(node_threshold=node_threshold, edge_threshold=edge_threshold,
                  node_f1=f1(totals['node_tp'], totals['node_fp'], totals['node_fn']),
                  node_precision=totals['node_tp'] / max(1, totals['node_tp'] + totals['node_fp']),
                  node_recall=totals['node_tp'] / max(1, totals['node_tp'] + totals['node_fn']),
                  edge_f1=f1(totals['tp'], totals['fp'], totals['fn']),
                  edge_precision=totals['tp'] / max(1, totals['tp'] + totals['fp']),
                  edge_recall=totals['tp'] / max(1, totals['tp'] + totals['fn']),
                  missed_edges_without_endpoint_pair=totals['missing_endpoint_edges'] / max(1, totals['fn']),
                  false_edges_MM_MU_UU=[totals['MM_fp'], totals['MU_fp'], totals['UU_fp']],
                  beta0_mae=totals['beta0_error'] / len(rows), beta1_mae=totals['beta1_error'] / len(rows),
                  patches=len(rows))
    if geometry:
        agg = aggregate_common_vessels(common)
        result.update({k: agg[k] for k in ('common_branch_f1', 'common_curve_f1', 'common_curve_precision',
                                           'common_curve_recall', 'common_guide_distance_mm', 'common_anchor_f1')})
    return result


def select(rows):
    """dense_graph_20261030.select: node threshold by node F1 at edge 0.5, then edge threshold by edge F1."""
    node = max(NODE_THRESHOLDS, key=lambda t: evaluate(rows, t, 0.5, geometry=False)['node_f1'])
    edge = max(EDGE_THRESHOLDS, key=lambda t: evaluate(rows, node, t, geometry=False)['edge_f1'])
    return node, edge


def anchor_bound(rows):
    """Upper bound on node recall: GT nodes with an anchor within 2 mm (one-to-one and any)."""
    from diagnose_branch_relations import match_nodes
    one, anyone, total, peaks, confident = 0, 0, 0, [], []
    for r in rows:
        with np.load(r['cache']) as z:
            gt, metric = z['nodes'] @ z['metric'].T, z['metric']
        anchors = r['anchors'] @ metric.T
        one += len(match_nodes(anchors, gt, TOLERANCE_MM))
        if len(gt):
            anyone += int((np.linalg.norm(anchors[:, None] - gt[None], axis=-1).min(0) <= TOLERANCE_MM).sum())
        total += len(gt)
        peaks.append(r['anchor_peak_count'])
        confident.append(int((r['anchor_scores'] >= 0.05).sum()))
    return dict(one_to_one_recall=one / max(1, total), any_anchor_within_2mm=anyone / max(1, total), gt_nodes=total,
                mean_real_peaks=float(np.mean(peaks)), padded_patch_fraction=float(np.mean(np.array(peaks) < 256)),
                mean_anchors_score_ge_0p05=float(np.mean(confident)), patches=len(rows))


def parity_check(model, plan, dataset, ids, split, rows):
    """Cached-row decoding at 0.5/0.5 == direct infer_graphs(0.5, 0.5, spline) graphs."""
    import torch
    from torch.utils.data import DataLoader
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    base = base_dataset(plan, dataset, ids, split, False)
    loader = DataLoader(IndexedDataset(base), batch_size=EVAL_BATCH, num_workers=0, collate_fn=collate)
    model.eval()
    by_id = {r['sample_id']: r for r in rows}
    checked = 0
    with torch.no_grad():
        for graph_batch, sample_ids in loader:
            volumes, targets = _move_graph_batch(graph_batch, 'cuda', 'image', True)
            tokens, predictions, _ = model(volumes, [padding_seed(dataset, i) for i in sample_ids])
            direct = infer_graphs(tokens, predictions, model.relation_embed, object_queries=OBJECT_QUERIES,
                                  relation_tokens=RELATION_TOKENS, node_threshold=.5, edge_threshold=.5,
                                  spline_head=model.spline_head,
                                  spline_metrics=[g['metric'] for g in targets['spline']])
            for j, sample in enumerate(sample_ids):
                row, d = by_id[sample], direct[j]
                keep, sub, scores, _ = graph_at(row, .5, .5)
                assert np.array_equal(row['query_ids'][keep], d['query_ids'].numpy())
                np.testing.assert_allclose(row['nodes'][keep], d['nodes'].numpy(), atol=1e-6)
                cached = {tuple(sorted(row['query_ids'][[a, b]])) for a, b in sub[scores > .5]}
                qd = d['query_ids'].numpy()
                direct_edges = {tuple(sorted(qd[[a, b]])) for a, b in d['edges'].numpy()}
                assert cached == direct_edges, (sample, len(cached), len(direct_edges))
                lookup = {tuple(sorted(row['query_ids'][list(e)])): c
                          for e, c in zip(row['spline_edges'], row['spline_controls'])}
                for e, c in zip(d['edges'].numpy(), d['spline_controls'].numpy()):
                    np.testing.assert_allclose(lookup[tuple(sorted(qd[list(e)]))], c, atol=1e-5)
                checked += 1
    model.train()
    return dict(patches=checked, edges_and_controls_identical=True)


def full_report(model, plan, dataset, workers, folder, monitor_ids=None, val_ids=None):
    entry = plan['datasets'][dataset]
    monitor_ids = entry['monitor_train'] if monitor_ids is None else monitor_ids
    val_ids = entry['val_ids'] if val_ids is None else val_ids
    start = time.monotonic()
    train_rows = predict(model, plan, dataset, monitor_ids, 'train', workers)
    val_rows = predict(model, plan, dataset, val_ids, 'val', workers)
    inference_seconds = time.monotonic() - start
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in [('monitor_train', train_rows), ('validation', val_rows)]:
        with (folder / f'{name}-rows.pkl').open('wb') as stream:
            pickle.dump(rows, stream)
    node_t, edge_t = select(train_rows)
    result = dict(
        thresholds=dict(node=node_t, edge=edge_t, policy='monitor_train only; node by node F1 at edge 0.5, then edge'),
        train_monitor_selected=evaluate(train_rows, node_t, edge_t),
        train_monitor_fixed_0p5=evaluate(train_rows, .5, .5),
        validation_selected=evaluate(val_rows, node_t, edge_t),
        validation_fixed_0p5=evaluate(val_rows, .5, .5),
        anchor_upper_bound=dict(train_monitor=anchor_bound(train_rows), validation=anchor_bound(val_rows)),
        populations=dict(monitor_train=len(train_rows), validation=len(val_rows)),
        inference_seconds=inference_seconds, total_seconds=time.monotonic() - start)
    return result


def quick_curve(model, plan, dataset, workers):
    entry = plan['datasets'][dataset]
    train_rows = predict(model, plan, dataset, entry['monitor_train'][:64], 'train', workers)
    val_rows = predict(model, plan, dataset, entry['val_ids'][:128], 'val', workers)
    return dict(train_monitor_first64=evaluate(train_rows, .5, .5, geometry=False),
                validation_first128=evaluate(val_rows, .5, .5, geometry=False),
                anchor_upper_bound_validation_first128=anchor_bound(val_rows))


# ----------------------------------------------------------------------------- smoke
def smoke(workers):
    import torch
    from torch.utils.data import DataLoader
    from models.anchor_matcher import AnchorMatcher
    from training.engine import _move_graph_batch
    plan = read(OUT / 'plan.json')
    assert sha(plan['source']['path']) == plan['source']['sha256']
    for name in DATASETS:
        assert sha(plan['datasets'][name]['probe_checkpoint']) == plan['datasets'][name]['probe_sha256']
    result = dict(job_id=os.environ.get('SLURM_JOB_ID'), dataset='BRAVA')
    t0 = time.monotonic()
    dataset = 'BRAVA'
    ids = plan['datasets'][dataset]['train_ids']

    # 1) Initialization, zero residual at start, determinism and matcher invariance on real data.
    cfg, model, criterion, opt, migration = setup(plan, dataset, 16)
    result['migration'] = migration
    base = base_dataset(plan, dataset, ids, 'train', True)
    seeded = SeededDataset(base)
    loader = DataLoader(seeded, batch_sampler=EpochBatches(16, len(ids)), num_workers=workers, collate_fn=collate)
    batch = next(iter(loader))
    volumes, targets = _move_graph_batch(batch[0], 'cuda', 'image', True)
    seeds = [padding_seed(dataset, i) for i in batch[1]]
    model.train()
    with torch.no_grad():
        tokens_a, pred_a, _ = model(volumes, seeds)
        tokens_b, pred_b, _ = model(volumes, seeds)
    assert not model.proposer.training
    assert torch.equal(pred_a['anchors'], pred_b['anchors']) and torch.equal(pred_a['anchor_scores'], pred_b['anchor_scores'])
    torch.testing.assert_close(pred_a['pred_nodes'][..., :3], pred_a['anchors'], rtol=0, atol=0)
    matcher = AnchorMatcher(TOLERANCE_MM)
    first, second = matcher(pred_a, targets), matcher(pred_b, targets)
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(first, second))
    permutation = torch.randperm(OBJECT_QUERIES, generator=torch.Generator().manual_seed(5))
    permuted = matcher(dict(anchors=pred_a['anchors'][:, permutation.to(volumes.device)]), targets)
    for (s, t), (ps, pt) in zip(first, permuted):
        assert {(int(a), int(b)) for a, b in zip(s, t)} == {(int(permutation[a]), int(b)) for a, b in zip(ps, pt)}
    result['checks'] = dict(anchors_identical_two_passes=True, initial_nodes_equal_anchors=True,
                            matcher_identical_two_passes=True, matcher_query_order_invariant=True,
                            proposer_eval_in_train_mode=True,
                            batch_matched_fraction=sum(len(s) for s, _ in first) / sum(len(n) for n in targets['nodes']),
                            batch_mean_real_peaks=float(pred_a['anchor_peak_counts'].float().mean()))
    result['diagnostic_update'] = update(model, criterion, opt, batch, 1, .01, dataset, diagnostic=True)
    print(json.dumps(result, indent=1), flush=True)
    del model, criterion, opt, loader, batch, volumes, targets, tokens_a, tokens_b, pred_a, pred_b
    torch.cuda.empty_cache()

    # 2) Throughput / memory at batch 16 and 32 (benchmark updates discarded).
    rows = []
    for size in plan['batch_candidates']:
        cfg, model, criterion, opt, _ = setup(plan, dataset, size)
        loader = DataLoader(SeededDataset(base), batch_sampler=EpochBatches(size, len(ids)), num_workers=workers,
                            collate_fn=collate, pin_memory=True, prefetch_factor=2)
        torch.cuda.reset_peak_memory_stats()
        timings, iterator = [], iter(loader)
        for number in range(1, 21):
            torch.cuda.synchronize()
            start = time.perf_counter()
            last = update(model, criterion, opt, next(iterator), number, .01, dataset)
            torch.cuda.synchronize()
            if number > 4:
                timings.append(time.perf_counter() - start)
        seconds = float(np.median(timings))
        rows.append(dict(batch_size=size, seconds_per_update=seconds, mean_seconds_per_update=float(np.mean(timings)),
                         patches_per_second=size / seconds, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                         total_memory_bytes=torch.cuda.get_device_properties(0).total_memory, last=last))
        print('benchmark', json.dumps(rows[-1]), flush=True)
        del model, criterion, opt, loader, iterator
        torch.cuda.empty_cache()
    valid = {r['batch_size']: r['peak_reserved_bytes'] < .85 * r['total_memory_bytes'] for r in rows}
    speed = {r['batch_size']: r['patches_per_second'] for r in rows}
    chosen = 32 if valid[32] and (not valid[16] or speed[32] >= .95 * speed[16]) else 16
    seconds = next(r['mean_seconds_per_update'] for r in rows if r['batch_size'] == chosen)
    result['benchmark'] = rows
    result['batch_size'] = chosen

    # 3) Memorize 8 BRAVA patches (no augmentation, fixed batch of 8, ~500 updates).
    cfg, model, criterion, opt, _ = setup(plan, dataset, 8)
    tiny_ids = ids[:8]
    tiny = IndexedDataset(base_dataset(plan, dataset, tiny_ids, 'train', False))
    memo_loader = DataLoader(tiny, batch_sampler=FixedBatches(8, 500), num_workers=2, collate_fn=collate)
    eval_start = time.monotonic()
    before = predict(model, plan, dataset, tiny_ids, 'train', 0)
    result['memorize_before'] = dict(fixed_0p5=evaluate(before, .5, .5), anchor_upper_bound=anchor_bound(before))
    eval_seconds = time.monotonic() - eval_start
    history = []
    memo_start = time.monotonic()
    for step, batch in enumerate(memo_loader, 1):
        row = update(model, criterion, opt, batch, step, min(1., .01 + .99 * step / 25), dataset)
        if step % 50 == 0:
            history.append({k: row[k] for k in ('step', 'total', 'class', 'nodes', 'edges', 'spline')})
            print('memorize', json.dumps(history[-1]), flush=True)
    result['memorize_seconds'] = time.monotonic() - memo_start
    after = predict(model, plan, dataset, tiny_ids, 'train', 0)
    node_t, edge_t = select(after)
    result['memorize'] = dict(patches=tiny_ids, updates=500, batch=8, augmentation=False, history=history,
                              fixed_0p5=evaluate(after, .5, .5), selected_on_same_patches=evaluate(after, node_t, edge_t),
                              anchor_upper_bound=anchor_bound(after),
                              gt_nodes=int(sum(len(np.load(r['cache'])['nodes']) for r in after)))
    result['evaluation_parity'] = parity_check(model, plan, dataset, tiny_ids, 'train', after)
    t = time.monotonic()
    full_report(model, plan, dataset, workers, LARGE / 'smoke' / 'report', monitor_ids=tiny_ids,
                val_ids=plan['datasets'][dataset]['val_ids'][:8])
    report_seconds_per_patch = (time.monotonic() - t) / 16
    result['evaluation_seconds_per_patch'] = dict(before_memorize_8=eval_seconds / 8,
                                                  full_report=report_seconds_per_patch)

    # 4) Production estimates (whole epochs around 15,000 updates at the chosen batch).
    estimates = {}
    for name in DATASETS:
        entry = plan['datasets'][name]
        per_epoch = len(entry['train_ids']) // chosen
        epochs = round(TARGET_UPDATES / per_epoch)
        updates = epochs * per_epoch
        curves = updates // 2500
        eval_patches = len(entry['monitor_train']) + len(entry['val_ids'])
        hours = (updates * seconds + 150 + curves * 192 * report_seconds_per_patch
                 + eval_patches * report_seconds_per_patch) / 3600
        estimates[name] = dict(updates_per_epoch=per_epoch, epochs=epochs, updates=updates,
                               estimated_hours=hours, walltime_hours=max(.75, 1.3 * hours))
    result['estimates'] = estimates
    result['seconds'] = time.monotonic() - t0
    result['passed'] = True
    write(OUT / 'smoke.json', result)
    write(OUT / 'batch-selection.json', dict(batch_size=chosen, rows=rows, estimates=estimates,
                                             job_id=os.environ.get('SLURM_JOB_ID')))
    print(json.dumps({k: result[k] for k in ('memorize', 'estimates', 'batch_size')}, indent=1), flush=True)


# ----------------------------------------------------------------------------- training
def train(dataset, workers):
    import torch
    from torch.utils.data import DataLoader
    assert read(OUT / 'smoke.json')['passed']
    plan = read(OUT / 'plan.json')
    selection = read(OUT / 'batch-selection.json')
    size = selection['batch_size']
    entry = plan['datasets'][dataset]
    per_epoch = len(entry['train_ids']) // size
    epochs = round(TARGET_UPDATES / per_epoch)
    total = epochs * per_epoch
    run = LARGE / 'runs' / dataset
    run.mkdir(parents=True, exist_ok=True)
    latest = run / 'latest.pt'
    initialize = not latest.exists()
    cfg, model, criterion, opt, migration = setup(plan, dataset, size, initialize=initialize)
    epoch, step, offset, history, curve = 1, 0, 0, [], []
    if initialize:
        write(run / 'config.json', cfg)
        write(run / 'initialization.json', dict(dataset=dataset, migration=migration, seed=SEED, batch_size=size,
                                                epochs=epochs, updates=total, updates_per_epoch=per_epoch))
    else:
        checkpoint = load_checkpoint(model, latest, opt)
        assert checkpoint['plan_sha256'] == sha(OUT / 'plan.json') and checkpoint['batch_size'] == size
        epoch, step, offset = checkpoint['epoch'], checkpoint['step'], checkpoint['offset']
        history = read(run / 'history.json') if (run / 'history.json').exists() else []
        curve = read(run / 'curve.json') if (run / 'curve.json').exists() else []
        print('resumed', dataset, 'epoch', epoch, 'offset', offset, 'step', step, flush=True)
        del checkpoint
    base = base_dataset(plan, dataset, entry['train_ids'], 'train', True)
    sampler = EpochBatches(size, len(entry['train_ids']))
    assert len(sampler) == per_epoch
    loader = DataLoader(SeededDataset(base), batch_sampler=sampler, num_workers=workers, collate_fn=collate,
                        pin_memory=True, persistent_workers=False, prefetch_factor=2)
    stop = [False]
    for sig in [signal.SIGUSR1, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    curve_every = 2500
    while step < total:
        sampler.epoch, sampler.start = epoch, offset
        losses = []
        for batch in loader:
            step += 1
            offset += 1
            row = update(model, criterion, opt, batch, step, lr_scale(step, per_epoch, epochs), dataset,
                         diagnostic=step == 1)
            losses.append(row)
            if step == 1:
                write(run / 'first-update-diagnostic.json', row)
            if step % 50 == 0:
                print(dataset, 'epoch', epoch, 'batch', offset, '/', per_epoch, 'step', step, '/', total,
                      'loss', round(row['total'], 5), 'matched', round(row['matched_fraction'], 3),
                      'lr', row['lr'], flush=True)
            if step % curve_every == 0 or step == total:
                curve.append(dict(step=step, epoch=epoch, seconds=time.monotonic() - start,
                                  **quick_curve(model, plan, dataset, min(workers, 4))))
                write(run / 'curve.json', curve)
                print('CURVE', json.dumps({k: v for k, v in curve[-1].items() if k != 'anchor_upper_bound_validation_first128'}),
                      flush=True)
            if stop[0] or step % 1000 == 0:
                save_checkpoint(latest, model, opt, epoch=epoch, step=step, offset=offset, batch_size=size,
                                dataset=dataset)
                write(run / 'history.json', history)
                if stop[0]:
                    print('Signal received; resumable checkpoint saved at step', step, flush=True)
                    sys.exit(3)
        keys = set.intersection(*(set(r) for r in losses))
        history.append(dict(epoch=epoch, step=step, seconds=time.monotonic() - start,
                            **{k: float(np.mean([r[k] for r in losses])) for k in sorted(keys)
                               if isinstance(losses[0][k], (int, float)) and k != 'step'}))
        write(run / 'history.json', history)
        epoch, offset = epoch + 1, 0
    elapsed = time.monotonic() - start
    save_checkpoint(run / 'model-final.pt', model, None, epoch=epochs, step=step, offset=0, batch_size=size,
                    dataset=dataset)
    save_checkpoint(latest, model, opt, epoch=epoch, step=step, offset=0, batch_size=size, dataset=dataset)
    write(run / 'training-complete.json', dict(dataset=dataset, updates=step, epochs=epochs, batch_size=size,
                                               seconds=elapsed, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                               job_id=os.environ.get('SLURM_JOB_ID')))
    report(dataset, workers, model=model)


def report(dataset, workers, model=None):
    plan = read(OUT / 'plan.json')
    run = LARGE / 'runs' / dataset
    complete = read(run / 'training-complete.json')
    if model is None:
        _, model, _, _, _ = setup(plan, dataset, complete['batch_size'], initialize=False)
        load_checkpoint(model, run / 'model-final.pt')
    result = full_report(model, plan, dataset, workers, run / 'evaluation')
    result.update(dataset=dataset, method='heatmap-anchored RelationFormer', updates=complete['updates'],
                  epochs=complete['epochs'], batch_size=complete['batch_size'],
                  training_seconds=complete['seconds'], curve=read(run / 'curve.json'),
                  checkpoint=str(run / 'model-final.pt'), job_id=os.environ.get('SLURM_JOB_ID'))
    write(run / 'final.json', result)
    write(OUT / f'final-{dataset}.json', result)
    print('FINAL', json.dumps({k: result[k] for k in ('thresholds', 'validation_selected', 'validation_fixed_0p5',
                                                       'anchor_upper_bound')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'smoke', 'train', 'evaluate'])
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'smoke':
        smoke(args.workers)
    elif args.mode == 'train':
        train(args.dataset, args.workers)
    else:
        report(args.dataset, args.workers)
