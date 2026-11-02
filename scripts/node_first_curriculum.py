"""L08: scratch B0 joint versus node-first training, with fixed data streams."""
import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[1]
MAIN = next(parent / 'Graph-Native-Betti-Matching-Representation'
            for parent in REPO.parents
            if (parent / 'Graph-Native-Betti-Matching-Representation/docs/repo-graph.md').is_file())
META = REPO / 'experiments/node_first_curriculum_20261028'
ARMS = ['joint', 'node_first']


def read(path):
    return json.loads(Path(path).read_text())


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def plan():
    return read(META / 'plan.json')


def seed_all(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare():
    assert not (META / 'plan.json').exists(), 'Preserve the registered immutable plan'
    prior = read(META / 'reference-plan.json')
    cfg, spec = copy.deepcopy(prior['config']), copy.deepcopy(prior['dataset'])
    assert prior['baseline_updates'] == 78200 and cfg['data']['batch_size'] == 32
    assert cfg['model']['decoder']['object_queries'] == 120
    assert cfg['model']['decoder']['relation_tokens'] == 2
    assert cfg['loss']['node']['classification']['name'] == 'focal'
    assert cfg['loss']['node']['classification']['class_weights'] == [1., 1.]
    assert cfg['loss']['node']['classification']['focal_gamma'] == 2
    assert cfg['loss']['edge']['classification']['name'] == 'cross_entropy'
    assert cfg['loss']['edge']['balancing']['positive_to_negative_ratio'] == .15
    assert not cfg['loss']['edge']['candidates']['include_unmatched']
    assert not cfg['model']['node_refinement']['enabled']
    assert cfg['model']['spline']['controls'] == 8
    assert cfg['training']['epochs'] == 100 and cfg['training']['warmup_epochs'] == 1
    root = Path(spec['root'])
    assert hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest() == spec['manifest_sha256']
    for split, key in [('train', 'train_ids'), ('val', 'val_ids')]:
        assert sorted(p.stem for p in (root / split / 'spline').glob('*.npz')) == spec[key]
    subjects = lambda ids: {spec['subjects'][s.split('_')[1]] for s in ids}
    assert not subjects(spec['train_ids']) & subjects(spec['val_ids'])
    assert (len(subjects(spec['train_ids'])), len(subjects(spec['val_ids']))) == (64, 13)
    cfg['tracking']['enabled'] = False
    cfg['runtime']['workers'] = 4
    cfg['experiment'].update(name='L08_node_first_scratch', seed=364505)
    spec['monitor_train'] = spec['monitor_train'][:128]
    p = dict(id='L08', config=cfg, dataset=spec, seeds=[364505, 92741],
             output_root='/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/node_first_curriculum_20261028',
             source_initialization='random CPU construction only; no graph-pretrained state',
             stages={'pilot': dict(updates=15640, epochs=20, seeds=[364505]),
                     'full': dict(updates=78200, epochs=100, seeds=[364505, 92741])},
             scheduler_horizon_updates=78200, gradient_route='graph_only',
             spline_policy='unchanged from update1; no spline gradient into detector',
             ramp='half cosine; zero through floor(0.1U), one from floor(0.2U)',
             inference_batch=1, node_threshold=.5, edge_threshold=.5,
             subject_bootstrap_draws=1000, bootstrap_seed=281028,
             max_h100=2, global_h100_limit=2,
             promotion=dict(auto_on_clear_positive=True, edge_f1_min_delta=.01,
                            edge_f1_ci95_lower_min=0., common_conditional_ap_min_delta=.005,
                            branch_f1_min_delta=0., curve_f1_min_delta=0.,
                            calibrated_edge_f1_min_delta=.005, require_non_decreasing_edge_tp=True,
                            note='Single-seed learning-curve gate, not a final efficacy conclusion; inconclusive/negative never auto-expand'),
             pilot_lr='First20epochs of unchanged100epoch polynomial schedule; curriculum percentages use pilot U',
             full_restart='Both arms restart from step0 random state; do not resume compressed pilot curriculum',
             candidate_policy='unchanged matched Hungarian CE; ratio upsampling .15, no unmatched mining')
    write(META / 'plan.json', p)
    print('Prepared L08:25000/480 patches;64/13 subjects; pilot15640 and conditional full78200 updates.')


def make_initial(seed):
    import fcntl
    import torch
    from models import build_model
    from scripts.extend_scratch_geometry import atomic_checkpoint
    p = plan()
    path = Path(p['output_root']) / 'initializers' / f'random-{seed}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            seed_all(seed)
            cfg = copy.deepcopy(p['config'])
            cfg['experiment']['seed'] = seed
            model = build_model(cfg)  # No checkpoint or pretrained backbone call.
            digest = state_digest(model.state_dict())
            atomic_checkpoint(path, dict(net=model.state_dict(), source_type='random',
                                         training_updates=0, seed=seed, state_sha256=digest,
                                         config=cfg, optimizer_state_empty=True))
            write(path.with_suffix('.json'), dict(path=str(path), seed=seed,
                state_sha256=digest, training_updates=0, initialization='random CPU model constructor; no pretrained tensors'))
            del model
    return path


def setup(seed, arm, stage, *, workers=None):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from configs import validate_config
    from models import build_model
    from models.matcher import build_matcher
    from models.ops.modules import MSDeformAttn
    from data.loaders import build_data_loaders
    from data.loaders.common import image_graph_collate, seed_data_worker
    from scripts.train_ixi_transfer import TransferCriterion
    from training.optim import build_optimizer, build_scheduler
    from training.node_first_curriculum import ScheduledGraphObjective
    p = plan()
    cfg = copy.deepcopy(p['config'])
    cfg['experiment']['seed'] = seed
    if workers is not None:
        cfg['runtime']['workers'] = workers
    validate_config(cfg)
    seed_all(seed)
    model = build_model(cfg)
    initial = torch.load(make_initial(seed), map_location='cpu', weights_only=False)
    assert initial['source_type'] == 'random' and initial['training_updates'] == 0
    assert initial['seed'] == seed and initial['optimizer_state_empty']
    model.load_state_dict(initial['net'], strict=True)
    assert state_digest(model.state_dict()) == initial['state_sha256']
    initial_sha = initial['state_sha256']
    del initial
    model.cuda().requires_grad_(True)
    for layer in model.decoder.decoder.layers:
        if layer.rln_attn:  # Same already-dormant legacy sub-block as modern B0.
            layer.cross_attn2.requires_grad_(False)
            layer.norm12.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, MSDeformAttn):
            module.record_level_diagnostics = False
    criterion = TransferCriterion(cfg, build_matcher(cfg), model.relation_embed,
                                  spline_head=model.spline_head, route='graph_only').cuda()
    original, _ = build_data_loaders(cfg)
    names = [r.sample_id for r in original.dataset.datasets[0].records]
    assert names == p['dataset']['train_ids'] and len(names) == 25000

    class Tagged(Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, index):
            return names[index], self.base[index]

    def collate(items):
        ids, samples = zip(*items)
        return list(ids), image_graph_collate(samples)

    def worker_init(worker_id):
        seed_data_worker(worker_id)
        def initialize(ds):
            if getattr(ds, 'intensity_transform', None) is not None:
                ds.intensity_transform.set_random_state(seed=torch.initial_seed() % (2 ** 32))
            if hasattr(ds, 'base'):
                initialize(ds.base)
            for sub in getattr(ds, 'datasets', []):
                initialize(sub)
        initialize(torch.utils.data.get_worker_info().dataset)

    loader = DataLoader(Tagged(original.dataset), batch_size=original.batch_size,
                        shuffle=True, generator=original.generator, num_workers=original.num_workers,
                        pin_memory=original.pin_memory, drop_last=original.drop_last,
                        worker_init_fn=worker_init, collate_fn=collate)
    assert len(loader) == 782
    optimizer = build_optimizer(cfg, model)
    assert not optimizer.state
    scheduler = build_scheduler(cfg, optimizer, len(loader))  # Unchanged100epoch horizon.
    objective = ScheduledGraphObjective(model, criterion, arm, p['stages'][stage]['updates'])
    return dict(plan=p, cfg=cfg, model=model, criterion=criterion, loader=loader,
                opt=optimizer, scheduler=scheduler, objective=objective, initial_sha256=initial_sha,
                seed=seed, arm=arm, stage=stage)


def chain_hash(previous, ids, batch):
    from scripts.local_refinement_long import hash_tree
    h = hashlib.sha256(previous.encode())
    hash_tree(h, ids)
    hash_tree(h, batch)
    return h.hexdigest()


def snapshot(b, out, step, *, initial=False, tiny=False):
    import numpy as np
    import torch
    from scripts.multidataset_graph_evaluation import loaders
    from scripts.train_spline_weight_pilot import monitoring_rng
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    model, criterion, p = b['model'], b['criterion'], b['plan']
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = copy.deepcopy(p['dataset'])
    if tiny:
        spec['monitor_train'] = spec['monitor_train'][:2]
        spec['val_ids'] = spec['val_ids'][:2]
    rows = []
    model.eval()
    saved_validation = criterion.validation
    criterion.validation = True
    temporary = out / 'caches.zip.tmp'
    try:
        with monitoring_rng(), torch.no_grad(), zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            seed_all(b['seed'] + 400000)
            for split, names, loader in loaders(b['cfg'], {'datasets': {'SyntheticMRI': spec}}, {'dataset': 'SyntheticMRI'}):
                for index, batch in enumerate(loader):
                    volumes, targets = _move_graph_batch(batch, 'cuda', 'image', True)
                    tokens, pred, _ = model(volumes)
                    source, target = criterion.matcher(pred, targets)[0]
                    source, target = source.cpu().numpy(), target.cpu().numpy()
                    cache = dict(raw_boxes=pred['pred_nodes'][0].cpu().numpy(),
                                 raw_logits=pred['pred_logits'][0].cpu().numpy(),
                                 scores=pred['pred_logits'][0].softmax(-1)[:, 1].cpu().numpy(),
                                 hungarian_source=source, hungarian_target=target,
                                 truth_normalized=targets['nodes'][0].cpu().numpy(),
                                 gt_edges=targets['edges'][0].cpu().numpy(),
                                 metric=targets['spline'][0]['metric'].cpu().numpy())
                    cache['physical'] = cache['raw_boxes'][:, :3] @ cache['metric'].T
                    cache['truth'] = cache['truth_normalized'] @ cache['metric'].T
                    loss_values = {}
                    if not initial:
                        raw = infer_graphs(tokens, pred, model.relation_embed,
                            object_queries=120, relation_tokens=2, node_threshold=-1., edge_threshold=-1.)[0]
                        graph = infer_graphs(tokens, pred, model.relation_embed,
                            object_queries=120, relation_tokens=2, node_threshold=.5, edge_threshold=.5,
                            spline_head=model.spline_head, spline_metrics=[targets['spline'][0]['metric']])[0]
                        cache.update(pairs=raw['query_ids'][raw['edges']].numpy(), relations=raw['edge_scores'].numpy(),
                                     decoder_tokens=tokens[0].cpu().numpy())
                        cache.update({'graph_' + k: v.numpy() for k, v in graph.items() if torch.is_tensor(v)})
                        assert np.allclose(cache['graph_spline_controls'][:, [0, -1]],
                                           cache['graph_nodes'][cache['graph_edges']], atol=1e-6)
                        losses = criterion(tokens, pred, targets)
                        loss_values = {k: float(v) for k, v in losses.items() if v.numel() == 1}
                    sample = names[index]
                    key = split + '/' + sample + '.npz'
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer, **cache)
                    archive.writestr(key, buffer.getvalue())
                    rows.append(dict(sample=sample, split=split,
                        subject=spec['subjects'][sample.split('_')[1]], cache_key=key,
                        target_path=str(Path(spec['root']) / split / 'spline' / (sample + '.npz')),
                        loss=loss_values))
        temporary.replace(out / 'caches.zip')
        write(out / 'records.json', rows)
        write(out / 'complete.json', dict(completed=True, step=step, initial=initial,
                                         inference_batch=1, patches=len(rows)))
    finally:
        criterion.validation = saved_validation


def smoke():
    import numpy as np
    import torch
    from training.engine import _move_graph_batch
    from training.node_first_curriculum import scheduled_total
    p = plan()
    root = Path(p['output_root'])
    gate = dict(completed=False, no_pretrained_weights=True, arms={})
    benchmarks, digests = [], []
    for arm in ARMS:
        b = setup(p['seeds'][0], arm, 'pilot')
        for package in ['models', 'training', 'data', 'configs']:
            source = Path(sys.modules[package].__file__).resolve()
            assert source.is_relative_to(REPO), ('Wrong worktree imported', package, str(source))
        digests.append(b['initial_sha256'])
        loader = b['loader']
        loader.generator.manual_seed(p['seeds'][0] + 1)
        ids, batch = next(iter(loader))
        data_digest = chain_hash('', ids, batch)
        gate['arms'][arm] = dict(initial_state_sha256=b['initial_sha256'], first_batch_sha256=data_digest)
        x, y = _move_graph_batch(batch, 'cuda', 'image', True)
        model, criterion = b['model'], b['criterion']
        model.train()
        # Both real detector and geometry paths are checked on the same batch32.
        relation_before = state_digest(model.relation_embed.state_dict())
        times = []
        for iteration in range(1, 5):
            seed_all(p['seeds'][0] + iteration)
            b['opt'].zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.monotonic()
            total, logs = b['objective'](x, y, iteration, probe=iteration == 1)
            if iteration == 1:
                gate['arms'][arm]['gradient_alignment'] = logs['gradient_alignment']
            assert torch.isfinite(total)
            total.backward()
            if arm == 'node_first':
                assert all(v.grad is None for v in model.relation_embed.parameters())
                assert all(v not in b['opt'].state for v in model.relation_embed.parameters())
            else:
                assert sum(float(v.grad.norm()) for v in model.relation_embed.parameters() if v.grad is not None) > 0
            b['opt'].step()
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
        if arm == 'node_first':
            assert state_digest(model.relation_embed.state_dict()) == relation_before
            b['opt'].zero_grad(set_to_none=True)
            total, logs = b['objective'](x, y, p['stages']['pilot']['updates'] // 10 + 1)
            total.backward()
            assert sum(float(v.grad.norm()) for v in model.relation_embed.parameters() if v.grad is not None) > 0
            b['opt'].step()
            assert all(int(b['opt'].state[v]['step']) == 1 for v in model.relation_embed.parameters())
        benchmarks.append(float(np.median(times[1:])))
        gate['arms'][arm]['step_seconds'] = times
        # Strict graph_only route: geometry cannot update the shared detector.
        model.train()
        tokens, pred, _ = model(x[:1])
        one = {k: v[:1] if isinstance(v, list) else v[:1] for k, v in y.items()}
        losses = criterion(tokens, pred, one)
        parameters = [v for n, v in model.named_parameters() if v.requires_grad and not n.startswith('spline_head.')]
        grads = torch.autograd.grad(losses['spline'], parameters, allow_unused=True)
        assert all(g is None or not g.count_nonzero() for g in grads), 'Spline gradients reached detector'
        assert scheduled_total(losses, criterion, 1) is losses['total']
        # Exercise raw-query cache serialization and CPU reconstruction on four real patches.
        snapshot(b, root / 'smoke' / arm, 4, tiny=True)
        del b, model, criterion, x, y, tokens, pred, losses, total, logs, grads
        torch.cuda.empty_cache()
    assert len(set(digests)) == 1
    assert len({v['first_batch_sha256'] for v in gate['arms'].values()}) == 1
    assert max(benchmarks) * 15640 * 1.3 + 900 <= 3 * 3600, 'Pilot throughput exceeds bounded3h allocation'
    gate.update(completed=True, benchmark_seconds=max(benchmarks),
                pilot_train_seconds_estimate=max(benchmarks) * 15640,
                full_train_seconds_estimate=max(benchmarks) * 78200,
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                job_id=os.getenv('SLURM_JOB_ID'))
    write(root / 'smoke.json', gate)
    print('L08 GPU correctness gate passed:', json.dumps(gate), flush=True)


def train(index, stage):
    import torch
    from training.engine import _move_graph_batch
    from scripts.extend_scratch_geometry import atomic_checkpoint, capture_rng, restore_rng
    p = plan()
    root = Path(p['output_root'])
    assert read(root / 'smoke.json')['completed']
    assert read(root / 'cpu-smoke.json')['completed']
    seeds = p['stages'][stage]['seeds']
    seed, arm = seeds[index // 2], ARMS[index % 2]
    b = setup(seed, arm, stage)
    model = b['model']
    out = root / stage / str(seed) / arm
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists(), 'Preserve completed runs'
    write(out / 'config.json', b['cfg'])
    steps = p['stages'][stage]['updates']
    checkpoints = [steps // 10, steps // 5, steps // 2, steps]
    assert all(step % len(b['loader']) == 0 for step in checkpoints)
    epoch, step, chain = 0, 0, ''
    latest = out / 'latest.pt'
    if latest.exists():
        cp = torch.load(latest, map_location='cpu', weights_only=False)
        assert cp['initial_sha256'] == b['initial_sha256'] and cp['arm'] == arm and cp['stage'] == stage
        model.load_state_dict(cp['net'], strict=True)
        b['opt'].load_state_dict(cp['optimizer'])
        b['scheduler'].load_state_dict(cp['scheduler'])
        restore_rng(cp['rng'], b['loader'])
        epoch, step, chain = cp['epoch'], cp['steps'], cp['batch_chain']
        del cp
    else:
        snapshot(b, out / 'initial-assignment', 0, initial=True)
    started = time.monotonic()
    probes = {1, 100, checkpoints[0], checkpoints[0] + 1, checkpoints[1], checkpoints[2], checkpoints[3]}
    def checkpoint():
        return dict(net=model.state_dict(), optimizer=b['opt'].state_dict(), scheduler=b['scheduler'].state_dict(),
                    config=b['cfg'], epoch=epoch, steps=step, rng=capture_rng(b['loader']), batch_chain=chain,
                    initial_sha256=b['initial_sha256'], arm=arm, stage=stage, seed=seed,
                    source_type='scratch continuation of own random step0 initializer')
    for epoch in range(epoch + 1, p['stages'][stage]['epochs'] + 1):
        b['loader'].generator.manual_seed(seed + epoch)
        for ids, batch in b['loader']:
            step += 1
            chain = chain_hash(chain, ids, batch)
            seed_all(seed + step)
            x, y = _move_graph_batch(batch, 'cuda', 'image', True)
            b['criterion'].set_training_progress(epoch, 100. * (step - 1) / max(1, steps - 1))
            model.train()
            b['opt'].zero_grad(set_to_none=True)
            total, metrics = b['objective'](x, y, step, probe=step in probes)
            if not torch.isfinite(total):
                raise FloatingPointError(f'nonfinite objective at{step}')
            total.backward()
            b['opt'].step()
            b['scheduler'].step()  # No additional clipping: exact modern scratch recipe.
            if step == 1 or step % 100 == 0 or step in probes:
                row = {k: float(v) if torch.is_tensor(v) else v for k, v in metrics.items()}
                relation_parameters = list(model.relation_embed.parameters())
                row['relation_gradient_norm'] = sum(float(v.grad.norm()) for v in relation_parameters if v.grad is not None)
                row['relation_optimizer_steps'] = max([int(b['opt'].state[v]['step'])
                    for v in relation_parameters if v in b['opt'].state] or [0])
                row.update(step=step, epoch=epoch, lr=b['opt'].param_groups[0]['lr'],
                           seconds=time.monotonic() - started, batch_chain=chain)
                with (out / 'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(clean(row)) + '\n')
                write(out / 'progress.json', row)
                print(stage, seed, arm, step, float(total), row['edge_multiplier'], flush=True)
        # All required comparisons are epoch boundaries; resume preserves the
        # exact sampler/worker seeds and a resumable chained full-batch digest.
        if step in checkpoints:
            atomic_checkpoint(out / f'model-{step}.pt', checkpoint())
            snapshot(b, out / f'step-{step}', step)
            write(out / f'step-{step}' / 'batch-proof.json', dict(batch_chain=chain, step=step))
        if epoch % 5 == 0 or step in checkpoints:
            atomic_checkpoint(latest, checkpoint())
    assert step == steps
    write(out / 'complete.json', dict(completed=True, steps=step, seed=seed, arm=arm, stage=stage,
        initial_state_sha256=b['initial_sha256'], batch_chain=chain, seconds=time.monotonic() - started,
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 2 ** 30, job_id=os.getenv('SLURM_JOB_ID')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'smoke', 'train'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--stage', choices=['pilot', 'full'], default='pilot')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        import torch
        torch.set_num_threads(1)
        if args.mode == 'smoke':
            smoke()
        else:
            train(args.index, args.stage)


if __name__ == '__main__':
    main()
