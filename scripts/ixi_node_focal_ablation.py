"""Bounded full-network IXI node-weight/focal continuation, with paired seeds."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.ixi_cached_mechanisms import save, sha

ARMS = ['ce_weighted', 'ce_equal', 'focal_equal']
SEED = 201021


def configure(cfg, arm):
    cfg = copy.deepcopy(cfg)
    node = cfg['loss']['node']['classification']
    node.update(name='focal' if arm == 'focal_equal' else 'weighted_cross_entropy',
                class_weights=[.25, .75] if arm == 'ce_weighted' else [1., 1.], focal_gamma=2.)
    node['curriculum']['enabled'] = False
    edge = cfg['loss']['edge']
    edge['classification'].update(name='cross_entropy', class_weights=[1., 1.])
    edge['classification']['curriculum']['enabled'] = False
    edge['balancing'].update(mode='ratio_upsample', positive_to_negative_ratio=.15, tolerance=.02)
    assert not edge['candidates']['include_unmatched']
    cfg['data'].update(batch_size=8, validation_batch_size=1, train_augmentation=True)
    cfg['runtime']['workers'] = 0
    cfg['experiment']['seed'] = SEED
    cfg['tracking']['enabled'] = False
    return cfg


def seeded(number):
    random.seed(SEED+number); np.random.seed(SEED+number); torch.manual_seed(SEED+number)


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def setup(root, arm):
    from scripts.diagnose_ixi_queries import load
    from data.loaders import build_data_loaders
    from scripts.train_ixi_transfer import TransferCriterion
    from models.matcher import build_matcher
    plan = json.loads((root/'plan.json').read_text())
    model, cfg, _ = load('local_image')
    cfg = configure(cfg, arm)
    assert sha(plan['checkpoint']) == plan['checkpoint_sha256']
    assert sha(plan['refinement']) == plan['refinement_sha256']
    model.requires_grad_(True).train()
    assert cfg['loss']['weights']['spline'] == 0., 'Preserve local-refinement checkpoint geometry policy'
    model.spline_head.requires_grad_(False)
    criterion = TransferCriterion(cfg, build_matcher(cfg), model.relation_embed,
        spline_head=model.spline_head, route='graph_only').cuda()
    criterion.set_training_progress(1, 0.)
    train, _ = build_data_loaders(cfg)
    assert len(train.dataset.datasets) == 1
    dataset = copy.copy(train.dataset.datasets[0]); lookup = {r.sample_id: r for r in dataset.records}
    dataset.records = [lookup[n] for n in plan['ids']['train']]
    assert dataset.augment
    settings = cfg['training']['optimizer']
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=7e-5, betas=tuple(settings['betas']),
        weight_decay=settings['weight_decay'], eps=settings['epsilon'])
    return model, cfg, criterion, opt, dataset, train.collate_fn


def order_for(updates, count):
    return np.random.default_rng(SEED).integers(count, size=(updates, 8))


def update(model, cfg, criterion, opt, dataset, collate, indices, number, diagnostic=False):
    from training.engine import _move_graph_batch
    seeded(number)
    batch = collate([dataset[int(i)] for i in indices])
    volumes, targets = _move_graph_batch(batch, 'cuda', cfg['training']['input'], True)
    model.train(); opt.zero_grad(set_to_none=True)
    tokens, predictions, _ = model(volumes)
    losses = criterion(tokens, predictions, targets)
    assert torch.isfinite(losses['total'])
    diagnostic_groups = {}
    if diagnostic:
        # Classification alone must reach the classifier AND shared representations.
        groups = ['class_embed.', 'decoder.', 'encoder.']
        selected = [(n, p) for n, p in model.named_parameters() if n.startswith(tuple(groups))]
        gradients = torch.autograd.grad(losses['class'], [p for _, p in selected], retain_graph=True, allow_unused=True)
        for group in groups:
            value = sum(float(g.detach().abs().sum()) for (n, p), g in zip(selected, gradients) if n.startswith(group) and g is not None)
            assert np.isfinite(value) and value > 0, ('Node classification gradient missing', group)
            diagnostic_groups[group] = value
    losses['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step()
    return dict(step=number, **{k: float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel() == 1},
                classification_gradient_l1=diagnostic_groups)


def run_smoke_or_benchmark(root, smoke):
    directory = root/('smoke' if smoke else 'benchmark'); directory.mkdir(parents=True, exist_ok=False)
    timings = {}; identities = []; parameters = []
    for arm in ARMS:
        model, cfg, criterion, opt, dataset, collate = setup(root, arm)
        identities.append(state_digest(model)); parameters.append([n for n,p in model.named_parameters() if p.requires_grad])
        steps = 2 if smoke else 35; order = order_for(steps, len(dataset))
        history = []; torch.cuda.reset_peak_memory_stats(); started = None
        for number, indices in enumerate(order, 1):
            if number == 4:
                torch.cuda.synchronize(); started = time.monotonic()
            row = update(model, cfg, criterion, opt, dataset, collate, indices, number, diagnostic=number == 1)
            history.append(row)
        torch.cuda.synchronize()
        timings[arm] = dict(seconds_per_update=(time.monotonic()-started)/32 if started else None,
            parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), history=history)
        assert state_digest(model) != identities[-1], 'Model did not learn'
        save(directory/(arm+'-config.json'), cfg)
        if smoke:
            atomic_checkpoint(directory/(arm+'.pt'), model, cfg, opt, steps)
        del model, criterion, opt; torch.cuda.empty_cache()
    assert len(set(identities)) == 1 and all(x == parameters[0] for x in parameters)
    save(directory/'summary.json', dict(completed=True, smoke=smoke, timings=timings,
        initial_model_sha256=identities[0], trainable_names=parameters[0]))
    if not smoke:
        slowest = max(x['seconds_per_update'] for x in timings.values())
        updates = min(20000, int(3300/(1.2*slowest))//100*100)
        assert updates >= 200, 'Insufficient full-network learning-curve budget'
        checkpoints = sorted(set([x for x in [250, 500, 1000, 2000, updates] if x <= updates]))
        save(directory/'budget.json', dict(completed=True, final_updates=updates, checkpoints=checkpoints,
             target_training_seconds=3300, safety_factor=1.2, slowest_seconds_per_update=slowest))


def atomic_checkpoint(path, model, cfg, opt, number):
    temporary = path.with_suffix('.partial.pt')
    torch.save(dict(net=model.state_dict(), config=cfg, optimizer=opt.state_dict(), steps=number), temporary)
    os.replace(temporary, path)


def train(root, arm):
    budget = json.loads((root/'benchmark/budget.json').read_text()); assert budget['completed']
    model, cfg, criterion, opt, dataset, collate = setup(root, arm)
    directory = root/'runs'/arm; directory.mkdir(parents=True, exist_ok=False)
    save(directory/'resolved-config.json', cfg)
    initial = state_digest(model); order = order_for(budget['final_updates'], len(dataset))
    history = []; started = time.monotonic(); stopped = [False]
    for sig in [signal.SIGTERM, signal.SIGUSR1]: signal.signal(sig, lambda *unused: stopped.__setitem__(0, True))
    for number, indices in enumerate(order, 1):
        row = update(model, cfg, criterion, opt, dataset, collate, indices, number, diagnostic=number == 1)
        if number == 1 or number % 50 == 0 or number == len(order):
            history.append(dict(elapsed_seconds=time.monotonic()-started, **row)); save(directory/'history.json', history)
            print(arm, history[-1], flush=True)
        if number in budget['checkpoints']: atomic_checkpoint(directory/f'model-{number:06d}.pt', model, cfg, opt, number)
        if number % 250 == 0: atomic_checkpoint(directory/'latest.pt', model, cfg, opt, number)
        if stopped[0] or time.monotonic()-started > 4800:
            atomic_checkpoint(directory/'interrupted.pt', model, cfg, opt, number)
            raise RuntimeError('Interrupted training; saved checkpoint; dependent evaluation blocked')
    save(directory/'summary.json', dict(completed=True, arm=arm, updates=len(order), checkpoints=budget['checkpoints'],
        initial_model_sha256=initial, final_model_sha256=state_digest(model),
        trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
        batch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(), elapsed_seconds=time.monotonic()-started,
        scheduler='constant 7e-5; optimizer/scheduler restarted equally across arms', augmentation=True))


def predict(root, arm, smoke=False):
    from data.loaders import build_data_loaders, build_evaluation_loader
    from scripts.frozen_spline_diagnostic import training_evaluation_loader
    from scripts.diagnose_ixi_queries import load, cache_patch
    from models.checkpoint import load_checkpoint_file
    plan = json.loads((root/'plan.json').read_text())
    if smoke:
        checkpoints = [2]; source = root/'smoke'
    else:
        summary = json.loads((root/'runs'/arm/'summary.json').read_text()); assert summary['completed']
        checkpoints = summary['checkpoints']; source = root/'runs'/arm
    # Inference must be repeated because these experiments update the node predictor.
    model, _, _ = load('local_image')
    for number in checkpoints:
        cp = load_checkpoint_file(source/(arm+'.pt' if smoke else f'model-{number:06d}.pt'))
        model.load_state_dict(cp['net'], strict=True); cfg = copy.deepcopy(cp['config']); del cp
        cfg['data'].update(train_augmentation=False, batch_size=1, validation_batch_size=1)
        model.eval().requires_grad_(False)
        output = root/('smoke-predictions' if smoke else 'predictions')/arm/f'{number:06d}'
        output.mkdir(parents=True, exist_ok=False); records = []
        train_loader, _ = build_data_loaders(cfg)
        for split, names in plan['evaluation_ids'].items():
            if smoke: names = names[:2]
            loader = training_evaluation_loader(train_loader, cfg, names) if split == 'train' else build_evaluation_loader(cfg, dataset_name='synthetic_mri', sample_ids=names)
            for index, batch in enumerate(loader):
                name = names[index]; c = cache_patch(model, cfg, batch)
                gt_path = Path(plan['dataset'])/split/'spline'/(name+'.npz')
                with np.load(gt_path) as gt: c.update(gt_nodes=gt['nodes'], gt_edges=gt['edges'], metric=gt['metric'])
                path = output/(name+'.npz'); np.savez_compressed(path, **c)
                records.append(dict(split=split, sample_id=name, subject=name.rsplit('_',1)[0], site=plan['sites'][name], cache=str(path), gt=str(gt_path)))
                if (index+1) % 64 == 0: print(arm, number, split, index+1, '/', len(names), flush=True)
        save(output/'records.json', records); save(output/'summary.json', dict(completed=True, arm=arm, step=number, batching=1, records=len(records)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['smoke','benchmark','train','predict'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--arm', choices=ARMS); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); torch.set_num_threads(1)
    if a.mode in ['smoke','benchmark']: run_smoke_or_benchmark(a.root, a.mode == 'smoke')
    elif a.mode == 'train': train(a.root, a.arm)
    else: predict(a.root, a.arm, a.smoke)
