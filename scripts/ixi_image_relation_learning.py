"""Shared frozen-feature preparation and residual-adapter training for IXI."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.image_relation_adapter import ARMS, ImageRelationAdapter, residual_logits
from models.matcher import build_matcher
from scripts.ixi_cached_mechanisms import CACHES, manifest, sha, save, table

SEED = 201020
PREVIOUS = Path(json.loads(Path('experiments/ixi_mechanisms/submission.json').read_text())['output_root'])
DATASET = Path(manifest()['dataset'])


def pair_indices(pairs, queries=256):
    a = pairs.amin(-1); b = pairs.amax(-1)
    return a*(2*queries-a-1)//2+b-a-1


def read_plan(root):
    return json.loads((root/'plan.json').read_text())


def prepare(root, smoke=False):
    from data.loaders import build_data_loaders, build_evaluation_loader
    from scripts.frozen_spline_diagnostic import training_evaluation_loader
    from scripts.diagnose_ixi_queries import load, cache_patch
    from scripts.controlled_spline_unfreeze import frozen_state_digest
    plan = read_plan(root); out = root/('smoke-cache' if smoke else 'cache')
    out.mkdir(parents=True, exist_ok=False)
    model, cfg, step = load('local_image'); frozen = frozen_state_digest(model)
    assert cfg['training']['input'] == 'image', 'GT masks must never be predictor inputs'
    assert sha(DATASET/'manifest.json') == plan['dataset_manifest_sha256']
    matcher = build_matcher(cfg)
    assert cfg['loss']['edge']['classification']['name'] == 'focal'
    assert cfg['loss']['edge']['balancing']['mode'] == 'none'
    assert not cfg['loss']['edge']['candidates']['include_unmatched']
    assert not cfg['loss']['edge']['classification']['curriculum']['enabled']
    assert cfg['loss']['edge']['candidates']['max_per_graph'] == 9999
    assert cfg['loss']['edge']['candidates']['positive_cap'] is None
    assert sha(manifest()['checkpoint']) == plan['checkpoint_sha256']
    replay = json.loads((PREVIOUS/'features/summary.json').read_text())
    assert replay['completed'] and len(replay['checks']) == 384
    assert max(r['spline_max_error_mm'] for r in replay['checks']) <= .001
    prior_inputs = Path(json.loads(Path('experiments/ixi_relation_matching_calibration/submission.json').read_text())['output_root'])/'prepared/inputs.json'
    checked = json.loads(prior_inputs.read_text())
    # Revalidate previous full 384-patch feature/cache/GT replay evidence without
    # another CNN pass; saved hashes establish that those exact files are reused.
    for item in checked:
        for file, expected in item['sources'].items():
            if '/features/' in file or '/pilot-1291329-' in file or '/spline/' in file:
                assert sha(file) == expected, ('Changed pilot input', file)
    records, old_replay = [], []
    ids = {s: names[:2]+[next(n for n in names if n not in manifest()['ids'][s])] if smoke else names for s, names in plan['ids'].items()}
    full, _ = build_data_loaders(cfg)
    captured = {}
    hook = model.encoder.layer0.relu1.register_forward_hook(lambda module, inputs, value: captured.update(early=value.detach()))
    for split, names in ids.items():
        directory = out/split; directory.mkdir()
        reused_smoke = root/'smoke-cache'/split
        novel = [n for n in names if not (CACHES['local_image']/split/(n+'.npz')).exists()
                 and not (not smoke and (reused_smoke/(n+'.npz')).exists())]
        loader = (training_evaluation_loader(full, cfg, novel) if split == 'train' else
                  build_evaluation_loader(cfg, dataset_name='synthetic_mri', sample_ids=novel)) if novel else []
        for index, batch in enumerate(loader):
            name = loader.dataset.records[index].sample_id
            c = cache_patch(model, cfg, batch)
            with np.load(DATASET/split/'spline'/(name+'.npz')) as gt:
                c.update(gt_nodes=gt['nodes'], gt_edges=gt['edges'], metric=gt['metric'])
            np.savez_compressed(directory/(name+'.npz'), **c)
            np.save(directory/(name+'-features.npy'), captured['early'][0].cpu().numpy())
            if (index+1) % 32 == 0:
                print('extracted', split, index+1, '/', len(novel), flush=True)
        for name in names:
            old = CACHES['local_image']/split/(name+'.npz')
            reused = not smoke and (reused_smoke/(name+'.npz')).exists()
            cache = old if old.exists() else (reused_smoke if reused else directory)/(name+'.npz')
            feature = PREVIOUS/'features'/split/(name+'.npy') if old.exists() else (reused_smoke if reused else directory)/(name+'-features.npy')
            with np.load(cache) as f:
                c = {k: f[k] for k in f.files}
            np.testing.assert_array_equal(c['edges'], torch.combinations(torch.arange(256)).numpy())
            np.testing.assert_array_equal(c['query_ids'], np.arange(256))
            with np.load(DATASET/split/'spline'/(name+'.npz')) as gt:
                np.testing.assert_array_equal(gt['nodes'], c['gt_nodes']); np.testing.assert_array_equal(gt['edges'], c['gt_edges'])
                assert len(gt['nodes']) <= 256
                assert len({tuple(sorted(e)) for e in gt['edges']}) == len(gt['edges'])
                assert not (gt['edges'][:, 0] == gt['edges'][:, 1]).any()
            if old.exists():
                # Small heads only: reproduce all old relation logits and current
                # refinement outputs using the already validated early feature maps.
                with torch.no_grad():
                    tokens = torch.tensor(c['tokens'], device='cuda')[None]
                    nodes = model.coord_embed(tokens[:, :256]).sigmoid()
                    logits = model.class_embed(tokens[:, :256])
                    xyz, cls = model.node_refinement(tokens[:, :256], nodes, logits,
                        torch.tensor(np.load(feature), device='cuda')[None], tuple(c['shape']))
                    displacement = (xyz[0, :, :3].cpu().numpy()-c['nodes']) @ c['metric'].T
                    error = float(np.linalg.norm(displacement, axis=-1).max(initial=0))
                    probability_error = float(np.abs(cls[0].softmax(-1).cpu().numpy()[:, 1]-c['node_scores']).max(initial=0))
                    assert error < .003 and probability_error < .0002, (name, error, probability_error)
                    np.testing.assert_array_equal(np.flatnonzero(cls[0].argmax(-1).cpu().numpy() == 1), c['retained'])
                    qp = torch.tensor(c['edges'], device='cuda'); tok = tokens[0]
                    shared = tok[256:].flatten()[None].expand(len(qp), -1)
                    a = torch.cat([tok[qp[:, 0]], tok[qp[:, 1]], shared], -1)
                    b = torch.cat([tok[qp[:, 1]], tok[qp[:, 0]], shared], -1)
                    result = (.5*(model.relation_embed(a)+model.relation_embed(b))).cpu().numpy()
                    np.testing.assert_allclose(result, c['pair_logits'], atol=3e-6, rtol=3e-6)
                    np.testing.assert_array_equal(result.argmax(-1), c['pair_logits'].argmax(-1))
                    old_replay.append(dict(split=split, sample_id=name, coordinate_error_mm=error, confidence_error=probability_error))
            source, target = matcher({'pred_nodes': torch.tensor(c['boxes'])[None], 'pred_logits': torch.tensor(c['node_logits'])[None]},
                                     {'nodes': [torch.tensor(c['gt_nodes'])]})[0]
            records.append(dict(split=split, sample_id=name, subject=name.rsplit('_', 1)[0],
                site=plan['sites'][name], cache=str(cache), features=str(feature), cache_sha256=sha(cache),
                features_sha256=sha(feature), gt=str(DATASET/split/'spline'/(name+'.npz')),
                gt_sha256=sha(DATASET/split/'spline'/(name+'.npz')), matched_queries=source.tolist(), matched_gt=target.tolist()))
    hook.remove(); assert frozen_state_digest(model) == frozen
    # Shared labels and offsets on every normally retained pair. GT attributes
    # below are supervision/evaluation metadata, never adapter inputs.
    from scripts.diagnose_branch_relations import match_nodes
    from scripts.ixi_relation_matching_calibration import pair_labels, mapping_array
    chunks = {}; offsets = [0]; training_arcs = []; training_chords = []
    for record in records:
        with np.load(record['cache']) as c, np.load(record['gt']) as gt:
            q = c['retained']; pairs = torch.combinations(torch.tensor(q), r=2).numpy().reshape(-1, 2)
            ids_pair = pair_indices(torch.tensor(pairs)).numpy()
            logits = c['pair_logits'][ids_pair]
            values = dict(original_margin=logits[:, 1].astype(float)-logits[:, 0].astype(float),
                endpoint_confidence=c['node_scores'][pairs].min(1) if len(pairs) else np.empty(0))
            metric = gt['metric']; truth = gt['nodes'] @ metric.T
            arcs = [float(np.linalg.norm(np.diff(gt['full_paths'][a:b] @ metric.T, axis=0), axis=1).sum())
                    for a, b in zip(gt['path_offsets'][:-1], gt['path_offsets'][1:])]
            chords = [float(np.linalg.norm(truth[a]-truth[b])) for a, b in gt['edges']]
            record['gt_arc_lengths'] = arcs; record['gt_endpoint_distances'] = chords
            if record['split'] == 'train': training_arcs.extend(arcs); training_chords.extend(chords)
            lookup = {tuple(sorted(e)): k for k, e in enumerate(gt['edges'])}
            for tol in (1, 2):
                mapping = {int(q[i]): j for i, j in match_nodes(c['nodes'][q] @ metric.T, truth, tol).items()}
                record[f'mapping_{tol}'] = {str(k): int(v) for k, v in mapping.items()}
                _, mm, y = pair_labels(pairs, mapping_array(mapping), gt['edges'], q)
                near = np.zeros(len(pairs), bool); arc = np.full(len(pairs), np.nan); chord = np.full(len(pairs), np.nan)
                for i in np.flatnonzero(mm):
                    a, b = (mapping[int(k)] for k in pairs[i]); chord[i] = np.linalg.norm(truth[a]-truth[b])
                    near[i] = not y[i] and chord[i] <= 5.
                    if y[i]: arc[i] = arcs[lookup[tuple(sorted((a, b)))]]
                values.update({f'labels_{tol}': y, f'conditional_{tol}': mm, f'near_negative_{tol}': near,
                               f'arc_{tol}': arc, f'chord_{tol}': chord})
            for key, value in values.items(): chunks.setdefault(key, []).append(value)
            offsets.append(offsets[-1]+len(pairs))
    np.savez_compressed(out/'common-pairs.npz', **{k: np.concatenate(v) for k, v in chunks.items()}, offsets=np.array(offsets))
    save(out/'strata.json', dict(training_arc_quartiles_mm=np.quantile(training_arcs, [.25, .5, .75]).tolist(),
                                training_chord_quartiles_mm=np.quantile(training_chords, [.25, .5, .75]).tolist()))
    save(out/'records.json', records); save(out/'replay.json', old_replay)
    save(out/'config.json', cfg)
    sizes = {str(p): p.stat().st_size for r in records for p in [Path(r['cache']), Path(r['features'])]}
    save(out/'summary.json', dict(completed=True, ids=ids, records=len(records), logical_bytes=sum(sizes.values()),
        new_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()), frozen_state_unchanged=True,
        checkpoint_sha256=plan['checkpoint_sha256'], adapter_sha256=manifest()['adapter_sha256'], batching=1,
        pilot_replay_cases=len(old_replay), previous_spline_replay_cases=384, parent_step=step))


class FrozenData:
    def __init__(self, root, smoke=False):
        self.root = root; self.directory = root/('smoke-cache' if smoke else 'cache')
        assert json.loads((self.directory/'summary.json').read_text())['completed']
        self.records = json.loads((self.directory/'records.json').read_text())
        self.config = json.loads((self.directory/'config.json').read_text())
        self.items = []
        free, total = torch.cuda.mem_get_info()
        estimate = len(self.records)*(64*32**3*4+258*552*4+32640*2*4)
        assert estimate < free*.7, f'Frozen GPU cache estimate {estimate} exceeds memory allowance {free*.7}'
        for i, record in enumerate(self.records):
            with np.load(record['cache']) as c:
                item = {k: torch.tensor(c[k], device='cuda') for k in ['tokens', 'nodes', 'node_scores', 'pair_logits', 'metric']}
                item.update(retained=c['retained'].copy(), gt_edges=torch.tensor(c['gt_edges'], device='cuda', dtype=torch.long))
            item['features'] = torch.tensor(np.load(record['features']), device='cuda')
            item['assignment'] = (torch.tensor(record['matched_queries'], device='cuda', dtype=torch.long), torch.tensor(record['matched_gt'], device='cuda', dtype=torch.long))
            assert all(not x.requires_grad for x in item.values() if torch.is_tensor(x))
            self.items.append(item)
        self.train = [i for i, r in enumerate(self.records) if r['split'] == 'train']
        self.frozen_digest = self.digest()

    def digest(self):
        # Inputs are never optimizer parameters. Compact tensors plus deterministic
        # early-map samples detect unintended in-place changes without 24GB copies.
        h = hashlib.sha256()
        for item in self.items:
            for k in ['tokens', 'nodes', 'node_scores', 'pair_logits', 'metric']:
                h.update(item[k].detach().cpu().numpy().tobytes())
            h.update(item['features'].flatten()[::4096].cpu().numpy().tobytes())
        return h.hexdigest()


def criterion(data):
    from training.losses.criterion import GraphCriterion
    cfg = data.config
    assert cfg['loss']['edge']['classification']['name'] == 'focal'
    assert cfg['loss']['edge']['balancing']['mode'] == 'none'
    assert not cfg['loss']['edge']['candidates']['include_unmatched']
    # Only the established pair sampler and focal implementation are used.
    return GraphCriterion(cfg, build_matcher(cfg), torch.nn.Identity()).cuda()


def new_adapter(arm, device='cuda'):
    torch.manual_seed(SEED)
    return ImageRelationAdapter(arm=arm).to(device)


def step(adapter, data, indices, loss_module, optimizer, number):
    torch.manual_seed(SEED+number)
    assignments = [data.items[i]['assignment'] for i in indices]
    # Sampling depends only on query IDs and graph structure; width-one placeholder
    # tokens avoid materializing unused full pair embeddings in the old sampler.
    contexts = []
    _, labels, _ = loss_module._matched_edge_pool(torch.zeros(len(indices), 258, 1, device='cuda'),
        [data.items[i]['gt_edges'] for i in indices], assignments, contexts)
    count = sum(len(y) for y in labels)
    optimizer.zero_grad(set_to_none=True); loss_value = 0.; positives = 0
    if not count:
        sum(p.sum()*0 for p in adapter.parameters()).backward(); optimizer.step()
        return dict(loss=0., candidates=0, positives=0)
    for (batch, pairs), y in zip(contexts, labels):
        item = data.items[indices[batch]]; positives += int(y.sum())
        for start in range(0, len(pairs), 512):
            p = pairs[start:start+512]; target = y[start:start+512]
            residual = adapter(item['tokens'], item['nodes'], item['node_scores'], item['metric'], item['features'], p)
            logits = residual_logits(item['pair_logits'][pair_indices(p)], residual)
            focal = loss_module._classification_loss(logits, target, loss_module.edge_config)
            loss = focal*(len(p)/count)*loss_module.weights['edges']
            assert torch.isfinite(loss)
            loss.backward(); loss_value += float(loss.detach())
    assert all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.network[-1].parameters())
    optimizer.step()
    return dict(loss=loss_value, candidates=count, positives=positives)


def optimizer(adapter, cfg):
    settings = cfg['training']['optimizer']
    return torch.optim.AdamW(adapter.parameters(), lr=7e-5, weight_decay=settings['weight_decay'],
                             eps=settings['epsilon'], betas=tuple(settings['betas']))


def evaluate(adapter, data, output):
    scores, offsets, baseline = [], [0], []
    adapter.eval()
    with torch.no_grad():
        for item in data.items:
            pairs = torch.combinations(torch.tensor(item['retained'], device='cuda'), r=2)
            result, original = [], []
            for p in pairs.split(512):
                if not len(p):
                    continue
                delta = adapter(item['tokens'], item['nodes'], item['node_scores'], item['metric'], item['features'], p)
                logits = item['pair_logits'][pair_indices(p)]
                original.append((logits[:, 1].double()-logits[:, 0].double()).cpu().numpy())
                result.append((logits[:, 1].double()-logits[:, 0].double()+delta.double()).cpu().numpy())
            margin = np.concatenate(result) if result else np.empty(0)
            base = np.concatenate(original) if original else np.empty(0)
            scores.append(margin); baseline.append(base); offsets.append(offsets[-1]+len(margin))
    np.savez_compressed(output, margin=np.concatenate(scores), original_margin=np.concatenate(baseline), offsets=np.array(offsets),
                        names=np.array([r['sample_id'] for r in data.records]), splits=np.array([r['split'] for r in data.records]))
    adapter.train()


def benchmark(root):
    plan = read_plan(root); data = FrozenData(root)
    output = root/'benchmark'; output.mkdir(exist_ok=False)
    loss_module = criterion(data); timings = {}; smoke = {}
    rng = np.random.default_rng(SEED)
    order = rng.choice(data.train, size=(208, plan['batch_size']), replace=True)
    for arm in ARMS:
        adapter = new_adapter(arm); opt = optimizer(adapter, data.config)
        torch.cuda.reset_peak_memory_stats()
        item = data.items[data.train[0]]; p = torch.tensor([[0, 1], [2, 3]], device='cuda')
        with torch.no_grad():
            delta = adapter(item['tokens'], item['nodes'], item['node_scores'], item['metric'], item['features'], p)
            assert torch.equal(delta, torch.zeros_like(delta))
            base = item['pair_logits'][pair_indices(p)]
            assert torch.equal(residual_logits(base, delta), base)
        last = None
        for i, batch in enumerate(order[:8], 1):
            last = step(adapter, data, batch.tolist(), loss_module, opt, i)
        torch.cuda.synchronize(); started = time.monotonic()
        for i, batch in enumerate(order[8:], 9):
            last = step(adapter, data, batch.tolist(), loss_module, opt, i)
        torch.cuda.synchronize(); elapsed = time.monotonic()-started
        timings[arm] = dict(seconds_per_update=elapsed/200, updates=200, elapsed_seconds=elapsed,
            parameters=sum(p.numel() for p in adapter.parameters()), last=last,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        # Saved tiny benchmark states are not used as training initializers.
        smoke[arm] = dict(finite_gradients=True, zero_initialized=True)
        del adapter, opt
    assert data.digest() == data.frozen_digest
    assert len({v['parameters'] for v in timings.values()}) == 1
    slowest = max(x['seconds_per_update'] for x in timings.values())
    updates = min(100000, int(3300/(slowest*1.15))//100*100)
    assert updates >= 2000, 'Throughput too low: simplify adapter before launching the larger experiment'
    save(output/'budget.json', dict(completed=True, final_updates=updates, checkpoints=[1000, 2000, updates],
        target_training_seconds=3300, safety_factor=1.15, timings=timings, smoke=smoke,
        frozen_cache_unchanged=True, batch_size=plan['batch_size'], seed=SEED))


def smoke(root):
    prepare(root, True); data = FrozenData(root, True); loss_module = criterion(data)
    order = np.random.default_rng(SEED).choice(data.train, size=(3, 2), replace=True)
    for arm in ARMS:
        out = root/'smoke-runs'/arm; out.mkdir(parents=True, exist_ok=False)
        adapter = new_adapter(arm); opt = optimizer(adapter, data.config)
        item = data.items[0]; pairs = torch.tensor([[0, 1], [2, 3]], device='cuda')
        with torch.no_grad():
            zero = adapter(item['tokens'], item['nodes'], item['node_scores'], item['metric'], item['features'], pairs)
            assert torch.equal(zero, torch.zeros_like(zero))
            original = item['pair_logits'][pair_indices(pairs)]
            assert torch.equal(original, residual_logits(original, zero))
        history = [step(adapter, data, batch.tolist(), loss_module, opt, i) for i, batch in enumerate(order, 1)]
        with torch.no_grad():
            a = adapter(item['tokens'], item['nodes'], item['node_scores'], item['metric'], item['features'], pairs)
            b = adapter(item['tokens'], item['nodes'], item['node_scores'], item['metric'], item['features'], pairs.flip(1))
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        evaluate(adapter, data, out/'scores-000003.npz')
        save(out/'history.json', history)
        save(out/'summary.json', dict(completed=True, smoke=True, arm=arm, updates=3, checkpoints=[3],
             batch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(), frozen_inputs_unchanged=True))
    assert data.digest() == data.frozen_digest


def train(root, arm):
    plan = read_plan(root); budget = json.loads((root/'benchmark/budget.json').read_text()); assert budget['completed']
    data = FrozenData(root); adapter = new_adapter(arm); opt = optimizer(adapter, data.config); loss_module = criterion(data)
    out = root/'runs'/arm; out.mkdir(parents=True, exist_ok=False)
    updates = budget['final_updates']; checkpoints = sorted(set(budget['checkpoints']))
    order = np.random.default_rng(SEED).choice(data.train, size=(updates, plan['batch_size']), replace=True)
    order_hash = hashlib.sha256(order.tobytes()).hexdigest(); history = []; started = time.monotonic(); training_seconds = 0.; scoring_seconds = 0.
    interrupted = [False]
    for sig in [signal.SIGTERM, signal.SIGUSR1]:
        signal.signal(sig, lambda signum, frame: interrupted.__setitem__(0, True))
    def checkpoint(path, number):
        temporary = path.with_suffix('.partial.pt')
        torch.save(dict(adapter={k: v.detach().cpu() for k, v in adapter.state_dict().items()}, optimizer=opt.state_dict(),
            step=number, arm=arm, final_updates=updates, batch_order_sha256=order_hash, plan=plan, budget=budget), temporary)
        os.replace(temporary, path)
    for number, indices in enumerate(order, 1):
        before = time.monotonic(); values = step(adapter, data, indices.tolist(), loss_module, opt, number)
        training_seconds += time.monotonic()-before
        if number == 1 or number % 100 == 0:
            history.append(dict(step=number, elapsed_seconds=time.monotonic()-started, training_seconds=training_seconds, scoring_seconds=scoring_seconds, **values))
            save(out/'history.json', history); print(arm, history[-1], flush=True)
        if number % 500 == 0:
            checkpoint(out/'latest.pt', number)
        if number in checkpoints:
            checkpoint(out/f'adapter-{number:06d}.pt', number)
            scoring_started = time.monotonic()
            evaluate(adapter, data, out/f'scores-{number:06d}.npz')
            scoring_seconds += time.monotonic()-scoring_started
        if interrupted[0] or time.monotonic()-started > 4800:
            checkpoint(out/'interrupted.pt', number)
            raise RuntimeError('Training incomplete; saved recovery checkpoint and blocked dependent evaluation')
    assert data.digest() == data.frozen_digest
    save(out/'summary.json', dict(completed=True, arm=arm, updates=updates, checkpoints=checkpoints,
        elapsed_seconds=time.monotonic()-started, training_seconds=training_seconds, scoring_seconds=scoring_seconds, batch_order_sha256=order_hash, frozen_inputs_unchanged=True,
        parameters=sum(p.numel() for p in adapter.parameters()), plan=plan, budget=budget))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['prepare', 'benchmark', 'train', 'smoke'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--arm', choices=ARMS); p.add_argument('--smoke', action='store_true')
    args = p.parse_args(); torch.set_num_threads(1)
    if args.mode == 'smoke': smoke(args.root)
    elif args.mode == 'prepare': prepare(args.root, args.smoke)
    elif args.mode == 'benchmark': benchmark(args.root)
    else: train(args.root, args.arm)
