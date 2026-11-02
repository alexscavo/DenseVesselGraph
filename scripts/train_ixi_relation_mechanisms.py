#!/usr/bin/env python3
"""Short relation-only updates on frozen IXI tokens, with separately normalized auxiliaries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from models.matcher import build_matcher
from training.losses.criterion import GraphCriterion, _ratio_upsample
from training.losses.focal import select_active_unmatched_queries, build_unmatched_relation_pairs
from scripts.ixi_cached_mechanisms import SEED, identities, heads, manifest, read_cache, save, provenance

ARMS = ['focal_control', 'regularized', 'hard_unmatched', 'hard_matched']


def symmetric(head, tokens, pairs, relations):
    def features(p):
        return torch.cat([tokens[p[:, 0]], tokens[p[:, 1]], tokens[256:256+relations].flatten()[None].expand(len(p), -1)], -1)
    return .5*(head(features(pairs)) + head(features(pairs[:, [1, 0]])))


def auxiliary_pairs(criterion, cache, assignment, kind):
    source, truth = assignment
    if kind == 'hard_unmatched':
        _, _, active, _ = select_active_unmatched_queries(cache['logits'], source, .5, 32)
        pairs = build_unmatched_relation_pairs(source, active)
    elif kind == 'hard_matched':
        pairs = torch.combinations(source, r=2)
        positive = criterion._local_true_edges(cache['edges'], truth, source.device)
        true_pairs = {tuple(sorted(e)) for e in source[positive].tolist()}
        keep = torch.tensor([tuple(sorted(e)) not in true_pairs for e in pairs.tolist()], dtype=torch.bool, device=source.device)
        pairs = pairs[keep]
    else:
        pairs = source.new_empty((0, 2))
    return pairs


def fit(args):
    ids = identities(args.smoke); cfg = copy.deepcopy(manifest()['config'])
    assert cfg['model']['matcher']['type'] == 'hungarian'
    assert cfg['loss']['edge']['classification']['name'] == 'focal'
    assert cfg['loss']['edge']['balancing']['mode'] == 'none'
    assert not cfg['loss']['edge']['candidates']['include_unmatched']
    assert not cfg['loss']['degree']['enabled'] and cfg['loss']['edge']['candidates']['positive_cap'] is None
    if args.arm != 'focal_control':
        cfg['loss']['edge']['classification']['name'] = 'cross_entropy'
        cfg['loss']['edge']['balancing']['mode'] = 'ratio_upsample'
    head, _ = heads(args.device); head.requires_grad_(True).eval()
    initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    matcher = build_matcher(cfg)
    criterion = GraphCriterion(cfg, matcher, head).to(args.device)
    data, assignments, pools, audit = [], [], [], []
    relations = cfg['model']['decoder']['relation_tokens']
    for name in ids['train']:
        c = read_cache('local_image', 'train', name)
        item = dict(tokens=torch.tensor(c['tokens'], device=args.device),
                    logits=torch.tensor(c['node_logits'], device=args.device),
                    edges=torch.tensor(c['gt_edges'], dtype=torch.long, device=args.device))
        assignment = matcher({'pred_nodes': torch.tensor(c['boxes'])[None], 'pred_logits': torch.tensor(c['node_logits'])[None]},
                             {'nodes': [torch.tensor(c['gt_nodes'])]})[0]
        assignment = tuple(v.to(args.device) for v in assignment)
        data.append(item); assignments.append(assignment)
        pools.append(auxiliary_pairs(criterion, item, assignment, args.arm))
        matched = len(assignment[0]); positive = len(criterion._local_true_edges(item['edges'], assignment[1], args.device))
        audit.append(dict(sample_id=name, matched=matched, positive_edges=positive, all_matched_pairs=matched*(matched-1)//2,
                          candidate_cap=criterion.candidates['max_per_graph'], auxiliary_candidates=len(pools[-1])))
    settings = cfg['training']['optimizer']
    optimizer = torch.optim.AdamW(head.parameters(), lr=7e-5, weight_decay=settings['weight_decay'],
                                 eps=settings['epsilon'], betas=tuple(settings['betas']))
    updates = 3 if args.smoke else 1000; batch_size = 2 if args.smoke else 32
    order = np.random.default_rng(SEED).integers(len(data), size=(updates, batch_size))
    history = []; batch_digest = hashlib.sha256(order.tobytes()).hexdigest()
    # Identical baseline sampling/orientation RNG across arms; auxiliaries consume no RNG.
    for step, indices in enumerate(order, 1):
        torch.manual_seed(SEED+step)
        tokens = torch.stack([data[i]['tokens'] for i in indices])
        logits = torch.stack([data[i]['logits'] for i in indices])
        target_edges = [data[i]['edges'] for i in indices]
        selected_assignments = [assignments[i] for i in indices]
        optimizer.zero_grad(set_to_none=True)
        baseline, degree = criterion.loss_edges(tokens, logits, target_edges, selected_assignments)
        assert float(degree.detach()) == 0.
        aux_logits, counts = [], dict(MU=0, UU=0, MM=0)
        for i in indices:
            pool = pools[i]
            if not len(pool):
                continue
            with torch.no_grad():
                scores = symmetric(head, data[i]['tokens'], pool, relations).softmax(-1)[:, 1]
                keep = torch.argsort(scores, descending=True, stable=True)[:64]
            chosen = pool[keep]
            aux_logits.append(symmetric(head, data[i]['tokens'], chosen, relations))
            matched = set(assignments[i][0].tolist())
            for a, b in chosen.tolist():
                count = int(a in matched)+int(b in matched)
                counts[['UU', 'MU', 'MM'][count]] += 1
        aux = F.cross_entropy(torch.cat(aux_logits), torch.zeros(sum(len(v) for v in aux_logits), dtype=torch.long, device=args.device)) if aux_logits else baseline*0
        # Smoke exercises auxiliary gradients immediately; pilot warms up then ramps.
        weight = .1 if args.smoke else .1*min(1., max(0., (step-100)/200))
        total = cfg['loss']['weights']['edges']*(baseline+weight*aux)
        check = step == 1 or step % 100 == 0 or step == updates
        norms = {}
        if check:
            for label, loss in [('base', baseline), ('aux', aux)]:
                gradients = torch.autograd.grad(loss, tuple(head.parameters()), retain_graph=True)
                norms[label+'_gradient_norm'] = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
        assert torch.isfinite(total)
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        assert not tokens.requires_grad and not logits.requires_grad
        optimizer.step()
        if check:
            positive_count = sum(audit[i]['positive_edges'] for i in indices)
            negative_count = sum(min(audit[i]['all_matched_pairs']-audit[i]['positive_edges'],
                                     max(0, audit[i]['candidate_cap']-audit[i]['positive_edges'])) for i in indices)
            duplicated = 0
            if criterion.balancing['mode'] == 'ratio_upsample':
                labels = torch.tensor([1]*positive_count+[0]*negative_count)
                _, balanced_labels = _ratio_upsample(torch.zeros(len(labels), 2), labels,
                    criterion.balancing['positive_to_negative_ratio'], criterion.balancing['tolerance'])
                duplicated = len(balanced_labels)-len(labels)
            history.append(dict(step=step, baseline=float(baseline.detach()), auxiliary=float(aux.detach()),
                                auxiliary_weight=weight, total=float(total.detach()), base_positive_count=positive_count,
                                base_negative_count=negative_count, base_upsampled_copies=duplicated,
                                base_reduction_count=positive_count+negative_count+duplicated,
                                auxiliary_reduction_count=sum(counts.values()), **counts, **norms))
            print(args.arm, history[-1], flush=True)
    assert any(not torch.equal(v.cpu(), initial[k]) for k, v in head.state_dict().items())
    torch.save({'state_dict': {k: v.cpu() for k, v in head.state_dict().items()}, 'arm': args.arm}, args.output/'relation-head.pt')
    # Head-only scoring on cached tokens; no image-model forwards. CPU collectors decode curves.
    parity_errors = []
    original, _ = heads(args.device)
    with torch.no_grad():
        for split, names in ids.items():
            (args.output/split).mkdir()
            for name in names:
                c = read_cache('local_image', split, name)
                t = torch.tensor(c['tokens'], device=args.device); p = torch.tensor(c['edges'], device=args.device)
                reference = symmetric(original, t, p, relations).cpu().numpy()
                np.testing.assert_allclose(reference, c['pair_logits'], atol=3e-6, rtol=3e-6)
                np.testing.assert_array_equal(reference.argmax(-1), c['pair_logits'].argmax(-1))
                parity_errors.append(float(np.max(np.abs(reference-c['pair_logits']))))
                scores = symmetric(head, t, p, relations).cpu().numpy()
                np.savez_compressed(args.output/split/(name+'.npz'), pair_logits=scores)
    save(args.output/'history.json', history); save(args.output/'pair-populations.json', audit)
    save(args.output/'resolved-config.json', cfg)
    save(args.output/'summary.json', dict(completed=True, arm=args.arm, updates=updates, batch_size=batch_size,
        learning_rate=7e-5, edge_weight=cfg['loss']['weights']['edges'], optimizer_settings=settings,
        batch_order_sha256=batch_digest, initial_head_sha256=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest(),
        maximum_initial_logit_replay_error=max(parity_errors), **provenance(ids),
        auxiliary=dict(top_k=64, active_unmatched_cap=32, active_unmatched_threshold=.5, weight=.1, warmup_updates=100, ramp_updates=200),
        frozen='Only relation MLP instantiated for training; cached node outputs and tokens immutable.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--device', default='cuda')
    args = p.parse_args(); torch.set_num_threads(1); torch.manual_seed(SEED)
    args.output.mkdir(parents=True, exist_ok=False); fit(args)
