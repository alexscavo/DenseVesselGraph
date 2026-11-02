#!/usr/bin/env python3
"""Two-stage dense model vs RelationFormer B0 vs straight-edge RelationFormer (3D, validation).

Why the straight arm needs no training. In the B0 recipe the spline head cannot change the
detector: the `graph_only` route detaches tokens and node coordinates in the spline loss, the
local corrector samples a detached feature map, no update clips gradients, AdamW updates each
parameter separately, the spline head is built after every detector module, and the RNG is
reseeded every step. A straight-only RelationFormer trained with the same recipe therefore
decodes B0's graph (up to GPU nondeterminism) and draws each edge as its endpoint chord. Each
B0 graph is scored twice: with its K8 splines ("b0") and with chords ("straight").

B0 checkpoints (fixed final updates of the lineage used everywhere else):
- SyntheticMRI: local_refinement_long seed 92741 B0 model-12000 (the IXI/BRAVA source);
- IXI: ixi_long_20261030 epoch 100 (its saved predictions are reused);
- BRAVA: D08 synthetic arm, epoch 300 (82.2k updates).
Inference and decoding are ixi_long_experiment_20261030.snapshot, unchanged: the dense model's
multidataset_continuous populations (monitor_train for threshold selection, val_ids), fixed
0.5/0.5 and train-selected thresholds, strict > as infer_graphs. Only the object-query count
passed to infer_graphs follows the checkpoint (SyntheticMRI B0 has 120 queries).

Dense rows/thresholds: dense_graph_20261030/<dataset>/spline6/final-v2*. Per-patch counts are
pooled and bootstrapped over validation subjects (2000 resamples, seed 261030), with parity
checks against each arm's reported results. No test data is read.

Modes: infer <dataset> (GPU; SyntheticMRI, BRAVA) | score <dataset> (CPU) | collect.
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import ixi_long_experiment_20261030 as ix  # noqa: E402
from compare_dense_anchored_20261030 import KEYS, SCRATCH, counts, dense_patch, metrics  # noqa: E402

OUT = ROOT / 'experiments/b0_straight_dense_20261101'
LARGE = SCRATCH / 'b0_straight_dense_20261101'
EVAL_PLAN = ROOT / 'experiments/multidataset_continuous/plan.json'
CHECKPOINTS = {
    'SyntheticMRI': ROOT / 'experiments/local_refinement_long/92741/runs/B0/model-12000.pt',
    'BRAVA': ix.D08 / 'large/runs/synthetic/model-epoch-300.pt',
}
PREDICTIONS = {'SyntheticMRI': LARGE / 'predictions/SyntheticMRI', 'BRAVA': LARGE / 'predictions/BRAVA',
               'IXI': ix.LARGE / 'predictions/epoch-100'}
# Separately trained straight-edge lineage (scripts/straight_lineage_20261101.py), same evaluator.
LINEAGE = SCRATCH / 'straight_lineage_20261101'
TRAINED_CHECKPOINTS = {'SyntheticMRI': LINEAGE / 's2/92741/runs/B0/model-12000.pt',
                       'BRAVA': LINEAGE / 'brava/large/runs/synthetic/model-epoch-300.pt'}
TRAINED_PREDICTIONS = {'SyntheticMRI': LARGE / 'predictions-trained-straight/SyntheticMRI',
                       'BRAVA': LARGE / 'predictions-trained-straight/BRAVA',
                       'IXI': LINEAGE / 'ixi/predictions/epoch-100'}
DATASETS = ('SyntheticMRI', 'IXI', 'BRAVA')
ARMS = ('b0', 'straight')


def infer(dataset, trained=False):
    import torch
    import training.evaluation.inference as inference
    from configs import validate_config
    from models import build_model
    from models.matcher import build_matcher
    from scripts.train_ixi_transfer import TransferCriterion
    checkpoint, folder = (TRAINED_CHECKPOINTS, TRAINED_PREDICTIONS) if trained else (CHECKPOINTS, PREDICTIONS)
    checkpoint, folder = checkpoint[dataset], folder[dataset]
    cp = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg, spec = cp['config'], ix.read(EVAL_PLAN)['datasets'][dataset]
    assert cfg['data']['datasets']['synthetic_mri']['root'] == spec['root']
    assert (float(cfg['loss']['weights']['spline']) == 0) == trained
    assert cfg['model']['spline']['enabled'] and not cfg['model']['node_refinement'].get('enabled', False)
    validate_config(cfg)
    ix.seed(ix.SEED)
    model = build_model(cfg)
    model.load_state_dict(cp['net'], strict=True)
    step = cp.get('steps', cp.get('step'))
    del cp
    model.cuda()
    criterion = TransferCriterion(cfg, build_matcher(cfg), model.relation_embed,
                                  spline_head=model.spline_head, route='graph_only').cuda()
    criterion.set_training_progress(1, 0.)
    queries = cfg['model']['decoder']['object_queries']
    original = inference.infer_graphs

    def with_queries(*args, **kwargs):  # ix.infer hard-codes the 256 queries of IXI/BRAVA
        return original(*args, **dict(kwargs, object_queries=queries))
    inference.infer_graphs = with_queries
    plan = dict(config=cfg, dataset=spec['root'], ids=dict(train=spec['monitor_train'], val=spec['val_ids']),
                populations=dict(train_monitor=spec['monitor_train'], val=spec['val_ids']))
    ix.snapshot(plan, cfg, model, criterion, step, folder, cpu_workers=6)
    ix.write(folder / 'checkpoint.json', dict(path=str(checkpoint), sha256=ix.sha(checkpoint), step=step, object_queries=queries))


def score_patch(item):
    from diagnose_branch_relations import match_nodes
    from ixi_query_diagnostics import edge_accounting
    from metrics.common_vessel_metrics import evaluate_common_vessels
    sample, cache, gt_path, policies = item
    gt = ix.load_gt(gt_path)
    m = gt['metric']
    out = {}
    for name, (tn, te) in policies.items():
        qids, pairs, scores = ix.subset(cache, tn)
        np.testing.assert_array_equal(cache[f'{name}__query_ids'], qids)
        nodes, edges = cache[f'{name}__nodes'], cache[f'{name}__edges'].reshape(-1, 2)
        decoded = {tuple(sorted(map(int, e))) for e in edges}
        selected = np.array([tuple(sorted(map(int, p))) in decoded for p in pairs], dtype=bool)
        mapping = match_nodes(nodes @ m.T, gt['nodes'] @ m.T, 2.)
        row, *_ = edge_accounting(np.arange(len(nodes)), pairs, scores, selected, mapping, gt['edges'], len(gt['nodes']))
        spline = evaluate_common_vessels(dict(nodes=nodes, edges=edges, spline_controls=cache[f'{name}__spline_controls']), gt_path)
        chord = evaluate_common_vessels(dict(nodes=nodes, edges=edges), gt_path)  # no curves -> endpoint chords
        out[name] = dict(b0=counts(None, row, spline), straight=counts(None, row, chord))
    return sample, out


def pooled(per, groups):
    total = dict.fromkeys(KEYS, 0)
    for group in groups:
        for i in group:
            for k in KEYS:
                total[k] += per[i][k]
    return metrics(total)


def model_counts(folder, spec, pool, label):
    """Per-patch counts of one RelationFormer's saved predictions, with parity against its own summary."""
    summary = ix.read(folder / 'summary.json')
    policies = {ix.FIXED: (.5, .5), ix.SELECTED: (summary['thresholds']['node'], summary['thresholds']['edge'])}
    per, parity = {}, {}
    for population, split in ix.POPULATIONS:
        expected = spec['val_ids' if population == 'val' else 'monitor_train']
        with zipfile.ZipFile(folder / f'{population}.zip') as archive:
            caches = [(n[:-4], dict(np.load(io.BytesIO(archive.read(n))))) for n in archive.namelist()]
        assert sorted(s for s, _ in caches) == sorted(expected)
        items = [(s, c, str(Path(spec['root']) / split / 'spline' / f'{s}.npz'), policies) for s, c in caches]
        per[population] = dict(pool.imap(score_patch, items, chunksize=4))
        del caches, items
        for name in policies:
            point = pooled({s: r[name]['b0'] for s, r in per[population].items()}, [expected])
            reported = summary['metrics'][population]['policies'][name]
            parity[f'{label}:{population}/{name}'] = {
                'node_f1': abs(point['node_f1']-reported['2.0']['node_f1']), 'edge_f1': abs(point['edge_f1']-reported['2.0']['f1']),
                'branch_f1': abs(point['branch_f1']-reported['common_v2']['common_branch_f1']),
                'curve_f1': abs(point['curve_f1']-reported['common_v2']['common_curve_f1'])}
        print('scored', label, population, len(per[population]), flush=True)
    return summary, policies, per, parity


def score(dataset, workers):
    spec = ix.read(EVAL_PLAN)['datasets'][dataset]
    trained = TRAINED_PREDICTIONS[dataset] / 'summary.json'
    with mp.get_context('fork').Pool(workers) as pool:
        summary, policies, per, parity = model_counts(PREDICTIONS[dataset], spec, pool, 'b0')
        if trained.exists():
            trained_summary, trained_policies, trained_per, trained_parity = model_counts(trained.parent, spec, pool, 'trained')
            parity.update(trained_parity)
    dense_result = json.loads((SCRATCH / f'dense_graph_20261030/{dataset}/spline6/final-v2.json').read_text())
    dense_rows = pickle.loads((SCRATCH / f'dense_graph_20261030/{dataset}/spline6/final-v2-validation-rows.pkl').read_bytes())
    dt = dense_result['thresholds']
    dense = {ix.SELECTED: {r['sample_id']: dense_patch(r, dt['node'], dt['edge']) for r in dense_rows},
             ix.FIXED: {r['sample_id']: dense_patch(r, .5, .5) for r in dense_rows}}
    del dense_rows
    ids = sorted(per['val'])
    assert ids == sorted(dense[ix.SELECTED]), 'validation populations differ'
    subjects = {}
    for i in ids:
        subjects.setdefault(i.split('_')[1], []).append(i)
    names = sorted(subjects)
    arms = {}
    for name in policies:
        arms[f'dense/{name}'] = dense[name]
        for arm in ARMS:
            arms[f'{arm}/{name}'] = {s: r[name][arm] for s, r in per['val'].items()}
        if trained.exists():  # its own decoding; its K8 controls are exact chords (spline weight 0)
            arms[f'trained/{name}'] = {s: r[name]['b0'] for s, r in trained_per['val'].items()}
    point = {a: pooled(p, [subjects[s] for s in names]) for a, p in arms.items()}
    reported = {f'dense/{ix.SELECTED}': dense_result['validation'], f'dense/{ix.FIXED}': dense_result['validation_fixed_0p5']}
    for a, r in reported.items():
        parity[a] = {m: abs(point[a][m]-r[k]) for m, k in (('node_f1', 'node_f1'), ('edge_f1', 'edge_f1'),
                                                            ('branch_f1', 'common_branch_f1'), ('curve_f1', 'common_curve_f1'))}
    contrasts = [(f'dense/{n}', f'{a}/{n}') for n in policies for a in ARMS] + [(f'b0/{n}', f'straight/{n}') for n in policies]
    if trained.exists():
        assert sorted(arms[f'trained/{ix.SELECTED}']) == ids
        contrasts += [(f'{a}/{n}', f'trained/{n}') for n in policies for a in ('dense', 'b0', 'straight')]
    rng = np.random.default_rng(261030)
    boot = {c: {m: [] for m in point[c[0]]} for c in contrasts}
    for _ in range(2000):
        draw = [subjects[names[j]] for j in rng.integers(len(names), size=len(names))]
        sample = {a: pooled(p, draw) for a, p in arms.items()}
        for left, right in contrasts:
            for m in boot[(left, right)]:
                boot[(left, right)][m].append(sample[left][m]-sample[right][m])
    train = {f'{arm}/{name}': pooled({s: r[name][arm] for s, r in per['train_monitor'].items()}, [list(per['train_monitor'])])
             for name in policies for arm in ARMS}
    if trained.exists():
        train.update({f'trained/{name}': pooled({s: r[name]['b0'] for s, r in trained_per['train_monitor'].items()},
                                                [list(trained_per['train_monitor'])]) for name in policies})
    train[f'dense/{ix.SELECTED}'] = {m: dense_result['train_monitor'][k] for m, k in (
        ('node_f1', 'node_f1'), ('edge_f1', 'edge_f1'), ('branch_f1', 'common_branch_f1'), ('curve_f1', 'common_curve_f1'))}
    b0_summary = summary['metrics']['val']['policies']
    result = dict(
        dataset=dataset, subjects=len(names), patches=len(ids),
        b0_checkpoint=ix.read(PREDICTIONS[dataset] / 'checkpoint.json') if (PREDICTIONS[dataset] / 'checkpoint.json').exists() else dict(path=str(ix.LARGE/'runs/synthetic/model-epoch-100.pt'), epoch=summary['epoch']),
        thresholds=dict(dense=dt, b0_and_straight=dict(node=policies[ix.SELECTED][0], edge=policies[ix.SELECTED][1])),
        validation=point, train_monitor=train,
        differences={f'{l} - {r}': {m: dict(point=point[l][m]-point[r][m],
                                            ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
                                    for m, v in boot[(l, r)].items()} for l, r in contrasts},
        b0_box_mAP_mAR={n: {k: b0_summary[n][f'relationformer_box_{k}'] for k in ('node_mAP', 'node_mAR', 'edge_mAP', 'edge_mAR')}
                        for n in policies},
        b0_node_ap_2mm=summary['metrics']['val']['node_ap']['2.0']['node_ap'],
        parity_abs_difference_vs_reported=parity)
    if trained.exists():
        result.update(trained_checkpoint=ix.read(trained.parent / 'checkpoint.json') if (trained.parent / 'checkpoint.json').exists() else dict(epoch=trained_summary['epoch']),
                      trained_thresholds=dict(node=trained_policies[ix.SELECTED][0], edge=trained_policies[ix.SELECTED][1]),
                      trained_chord_vs_own_decoding={n: {m: abs(point[f'trained/{n}'][m]-pooled({s: r[n]['straight'] for s, r in trained_per['val'].items()}, [ids])[m])
                                                         for m in point[f'trained/{n}']} for n in policies})
    ix.write(OUT / (f'{dataset}-with-trained.json' if trained.exists() else f'{dataset}.json'), result)
    print(json.dumps(dict(validation=point, parity=parity), indent=1))


def collect():
    rows = []
    for d in DATASETS:
        r = ix.read(OUT / f'{d}.json')
        for arm in ('dense', 'b0', 'straight'):
            for name in (ix.SELECTED, ix.FIXED):
                v = r['validation'][f'{arm}/{name}']
                rows.append(f"| {d} | {arm} | {name} | {v['node_f1']:.3f} | {v['edge_f1']:.3f} | {v['branch_f1']:.3f} | {v['curve_f1']:.3f} |")
    print('| Dataset | Arm | Thresholds | Node F1 | Edge F1 | Branch F1 | Curve F1 |\n|---|---|---|---|---|---|---|')
    print('\n'.join(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['infer', 'score', 'collect'])
    p.add_argument('dataset', nargs='?', choices=DATASETS)
    p.add_argument('--trained', action='store_true', help='infer: the separately trained straight-edge lineage')
    p.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    a = p.parse_args()
    if a.mode == 'infer':
        assert a.dataset in CHECKPOINTS, 'IXI predictions come from the training runs themselves'
        infer(a.dataset, a.trained)
    elif a.mode == 'score':
        score(a.dataset, a.workers)
    else:
        collect()


if __name__ == '__main__':
    main()
