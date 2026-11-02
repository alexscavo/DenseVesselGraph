#!/usr/bin/env python3
"""Topology metrics on validation for every arm, each decoded at its train-selected thresholds.

Per patch: predicted / GT beta0 (connected components) and beta1 (independent cycles = loops)
via scripts/ixi_query_diagnostics.betti, and the street mover distance (SMD) of
training/evaluation/metrics.street_mover_distance with the native protocol of configs/base.yaml
(100 points per graph, Sinkhorn eps 1e-7, 100 iterations, normalized coordinates, straight
segments between nodes as in the original Relationformer metric). Arms:
- dense (two-stage, 16-NN), traced (dense + traced connectivity + adaptive K): saved rows, >=;
- anchored RelationFormer: saved rows, strict >;
- RelationFormer B0 and the trained straight-edge RelationFormer: ixi_long-style prediction
  caches (decoded graph of the train_selected policy).
CPU only. Validation only.
"""
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
from ixi_query_diagnostics import betti  # noqa: E402
from b0_straight_dense_20261101 import PREDICTIONS, TRAINED_PREDICTIONS  # noqa: E402
import ixi_long_experiment_20261030 as ix  # noqa: E402

SCRATCH = Path(os.environ['SCRATCH']) / 'experiments/gnbm'
DENSE = SCRATCH / 'dense_graph_20261030'


def patch_metrics(item):
    import torch
    from training.evaluation.metrics import street_mover_distance
    nodes, edges, cache = item
    with np.load(cache) as z:
        gn, ge = z['nodes'].astype(float), z['edges'].reshape(-1, 2)
    p0, p1 = betti(len(nodes), edges)
    g0, g1 = betti(len(gn), ge)
    smd = street_mover_distance(torch.as_tensor(gn).float(), torch.as_tensor(ge).long(),
                                torch.as_tensor(nodes).float().reshape(-1, 3), torch.as_tensor(edges).long().reshape(-1, 2),
                                num_points=100, epsilon=1e-7, max_iterations=100) if len(ge) else float('nan')
    return dict(p0=p0, p1=p1, g0=g0, g1=g1, smd=float(smd), nodes=len(nodes), edges=len(edges))


def from_rows(rows, node_t, edge_t, strict):
    for r in rows:
        keep = np.nonzero(r['node_scores'] > node_t if strict else r['node_scores'] >= node_t)[0]
        remap = {int(k): i for i, k in enumerate(keep)}
        edges = np.zeros((0, 2), int)
        if r['edge_scores'] is not None and len(keep) >= 2:
            valid = np.isin(r['pairs'], keep).all(1)
            sel = valid & ((r['edge_scores'] > edge_t) if strict else (r['edge_scores'] >= edge_t))
            edges = np.array([[remap[int(a)], remap[int(b)]] for a, b in r['pairs'][sel]], int).reshape(-1, 2)
        yield r['nodes'][keep], edges, r['cache']


def from_caches(folder, dataset):
    root = Path(ix.read(ROOT / 'experiments/multidataset_continuous/plan.json')['datasets'][dataset]['root'])
    with zipfile.ZipFile(folder / 'val.zip') as archive:
        for name in archive.namelist():
            c = dict(np.load(io.BytesIO(archive.read(name))))
            yield (c[f'{ix.SELECTED}__nodes'], c[f'{ix.SELECTED}__edges'].reshape(-1, 2),
                   str(root / 'val' / 'spline' / f'{name[:-4]}.npz'))


def arms(dataset):
    out = {}
    for name, sub, rows, result in (('dense', 'spline6', 'final-v2-validation-rows.pkl', 'final-v2.json'),
                                    ('traced', 'traced_adaptive', 'final-validation-rows.pkl', 'final.json')):
        thr = json.loads((DENSE / dataset / sub / result).read_text())['thresholds']
        out[name] = (DENSE / dataset / sub / rows, thr, False)
    for name in ('traced_gap', 'traced_soft'):
        if (DENSE / dataset / name / 'final.json').exists():
            thr = json.loads((DENSE / dataset / name / 'final.json').read_text())['thresholds']
            out[name] = (DENSE / dataset / name / 'final-validation-rows.pkl', thr, False)
    if dataset != 'SyntheticMRI':
        thr = json.loads((ROOT / f'experiments/anchored_rf_20261030/final-{dataset}.json').read_text())['thresholds']
        out['anchored'] = (SCRATCH / f'anchored_rf_20261030/runs/{dataset}/evaluation/validation-rows.pkl', thr, True)
    out['b0'] = PREDICTIONS[dataset]
    if (TRAINED_PREDICTIONS[dataset] / 'val.zip').exists():
        out['straight_trained'] = TRAINED_PREDICTIONS[dataset]
    return out


def summarize(per):
    a = {k: np.array([p[k] for p in per], float) for k in per[0]}
    smd = a['smd'][np.isfinite(a['smd'])]
    return dict(patches=len(per), beta0_mae=float(np.abs(a['p0'] - a['g0']).mean()), beta1_mae=float(np.abs(a['p1'] - a['g1']).mean()),
                components_per_patch=float(a['p0'].mean()), gt_components_per_patch=float(a['g0'].mean()),
                loops_per_patch=float(a['p1'].mean()), gt_loops_per_patch=float(a['g1'].mean()),
                smd_mean=float(smd.mean()), smd_median=float(np.median(smd)), smd_undefined_patches=int((~np.isfinite(a['smd'])).sum()),
                nodes_per_patch=float(a['nodes'].mean()), edges_per_patch=float(a['edges'].mean()))


def main(only=()):
    """Compute every arm, or with `only` just those arms, merged into the existing output."""
    output = ROOT / 'experiments/dense_graph_20261030/topology-validation.json'
    result = json.loads(output.read_text()) if only else {}
    with mp.get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        for dataset in ('SyntheticMRI', 'IXI', 'BRAVA'):
            for name, spec in arms(dataset).items():
                if only and name not in only:
                    continue
                if isinstance(spec, tuple):
                    path, thr, strict = spec
                    items = list(from_rows(pickle.loads(Path(path).read_bytes()), thr['node'], thr['edge'], strict))
                else:
                    items = list(from_caches(spec, dataset))
                result[f'{dataset}/{name}'] = summarize(pool.map(patch_metrics, items, chunksize=8))
                print(dataset, name, json.dumps(result[f'{dataset}/{name}']), flush=True)
    output.write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    # usage: topology_metrics_20261102.py [<arm> ...]
    main(tuple(sys.argv[1:]))
