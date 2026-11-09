#!/usr/bin/env python3
"""Test-time check of the minimum-arc K rule (9 Nov 2026) on an already trained traced model.

The same model predicts the validation set twice: with the rule it was trained and evaluated with (original: the
arc/chord ratio may raise K at any length) and with the new rule (ratio counts only once the traced arc reaches
8 mm for K16 / 20 mm for K24; scripts/dense_graph_traced_20261030.py MODE['k_min_arc']). Nodes, candidates, paths and
edge scores do not depend on K (asserted), so only curves change: this isolates geometry exactly.
Reported at the run's own train-selected thresholds (final.json):
  - Curve F1 / Branch F1 (and the unchanged node/edge F1, as a check) with a subject-paired bootstrap, new - old;
  - per accepted true-positive edge whose K changed, by chord length: mean and Hausdorff distance (mm) of the curve to
    the GT dense path, and shape (arc/chord, total turning in degrees at 1 mm steps) of old curve, new curve and GT.
The K8 head was trained on traced pairs too, so routing short traced pairs to it is in-distribution, but the heads
were trained under the old routing: a retrained model is the real test.
Writes experiments/k_rule_check_20261109/<dataset>.json. usage: k_rule_check_20261109.py <dataset> [--run name] [--smoke]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402
from compare_dense_anchored_20261030 import KEYS, dense_patch  # noqa: E402
from data.spline_targets import resample  # noqa: E402
from decode_study_20261105 import gt  # noqa: E402
from ixi_query_diagnostics import betti  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402
from protocol_report_20261106 import paired, pooled  # noqa: E402

RUNS = {'IXI-loops': 'traced_adaptive', 'BRAVA': 'traced_offset', 'TopBrainCTA-loops': 'traced_adaptive',
        'TopBrainMRA-loops': 'traced_adaptive', 'SyntheticMRI-loops': 'traced_adaptive'}
CHORD_BINS = [(0, 5), (5, 10), (10, 20), (20, 1e9)]
CTX = {}


def dense(p, step=0.25):
    n = max(2, int(np.ceil(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() / step)) + 1)
    return resample(p, n)


def shape(p_mm, step=1.0):
    """(arc/chord, total turning in degrees) after arc-length resampling at `step` mm, so voxel-scale noise in
    either curve or GT does not count as bending."""
    arc = np.linalg.norm(np.diff(p_mm, axis=0), axis=1).sum()
    q = resample(p_mm, max(3, int(round(arc / step)) + 1))
    d = np.diff(q, axis=0)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    turn = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1))).sum()
    return float(arc / max(np.linalg.norm(p_mm[-1] - p_mm[0]), 1e-6)), float(turn)


def counts(r):
    c = dense_patch(r, CTX['node_t'], CTX['edge_t'])
    keep = r['node_scores'] >= CTX['node_t']
    acc = (r['edge_scores'] >= CTX['edge_t']) & keep[r['pairs']].all(1) if r['edge_scores'] is not None else np.zeros(0, bool)
    _, _, _, (g0, g1), _ = gt(r['cache'])
    p0, p1 = betti(int(keep.sum()), r['pairs'][acc] if len(acc) else np.zeros((0, 2), int))
    return [c[k] for k in KEYS] + [abs(p0 - g0), abs(p1 - g1)]


def patch_task(i):
    """Counts for old and new rows + distance records for accepted true-positive edges whose K changed."""
    old, new = CTX['old'][i], CTX['new'][i]
    out = dict(old=counts(old), new=counts(new), edges=[])
    if old['edge_scores'] is None:
        return out
    z = np.load(old['cache'])
    M = z['metric']
    keep = np.flatnonzero(old['node_scores'] >= CTX['node_t'])
    mp_ = D.match(torch.as_tensor(old['nodes'][keep] @ M.T), torch.as_tensor(z['nodes'] @ M.T))
    mp_ = {int(keep[a]): b for a, b in mp_.items()}
    ge = {}
    for e, (a, b) in enumerate(z['edges'].reshape(-1, 2)):
        ge.setdefault(tuple(sorted((int(a), int(b)))), []).append(e)
    off = z['path_offsets']
    for k in np.flatnonzero((old['edge_scores'] >= CTX['edge_t']) & (old['interior'] != new['interior'])):
        a, b = map(int, old['pairs'][k])
        es = ge.get(tuple(sorted((mp_.get(a, -1), mp_.get(b, -2)))))
        if a not in mp_ or b not in mp_ or not es:
            continue
        rec = dict(K_old=int(old['interior'][k]) + 2, K_new=int(new['interior'][k]),
                   chord=float(np.linalg.norm((old['nodes'][b] - old['nodes'][a]) @ M.T)))
        rec['K_new'] += 2
        for name, row in (('old', old), ('new', new)):
            curve_mm = np.concatenate([row['nodes'][a][None], row['curves'][k][1:-1], row['nodes'][b][None]]) @ M.T
            rec[f'tort_{name}'], rec[f'turn_{name}'] = shape(curve_mm)
            cd = dense(curve_mm)
            best = None
            for e in es:
                ref = dense(z['full_paths'][off[e]:off[e + 1]] @ M.T)
                d, back = cKDTree(ref).query(cd)[0], cKDTree(cd).query(ref)[0]
                cand = (float(d.mean()), float(max(d.max(), back.max())), e)
                best = cand if best is None or cand[0] < best[0] else best
            rec[f'mean_{name}'], rec[f'haus_{name}'] = best[:2]
            if name == 'old':
                rec['tort_gt'], rec['turn_gt'] = shape(z['full_paths'][off[best[2]]:off[best[2] + 1]] @ M.T)
        out['edges'].append(rec)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('--run', default=None)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    run = a.run or RUNS[a.dataset]
    run_dir = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030' / a.dataset / run
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(final)
    assert tuple(T.MODE['k_min_arc']) == (0.0, 0.0), 'this check compares against the original rule'
    assert not T.MODE['passthrough'], 'pass-through runs are not supported here'
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], two_hop=T.MODE['two_hop'],
                               backbone=final.get('backbone', 'unet')).to('cuda')
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cuda')['model'])
    val_ids = json.loads(PLAN.read_text())['datasets'][a.dataset]['val_ids']
    if a.smoke:
        val_ids = val_ids[:24]
    ds = make_dataset(a.dataset, val_ids, augment=False)
    node_t, edge_t = final['thresholds']['node'], final['thresholds']['edge']
    t0 = time.time()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # identical U-Net outputs in both passes
    old = T.predict(model, ds, 'cuda', node_t, workers)
    T.MODE['k_min_arc'] = T.K_MIN_ARC_DEFAULT
    new = T.predict(model, ds, 'cuda', node_t, workers)
    print(f'predicted twice {time.time() - t0:.0f}s', flush=True)
    max_diff = 0.0
    for o, n in zip(old, new):  # only the curves may differ
        assert o['sample_id'] == n['sample_id'] and np.array_equal(o['nodes'], n['nodes']) and np.array_equal(o['pairs'], n['pairs'])
        assert (o['edge_scores'] is None) == (n['edge_scores'] is None)
        if o['edge_scores'] is not None:
            max_diff = max(max_diff, float(np.abs(o['edge_scores'] - n['edge_scores']).max()) if len(o['edge_scores']) else 0.0)
            assert np.array_equal(o['traced'], n['traced'])
    print(f'max edge-score difference between passes {max_diff:.2e}', flush=True)
    assert max_diff < 1e-3, 'edge scores differ beyond numerical noise'
    CTX.update(old=old, new=new, node_t=node_t, edge_t=edge_t)
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(patch_task, range(len(old)), chunksize=8)
    ids = [r['sample_id'] for r in old]
    A_old, A_new = np.asarray([r['old'] for r in res], float), np.asarray([r['new'] for r in res], float)
    interior_old = np.concatenate([r['interior'] for r in old if r['interior'] is not None]) + 2
    interior_new = np.concatenate([r['interior'] for r in new if r['interior'] is not None]) + 2
    traced = np.concatenate([r['traced'] for r in old if r['traced'] is not None])
    edges = [e for r in res for e in r['edges']]
    by_chord = {}
    for lo, hi in CHORD_BINS:
        sel = [e for e in edges if lo <= e['chord'] < hi]
        if not sel:
            continue
        dm = np.array([e['mean_new'] - e['mean_old'] for e in sel])
        dh = np.array([e['haus_new'] - e['haus_old'] for e in sel])
        by_chord[f'{lo}-{hi if hi < 1e8 else "inf"}mm'] = dict(
            tp_edges_with_K_changed=len(sel),
            mean_mm_old_median=float(np.median([e['mean_old'] for e in sel])), mean_mm_new_median=float(np.median([e['mean_new'] for e in sel])),
            mean_mm_change_median=float(np.median(dm)), mean_mm_improved_fraction=float((dm < 0).mean()),
            hausdorff_mm_old_median=float(np.median([e['haus_old'] for e in sel])), hausdorff_mm_new_median=float(np.median([e['haus_new'] for e in sel])),
            hausdorff_mm_change_median=float(np.median(dh)),
            **{f'{q}_{w}_median': float(np.median([e[f'{q}_{w}'] for e in sel])) for q in ('tort', 'turn') for w in ('gt', 'old', 'new')},
            **{f'{q}_excess_over_gt_{w}_median': float(np.median([e[f'{q}_{w}'] - e[f'{q}_gt'] for e in sel])) for q in ('tort', 'turn') for w in ('old', 'new')},
            **{f'turn_excess_over_45deg_{w}': float(np.mean([e[f'turn_{w}'] - e['turn_gt'] > 45 for e in sel])) for w in ('old', 'new')},
            turn_new_lower_fraction=float(np.mean([e['turn_new'] < e['turn_old'] for e in sel])))
    out = dict(dataset=a.dataset, run=run, smoke=a.smoke, thresholds=dict(node=node_t, edge=edge_t),
               rule_old=(0.0, 0.0), rule_new=T.K_MIN_ARC_DEFAULT, max_edge_score_difference=max_diff, validation_patches=len(ids),
               candidates=dict(traced=int(traced.sum()), K_changed=int((interior_old != interior_new).sum()),
                               K_old_share={f'K{k}': float((interior_old[traced] == k).mean()) for k in (8, 16, 24)},
                               K_new_share={f'K{k}': float((interior_new[traced] == k).mean()) for k in (8, 16, 24)}),
               old=pooled(A_old), new=pooled(A_new), new_minus_old=paired(A_new, A_old, ids)['difference'],
               accepted_tp_edges_with_K_changed=by_chord, seconds=time.time() - t0)
    dst = ROOT / 'experiments/k_rule_check_20261109' / ('smoke' if a.smoke else '')
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f'{a.dataset}.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    d = out['new_minus_old']
    print('RESULT', json.dumps(dict(candidates=out['candidates'], **{k: d[k] for k in ('node_f1', 'edge_f1', 'branch_f1', 'curve_f1')},
                                    by_chord=by_chord), default=float), flush=True)
    print(f'DONE {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
