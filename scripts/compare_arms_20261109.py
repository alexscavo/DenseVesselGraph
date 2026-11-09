#!/usr/bin/env python3
"""Validation comparison of trained traced-model arms against the reference run of the same dataset (9 Nov 2026).

Every run (reference and arms) is decoded at the evaluation protocol's frozen global setting (V0: node 0.30, edge
0.75), so no arm benefits from its own threshold choice; each run is evaluated with the K rule / feature channels /
backbone recorded in its own final.json (T.apply_saved_mode).
Per run:
  - node AP @2 mm (threshold-free), loop-aware node/edge/Branch/Curve F1 and |beta0|, |beta1| errors (V0);
  - arm - reference: subject-paired bootstrap (protocol_report_20261106.paired);
  - shape of accepted true-positive edges (both endpoints matched to the endpoints of one GT edge), by chord length:
    arc/chord and total turning (degrees at 1 mm steps) of curve vs GT, the share turning > 45 deg more than the GT,
    mean distance to the GT path (mm); and distance by GT arc length (long edges);
  - share of K8/K16/K24 among traced candidates.
Writes experiments/arm_compare_20261109/<dataset>.json.
Runs trained with k_rule 'fixed24' are scored twice: raw K24 curves and curves compressed to the fewest controls within
--compress-tol voxels ('<run>@compressed', a tolerance fixed in advance = the needed-K tolerance of
k_selection_check_20261109).
usage: compare_arms_20261109.py <dataset> <arm run> [<arm run> ...] [--reference run] [--compress-tol 0.5]
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
from decode_study_20261105 import as_row, decode, gt  # noqa: E402
from ixi_query_diagnostics import betti  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402
from protocol_report_20261106 import paired, pooled  # noqa: E402
from traced_ap_20261105 import node_ap_from_model  # noqa: E402

REFERENCE = {'IXI-loops': 'traced_adaptive', 'BRAVA': 'traced_offset', 'TopBrainCTA-loops': 'traced_adaptive',
             'TopBrainMRA-loops': 'traced_adaptive', 'SyntheticMRI-loops': 'traced_adaptive'}
V0 = dict(t_high=0.30, t_edge=0.75)
CHORD_BINS = [(0, 5), (5, 10), (10, 20), (20, 1e9)]
ARC_BINS = [(0, 10), (10, 20), (20, 40), (40, 1e9)]
ROWS = []


def dense(p, step=0.25):
    n = max(2, int(np.ceil(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() / step)) + 1)
    return resample(p, n)


def shape(p_mm, step=1.0):
    arc = np.linalg.norm(np.diff(p_mm, axis=0), axis=1).sum()
    q = resample(p_mm, max(3, int(round(arc / step)) + 1))
    d = np.diff(q, axis=0)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    turn = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1))).sum()
    return float(arc / max(np.linalg.norm(p_mm[-1] - p_mm[0]), 1e-6)), float(turn)


def per_patch(i):
    r = ROWS[i]
    nodes, pairs = decode(r, **V0)
    dr = as_row(r, nodes, pairs)
    c = dense_patch(dr, 0.5, 0.5)
    _, _, _, (g0, g1), _ = gt(r['cache'])
    p0, p1 = betti(len(dr['nodes']), dr['pairs'])
    counts = [c[k] for k in KEYS] + [abs(p0 - g0), abs(p1 - g1)]
    kkey = 'interior_compressed' if r.get('interior_compressed') is not None else 'interior'
    accepted_k = (np.asarray(r[kkey])[pairs] + 2).tolist() if r.get(kkey) is not None and len(pairs) else []
    edges = []
    if dr['edge_scores'] is not None and len(dr['pairs']):
        z = np.load(r['cache'])
        M = z['metric']
        mp_ = D.match(torch.as_tensor(dr['nodes'] @ M.T), torch.as_tensor(z['nodes'] @ M.T))
        ge = {}
        for e, (a, b) in enumerate(z['edges'].reshape(-1, 2)):
            ge.setdefault(tuple(sorted((int(a), int(b)))), []).append(e)
        off = z['path_offsets']
        for k, (a, b) in enumerate(dr['pairs']):
            es = ge.get(tuple(sorted((mp_.get(int(a), -1), mp_.get(int(b), -2))))) if a in mp_ and b in mp_ else None
            if not es:
                continue
            curve = np.concatenate([dr['nodes'][a][None], dr['curves'][k][1:-1], dr['nodes'][b][None]]) @ M.T
            cd = dense(curve)
            best = None
            for e in es:
                P = z['full_paths'][off[e]:off[e + 1]] @ M.T
                d = cKDTree(dense(P)).query(cd)[0].mean()
                best = (d, P) if best is None or d < best[0] else best
            tc, uc = shape(curve)
            tg, ug = shape(best[1])
            edges.append(dict(chord=float(np.linalg.norm(curve[-1] - curve[0])), gt_arc=float(np.linalg.norm(np.diff(best[1], axis=0), axis=1).sum()),
                              dist=float(best[0]), tort=tc, tort_gt=tg, turn=uc, turn_gt=ug))
    return counts, edges, accepted_k


def shape_summary(edges):
    out = {}
    for lo, hi in CHORD_BINS:
        s = [e for e in edges if lo <= e['chord'] < hi]
        if s:
            out[f'chord_{lo}-{hi if hi < 1e8 else "inf"}mm'] = dict(
                tp_edges=len(s), tortuosity_median=float(np.median([e['tort'] for e in s])),
                tortuosity_gt_median=float(np.median([e['tort_gt'] for e in s])),
                turning_deg_median=float(np.median([e['turn'] for e in s])), turning_gt_deg_median=float(np.median([e['turn_gt'] for e in s])),
                turning_excess_over_45deg=float(np.mean([e['turn'] - e['turn_gt'] > 45 for e in s])),
                distance_mm_median=float(np.median([e['dist'] for e in s])))
    for lo, hi in ARC_BINS:
        s = [e for e in edges if lo <= e['gt_arc'] < hi]
        if s:
            out[f'gt_arc_{lo}-{hi if hi < 1e8 else "inf"}mm'] = dict(tp_edges=len(s), distance_mm_median=float(np.median([e['dist'] for e in s])),
                                                                      distance_mm_p90=float(np.percentile([e['dist'] for e in s], 90)),
                                                                      turning_excess_over_45deg=float(np.mean([e['turn'] - e['turn_gt'] > 45 for e in s])))
    return out


def evaluate_run(dataset, run, ds, workers, compress_tol=0.0):
    run_dir = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030' / dataset / run
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(final)
    T.MODE['compress_tol'] = compress_tol
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], two_hop=T.MODE['two_hop'], passthrough=T.MODE['passthrough'],
                               backbone=final.get('backbone', 'unet'), feature_channels=final.get('feature_channels', 32)).to('cuda')
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cuda')['model'])
    assert not T.MODE['passthrough'], 'pass-through runs need contraction; not supported here'
    ap = node_ap_from_model(model, ds, 'cuda')
    ROWS[:] = T.predict(model, ds, 'cuda', V0['t_high'], workers)
    ids = [r['sample_id'] for r in ROWS]
    key = 'interior_compressed' if compress_tol else 'interior'
    interior = np.concatenate([r[key][r['traced']] for r in ROWS if r['interior'] is not None]) + 2
    interior_all = np.concatenate([r[key] for r in ROWS if r['interior'] is not None]) + 2
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(per_patch, range(len(ROWS)), chunksize=16)
    ROWS.clear()
    del model
    torch.cuda.empty_cache()
    counts = np.asarray([r[0] for r in res], float)
    edges = [e for r in res for e in r[1]]
    acc_k = np.asarray([k for r in res for k in r[2]])
    entry = dict(run=run, mode=dict(T.MODE), feature_channels=final.get('feature_channels', 32), backbone=final.get('backbone', 'unet'),
                 own_thresholds=final['thresholds'], own_validation={k: final['validation'].get(k) for k in
                 ('node_f1', 'edge_f1', 'common_branch_f1', 'common_curve_f1', 'beta0_mae', 'beta1_mae')},
                 node_ap_2mm=ap, V0_global=pooled(counts), compress_tol_voxels=compress_tol,
                 traced_K_share={f'K{k}': float((interior == k).mean()) for k in (8, 16, 24)},
                 all_candidates_K_share={f'K{k}': float((interior_all == k).mean()) for k in (8, 16, 24)},
                 accepted_edges_K_share={f'K{k}': float((acc_k == k).mean()) for k in (8, 16, 24)} if len(acc_k) else {},
                 accepted_edges_mean_controls=float(acc_k.mean()) if len(acc_k) else None, accepted_edges=int(len(acc_k)),
                 shape=shape_summary(edges))
    return entry, counts, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('arms', nargs='+')
    p.add_argument('--reference', default=None)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--compress-tol', type=float, default=0.5,
                   help="Arms whose run uses k_rule 'fixed24' are also scored with their curves compressed at this "
                        "tolerance (voxels), as '<run>@compressed'")
    p.add_argument('--compress-all', action='store_true',
                   help="Also score every listed arm (any K rule) with compressed curves, as '<run>@compressed'")
    p.add_argument('--skip-raw', action='store_true',
                   help='Do not re-score the raw (uncompressed) arms already in the results file; only add @compressed')
    a = p.parse_args()
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    val_ids = json.loads(PLAN.read_text())['datasets'][a.dataset]['val_ids']
    if a.smoke:
        val_ids = val_ids[:24]
    ds = make_dataset(a.dataset, val_ids, augment=False)
    reference = a.reference or REFERENCE[a.dataset]
    t0 = time.time()
    dst = ROOT / 'experiments/arm_compare_20261109' / ('smoke' if a.smoke else '')
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / f'{a.dataset}.json'
    out = json.loads(path.read_text()) if path.exists() else dict(dataset=a.dataset, decoding=V0, runs={}, minus_reference={})
    ref, ref_counts, ref_ids = evaluate_run(a.dataset, reference, ds, workers)
    out['reference'] = reference
    out['runs'][reference] = ref
    variants = []
    for arm in a.arms:
        fj = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030' / a.dataset / arm / 'final.json'
        if not fj.exists():
            print('MISSING', arm, flush=True)
            continue
        if not (a.skip_raw and arm in out['runs']):
            variants.append((arm, arm, 0.0))
        if a.compress_all or json.loads(fj.read_text()).get('mode', {}).get('k_rule') == 'fixed24':
            variants.append((f'{arm}@compressed', arm, a.compress_tol))
    for name, arm, tol in variants:
        entry, counts, ids = evaluate_run(a.dataset, arm, ds, workers, tol)
        assert ids == ref_ids
        out['runs'][name] = entry
        out['minus_reference'][name] = paired(counts, ref_counts, ids)['difference']
        d = out['minus_reference'][name]
        print('RESULT', a.dataset, name, json.dumps({k: [round(d[k]['point'], 4)] + [round(x, 4) for x in d[k]['ci95']]
                                                   for k in ('node_f1', 'edge_f1', 'branch_f1', 'curve_f1', 'mae_beta0', 'mae_beta1')}),
              'nodeAP', round(entry['node_ap_2mm']['node_ap'] - ref['node_ap_2mm']['node_ap'], 4),
              'K', {k: round(v, 2) for k, v in entry['traced_K_share'].items()}, flush=True)
    out['seconds'] = time.time() - t0
    path.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(f'DONE {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
