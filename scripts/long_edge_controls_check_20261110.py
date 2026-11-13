#!/usr/bin/env python3
"""Is the number of spline controls what loses long edges? Three checks (10 Nov 2026, CPU).

1. Representation: endpoint-pinned least-squares cubic B-spline fit of each GT validation path (128 arc-length
   samples) with K = 8 / 16 / 24 / 32 / 48 controls; Hausdorff error in voxels, by GT arc length.
2. Training arms on the same stage-1 nodes: fraction of GT edges recovered by arc length for the K rule
   (traced_arcpen_r1109) and the single K24 head (traced_k24_r1109), from each run's final.json.
3. Acceptance with perfect nodes (oracle rows of node_miss_diagnosis_20261110): for each GT edge, was it proposed by
   tracing, only as a straight 16-NN fallback pair, or not at all, how often the edge head accepts it (>= 0.75), and
   the distance of the predicted curve to the GT path, by arc length and by the K the edge was drawn with.
Writes experiments/node_miss_diagnosis_20261110/long-edge-controls.json
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_traced_20261030 as T  # noqa: E402
from node_miss_diagnosis_20261110 import dense_path  # noqa: E402

S = Path(os.environ['SCRATCH']) / 'experiments/gnbm'
BINS = [(0, 10), (10, 20), (20, 40), (40, 1e9)]
lab = lambda lo, hi: f'{lo}-{hi} mm' if hi < 1e8 else f'>= {lo} mm'
DATASETS = ['IXI-loops', 'BRAVA', 'TopBrainCTA-loops']
out = dict(fit={}, arms={}, oracle={})
plan = json.loads((ROOT / 'experiments/multidataset_continuous/plan.json').read_text())['datasets']

print('CHECK 1: K-control fit of the GT path, Hausdorff error in voxels', flush=True)
for d in DATASETS:
    root, paths, arcs, vox = Path(plan[d]['root']), [], [], []
    for i in plan[d]['val_ids']:
        with np.load(root / 'val' / 'spline' / f'{i}.npz') as z:
            M, o, fp = z['metric'], z['path_offsets'], z['full_paths']
        v = float(np.linalg.norm(M, axis=0).mean() / 64)
        for k in range(len(o) - 1):
            P = fp[o[k]:o[k + 1]] @ M.T
            arc = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
            if arc >= 10:
                paths.append(T.resample(torch.as_tensor(P, dtype=torch.float32)[None], 128)[0])
                arcs.append(arc)
                vox.append(v)
    paths, arcs, vox = torch.stack(paths), np.array(arcs), np.array(vox)
    out['fit'][d] = {}
    print(' ', d, flush=True)
    for lo, hi in BINS[1:]:
        s = np.flatnonzero((arcs >= lo) & (arcs < hi))
        if not len(s):
            continue
        entry = dict(edges=int(len(s)))
        for K in (8, 16, 24, 32, 48):
            e = np.concatenate([T.fit_errors(paths[torch.as_tensor(c)], K).numpy() for c in np.array_split(s, max(1, len(s) // 64))]) / vox[s]
            entry[f'K{K}'] = dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)), over_half_voxel=float(np.mean(e > 0.5)), over_1_voxel=float(np.mean(e > 1)))
        out['fit'][d][lab(lo, hi)] = entry
        print(f"    {lab(lo, hi)} (n={len(s)}): " + ' | '.join(f"{k}: median {v['median']:.2f}, p90 {v['p90']:.2f}, >0.5 vox {v['over_half_voxel']:.2f}, >1 vox {v['over_1_voxel']:.2f}"
                                                              for k, v in entry.items() if k != 'edges'), flush=True)

print('\nCHECK 2: GT edges recovered on validation, same stage-1 nodes: K rule vs one K24 head (own thresholds)', flush=True)
for d in DATASETS:
    out['arms'][d] = {}
    print(' ', d)
    for run in ('traced_arcpen_r1109', 'traced_k24_r1109'):
        f = json.loads((S / 'dense_graph_20261030' / d / run / 'final.json').read_text())
        b = {k: {x: v[x] for x in ('edges', 'recovered_fraction', 'rejected_fraction', 'not_candidate_fraction', 'endpoints_missing_fraction')}
             for k, v in f['validation_by_arc_length'].items()}
        out['arms'][d][run] = dict(thresholds=f['thresholds'], by_arc=b)
        print(f"    {run} (node {f['thresholds']['node']}, edge {f['thresholds']['edge']}): "
              + ', '.join(f"{k}: recovered {v['recovered_fraction']:.3f}, rejected {v['rejected_fraction']:.3f}, endpoint missing {v['endpoints_missing_fraction']:.2f}"
                          for k, v in b.items() if k in ('10-20', '20-40', '40-inf')), flush=True)

print('\nCHECK 3: GT nodes given to stage 2; GT edges by how they were proposed; accepted at 0.75; curve-to-GT distance', flush=True)
for d in DATASETS:
    rows = pickle.loads((S / 'node_miss_20261110' / f'{d}-traced_arcpen_r1109-oracle-rows.pkl').read_bytes())['val']['gt']
    acc = {}
    for r in rows:
        if r['edge_scores'] is None:
            continue
        with np.load(r['cache']) as z:
            M, ed, o, P = z['metric'], z['edges'].reshape(-1, 2), z['path_offsets'], z['full_paths'] @ z['metric'].T
        idx = {tuple(sorted(map(int, p))): k for k, p in enumerate(r['pairs'])}
        for e, (a, b) in enumerate(ed):
            g = P[o[e]:o[e + 1]]
            arc = np.linalg.norm(np.diff(g, axis=0), axis=1).sum()
            k = idx.get(tuple(sorted((int(a), int(b)))))
            name = next(lab(lo, hi) for lo, hi in BINS if lo <= arc < hi)
            kind = 'not proposed' if k is None else 'traced' if r['traced'][k] else 'straight fallback'
            c = acc.setdefault((name, kind), dict(n=0, accepted=0, K=[], dist=[], dist_by_K={}))
            c['n'] += 1
            if k is not None:
                ok = r['edge_scores'][k] >= 0.75
                K = int(r['interior'][k]) + 2
                dist = float(cKDTree(dense_path(g)).query(dense_path(r['curves'][k] @ M.T))[0].mean())
                c['accepted'] += int(ok)
                c['K'].append(K)
                c['dist'].append(dist)
                c['dist_by_K'].setdefault(K, []).append(dist)
    del rows
    out['oracle'][d] = {}
    print(' ', d)
    for lo, hi in BINS:
        n = lab(lo, hi)
        tot = sum(v['n'] for (a, _), v in acc.items() if a == n)
        entry, parts = dict(edges=tot), []
        for kind in ('traced', 'straight fallback', 'not proposed'):
            v = acc.get((n, kind))
            if not v:
                continue
            e = dict(share=v['n'] / tot)
            if kind != 'not proposed':
                e.update(accepted=v['accepted'] / v['n'], K_share={f'K{k}': float(np.mean(np.array(v['K']) == k)) for k in (8, 16, 24)},
                         curve_distance_mm_median=float(np.median(v['dist'])), curve_distance_mm_p90=float(np.percentile(v['dist'], 90)),
                         curve_distance_mm_median_by_K={f'K{k}': float(np.median(x)) for k, x in sorted(v['dist_by_K'].items())})
                parts.append(f"{kind}: {e['share']:.2f} of edges, accepted {e['accepted']:.2f}, K24 {e['K_share']['K24']:.2f}, curve distance median {e['curve_distance_mm_median']:.2f} / p90 {e['curve_distance_mm_p90']:.2f} mm")
            else:
                parts.append(f"{kind}: {e['share']:.2f}")
            entry[kind] = e
        out['oracle'][d][n] = entry
        print(f"    {n} (n={tot}): " + ' | '.join(parts), flush=True)
(ROOT / 'experiments/node_miss_diagnosis_20261110/long-edge-controls.json').write_text(json.dumps(out, indent=1) + '\n')
