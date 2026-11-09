#!/usr/bin/env python3
"""Which K rule picks the number of controls an edge actually needs? Offline check on the TRAINING monitor (9 Nov 2026).

For each dataset's reference traced model, on its train monitor (256 training patches), at its own node threshold:
every traced candidate pair whose endpoints are matched (2 mm) to the endpoints of one GT edge gets
  - the K chosen by each rule from the PREDICTED traced path (what inference sees):
      ratio_original (k_min_arc 0 0), ratio_min_arc (8 20), fit_tol<t> and fit_length_tol<t> (fit + arc floors) for t in TOLS voxels;
  - the K the GT shape needs ("needed K", fixed before looking at any rule): the fewest controls (8 / 16 / 24) whose
    endpoint-pinned least-squares fit of the GT dense path (resampled to 64 by arc length, smoothed at 1 mm exactly
    like the traced path) is within NEEDED_TOL_VOX = 0.5 voxel (Hausdorff); 24 if none is.
Reports per rule: agreement with the needed K, over- and under-selection, mean controls, by chord length; and the
fit tolerance that maximises pooled agreement over all datasets (one global setting, chosen on training data only).
Writes experiments/k_selection_check_20261109/k_selection.json. usage: k_selection_check_20261109.py [datasets...] [--smoke]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402

RUNS = {'IXI-loops': 'traced_adaptive', 'BRAVA': 'traced_offset', 'TopBrainCTA-loops': 'traced_adaptive',
        'TopBrainMRA-loops': 'traced_adaptive', 'SyntheticMRI-loops': 'traced_adaptive'}
TOLS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
NEEDED_TOL_VOX = 0.5
SMOOTH_MM = 1.0
CHORD_BINS = [(0, 5), (5, 10), (10, 20), (20, 1e9)]


def rule_k(paths_mm, voxel, rule, tol=None):
    traced = torch.ones(len(paths_mm), dtype=torch.bool, device=paths_mm.device)
    if rule in ('fit', 'fit_length'):
        T.MODE.update(k_rule=rule, k_fit_tol=tol, k_fit_smooth_mm=SMOOTH_MM)
    else:
        T.MODE.update(k_rule='ratio', k_min_arc=(0.0, 0.0) if rule == 'ratio_original' else T.K_MIN_ARC_DEFAULT)
    return (T.choose_interior(paths_mm, traced, voxel) + 2).cpu().numpy()


@torch.no_grad()
def collect(dataset, smoke):
    run_dir = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030' / dataset / RUNS[dataset]
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(final)
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], backbone=final.get('backbone', 'unet'),
                               feature_channels=final.get('feature_channels', 32)).to('cuda').eval()
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cuda')['model'])
    ids = json.loads(PLAN.read_text())['datasets'][dataset]['monitor_train']
    ds = make_dataset(dataset, ids[:16] if smoke else ids, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')),
                                         collate_fn=D.collate)
    node_t = final['thresholds']['node']
    recs, offset = [], 0
    for images, nodes, edges, geometry in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model.forward_image(images.to('cuda'))
        prob = out[:, 4].float().sigmoid()
        found = D.peaks(out, node_t)
        contacts, dist, label = T.traced_pairs(prob, [p for p, _ in found])
        for b, (p, _) in enumerate(found):
            if len(p) < 2 or not contacts[b]:
                continue
            metric = geometry[b]['metric'].to('cuda') / D.SIZE
            voxel = T.voxel_size(metric)
            rec = ds.records[offset + b]
            z = np.load(rec.graph.parent.parent / 'spline' / f'{rec.sample_id}.npz')
            gt_vox = torch.as_tensor(z['nodes'] * D.SIZE, device='cuda', dtype=torch.float32)
            mapping = D.match(p @ metric.T, gt_vox @ metric.T)
            ge = {}
            for e, (u, v) in enumerate(z['edges'].reshape(-1, 2)):
                ge.setdefault(tuple(sorted((int(u), int(v)))), e)
            pairs = torch.tensor(sorted(contacts[b]), dtype=torch.long, device='cuda')
            keep, gidx = [], []
            for k, (i, j) in enumerate(pairs.tolist()):
                e = ge.get(tuple(sorted((mapping.get(i, -1), mapping.get(j, -2))))) if i in mapping and j in mapping else None
                if e is not None:
                    keep.append(k)
                    gidx.append(e)
            if not keep:
                continue
            pairs = pairs[keep]
            paths, traced = T.build_paths(p, pairs, contacts[b], dist[b], label[b])
            m = metric[None].expand(len(paths), 3, 3)
            paths_mm = T.to_mm(paths, m)
            off = z['path_offsets']
            gt_paths = torch.stack([T.resample(torch.as_tensor(z['full_paths'][off[e]:off[e + 1]] * D.SIZE, device='cuda',
                                                               dtype=torch.float32)[None], 64)[0] for e in gidx])
            gt_mm = T.smooth_physical(T.to_mm(gt_paths, m), SMOOTH_MM)
            needed = np.full(len(gidx), 24)
            open_ = np.ones(len(gidx), bool)
            for c in (8, 16):
                ok = (T.fit_errors(gt_mm, c) <= NEEDED_TOL_VOX * voxel).cpu().numpy() & open_
                needed[ok] = c
                open_ &= ~ok
            chord = (paths_mm[:, -1] - paths_mm[:, 0]).norm(dim=-1).cpu().numpy()
            ks = {r: rule_k(paths_mm, voxel, r) for r in ('ratio_original', 'ratio_min_arc')}
            ks.update({f'fit_tol{t}': rule_k(paths_mm, voxel, 'fit', t) for t in TOLS})
            ks.update({f'fit_length_tol{t}': rule_k(paths_mm, voxel, 'fit_length', t) for t in TOLS})
            for k in range(len(gidx)):
                recs.append(dict(chord=float(chord[k]), needed=int(needed[k]), **{r: int(v[k]) for r, v in ks.items()}))
        offset += len(images)
    return recs


def summary(recs, rules):
    out = {}
    for r in rules:
        k = np.array([x[r] for x in recs])
        n = np.array([x['needed'] for x in recs])
        out[r] = dict(agreement=float((k == n).mean()), over=float((k > n).mean()), under=float((k < n).mean()),
                      mean_controls=float(k.mean()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('datasets', nargs='*', default=list(RUNS))
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    rules = ['ratio_original', 'ratio_min_arc'] + [f'fit_tol{t}' for t in TOLS] + [f'fit_length_tol{t}' for t in TOLS]
    per, pooled = {}, []
    for ds in a.datasets:
        recs = collect(ds, a.smoke)
        pooled += recs
        n = np.array([x['needed'] for x in recs])
        per[ds] = dict(pairs=len(recs), needed_share={f'K{c}': float((n == c).mean()) for c in (8, 16, 24)},
                       overall=summary(recs, rules),
                       by_chord={f'{lo}-{hi if hi < 1e8 else "inf"}mm': dict(pairs=len(s), needed_share={f'K{c}': float(np.mean([x['needed'] == c for x in s])) for c in (8, 16, 24)}, **summary(s, rules))
                                 for lo, hi in CHORD_BINS for s in [[x for x in recs if lo <= x['chord'] < hi]] if s})
        print(ds, json.dumps({r: round(v['agreement'], 3) for r, v in per[ds]['overall'].items()}), flush=True)
    macro = {r: float(np.mean([per[d]['overall'][r]['agreement'] for d in per])) for r in rules}
    best = max(TOLS, key=lambda t: macro[f'fit_tol{t}'])
    best_hybrid = max(TOLS, key=lambda t: macro[f'fit_length_tol{t}'])
    out = dict(needed_tol_voxels=NEEDED_TOL_VOX, smooth_mm=SMOOTH_MM, split='monitor_train (training patches only)',
               per_dataset=per, macro_agreement=macro, pooled=summary(pooled, rules), selected_fit_tol_voxels=best, selected_fit_length_tol_voxels=best_hybrid,
               selection_rule='fit tolerance maximising macro-average agreement with the needed K over datasets')
    dst = ROOT / 'experiments/k_selection_check_20261109' / ('smoke' if a.smoke else '')
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'k_selection.json').write_text(json.dumps(out, indent=1) + '\n')
    print('MACRO', json.dumps({k: round(v, 3) for k, v in macro.items()}), 'SELECTED', best, flush=True)


if __name__ == '__main__':
    main()
