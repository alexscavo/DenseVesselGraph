#!/usr/bin/env python3
"""Decoding sweep (5 Nov 2026), GPU part: predictions of a trained traced model at a grid of node thresholds.

Node selection changes the traced candidates (every detected peak seeds the geodesic regions), so each node
threshold needs its own prediction. Saves raw rows (as dense_graph_traced.predict) for the train monitor and the
validation list at NODE_GRID, for the CPU decoding study (decode_study_20261105.py): joint threshold selection,
hysteresis with edge support, degree pruning. Inference only; the model is not changed.
Output: $SCRATCH/experiments/gnbm/decode_sweep_20261105/<dataset>/<run>/{monitor,val}_<t>.pkl
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_traced_20261030 as T  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402

NODE_GRID = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--run', default='traced_adaptive')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--grid', type=float, nargs='*', default=NODE_GRID)
    p.add_argument('--lists', nargs='*', default=['monitor:monitor_train', 'val:val_ids'],
                   help='name:plan_field pairs to predict, e.g. oof:oof_ids for cross-fitting folds')
    a = p.parse_args()
    scratch = Path(os.environ['SCRATCH']) / 'experiments/gnbm'
    run_dir = scratch / 'dense_graph_20261030' / a.dataset / a.run
    out = scratch / 'decode_sweep_20261105' / a.dataset / a.run
    out.mkdir(parents=True, exist_ok=True)
    final = json.loads((run_dir / 'final.json').read_text())
    T.MODE.update(final.get('mode', {}))  # runs before 3 Nov have no 'mode' field: defaults apply
    plan = json.loads(PLAN.read_text())['datasets'][a.dataset]
    device = 'cuda'
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], two_hop=T.MODE['two_hop'],
                               passthrough=T.MODE['passthrough'], backbone=final.get('backbone', 'unet')).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device)['model'])
    for split, ids in [(x.split(':')[0], plan[x.split(':')[1]]) for x in a.lists]:
        ds = make_dataset(a.dataset, ids, augment=False)
        for t in a.grid:
            path = out / f'{split}_{t:.2f}.pkl'
            if path.exists():
                continue
            start = time.time()
            rows = T.predict(model, ds, device, t, a.workers)
            path.write_bytes(pickle.dumps(rows))
            print(f'{a.dataset} {split} t={t:.2f} patches={len(rows)} seconds={time.time() - start:.0f}', flush=True)
    (out / 'source.json').write_text(json.dumps(dict(run_dir=str(run_dir), thresholds=final['thresholds'], grid=a.grid,
                                                     monitor=len(plan['monitor_train']), val=len(plan['val_ids'])), indent=1))


if __name__ == '__main__':
    main()
