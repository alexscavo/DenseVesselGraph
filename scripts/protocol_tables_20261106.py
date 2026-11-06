#!/usr/bin/env python3
"""Evaluation protocol, step 1 (6 Nov 2026): per-setting, per-patch counts for one dataset. CPU.

Settings: the V0 family (single node threshold t_node x edge threshold t_edge) and the V4 family (tree-rule hysteresis:
t_low < t_high, t_edge), over the decode-sweep grid. For every setting and patch:
  monitor (train patches, used for SELECTION): full loop-aware counts (node, edge, Branch, Curve; compare_dense_anchored
           KEYS) and the predicted / GT Betti numbers;
  val     (validation, used for REPORTING): node/edge counts and Betti numbers (no geometry: cheap enough for every
           setting, used for the oracle upper bound, sensitivity curves and monitor-size variance only).
Full validation metrics are computed later, only for the settings the rules choose (protocol_report_20261106).
Output: experiments/evaluation_protocol_20261106/tables/<dataset>.npz (+ .json with settings and patch ids).
"""
import json
import os
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compare_dense_anchored_20261030 import KEYS, dense_patch  # noqa: E402
from decode_study_20261105 import EDGE_GRID, as_row, decode, gt, load  # noqa: E402
from diagnose_branch_relations import match_nodes  # noqa: E402
from ixi_query_diagnostics import betti, edge_accounting  # noqa: E402

OUT = ROOT / 'experiments/evaluation_protocol_20261106/tables'
FAST = ('node_tp', 'node_fp', 'node_fn', 'tp', 'fp', 'fn')
BETTI = ('p0', 'p1', 'g0', 'g1')
ROWS = {}


def settings(grid):
    v0 = [dict(family='V0', t_high=tn, t_edge=te) for tn, te in product(grid, EDGE_GRID)]
    v4 = [dict(family='V4', t_low=tl, t_high=th, t_edge=te, weak_tree=True) for tl, th, te in product(grid, grid, EDGE_GRID) if th > tl]
    return v0 + v4


def cfg_args(s):
    return {k: v for k, v in s.items() if k != 'family'}


def rows_threshold(s):
    return s.get('t_low', s['t_high'])


def betti_counts(r, dr):
    _, _, _, (g0, g1), _ = gt(r['cache'])
    p0, p1 = betti(len(dr['nodes']), dr['pairs'])
    return [p0, p1, g0, g1]


def fast_counts(r, dr):
    nodes_mm, gt_edges, inserted, _, m = gt(r['cache'])
    pred_mm = dr['nodes'] @ m.T
    mapping = match_nodes(pred_mm, nodes_mm, 2.0) if len(pred_mm) and len(nodes_mm) else {}
    local = dr['pairs']
    row, *_ = edge_accounting(np.arange(len(dr['nodes'])), local, np.ones(len(local)), np.ones(len(local), bool), mapping,
                              gt_edges, len(nodes_mm), inserted)
    return [row[k] for k in FAST]


def run_setting(args):
    split, s = args
    out = []
    for r in ROWS[(split, rows_threshold(s))]:
        dr = as_row(r, *decode(r, **cfg_args(s)))
        if split == 'monitor':
            c = dense_patch(dr, 0.5, 0.5)
            out.append([c[k] for k in KEYS] + betti_counts(r, dr))
        else:
            out.append(fast_counts(r, dr) + betti_counts(r, dr))
    return np.asarray(out, np.float64)  # curve sample counts may be fractional: never truncate


def main(dataset, run='traced_adaptive'):
    sweep_src = Path(os.environ['SCRATCH']) / 'experiments/gnbm/decode_sweep_20261105' / dataset / run / 'source.json'
    src = json.loads(sweep_src.read_text())
    grid = src['grid']
    S = settings(grid)
    for split in ('monitor', 'val'):
        for t in grid:
            ROWS[(split, t)] = load(dataset, run, split, t)
    ids = {split: [r['sample_id'] for r in ROWS[(split, grid[0])]] for split in ('monitor', 'val')}
    for split in ids:  # every threshold's rows must list the same patches in the same order
        assert all([r['sample_id'] for r in ROWS[(split, t)]] == ids[split] for t in grid), split
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {}
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))) as pool:
        for split in ('monitor', 'val'):
            tables[split] = np.stack(pool.map(run_setting, [(split, s) for s in S], chunksize=1))
            print(dataset, split, tables[split].shape, flush=True)
    np.savez_compressed(OUT / f'{dataset}.npz', monitor=tables['monitor'], val=tables['val'])
    (OUT / f'{dataset}.json').write_text(json.dumps(dict(
        dataset=dataset, run=run, grid=grid, edge_grid=EDGE_GRID, settings=S, final_thresholds=src['thresholds'],
        monitor_columns=list(KEYS) + list(BETTI), val_columns=list(FAST) + list(BETTI),
        monitor_ids=ids['monitor'], val_ids=ids['val']), indent=1) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
