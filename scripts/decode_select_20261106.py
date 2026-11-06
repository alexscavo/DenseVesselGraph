#!/usr/bin/env python3
"""Selection rules for the tree-rule hysteresis decoding (V4), 6 Nov 2026.

decode_study_20261105 chose V4's settings by monitor edge F1 subject to the Betti errors not exceeding V0's. On IXI
and TopBrain CTA that traded some Curve F1 for fewer loops. Here, every V4 setting is evaluated on the TRAIN MONITOR
with the full protocol (node/edge/Branch/Curve F1, loop-aware; Betti errors), and three rules pick one:
  A  max edge F1   s.t. |beta0|, |beta1| errors <= V0 and Curve F1 >= V0
  B  max Branch F1 s.t. |beta0|, |beta1| errors <= V0
  C  max Branch F1 s.t. |beta0|, |beta1| errors <= V0 and Curve F1 >= V0
If no setting satisfies a rule, it falls back to V0 (reported). Validation is used once, for each chosen setting vs V0
(subject-paired bootstrap). Writes experiments/decode_study_20261105/select_<dataset>.json.
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
from compare_dense_anchored_20261030 import KEYS, dense_patch, metrics  # noqa: E402
from decode_study_20261105 import EDGE_GRID, OUT, as_row, decode, gt, load, paired  # noqa: E402
from ixi_query_diagnostics import betti  # noqa: E402

ROWS = {}


def full(args):
    """Pooled F1s and mean Betti errors of one setting over the monitor rows at its tracing threshold."""
    t_rows, cfg = args
    total = dict.fromkeys(KEYS, 0)
    e0 = e1 = 0.0
    rows = ROWS[t_rows]
    for r in rows:
        nodes, pairs = decode(r, **cfg)
        dr = as_row(r, nodes, pairs)
        c = dense_patch(dr, 0.5, 0.5)
        for k in KEYS:
            total[k] += c[k]
        _, _, _, (g0, g1), _ = gt(r['cache'])
        p0, p1 = betti(len(dr['nodes']), dr['pairs'])
        e0 += abs(p0 - g0)
        e1 += abs(p1 - g1)
    return dict(metrics(total), mae_beta0=e0 / len(rows), mae_beta1=e1 / len(rows))


def main(dataset, run='traced_adaptive'):
    study = json.loads((OUT / f'{dataset}.json').read_text())
    grid = study['grid']
    v0 = study['chosen']['V0_current']['config']
    for t in grid:
        ROWS[t] = load(dataset, run, 'monitor', t)
    jobs = [(v0['t_high'], v0)]
    jobs += [(tl, dict(t_low=tl, t_high=th, t_edge=te, weak_tree=True)) for tl, th, te in product(grid, grid, EDGE_GRID) if th > tl]
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))) as pool:
        scores = pool.map(full, jobs, chunksize=2)
    base, options = scores[0], list(zip(jobs[1:], scores[1:]))
    topo = lambda s: s['mae_beta0'] <= base['mae_beta0'] + 1e-9 and s['mae_beta1'] <= base['mae_beta1'] + 1e-9  # noqa: E731
    curve = lambda s: s['curve_f1'] >= base['curve_f1'] - 1e-9  # noqa: E731
    rules = dict(A_edge_topology_curve=(lambda s: topo(s) and curve(s), 'edge_f1'),
                 B_branch_topology=(topo, 'branch_f1'),
                 C_branch_topology_curve=(lambda s: topo(s) and curve(s), 'branch_f1'))
    chosen = {}
    for name, (ok, key) in rules.items():
        feasible = [o for o in options if ok(o[1])]
        if feasible:
            (t_rows, cfg), s = max(feasible, key=lambda o: o[1][key])
        else:
            (t_rows, cfg), s = (v0['t_high'], v0), base
        chosen[name] = dict(config=cfg, rows_threshold=t_rows, monitor=s, feasible_options=len(feasible), options=len(options),
                            fell_back_to_V0=not feasible)
        print(dataset, name, json.dumps(chosen[name]), flush=True)
    print(dataset, 'V0 monitor', json.dumps(base), flush=True)
    # validation, once per chosen setting
    val = {}
    def per_patch(t_rows, cfg):  # noqa: E306
        if t_rows not in val:
            val[t_rows] = load(dataset, run, 'val', t_rows)
        out, bet = {}, []
        for r in val[t_rows]:
            nodes, pairs = decode(r, **cfg)
            dr = as_row(r, nodes, pairs)
            out[r['sample_id']] = dense_patch(dr, 0.5, 0.5)
            _, _, _, (g0, g1), _ = gt(r['cache'])
            bet.append(np.subtract(betti(len(dr['nodes']), dr['pairs']), (g0, g1)))
        b = np.abs(np.array(bet, float))
        return out, dict(mae_beta0=b[:, 0].mean(), mae_beta1=b[:, 1].mean())
    per0, bet0 = per_patch(v0['t_high'], v0)
    result = dict(dataset=dataset, V0=dict(config=v0, monitor=base, validation=paired(per0, per0)['a'], validation_betti=bet0))
    for name, c in chosen.items():
        per, bet = per_patch(c['rows_threshold'], c['config'])
        c['validation_vs_V0'] = paired(per, per0)
        c['validation_betti'] = bet
        d = c['validation_vs_V0']['difference']
        print(dataset, name, 'VAL minus V0:', ' '.join(f"{m} {v['point']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for m, v in d.items()),
              '| betti', json.dumps(bet), '| V0 betti', json.dumps(bet0), flush=True)
    result['chosen'] = chosen
    (OUT / f'select_{dataset}.json').write_text(json.dumps(result, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
