#!/usr/bin/env python3
"""Score saved validation predictions of existing runs against the simplified GT (10 Nov 2026, CPU).

The runs were trained on the original GT; nothing is re-run. Each run's saved validation rows (decoded at the run's
own train-monitor thresholds) are scored twice by D.evaluate: against the original cache and against the cache
simplified by scripts/simplify_graph_caches_20261110.py. This is NOT the effect of training on the simplified GT
(the model still fires at the nodes that were removed, which now count as false detections); it only shows how much
of the measured error sits on GT structure below the simplification scale.

usage: rescore_on_simplified_20261110.py <dataset entry> <simplified root> <run> [<run> ...]
Writes experiments/node_miss_diagnosis_20261110/rescore-<dataset>-<simplified root name>.json
"""
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402

RUNS = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
KEYS = ('node_f1', 'node_precision', 'node_recall', 'edge_f1', 'edge_precision', 'edge_recall', 'common_branch_f1', 'common_curve_f1',
        'missed_edges_without_endpoint_pair', 'beta0_mae', 'beta1_mae', 'patches')


def main(dataset, new_root, *runs):
    new_root = Path(new_root)
    out = dict(dataset=dataset, simplified_root=str(new_root), runs={})
    for run in runs:
        d = RUNS / dataset / run
        final = json.loads((d / 'final.json').read_text())
        t = final['thresholds']
        rows = pickle.loads((d / 'final-validation-rows.pkl').read_bytes())
        moved = []
        for r in rows:
            c = Path(r['cache'])
            n = new_root / c.parent.parent.name / 'spline' / c.name
            if n.exists():
                moved.append((r, dict(r, cache=str(n))))
        entry = dict(thresholds=t, patches_dropped=len(rows) - len(moved),
                     original_gt={k: v for k, v in D.evaluate([a for a, _ in moved], t['node'], t['edge']).items() if k in KEYS},
                     simplified_gt={k: v for k, v in D.evaluate([b for _, b in moved], t['node'], t['edge']).items() if k in KEYS})
        out['runs'][run] = entry
        for name in ('original_gt', 'simplified_gt'):
            e = entry[name]
            print(f"RESCORE {dataset} {run} {name:13s} node F1 {e['node_f1']:.3f} (P {e['node_precision']:.3f} R {e['node_recall']:.3f}) "
                  f"edge F1 {e['edge_f1']:.3f} (P {e['edge_precision']:.3f} R {e['edge_recall']:.3f}) branch {e['common_branch_f1']:.3f} "
                  f"curve {e['common_curve_f1']:.3f} b0 {e['beta0_mae']:.2f} b1 {e['beta1_mae']:.2f}", flush=True)
    dst = ROOT / 'experiments/node_miss_diagnosis_20261110'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f'rescore-{dataset}-{new_root.name}.json').write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
