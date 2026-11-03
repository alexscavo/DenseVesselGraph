#!/usr/bin/env python3
"""Node quality per arm: matched-node position error and matched vs spurious heatmap scores (3 Nov 2026).

CPU, saved validation rows, each arm at its own train-monitor-selected thresholds. Reuses
twohop_analysis_20261103.analyse, so spurious in-between detections and blocked edges are reported too.
Results are merged into experiments/dense_graph_20261030/node-quality.json.

usage: node_quality_20261103.py <arm> [<arm> ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compare_dense_anchored_20261030 import SCRATCH  # noqa: E402
from compare_traced_knn_20261030 import load_arm  # noqa: E402
from twohop_analysis_20261103 import analyse  # noqa: E402


def main(arms):
    path = ROOT / 'experiments/dense_graph_20261030/node-quality.json'
    out = json.loads(path.read_text()) if path.exists() else {}
    for dataset in ('SyntheticMRI', 'IXI', 'BRAVA'):
        for arm in arms:
            result, rows = load_arm(SCRATCH / f'dense_graph_20261030/{dataset}', arm)
            t = result['thresholds']
            out.setdefault(dataset, {})[arm] = dict(thresholds=dict(node=t['node'], edge=t['edge']), **analyse(rows, t['node'], t['edge']))
            print('NODES', dataset, arm, json.dumps(out[dataset][arm]), flush=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
