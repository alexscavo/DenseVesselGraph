#!/usr/bin/env python3
"""Paired comparison of two traced arms at shared thresholds, to size a threshold-shift confound (4 Nov 2026).

CPU, saved validation rows, subject-paired bootstrap as in passthrough_effect_20261104. Both arms are decoded at
the higher of their two node thresholds and at arm_b's edge threshold. A lower edge threshold is exact, because all
scores are saved. Raising an arm's node threshold only filters its rows and does not repeat tracing among the
retained nodes, so edges through a removed node are lost and that arm's Curve F1 is penalised: such rows are
marked exact=false and are an approximation. No selection happens on validation.
Results are merged into experiments/dense_graph_20261030/shared-thresholds.json.

usage: compare_shared_threshold_20261104.py <arm_a> <arm_b> [<arm_a> <arm_b> ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compare_dense_anchored_20261030 import SCRATCH  # noqa: E402
from passthrough_effect_20261104 import decode, load, paired  # noqa: E402


def main(pairs):
    path = ROOT / 'experiments/dense_graph_20261030/shared-thresholds.json'
    out = json.loads(path.read_text()) if path.exists() else {}
    for dataset in ('SyntheticMRI', 'IXI', 'BRAVA'):
        run = SCRATCH / f'dense_graph_20261030/{dataset}'
        for a, b in pairs:
            (ra, rows_a), (rb, rows_b) = load(run, a), load(run, b)
            ta, tb = ra['thresholds'], rb['thresholds']
            key = f'{a}_minus_{b}'
            own = dict(a=dict(node=ta['node'], edge=ta['edge']), b=dict(node=tb['node'], edge=tb['edge']))
            if (ta['node'], ta['edge']) == (tb['node'], tb['edge']):
                out.setdefault(dataset, {})[key] = dict(own, same_thresholds=True)
                continue
            node, edge = max(ta['node'], tb['node']), tb['edge']
            result = paired(decode(rows_a, node, edge, None), decode(rows_b, node, edge, None))
            out.setdefault(dataset, {})[key] = dict(own, same_thresholds=False, shared=dict(node=node, edge=edge),
                                                    exact=ta['node'] == tb['node'], **result)
            print('SHARED', dataset, key, json.dumps(out[dataset][key]), flush=True)
            path.write_text(json.dumps(out, indent=1))
    path.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(list(zip(sys.argv[1::2], sys.argv[2::2])))
