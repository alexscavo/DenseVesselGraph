#!/usr/bin/env python3
"""Subject-paired recovery of long GT edges (>= 20 mm, >= 40 mm) between two traced-model arms.

CPU only, from saved validation rows. Each arm is decoded at its own train-monitor-selected
node threshold and edge threshold through dense_graph_traced_20261030.length_breakdown. Counts
are pooled and bootstrapped over validation subjects (2000 resamples, seed 261030), as in
compare_traced_knn_20261030. Also records each arm's traced-candidate share on validation and
the last logged training traced_pair_fraction when the arm has a training log.

usage: compare_arc_length_20261102.py <dataset> <arm_a> <arm_b>   (reports arm_a - arm_b)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_dense_anchored_20261030 import SCRATCH  # noqa: E402
from compare_traced_knn_20261030 import load_arm  # noqa: E402
from dense_graph_traced_20261030 import length_breakdown  # noqa: E402

RANGES = {'ge20mm': ('20-40', '40-inf'), 'ge40mm': ('40-inf',)}
FIELDS = ('edges', 'recovered', 'endpoints_missing', 'not_candidate', 'rejected')


def main(dataset, arms):
    run = SCRATCH / f'dense_graph_20261030/{dataset}'
    result, per = {}, {}
    for a in arms:
        result[a], rows = load_arm(run, a)
        groups = {}
        for r in rows:
            groups.setdefault(r['sample_id'].split('_')[1], []).append(r)
        thresholds = result[a]['thresholds']
        per[a] = {s: length_breakdown(rows, thresholds['node'], thresholds['edge']) for s, rows in groups.items()}
    names = sorted(per[arms[0]])
    assert names == sorted(per[arms[1]]), 'validation subjects differ'

    def totals(arm, chosen, bins):
        return {f: sum(per[arm][s][b][f] for s in chosen for b in bins) for f in FIELDS}

    out = dict(dataset=dataset, arms=list(arms), subjects=len(names),
               thresholds={a: result[a]['thresholds'] for a in arms}, ranges={})
    rng = np.random.default_rng(261030)
    draws = [[names[j] for j in rng.integers(len(names), size=len(names))] for _ in range(2000)]
    for label, bins in RANGES.items():
        point = {a: totals(a, names, bins) for a in arms}
        assert point[arms[0]]['edges'] == point[arms[1]]['edges'], 'GT edge counts differ'
        boot = []
        for draw in draws:
            t, k = totals(arms[0], draw, bins), totals(arms[1], draw, bins)
            if t['edges']:
                boot.append(t['recovered'] / t['edges'] - k['recovered'] / k['edges'])
        fraction = {a: point[a]['recovered'] / max(1, point[a]['edges']) for a in arms}
        out['ranges'][label] = dict(counts=point, recovered_fraction=fraction,
                                    difference=dict(point=fraction[arms[0]] - fraction[arms[1]],
                                                    ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                                    resamples_with_edges=len(boot)))
    out['validation_traced_fraction'] = {a: result[a].get('validation_candidates', {}).get('traced_fraction') for a in arms}
    out['last_training_traced_pair_fraction'] = {}
    for a in arms:
        log = run / a / 'training.jsonl'
        lines = log.read_text().splitlines() if log.exists() else []
        out['last_training_traced_pair_fraction'][a] = json.loads(lines[-1]).get('traced_pair_fraction') if lines else None
    path = ROOT / f'experiments/dense_graph_20261030/paired-arc-{arms[0]}-vs-{arms[1]}-{dataset}.json'
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], tuple(sys.argv[2:4]))
