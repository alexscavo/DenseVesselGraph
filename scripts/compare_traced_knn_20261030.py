#!/usr/bin/env python3
"""Subject-paired comparison: traced + adaptive-K connectivity vs the 16-NN continued control.

CPU only, from saved validation rows. Both arms come from the same spline6 warm start with the
same budget; both use >= thresholds and carry per-pair `curves`, so both are decoded with
compare_dense_anchored_20261030.dense_patch at their own train-monitor-selected thresholds.
Per-patch counts are pooled and bootstrapped over validation subjects (2000 resamples, seed
261030). Pooled micro values are checked against each arm's own reported validation results.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_dense_anchored_20261030 import KEYS, SCRATCH, dense_patch, metrics  # noqa: E402

ARMS = ('traced_adaptive', 'knn_continued')


def main(dataset, arms=ARMS):
    """Report arms[0] - arms[1]. The default arms keep their original output file and key."""
    run = SCRATCH / f'dense_graph_20261030/{dataset}'
    result = {a: json.loads((run / a / 'final.json').read_text()) for a in arms}
    rows = {a: pickle.loads((run / a / 'final-validation-rows.pkl').read_bytes()) for a in arms}
    per = {a: {r['sample_id']: dense_patch(r, result[a]['thresholds']['node'], result[a]['thresholds']['edge'])
               for r in rows[a]} for a in arms}
    ids = sorted(per[arms[0]])
    assert ids == sorted(per[arms[1]]), 'validation populations differ'
    subjects = {}
    for i in ids:
        subjects.setdefault(i.split('_')[1], []).append(i)
    names = sorted(subjects)

    def pooled(arm, chosen):
        total = dict.fromkeys(KEYS, 0)
        for s in chosen:
            for i in subjects[s]:
                for k in KEYS:
                    total[k] += per[arm][i][k]
        return metrics(total)

    point = {a: pooled(a, names) for a in arms}
    parity = {a: {m: abs(point[a][m] - result[a]['validation'][k])
                  for m, k in (('node_f1', 'node_f1'), ('edge_f1', 'edge_f1'), ('branch_f1', 'common_branch_f1'), ('curve_f1', 'common_curve_f1'))}
              for a in arms}
    rng = np.random.default_rng(261030)
    boot = {m: [] for m in point[arms[0]]}
    for _ in range(2000):
        draw = [names[j] for j in rng.integers(len(names), size=len(names))]
        t, k = pooled(arms[0], draw), pooled(arms[1], draw)
        for m in boot:
            boot[m].append(t[m] - k[m])
    default = tuple(arms) == ARMS
    out = dict(dataset=dataset, subjects=len(names), patches=len(ids),
               thresholds={a: result[a]['thresholds'] for a in arms}, **point,
               **{'traced_minus_knn' if default else f'{arms[0]}_minus_{arms[1]}':
                  {m: dict(point=point[arms[0]][m] - point[arms[1]][m],
                           ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
                   for m, v in boot.items()}},
               parity_abs_difference_vs_reported=parity)
    stem = 'traced-vs-knn' if default else f'{arms[0]}-vs-{arms[1]}'
    path = ROOT / f'experiments/dense_graph_20261030/paired-{stem}-{dataset}.json'
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    # usage: compare_traced_knn_20261030.py <dataset> [<arm_a> <arm_b>]
    main(sys.argv[1], tuple(sys.argv[2:4]) or ARMS)
