#!/usr/bin/env python3
"""Subject-paired comparison: two-stage dense model vs heatmap-anchored RelationFormer.

CPU only, from saved validation rows. Each arm is decoded with its own rule and its own
train-monitor-selected thresholds (dense: >=, 16-NN candidates on retained nodes; anchored:
strict >, all retained pairs, K8 controls). Per-patch node/edge/Branch/Curve counts are
pooled and bootstrapped over validation subjects (2000 resamples, seed 261030). The pooled
micro values are checked against each arm's own reported validation results.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from diagnose_branch_relations import match_nodes  # noqa: E402
from ixi_query_diagnostics import edge_accounting  # noqa: E402
from metrics.common_vessel_metrics import evaluate_common_vessels  # noqa: E402

SCRATCH = Path(os.environ['SCRATCH']) / 'experiments/gnbm'
KEYS = ('node_tp', 'node_fp', 'node_fn', 'tp', 'fp', 'fn', 'branch_tp', 'branch_fp', 'branch_fn',
        'curve_pred', 'curve_pred_ok', 'curve_gt', 'curve_gt_ok')


def counts(row, edge_row, common):
    return dict(node_tp=edge_row['node_tp'], node_fp=edge_row['node_fp'], node_fn=edge_row['node_fn'],
                tp=edge_row['tp'], fp=edge_row['fp'], fn=edge_row['fn'],
                branch_tp=common['common_branch_tp'], branch_fp=common['common_branch_fp'], branch_fn=common['common_branch_fn'],
                curve_pred=common['common_curve_pred_samples'], curve_pred_ok=common['common_curve_supported_pred_samples'],
                curve_gt=common['common_curve_gt_samples'], curve_gt_ok=common['common_curve_supported_gt_samples'])


def dense_patch(r, node_t, edge_t):
    with np.load(r['cache']) as z:
        gt_nodes, gt_edges, metric = z['nodes'], z['edges'].reshape(-1, 2), z['metric']
        inserted = z['inserted'] if 'inserted' in z.files else None  # breaking-point anchors (loop caches)
    keep = np.nonzero(r['node_scores'] >= node_t)[0]
    remap = {int(k): i for i, k in enumerate(keep)}
    nodes = r['nodes'][keep]
    pairs = r['pairs'] if r['edge_scores'] is not None else np.zeros((0, 2), int)
    valid = np.isin(pairs, keep).all(1) if len(pairs) else np.zeros(0, bool)
    sub = pairs[valid]
    scores = r['edge_scores'][valid] if len(sub) else np.zeros(0)
    local = np.array([[remap[a], remap[b]] for a, b in sub], int).reshape(-1, 2)
    mapping = match_nodes(nodes @ metric.T, gt_nodes @ metric.T, 2.0)
    row, chosen, *_ = edge_accounting(np.arange(len(nodes)), local, scores, scores >= edge_t, mapping, gt_edges, len(gt_nodes), inserted)
    sel, edges = np.nonzero(valid)[0][chosen], local[chosen]
    prediction = dict(nodes=nodes, edges=edges)
    if r['curves'] is not None and len(edges):
        prediction['edge_polylines'] = [np.concatenate([nodes[e[0]][None], r['curves'][s][1:-1], nodes[e[1]][None]])
                                        for s, e in zip(sel, edges)]
    return counts(r, row, evaluate_common_vessels(prediction, r['cache']))


def anchored_patch(r, node_t, edge_t):
    with np.load(r['cache']) as z:
        gt_nodes, gt_edges, metric = z['nodes'], z['edges'].reshape(-1, 2), z['metric']
        inserted = z['inserted'] if 'inserted' in z.files else None  # breaking-point anchors (loop caches)
    keep = np.nonzero(r['node_scores'] > node_t)[0]
    remap = {int(k): i for i, k in enumerate(keep)}
    pairs = r['pairs']
    valid = np.isin(pairs, keep).all(1) if len(pairs) else np.zeros(0, bool)
    sub, scores = pairs[valid], r['edge_scores'][valid]
    local = np.array([[remap[int(a)], remap[int(b)]] for a, b in sub], int).reshape(-1, 2)
    nodes = r['nodes'][keep]
    mapping = match_nodes(nodes @ metric.T, gt_nodes @ metric.T, 2.0)
    row, chosen, *_ = edge_accounting(np.arange(len(nodes)), local, scores, scores > edge_t, mapping, gt_edges, len(gt_nodes))
    lookup = {tuple(sorted(map(int, e))): (tuple(map(int, e)), c) for e, c in zip(r['spline_edges'], r['spline_controls'])}
    edges, controls = [], []
    for a, b in sub[chosen]:
        (u, v), c = lookup[tuple(sorted((int(a), int(b))))]
        edges.append((remap[u], remap[v]))
        controls.append(c)
    prediction = dict(nodes=nodes, edges=np.array(edges, int).reshape(-1, 2),
                      spline_controls=np.array(controls).reshape(-1, 8, 3), spline_family='bspline')
    return counts(r, row, evaluate_common_vessels(prediction, r['cache']))


def metrics(c):
    f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0  # noqa: E731
    p = c['curve_pred_ok'] / c['curve_pred'] if c['curve_pred'] else 1.0
    r = c['curve_gt_ok'] / c['curve_gt'] if c['curve_gt'] else 1.0
    return dict(node_f1=f1(c['node_tp'], c['node_fp'], c['node_fn']), edge_f1=f1(c['tp'], c['fp'], c['fn']),
                branch_f1=f1(c['branch_tp'], c['branch_fp'], c['branch_fn']), curve_f1=2 * p * r / (p + r) if p + r else 0.0)


def main(dataset):
    dense_result = json.loads((SCRATCH / f'dense_graph_20261030/{dataset}/spline6/final-v2.json').read_text())
    dense_rows = pickle.loads((SCRATCH / f'dense_graph_20261030/{dataset}/spline6/final-v2-validation-rows.pkl').read_bytes())
    anchored_result = json.loads((ROOT / f'experiments/anchored_rf_20261030/final-{dataset}.json').read_text())
    anchored_rows = pickle.loads((SCRATCH / f'anchored_rf_20261030/runs/{dataset}/evaluation/validation-rows.pkl').read_bytes())
    dt, at = dense_result['thresholds'], anchored_result['thresholds']
    per = {'dense': {r['sample_id']: dense_patch(r, dt['node'], dt['edge']) for r in dense_rows},
           'anchored': {r['sample_id']: anchored_patch(r, at['node'], at['edge']) for r in anchored_rows}}
    ids = sorted(per['dense'])
    assert ids == sorted(per['anchored']), 'validation populations differ'
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

    point = {arm: pooled(arm, names) for arm in per}
    # Parity with each arm's own reported validation results (same rows, same thresholds).
    reported = {'dense': dense_result['validation'], 'anchored': anchored_result['validation_selected']}
    parity = {arm: {m: abs(point[arm][m] - reported[arm][k])
                    for m, k in (('node_f1', 'node_f1'), ('edge_f1', 'edge_f1'), ('branch_f1', 'common_branch_f1'), ('curve_f1', 'common_curve_f1'))}
              for arm in per}
    rng = np.random.default_rng(261030)
    boot = {m: [] for m in point['dense']}
    for _ in range(2000):
        draw = [names[j] for j in rng.integers(len(names), size=len(names))]
        d, a = pooled('dense', draw), pooled('anchored', draw)
        for m in boot:
            boot[m].append(d[m] - a[m])
    result = dict(dataset=dataset, subjects=len(names), patches=len(ids),
                  thresholds=dict(dense=dt, anchored=at), dense=point['dense'], anchored=point['anchored'],
                  dense_minus_anchored={m: dict(point=point['dense'][m] - point['anchored'][m],
                                                ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
                                        for m, v in boot.items()},
                  parity_abs_difference_vs_reported=parity)
    out = ROOT / f'experiments/dense_graph_20261030/paired-dense-vs-anchored-{dataset}.json'
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
