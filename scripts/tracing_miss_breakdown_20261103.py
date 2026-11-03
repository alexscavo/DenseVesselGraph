#!/usr/bin/env python3
"""GT edges whose two endpoints are both detected, but the edge is missed: how much is tracing? (3 Nov 2026)

CPU only, from saved validation rows, at each arm's train-monitor-selected thresholds (as in
length_breakdown). Each GT edge with both endpoints matched (2 mm) falls into one of:
  traced_recovered / traced_rejected    the geodesic regions touch; the edge head accepts / rejects
  fallback_recovered / fallback_rejected no geodesic contact, pair proposed only by the 16-NN straight fallback
  not_candidate                         neither tracing nor 16-NN proposes the pair
Tracing failure = fallback_* + not_candidate. Arms without a `traced` field (knn_continued) count as fallback.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
from compare_dense_anchored_20261030 import SCRATCH  # noqa: E402

ARMS = ('knn_continued', 'traced_adaptive', 'traced_gap', 'traced_soft')
KEYS = ('edges', 'endpoints_missing', 'both_found', 'traced_recovered', 'traced_rejected',
        'fallback_recovered', 'fallback_rejected', 'not_candidate')
BINS = {'all': (0, 1e9), '<20mm': (0, 20), '>=20mm': (20, 1e9), '>=40mm': (40, 1e9)}


def edge_outcomes(rows, node_t, edge_t):
    """Per validation row: list of (arc_mm, outcome) over its GT edges, in cache order."""
    for r in rows:
        with np.load(r['cache']) as z:
            M, gn, ge = z['metric'], z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2)
            P, o = z['full_paths'] @ z['metric'].T, z['path_offsets']
        keep = np.nonzero(r['node_scores'] >= node_t)[0]
        inv = {g: i for i, g in D.match(torch.as_tensor(r['nodes'][keep] @ M.T), torch.as_tensor(gn)).items()}
        score = {tuple(sorted(map(int, p))): s for p, s in zip(r['pairs'], r['edge_scores'])} if r['edge_scores'] is not None else {}
        traced = {tuple(sorted(map(int, p))): bool(t) for p, t in zip(r['pairs'], r['traced'])} if r.get('traced') is not None else {}
        out = []
        for e, (a, b) in enumerate(ge):
            arc = np.linalg.norm(np.diff(P[o[e]:o[e + 1]], axis=0), axis=1).sum()
            if a not in inv or b not in inv:
                outcome = 'endpoints_missing'
            else:
                k = tuple(sorted((int(keep[inv[a]]), int(keep[inv[b]]))))
                if k not in score:
                    outcome = 'not_candidate'
                else:
                    kind = 'traced' if traced.get(k, False) else 'fallback'
                    outcome = f"{kind}_{'recovered' if score[k] >= edge_t else 'rejected'}"
            out.append((arc, outcome))
        yield r, out


def breakdown(rows, node_t, edge_t):
    tally = {b: dict.fromkeys(KEYS, 0) for b in BINS}
    for _, edges in edge_outcomes(rows, node_t, edge_t):
        for arc, outcome in edges:
            for name, (lo, hi) in BINS.items():
                if lo <= arc < hi:
                    t = tally[name]
                    t['edges'] += 1
                    t[outcome] += 1
                    t['both_found'] += outcome != 'endpoints_missing'
    for t in tally.values():
        n = max(1, t['both_found'])
        t['missed_given_both_found'] = (t['both_found'] - t['traced_recovered'] - t['fallback_recovered']) / n
        t['tracing_failure_given_both_found'] = (t['fallback_recovered'] + t['fallback_rejected'] + t['not_candidate']) / n
        t['missed_by_tracing_given_both_found'] = (t['fallback_rejected'] + t['not_candidate']) / n
        t['missed_by_edge_head_given_both_found'] = t['traced_rejected'] / n
    return tally


def main():
    out = {}
    for ds in ['SyntheticMRI', 'IXI', 'BRAVA']:
        for arm in ARMS:
            d = SCRATCH / f'dense_graph_20261030/{ds}/{arm}'
            if not (d / 'final-validation-rows.pkl').exists():
                continue
            th = json.loads((d / 'final.json').read_text())['thresholds']
            rows = pickle.loads((d / 'final-validation-rows.pkl').read_bytes())
            out.setdefault(ds, {})[arm] = dict(thresholds=dict(node=th['node'], edge=th['edge']),
                                               **breakdown(rows, th['node'], th['edge']))
            del rows
            print(ds, arm, json.dumps(out[ds][arm]['all']), json.dumps(out[ds][arm]['>=20mm']), flush=True)
    (ROOT / 'experiments/dense_graph_20261030/tracing-miss-breakdown.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
