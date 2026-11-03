#!/usr/bin/env python3
"""Two-hop candidates and neighbourhood offsets: node confidence, triangles, blocked-edge recovery (3 Nov 2026).

CPU, saved validation rows, each arm at its own train-monitor-selected thresholds.
Arms: traced_adaptive (reference), traced_twohop, traced_twohop_offset.
  node_scores     heatmap score of retained detections: matched (2 mm) vs spurious, and the
                  spurious "in-between" detections C (within 1.5 mm of the interior of a GT edge
                  whose endpoints are both matched, >= 2 mm from either end)
  two_hop         accepted two-hop edges A-B (pass-through C), their precision, and triangles:
                  accepted two-hop A-B whose A-C and C-B are also accepted (a spurious 3-cycle
                  unless the GT has one)
  blocked_edges   recovery of GT edges blocked by a spurious in-between detection; whether A-B is a
                  candidate at all, a two-hop candidate, and its edge score
  node_error_mm   matched-node position error (offset fix)
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

ARMS = ('traced_adaptive', 'traced_twohop', 'traced_twohop_offset')
NEAR_MM, END_MM = 1.5, 2.0


def quantiles(x):
    x = np.asarray(x, float)
    if not len(x):
        return dict(n=0)
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return dict(n=int(len(x)), mean=float(x.mean()), p10=float(q[0]), p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p90=float(q[4]))


def analyse(rows, node_t, edge_t):
    matched_s, spurious_s, between_s, err = [], [], [], []
    hop = dict(accepted=0, correct=0, triangles=0, patches=0)
    blocked = dict(edges=0, recovered=0, candidate=0, two_hop_candidate=0)
    blocked_scores = []
    for r in rows:
        with np.load(r['cache']) as z:
            M, gn, ge = z['metric'], z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2)
            P, o = z['full_paths'] @ z['metric'].T, z['path_offsets']
        hop['patches'] += 1
        keep = np.nonzero(r['node_scores'] >= node_t)[0]
        pred = r['nodes'][keep] @ M.T
        score = r['node_scores'][keep]
        m = D.match(torch.as_tensor(pred), torch.as_tensor(gn))  # local index -> GT index
        inv = {g: i for i, g in m.items()}
        is_m = np.array([i in m for i in range(len(keep))], bool)
        matched_s += list(score[is_m])
        spurious_s += list(score[~is_m])
        err += [float(np.linalg.norm(pred[i] - gn[g])) for i, g in m.items()]
        local = {int(k): i for i, k in enumerate(keep)}
        acc, cand = {}, {}
        if r['edge_scores'] is not None:
            for j, (u, v) in enumerate(r['pairs']):
                if int(u) in local and int(v) in local:
                    cand[tuple(sorted((local[int(u)], local[int(v)])))] = j
            for j, (u, v) in enumerate(r['pairs']):
                if r['edge_scores'][j] >= edge_t and int(u) in local and int(v) in local:
                    acc[tuple(sorted((local[int(u)], local[int(v)])))] = j
        gt = {tuple(sorted(map(int, e))) for e in ge}
        via = r.get('via')
        if via is not None:
            for (u, v), j in acc.items():
                c = int(via[j])
                if c < 0 or c not in local:
                    continue
                c = local[c]
                hop['accepted'] += 1
                hop['correct'] += int(u in m and v in m and tuple(sorted((m[u], m[v]))) in gt)
                hop['triangles'] += int(tuple(sorted((u, c))) in acc and tuple(sorted((c, v))) in acc)
        for e, (a, b) in enumerate(ge):
            if a not in inv or b not in inv:
                continue
            path = P[o[e]:o[e + 1]]
            along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
            inner = path[(along > END_MM) & (along < along[-1] - END_MM)]
            rest = np.setdiff1d(np.nonzero(~is_m)[0], [inv[a], inv[b]])
            if not len(inner) or not len(rest):
                continue
            near = rest[np.linalg.norm(pred[rest][:, None] - inner[None], axis=-1).min(1) < NEAR_MM]
            if not len(near):
                continue
            between_s += list(score[near])
            blocked['edges'] += 1
            k = tuple(sorted((inv[a], inv[b])))
            blocked['recovered'] += int(k in acc)
            if k in cand:
                blocked['candidate'] += 1
                blocked['two_hop_candidate'] += int(via is not None and int(via[cand[k]]) >= 0)
                blocked_scores.append(float(r['edge_scores'][cand[k]]))
    return dict(node_scores=dict(matched=quantiles(matched_s), spurious=quantiles(spurious_s), spurious_in_between=quantiles(between_s)),
                two_hop=dict(hop, precision=hop['correct'] / max(1, hop['accepted']), triangles_per_patch=hop['triangles'] / max(1, hop['patches'])),
                blocked_edges=dict(blocked, recovered_fraction=blocked['recovered'] / max(1, blocked['edges']),
                                   candidate_fraction=blocked['candidate'] / max(1, blocked['edges']),
                                   two_hop_candidate_fraction=blocked['two_hop_candidate'] / max(1, blocked['edges']),
                                   candidate_edge_score=quantiles(blocked_scores)),
                node_error_mm=quantiles(err))


def main():
    out = {}
    for ds in ['SyntheticMRI', 'IXI', 'BRAVA']:
        for arm in ARMS:
            d = SCRATCH / f'dense_graph_20261030/{ds}/{arm}'
            th = json.loads((d / 'final.json').read_text())['thresholds']
            rows = pickle.loads((d / 'final-validation-rows.pkl').read_bytes())
            out.setdefault(ds, {})[arm] = dict(thresholds=dict(node=th['node'], edge=th['edge']), **analyse(rows, th['node'], th['edge']))
            del rows
            print('TWOHOP', ds, arm, json.dumps(out[ds][arm]), flush=True)
    (ROOT / 'experiments/dense_graph_20261030/twohop-analysis.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
