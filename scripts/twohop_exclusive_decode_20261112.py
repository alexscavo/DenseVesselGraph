#!/usr/bin/env python3
"""Exclusive decoding of two-hop chains, and which endpoint is missing on long edges (12 Nov 2026, CPU, validation).

Saved validation rows of a --two-hop run (pairs separated by one traced node C are candidates along the joined
path A -> C -> B, scored by the ordinary edge head), decoded at the run's own thresholds.
Decodings compared:
  plain        every candidate is an ordinary edge (what the arm was evaluated with: A-B, A-C and C-B can all be kept);
  no two-hop   two-hop candidates are ignored (the model used as a plain traced model);
  exclusive    two-hop candidates are never ordinary edges. Visited by decreasing score: if the score of A-B is at
               least tau AND at least the smaller of the scores of A-C and C-B, both halves are accepted, and C has no
               other accepted edge, then C and its two edges are replaced by A-B. ('halves not required': the two
               halves need not be accepted.)
  dedupe       plain decoding, then only the duplicates are resolved: when an accepted two-hop edge A-B has both
               halves A-C and C-B accepted too, C and the halves are removed if C has no other accepted edge (C was a
               pass-through point), otherwise A-B is dropped (C is a junction and the chain stands). A two-hop edge with
               a rejected half stays: that is where the joined candidate recovers an edge the chain does not give.
tau is swept on validation here, so the numbers are descriptive; a real use would choose it on the training monitor.
Also: for GT edges >= 20 mm with a missing endpoint (plain decoding), the type of the missing GT node.
usage: twohop_exclusive_decode_20261112.py <plan entry> <two-hop run>
Writes experiments/long_edges_20261111/exclusive-<entry>-<run>.json
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
import long_edge_breaks_20261111 as L  # noqa: E402
from diagnose_branch_relations import match_nodes  # noqa: E402

KEYS = ('node_f1', 'node_precision', 'node_recall', 'edge_f1', 'edge_precision', 'edge_recall', 'beta0_mae', 'beta1_mae')


def decode(row, node_t, edge_t, mode, tau=None, need_halves=True):
    if row['edge_scores'] is None or row.get('via') is None:
        return row, 0
    via = np.asarray(row['via'])
    hop = via >= 0
    if mode == 'plain':
        return row, 0
    if mode == 'dedupe':
        ns, es, pairs, s = row['node_scores'].copy(), row['edge_scores'].copy(), row['pairs'], row['edge_scores']
        acc = (es >= edge_t) & (ns[pairs[:, 0]] >= node_t) & (ns[pairs[:, 1]] >= node_t)
        index = {tuple(map(int, pr)): k for k, pr in enumerate(pairs) if not hop[k]}
        done = 0
        for j in np.argsort(-np.where(hop & acc, s, -1.0)):
            if not (hop[j] and acc[j]):
                break
            a, b, c = int(pairs[j, 0]), int(pairs[j, 1]), int(via[j])
            ka, kb = index.get((min(a, c), max(a, c))), index.get((min(c, b), max(c, b)))
            if ka is None or kb is None or not (acc[ka] and acc[kb]) or ns[c] < node_t:
                continue
            inc = np.flatnonzero(acc & ((pairs[:, 0] == c) | (pairs[:, 1] == c)))
            if set(pairs[inc].ravel().tolist()) - {c} <= {a, b}:
                es[inc], acc[inc] = 0.0, False
                ns[c] = -1.0
                done += 1
            else:
                es[j], acc[j] = 0.0, False
        return dict(row, node_scores=ns, edge_scores=es), done
    ns, es = row['node_scores'].copy(), row['edge_scores'].copy()
    es[hop] = 0.0
    done = 0
    if mode == 'exclusive':
        pairs, s = row['pairs'], row['edge_scores']
        acc = (es >= edge_t) & (ns[pairs[:, 0]] >= node_t) & (ns[pairs[:, 1]] >= node_t)
        index = {tuple(map(int, pr)): k for k, pr in enumerate(pairs) if not hop[k]}
        removed = set()
        for j in np.argsort(-np.where(hop, s, -1.0)):
            if not hop[j] or s[j] < tau:
                break
            a, b, c = int(pairs[j, 0]), int(pairs[j, 1]), int(via[j])
            if {a, b, c} & removed or min(ns[a], ns[b], ns[c]) < node_t:
                continue
            ka, kb = index.get((min(a, c), max(a, c))), index.get((min(c, b), max(c, b)))
            if ka is None or kb is None:
                continue
            if need_halves and not (acc[ka] and acc[kb]):
                continue
            if s[j] < min(s[ka], s[kb]):
                continue
            inc = np.flatnonzero(acc & ((pairs[:, 0] == c) | (pairs[:, 1] == c)))
            if not set(pairs[inc].ravel().tolist()) - {c} <= {a, b}:
                continue
            es[inc], acc[inc] = 0.0, False
            ns[c] = -1.0
            removed.add(c)
            es[j], acc[j] = 1.0, True
            done += 1
    return dict(row, node_scores=ns, edge_scores=es), done


def main(entry, run, *_):
    d = L.RUNS / entry / run
    final = json.loads((d / 'final.json').read_text())
    node_t, edge_t = final['thresholds']['node'], final['thresholds']['edge']
    rows = pickle.loads((d / 'final-validation-rows.pkl').read_bytes())
    out = dict(entry=entry, run=run, thresholds=dict(node=node_t, edge=edge_t), decodings={})
    configs = [('plain', dict(mode='plain')), ('no two-hop', dict(mode='none')), ('dedupe', dict(mode='dedupe'))]
    if not (len(sys.argv) > 3 and sys.argv[3] == 'dedupe-only'):
        for tau in (0.5, 0.7, edge_t, 0.95):
            configs.append((f'exclusive, tau {tau:g}', dict(mode='exclusive', tau=tau)))
        configs.append((f'exclusive, tau {edge_t:g}, halves not required', dict(mode='exclusive', tau=edge_t, need_halves=False)))
    for name, kw in configs:
        new, count = zip(*[decode(r, node_t, edge_t, **kw) for r in rows])
        res = D.evaluate(list(new), node_t, edge_t, geometry=False)
        rec = L.recovered_long(new, node_t, edge_t)
        out['decodings'][name] = dict({k: res[k] for k in KEYS}, contractions_per_patch=float(np.mean(count)), recovered=rec)
        print('DECODE', entry, f'{name}:', 'node F1 %.4f | edge P/R/F1 %.4f %.4f %.4f | b0 %.2f b1 %.2f | %.2f contractions per patch | recovered' % (
            res['node_f1'], res['edge_precision'], res['edge_recall'], res['edge_f1'], res['beta0_mae'], res['beta1_mae'], np.mean(count)),
            {k: round(v, 3) for k, v in rec.items()}, flush=True)
    # which endpoint is missing on long edges
    kinds = Counter()
    for r in rows:
        with np.load(r['cache']) as z:
            M, gn, ed, P, off = z['metric'], z['nodes'], z['edges'].reshape(-1, 2), z['full_paths'] @ z['metric'].T, z['path_offsets']
            ins = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(gn), bool)
            boundary = z['boundary'].astype(bool)
        g = gn @ M.T
        deg = np.bincount(ed.ravel(), minlength=len(g))
        keep = r['node_scores'] >= node_t
        m = match_nodes(r['nodes'][keep] @ M.T, g, 2.0) if keep.any() and len(g) else {}
        hit = set(m.values())
        kind = lambda i: 'boundary' if boundary[i] else 'junction' if deg[i] >= 3 else 'terminal' if deg[i] == 1 else 'other'
        for e, (a, b) in enumerate(ed):
            if ins[a] or ins[b]:
                continue
            arc = float(np.linalg.norm(np.diff(P[off[e]:off[e + 1]], axis=0), axis=1).sum())
            if arc < 20 or (a in hit and b in hit):
                continue
            kinds['edges with a missing endpoint'] += 1
            kinds['both ends missing'] += int(a not in hit and b not in hit)
            for i in (a, b):
                if i not in hit:
                    kinds['missing ' + kind(i)] += 1
    total = kinds.pop('edges with a missing endpoint')
    out['missing_endpoint_edges_ge_20mm'] = dict(edges=total, **{k: v / max(1, total) for k, v in kinds.most_common()})
    print('MISSING', entry, json.dumps(out['missing_endpoint_edges_ge_20mm']), flush=True)
    suffix = '-dedupe' if len(sys.argv) > 3 else ''
    (ROOT / 'experiments/long_edges_20261111' / f'exclusive-{entry}-{run}{suffix}.json').write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
