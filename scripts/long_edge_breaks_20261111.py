#!/usr/bin/env python3
"""Why are long GT edges not recovered: broken into pieces by intermediate nodes, or not drawn? (11 Nov 2026, CPU)

Saved validation rows of one traced run, decoded at the run's own train-selected thresholds.

Part 1, per GT edge of arc length >= 10 mm (edges touching an inserted loop anchor are left out):
  recovered            the predicted nodes matched to its two ends are joined by one accepted edge;
  broken, all pieces   both ends matched, and the accepted graph joins them by a chain whose intermediate nodes all
                       lie within 2 mm of the GT path (the vessel is drawn, but as several edges);
  broken, piece missing  both ends matched, at least one kept detection sits on the interior of the GT path
                       (within 1.5 mm of it, more than 2 mm of arc from both ends), no complete chain;
  rejected / not proposed  both ends matched, no detection on the interior: the direct pair was a candidate with a
                       score below the edge threshold (traced or straight fallback), or was not a candidate;
  endpoint missing     at least one end has no matched detection.
For every class: 'coverage' = share of the GT path within 1.5 mm of any accepted predicted curve.

Part 2, decoding rule tried on the same rows (descriptive: thresholds of the rule are swept on validation here, a
proper choice must be made on the training monitor): an interior predicted node (more than 2 voxels from the patch
face) with exactly two accepted edges A-C and C-B, A-B not accepted, is removed and replaced by one edge A-B drawn
along the two joined curves. Options: only when the node's heat score is below a bound; only when the two curves meet
at C with an angle below a bound. In the simplified GT an interior node always has one or at least three branches,
so such a node is never a correct node as it stands.

usage: long_edge_breaks_20261111.py <plan entry> <run>
Writes experiments/long_edges_20261111/<entry>-<run>.json
"""
import json
import os
import pickle
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
from diagnose_branch_relations import match_nodes  # noqa: E402
from node_miss_diagnosis_20261110 import dense_path  # noqa: E402

RUNS = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
BINS = [(10, 20), (20, 40), (40, 1e9)]
lab = lambda lo, hi: f'{lo}-{hi} mm' if hi < 1e8 else f'>= {lo} mm'


def resample(p, n=64):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.r_[0., np.cumsum(seg)]
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, p[:, k]) for k in range(3)], 1)


def oriented(row, k, a):
    """Curve of pair k (normalised coordinates, with exact node ends) starting at node a."""
    i, j = map(int, row['pairs'][k])
    c = np.concatenate([row['nodes'][i][None], row['curves'][k][1:-1], row['nodes'][j][None]])
    return c if i == a else c[::-1]


def accepted_graph(row, node_t, edge_t):
    keep = row['node_scores'] >= node_t
    acc = {}
    if row['edge_scores'] is not None:
        for k, (i, j) in enumerate(row['pairs']):
            if keep[i] and keep[j] and row['edge_scores'][k] >= edge_t:
                acc[(int(i), int(j))] = k
    return keep, acc


def classify(row, node_t, edge_t):
    with np.load(row['cache']) as z:
        M, gn, ed = z['metric'], z['nodes'], z['edges'].reshape(-1, 2)
        ins = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(gn), bool)
        P, off = z['full_paths'] @ M.T, z['path_offsets']
    g = gn @ M.T
    keep, acc = accepted_graph(row, node_t, edge_t)
    ids = np.flatnonzero(keep)
    p = row['nodes'][ids] @ M.T
    mapping = match_nodes(p, g, 2.0) if len(p) and len(g) else {}
    inv = {gt: int(ids[q]) for q, gt in mapping.items()}  # GT node -> predicted node index
    matched = {int(ids[q]) for q in mapping}
    adj = {}
    for (i, j) in acc:
        adj.setdefault(i, set()).add(j), adj.setdefault(j, set()).add(i)
    pts = [dense_path(oriented(row, k, i) @ M.T) for (i, j), k in acc.items()]
    curve_tree = cKDTree(np.concatenate(pts)) if pts else None
    cand = {tuple(sorted(map(int, pr))): k for k, pr in enumerate(row['pairs'])} if row['edge_scores'] is not None else {}
    out = []
    for e, (a, b) in enumerate(ed):
        if ins[a] or ins[b]:
            continue
        G = dense_path(P[off[e]:off[e + 1]])
        arc = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
        if arc < BINS[0][0]:
            continue
        cover = float((curve_tree.query(G)[0] <= 1.5).mean()) if curve_tree is not None else 0.0
        rec = dict(arc=arc, coverage=cover, interrupting=0, interrupting_unmatched=0, chain_nodes=0)
        if a not in inv or b not in inv:
            out.append(dict(rec, kind='endpoint missing', both_missing=a not in inv and b not in inv))
            continue
        A, B = inv[a], inv[b]
        if (min(A, B), max(A, B)) in acc:
            out.append(dict(rec, kind='recovered'))
            continue
        tree = cKDTree(G)
        d, pos = tree.query(p) if len(p) else (np.zeros(0), np.zeros(0, int))
        s = pos * (arc / max(1, len(G) - 1))  # arc position of each detection's projection
        near = {int(ids[q]) for q in range(len(p)) if d[q] <= 2.0}
        interior = [int(ids[q]) for q in range(len(p)) if d[q] <= 1.5 and 2.0 < s[q] < arc - 2.0 and int(ids[q]) not in (A, B)]
        rec.update(interrupting=len(interior), interrupting_unmatched=sum(c not in matched for c in interior))
        # chain of accepted edges from A to B through detections near the path
        prev, queue = {A: None}, deque([A])
        while queue and B not in prev:
            x = queue.popleft()
            for y in adj.get(x, ()):
                if y not in prev and (y == B or y in near):
                    prev[y] = x
                    queue.append(y)
        if B in prev:
            n, x = 0, prev[B]
            while x != A:
                n, x = n + 1, prev[x]
            out.append(dict(rec, kind='broken, all pieces accepted', chain_nodes=n))
        elif interior:
            out.append(dict(rec, kind='broken, a piece missing'))
        else:
            k = cand.get((min(A, B), max(A, B)))
            kind = 'not proposed' if k is None else ('traced, rejected' if row['traced'][k] else 'straight fallback, rejected')
            out.append(dict(rec, kind=kind))
    return out


def summarise(edges):
    out = {}
    for lo, hi in BINS:
        s = [e for e in edges if lo <= e['arc'] < hi]
        if not s:
            continue
        kinds = Counter(e['kind'] for e in s)
        entry = dict(edges=len(s))
        for k, n in kinds.most_common():
            x = [e for e in s if e['kind'] == k]
            entry[k] = dict(share=n / len(s), coverage_mean=float(np.mean([e['coverage'] for e in x])),
                            coverage_over_0p8=float(np.mean([e['coverage'] >= 0.8 for e in x])))
            if k.startswith('broken'):
                entry[k].update(interrupting_mean=float(np.mean([e['interrupting'] for e in x])),
                                interrupting_unmatched_share=float(sum(e['interrupting_unmatched'] for e in x) / max(1, sum(e['interrupting'] for e in x))),
                                chain_nodes_mean=float(np.mean([e['chain_nodes'] for e in x])))
        out[lab(lo, hi)] = entry
    return out


def contract(row, node_t, edge_t, max_score=None, min_cos=None):
    """Remove interior degree-2 nodes of the accepted graph and join their two edges. Returns (new row, contractions)."""
    if row['edge_scores'] is None:
        return row, 0
    keep, acc = accepted_graph(row, node_t, edge_t)
    ns, es = row['node_scores'].copy(), row['edge_scores'].copy()
    pairs, curves, traced = [tuple(map(int, pr)) for pr in row['pairs']], list(row['curves']), list(row['traced'])
    es = list(es)
    vox = row['nodes'] * 64
    face = np.minimum(vox + 0.5, 63.5 - vox).min(1)
    geometry = {}  # accepted edge -> curve from i to j (normalised)

    def curve(i, j):
        key = (min(i, j), max(i, j))
        c = geometry.get(key)
        if c is None:
            c = oriented(row, acc[key], key[0])
        return c if key[0] == i else c[::-1]

    adj = {}
    for (i, j) in acc:
        adj.setdefault(i, set()).add(j), adj.setdefault(j, set()).add(i)
    done, changed = 0, True
    while changed:
        changed = False
        for c in sorted(adj, key=lambda q: ns[q]):
            if len(adj.get(c, ())) != 2 or face[c] <= 2.0 or (max_score is not None and ns[c] >= max_score):
                continue
            a, b = sorted(adj[c])
            if b in adj[a]:
                continue
            ca, cb = curve(a, c), curve(c, b)
            if min_cos is not None:
                t1, t2 = ca[-1] - ca[-6], cb[5] - cb[0]
                if float(t1 @ t2) / max(1e-9, np.linalg.norm(t1) * np.linalg.norm(t2)) < min_cos:
                    continue
            for x in (a, b):
                key = (min(x, c), max(x, c))
                es[acc.pop(key)] = 0.0
                geometry.pop(key, None)
                adj[x].discard(c)
            del adj[c]
            ns[c] = -1.0
            adj[a].add(b), adj[b].add(a)
            pairs.append((a, b))
            curves.append(resample(np.concatenate([ca, cb[1:]])).astype(np.float32))
            es.append(1.0)
            traced.append(True)
            acc[(a, b)] = len(pairs) - 1
            geometry[(a, b)] = curves[-1]
            done, changed = done + 1, True
            break
    if not done:
        return row, 0
    return dict(row, node_scores=ns, edge_scores=np.asarray(es, np.float32), pairs=np.asarray(pairs, int),
                curves=np.stack(curves), traced=np.asarray(traced, bool)), done


def recovered_long(rows, node_t, edge_t):
    edges = [e for r in rows for e in classify(r, node_t, edge_t)]
    return {lab(lo, hi): float(np.mean([e['kind'] == 'recovered' for e in edges if lo <= e['arc'] < hi] or [np.nan])) for lo, hi in BINS}


def main(entry, run):
    d = RUNS / entry / run
    final = json.loads((d / 'final.json').read_text())
    node_t, edge_t = final['thresholds']['node'], final['thresholds']['edge']
    rows = pickle.loads((d / 'final-validation-rows.pkl').read_bytes())
    out = dict(entry=entry, run=run, thresholds=dict(node=node_t, edge=edge_t), validation_patches=len(rows))
    edges = [e for r in rows for e in classify(r, node_t, edge_t)]
    out['long_edges'] = summarise(edges)
    for b, v in out['long_edges'].items():
        print(f"LONG {entry} {b} (n={v['edges']}): " + ' | '.join(
            f"{k}: {x['share']:.2f} (coverage {x['coverage_mean']:.2f}" + (f", {x['chain_nodes_mean']:.1f} nodes in chain" if k == 'broken, all pieces accepted' else '')
            + (f", {x['interrupting_mean']:.1f} on path, {x['interrupting_unmatched_share']:.2f} unmatched" if k == 'broken, a piece missing' else '') + ')'
            for k, x in v.items() if k != 'edges'), flush=True)
    keys = ('node_f1', 'node_precision', 'node_recall', 'edge_f1', 'edge_precision', 'edge_recall', 'beta0_mae', 'beta1_mae')
    out['contraction'] = {}
    base = D.evaluate(rows, node_t, edge_t, geometry=False)
    out['contraction']['none'] = dict({k: base[k] for k in keys}, recovered=recovered_long(rows, node_t, edge_t))
    print('CONTRACT', entry, 'none', json.dumps({k: round(base[k], 4) for k in keys}), out['contraction']['none']['recovered'], flush=True)
    for name, kw in (('all', {}), ('score<0.5', dict(max_score=0.5)), ('score<0.7', dict(max_score=0.7)), ('angle<60', dict(min_cos=0.5)),
                     ('angle<60, score<0.7', dict(min_cos=0.5, max_score=0.7)), ('angle<30', dict(min_cos=0.866))):
        new, count = zip(*[contract(r, node_t, edge_t, **kw) for r in rows])
        res = D.evaluate(list(new), node_t, edge_t, geometry=False)
        out['contraction'][name] = dict({k: res[k] for k in keys}, contractions_per_patch=float(np.mean(count)), recovered=recovered_long(new, node_t, edge_t))
        print('CONTRACT', entry, name, json.dumps({k: round(res[k], 4) for k in keys}), f'{np.mean(count):.2f} per patch', out['contraction'][name]['recovered'], flush=True)
    dst = ROOT / 'experiments/long_edges_20261111'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f'{entry}-{run}.json').write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
