"""Loop-aware graph scoring for targets with inserted breaking-point anchors (5 Nov 2026).

Loop patches are kept in the training targets by subdividing parallel branches, self-loops and anchor-free rings
with degree-2 'breaking point' anchors (data/spline_targets.subdivide_loops for 3D; for 2D the same rule in
native2d/targets.crop_graph since 5 Nov 2026, earlier 2D targets used native2d/targets._simple_graph). These anchors
have no anatomical meaning, so evaluation must not reward or penalise them:
- a prediction matched to an inserted GT anchor is neither a true nor a false node;
- edges are scored on the original branches: GT chains through inserted anchors are merged back (a parallel pair
  is two A-B edges, a self-loop an A-A edge), and so are predicted chains through degree-2 predictions matched to
  inserted anchors; edges are then matched as a multiset of endpoint pairs;
- rings made only of inserted anchors have no endpoint and are left out of node/edge counts (they stay in Curve
  F1, Branch F1 and beta_1, which do not depend on the subdivision).
On a target without inserted anchors every count equals the plain protocol.
"""
from collections import Counter

import numpy as np


def _merge_chains(edges, through, count):
    """Multigraph over the nodes not in `through` (all of degree 2): each chain via `through` nodes is one edge.

    Returns [(u, v, [edge indices])] with u <= v (u == v: a loop back to the same node) and the number of rings
    made only of `through` nodes, which have no endpoint and are dropped."""
    adjacency = [[] for _ in range(count)]
    for k, (a, b) in enumerate(edges):
        adjacency[a].append((b, k))
        adjacency[b].append((a, k))
    seen, out, rings = set(), [], 0

    def walk(cur, members):
        while cur in through:
            options = [(n, j) for n, j in adjacency[cur] if j not in seen]
            if not options:
                break
            cur, j = options[0]
            seen.add(j)
            members.append(j)
        return cur

    for k, (a, b) in enumerate(edges):
        if k in seen or (a in through and b in through):
            continue
        start, other = (a, b) if a not in through else (b, a)
        seen.add(k)
        members = [k]
        end = walk(other, members)
        out.append((min(start, end), max(start, end), members))
    for k, (a, b) in enumerate(edges):  # what is left are rings of `through` nodes only
        if k not in seen:
            seen.add(k)
            walk(b, [k])
            rings += 1
    return out, rings


def loop_aware_counts(n_pred, pred_edges, mapping, gt_edges, n_gt, inserted):
    """Core counts. pred_edges: accepted predicted edges (local ids 0..n_pred-1); mapping: pred id -> GT id (one-to-one
    match against all GT nodes, inserted included); inserted: per-GT-node mask."""
    inserted = np.asarray(inserted, bool)
    real = set(np.flatnonzero(~inserted).tolist())
    gt_merged, gt_rings = _merge_chains([tuple(map(int, e)) for e in np.asarray(gt_edges).reshape(-1, 2)],
                                        set(np.flatnonzero(inserted).tolist()), n_gt)
    gt_pairs = Counter((u, v) for u, v, _ in gt_merged)
    edges = [tuple(map(int, e)) for e in np.asarray(pred_edges).reshape(-1, 2)]
    degree = np.bincount(np.asarray(edges, int).ravel(), minlength=n_pred) if edges else np.zeros(n_pred, int)
    ignored = {q for q, g in mapping.items() if inserted[g]}
    through = {q for q in ignored if degree[q] == 2}
    real_map = {q: g for q, g in mapping.items() if g in real}
    pred_merged, pred_rings = _merge_chains(edges, through, n_pred)
    pool, tp, kinds = Counter(gt_pairs), 0, {}
    labels, types = np.zeros(len(edges), bool), np.array(['UU'] * len(edges), dtype='<U2')
    for u, v, members in pred_merged:
        key = tuple(sorted((real_map[u], real_map[v]))) if u in real_map and v in real_map else None
        kind = 'MM' if key is not None else 'MU' if (u in real_map or v in real_map) else 'UU'
        hit = key is not None and pool[key] > 0
        if hit:
            pool[key] -= 1
        tp += hit
        kinds.setdefault(kind, [0, 0])[0 if hit else 1] += 1
        labels[members] = hit
        types[members] = kind
    recovered = set(real_map.values())
    return dict(node_tp=len(real_map), pred_nodes=n_pred - len(ignored), node_fp=n_pred - len(mapping),
                gt_nodes=len(real), tp=tp, pred_edges=len(pred_merged), gt_edges=sum(gt_pairs.values()),
                recoverable=sum(c for (u, v), c in gt_pairs.items() if u in recovered and v in recovered),
                kinds=kinds, labels=labels, types=types, ignored=len(ignored), dropped_gt_rings=gt_rings,
                dropped_pred_rings=pred_rings, degree=degree)


def inserted_from_degree(n_nodes, edges):
    """Fallback for 2D targets prepared before 5 Nov 2026, which store no inserted flag (recontracted targets from
    native2d/targets.crop_graph store an explicit `inserted` mask; use that instead). A breaking point has degree 2
    AND lies on a cycle (it splits a parallel pair, a self-loop or a ring). Degree alone is not enough: the 5 Nov 2026 audit found ordinary degree-2 nodes off
    any cycle in AV-WIDE, REAVER and plants crops (experiments/loop_subdivision_20261105/audit_2d_degree_two.json),
    which must stay ordinary GT nodes. A non-inserted degree-2 node that happens to lie on a cycle would still be
    mislabelled; the 3D caches avoid this with an explicit `inserted` flag."""
    edges = [tuple(map(int, e)) for e in np.asarray(edges, int).reshape(-1, 2)]
    out = np.zeros(n_nodes, bool)
    if not edges:
        return out
    degree = np.bincount(np.asarray(edges).ravel(), minlength=n_nodes)
    for v in np.flatnonzero(degree == 2):
        nb = [q if p == v else p for p, q in edges if v in (p, q)]
        a, b = nb[0], nb[1]
        if a == b:  # both edges to one neighbour: a 2-cycle
            out[v] = True
            continue
        adjacency = {}
        for p, q in edges:
            if v not in (p, q):
                adjacency.setdefault(p, []).append(q)
                adjacency.setdefault(q, []).append(p)
        seen, stack = {a}, [a]
        while stack:
            x = stack.pop()
            if x == b:
                out[v] = True
                break
            for y in adjacency.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    return out
