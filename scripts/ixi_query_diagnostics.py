"""CPU accounting for the IXI P0 audit; diagnostic matches are position-only."""
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from scripts.diagnose_branch_relations import match_nodes


def betti(n, edges):
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edges:
        parent[root(int(a))] = root(int(b))
    b0 = len({root(i) for i in range(n)})
    return b0, len(edges) - n + b0


def nms(nodes_mm, scores, query_ids, radius):
    """Confidence-first physical suppression; query ID breaks exact score ties."""
    kept = []
    for i in sorted(query_ids, key=lambda i: (-float(scores[i]), int(i))):
        if not kept or np.all(np.linalg.norm(nodes_mm[kept] - nodes_mm[i], axis=1) > radius):
            kept.append(int(i))
    return np.array(sorted(kept), dtype=int)


def categorize(nodes, gt, metric, mask, query_ids, tolerance):
    physical = nodes @ metric.T
    truth = gt @ metric.T
    local = match_nodes(physical[query_ids], truth, tolerance)
    mapping = {int(query_ids[i]): j for i, j in local.items()}
    recovered = set(mapping.values())
    shape = np.array(mask.shape)
    vox = nodes * shape
    inside = ((vox >= -.5 - 1e-5) & (vox <= shape - .5 + 1e-5)).all(1)
    fg = np.argwhere(mask) / shape @ metric.T
    distance = cKDTree(fg).query(physical)[0] if len(fg) else np.full(len(nodes), np.inf)
    near = np.linalg.norm(physical[:, None] - truth[None], axis=-1) <= tolerance
    categories = {}
    for i in query_ids:
        if i in mapping:
            category = 'matched'
        elif any(near[i, j] for j in recovered):
            category = 'duplicate_candidate'
        elif near[i].any():
            category = 'near_unrecovered_gt'
        elif inside[i] and distance[i] <= 1.:
            category = 'foreground_supported_extra'
        else:
            category = 'off_vessel_or_outside'
        categories[int(i)] = category
    return mapping, categories, distance, inside


def edge_accounting(query_ids, pairs, scores, selected, mapping, gt_edges, gt_count):
    """Keep query identity (and oracle assignments) fixed through deletions."""
    links = {tuple(sorted(map(int, edge))) for edge in gt_edges}
    if len(links) != len(gt_edges) or any(a == b for a, b in links):
        raise ValueError('Closed/parallel GT requires an explicit multigraph protocol')
    active = np.isin(pairs, query_ids).all(1)
    chosen = active & selected
    labels = np.array([a in mapping and b in mapping and
                       tuple(sorted((mapping[a], mapping[b]))) in links
                       for a, b in pairs], dtype=bool)
    types = np.array(['MM' if a in mapping and b in mapping else
                      'MU' if a in mapping or b in mapping else 'UU'
                      for a, b in pairs])
    tp = int(labels[chosen].sum())
    fp = int(chosen.sum()) - tp
    recovered = set(mapping.values())
    recoverable = sum(a in recovered and b in recovered for a, b in links)
    remap = {int(q): i for i, q in enumerate(query_ids)}
    edges = np.array([[remap[int(a)], remap[int(b)]] for a, b in pairs[chosen]], dtype=int).reshape(-1, 2)
    p0, p1 = betti(len(query_ids), edges)
    g0, g1 = betti(gt_count, gt_edges)
    degree = np.bincount(edges.ravel(), minlength=len(query_ids))
    row = dict(nodes=len(query_ids), gt_nodes=gt_count, node_tp=len(mapping),
               node_fp=len(query_ids)-len(mapping), node_fn=gt_count-len(mapping),
               edges=int(chosen.sum()), gt_edges=len(links), tp=tp, fp=fp, fn=len(links)-tp,
               recoverable_edges=recoverable, missing_endpoint_edges=len(links)-recoverable,
               missing_relation_edges=recoverable-tp, beta0=p0, beta1=p1,
               beta0_error=abs(p0-g0), beta1_error=abs(p1-g1),
               isolates=int((degree == 0).sum()), max_degree=int(degree.max()) if len(degree) else 0,
               degree_histogram=dict(Counter(map(int, degree))))
    for kind in ('MM', 'MU', 'UU'):
        subset = chosen & (types == kind)
        row[kind+'_tp'] = int(labels[subset].sum())
        row[kind+'_fp'] = int((~labels[subset]).sum())
    assert row['tp']+row['fn'] == len(links)
    assert sum(row[k+'_fp'] for k in ('MM', 'MU', 'UU')) == fp
    return row, chosen, labels, types, edges


def summarize(rows):
    count_keys = ['nodes', 'gt_nodes', 'node_tp', 'node_fp', 'node_fn', 'edges', 'gt_edges',
                  'tp', 'fp', 'fn', 'recoverable_edges', 'missing_endpoint_edges',
                  'missing_relation_edges', 'MM_tp', 'MM_fp', 'MU_fp', 'UU_fp']
    out = {k: sum(r[k] for r in rows) for k in count_keys}
    ratio = lambda a, b: a/b if b else None
    out.update(samples=len(rows), precision=ratio(out['tp'], out['tp']+out['fp']),
               recall=ratio(out['tp'], out['gt_edges']),
               f1=ratio(2*out['tp'], 2*out['tp']+out['fp']+out['fn']),
               node_precision=ratio(out['node_tp'], out['nodes']),
               node_recall=ratio(out['node_tp'], out['gt_nodes']),
               node_f1=ratio(2*out['node_tp'], out['nodes']+out['gt_nodes']),
               conditional_recall=ratio(out['tp'], out['recoverable_edges']),
               unmatched_fraction_of_fp=ratio(out['MU_fp']+out['UU_fp'], out['fp']),
               beta0_mae=float(np.mean([r['beta0_error'] for r in rows])),
               beta1_mae=float(np.mean([r['beta1_error'] for r in rows])))
    return out
