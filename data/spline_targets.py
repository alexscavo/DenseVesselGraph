"""Full-path crop and endpoint-constrained oracle fit for supplied vessel guides."""
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import BSpline


def _point_at(poly, s):
    """Point at arc length s along poly, and the vertex index after which it lies."""
    d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    k = int(np.clip(np.searchsorted(d, s, side='right') - 1, 0, len(poly) - 2))
    t = 0. if d[k + 1] - d[k] <= 0 else (s - d[k]) / (d[k + 1] - d[k])
    return poly[k] + t * (poly[k + 1] - poly[k]), k


def _split(poly, fractions):
    """Split poly at arc-length fractions; returns the pieces and the cut points (geometry unchanged)."""
    total = np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()
    pieces, cuts, start, rest = [], [], 0., poly
    for f in fractions:
        p, k = _point_at(rest, f * total - start)
        pieces.append(np.vstack([rest[:k + 1], p[None]]))
        rest = np.vstack([p[None], rest[k + 1:]])
        start = f * total
        cuts.append(p)
    pieces.append(rest)
    return pieces, cuts


def subdivide_loops(graph, next_id):
    """Make a contracted multigraph simple by inserting degree-2 'breaking point' anchors (training encoding).

    The adjacency target stores one edge per unordered node pair and no self-loop, so:
    - a branch repeating an existing pair (parallel branch) is split at its arc-length midpoint;
    - a self-loop on an anchor is split at 1/3 and 2/3 of its length;
    - a ring with no junction/end (contractor's virtual centroid anchor, negative id) gets three anchors on the
      curve itself (its start point and 1/3, 2/3), replacing the virtual centroid, which is not on the curve.
    Every polyline point is kept, so centreline geometry, length, beta_0 and beta_1 are unchanged. Inserted
    anchors get ids >= next_id and are returned so evaluation can treat them as 'don't care' (they have no
    anatomical meaning). Returns (anchors, branches, inserted_ids).
    """
    from metrics.branch_connectivity import Branch
    anchors = {k: v.copy() for k, v in graph.anchors.items() if k >= 0}
    inserted, branches, used = set(), [], set()

    def new(point):
        nonlocal next_id
        anchors[next_id] = np.asarray(point, float).copy()
        inserted.add(next_id)
        next_id += 1
        return next_id - 1

    def chain(ids, pieces):
        for a, b, piece in zip(ids[:-1], ids[1:], pieces):
            branches.append(Branch(a, b, piece))
            used.add(tuple(sorted((a, b))))

    for branch in graph.branches:
        a, b, poly = branch.start, branch.end, np.asarray(branch.polyline, float)
        if np.linalg.norm(np.diff(poly, axis=0), axis=1).sum() < 1e-10:
            raise ValueError('degenerate_branch')
        if a == b:
            if a < 0:  # ring without anchors: start point of the polyline becomes an anchor too
                a = new(poly[0])
            pieces, cuts = _split(poly, (1 / 3, 2 / 3))
            chain([a, new(cuts[0]), new(cuts[1]), a], pieces)
        elif tuple(sorted((a, b))) in used:
            pieces, cuts = _split(poly, (0.5,))
            chain([a, new(cuts[0]), b], pieces)
        else:
            chain([a, b], [poly])
    return anchors, tuple(branches), inserted


def contract_spline_graph(nodes, edges, max_anchors=120, *, closed_components=0, preserve_nodes=(), subdivide=False):
    """Return the contracted target and its whole-patch exclusion reason.

    subdivide=False (default, all caches before 5 Nov 2026): patches whose contracted graph has parallel branches,
    self-loops or anchor-free rings are excluded. subdivide=True: such branches get inserted degree-2 anchors
    (subdivide_loops) and the patch is kept; the returned graph then has an `inserted` set of anchor ids.
    """
    from metrics.branch_connectivity import contract_degree_two
    links = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if subdivide:
        from types import SimpleNamespace as NS
        nodes = np.asarray(nodes, float)
        # Zero-length dense self-loop records (a clipped closed part with no interior sample) carry no geometry.
        links = links[links[:, 0] != links[:, 1]]
        # A repeated dense record is a distinct parallel segment: give the repeat a midpoint sample (a degree-2
        # dense node, contracted away) so the contractor sees two embedded paths instead of an ambiguous duplicate.
        seen, extra, kept = set(), [], []
        for a, b in links.tolist():
            key = tuple(sorted((a, b)))
            if key in seen:
                m = len(nodes) + len(extra)
                extra.append((nodes[a] + nodes[b]) / 2)
                kept.extend(((a, m), (m, b)))
            else:
                seen.add(key)
                kept.append((a, b))
        if extra:
            nodes = np.vstack([nodes, np.asarray(extra)])
        links = np.asarray(kept, dtype=np.int64).reshape(-1, 2)
        graph = contract_degree_two(nodes, links, preserve_nodes=preserve_nodes)
        try:
            anchors, branches, inserted = subdivide_loops(graph, len(nodes))
        except ValueError as error:
            return None, str(error)
        out = NS(anchors=anchors, branches=branches, inserted=inserted, betti_0=graph.betti_0, betti_1=graph.betti_1)
        return out, ('object_capacity' if len(anchors) > max_anchors else None)
    # The generic contractor only accepts simple embedded graphs. Check the
    # current decoder's unsupported cases before invoking it: never deduplicate
    # records, which could silently erase a parallel vessel branch.
    if closed_components or np.any(links[:, 0] == links[:, 1]):
        return None, 'closed_branch'
    source_pairs = {tuple(sorted(pair)) for pair in links.tolist()}
    if len(source_pairs) != len(links):
        return None, 'parallel_branches'
    graph = contract_degree_two(nodes, edges, preserve_nodes=preserve_nodes)
    pairs = [tuple(sorted((b.start, b.end))) for b in graph.branches]
    reason = ('closed_branch' if any(a == b or a < 0 for a, b in pairs) else
              'parallel_branches' if len(set(pairs)) != len(pairs) else
              'object_capacity' if len(graph.anchors) > max_anchors else None)
    return graph, reason


def crop_dense_paths(source, bounds, *, retain_tangent_contacts=False, boundary_policy='exact'):
    if boundary_policy not in ('exact', 'last_inside_sample'):
        raise ValueError('Unknown boundary policy')
    ids = [int(i) for i in source.node_ids if source._inside(source.nodes[int(i)], bounds)]
    positions = [source.nodes[i].copy() for i in ids]
    mapping = {i: j for j, i in enumerate(ids)}
    boundary = [False]*len(ids)
    edges, contacts, snapped, fallbacks, closed = [], 0, 0, 0, 0

    def endpoint(point, edge):
        for i in (edge.node1, edge.node2):
            if i in mapping and np.allclose(point, source.nodes[i], atol=1e-7, rtol=0):
                return mapping[i]
        positions.append(point.copy()); boundary.append(True)
        return len(positions)-1

    for k in source._candidate_edge_indices(bounds):
        edge = source.edge_geometries[int(k)]
        polyline = source.edge_polylines[int(k)]
        for part in source._clip_polyline(polyline, bounds):
            if len(part) == 1:
                contacts += 1
                if retain_tangent_contacts:
                    endpoint(part[0], edge)
                continue
            if boundary_policy == 'last_inside_sample':
                # A clipped component retains original samples between its two
                # intersections. Trim only interpolated endpoints, never choose
                # a sample from another component of a re-entering vessel.
                first_original = np.any(np.all(np.isclose(polyline, part[0], atol=1e-8, rtol=0), axis=1))
                last_original = np.any(np.all(np.isclose(polyline, part[-1], atol=1e-8, rtol=0), axis=1))
                first = 0 if first_original else 1
                stop = len(part) if last_original else len(part)-1
                trimmed = part[first:stop]
                if len(trimmed) >= 2 and np.linalg.norm(trimmed[-1]-trimmed[0]) > 1e-10:
                    snapped += int(not first_original)+int(not last_original)
                    part = trimmed
                elif not first_original or not last_original:
                    # Preserve positive-length foreground-guided components when
                    # the sampling is too sparse for two existing endpoints.
                    fallbacks += 1
            left, right = endpoint(part[0], edge), endpoint(part[-1], edge)
            closed += int(left == right)
            chain = [left]
            for point in part[1:-1]:
                chain.append(len(positions)); positions.append(point.copy()); boundary.append(False)
            chain.append(right); edges.extend(zip(chain[:-1], chain[1:]))
    # Retain isolated original nodes: do not silently alter target topology.
    return SimpleNamespace(positions=np.asarray(positions).reshape(-1, 3),
                           edges=np.asarray(edges, dtype=int).reshape(-1, 2),
                           boundary=np.asarray(boundary), tangent_contacts=contacts,
                           snapped_endpoints=snapped, sparse_sample_fallbacks=fallbacks,
                           closed_components=closed)


def resample(points, samples):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance)>1e-10]
    points, distance = points[keep], distance[keep]
    if distance[-1] <= 1e-10:
        raise ValueError('Degenerate guide')
    return np.column_stack([np.interp(np.linspace(0, distance[-1], samples), distance, points[:, i]) for i in range(3)])


def fit_spline(points, controls=8, samples=64):
    knots = np.r_[np.zeros(4), np.arange(1, controls-3)/(controls-3), np.ones(4)]
    guide = resample(points, 257)
    train_basis = BSpline(knots, np.eye(controls), 3)(np.linspace(0, 1, 257))
    target = guide-train_basis[:, :1]*guide[0]-train_basis[:, -1:]*guide[-1]
    internal = np.linalg.lstsq(train_basis[:, 1:-1], target, rcond=None)[0]
    fitted = np.vstack((guide[0], internal, guide[-1]))
    return fitted, BSpline(knots, fitted, 3)(np.linspace(0, 1, samples))
