#!/usr/bin/env python3
"""Stitching patch graphs into one graph per volume, using the overlap between patches (12 Nov 2026, CPU).

Setting. The validation patches of a subject form a regular grid of 64^3 patches whose starts are 36-38 voxels
apart, so neighbours overlap by about 27 voxels. Every patch has a predicted graph (saved validation rows of a traced
run) and a GT graph (the simplified cache).

Method 'core' (the overlap idea of Blayney et al. 2024, Bezier graphs, with the node merging of Vesselmerger,
Prabhakar et al. 2024):
  1. Each patch keeps only its core: the box bounded, on each axis, by the mid-planes of its overlaps with the two
     neighbouring grid positions (the volume border where there is none). Cores tile the volume without overlap.
  2. Nodes outside the core are dropped. An edge that crosses the core boundary is cut at the plane; the cut point is
     a new 'cut node'. The same vessel is cut at the same plane in the neighbouring patch, so the cut node appears
     twice, once per patch.
  3. On every plane, cut nodes coming from the two sides are matched one-to-one (Hungarian; cost = distance in mm,
     plus --direction-penalty when the two cut edges do not continue each other) and pairs with cost <= --tolerance
     are merged at their mean position. A merged cut node has two edges and is not a node of the graph: its two edges
     become one.
  4. Cleaning (optional): the simplification rule of the GT at 2 mm (scripts/simplify_graph_caches_20261110.simplify)
     with every predicted node protected: unmatched cut stubs shorter than 2 mm go, two predictions of one junction on
     either side of a plane merge, isolated nodes go (as the post-processing of Blayney et al.).
Baseline 'union' (the aggregation of Blayney et al. applied directly, no cutting): patches are added one by one; the
  nodes of the new patch are matched (Hungarian, <= tolerance) with the nodes already in the volume graph, matched
  pairs are averaged, everything else is added. With a 27-voxel overlap the same vessel piece is then present twice.
'matching' = 'hungarian' (one-to-one) or 'greedy' (every cross-patch pair closer than the tolerance is merged, the
  dilated-sphere overlap rule of Vesselmerger).

Graphs stitched: 'gt' (the GT patch graphs: tests the stitching alone; --jitter-mm shifts every patch graph by an
  independent Gaussian offset to imitate disagreement between patches) and 'pred' (a run's validation rows at its own
  thresholds).
Volume GT: the subject's full graph (vascular_graph_sources), simplified at 2 mm with the same rule as the patches.
Scores per volume graph: node F1 (one-to-one within 2 mm), edge F1 (both ends matched to the ends of a GT edge),
  centreline precision / recall at 1.5 mm, components and loops.

usage: stitch_volume_20261112.py <plan entry> [<run>] [--subjects N]
Writes experiments/stitching_20261112/<entry>[-<run>].json

Whole volumes (added 12 Nov 2026): with --rows-dir DIR --grid overlap|thin|tile0 the predicted graphs come from
scripts/predict_volume_grid_20261112.py (a prediction at EVERY grid position, not only the stored validation patches;
grid starts and affine from DIR/<grid>/meta.json) and only the prediction is stitched, at --node-threshold /
--edge-threshold (default: the run's own), for the --subjects first patients of --split. Configurations: overlap and
thin: 'core, hungarian 2 mm', 'core, greedy 2 mm (sphere overlap)', 'union, hungarian 2 mm (no cutting)' (no rescue:
it lowered F1 everywhere); tile0 (zero overlap, the padding beyond the volume dropped): 'face, hungarian 2 mm' /
'face, greedy 2 mm' (Vesselmerger's matching of the nodes within 3 voxels of a shared face) and 'core, hungarian 2 mm'
(on tile0 it never merges: a no-merging reference). Writes experiments/stitching_20261112/<entry>-<run>-<grid>[-<tag>].json
(pooled results and per-subject scores).
--select-thresholds (--split train only): pooled cleaned volume edge F1 (ties: node F1) of 'core, hungarian 2 mm'
(tile0: 'face, hungarian 2 mm') over the node thresholds present in the rows and edge thresholds 0.50..0.95; writes
<entry>-<run>-<grid>-thresholds.json.
"""
import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from simplify_graph_caches_20261110 import Dense, simplify  # noqa: E402

PLAN = ROOT / 'experiments/multidataset_continuous/plan.json'
RUNS = Path(os.environ.get('SCRATCH', '/tmp')) / 'experiments/gnbm/dense_graph_20261030'
SIZE = 64
DUMP = Path(os.environ.get('SCRATCH', '/tmp')) / 'experiments/gnbm/stitching_20261112'


# ------------------------------------------------------------------------------------------------ patch graphs
def gt_patch(cache):
    """(nodes voxels in the patch, keep flags, edges [(a, b, polyline)]) of a simplified GT cache."""
    with np.load(cache) as z:
        nodes, ed, fp, off = z['nodes'] * SIZE, z['edges'].reshape(-1, 2), z['full_paths'] * SIZE, z['path_offsets']
        ins = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(nodes), bool)
    edges = []
    for k, (a, b) in enumerate(ed):
        p = fp[off[k]:off[k + 1]]
        if np.linalg.norm(p[0] - nodes[a]) > np.linalg.norm(p[-1] - nodes[a]):
            p = p[::-1]
        edges.append((int(a), int(b), p.astype(float)))
    return nodes.astype(float), ~ins, edges


def pred_patch(row, node_t, edge_t):
    nodes = row['nodes'] * SIZE
    keep = row['node_scores'] >= node_t
    edges = []
    if row['edge_scores'] is not None:
        for k, (a, b) in enumerate(row['pairs']):
            if keep[a] and keep[b] and row['edge_scores'][k] >= edge_t:
                p = np.concatenate([row['nodes'][a][None], row['curves'][k][1:-1], row['nodes'][b][None]]) * SIZE
                edges.append((int(a), int(b), p.astype(float)))
    return nodes.astype(float), keep, edges


def densify(p, step=1.0):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.r_[0., np.cumsum(seg)]
    if cum[-1] <= 0:
        return p[:1]
    t = np.unique(np.r_[np.linspace(0, cum[-1], max(2, int(np.ceil(cum[-1] / step)) + 1)), cum])
    return np.stack([np.interp(t, cum, p[:, k]) for k in range(3)], 1)


# ------------------------------------------------------------------------------------------------ stitching
class Volume(Dense):
    def merge(self, keep, other):
        """Unify two vertices at their mean position."""
        self.pos[keep] = (self.pos[keep] + self.pos[other]) / 2
        for n in list(self.adj[other]):
            if n != keep:
                self.link(keep, n)
        self.remove(other)

    def graph(self):
        """(node positions mm, [(u, v, polyline mm)]) of the current graph; nodes = anchors."""
        branches, rings = self.branches()
        ids = sorted(self.anchors())
        index = {k: i for i, k in enumerate(ids)}
        edges = [(index[a], index[b], np.asarray([self.pos[k] for k in path])) for a, b, path in branches]
        for ring in rings:  # a closed loop without any node: keep it as an edge on an extra node
            index[ring[0]] = len(ids)
            ids.append(ring[0])
            edges.append((index[ring[0]], index[ring[0]], np.asarray([self.pos[k] for k in ring + ring[:1]])))
        return np.asarray([self.pos[k] for k in ids]).reshape(-1, 3), edges


def match_pairs(a, b, tol, extra=None, greedy=False):
    """Index pairs (i, j) of positions a, b (mm) to merge. extra: additive cost matrix."""
    if not len(a) or not len(b):
        return []
    cost = np.linalg.norm(a[:, None] - b[None], axis=-1)
    if extra is not None:
        cost = cost + extra
    if greedy:
        return [(int(i), int(j)) for i, j in zip(*np.nonzero(cost <= tol))]
    r, c = linear_sum_assignment(np.where(cost <= tol, cost, 1e6))
    return [(int(i), int(j)) for i, j in zip(r, c) if cost[i, j] <= tol]


def core_bounds(grid):
    """Per axis: start -> (lower, upper) plane of the core of the grid position starting there."""
    out = []
    for s in grid:
        mids = [(s[i] + SIZE + s[i + 1]) / 2 for i in range(len(s) - 1)]
        out.append({st: ((mids[i - 1] if i else -np.inf), (mids[i] if i < len(mids) else np.inf)) for i, st in enumerate(s)})
    return out


def clip(vox, keep, edges, lo, hi):
    """Part of a patch graph (volume voxels) inside the box [lo, hi): the kept nodes inside, and the edge pieces
    [(end, end, points)] with end = ('node', index) or ('cut', (axis, plane, side), outward direction, point)."""
    inside = [int(i) for i in np.flatnonzero(keep) if np.all(vox[i] >= lo) and np.all(vox[i] < hi)]
    pieces = []
    for a, b, p in edges:
        flags = np.all(p >= lo, axis=1) & np.all(p < hi, axis=1)
        i = 0
        while i < len(p):
            if not flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(p) and flags[j + 1]:
                j += 1
            ends = []
            for idx, out in ((i, i - 1), (j, j + 1)):
                if 0 <= out < len(p):  # the run ends where the edge crosses the box boundary
                    pin, pout = p[idx], p[out]
                    best = (2.0, None)
                    for k in range(3):
                        for bound, side in ((lo[k], 'lo'), (hi[k], 'hi')):
                            if np.isfinite(bound) and ((side == 'lo' and pout[k] < bound) or (side == 'hi' and pout[k] >= bound)):
                                t = (bound - pin[k]) / (pout[k] - pin[k]) if pout[k] != pin[k] else 0.0
                                if t < best[0]:
                                    best = (t, (k, float(bound), side))
                    t, face = best
                    ends.append(('cut', face, pout - pin, pin + min(max(t, 0.0), 1.0) * (pout - pin)))
                else:
                    ends.append(('node', a if idx == 0 else b))
            pts = p[i:j + 1]
            if ends[0][0] == 'node':
                pts = pts[1:]
            if ends[1][0] == 'node':
                pts = pts[:-1]
            pieces.append((ends[0], ends[1], pts))
            i = j + 1
    return inside, pieces


def stitch_core(patches, grid, A, tol, direction_penalty, greedy, rescue=False, face_band=0.0, extent=None):
    """patches: [(start, nodes, keep, edges)] in patch voxels. grid: per axis sorted starts. A: voxel -> mm (3x3).
    rescue: after the first matching, pieces that another patch draws inside a core and that the stitched graph does
    not contain (less than half of the piece within 1.5 mm of it) are added, and the cut points are matched again.
    face_band > 0 (zero-overlap tilings, Vesselmerger): a node within this many voxels of a core plane is treated as a
    cut node of that plane, i.e. the nodes detected at the faces of adjacent patches are what gets matched.
    extent (per axis, volume voxels; e.g. a tiling padded beyond the volume): every core is also bounded above by it, so
    nodes and edge points beyond it are dropped; an edge leaving the volume ends at that plane (a plain end, never
    matched, not protected from cleaning)."""
    g = Volume()
    cuts = defaultdict(lambda: dict(lo=[], hi=[]))  # (axis, plane) -> side -> [(vertex, outward direction mm)]
    bounds = core_bounds(grid)
    stats = Counter()
    dense_patches = [(np.asarray(start), nodes + np.asarray(start), keep, [(x, y, densify(poly + np.asarray(start))) for x, y, poly in edges])
                     for start, nodes, keep, edges in patches]
    outer = set()  # (axis, plane) of the extent planes
    if extent is not None:
        outer = {(k, float(extent[k])) for k in range(3)}

    def box(start):
        lo = np.array([bounds[k][int(start[k])][0] for k in range(3)])
        hi = np.array([bounds[k][int(start[k])][1] for k in range(3)])
        if extent is not None:
            hi = np.minimum(hi, np.asarray(extent, float))
        return lo, hi

    def add(vox, inside, pieces, lo, hi, nodes_by_index, protect=True):
        vertex = dict(nodes_by_index)
        for i in inside:
            if i in vertex:
                continue
            face = None
            if face_band:
                for k in range(3):
                    for bound, side in ((lo[k], 'lo'), (hi[k], 'hi')):
                        if np.isfinite(bound) and (k, float(bound)) not in outer and abs(vox[i][k] - bound) <= face_band:
                            face = (k, float(bound), side)
            if face is None:
                vertex[i] = g.add(vox[i] @ A.T, boundary=protect)
            else:  # a node detected at the patch face: a cut node of that plane
                vertex[i] = g.add(vox[i] @ A.T)
                d = np.zeros(3)
                d[face[0]] = 1.0 if face[2] == 'hi' else -1.0
                cuts[face[:2]][face[2]].append((vertex[i], d @ A.T))
                stats['cut_nodes'] += 1
        for e0, e1, pts in pieces:
            ends = []
            for e, fallback in ((e0, pts[0] if len(pts) else None), (e1, pts[-1] if len(pts) else None)):
                if e[0] == 'cut' and e[1][:2] in outer:  # the edge leaves the volume: a plain end there
                    ends.append(g.add(e[3] @ A.T))
                    stats['volume_exits'] += 1
                elif e[0] == 'cut':
                    v = g.add(e[3] @ A.T)
                    d = e[2] @ A.T
                    cuts[e[1][:2]][e[1][2]].append((v, d / max(1e-9, np.linalg.norm(d))))
                    stats['cut_nodes'] += 1
                    ends.append(v)
                else:
                    if e[1] not in vertex:  # e.g. an inserted loop anchor of the GT, or a node on the boundary by rounding
                        vertex[e[1]] = g.add(vox[e[1]] @ A.T)
                    ends.append(vertex[e[1]])
            g.chain(ends[0], [q @ A.T for q in pts], ends[1])
        return vertex

    def match_planes():
        for face, sides in cuts.items():
            hi_side = [(v, d) for v, d in sides['hi'] if v in g.adj and v not in merged]
            lo_side = [(v, d) for v, d in sides['lo'] if v in g.adj and v not in merged]
            a = np.asarray([g.pos[v] for v, _ in hi_side]).reshape(-1, 3)
            b = np.asarray([g.pos[v] for v, _ in lo_side]).reshape(-1, 3)
            extra = None
            if direction_penalty and len(a) and len(b):
                da, db = np.asarray([d for _, d in hi_side]), np.asarray([d for _, d in lo_side])
                extra = direction_penalty * ((da @ db.T) > 0)  # the two outward directions must point against each other
            for i, j in match_pairs(a, b, tol, extra, greedy):
                u, v = hi_side[i][0], lo_side[j][0]
                if u in g.adj and v in g.adj and u != v:
                    g.merge(u, v)
                    merged.update((u, v))
                    stats['merged_pairs'] += 1

    merged = set()
    for start, vox, keep, edges in dense_patches:
        lo, hi = box(start)
        add(vox, *clip(vox, keep, edges, lo, hi), lo, hi, {})
    match_planes()
    if rescue:
        starts = {tuple(int(x) for x in st) for st, *_ in dense_patches} | {(d, h, w) for d in grid[0] for h in grid[1] for w in grid[2]}
        covered = cKDTree(np.asarray(g.pos)[sorted(g.adj)])
        added = np.zeros((0, 3))
        for start, vox, keep, edges in dense_patches:
            fresh = []
            for other in starts:
                if other == tuple(int(x) for x in start) or any(abs(other[k] - start[k]) >= SIZE for k in range(3)):
                    continue  # the patch's own core, or a core its box does not reach
                lo, hi = box(other)
                inside, pieces = clip(vox, keep, edges, lo, hi)
                keep_pieces = []
                for e0, e1, pts in pieces:
                    q = np.asarray([e[3] for e in (e0, e1) if e[0] == 'cut'] + list(pts)).reshape(-1, 3) @ A.T
                    if not len(q):
                        continue
                    near = covered.query(q)[0] <= 1.5
                    if len(added):
                        near |= cKDTree(added).query(q)[0] <= 1.5
                    if near.mean() < 0.5:
                        keep_pieces.append((e0, e1, pts))
                        fresh.append(q)
                if not keep_pieces:
                    continue
                # ends that are nodes: reuse a node already in the volume within the tolerance, else add the node
                index = {}
                used = sorted({e[1] for e0, e1, _ in keep_pieces for e in (e0, e1) if e[0] == 'node'})
                anchors = [v for v in g.boundary if v in g.adj]
                if used and anchors:
                    tree = cKDTree(np.asarray([g.pos[v] for v in anchors]))
                    for i in used:
                        dist, j = tree.query(vox[i] @ A.T)
                        if dist <= tol:
                            index[i] = anchors[j]
                add(vox, [i for i in used if i not in index and keep[i]], keep_pieces, lo, hi, index)
                stats['rescued_pieces'] += len(keep_pieces)
            if fresh:
                added = np.concatenate([added] + fresh)
        match_planes()
    open_cuts = [v for sides in cuts.values() for side in sides.values() for v, _ in side if v in g.adj and v not in merged]
    stats['unmatched_cuts'] = len(open_cuts)
    g.open_cuts = open_cuts
    g.core_bounds = bounds
    return g, stats


def stitch_union(patches, A, tol, greedy):
    """Aggregation without cutting: nodes of each new patch are matched with the nodes already in the volume."""
    g = Volume()
    stats = Counter()
    for start, nodes, keep, edges in patches:
        vox = nodes + np.asarray(start)
        new = [int(i) for i in np.flatnonzero(keep)]
        mm = vox[new] @ A.T if new else np.zeros((0, 3))
        old = [k for k in g.boundary]
        lo, hi = (np.asarray(start) @ A.T), (np.asarray(start) + SIZE) @ A.T
        box_lo, box_hi = np.minimum(lo, hi) - tol, np.maximum(lo, hi) + tol
        old = [k for k in old if np.all(g.pos[k] >= box_lo) and np.all(g.pos[k] <= box_hi)]
        pairs = dict((i, old[j]) for i, j in match_pairs(mm, np.asarray([g.pos[k] for k in old]).reshape(-1, 3), tol, None, greedy)) if not greedy else {}
        if greedy:
            for i, j in match_pairs(mm, np.asarray([g.pos[k] for k in old]).reshape(-1, 3), tol, None, True):
                pairs.setdefault(i, old[j])
        vertex = {}
        for q, i in enumerate(new):
            if q in pairs:
                v = pairs[q]
                g.pos[v] = (g.pos[v] + mm[q]) / 2
                vertex[i] = v
                stats['merged_pairs'] += 1
            else:
                vertex[i] = g.add(mm[q], boundary=True)
        for a, b, poly in edges:
            p = densify(poly + np.asarray(start))
            for node, q in ((a, p[0]), (b, p[-1])):
                if node not in vertex:  # e.g. an inserted loop anchor of the GT: a plain point
                    vertex[node] = g.add(q @ A.T)
            g.chain(vertex[a], [q @ A.T for q in p[1:-1]], vertex[b])
    return g, stats


# ------------------------------------------------------------------------------------------------ evaluation
def sample(edges, step=0.5):
    pts = [densify(p, step) for _, _, p in edges if len(p) > 1]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def node_matching(p, q, tol=2.0):
    """One-to-one matching within tol (optimal inside every connected group of candidate pairs)."""
    if not len(p) or not len(q):
        return {}
    pairs = cKDTree(p).query_ball_tree(cKDTree(q), tol)
    rows = [i for i, js in enumerate(pairs) for _ in js]
    cols = [j for js in pairs for j in js]
    if not rows:
        return {}
    n = len(p) + len(q)
    comp = connected_components(coo_matrix((np.ones(len(rows)), (rows, [len(p) + j for j in cols])), shape=(n, n)), directed=False)[1]
    out = {}
    groups = defaultdict(lambda: (set(), set()))
    for i, j in zip(rows, cols):
        groups[comp[i]][0].add(i), groups[comp[i]][1].add(j)
    for ps, qs in groups.values():
        ps, qs = sorted(ps), sorted(qs)
        d = np.linalg.norm(p[ps][:, None] - q[qs][None], axis=-1)
        r, c = linear_sum_assignment(np.where(d <= tol, d, 1e6))
        out.update({ps[i]: qs[j] for i, j in zip(r, c) if d[i, j] <= tol})
    return out


def betti(n, edges):
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, _ in edges:
        parent[root(a)] = root(b)
    b0 = len({root(i) for i in range(n)})
    return b0, len(edges) - n + b0


def score(pred, gt):
    (pn, pe), (gn, ge) = pred, gt
    m = node_matching(pn, gn)
    pool = Counter(tuple(sorted((a, b))) for a, b, _ in ge)
    tp = 0
    for a, b, _ in pe:
        if a in m and b in m:
            key = tuple(sorted((m[a], m[b])))
            if pool[key] > 0:
                pool[key] -= 1
                tp += 1
    ps, gs = sample(pe), sample(ge)
    p_ok = int((cKDTree(gs).query(ps)[0] <= 1.5).sum()) if len(ps) and len(gs) else 0
    g_ok = int((cKDTree(ps).query(gs)[0] <= 1.5).sum()) if len(ps) and len(gs) else 0
    pb, gb = betti(len(pn), pe), betti(len(gn), ge)
    return dict(node_tp=len(m), nodes=len(pn), gt_nodes=len(gn), edge_tp=tp, edges=len(pe), gt_edges=len(ge), cl_pred=len(ps), cl_pred_ok=p_ok,
                cl_gt=len(gs), cl_gt_ok=g_ok, beta0=pb[0], beta1=pb[1], gt_beta0=gb[0], gt_beta1=gb[1])


def pooled(rows):
    t = {k: sum(r[k] for r in rows) for k in rows[0]}
    f = lambda tp, a, b: 2 * tp / max(1, a + b)  # noqa: E731
    return dict(node_precision=t['node_tp'] / max(1, t['nodes']), node_recall=t['node_tp'] / max(1, t['gt_nodes']), node_f1=f(t['node_tp'], t['nodes'], t['gt_nodes']),
                edge_precision=t['edge_tp'] / max(1, t['edges']), edge_recall=t['edge_tp'] / max(1, t['gt_edges']), edge_f1=f(t['edge_tp'], t['edges'], t['gt_edges']),
                centreline_precision=t['cl_pred_ok'] / max(1, t['cl_pred']), centreline_recall=t['cl_gt_ok'] / max(1, t['cl_gt']),
                nodes_per_volume=t['nodes'] / len(rows), gt_nodes_per_volume=t['gt_nodes'] / len(rows), edges_per_volume=t['edges'] / len(rows),
                gt_edges_per_volume=t['gt_edges'] / len(rows),
                beta0_per_volume=t['beta0'] / len(rows), gt_beta0_per_volume=t['gt_beta0'] / len(rows), beta1_per_volume=t['beta1'] / len(rows),
                gt_beta1_per_volume=t['gt_beta1'] / len(rows),
                beta0_abs_error=float(np.mean([abs(r['beta0'] - r['gt_beta0']) for r in rows])), beta1_abs_error=float(np.mean([abs(r['beta1'] - r['gt_beta1']) for r in rows])))


def volume_gt_dense(source_root, patient):
    """(Volume graph of the subject's full source graph in mm, A, SourceGraph) before any simplification."""
    import nibabel as nib
    from scripts.audit_synthetic_mri_grid import SourceGraph
    affine = nib.load(str(source_root / 'seg' / f'{patient}.nii.gz')).affine
    A = affine[:3, :3]
    directory = source_root / 'graphs'
    directory = directory / 'adaptive' / patient if (directory / 'adaptive').is_dir() else directory / patient
    full = SourceGraph.from_directory(directory).transformed(np.linalg.inv(affine))
    g = Volume()
    ids = {int(k): g.add(full.nodes[int(k)] @ A.T) for k in full.node_ids}
    for geo, poly in zip(full.edge_geometries, full.edge_polylines):
        g.chain(ids[int(geo.node1)], [q @ A.T for q in poly[1:-1]], ids[int(geo.node2)])
    return g, A, full


def volume_gt(source_root, patient, limit):
    """Volume GT simplified at `limit` mm. limit=None: the untouched GT (no simplification call at all, so isolated
    nodes stay too); only the contraction to nodes / edges of Volume.graph() applies (degree-2 vertices are not nodes)."""
    g, A, _ = volume_gt_dense(source_root, patient)
    if limit is not None:
        simplify(g, limit)
    return g.graph(), A


_ONE = None


def _call(patient):
    return _ONE(patient)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entry')
    ap.add_argument('run', nargs='?')
    ap.add_argument('--subjects', type=int, default=0)
    ap.add_argument('--scale-mm', type=float, default=2.0, help='GT simplification scale and cleaning scale')
    ap.add_argument('--dump', action='store_true', help='Save the stitched graphs of the subjects processed (for the figures)')
    ap.add_argument('--dump-config', default='core, hungarian 2 mm', help='Which stitching the dump holds')
    ap.add_argument('--only', nargs='*', help='Run only these stitching configurations (names as printed)')
    ap.add_argument('--tag', default='', help='Suffix of the results file')
    # whole-volume predictions (scripts/predict_volume_grid_20261112.py); without --rows-dir nothing below is used
    ap.add_argument('--rows-dir', type=Path, help='Rows of predict_volume_grid_20261112.py (<dir>/<grid>/...): stitch every grid position')
    ap.add_argument('--grid', choices=('overlap', 'thin', 'tile0'), help='With --rows-dir: which grid')
    ap.add_argument('--node-threshold', type=float, help="With --rows-dir: node threshold (default: the run's own)")
    ap.add_argument('--edge-threshold', type=float, help="With --rows-dir: edge threshold (default: the run's own)")
    ap.add_argument('--gt-limit-mm', type=float,
                    help='With --rows-dir: simplification scale of the volume GT only (default: --scale-mm; 0 = the untouched GT). '
                         'The prediction is always cleaned at --scale-mm')
    ap.add_argument('--thresholds-from', type=Path, help='With --rows-dir: node / edge thresholds = "selected" of a --select-thresholds file')
    ap.add_argument('--split', choices=('val', 'train'), default='val', help='With --rows-dir: subjects of this split')
    ap.add_argument('--select-thresholds', action='store_true',
                    help="With --rows-dir --split train: choose node / edge thresholds by the pooled volume edge F1 of 'core, hungarian 2 mm' "
                         "(tile0: 'face, hungarian 2 mm')")
    a = ap.parse_args()
    if a.rows_dir is not None:
        return rows_main(a)
    plan = json.loads(PLAN.read_text())['datasets'][a.entry]
    root = Path(plan['root'])
    manifest = json.loads((root / 'simplify_manifest.json').read_text())
    original = Path(manifest['source'])  # the cache the simplified one was made from
    source_root = Path(json.loads((original / 'generation_config.json').read_text())['source_root'])
    index = {r['sample_id']: r for r in csv.DictReader((original / 'patch_index.csv').open())}
    by = defaultdict(list)
    for i in plan['val_ids']:
        by[index[i]['patient_id']].append(i)
    patients = sorted(by)[:a.subjects] if a.subjects else sorted(by)
    rows, thr = {}, None
    if a.run:
        final = json.loads((RUNS / a.entry / a.run / 'final.json').read_text())
        thr = final['thresholds']
        rows = {r['sample_id']: r for r in pickle.loads((RUNS / a.entry / a.run / 'final-validation-rows.pkl').read_bytes())}
    configs = [('core, hungarian 2 mm', dict(mode='core', tol=2.0, direction_penalty=0.0, greedy=False)),
               ('core, hungarian 1 mm', dict(mode='core', tol=1.0, direction_penalty=0.0, greedy=False)),
               ('core, hungarian 3 mm', dict(mode='core', tol=3.0, direction_penalty=0.0, greedy=False)),
               ('core, hungarian 2 mm + direction', dict(mode='core', tol=2.0, direction_penalty=2.0, greedy=False)),
               ('core, greedy 2 mm (sphere overlap)', dict(mode='core', tol=2.0, direction_penalty=0.0, greedy=True)),
               ('union, hungarian 2 mm (no cutting)', dict(mode='union', tol=2.0, greedy=False)),
               ('core, hungarian 2 mm + rescue', dict(mode='core', tol=2.0, direction_penalty=0.0, greedy=False, rescue=True))]
    if a.only:
        configs = [c for c in configs if c[0] in a.only]
    sources = [('gt', 0.0), ('gt', 0.5), ('gt', 1.0)] + ([('pred', 0.0)] if a.run else [])
    per = defaultdict(list)
    counts = defaultdict(Counter)

    def one(patient):
        rng = np.random.default_rng(int(patient))
        per, counts, dump = defaultdict(list), defaultdict(Counter), {}
        gt, A = volume_gt(source_root, patient, a.scale_mm)
        starts = {i: tuple(int(index[i][k]) for k in ('start_d', 'start_h', 'start_w')) for i in by[patient]}
        all_rows = [r for r in index.values() if r['patient_id'] == patient]
        grid = [sorted({int(r[k]) for r in all_rows}) for k in ('start_d', 'start_h', 'start_w')]
        for source, jitter in sources:
            patches = []
            for i in by[patient]:
                if source == 'gt':
                    nodes, keep, edges = gt_patch(root / 'val' / 'spline' / f'{i}.npz')
                    if jitter:
                        shift = rng.normal(0, jitter, 3) @ np.linalg.inv(A).T  # the same mm offset for the whole patch graph
                        nodes, edges = nodes + shift, [(x, y, p + shift) for x, y, p in edges]
                else:
                    nodes, keep, edges = pred_patch(rows[i], thr['node'], thr['edge'])
                patches.append((starts[i], nodes, keep, edges))
            for name, cfg in configs:
                if jitter and not (cfg['mode'] == 'core' and not cfg['greedy'] and not cfg['direction_penalty'] and not cfg.get('rescue')):
                    continue  # jittered GT: only the tolerance of the core method
                if cfg['mode'] == 'core':
                    g, st = stitch_core(patches, grid, A, cfg['tol'], cfg['direction_penalty'], cfg['greedy'], cfg.get('rescue', False),
                                        cfg.get('face_band', 0.0))
                else:
                    g, st = stitch_union(patches, A, cfg['tol'], cfg['greedy'])
                key = (f'{source}' + (f', jitter {jitter:g} mm' if jitter else ''), name)
                per[key + ('raw',)].append(score(g.graph(), gt))
                simplify(g, a.scale_mm)
                per[key + ('cleaned',)].append(score(g.graph(), gt))
                counts[key].update(st)
                if a.dump and not jitter and name == a.dump_config:  # graphs for the figures
                    dump[source] = g.graph()
                    dump[source + '_open_cuts'] = np.asarray([g.pos[v] for v in getattr(g, 'open_cuts', []) if v in g.adj and len(g.adj[v]) == 1]).reshape(-1, 3)
                    dump[source + '_score'] = per[key + ('cleaned',)][-1]
        if a.dump:
            dump.update(patient=patient, entry=a.entry, run=a.run, A=A, gt_volume=gt, grid=grid, thresholds=thr,
                        raw=str(source_root / 'raw' / f'{patient}.nii.gz'))
            DUMP.mkdir(parents=True, exist_ok=True)
            (DUMP / f'{a.entry}-{patient}.pkl').write_bytes(pickle.dumps(dump))
        print(f'{patient}: {len(by[patient])} patches, GT {len(gt[0])} nodes / {len(gt[1])} edges', flush=True)
        return dict(per), dict(counts)

    import multiprocessing as mp
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    global _ONE
    _ONE = one
    with mp.get_context('fork').Pool(min(workers, len(patients))) as pool:
        for p_, c_ in pool.imap_unordered(_call, patients):
            for k, v in p_.items():
                per[k] += v
            for k, v in c_.items():
                counts[k].update(v)
    out = dict(entry=a.entry, run=a.run, thresholds=thr, subjects=len(patients), scale_mm=a.scale_mm, results={})
    for (source, name, stage), rs in per.items():
        res = pooled(rs)
        res.update({k: v / len(rs) for k, v in counts[(source, name)].items()})
        out['results'][f'{source} | {name} | {stage}'] = res
        print(f"STITCH {a.entry} {source} | {name} | {stage}: node P/R/F1 {res['node_precision']:.3f} {res['node_recall']:.3f} {res['node_f1']:.3f} | "
              f"edge P/R/F1 {res['edge_precision']:.3f} {res['edge_recall']:.3f} {res['edge_f1']:.3f} | centreline P/R {res['centreline_precision']:.3f} {res['centreline_recall']:.3f} | "
              f"components {res['beta0_per_volume']:.0f} (GT {res['gt_beta0_per_volume']:.0f}) loops {res['beta1_per_volume']:.0f} (GT {res['gt_beta1_per_volume']:.0f}) | "
              f"nodes {res['nodes_per_volume']:.0f} (GT {res['gt_nodes_per_volume']:.0f})", flush=True)
    dst = ROOT / 'experiments/stitching_20261112'
    dst.mkdir(parents=True, exist_ok=True)
    if a.dump:
        return  # a partial run for the figures: do not overwrite the full results
    (dst / ((f'{a.entry}-{a.run}' if a.run else a.entry) + (f'-{a.tag}' if a.tag else '') + '.json')).write_text(json.dumps(out, indent=1, default=float) + '\n')


# ------------------------------------------------------------------------------------------------ whole-volume rows
# Rows of scripts/predict_volume_grid_20261112.py: a prediction at EVERY position of a grid ('overlap' = the dataset's
# own grid, 'thin' = maximum stride 64, 'tile0' = zero overlap on a volume zero-padded to a multiple of 64). Only the
# predicted graph is stitched (no per-patch GT here); it is scored against the same volume GT (volume_gt).
FACE = dict(mode='core', tol=2.0, direction_penalty=0.0, greedy=False, face_band=3.0)
ROWS_CONFIGS = {
    # '+ rescue' is left out: it lowered node and edge F1 on all three entries (jobs 96417/96419/96421, stored patches)
    'overlap': [('core, hungarian 2 mm', dict(mode='core', tol=2.0, direction_penalty=0.0, greedy=False)),
                ('core, greedy 2 mm (sphere overlap)', dict(mode='core', tol=2.0, direction_penalty=0.0, greedy=True)),
                ('union, hungarian 2 mm (no cutting)', dict(mode='union', tol=2.0, greedy=False))],
    # zero overlap: the cores are the patch boxes. Vesselmerger's face matching: nodes within 3 voxels of a shared
    # face are matched (one-to-one, or every pair within the tolerance); 'core' (cut points of crossing edges only) as
    # the reference.
    'tile0': [('face, hungarian 2 mm', FACE),
              ('face, greedy 2 mm', dict(FACE, greedy=True)),
              ('core, hungarian 2 mm', dict(mode='core', tol=2.0, direction_penalty=0.0, greedy=False))],
}
ROWS_CONFIGS['thin'] = ROWS_CONFIGS['overlap']
SELECT_EDGE_THRESHOLDS = [round(t, 2) for t in np.arange(0.50, 0.951, 0.05)]
# on tile0 'core' never merges (one side of a face only ever yields cut points), so selection there uses face matching
SELECT_CONFIG = {'overlap': ROWS_CONFIGS['overlap'][0], 'thin': ROWS_CONFIGS['thin'][0], 'tile0': ROWS_CONFIGS['tile0'][0]}


def load_meta(grid_dir):
    """Per patient meta of a grid: meta.json, completed by the per-patient files (written first by the predictor)."""
    meta = json.loads((grid_dir / 'meta.json').read_text())['patients'] if (grid_dir / 'meta.json').exists() else {}
    for f in sorted((grid_dir / 'meta').glob('*.json')):
        m = json.loads(f.read_text())
        meta[m['patient']] = m
    return meta


def rows_patches(rows, node_t, edge_t):
    """rows: a rows file (path) or the loaded rows."""
    if isinstance(rows, (str, Path)):
        rows = pickle.loads(Path(rows).read_bytes())
    return [(tuple(int(x) for x in r['start']), *pred_patch(r, node_t, edge_t)) for r in rows]


def stitch_rows(patches, meta, grid, A, cfg):
    extent = np.asarray(meta['shape'], float) - 0.5 + 1e-3 if grid == 'tile0' else None  # drop what lies in the padding
    if cfg['mode'] == 'core':
        return stitch_core(patches, meta['starts'], A, cfg['tol'], cfg['direction_penalty'], cfg['greedy'], cfg.get('rescue', False),
                           cfg.get('face_band', 0.0), extent=extent)
    assert extent is None, 'union stitching is not defined on a padded tiling'
    return stitch_union(patches, A, cfg['tol'], cfg['greedy'])


def print_stitch(entry, key, res):
    print(f"STITCH {entry} {key}: node P/R/F1 {res['node_precision']:.3f} {res['node_recall']:.3f} {res['node_f1']:.3f} | "
          f"edge P/R/F1 {res['edge_precision']:.3f} {res['edge_recall']:.3f} {res['edge_f1']:.3f} | centreline P/R {res['centreline_precision']:.3f} {res['centreline_recall']:.3f} | "
          f"components {res['beta0_per_volume']:.0f} (GT {res['gt_beta0_per_volume']:.0f}) loops {res['beta1_per_volume']:.0f} (GT {res['gt_beta1_per_volume']:.0f}) | "
          f"nodes {res['nodes_per_volume']:.0f} (GT {res['gt_nodes_per_volume']:.0f})", flush=True)


def rows_main(a):
    global _ONE
    import multiprocessing as mp
    if not a.run or not a.grid:
        raise SystemExit('--rows-dir needs <run> and --grid')
    plan = json.loads(PLAN.read_text())['datasets'][a.entry]
    root = Path(plan['root'])
    original = Path(json.loads((root / 'simplify_manifest.json').read_text())['source'])
    source_root = Path(json.loads((original / 'generation_config.json').read_text())['source_root'])
    index = {r['sample_id']: r for r in csv.DictReader((original / 'patch_index.csv').open())}
    ids = plan['val_ids' if a.split == 'val' else 'train_ids']
    split_patients = sorted({index[i]['patient_id'] for i in ids})
    patients = split_patients[:a.subjects] if a.subjects else split_patients
    own = json.loads((RUNS / a.entry / a.run / 'final.json').read_text())['thresholds']
    grid_dir = a.rows_dir / a.grid
    meta = load_meta(grid_dir)
    missing = [p for p in patients if p not in meta]
    if missing:
        raise SystemExit(f'no rows for patients {missing} in {grid_dir}')
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    dst = ROOT / 'experiments/stitching_20261112'
    dst.mkdir(parents=True, exist_ok=True)
    print(f'{a.entry} {a.run} grid {a.grid}: split {a.split}, {len(patients)} patients {patients}', flush=True)

    # volume GT once per patient (simplified at --gt-limit-mm, default --scale-mm; 0 = untouched)
    gt_limit = a.scale_mm if a.gt_limit_mm is None else (None if a.gt_limit_mm == 0 else a.gt_limit_mm)

    def gt_of(patient):
        return volume_gt(source_root, patient, gt_limit)
    _ONE = gt_of
    with mp.get_context('fork').Pool(min(workers, len(patients))) as pool:
        gts = dict(zip(patients, pool.map(_call, patients)))
    for p in patients:
        if not np.allclose(gts[p][1], np.asarray(meta[p]['A']), atol=1e-4):
            raise SystemExit(f'affine mismatch for {p}: seg {gts[p][1].tolist()} vs rows meta {meta[p]["A"]}')

    if a.select_thresholds:
        if a.split != 'train':
            raise SystemExit('thresholds are only ever chosen on training subjects: use --split train')
        bad = [p for p in patients if meta[p].get('split') != 'train' or any(index[i]['split'] != 'train' for i in ids if index[i]['patient_id'] == p)]
        if bad:
            raise SystemExit(f'not training subjects: {bad}')
        node_ts = sorted(set.intersection(*[set(meta[p]['node_thresholds']) for p in patients]))
        node_ts = [t for t in node_ts if all((grid_dir / f'{p}-node{t:g}.pkl').exists() for p in patients)]
        name, cfg = SELECT_CONFIG[a.grid]

        def one_select(task):
            patient, node_t = task
            gt, A = gts[patient]
            out = {}
            rows = pickle.loads((grid_dir / f'{patient}-node{node_t:g}.pkl').read_bytes())
            for edge_t in SELECT_EDGE_THRESHOLDS:
                g, _ = stitch_rows(rows_patches(rows, node_t, edge_t), meta[patient], a.grid, A, cfg)
                simplify(g, a.scale_mm)
                out[edge_t] = score(g.graph(), gt)
            print(f'{patient} node {node_t:g}: done', flush=True)
            return node_t, out
        _ONE = one_select
        tasks = [(p, t) for p in patients for t in node_ts]
        per = defaultdict(list)
        with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
            for node_t, out in pool.imap_unordered(_call, tasks):
                for edge_t, s in out.items():
                    per[(node_t, edge_t)].append(s)
        table = []
        for (node_t, edge_t), rs in sorted(per.items()):
            assert len(rs) == len(patients)
            table.append(dict(node=node_t, edge=edge_t, **pooled(rs)))
        best = max(table, key=lambda r: (r['edge_f1'], r['node_f1']))
        for r in sorted(table, key=lambda r: (-r['edge_f1'], -r['node_f1'])):
            print(f"SELECT {a.entry} {a.grid} node {r['node']:.2f} edge {r['edge']:.2f}: edge P/R/F1 {r['edge_precision']:.3f} {r['edge_recall']:.3f} {r['edge_f1']:.3f} | "
                  f"node P/R/F1 {r['node_precision']:.3f} {r['node_recall']:.3f} {r['node_f1']:.3f} | centreline P/R {r['centreline_precision']:.3f} {r['centreline_recall']:.3f}", flush=True)
        print(f"SELECTED {a.entry} {a.grid}: node {best['node']:g} edge {best['edge']:g} (own {own['node']} / {own['edge']}); train edge F1 {best['edge_f1']:.3f}, node F1 {best['node_f1']:.3f}", flush=True)
        result = dict(entry=a.entry, run=a.run, grid=a.grid, split='train', subjects=patients, configuration=name, stage='cleaned',
                      criterion='pooled volume edge F1 (ties: node F1), training subjects only', node_thresholds=node_ts,
                      edge_thresholds=SELECT_EDGE_THRESHOLDS, run_thresholds=own, selected=dict(node=best['node'], edge=best['edge']),
                      selected_row=best, table=table, scale_mm=a.scale_mm, rows_dir=str(a.rows_dir))
        (dst / f'{a.entry}-{a.run}-{a.grid}-thresholds.json').write_text(json.dumps(result, indent=1, default=float) + '\n')
        return

    node_t = own['node'] if a.node_threshold is None else a.node_threshold
    edge_t = own['edge'] if a.edge_threshold is None else a.edge_threshold
    selection = None
    if a.thresholds_from is not None:
        selection = json.loads(a.thresholds_from.read_text())
        if selection['split'] != 'train' or selection['entry'] != a.entry or selection['run'] != a.run or selection['grid'] != a.grid:
            raise SystemExit(f'{a.thresholds_from}: not a training-subject selection for {a.entry} {a.run} {a.grid}')
        node_t, edge_t = selection['selected']['node'], selection['selected']['edge']
        print(f'thresholds from {a.thresholds_from}: node {node_t:g} edge {edge_t:g} (chosen on training subjects {selection["subjects"]})', flush=True)
    missing = [p for p in patients if not (grid_dir / f'{p}-node{node_t:g}.pkl').exists()]
    if missing:
        raise SystemExit(f'no rows at node threshold {node_t:g} for {missing} in {grid_dir}')
    configs = ROWS_CONFIGS[a.grid]
    if a.only:
        configs = [c for c in configs if c[0] in a.only]

    def one(patient):
        gt, A = gts[patient]
        per, counts, dump = defaultdict(list), defaultdict(Counter), {}
        patches = rows_patches(grid_dir / f'{patient}-node{node_t:g}.pkl', node_t, edge_t)
        for name, cfg in configs:
            g, st = stitch_rows(patches, meta[patient], a.grid, A, cfg)
            key = ('pred', name)
            per[key + ('raw',)].append(score(g.graph(), gt))
            simplify(g, a.scale_mm)
            per[key + ('cleaned',)].append(score(g.graph(), gt))
            counts[key].update(st)
            if a.dump and name == a.dump_config:
                dump.update(pred=g.graph(), pred_score=per[key + ('cleaned',)][-1],
                            pred_open_cuts=np.asarray([g.pos[v] for v in getattr(g, 'open_cuts', []) if v in g.adj and len(g.adj[v]) == 1]).reshape(-1, 3))
        if a.dump:
            dump.update(patient=patient, entry=a.entry, run=a.run, A=A, gt_volume=gt, grid=meta[patient]['starts'], grid_name=a.grid,
                        thresholds=dict(node=node_t, edge=edge_t), raw=str(source_root / 'raw' / f'{patient}.nii.gz'))
            DUMP.mkdir(parents=True, exist_ok=True)
            (DUMP / f'{a.entry}-{a.grid}-{patient}.pkl').write_bytes(pickle.dumps(dump))
        print(f'{patient}: {len(patches)} grid positions ({a.grid}), GT {len(gt[0])} nodes / {len(gt[1])} edges', flush=True)
        return patient, dict(per), dict(counts)

    _ONE = one
    per, counts, per_subject = defaultdict(list), defaultdict(Counter), {}
    with mp.get_context('fork').Pool(min(workers, len(patients))) as pool:
        for patient, p_, c_ in pool.imap_unordered(_call, patients):
            per_subject[patient] = {' | '.join(k): v[0] for k, v in p_.items()}
            for k, v in p_.items():
                per[k] += v
            for k, v in c_.items():
                counts[k].update(v)
    out = dict(entry=a.entry, run=a.run, grid=a.grid, split=a.split, thresholds=dict(node=node_t, edge=edge_t), run_thresholds=own,
               subjects=len(patients), patients=patients, scale_mm=a.scale_mm, gt_limit_mm=gt_limit, rows_dir=str(a.rows_dir), results={},
               per_subject={p: per_subject[p] for p in sorted(per_subject)},
               thresholds_from=str(a.thresholds_from) if a.thresholds_from else None)
    for (source, name, stage), rs in per.items():
        res = pooled(rs)
        res.update({k: v / len(rs) for k, v in counts[(source, name)].items()})
        out['results'][f'{source} | {name} | {stage}'] = res
        print_stitch(a.entry, f'{source} | {name} | {stage}', res)
    if a.dump:
        return
    (dst / (f'{a.entry}-{a.run}-{a.grid}' + (f'-{a.tag}' if a.tag else '') + '.json')).write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
