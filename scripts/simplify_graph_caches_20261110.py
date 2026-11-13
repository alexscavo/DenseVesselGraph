#!/usr/bin/env python3
"""Simplify the patch GT graphs at a fixed physical scale (10 Nov 2026, journal open problem B1).

Why. In the skeleton-derived GTs (IXI, TopBrain) 37-38% of the edges are shorter than the 2 mm node-matching
tolerance and about half of the nodes sit on such an edge: one-voxel skeleton hairs (a junction and an end 0.5 mm
apart, shorter than the vessel radius), junctions split in two, and stubs where a vessel leaves the patch right
after a junction. Such nodes cannot be told apart at the matching tolerance, the model does not find them (recall
0.38 on IXI against 0.76 for the other nodes, on training patches too), and they hold 57% of all missed IXI nodes.

Rule (one parameter, --min-edge-mm L, the same for every dataset). Repeat on the shortest branch shorter than L:
  - both ends are ends (free end or patch-boundary exit, nothing else attached): delete the branch (fragment / stub);
  - one end is a free end (not on the boundary), the other carries other branches: delete it (spur);
  - one end is a boundary exit with nothing else attached, the other a junction: the junction moves to the exit
    (the exit node stays, the junction's other branches are extended along the stub to it);
  - otherwise (two junctions): merge them at the arc midpoint of the branch; their branches are extended to it;
  - a closed loop shorter than L is deleted.
Then isolated nodes are dropped and the graph is contracted again with the standard code (degree-2 chains merged,
boundary exits kept, loops subdivided with inserted anchors): data.spline_targets.contract_spline_graph.
Extending the branches keeps the centreline geometry; only spurs, fragments and tiny loops lose length (reported).

Original caches are not modified. The output root has the same layout (raw/ and seg/ are symlinks to the same
patches), so it is used through a new plan entry.

usage: simplify_graph_caches_20261110.py --source <cache root> --output <new root> --min-edge-mm 2.0 [--splits train val]
       simplify_graph_caches_20261110.py --plan-entry <name> --from-entry <existing plan entry> --root <new root>
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from data.spline_targets import contract_spline_graph, fit_spline, resample  # noqa: E402

PLAN = ROOT / 'experiments/multidataset_continuous/plan.json'


class Dense:
    """Simple undirected graph over centreline points (positions in mm)."""

    def __init__(self):
        self.pos, self.adj, self.boundary = [], {}, set()

    def add(self, p, boundary=False):
        k = len(self.pos)
        self.pos.append(np.asarray(p, float))
        self.adj[k] = set()
        if boundary:
            self.boundary.add(k)
        return k

    def link(self, a, b):
        if a == b:
            return
        if b in self.adj[a]:  # a second direct link between the same points: keep it distinct with a midpoint
            m = self.add((self.pos[a] + self.pos[b]) / 2)
            self.adj[a].add(m), self.adj[m].update((a, b)), self.adj[b].add(m)
            return
        self.adj[a].add(b), self.adj[b].add(a)

    def chain(self, start, points, end):
        """Link start -> new points -> end."""
        cur = start
        for p in points:
            k = self.add(p)
            self.link(cur, k)
            cur = k
        self.link(cur, end)

    def remove(self, k):
        for n in self.adj.pop(k):
            self.adj[n].discard(k)
        self.boundary.discard(k)

    def anchors(self):
        return {k for k, n in self.adj.items() if len(n) != 2 or k in self.boundary}

    def branches(self):
        """[(a, b, [vertex ids from a to b])] between anchors, and rings without any anchor."""
        anchors, seen, out = self.anchors(), set(), []
        for a in sorted(anchors):
            for n in sorted(self.adj[a]):
                if (a, n) in seen:
                    continue
                path, prev, cur = [a, n], a, n
                while cur not in anchors:
                    nxt = next(x for x in self.adj[cur] if x != prev) if len(self.adj[cur]) == 2 else None
                    prev, cur = cur, nxt
                    path.append(cur)
                seen.add((a, n)), seen.add((path[-1], path[-2]))
                out.append((a, path[-1], path))
        used = anchors | {k for _, _, p in out for k in p}
        rings, left = [], set(self.adj) - used
        while left:
            start = min(left)
            path, prev, cur = [start], None, start
            while True:
                nxt = [x for x in self.adj[cur] if x != prev]
                prev, cur = cur, nxt[0]
                if cur == start:
                    break
                path.append(cur)
            left -= set(path)
            rings.append(path)
        return out, rings

    def arc(self, path, closed=False):
        p = np.asarray([self.pos[k] for k in (path + path[:1] if closed else path)])
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def simplify(g, limit):
    """Returns the operation counts and, for every merged node that survives, how far (mm, along the vessel) the
    furthest original node it replaces was moved ('moved'; merges can chain, so this is measured, not assumed)."""
    ops, moved = Counter(), {}
    while True:
        branches, rings = g.branches()
        for ring in rings:
            if g.arc(ring, closed=True) < limit:
                for k in ring:
                    g.remove(k)
                ops['tiny_loop'] += 1
        short = [(g.arc(p), a, b, p) for a, b, p in branches if all(k in g.adj for k in p)]
        short = [s for s in short if s[0] < limit]
        if not short:
            break
        g_arc, a, b, path = min(short, key=lambda s: (s[0], s[1], s[2]))
        if a == b:  # closed loop on one anchor
            for k in path[1:-1]:
                g.remove(k)
            ops['tiny_loop'] += 1
            continue
        free_a, free_b = len(g.adj[a]) == 1, len(g.adj[b]) == 1
        if free_a and free_b:
            for k in path:
                g.remove(k)
            ops['fragment'] += 1
        elif (free_a and a not in g.boundary) or (free_b and b not in g.boundary):
            end = a if free_a and a not in g.boundary else b
            for k in path[1:-1] + [end]:
                g.remove(k)
            ops['spur'] += 1
        else:
            pts = [g.pos[k] for k in path]
            if a in g.boundary or b in g.boundary:  # the junction moves to the boundary exit
                keep, other = (a, b) if a in g.boundary else (b, a)
                trail = pts[1:] if keep == a else pts[-2::-1]  # from next to keep ... up to `other`
                for n in [x for x in g.adj[other] if x not in path]:
                    g.adj[n].discard(other), g.adj[other].discard(n)
                    g.chain(n, trail[::-1], keep)
                for k in path:
                    if k != keep:
                        g.remove(k)
                moved[keep] = max(moved.get(keep, 0.), moved.pop(other, 0.) + g_arc)
                ops['junction_to_boundary'] += 1
            else:  # two junctions merge at the arc midpoint
                seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                cum = np.r_[0., np.cumsum(seg)]
                half = cum[-1] / 2
                j = int(np.clip(np.searchsorted(cum, half, side='right') - 1, 0, len(pts) - 2))
                t = 0. if seg[j] <= 0 else (half - cum[j]) / seg[j]
                mid = g.add(pts[j] + t * (pts[j + 1] - pts[j]))
                for end, trail in ((a, pts[:j + 1]), (b, pts[:j:-1])):  # points from the end towards the midpoint
                    for n in [x for x in g.adj[end] if x not in path]:
                        g.adj[n].discard(end), g.adj[end].discard(n)
                        g.chain(n, trail, mid)
                for k in path:
                    g.remove(k)
                moved[mid] = max(moved.pop(a, 0.), moved.pop(b, 0.)) + half
                ops['junction_merge'] += 1
    for k in [k for k, n in g.adj.items() if not n]:
        g.remove(k)
        ops['isolated_node'] += 1
    return ops, [v for k, v in moved.items() if k in g.adj]


def betti(n, links):
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in links:
        parent[root(a)] = root(b)
    b0 = len({root(i) for i in range(n)})
    return b0, len(links) - n + b0


def simplify_cache(z, limit):
    """npz-like dict -> (new arrays dict | None, stats)."""
    M = z['metric'].astype(np.float64)
    inverse = np.linalg.inv(M)
    nodes, edges = z['nodes'].astype(np.float64), z['edges'].reshape(-1, 2)
    boundary = z['boundary'].astype(bool)
    off = z['path_offsets']
    g = Dense()
    ids = [g.add(p, boundary=bool(b)) for p, b in zip(nodes @ M.T, boundary)]  # inserted anchors become plain points
    length = 0.
    for k, (a, b) in enumerate(edges):
        path = z['full_paths'][off[k]:off[k + 1]].astype(np.float64) @ M.T
        if np.linalg.norm(path[0] - g.pos[ids[a]]) > np.linalg.norm(path[-1] - g.pos[ids[a]]):
            path = path[::-1]
        g.chain(ids[a], path[1:-1], ids[b])
        length += float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    inserted_before = z['inserted'].astype(bool) if 'inserted' in z else np.zeros(len(nodes), bool)
    stats = dict(nodes_before=int((~inserted_before).sum()), edges_before=int(len(edges) - inserted_before.sum()),
                 length_before=length, betti_before=betti(len(nodes), edges.tolist()))
    ops, moved = simplify(g, limit)
    stats['moved_mm'] = moved
    order = sorted(g.adj)
    remap = {k: i for i, k in enumerate(order)}
    positions = np.asarray([g.pos[k] for k in order]).reshape(-1, 3) @ inverse.T
    links = np.asarray(sorted({(min(remap[a], remap[b]), max(remap[a], remap[b])) for a in order for b in g.adj[a]}), dtype=np.int64).reshape(-1, 2)
    graph, reason = contract_spline_graph(positions, links, max_anchors=256, preserve_nodes=[remap[k] for k in sorted(g.boundary)], subdivide=True)
    if reason:
        return None, dict(stats, reason=reason, ops=dict(ops))
    anchor_ids = sorted(graph.anchors)
    mapping = {k: i for i, k in enumerate(anchor_ids)}
    preserved = {remap[k] for k in g.boundary}
    new_nodes = np.asarray([graph.anchors[i] for i in anchor_ids]).reshape(-1, 3)
    links, controls, curves, guides, paths, offsets = [], [], [], [], [], [0]
    for branch in graph.branches:  # same conventions as prepare_transfer_splines.prepare
        path = branch.polyline @ M.T
        a, b = branch.start, branch.end
        if tuple(path[0]) > tuple(path[-1]):
            path, a, b = path[::-1], b, a
        if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < 1e-10:
            return None, dict(stats, reason='degenerate_branch', ops=dict(ops))
        c, curve = fit_spline(path, 8, 64)
        controls.append(c @ inverse.T), curves.append(curve @ inverse.T), guides.append(resample(path, 64) @ inverse.T)
        links.append((mapping[a], mapping[b])), paths.append(path @ inverse.T), offsets.append(offsets[-1] + len(path))
    new_edges = np.asarray(links, dtype=np.int64).reshape(-1, 2)
    inserted = np.array([k in graph.inserted for k in anchor_ids], dtype=bool)
    out = dict(nodes=new_nodes.astype('float32'), edges=new_edges,
               controls=np.asarray(controls, dtype='float32').reshape(-1, 8, 3), curves=np.asarray(curves, dtype='float32').reshape(-1, 64, 3),
               guides=np.asarray(guides, dtype='float32').reshape(-1, 64, 3), metric=z['metric'].astype('float32'),
               full_paths=(np.concatenate(paths) if paths else np.empty((0, 3))).astype('float32'), path_offsets=np.asarray(offsets),
               boundary=np.array([k in preserved for k in anchor_ids], dtype=bool), inserted=inserted)
    stats.update(nodes_after=int((~inserted).sum()), edges_after=int(len(new_edges) - inserted.sum()), ops=dict(ops),
                 length_after=float(sum(np.linalg.norm(np.diff(p @ M.T, axis=0), axis=1).sum() for p in paths)),
                 betti_after=betti(len(new_nodes), new_edges.tolist()))
    return out, stats


def work(job):
    src, split, name, limit = job
    with np.load(src / split / 'spline' / f'{name}.npz') as z:
        z = {k: z[k] for k in z.files}
    out, stats = simplify_cache(z, limit)
    stats.update(sample_id=name, split=split)
    return out, stats


def write_vtp(path, nodes, edges):
    """Minimal ASCII VTK XML PolyData (points + 2-point lines), read back by pyvista in the loader. Written by hand
    because pyvista cannot be imported on the CPU nodes (jobs 25058, 33564 of 10 Nov 2026)."""
    pts = ' '.join(f'{float(v):.9g}' for v in np.asarray(nodes, np.float32).ravel())
    conn = ' '.join(str(int(v)) for v in np.asarray(edges).ravel())
    offs = ' '.join(str(2 * (k + 1)) for k in range(len(edges)))
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n <PolyData>\n'
        f'  <Piece NumberOfPoints="{len(nodes)}" NumberOfVerts="0" NumberOfLines="{len(edges)}" NumberOfStrips="0" NumberOfPolys="0">\n'
        f'   <Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">{pts}</DataArray></Points>\n'
        f'   <Lines><DataArray type="Int64" Name="connectivity" format="ascii">{conn}</DataArray>'
        f'<DataArray type="Int64" Name="offsets" format="ascii">{offs}</DataArray></Lines>\n'
        '  </Piece>\n </PolyData>\n</VTKFile>\n')


def write(src, dst, split, name, out):
    leaf = dst / split
    for folder, suffix in (('raw', '_data.nii.gz'), ('seg', '_seg.nii.gz')):
        link = leaf / folder / f'{name}{suffix}'
        if not link.is_symlink():
            link.symlink_to((src / split / folder / f'{name}{suffix}').resolve())
    write_vtp(leaf / 'vtp' / f'{name}_graph.vtp', out['nodes'], out['edges'])
    np.savez_compressed(leaf / 'spline' / f'{name}.npz', **out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--min-edge-mm', type=float, default=2.0)
    p.add_argument('--splits', nargs='+', default=['train', 'val'])
    p.add_argument('--limit', type=int, default=0, help='Only the first N patches of each split (test runs)')
    p.add_argument('--dry-run', action='store_true', help='Statistics only, nothing written')
    p.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    p.add_argument('--plan-entry', help='Add this entry to the plan: a copy of --from-entry pointing at --root, without the '
                                        'patches whose simplified graph is empty or was rejected')
    p.add_argument('--from-entry')
    p.add_argument('--drop-empty', action='store_true', help='With --plan-entry: also leave out patches whose simplified graph is empty')
    p.add_argument('--root', type=Path)
    a = p.parse_args()
    if a.plan_entry:
        plan = json.loads(PLAN.read_text())
        entry = dict(plan['datasets'][a.from_entry])
        manifest = json.loads((a.root / 'simplify_manifest.json').read_text())
        bad = set(manifest['rejected']) | (set(manifest['empty']) if a.drop_empty else set())
        for key in ('train_ids', 'val_ids', 'monitor_train', 'monitor_val'):
            if key in entry:
                entry[key] = [i for i in entry[key] if i not in bad]
        entry.update(root=str(a.root), design=f"{a.from_entry} with GT graphs simplified at {manifest['min_edge_mm']} mm "
                     f"(scripts/simplify_graph_caches_20261110.py); same patches, {len(bad)} left out of the id lists (rejected by the contractor{' or empty after simplification' if a.drop_empty else ''})")
        entry.pop('manifest_sha256', None)
        plan['datasets'][a.plan_entry] = entry
        PLAN.write_text(json.dumps(plan, indent=1) + '\n')
        print(a.plan_entry, {k: len(entry[k]) for k in ('train_ids', 'val_ids', 'monitor_train')}, 'left out', len(bad))
        return
    jobs = []
    for split in a.splits:
        names = sorted(f.stem for f in (a.source / split / 'spline').glob('*.npz'))
        for folder in ('raw', 'seg', 'vtp', 'spline'):
            if not a.dry_run:
                (a.output / split / folder).mkdir(parents=True, exist_ok=True)
        jobs += [(a.source, split, n, a.min_edge_mm) for n in (names[:a.limit] if a.limit else names)]
    stats = []
    with mp.get_context('fork').Pool(a.workers) as pool:
        for k, (out, st) in enumerate(pool.imap(work, jobs, chunksize=32)):
            stats.append(st)
            if out is not None and not a.dry_run:
                write(a.source, a.output, st['split'], st['sample_id'], out)
            if (k + 1) % 1000 == 0:
                print(f'{k + 1}/{len(jobs)}', flush=True)
    ok = [s for s in stats if 'reason' not in s]
    ops = Counter()
    for s in ok:
        ops.update(s['ops'])
    tot = lambda k: sum(s[k] for s in ok)
    summary = dict(source=str(a.source), min_edge_mm=a.min_edge_mm, patches=len(stats),
                   rejected=sorted(s['sample_id'] for s in stats if 'reason' in s), rejection_reasons=dict(Counter(s['reason'] for s in stats if 'reason' in s)),
                   empty=sorted(s['sample_id'] for s in ok if s['nodes_after'] == 0 and s['nodes_before'] > 0),
                   nodes_before=tot('nodes_before'), nodes_after=tot('nodes_after'), edges_before=tot('edges_before'), edges_after=tot('edges_after'),
                   centreline_mm_before=tot('length_before'), centreline_mm_after=tot('length_after'), operations=dict(ops),
                   beta0_before=sum(s['betti_before'][0] for s in ok), beta0_after=sum(s['betti_after'][0] for s in ok),
                   beta1_before=sum(s['betti_before'][1] for s in ok), beta1_after=sum(s['betti_after'][1] for s in ok))
    moved = np.array([m for s in ok for m in s['moved_mm']])
    if len(moved):
        summary['merged_node_displacement_mm'] = dict(merged_nodes=int(len(moved)), median=float(np.median(moved)), p95=float(np.percentile(moved, 95)),
                                                      p99=float(np.percentile(moved, 99)), max=float(moved.max()), over_2mm=float(np.mean(moved > 2.0)))
    if a.dry_run:
        print(json.dumps({k: v for k, v in summary.items() if k not in ('rejected', 'empty')}, indent=1, default=float))
        print('rejected', len(summary['rejected']), 'empty', len(summary['empty']))
        return
    (a.output / 'simplify_manifest.json').write_text(json.dumps(dict(summary, per_patch=stats), indent=1, default=float) + '\n')
    for name in ('generation_config.json', 'source_manifest.json'):
        if (a.source / name).exists() and not (a.output / name).exists():
            (a.output / name).write_text((a.source / name).read_text())
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rejected', 'empty')}, indent=1, default=float))
    print('rejected', len(summary['rejected']), 'empty', len(summary['empty']))


if __name__ == '__main__':
    main()
