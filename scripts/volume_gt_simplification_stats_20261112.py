#!/usr/bin/env python3
"""How the 2 mm simplification changes the full-volume GT graph (12 Nov 2026, CPU; validation subjects).

Per validation subject of each plan entry, the volume GT used by stitch_volume_20261112.py:
  source     the subject's source graph (vascular_graph_sources): nodes, edges, self-loop edges;
  untouched  the same graph as a Volume (stitch_volume_20261112.volume_gt with limit None), contracted to nodes / edges
             by Volume.graph(): degree-2 source nodes are not nodes there, isolated source nodes stay; a self-loop edge
             without interior points cannot be represented (Dense.link ignores a == b) and is counted;
  limit0     simplify(g, 0): checked to change nothing but the removal of isolated nodes;
  2 mm       simplify(g, 2.0): operation counts (spur, fragment, junction_merge, junction_to_boundary, tiny_loop,
             isolated_node) and the distance merged nodes moved.
Centreline length = sum of the edge polyline lengths (mm) of Volume.graph().
usage: volume_gt_simplification_stats_20261112.py <entry> ... [--scale-mm 2]
Writes experiments/stitching_20261112/volume-gt-simplification.json
"""
import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from stitch_volume_20261112 import PLAN, betti, simplify, volume_gt_dense  # noqa: E402

_ONE = None


def _call(x):
    return _ONE(x)


def describe(g):
    nodes, edges = g.graph()
    b0, b1 = betti(len(nodes), edges)
    length = float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for _, _, p in edges if len(p) > 1))
    deg = Counter(len(n) for n in g.adj.values())
    return dict(nodes=len(nodes), edges=len(edges), components=b0, loops=b1, length_mm=length, isolated_vertices=deg[0])


def main():
    global _ONE
    ap = argparse.ArgumentParser()
    ap.add_argument('entries', nargs='+')
    ap.add_argument('--scale-mm', type=float, default=2.0)
    a = ap.parse_args()
    out = {}
    for entry in a.entries:
        plan = json.loads(PLAN.read_text())['datasets'][entry]
        original = Path(json.loads((Path(plan['root']) / 'simplify_manifest.json').read_text())['source'])
        source_root = Path(json.loads((original / 'generation_config.json').read_text())['source_root'])
        index = {r['sample_id']: r for r in csv.DictReader((original / 'patch_index.csv').open())}
        patients = sorted({index[i]['patient_id'] for i in plan['val_ids']})

        def one(patient):
            g, A, full = volume_gt_dense(source_root, patient)
            src_deg = Counter()
            self_loops = self_loops_bare = 0
            for geo, poly in zip(full.edge_geometries, full.edge_polylines):
                u, v = int(geo.node1), int(geo.node2)
                src_deg[u] += 1
                src_deg[v] += 1
                if u == v:
                    self_loops += 1
                    self_loops_bare += len(poly) <= 2
            source_length = float(sum(np.linalg.norm(np.diff(np.asarray(p) @ A.T, axis=0), axis=1).sum() for p in full.edge_polylines))
            r = dict(source=dict(nodes=len(full.node_ids), edges=len(full.edge_geometries), self_loop_edges=self_loops,
                                 self_loop_edges_without_interior_points=self_loops_bare, length_mm=source_length,
                                 degree2_nodes=sum(1 for k in full.node_ids if src_deg[int(k)] == 2),
                                 degree0_nodes=sum(1 for k in full.node_ids if src_deg[int(k)] == 0)))
            r['untouched'] = describe(g)
            g0 = copy.deepcopy(g)
            ops0, _ = simplify(g0, 0.0)
            r['limit0'] = dict(describe(g0), ops=dict(ops0))
            g2 = copy.deepcopy(g)
            ops2, moved = simplify(g2, a.scale_mm)
            r['simplified'] = dict(describe(g2), ops=dict(ops2), moved_mm_max=float(max(moved, default=0.0)),
                                   moved_mm_mean=float(np.mean(moved)) if moved else 0.0, merged_nodes=len(moved))
            u, z = r['untouched'], r['limit0']
            r['limit0_only_removes_isolated'] = (set(ops0) <= {'isolated_node'} and z['nodes'] == u['nodes'] - u['isolated_vertices']
                                                 and z['edges'] == u['edges'] and abs(z['length_mm'] - u['length_mm']) < 1e-6)
            return patient, r

        _ONE = one
        res = {}
        with mp.get_context('fork').Pool(min(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')), len(patients))) as pool:
            for p, r in pool.imap_unordered(_call, patients):
                res[p] = r
        tot = defaultdict(float)
        ops = Counter()
        for p in sorted(res):
            r = res[p]
            for stage in ('source', 'untouched', 'simplified'):
                for k, v in r[stage].items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        tot[f'{stage}.{k}'] += v
            ops.update(r['simplified']['ops'])
            print(f"GTSUBJ {entry} {p}: source {r['source']['nodes']} nodes / {r['source']['edges']} edges ({r['source']['degree2_nodes']} degree-2, "
                  f"{r['source']['degree0_nodes']} isolated, {r['source']['self_loop_edges']} self-loops) | untouched {r['untouched']['nodes']} / {r['untouched']['edges']}, "
                  f"{r['untouched']['components']} comp, {r['untouched']['loops']} loops, {r['untouched']['length_mm']:.0f} mm | 2 mm {r['simplified']['nodes']} / "
                  f"{r['simplified']['edges']}, {r['simplified']['components']} comp, {r['simplified']['loops']} loops, {r['simplified']['length_mm']:.0f} mm | "
                  f"ops {dict(r['simplified']['ops'])} | limit0 only removes isolated: {r['limit0_only_removes_isolated']}", flush=True)
        n = len(res)
        summary = {k: v / n for k, v in tot.items()}
        summary['ops_total'] = dict(ops)
        summary['subjects'] = n
        summary['limit0_only_removes_isolated_all'] = all(r['limit0_only_removes_isolated'] for r in res.values())
        print(f"GTSTAT {entry} ({n} validation subjects, per volume): source {summary['source.nodes']:.1f} nodes / {summary['source.edges']:.1f} edges, "
              f"{summary['source.length_mm']:.0f} mm | untouched {summary['untouched.nodes']:.1f} / {summary['untouched.edges']:.1f}, {summary['untouched.components']:.1f} comp, "
              f"{summary['untouched.loops']:.1f} loops, {summary['untouched.length_mm']:.0f} mm, {summary['untouched.isolated_vertices']:.2f} isolated | 2 mm "
              f"{summary['simplified.nodes']:.1f} / {summary['simplified.edges']:.1f}, {summary['simplified.components']:.1f} comp, {summary['simplified.loops']:.1f} loops, "
              f"{summary['simplified.length_mm']:.0f} mm | ops total {dict(ops)} | limit0 only removes isolated (all subjects): {summary['limit0_only_removes_isolated_all']}", flush=True)
        out[entry] = dict(summary=summary, per_subject={p: res[p] for p in sorted(res)})
    dst = ROOT / 'experiments/stitching_20261112'
    (dst / 'volume-gt-simplification.json').write_text(json.dumps(dict(scale_mm=a.scale_mm, entries=out), indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
