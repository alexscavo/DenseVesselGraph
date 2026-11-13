#!/usr/bin/env python3
"""Why are some GT edges not proposed by tracing, and would a route through other nodes' regions follow the vessel?
(11 Nov 2026, GPU, validation, diagnostic only.)

For GT edges >= 10 mm whose two ends are available as nodes and whose pair is NOT a traced contact:
  - what separates the two regions along the GT path: a stretch outside the predicted centreline mask ('gap', at
    least 2 mm of the path), or the region of a third node ('third region'), or neither;
  - 'route': the cheapest chain of traced contacts A - C1 - ... - B (Dijkstra on the contact graph, at most 5 hops),
    with its path made of the traced pieces. Each piece runs through the intermediate node, so a node sitting on a
    side structure gives a detour; 'spliced' removes such detours (where the path comes back within 1.5 voxels of
    itself, the excursion in between is cut out).
    Reported: share of edges with a route, hops, share of intermediate nodes that are unmatched, and the mean distance
    of the path to the GT centreline for the straight chord, the route and the spliced route.
Node sets: the model's own peaks >= its node threshold, and the GT nodes (to explain the fallback-only edges that
remain with perfect nodes).
usage: fallback_route_check_20261111.py <plan entry> <run> [--patches 600]
Writes experiments/long_edges_20261111/routes-<entry>-<run>.json
"""
import argparse
import heapq
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402
from node_miss_diagnosis_20261110 import dense_path  # noqa: E402

RUNS = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
BINS = [(10, 20), (20, 40), (40, 1e9)]
lab = lambda lo, hi: f'{lo}-{hi} mm' if hi < 1e8 else f'>= {lo} mm'


def route(contacts, a, b, max_hops=5):
    """Cheapest chain of contacts from a to b: list of nodes, or None."""
    nbr = {}
    for (i, j), (c, _) in contacts.items():
        nbr.setdefault(i, []).append((j, c)), nbr.setdefault(j, []).append((i, c))
    best, heap = {(a, 0): 0.0}, [(0.0, a, 0, (a,))]
    while heap:
        cost, x, h, path = heapq.heappop(heap)
        if x == b:
            return list(path)
        if h == max_hops:
            continue
        for y, c in nbr.get(x, ()):
            if y in path:
                continue
            key = (y, h + 1)
            if cost + c < best.get(key, np.inf):
                best[key] = cost + c
                heapq.heappush(heap, (cost + c, y, h + 1, path + (y,)))
    return None


def splice(p, tol=1.5, min_skip=6):
    """Cut out excursions: when the polyline (voxels) returns within tol of an earlier point, drop what is in between."""
    out = [p[0]]
    i = 0
    n = len(p)
    while i < n - 1:
        d = np.linalg.norm(p[i + min_skip:] - p[i], axis=1) if i + min_skip < n else np.zeros(0)
        close = np.flatnonzero(d <= tol)
        i = i + min_skip + int(close[-1]) if len(close) else i + 1
        out.append(p[i])
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entry')
    ap.add_argument('run')
    ap.add_argument('--patches', type=int, default=600)
    a = ap.parse_args()
    run_dir = RUNS / a.entry / a.run
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(final)
    node_t = final['thresholds']['node']
    device = 'cuda'
    model = T.TracedDenseGraph(backbone=final.get('backbone', 'unet'), feature_channels=final.get('feature_channels', 32)).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device)['model'])
    model.eval()
    ids = json.loads(PLAN.read_text())['datasets'][a.entry]['val_ids'][:a.patches]
    ds = make_dataset(a.entry, ids, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=8, collate_fn=D.collate)
    records = {'predicted': [], 'gt': []}
    offset = 0
    for images, nodes, edges, geometry in loader:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model.forward_image(images.to(device))
        prob = out[:, 4].float().sigmoid()
        found = D.peaks(out, node_t)
        for name in records:
            seeds = [p for p, _ in found] if name == 'predicted' else [n.to(device) * D.SIZE for n in nodes]
            with torch.no_grad():
                contacts, dist, label = T.traced_pairs(prob, seeds)
                mask = T.geodesic_regions(prob, seeds)[0]
            for b in range(len(images)):
                record = ds.records[offset + b]
                with np.load(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz') as z:
                    M, gn, ed, P, off = z['metric'], z['nodes'], z['edges'].reshape(-1, 2), z['full_paths'] @ z['metric'].T, z['path_offsets']
                    ins = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(gn), bool)
                g, p = gn @ M.T, seeds[b]
                pm = (p / D.SIZE).cpu().numpy() @ M.T
                if name == 'predicted':
                    m = D.match(torch.as_tensor(pm), torch.as_tensor(g)) if len(pm) and len(g) else {}
                else:
                    m = {i: i for i in range(len(g))}
                inv = {gt: q for q, gt in m.items()}
                lab_b, mask_b = label[b].cpu().numpy(), mask[b].cpu().numpy()
                Minv = np.linalg.inv(M)
                vox_mm = float(np.linalg.norm(M, axis=0).mean() / D.SIZE)
                for e, (x, y) in enumerate(ed):
                    if ins[x] or ins[y] or x not in inv or y not in inv:
                        continue
                    A, B = inv[x], inv[y]
                    if (min(A, B), max(A, B)) in contacts[b]:
                        continue
                    G = dense_path(P[off[e]:off[e + 1]])
                    arc = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
                    if arc < BINS[0][0]:
                        continue
                    v = np.clip(np.round(G @ Minv.T * D.SIZE).astype(int), 0, D.SIZE - 1)
                    inside, labs = mask_b[v[:, 0], v[:, 1], v[:, 2]], lab_b[v[:, 0], v[:, 1], v[:, 2]]
                    run, longest = 0, 0
                    for flag in ~inside:
                        run = run + 1 if flag else 0
                        longest = max(longest, run)
                    gap_mm = longest * 0.25
                    third = float(np.mean(inside & (labs >= 0) & (labs != A) & (labs != B)))
                    rec = dict(arc=arc, gap_mm=gap_mm, third_share=third,
                               why='gap' if gap_mm >= 2.0 else 'third region' if third > 0.05 else 'neither')
                    tree = cKDTree(G)
                    chord = np.linspace(pm[A], pm[B], 64)
                    rec['straight'] = float(tree.query(chord)[0].mean())
                    r = route(contacts[b], A, B)
                    if r is not None:
                        hops = torch.tensor([[min(i, j), max(i, j)] for i, j in zip(r[:-1], r[1:])], device=device)
                        with torch.no_grad():
                            pieces = T.build_paths(p, hops, contacts[b], dist[b], label[b])[0].cpu().numpy()
                        joined = np.concatenate([pc if hops[k, 0].item() == r[k] else pc[::-1] for k, pc in enumerate(pieces)])
                        sp = splice(joined)
                        to_mm = lambda q: (q / D.SIZE) @ M.T
                        rec.update(hops=len(r) - 1, via_unmatched=float(np.mean([c not in m for c in r[1:-1]])),
                                   route=float(tree.query(dense_path(to_mm(joined)))[0].mean()), spliced=float(tree.query(dense_path(to_mm(sp)))[0].mean()),
                                   spliced_arc_ratio=float(np.linalg.norm(np.diff(to_mm(sp), axis=0), axis=1).sum() / arc))
                    records[name].append(rec)
        offset += len(images)
    result = dict(entry=a.entry, run=a.run, node_threshold=node_t, validation_patches=len(ids), node_sets={})
    for name, recs in records.items():
        res = {}
        for lo, hi in BINS:
            s = [r for r in recs if lo <= r['arc'] < hi]
            if not s:
                continue
            w = [r for r in s if 'route' in r]
            med = lambda k, rows: float(np.median([r[k] for r in rows])) if rows else None
            ok = lambda k, rows: float(np.mean([r[k] <= 1.5 for r in rows])) if rows else None
            res[lab(lo, hi)] = dict(
                edges=len(s), why={k: float(np.mean([r['why'] == k for r in s])) for k in ('gap', 'third region', 'neither')},
                route_available=len(w) / len(s), hops_mean=med('hops', w), intermediate_unmatched=float(np.mean([r['via_unmatched'] for r in w])) if w else None,
                straight_median_mm=med('straight', s), straight_within_1p5=ok('straight', s),
                with_route=dict(straight_median_mm=med('straight', w), route_median_mm=med('route', w), spliced_median_mm=med('spliced', w),
                                straight_within_1p5=ok('straight', w), route_within_1p5=ok('route', w), spliced_within_1p5=ok('spliced', w),
                                spliced_arc_ratio_median=med('spliced_arc_ratio', w)))
            v = res[lab(lo, hi)]
            print('ROUTE', a.entry, name, lab(lo, hi), f"n={len(s)} | why: " + ', '.join(f'{k} {x:.2f}' for k, x in v['why'].items())
                  + f" | route available {v['route_available']:.2f}, hops {v['hops_mean']}, intermediates unmatched {v['intermediate_unmatched']}"
                  + f" | path to GT (edges with a route) median mm: straight {v['with_route']['straight_median_mm']}, route {v['with_route']['route_median_mm']}, spliced {v['with_route']['spliced_median_mm']}"
                  + f" | within 1.5 mm: straight {v['with_route']['straight_within_1p5']}, route {v['with_route']['route_within_1p5']}, spliced {v['with_route']['spliced_within_1p5']}", flush=True)
        result['node_sets'][name] = res
    dst = ROOT / 'experiments/long_edges_20261111'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f'routes-{a.entry}-{a.run}.json').write_text(json.dumps(result, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
