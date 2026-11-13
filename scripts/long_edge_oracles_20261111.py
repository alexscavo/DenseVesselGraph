#!/usr/bin/env python3
"""Sizing three ways to recover long edges, on a trained traced model (11 Nov 2026, GPU; validation, oracles).

Nothing is trained here. The same model is run with a changed node set and / or changed fallback paths:
  node sets
    predicted         the model's peaks >= its node threshold (reference);
    no_intermediate   ORACLE for "classify nodes and leave the intermediate ones out of the tracing": peaks that are
                      unmatched and within 1.5 mm of a GT centreline are removed before tracing;
    tips              predicted + centreline tips: voxels where a node's traced region ends (a local maximum of the
                      geodesic distance, not next to another region, at least MIN_TIP cost units from the node), kept
                      by a 5x5x5 non-maximum suppression. A tip is where the predicted centreline stops without a node:
                      a candidate for a missed terminal (inside the patch) or boundary node (at the face). No GT used;
    no_intermediate+tips
  fallback paths (pairs that hard tracing does not connect)
    straight          the chord (current model);
    soft              where the two nodes' regions touch when the centreline mask is dropped (cost 1/(p+0.05) in every
                      voxel), the path read back from that soft field; such pairs are also added as candidates.
The edge head was never trained on soft paths or on tip nodes, so its scores there are a lower bound of what a
trained arm would give; the geometric quantities do not depend on it:
  - distance to the GT centreline of the fallback path of GT edges that are not hard-traced (straight vs soft);
  - what the tips are (hit a missed GT node, duplicate a matched one, lie on a GT centreline, elsewhere);
  - share of the GT centreline inside the predicted centreline mask, and inside a region that has a node.
usage: long_edge_oracles_20261111.py <plan entry> <run> [--patches 600]
Writes experiments/long_edges_20261111/oracles-<entry>-<run>.json
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402
from node_miss_diagnosis_20261110 import dense_path  # noqa: E402

RUNS = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
MIN_TIP = 6.0
KEYS = ('node_f1', 'node_precision', 'node_recall', 'edge_f1', 'edge_precision', 'edge_recall', 'beta0_mae', 'beta1_mae')


@torch.no_grad()
def tips(prob, seeds):
    """Per patch: (n,3) voxel positions of centreline tips of the seeded regions, and the mask / label volumes."""
    mask, dist, label, _ = T.geodesic_regions(prob, seeds)
    ok = mask & (label >= 0) & torch.isfinite(dist)
    d = torch.where(ok, dist, torch.full_like(dist, -1.0))
    local_max = d >= T.neighbours(d, -1.0).amax(1)
    nl = T.neighbours(label, -1.0)
    contact = ((nl >= 0) & (nl != label[:, None])).any(1)
    cand = torch.where(ok & local_max & ~contact & (dist >= MIN_TIP), dist, torch.full_like(dist, -1.0))
    keep = (cand > 0) & (cand == F.max_pool3d(cand[:, None], 5, 1, 2)[:, 0])
    return [keep[b].nonzero().float() for b in range(len(prob))], mask, label


@torch.no_grad()
def rows_for(model, feats, heat, prob, found, geometry, ds, offset, soft):
    """As T.predict for given nodes; with soft=True non-traced pairs take the soft-field path where one exists."""
    device = prob.device
    seeds = [p for p, _ in found]
    contacts, dist, label = T.traced_pairs(prob, seeds)
    if soft:
        T.MODE['soft'] = True
        s_contacts, s_dist, s_label = T.traced_pairs(prob, seeds)
        T.MODE['soft'] = False
    rows = []
    for b, (p, score) in enumerate(found):
        metric = geometry[b]['metric'].to(device) / D.SIZE
        row = dict(nodes=(p / D.SIZE).cpu().numpy(), node_scores=score.cpu().numpy(), pairs=np.zeros((0, 2), int),
                   edge_scores=None, curves=None, traced=None, interior=None, soft=None, paths=None)
        pairs = T.patch_candidates(p, metric, contacts[b], s_contacts[b] if soft else None) if len(p) >= 2 else None
        if pairs is not None and len(pairs):
            paths, traced = T.build_paths(p, pairs, contacts[b], dist[b], label[b])
            soft_flag = torch.zeros(len(pairs), dtype=torch.bool, device=device)
            if soft:
                idx = [k for k, pr in enumerate(pairs.tolist()) if not traced[k] and tuple(pr) in s_contacts[b]]
                if idx:
                    paths[idx] = T.build_paths(p, pairs[idx], s_contacts[b], s_dist[b], s_label[b])[0]
                    soft_flag[idx] = True
            x = model.path_features(feats[b], heat[b], paths, prob[b])
            interior = T.choose_interior(T.to_mm(paths, metric[None].expand(len(paths), 3, 3)), traced, T.voxel_size(metric))
            curves = model.path_curves(x, paths, interior, traced)
            row.update(pairs=pairs.cpu().numpy(), edge_scores=model.edge_head(x).squeeze(-1).sigmoid().cpu().numpy(),
                       curves=(curves / D.SIZE).cpu().numpy(), traced=traced.cpu().numpy(), interior=interior.cpu().numpy(),
                       soft=soft_flag.cpu().numpy(), paths=(paths / D.SIZE).cpu().numpy())
        record = ds.records[offset + b]
        row.update(sample_id=record.sample_id, cache=str(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz'))
        rows.append(row)
    return rows


def path_quality(rows_straight, rows_soft, node_t):
    """GT edges with both ends matched whose pair is not hard-traced: fallback path vs GT centreline (mm)."""
    out = dict(straight=[], soft=[], has_soft=0, total=0, arcs=[])
    for rs, rf in zip(rows_straight, rows_soft):
        if rs['edge_scores'] is None:
            continue
        with np.load(rs['cache']) as z:
            M, g, ed, P, off = z['metric'], z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2), z['full_paths'] @ z['metric'].T, z['path_offsets']
        m = D.match(torch.as_tensor(rs['nodes'] @ M.T), torch.as_tensor(g))
        inv = {gt: q for q, gt in m.items()}
        ks = {tuple(sorted(map(int, pr))): k for k, pr in enumerate(rs['pairs'])}
        kf = {tuple(sorted(map(int, pr))): k for k, pr in enumerate(rf['pairs'])}
        for e, (a, b) in enumerate(ed):
            if a not in inv or b not in inv:
                continue
            key = tuple(sorted((inv[a], inv[b])))
            if key not in ks or rs['traced'][ks[key]]:
                continue
            G = dense_path(P[off[e]:off[e + 1]])
            arc = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
            if arc < 10:
                continue
            tree = cKDTree(G)
            out['total'] += 1
            out['arcs'].append(arc)
            out['straight'].append(float(tree.query(dense_path(rs['paths'][ks[key]] @ M.T))[0].mean()))
            k = kf.get(key)
            if k is not None and rf['soft'][k]:
                out['has_soft'] += 1
                out['soft'].append(float(tree.query(dense_path(rf['paths'][k] @ M.T))[0].mean()))
            else:
                out['soft'].append(out['straight'][-1])
    s, f, arcs = np.array(out['straight']), np.array(out['soft']), np.array(out['arcs'])
    res = {}
    for lo, hi in ((10, 20), (20, 40), (40, 1e9)):
        sel = (arcs >= lo) & (arcs < hi)
        if sel.any():
            res[f'{lo}-{hi} mm' if hi < 1e8 else f'>= {lo} mm'] = dict(
                edges=int(sel.sum()), straight_median_mm=float(np.median(s[sel])), soft_median_mm=float(np.median(f[sel])),
                straight_within_1p5mm=float(np.mean(s[sel] <= 1.5)), soft_within_1p5mm=float(np.mean(f[sel] <= 1.5)))
    res['soft_path_available'] = out['has_soft'] / max(1, out['total'])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entry')
    ap.add_argument('run')
    ap.add_argument('--patches', type=int, default=600)
    a = ap.parse_args()
    run_dir = RUNS / a.entry / a.run
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(final)
    T.MODE['compress_tol'] = 0.0
    node_t, edge_t = final['thresholds']['node'], final['thresholds']['edge']
    device = 'cuda'
    model = T.TracedDenseGraph(backbone=final.get('backbone', 'unet'), feature_channels=final.get('feature_channels', 32)).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device)['model'])
    model.eval()
    ids = json.loads(PLAN.read_text())['datasets'][a.entry]['val_ids'][:a.patches]
    ds = make_dataset(a.entry, ids, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=8, collate_fn=D.collate)
    names = ('predicted', 'no_intermediate', 'tips', 'no_intermediate+tips')
    rows = {(n, s): [] for n in names for s in ('straight', 'soft')}
    tip_kinds, coverage, offset, n_tips = Counter(), Counter(), 0, 0
    for images, nodes, edges, geometry in loader:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model.forward_image(images.to(device))
        feats, heat, prob = out[:, 5:].float(), out[:, :1].float().sigmoid(), out[:, 4].float().sigmoid()
        found = D.peaks(out, node_t)
        tip_pos, mask, label = tips(prob, [p for p, _ in found])
        sets = {n: [] for n in names}
        for b, (p, s) in enumerate(found):
            record = ds.records[offset + b]
            with np.load(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz') as z:
                M, gn, ed, P, off = z['metric'], z['nodes'], z['edges'].reshape(-1, 2), z['full_paths'] @ z['metric'].T, z['path_offsets']
                boundary = z['boundary'].astype(bool)
            g = gn @ M.T
            deg = np.bincount(ed.ravel(), minlength=len(g))
            pm = (p / D.SIZE).cpu().numpy() @ M.T
            m = D.match(torch.as_tensor(pm), torch.as_tensor(g)) if len(pm) and len(g) else {}
            dense = np.concatenate([dense_path(P[off[k]:off[k + 1]]) for k in range(len(ed))]) if len(ed) else np.zeros((0, 3))
            tree = cKDTree(dense) if len(dense) else None
            d_path = tree.query(pm)[0] if tree is not None and len(pm) else np.full(len(pm), np.inf)
            keep = torch.tensor([q in m or d_path[q] > 1.5 for q in range(len(pm))], dtype=torch.bool, device=device)
            t = tip_pos[b]
            n_tips += len(t)
            one = torch.ones(len(t), device=device)
            sets['predicted'].append((p, s))
            sets['no_intermediate'].append((p[keep], s[keep]))
            sets['tips'].append((torch.cat([p, t]), torch.cat([s, one])))
            sets['no_intermediate+tips'].append((torch.cat([p[keep], t]), torch.cat([s[keep], one])))
            # what the tips are
            if len(t):
                tm = (t / D.SIZE).cpu().numpy() @ M.T
                missed = np.array(sorted(set(range(len(g))) - set(m.values())), int)
                dg = np.linalg.norm(tm[:, None] - g[None], axis=-1) if len(g) else np.full((len(tm), 1), np.inf)
                dp = tree.query(tm)[0] if tree is not None else np.full(len(tm), np.inf)
                face = np.minimum(t.cpu().numpy() + 0.5, 63.5 - t.cpu().numpy()).min(1)
                for q in range(len(tm)):
                    j = int(dg[q].argmin()) if len(g) else -1
                    if j >= 0 and dg[q, j] <= 2.0 and j in missed:
                        kind = 'hits a missed GT node: ' + ('boundary' if boundary[j] else 'junction' if deg[j] >= 3 else 'terminal' if deg[j] == 1 else 'other')
                    elif j >= 0 and dg[q, j] <= 2.0:
                        kind = 'within 2 mm of an already matched GT node'
                    elif dp[q] <= 1.5:
                        kind = 'on a GT centreline, away from nodes'
                    else:
                        kind = 'off the GT graph' + (' (at the patch face)' if face[q] < 3 else ' (interior)')
                    tip_kinds[kind] += 1
            # GT centreline: inside the predicted mask? inside a region with a node?
            if len(dense):
                v = np.clip(np.round(dense @ np.linalg.inv(M).T * D.SIZE).astype(int), 0, D.SIZE - 1)
                inside = mask[b].cpu().numpy()[v[:, 0], v[:, 1], v[:, 2]]
                seeded = label[b].cpu().numpy()[v[:, 0], v[:, 1], v[:, 2]] >= 0
                coverage['points'] += len(v)
                coverage['in_mask'] += int(inside.sum())
                coverage['in_seeded_region'] += int((inside & seeded).sum())
        for n in names:
            for mode in ('straight', 'soft'):
                rows[(n, mode)] += rows_for(model, feats, heat, prob, sets[n], geometry, ds, offset, mode == 'soft')
        offset += len(images)
    result = dict(entry=a.entry, run=a.run, thresholds=dict(node=node_t, edge=edge_t), validation_patches=len(ids), min_tip_cost=MIN_TIP,
                  tips_per_patch=n_tips / len(ids), tip_kinds={k: v / max(1, n_tips) for k, v in tip_kinds.most_common()},
                  gt_centreline=dict(in_predicted_mask=coverage['in_mask'] / max(1, coverage['points']),
                                     in_region_with_a_node=coverage['in_seeded_region'] / max(1, coverage['points'])),
                  fallback_path_quality=path_quality(rows[('predicted', 'straight')], rows[('predicted', 'soft')], node_t), configurations={})
    print('TIPS', a.entry, f"{result['tips_per_patch']:.2f} per patch", json.dumps({k: round(v, 3) for k, v in result['tip_kinds'].items()}), flush=True)
    print('CENTRELINE', a.entry, json.dumps(result['gt_centreline']), flush=True)
    print('FALLBACK PATH', a.entry, json.dumps(result['fallback_path_quality']), flush=True)
    for (n, mode), r in rows.items():
        best = max(D.EDGE_THRESHOLDS, key=lambda e: D.evaluate(r, 0.0, e, geometry=False)['edge_f1'])
        e = D.evaluate(r, 0.0, edge_t, geometry=False)
        by_arc = T.length_breakdown(r, 0.0, edge_t)
        entry = dict({k: e[k] for k in KEYS}, best_edge_threshold_on_these_patches=best, edge_f1_at_best=D.evaluate(r, 0.0, best, geometry=False)['edge_f1'],
                     by_arc={k: {x: v[x] for x in ('edges', 'recovered_fraction', 'endpoints_missing_fraction', 'rejected_fraction', 'not_candidate_fraction')} for k, v in by_arc.items()})
        result['configurations'][f'{n} / {mode}'] = entry
        print('CONFIG', a.entry, f'{n} / {mode}:', 'node P/R/F1 %.3f %.3f %.3f | edge P/R/F1 %.3f %.3f %.3f (best thr %.2f: %.3f) | b0 %.2f b1 %.2f | recovered 10-20 %.3f, 20-40 %.3f, >=40 %.3f | endpoint missing >=40 %.2f'
              % (e['node_precision'], e['node_recall'], e['node_f1'], e['edge_precision'], e['edge_recall'], e['edge_f1'], best, entry['edge_f1_at_best'], e['beta0_mae'], e['beta1_mae'],
                 by_arc['10-20']['recovered_fraction'], by_arc['20-40']['recovered_fraction'], by_arc['40-inf']['recovered_fraction'], by_arc['40-inf']['endpoints_missing_fraction']), flush=True)
    dst = ROOT / 'experiments/long_edges_20261111'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f'oracles-{a.entry}-{a.run}.json').write_text(json.dumps(result, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
