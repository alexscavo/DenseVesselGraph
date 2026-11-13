#!/usr/bin/env python3
"""Missed-node diagnosis of the dense traced model (journal open problem B1, 10 Nov 2026).

Why is node F1 only 0.48-0.65 on real data, on the training monitor as much as on validation?

collect (GPU): one trained traced run, on the training monitor and on validation.
  - per patch: all heatmap peaks >= 0.05 (position, score, centreline probability, vessel-mask radius) and, at every
    GT node, the heat response (at the node, max within 1 and 2 mm), centreline probability, image value and
    vessel-mask radius (Euclidean distance transform of the patch segmentation, mm);
  - node oracles for stage 2, same model, same tracing and edge head, only the node set changes:
      'gt'        all GT nodes at their exact positions (ceiling of stage 2 given perfect stage 1);
      'no_fp'     the model's peaks >= t with the unmatched ones removed (perfect precision, same recall);
      'no_fn'     the model's peaks >= t plus every unmatched GT node (perfect recall, same precision);
      'predicted' the model's peaks >= t (reference, = the ordinary evaluation at the frozen setting).
    Scored by D.evaluate at the frozen global decoding (node 0.30, edge 0.75) and at the edge threshold that is best
    on the training monitor for that node set.
analyse (CPU): per GT node type (boundary / junction / terminal), recall, the reason of each miss, and recall against
  the node's local properties (length of its shortest incident edge, distance to the nearest other GT node, vessel
  radius, distance to the patch face); unmatched predictions by where they sit (duplicate of a GT node, on a GT
  centreline between nodes, inside the vessel mask but off the GT graph, outside the mask).

Inserted loop anchors are "don't care" as in the evaluation: they take part in the matching, are never counted as
missed, and a prediction matched to one is not a false positive.

usage: node_miss_diagnosis_20261110.py collect <dataset> [--run name] [--smoke]
       node_miss_diagnosis_20261110.py analyse <dataset> [--run name] [--node-threshold 0.30]
Outputs: $SCRATCH/experiments/gnbm/node_miss_20261110/<dataset>-<run>-<split>.pkl (per-patch records),
         experiments/node_miss_diagnosis_20261110/<dataset>-oracle.json and <dataset>.json.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
OUT = ROOT / 'experiments/node_miss_diagnosis_20261110'
STORE = Path(os.environ.get('SCRATCH', '/tmp')) / 'experiments/gnbm/node_miss_20261110'
RUNS = Path(os.environ.get('SCRATCH', '/tmp')) / 'experiments/gnbm/dense_graph_20261030'
DEFAULT_RUN = {'SyntheticMRI': 'traced_offset'}
V0 = dict(node=0.30, edge=0.75)
LOW = 0.05
TOL = 2.0


# ----------------------------------------------------------------------------------------------------------- collect
def collect(a):
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import distance_transform_edt, maximum_filter
    import dense_graph_20261030 as D
    import dense_graph_traced_20261030 as T
    from node_heatmap_probe_20261030 import PLAN, make_dataset

    def collate(batch):
        return (*D.collate(batch), torch.stack([b[1][0].reshape(D.SIZE, D.SIZE, D.SIZE) for b in batch]))

    run = a.run or DEFAULT_RUN.get(a.dataset, 'traced_arcpen_r1109')
    run_dir = RUNS / a.dataset / run
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(final)
    assert not T.MODE['passthrough'] and not T.MODE['two_hop']
    device = 'cuda'
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], backbone=final.get('backbone', 'unet'),
                               feature_channels=final.get('feature_channels', 32)).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device)['model'])
    model.eval()
    plan = json.loads(PLAN.read_text())['datasets'][a.dataset]
    splits = dict(monitor=plan['monitor_train'], val=plan['val_ids'])
    if a.smoke:
        splits = {k: v[:16] for k, v in splits.items()}
    STORE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8')) - 2
    oracle = dict(dataset=a.dataset, run=run, own_thresholds=final['thresholds'], frozen=V0, splits={})
    rows_by = {}
    for split, ids in splits.items():
        ds = make_dataset(a.dataset, ids, augment=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=workers, collate_fn=collate)
        records, sets, offset = [], {k: [] for k in ('gt', 'no_fp', 'no_fn', 'predicted')}, 0
        t0 = time.time()
        for images, nodes, edges, geometry, seg in loader:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_image(images.to(device))
            feats, heat, prob = out[:, 5:].float(), out[:, :1].float().sigmoid(), out[:, 4].float().sigmoid()
            low = D.peaks(out, LOW)
            node_sets = {k: [] for k in sets}
            for b in range(len(images)):
                metric = geometry[b]['metric'].to(device) / D.SIZE  # voxel -> mm
                vox_mm = metric.norm(dim=0).cpu().numpy()
                gt = nodes[b].to(device) * D.SIZE
                p, s = low[b]
                mask = seg[b].numpy() > 0
                radius = distance_transform_edt(mask, sampling=vox_mm).astype(np.float32)
                radius3 = maximum_filter(radius, 3)
                prob3 = F.max_pool3d(prob[b][None, None], 3, 1, 1)[0, 0]
                image3 = F.max_pool3d(images[b][None].to(device), 3, 1, 1)[0, 0]
                h = heat[b, 0]

                def at(points, volume):
                    i = points.round().clamp(0, D.SIZE - 1).long()
                    return volume[i[:, 0], i[:, 1], i[:, 2]]

                def ball_max(points, r_mm):
                    reach = [int(np.ceil(r_mm / v)) for v in vox_mm]
                    g = torch.stack(torch.meshgrid(*[torch.arange(-k, k + 1, device=device) for k in reach], indexing='ij'), -1).view(-1, 3)
                    g = g[((g.float() @ metric.T).norm(dim=-1)) <= r_mm]
                    i = points.round().long()[:, None] + g[None]
                    ok = ((i >= 0) & (i < D.SIZE)).all(-1)
                    i = i.clamp(0, D.SIZE - 1)
                    return torch.where(ok, h[i[..., 0], i[..., 1], i[..., 2]], torch.zeros((), device=device)).amax(1)

                record = ds.records[offset + b]
                r3 = torch.as_tensor(radius3, device=device)
                rec = dict(sample_id=record.sample_id, cache=str(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz'),
                           peaks=(p / D.SIZE).cpu().numpy(), peak_scores=s.cpu().numpy(),
                           peak_cl=at(p, prob3).cpu().numpy() if len(p) else np.zeros(0, np.float32),
                           peak_radius=at(p, r3).cpu().numpy() if len(p) else np.zeros(0, np.float32),
                           foreground_fraction=float(mask.mean()))
                if len(gt):
                    rec.update(gt_heat=at(gt, h).cpu().numpy(), gt_heat_1mm=ball_max(gt, 1.0).cpu().numpy(),
                               gt_heat_2mm=ball_max(gt, 2.0).cpu().numpy(), gt_cl=at(gt, prob3).cpu().numpy(),
                               gt_image=at(gt, image3).float().cpu().numpy(), gt_radius=at(gt, r3).cpu().numpy())
                records.append(rec)
                keep = s >= V0['node']
                pk, sk = p[keep], s[keep]
                mapping = D.match(pk @ metric.T, gt @ metric.T) if len(pk) and len(gt) else {}
                hit = torch.tensor(sorted(mapping), dtype=torch.long, device=device)
                missed = torch.tensor(sorted(set(range(len(gt))) - set(mapping.values())), dtype=torch.long, device=device)
                one = lambda n: torch.ones(n, device=device)
                node_sets['gt'].append((gt, one(len(gt))))
                node_sets['predicted'].append((pk, sk))
                node_sets['no_fp'].append((pk[hit], sk[hit]))
                node_sets['no_fn'].append((torch.cat([pk, gt[missed]]), torch.cat([sk, one(len(missed))])))
            for name, found in node_sets.items():
                sets[name] += stage2_rows(model, T, D, feats, heat, prob, found, geometry, ds, offset, device)
            offset += len(images)
        (STORE / f'{a.dataset}-{run}-{split}.pkl').write_bytes(pickle.dumps(records))
        rows_by[split] = sets
        print(f'COLLECTED {a.dataset} {split} {len(records)} patches {time.time() - t0:.0f}s', flush=True)
    (STORE / f"{'smoke-' if a.smoke else ''}{a.dataset}-{run}-oracle-rows.pkl").write_bytes(pickle.dumps(rows_by))  # scoring can be redone on CPU
    for split, sets in rows_by.items():
        oracle['splits'][split] = {}
        for name, rows in sets.items():
            best = max(D.EDGE_THRESHOLDS, key=lambda e: D.evaluate(rows_by['monitor'][name], 0.0, e, geometry=False)['edge_f1'])
            entry = dict(frozen_edge_threshold=D.evaluate(rows, 0.0, V0['edge']),
                         monitor_selected_edge_threshold={**D.evaluate(rows, 0.0, best, geometry=False), 'edge_threshold': best})
            if split == 'val':
                entry['by_arc_length'] = T.length_breakdown(rows, 0.0, V0['edge'])
            oracle['splits'][split][name] = entry
            e = entry['frozen_edge_threshold']
            print('ORACLE', a.dataset, split, name, 'node F1 %.3f edge F1 %.3f (P %.3f R %.3f) branch %.3f curve %.3f | best edge thr %.2f -> edge F1 %.3f'
                  % (e['node_f1'], e['edge_f1'], e['edge_precision'], e['edge_recall'], e['common_branch_f1'], e['common_curve_f1'],
                     best, entry['monitor_selected_edge_threshold']['edge_f1']), flush=True)
    (OUT / f"{'smoke-' if a.smoke else ''}{a.dataset}-oracle.json").write_text(json.dumps(oracle, indent=1, default=float) + '\n')


def stage2_rows(model, T, D, feats, heat, prob, found, geometry, ds, offset, device):
    """T.predict for a given node set (positions in voxels, scores) of one batch."""
    import torch
    rows = []
    with torch.no_grad():
        contacts, dist, label = T.traced_pairs(prob, [p for p, _ in found])
        for b, (p, score) in enumerate(found):
            metric = geometry[b]['metric'].to(device) / D.SIZE
            row = dict(nodes=(p / D.SIZE).cpu().numpy(), node_scores=score.cpu().numpy(), pairs=np.zeros((0, 2), int),
                       edge_scores=None, curves=None, traced=None, interior=None)
            pairs = T.patch_candidates(p, metric, contacts[b]) if len(p) >= 2 else None
            if pairs is not None and len(pairs):
                paths, traced = T.build_paths(p, pairs, contacts[b], dist[b], label[b])
                x = model.path_features(feats[b], heat[b], paths, prob[b])
                interior = T.choose_interior(T.to_mm(paths, metric[None].expand(len(paths), 3, 3)), traced, T.voxel_size(metric))
                curves = model.path_curves(x, paths, interior, traced)
                row.update(pairs=pairs.cpu().numpy(), edge_scores=model.edge_head(x).squeeze(-1).sigmoid().cpu().numpy(),
                           curves=(curves / D.SIZE).cpu().numpy(), traced=traced.cpu().numpy(), interior=interior.cpu().numpy())
            record = ds.records[offset + b]
            row.update(sample_id=record.sample_id, cache=str(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz'))
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------------------- analyse
ARC_BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]
NN_BINS = [(0, 1), (1, 2), (2, 4), (4, 1e9)]
RADIUS_BINS = [(0, 0.01), (0.01, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 1e9)]
FACE_BINS = [(0, 1), (1, 3), (3, 8), (8, 1e9)]
SCORE_BINS = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.01)]


def label(lo, hi, unit):
    return f'{lo:g}-{hi:g} {unit}' if hi < 1e8 else f'>= {lo:g} {unit}'


def binned(values, hits, bins, unit):
    values, hits = np.asarray(values, float), np.asarray(hits, bool)
    out = {}
    for lo, hi in bins:
        s = (values >= lo) & (values < hi)
        if s.any():
            out[label(lo, hi, unit)] = dict(nodes=int(s.sum()), recall=float(hits[s].mean()), share_of_misses=float((~hits[s]).sum() / max(1, (~hits).sum())))
    return out


def dense_path(p, step=0.25):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return p[:1]
    t = np.linspace(0, cum[-1], max(2, int(np.ceil(cum[-1] / step)) + 1))
    return np.stack([np.interp(t, cum, p[:, k]) for k in range(3)], 1)


def analyse_patch(r, t):
    from scipy.spatial import cKDTree
    from diagnose_branch_relations import match_nodes
    with np.load(r['cache']) as z:
        M, gn, ed = z['metric'], z['nodes'], z['edges'].reshape(-1, 2)
        boundary = z['boundary'].astype(bool)
        inserted = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(gn), bool)
        P, off = z['full_paths'] @ M.T, z['path_offsets']
    g = gn @ M.T
    n = len(g)
    deg = np.bincount(ed.ravel(), minlength=n)
    arcs = np.array([np.linalg.norm(np.diff(P[off[k]:off[k + 1]], axis=0), axis=1).sum() for k in range(len(ed))])
    min_arc = np.full(n, np.inf)
    for k, (i, j) in enumerate(ed):
        min_arc[i], min_arc[j] = min(min_arc[i], arcs[k]), min(min_arc[j], arcs[k])
    if n > 1:
        dd = np.linalg.norm(g[:, None] - g[None], axis=-1)
        np.fill_diagonal(dd, np.inf)
        nn = dd.min(1)
    else:
        nn = np.full(n, np.inf)
    vox = gn * 64
    face = np.minimum(vox + 0.5, 63.5 - vox).min(1) if n else np.zeros(0)
    p, s = r['peaks'] @ M.T, r['peak_scores']
    keep = np.flatnonzero(s >= t)
    high = match_nodes(p[keep], g, TOL) if len(keep) and n else {}
    low = match_nodes(p, g, TOL) if len(p) and n else {}
    hit = np.zeros(n, bool)
    hit[list(high.values())] = True
    low_gt = {gt: q for q, gt in low.items()}
    free_high = np.array([k for q, k in enumerate(keep) if q not in high], int)  # unmatched kept peaks
    tree = cKDTree(p) if len(p) else None
    nodes = []
    for i in range(n):
        if inserted[i]:
            continue
        kind = 'boundary' if boundary[i] else 'junction' if deg[i] >= 3 else 'terminal' if deg[i] == 1 else 'isolated' if deg[i] == 0 else 'degree2'
        reason, sub = 'matched', np.nan
        if not hit[i]:
            near = tree.query_ball_point(g[i], TOL) if tree is not None else []
            if i in low_gt:
                reason, sub = 'peak below threshold', float(s[low_gt[i]])
            elif near:
                reason = 'only peak belongs to another GT node'
            elif len(free_high) and np.linalg.norm(p[free_high] - g[i], axis=1).min() <= 2 * TOL:
                reason = 'detection 2-4 mm away'
            elif r['gt_heat_2mm'][i] >= LOW:
                reason = 'heat but no peak'
            else:
                reason = 'no response'
        nodes.append(dict(kind=kind, hit=bool(hit[i]), reason=reason, sub_score=sub, deg=int(deg[i]), min_arc=float(min_arc[i]), nn=float(nn[i]),
                          face=float(face[i]), radius=float(r['gt_radius'][i]), heat=float(r['gt_heat'][i]), heat2=float(r['gt_heat_2mm'][i]),
                          cl=float(r['gt_cl'][i]), image=float(r['gt_image'][i])))
    # unmatched predictions
    path_tree = cKDTree(np.concatenate([dense_path(P[off[k]:off[k + 1]]) for k in range(len(ed))])) if len(ed) else None
    fps, ignored = [], 0
    for q, k in enumerate(keep):
        if q in high:
            ignored += int(inserted[high[q]])
            continue
        d_node = float(np.linalg.norm(g - p[k], axis=1).min()) if n else np.inf
        d_path = float(path_tree.query(p[k])[0]) if path_tree is not None else np.inf
        pv = r['peaks'][k] * 64
        where = ('within 2 mm of a GT node' if d_node <= TOL else 'on a GT centreline, between nodes' if d_path <= 1.5
                 else 'inside the vessel mask, off the GT graph' if r['peak_radius'][k] > 0 else 'outside the vessel mask')
        fps.append(dict(where=where, score=float(s[k]), d_node=d_node, d_path=d_path, cl=float(r['peak_cl'][k]),
                        radius=float(r['peak_radius'][k]), face=float(np.minimum(pv + 0.5, 63.5 - pv).min())))
    # edges: are both endpoints matched, by arc length (anchors count as matched when their peak exists or they are inserted)
    edges = [dict(arc=float(arcs[k]), both=bool((hit[i] or inserted[i]) and (hit[j] or inserted[j]))) for k, (i, j) in enumerate(ed)]
    return nodes, fps, edges, dict(pred=len(keep), matched_real=int(sum(not inserted[v] for v in high.values())), ignored=ignored,
                                   gt_real=int((~inserted).sum()))


def summarise(rows, t):
    nodes, fps, edges, tot = [], [], [], dict(pred=0, matched_real=0, ignored=0, gt_real=0)
    for r in rows:
        if 'gt_heat' not in r:  # patch without GT nodes
            r = dict(r, gt_heat=np.zeros(0), gt_heat_2mm=np.zeros(0), gt_cl=np.zeros(0), gt_image=np.zeros(0), gt_radius=np.zeros(0))
        a, b, c, d = analyse_patch(r, t)
        nodes += a
        fps += b
        edges += c
        for k in tot:
            tot[k] += d[k]
    precision = tot['matched_real'] / max(1, tot['pred'] - tot['ignored'])
    recall = tot['matched_real'] / max(1, tot['gt_real'])
    out = dict(node_threshold=t, patches=len(rows), gt_nodes=tot['gt_real'], predicted_nodes=tot['pred'] - tot['ignored'],
               precision=precision, recall=recall, f1=2 * precision * recall / max(1e-9, precision + recall))
    misses = [x for x in nodes if not x['hit']]
    out['by_type'] = {}
    for kind in ('boundary', 'junction', 'terminal', 'isolated', 'degree2'):
        s = [x for x in nodes if x['kind'] == kind]
        if not s:
            continue
        m = [x for x in s if not x['hit']]
        reasons = {}
        for x in m:
            reasons[x['reason']] = reasons.get(x['reason'], 0) + 1
        hits = [x['hit'] for x in s]
        out['by_type'][kind] = dict(
            nodes=len(s), share_of_gt=len(s) / len(nodes), recall=float(np.mean(hits)), share_of_all_misses=len(m) / max(1, len(misses)),
            miss_reasons={k: v / max(1, len(m)) for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])},
            by_shortest_incident_edge=binned([x['min_arc'] for x in s], hits, ARC_BINS, 'mm'),
            by_nearest_gt_node=binned([x['nn'] for x in s], hits, NN_BINS, 'mm'),
            by_vessel_radius=binned([x['radius'] for x in s], hits, RADIUS_BINS, 'mm'),
            by_distance_to_face=binned([x['face'] for x in s], hits, FACE_BINS, 'voxels'),
            median_heat_at_node=dict(matched=float(np.median([x['heat'] for x in s if x['hit']] or [np.nan])),
                                     missed=float(np.median([x['heat'] for x in m] or [np.nan]))),
            median_centreline_probability=dict(matched=float(np.median([x['cl'] for x in s if x['hit']] or [np.nan])),
                                               missed=float(np.median([x['cl'] for x in m] or [np.nan]))))
    reasons = {}
    for x in misses:
        reasons[x['reason']] = reasons.get(x['reason'], 0) + 1
    out['miss_reasons'] = {k: v / max(1, len(misses)) for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])}
    hits = [x['hit'] for x in nodes]
    out['all_nodes'] = dict(by_shortest_incident_edge=binned([x['min_arc'] for x in nodes], hits, ARC_BINS, 'mm'),
                            by_nearest_gt_node=binned([x['nn'] for x in nodes], hits, NN_BINS, 'mm'),
                            by_vessel_radius=binned([x['radius'] for x in nodes], hits, RADIUS_BINS, 'mm'))
    # the same recall with the nodes of short edges set aside: what a GT without sub-tolerance edges would look like
    for cut in (1.0, 2.0):
        s = [x for x in nodes if x['min_arc'] >= cut]
        out[f'recall_nodes_with_all_edges_>=_{cut:g}mm'] = dict(nodes=len(s), share_of_gt=len(s) / max(1, len(nodes)),
                                                               recall=float(np.mean([x['hit'] for x in s])) if s else None)
    where = {}
    for x in fps:
        where.setdefault(x['where'], []).append(x)
    out['unmatched_predictions'] = dict(total=len(fps), by_place={
        k: dict(share=len(v) / max(1, len(fps)), median_score=float(np.median([x['score'] for x in v])),
                median_distance_to_gt_node_mm=float(np.median([x['d_node'] for x in v])),
                within_3_voxels_of_face=float(np.mean([x['face'] < 3 for x in v])),
                by_score={label(lo, hi, ''): int(sum(lo <= x['score'] < hi for x in v)) for lo, hi in SCORE_BINS})
        for k, v in sorted(where.items(), key=lambda kv: -len(kv[1]))})
    arcs, both = np.array([e['arc'] for e in edges]), np.array([e['both'] for e in edges], bool)
    out['gt_edges'] = dict(total=len(edges), by_arc={label(lo, hi, 'mm'): dict(
        edges=int(((arcs >= lo) & (arcs < hi)).sum()), share=float(((arcs >= lo) & (arcs < hi)).mean()),
        both_endpoints_matched=float(both[(arcs >= lo) & (arcs < hi)].mean())) for lo, hi in ARC_BINS if ((arcs >= lo) & (arcs < hi)).any()})
    return out


def analyse(a):
    run = a.run or DEFAULT_RUN.get(a.dataset, 'traced_arcpen_r1109')
    result = dict(dataset=a.dataset, run=run, matching_tolerance_mm=TOL, splits={})
    for split in ('monitor', 'val'):
        rows = pickle.loads((STORE / f'{a.dataset}-{run}-{split}.pkl').read_bytes())
        result['splits'][split] = summarise(rows, a.node_threshold)
        s = result['splits'][split]
        print(f"\n== {a.dataset} {split}: {s['patches']} patches, node P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} at {a.node_threshold}")
        for kind, v in s['by_type'].items():
            print(f"  {kind:9s} n={v['nodes']:6d} ({v['share_of_gt']:.2f} of GT) recall {v['recall']:.3f}  share of misses {v['share_of_all_misses']:.2f}  "
                  + '; '.join(f'{k} {x:.2f}' for k, x in v['miss_reasons'].items()))
            for name in ('by_shortest_incident_edge', 'by_nearest_gt_node', 'by_vessel_radius'):
                print(f"      {name}: " + ' | '.join(f"{k}: {x['recall']:.2f} (n={x['nodes']})" for k, x in v[name].items()))
        for k in ('recall_nodes_with_all_edges_>=_1mm', 'recall_nodes_with_all_edges_>=_2mm'):
            print(' ', k, s[k])
        u = s['unmatched_predictions']
        print(f"  unmatched predictions: {u['total']}  " + '; '.join(f"{k} {v['share']:.2f}" for k, v in u['by_place'].items()))
        print('  GT edges by arc: ' + ' | '.join(f"{k}: share {v['share']:.2f}, both endpoints {v['both_endpoints_matched']:.2f}" for k, v in s['gt_edges']['by_arc'].items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{a.dataset}.json').write_text(json.dumps(result, indent=1, default=float) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['collect', 'analyse'])
    p.add_argument('dataset')
    p.add_argument('--run', default=None)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--node-threshold', type=float, default=V0['node'])
    a = p.parse_args()
    collect(a) if a.command == 'collect' else analyse(a)
