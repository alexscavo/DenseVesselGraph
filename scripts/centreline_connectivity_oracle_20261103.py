#!/usr/bin/env python3
"""How much does broken predicted-centreline connectivity cost edge prediction? (3 Nov 2026)

Validation only, one trained traced model, at its own train-monitor-selected thresholds from
final.json (nothing is re-selected). Two inference conditions, identical except for the
centreline probability that the tracer reads:
  predicted  p = sigmoid(U-Net channel 4), exactly as in final_evaluation
  repaired   p = max(p, GT centreline target), the 3^3-dilated GT guides used as the training
             target, i.e. every gap along a GT vessel is filled; nothing else is added.
Nodes, node scores, corridor features and both heads are unchanged, so the difference in edge
recovery is the cost of broken centreline connectivity alone. (The repaired map can also merge
or split geodesic regions elsewhere, so precision may move too; both are reported.)

Per GT edge whose two endpoints are matched (2 mm): the predicted centreline is sampled along
the dense GT path (cache full_paths, <= 0.5 voxel spacing) after a 3^3 max-pool, so a 1-voxel
misregistration of the predicted ridge does not count as a gap. Samples within 2 voxels of
either GT endpoint are ignored (the tracer forces a radius-2 ball around every node). The edge
is "broken" if any remaining sample is < 0.5 (TAU), with the longest gap in mm.
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dense_graph_traced_20261030 as T
from tracing_miss_breakdown_20261103 import BINS, breakdown, edge_outcomes
from compare_dense_anchored_20261030 import SCRATCH

D, SIZE = T.D, T.SIZE
GAP_BINS = {'intact': (-1, 0), 'gap <2mm': (0, 2), 'gap 2-5mm': (2, 5), 'gap >=5mm': (5, 1e9)}


def repaired(prob, nodes, geometry):
    line = T.targets(nodes, geometry, prob.device)[3][:, 0]
    return torch.maximum(prob, line)


@torch.no_grad()
def gt_path_gaps(model, ds, device, workers, batch):
    """sample_id -> per GT edge (cache order): longest gap in mm along the GT path (-1 = intact)."""
    model.eval()
    gaps, offset = {}, 0
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=T.collate)
    for images, *_ in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model.forward_image(images.to(device))
        prob = F.max_pool3d(out[:, 4:5].float().sigmoid(), 3, 1, 1)[:, 0]
        for b in range(len(images)):
            record = ds.records[offset + b]
            with np.load(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz') as z:
                M, P, o = z['metric'], z['full_paths'] * SIZE, z['path_offsets']
            row = []
            for e in range(len(o) - 1):
                path = torch.as_tensor(P[o[e]:o[e + 1]], dtype=torch.float32, device=device)
                step = (path[1:] - path[:-1]).norm(dim=-1)
                n = int(max(2, step.sum().item() * 2 + 1))
                dense = F.interpolate(path.T[None], size=n, mode='linear', align_corners=True)[0].T
                g = (dense / (SIZE - 1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3)
                p = F.grid_sample(prob[b][None, None], g, align_corners=True).view(-1)
                inner = ((dense - path[0]).norm(dim=-1) > T.SEED_RADIUS) & ((dense - path[-1]).norm(dim=-1) > T.SEED_RADIUS)
                low = ((p < T.TAU) & inner).cpu().numpy()
                mm = np.linalg.norm(np.diff((dense.cpu().numpy() / SIZE) @ M.T, axis=0), axis=1)
                seg = np.concatenate([[0.0], mm])  # length attributed to each sample
                best = run = 0.0
                for is_low, length in zip(low, seg):
                    run = run + length if is_low else 0.0
                    best = max(best, run)
                row.append(best if low.any() else -1.0)
            gaps[record.sample_id] = np.array(row)
        offset += len(images)
    model.train()
    return gaps


def subject_bootstrap(per_patch, n=2000, seed=261103):
    """per_patch: sample_id -> (edges, recovered_predicted, recovered_repaired). Paired CI on the recall gain."""
    subjects = {}
    for sid, v in per_patch.items():
        subjects.setdefault(sid.split('_')[1], []).append(v)
    sums = np.array([np.sum(v, 0) for v in subjects.values()], float)
    rng = np.random.default_rng(seed)
    draws = sums[rng.integers(0, len(sums), (n, len(sums)))].sum(1)
    delta = (draws[:, 2] - draws[:, 1]) / np.maximum(1, draws[:, 0])
    point = (sums[:, 2].sum() - sums[:, 1].sum()) / max(1, sums[:, 0].sum())
    return dict(recall_gain=float(point), ci95=[float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))], subjects=len(sums))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['IXI', 'BRAVA', 'SyntheticMRI'], required=True)
    p.add_argument('--arm', default='traced_adaptive')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--eval-batch', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run = SCRATCH / f'dense_graph_20261030/{a.dataset}/{a.arm}'
    final = json.loads((run / 'final.json').read_text())
    for k, v in (final.get('mode') or {}).items():
        T.MODE[k] = v
    node_t, edge_t = final['thresholds']['node'], final['thresholds']['edge']
    val_ids = json.loads(T.PLAN.read_text())['datasets'][a.dataset]['val_ids']
    if a.smoke:
        val_ids = val_ids[:16]
    device = 'cuda'
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features']).to(device)
    model.load_state_dict(torch.load(run / 'model.pt', map_location=device)['model'])
    ds = T.make_dataset(a.dataset, val_ids, augment=False)

    rows = {}
    for name, hook in (('predicted', None), ('repaired', repaired)):
        T.PROB_HOOK = hook
        rows[name] = T.predict(model, ds, device, node_t, a.workers, a.eval_batch)
        print('PREDICTED', name, flush=True)
    T.PROB_HOOK = None
    gaps = gt_path_gaps(model, ds, device, a.workers, a.eval_batch)

    outcomes = {name: {r['sample_id']: e for r, e in edge_outcomes(rows[name], node_t, edge_t)} for name in rows}
    joint = {g: {b: dict(edges=0, predicted=Counter(), repaired=Counter()) for b in BINS} for g in GAP_BINS}
    per_patch = {}
    for sid, pred in outcomes['predicted'].items():
        rep, gap = outcomes['repaired'][sid], gaps[sid]
        assert len(pred) == len(rep) == len(gap), sid
        per_patch[sid] = (len(pred), sum(o.endswith('recovered') for _, o in pred), sum(o.endswith('recovered') for _, o in rep))
        for (arc, o_pred), (_, o_rep), g in zip(pred, rep, gap):
            if o_pred == 'endpoints_missing':
                continue
            gb = next(k for k, (lo, hi) in GAP_BINS.items() if (g < 0 and k == 'intact') or (g >= 0 and lo <= g < hi and k != 'intact'))
            for b, (lo, hi) in BINS.items():
                if lo <= arc < hi:
                    t = joint[gb][b]
                    t['edges'] += 1
                    t['predicted'][o_pred] += 1
                    t['repaired'][o_rep] += 1
    result = dict(dataset=a.dataset, arm=a.arm, thresholds=dict(node=node_t, edge=edge_t), patches=len(per_patch),
                  validation={n: D.evaluate(r, node_t, edge_t) for n, r in rows.items()},
                  breakdown={n: breakdown(r, node_t, edge_t) for n, r in rows.items()},
                  both_endpoints_found_by_gap={g: {b: dict(edges=t['edges'], predicted=dict(t['predicted']), repaired=dict(t['repaired']))
                                                   for b, t in v.items()} for g, v in joint.items()},
                  recall_gain_all_gt_edges=subject_bootstrap(per_patch))
    name = f'centreline-oracle-{a.dataset}-{a.arm}' + ('-smoke' if a.smoke else '')
    out = T.ROOT / 'experiments/dense_graph_20261030'
    (out / f'{name}.json').write_text(json.dumps(result, indent=1))
    (out / f'{name}-gaps.pkl').write_bytes(pickle.dumps(gaps))
    print('RESULT ' + json.dumps(dict(validation=result['validation'], gain=result['recall_gain_all_gt_edges'])), flush=True)


if __name__ == '__main__':
    main()
