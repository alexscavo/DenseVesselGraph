#!/usr/bin/env python3
"""Average precision / recall of the traced (stage 2) model (5 Nov 2026); definitions of dense_ap_20261030.

- node AP @2 mm: every heatmap peak >= 0.01 of the model, score order, greedy match to unmatched GT within 2 mm,
  recall over all GT nodes (max_recall = recall at the lowest score).
- box node / edge mAP and mAR: RelationFormer/GrEVe protocol (IoU 0.50:0.95, 40 detections, GT node box side 0.2,
  edge half-width 0.1) on the decoded graph, nodes and edges ranked by their own scores; the dense model has no box
  head, so predicted nodes get GT-size boxes.
- edge ranking AP: over all candidate pairs among the retained nodes ('all_retained') and over pairs with both
  endpoints matched ('conditional').
Decodings: V0 (final.json thresholds) and V4 (hysteresis with the tree rule, the monitor-chosen configuration of
decode_study_20261105). The stage-1 model (spline6) is reported on the same validation list for reference.
Note: GT includes the loop breaking-point anchors of the loop cache in these protocols (they are not loop-aware).
Validation only. usage: traced_ap_20261105.py <dataset key> [run]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402
from dense_ap_20261030 import SCRATCH, box_map, edge_parts, node_ap, node_ap_state  # noqa: E402
from decode_study_20261105 import SWEEP, decode  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402


def node_ap_from_model(model, ds, device):
    states, offset = [], 0
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=8, collate_fn=D.collate)
    model.eval()
    with torch.no_grad():
        for images, _, _, _ in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_image(images.to(device))
            for b, (p, score) in enumerate(D.peaks(out, 0.01)):
                record = ds.records[offset + b]
                with np.load(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz') as z:
                    states.append(node_ap_state((p / D.SIZE).cpu().numpy(), score.cpu().numpy(), z['nodes'], z['metric']))
            offset += len(images)
    return node_ap(states)


def decoded_rows(rows, cfg):
    """Rows holding the decoded graph with the ORIGINAL node and edge scores (box mAP ranks by score)."""
    out = []
    for r in rows:
        nodes, pairs = decode(r, **cfg)
        remap = {int(k): i for i, k in enumerate(nodes)}
        row = dict(r, nodes=r['nodes'][nodes], node_scores=r['node_scores'][nodes])
        if r['edge_scores'] is not None and len(pairs):
            row.update(pairs=np.array([[remap[int(a)], remap[int(b)]] for a, b in r['pairs'][pairs]], int),
                       edge_scores=r['edge_scores'][pairs], curves=r['curves'][pairs])
        else:
            row.update(pairs=np.zeros((0, 2), int), edge_scores=None)
        out.append(row)
    return out


def main(dataset, run='traced_adaptive'):
    device = 'cuda'
    run_dir = SCRATCH / 'dense_graph_20261030' / dataset / run
    final = json.loads((run_dir / 'final.json').read_text())
    thr = final['thresholds']
    T.MODE.update(final.get('mode', {}))
    val_ids = json.loads(PLAN.read_text())['datasets'][dataset]['val_ids']
    ds = make_dataset(dataset, val_ids, augment=False)
    result = dict(dataset=dataset, run=run, validation_patches=len(val_ids))
    model = T.TracedDenseGraph(backbone=final.get('backbone', 'unet')).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device)['model'])
    result['node_ap_2mm'] = node_ap_from_model(model, ds, device)
    rows = pickle.loads((SWEEP / dataset / run / f"val_{thr['node']:.2f}.pkl").read_bytes())
    result['V0'] = dict(thresholds=thr, **edge_parts(rows, thr['node'], strict=False), **box_map(rows, thr['node'], thr['edge'], strict=False))
    study = ROOT / f'experiments/decode_study_20261105/{dataset}.json'
    if study.exists():
        cfg = json.loads(study.read_text())['chosen']['V4_hysteresis_weak_tree']['config']
        low = pickle.loads((SWEEP / dataset / run / f"val_{cfg['t_low']:.2f}.pkl").read_bytes())
        result['V4'] = dict(config=cfg, **box_map(decoded_rows(low, cfg), 0.0, 0.0, strict=False))
    print(dataset, json.dumps(result), flush=True)
    # stage 1 on the same validation list, for reference
    s1_dir = SCRATCH / 'dense_graph_20261030' / dataset / 'spline6'
    s1_final = json.loads((s1_dir / 'final.json').read_text())
    s1 = D.DenseGraph('spline', 6, backbone=s1_final.get('backbone', 'unet')).to(device)
    s1.load_state_dict(torch.load(s1_dir / 'model.pt', map_location=device)['model'])
    s1_rows = pickle.loads((s1_dir / 'final-validation-rows.pkl').read_bytes())
    t1 = s1_final['thresholds']
    result['stage1_spline6'] = dict(thresholds=t1, node_ap_2mm=node_ap_from_model(s1, ds, device),
                                    **edge_parts(s1_rows, t1['node'], strict=False), **box_map(s1_rows, t1['node'], t1['edge'], strict=False))
    out = ROOT / f'experiments/decode_study_20261105/ap_{dataset}.json'
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(dataset, 'stage1', json.dumps(result['stage1_spline6']), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
