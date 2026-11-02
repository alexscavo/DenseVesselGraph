#!/usr/bin/env python3
"""Physical node AP @2 mm and edge ranking AP for the dense model (and the anchored arm).

Definitions copied from scripts/ixi_long_experiment_20261030.py (node_ap_state, node_ap,
ranking_ap) so the numbers are comparable with the literature-scale RelationFormer control:
- node AP: all proposals (dense: every heatmap peak >= 0.01; RelationFormer: all queries),
  score-ordered greedy matching to unmatched GT within 2 mm, recall over all GT nodes.
- edge ranking AP: sklearn AP over all candidate pairs among nodes retained at the
  train-selected node threshold ("all_retained"), and over pairs whose two endpoints are both
  matched ("conditional"). It ranks candidates; GT edges never proposed are not counted.
- RelationFormer box mAP/mAR (training/evaluation/metrics.graph_detection_states, protocol of
  configs/base.yaml: IoU 0.50:0.95, 40 detections, GT node box side 0.2, edge half-width 0.1),
  on the graph decoded at the train-selected thresholds. Arms without a box head (dense,
  anchored rows) use fixed side-0.2 boxes centred on the predicted nodes, i.e. the GT box
  convention; RelationFormer B0 uses its regressed boxes, so its value is reported separately.
Validation only.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import dense_graph_20261030 as D  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402
from diagnose_branch_relations import match_nodes  # noqa: E402
from ixi_query_diagnostics import edge_accounting  # noqa: E402

SCRATCH = Path(os.environ['SCRATCH']) / 'experiments/gnbm'


def node_ap_state(raw_nodes, raw_scores, gt_nodes, metric, tolerance=2.0):
    order = np.argsort(-raw_scores, kind='stable')
    d = np.linalg.norm((raw_nodes[order][:, None] - gt_nodes[None]) @ metric.T, axis=-1)
    matched = np.zeros(len(gt_nodes), bool)
    tp = np.zeros(len(order), bool)
    for i in range(len(order)):
        c = np.flatnonzero(~matched & (d[i] <= tolerance))
        if len(c):
            j = c[np.argmin(d[i, c])]
            matched[j] = True
            tp[i] = True
    return raw_scores[order], tp, len(gt_nodes)


def node_ap(states):
    scores = np.concatenate([s[0] for s in states])
    tp = np.concatenate([s[1] for s in states])
    gt = sum(s[2] for s in states)
    tp = tp[np.argsort(-scores, kind='stable')]
    cumulative = np.cumsum(tp)
    precision = cumulative / np.arange(1, len(tp) + 1)
    return dict(node_ap=float(precision[tp].sum() / gt), max_recall=float(cumulative[-1] / gt), proposals=len(tp), gt_nodes=gt)


def ranking_ap(scores, labels):
    from sklearn.metrics import average_precision_score
    labels = labels.astype(bool)
    return dict(ap=float(average_precision_score(labels, scores)) if labels.any() else None,
                candidates=int(len(scores)), positives=int(labels.sum()))


def edge_parts(rows, node_t, strict):
    s_all, l_all, s_cond, l_cond = [], [], [], []
    for r in rows:
        with np.load(r['cache']) as z:
            gt_nodes, gt_edges, metric = z['nodes'], z['edges'].reshape(-1, 2), z['metric']
        keep = np.nonzero(r['node_scores'] > node_t if strict else r['node_scores'] >= node_t)[0]
        if r['edge_scores'] is None or len(keep) < 2:
            continue
        remap = {int(k): i for i, k in enumerate(keep)}
        valid = np.isin(r['pairs'], keep).all(1)
        sub, scores = r['pairs'][valid], r['edge_scores'][valid]
        local = np.array([[remap[int(a)], remap[int(b)]] for a, b in sub], int).reshape(-1, 2)
        mapping = match_nodes(r['nodes'][keep] @ metric.T, gt_nodes @ metric.T, 2.0)
        _, _, labels, _, _ = edge_accounting(np.arange(len(keep)), local, scores, np.zeros(len(scores), bool), mapping, gt_edges, len(gt_nodes))
        cond = np.array([a in mapping and b in mapping for a, b in local], bool).reshape(-1)
        s_all.append(scores); l_all.append(labels); s_cond.append(scores[cond]); l_cond.append(labels[cond])
    cat = np.concatenate
    return dict(edge_ap_all_retained=ranking_ap(cat(s_all), cat(l_all)), edge_ap_conditional=ranking_ap(cat(s_cond), cat(l_cond)))


PROTOCOL = dict(iou_thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                max_detections=40, target_node_size=0.2, edge_half_width=0.1)


def box_map(rows, node_t, edge_t, strict):
    from training.evaluation.metrics import graph_detection_states, aggregate_detection_ap_ar
    node_states, edge_states = [], []
    for r in rows:
        with np.load(r['cache']) as z:
            gt_nodes, gt_edges = z['nodes'], z['edges'].reshape(-1, 2)
        keep = np.nonzero(r['node_scores'] > node_t if strict else r['node_scores'] >= node_t)[0]
        remap = {int(k): i for i, k in enumerate(keep)}
        nodes = r['nodes'][keep]
        edges, scores = np.zeros((0, 2), int), np.zeros(0)
        if r['edge_scores'] is not None and len(keep) >= 2:
            valid = np.isin(r['pairs'], keep).all(1)
            sel = valid & ((r['edge_scores'] > edge_t) if strict else (r['edge_scores'] >= edge_t))
            edges = np.array([[remap[int(a)], remap[int(b)]] for a, b in r['pairs'][sel]], int).reshape(-1, 2)
            scores = r['edge_scores'][sel]
        prediction = dict(nodes=nodes, edges=edges, node_scores=r['node_scores'][keep], edge_scores=scores,
                          boxes=np.concatenate([nodes, np.full_like(nodes, PROTOCOL['target_node_size'])], 1))
        n, e = graph_detection_states(prediction, gt_nodes, gt_edges, protocol=PROTOCOL)
        node_states.append(n); edge_states.append(e)
    result = {}
    for kind, states in (('node', node_states), ('edge', edge_states)):
        ap, ar = aggregate_detection_ap_ar(states, iou_thresholds=PROTOCOL['iou_thresholds'])
        result[f'box_{kind}_mAP'], result[f'box_{kind}_mAR'] = float(ap), float(ar)
    return result


def dense(dataset, device):
    out_dir = SCRATCH / f'dense_graph_20261030/{dataset}/spline6'
    thr = json.loads((out_dir / 'final-v2.json').read_text())['thresholds']
    model = D.DenseGraph('spline', 6).to(device)
    model.load_state_dict(torch.load(out_dir / 'model.pt', map_location=device)['model'])
    ds = make_dataset(dataset, json.loads(PLAN.read_text())['datasets'][dataset]['val_ids'], augment=False)
    states = []  # node AP from all peaks >= 0.01 (no pair features needed)
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=8, collate_fn=D.collate)
    offset = 0
    with torch.no_grad():
        for images, _, _, _ in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_image(images.to(device))
            for b, (p, score) in enumerate(D.peaks(out, 0.01)):
                record = ds.records[offset + b]
                with np.load(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz') as z:
                    states.append(node_ap_state((p / D.SIZE).cpu().numpy(), score.cpu().numpy(), z['nodes'], z['metric']))
            offset += len(images)
    rows_path = out_dir / 'final-v2-validation-rows.pkl'
    if not rows_path.exists():
        rows_path.write_bytes(pickle.dumps(D.predict(model, ds, device, thr['node'], 8)))
    rows = pickle.loads(rows_path.read_bytes())
    return dict(node_ap_2mm=node_ap(states), **edge_parts(rows, thr['node'], strict=False),
                **box_map(rows, thr['node'], thr['edge'], strict=False), thresholds=thr)


def anchored(dataset):
    rows = pickle.loads((SCRATCH / f'anchored_rf_20261030/runs/{dataset}/evaluation/validation-rows.pkl').read_bytes())
    thr = json.loads((ROOT / f'experiments/anchored_rf_20261030/final-{dataset}.json').read_text())['thresholds']
    states = []
    for r in rows:
        with np.load(r['cache']) as z:
            states.append(node_ap_state(r['raw_nodes'], r['raw_scores'], z['nodes'], z['metric']))
    return dict(node_ap_2mm=node_ap(states), **edge_parts(rows, thr['node'], strict=True),
                **box_map(rows, thr['node'], thr['edge'], strict=True), thresholds=thr)


def main():
    device = 'cuda'
    result = {}
    for dataset in ('SyntheticMRI', 'IXI', 'BRAVA'):
        result[f'dense/{dataset}'] = dense(dataset, device)
        print(dataset, json.dumps(result[f'dense/{dataset}']), flush=True)
    for dataset in ('IXI', 'BRAVA'):
        result[f'anchored/{dataset}'] = anchored(dataset)
        print('anchored', dataset, json.dumps(result[f'anchored/{dataset}']), flush=True)
    (ROOT / 'experiments/dense_graph_20261030/average-precision-validation.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
