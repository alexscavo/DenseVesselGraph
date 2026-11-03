#!/usr/bin/env python3
"""WS3 two-stage dense vessel-graph model (30 Oct 2026).

Stage 1 (nodes): 3D U-Net -> CenterNet node heatmap + sub-voxel offsets + centerline
aux + 32 feature channels. Nodes = heatmap local maxima (3^3 NMS) above a threshold.
Stage 2 (edges): candidate pairs = each node's 16 nearest nodes (mm). For each pair the
features in a corridor around the chord feed a shared 3D CNN with two heads: an edge
logit and the M interior points of a cubic B-spline (or polyline / chord arm).

There is no Hungarian assignment. Edge labels come from the same physical matching
used by evaluation: candidate pairs are built on *predicted* peaks, a peak inherits the
GT node it is matched to within 2 mm, and a pair is positive iff its matched GT nodes
share a GT edge. GT-node pairs (with 0.5 mm jitter) are used alone during warm-up and
alongside predicted-peak pairs afterwards. The candidate rule (16 nearest retained nodes)
is the same at training and inference; at inference it is applied to the nodes retained
at the selected node threshold. Thresholds (node, edge) are selected on the training
monitor only: node threshold by node F1, then edge threshold by edge F1.
Evaluation reuses match_nodes / edge_accounting / evaluate_common_vessels.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from models.spline import cubic_basis  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, collate, make_dataset, targets, focal  # noqa: E402
from oracle_geometry_20261030 import frame, polyline, chamfer, to_mm  # noqa: E402

SIZE = 64
L, S = 24, 11
NEIGHBOURS = 16
MAX_NEGATIVES = 256
MAX_PEAKS = 200  # IXI has up to 153 GT nodes per patch
NODE_THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.71, 0.05)]
EDGE_THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.91, 0.05)]


class DenseGraph(nn.Module):
    def __init__(self, mode='spline', interior=6, radius=12.0, backbone='unet'):
        super().__init__()
        from monai.networks.nets import BasicUNet
        self.mode, self.m, self.radius = mode, interior, float(radius)
        features = (32, 32, 64, 128, 256, 32)
        self.unet = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5 + 32, features=features)
        if backbone.startswith('ukan'):  # tokenized KAN (or MLP control) in the two deepest encoder / first two decoder stages
            from models.ukan3d import kanify_basic_unet
            kanify_basic_unet(self.unet, features, kan=backbone != 'ukan_mlp', skip='add' if backbone.endswith('_add') else 'cat')
        with torch.no_grad():
            self.unet.final_conv.bias[0] = -4.6
        self.corridor = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv3d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(64, 64, 3, stride=(1, 2, 2), padding=1), nn.GELU())
        width = 64 * 12 * 3 * 3 + 4 + 2 * 32 + 2
        self.edge_head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 1))
        self.geometry_head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 3 * interior))
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)
        if mode == 'spline':
            basis, greville = cubic_basis(interior + 2, 64)
            self.register_buffer('basis', basis)
            self.register_buffer('anchor', greville[1:-1])
        else:
            self.register_buffer('anchor', torch.arange(1, interior + 1) / (interior + 1))

    def forward_image(self, images):
        return self.unet(images)

    def pair_features(self, volume, heat, a, b):
        """Corridor + endpoint features for pairs a->b (voxels) of one patch."""
        d, t, u, w = frame(a, b)
        s = torch.linspace(-0.1, 1.1, L, device=a.device)
        o = torch.linspace(-self.radius, self.radius, S, device=a.device)
        p = (a[:, None, None, None] + s[None, :, None, None, None] * d[:, None, None, None]
             + o[None, None, :, None, None] * u[:, None, None, None] + o[None, None, None, :, None] * w[:, None, None, None])
        grid = (p / (SIZE - 1) * 2 - 1).flip(-1)
        corridor = F.grid_sample(volume[None], grid[None].reshape(1, -1, S, S, 3), align_corners=True)
        corridor = corridor.view(volume.shape[0], len(a), L, S, S).transpose(0, 1)
        ends = torch.stack([a, b], 1)
        g = (ends / (SIZE - 1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3)
        point = F.grid_sample(torch.cat([volume, heat], 0)[None], g, align_corners=True).view(volume.shape[0] + 1, len(a), 2)
        endpoint = point[:-1].permute(1, 2, 0).reshape(len(a), -1)
        return torch.cat([self.corridor(corridor).flatten(1), d.norm(dim=-1, keepdim=True) / SIZE, d / SIZE,
                          endpoint, point[-1]], -1)

    def curves(self, x, a, b):
        d, t, u, w = frame(a, b)
        local = self.geometry_head(x).view(len(a), self.m, 3) * self.radius
        interior = (a[:, None] + self.anchor[None, :, None] * d[:, None]
                    + local[..., :1] * t[:, None] + local[..., 1:2] * u[:, None] + local[..., 2:] * w[:, None])
        points = torch.cat([a[:, None], interior, b[:, None]], 1)
        if self.mode == 'chord':
            return torch.stack([a, b], 1)
        return self.basis @ points if self.mode == 'spline' else polyline(points)


@torch.no_grad()
def peaks(out, threshold):
    heat = out[:, :1].float().sigmoid()
    keep = (heat == F.max_pool3d(heat, 3, 1, 1)) & (heat >= threshold)
    result = []
    for b in range(len(out)):
        idx = keep[b, 0].nonzero()
        off = out[b, 1:4].float()[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
        score = heat[b, 0][keep[b, 0]]
        order = score.argsort(descending=True)
        result.append(((idx.float() + off).clamp(-0.5, SIZE - 0.5)[order], score[order]))
    return result


def candidate_pairs(points_mm):
    """Undirected pairs (i<j) where j is among i's 16 nearest nodes or vice versa."""
    n = len(points_mm)
    if n < 2:
        return torch.zeros(0, 2, dtype=torch.long, device=points_mm.device)
    d = torch.cdist(points_mm, points_mm)
    d.fill_diagonal_(float('inf'))
    k = min(NEIGHBOURS, n - 1)
    nn_idx = d.topk(k, largest=False).indices
    i = torch.arange(n, device=d.device)[:, None].expand(-1, k)
    pairs = torch.stack([torch.minimum(i, nn_idx), torch.maximum(i, nn_idx)], -1).reshape(-1, 2)
    return torch.unique(pairs, dim=0)


def match(pred_mm, gt_mm, tol=2.0):
    from scipy.optimize import linear_sum_assignment
    if len(pred_mm) == 0 or len(gt_mm) == 0:
        return {}
    dist = torch.cdist(pred_mm, gt_mm).cpu().numpy()
    r, c = linear_sum_assignment(np.where(dist <= tol, dist, 1e6))
    return {int(i): int(j) for i, j in zip(r, c) if dist[i, j] <= tol}


def pair_labels(pairs, mapping, gt_edges):
    links = {tuple(sorted(map(int, e))) for e in gt_edges.tolist()}
    label = [int(a) in mapping and int(b) in mapping and tuple(sorted((mapping[int(a)], mapping[int(b)]))) in links
             for a, b in pairs.tolist()]
    return torch.tensor(label, dtype=torch.float32, device=pairs.device)


def guide_for(pairs, mapping, gt_edges, guides, gt_vox):
    """Dense reference (voxels) for positive pairs, oriented from the GT node matched to a."""
    index = {tuple(sorted(map(int, e))): k for k, e in enumerate(gt_edges.tolist())}
    refs = []
    for a, b in pairs.tolist():
        g = guides[index[tuple(sorted((mapping[a], mapping[b])))]]
        start = gt_vox[mapping[a]]
        refs.append(g if (g[0] - start).norm() <= (g[-1] - start).norm() else g.flip(0))
    return torch.stack(refs) if refs else guides[:0]


def patch_terms(model, feats, heat_map, nodes_vox, gt_vox, gt_edges, guides, metric, use_pred, jitter_mm, voxel_mm):
    """Edge BCE + geometry chamfer for one patch's candidate set."""
    if use_pred:
        cand = nodes_vox
    else:
        cand = gt_vox + torch.randn_like(gt_vox) * (jitter_mm / voxel_mm)
    if len(cand) < 2:
        return None
    cand_mm = cand @ metric.T
    pairs = candidate_pairs(cand_mm)
    mapping = match(cand_mm, gt_vox @ metric.T) if use_pred else {i: i for i in range(len(gt_vox))}
    labels = pair_labels(pairs, mapping, gt_edges)
    negatives = torch.nonzero(labels == 0).squeeze(-1)
    if len(negatives) > MAX_NEGATIVES:  # bound corridor memory; keep every positive
        keep = torch.cat([torch.nonzero(labels > 0).squeeze(-1), negatives[torch.randperm(len(negatives), device=negatives.device)[:MAX_NEGATIVES]]])
        pairs, labels = pairs[keep], labels[keep]
    x = model.pair_features(feats, heat_map, cand[pairs[:, 0]], cand[pairs[:, 1]])
    logits = model.edge_head(x).squeeze(-1)
    pos = labels.sum()
    edge = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=((len(labels) - pos) / pos.clamp_min(1)).clamp(max=20.0))
    geo = None
    positive = labels.bool()
    if positive.any() and model.mode != 'chord':
        pp = pairs[positive]
        curve = model.curves(x[positive], cand[pp[:, 0]], cand[pp[:, 1]])
        ref = guide_for(pp, mapping, gt_edges, guides, gt_vox)
        m = metric[None].expand(len(pp), 3, 3)
        geo = chamfer(to_mm(curve, m), to_mm(ref, m))
    return edge, geo


@torch.no_grad()
def predict(model, ds, device, node_threshold, workers, batch=8):
    """Raw per-patch predictions with all candidate edge probabilities (for threshold sweeps)."""
    model.eval()
    rows = []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=collate)
    offset = 0
    for images, nodes, edges, geometry in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model.forward_image(images.to(device))
        feats = out[:, 5:].float()
        heat = out[:, :1].float().sigmoid()
        for b, (p, score) in enumerate(peaks(out, node_threshold)):
            metric = geometry[b]['metric'].to(device) / SIZE
            pairs = candidate_pairs(p @ metric.T)
            probs = curves = None
            if len(pairs):
                x = model.pair_features(feats[b], heat[b], p[pairs[:, 0]], p[pairs[:, 1]])
                probs = model.edge_head(x).squeeze(-1).sigmoid().cpu().numpy()
                curves = (model.curves(x, p[pairs[:, 0]], p[pairs[:, 1]]) / SIZE).cpu().numpy()
            record = ds.records[offset + b]
            rows.append(dict(sample_id=record.sample_id, cache=str(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz'),
                             nodes=(p / SIZE).cpu().numpy(), node_scores=score.cpu().numpy(),
                             pairs=pairs.cpu().numpy(), edge_scores=probs, curves=curves))
        offset += len(images)
    model.train()
    return rows


def evaluate(rows, node_threshold, edge_threshold, geometry=True):
    from diagnose_branch_relations import match_nodes
    from ixi_query_diagnostics import edge_accounting
    from metrics.common_vessel_metrics import evaluate_common_vessels, aggregate_common_vessels
    totals = dict(node_tp=0, node_fp=0, node_fn=0, tp=0, fp=0, fn=0, missing_endpoint_edges=0, MM_fp=0, MU_fp=0, UU_fp=0,
                  beta0_error=0, beta1_error=0)
    common = []
    for r in rows:
        with np.load(r['cache']) as z:
            gt_nodes, gt_edges, metric = z['nodes'], z['edges'].reshape(-1, 2), z['metric']
        keep = np.nonzero(r['node_scores'] >= node_threshold)[0]
        remap = {int(k): i for i, k in enumerate(keep)}
        nodes = r['nodes'][keep]
        pairs = r['pairs'] if r['edge_scores'] is not None else np.zeros((0, 2), int)
        valid = np.isin(pairs, keep).all(1) if len(pairs) else np.zeros(0, bool)
        sub = pairs[valid]
        scores = r['edge_scores'][valid] if len(sub) else np.zeros(0)
        local = np.array([[remap[a], remap[b]] for a, b in sub], int).reshape(-1, 2)
        mapping = match_nodes(nodes @ metric.T, gt_nodes @ metric.T, 2.0)
        row, chosen, *_ = edge_accounting(np.arange(len(nodes)), local, scores, scores >= edge_threshold, mapping, gt_edges, len(gt_nodes))
        for k in totals:
            totals[k] += row[k]
        if geometry:
            sel = np.nonzero(valid)[0][chosen]
            edges = local[chosen]
            prediction = dict(nodes=nodes, edges=edges)
            if r['curves'] is not None and len(edges):
                prediction['edge_polylines'] = [np.concatenate([nodes[e[0]][None], r['curves'][s][1:-1], nodes[e[1]][None]])
                                                for s, e in zip(sel, edges)]
            common.append(evaluate_common_vessels(prediction, r['cache']))
    f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    result = dict(node_threshold=node_threshold, edge_threshold=edge_threshold,
                  node_f1=f1(totals['node_tp'], totals['node_fp'], totals['node_fn']),
                  node_precision=totals['node_tp'] / max(1, totals['node_tp'] + totals['node_fp']),
                  node_recall=totals['node_tp'] / max(1, totals['node_tp'] + totals['node_fn']),
                  edge_f1=f1(totals['tp'], totals['fp'], totals['fn']),
                  edge_precision=totals['tp'] / max(1, totals['tp'] + totals['fp']),
                  edge_recall=totals['tp'] / max(1, totals['tp'] + totals['fn']),
                  missed_edges_without_endpoint_pair=totals['missing_endpoint_edges'] / max(1, totals['fn']),
                  false_edges_MM_MU_UU=[totals['MM_fp'], totals['MU_fp'], totals['UU_fp']],
                  beta0_mae=totals['beta0_error'] / len(rows), beta1_mae=totals['beta1_error'] / len(rows), patches=len(rows))
    if geometry:
        agg = aggregate_common_vessels(common)
        result.update({k: agg[k] for k in ('common_branch_f1', 'common_curve_f1', 'common_curve_precision',
                                           'common_curve_recall', 'common_guide_distance_mm', 'common_anchor_f1')})
    return result


def select_node_threshold(rows):
    """Node F1 does not depend on edges; rows built at the lowest threshold suffice."""
    return max(NODE_THRESHOLDS, key=lambda t: evaluate(rows, t, 0.5, geometry=False)['node_f1'])


def select_edge_threshold(rows, node):
    """Rows must be built at `node` so candidates are kNN over the retained nodes."""
    return max(EDGE_THRESHOLDS, key=lambda t: evaluate(rows, node, t, geometry=False)['edge_f1'])


def warm_start(model, path, device):
    """Load every tensor whose name and shape match (U-Net, corridor CNN, ...); report the rest."""
    source = torch.load(path, map_location=device)['model']
    own = model.state_dict()
    loaded = {k: v for k, v in source.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(loaded, strict=False)
    return dict(source=str(path), loaded=len(loaded), fresh=sorted(set(own) - set(loaded)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['IXI', 'BRAVA', 'SyntheticMRI'], required=True)
    p.add_argument('--mode', choices=['spline', 'polyline', 'chord'], default='spline')
    p.add_argument('--interior', type=int, default=6)
    p.add_argument('--updates', type=int, default=15000)
    p.add_argument('--warmup-gt', type=int, default=2000, help='Updates with GT-node candidates only')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--train-peak-threshold', type=float, default=0.2)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--eval-batch', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--init-from', type=Path, help='Warm start from a dense-model checkpoint (matching tensors)')
    p.add_argument('--run-name', default=None, help='Output subfolder name')
    p.add_argument('--evaluate-only', action='store_true', help='Re-evaluate the saved model.pt')
    p.add_argument('--backbone', choices=['unet', 'ukan', 'ukan_mlp', 'ukan_add'], default='unet',
                   help='ukan: U-KAN-style deepest stages (models/ukan3d.py); ukan_mlp: same blocks with MLPs; ukan_add: additive skips')
    a = p.parse_args()
    torch.manual_seed(364505)
    np.random.seed(364505)
    out_dir = a.output / a.dataset / (a.run_name or (a.mode + (str(a.interior) if a.mode != 'chord' else '')))
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text())['datasets'][a.dataset]
    train_ids, val_ids, monitor = plan['train_ids'], plan['val_ids'], plan['monitor_train']
    if a.smoke:
        train_ids, val_ids, monitor = train_ids[:8], val_ids[:4], train_ids[:4]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = make_dataset(a.dataset, train_ids, augment=True)
    model = DenseGraph(a.mode, a.interior, backbone=a.backbone).to(device)
    if a.init_from and not a.evaluate_only:
        (out_dir / 'warm-start.json').write_text(json.dumps(warm_start(model, a.init_from, device), indent=1))
    if a.evaluate_only:
        model.load_state_dict(torch.load(out_dir / 'model.pt', map_location=device)['model'])
        final_evaluation(model, a, monitor, val_ids, device, out_dir, name='final-v2.json')
        return
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.updates, pct_start=0.05)
    sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=a.updates * a.batch)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                         collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
    log = (out_dir / 'training.jsonl').open('a')
    start, step = time.time(), 0
    for images, nodes, edges, geometry in loader:
        step += 1
        heat_t, offset_t, peak_t, line_t = targets(nodes, geometry, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            out = model.forward_image(images.to(device))
        node_loss = focal(out[:, :1].float(), heat_t)
        m = peak_t.unsqueeze(1).expand_as(offset_t)
        off_loss = F.l1_loss(out[:, 1:4].float()[m], offset_t[m]) if m.any() else out.sum() * 0
        line_loss = F.binary_cross_entropy_with_logits(out[:, 4:5].float(), line_t, pos_weight=torch.tensor(5.0, device=device))
        feats, heat = out[:, 5:].float(), out[:, :1].float().sigmoid().detach()
        use_pred = step > a.warmup_gt
        predicted = peaks(out.detach(), a.train_peak_threshold) if use_pred else [(None, None)] * len(images)
        edge_terms, geo_terms = [], []
        for b in range(len(images)):
            metric = geometry[b]['metric'].to(device) / SIZE
            gt_vox = nodes[b].to(device) * SIZE
            for pred_mode in ([False, True] if use_pred else [False]):
                cand = predicted[b][0][:MAX_PEAKS] if pred_mode else None
                res = patch_terms(model, feats[b], heat[b], cand, gt_vox, edges[b].to(device),
                                  geometry[b]['guides'].to(device) * SIZE, metric, pred_mode, 0.5, float(metric[0].norm()))
                if res is None:
                    continue
                edge_terms.append(res[0])
                if res[1] is not None:
                    geo_terms.append(res[1])
        edge_loss = torch.stack(edge_terms).mean() if edge_terms else out.sum() * 0
        geo_loss = torch.cat(geo_terms).mean() if geo_terms else out.sum() * 0
        loss = node_loss + off_loss + line_loss + edge_loss + geo_loss
        assert torch.isfinite(loss), 'nonfinite loss'
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or a.smoke:
            row = dict(step=step, seconds=time.time() - start, node=float(node_loss), offset=float(off_loss), centerline=float(line_loss),
                       edge=float(edge_loss), geometry_mm=float(geo_loss), predicted_candidates=use_pred)
            log.write(json.dumps(row) + '\n')
            log.flush()
            print('STEP ' + json.dumps(row), flush=True)
    torch.save(dict(model=model.state_dict(), args={k: str(v) for k, v in vars(a).items()}), out_dir / 'model.pt')
    final_evaluation(model, a, monitor, val_ids, device, out_dir)


def final_evaluation(model, a, monitor, val_ids, device, out_dir, name='final.json'):
    monitor_ds = make_dataset(a.dataset, monitor, augment=False)
    val_ds = make_dataset(a.dataset, val_ids, augment=False)
    node_t = select_node_threshold(predict(model, monitor_ds, device, min(NODE_THRESHOLDS), a.workers, a.eval_batch))
    train_rows = predict(model, monitor_ds, device, node_t, a.workers, a.eval_batch)
    edge_t = select_edge_threshold(train_rows, node_t)
    val_rows = predict(model, val_ds, device, node_t, a.workers, a.eval_batch)
    import pickle  # per-patch rows for paired subject-level comparisons
    (out_dir / name.replace('.json', '-validation-rows.pkl')).write_bytes(pickle.dumps(val_rows))
    fixed = predict(model, val_ds, device, 0.5, a.workers, a.eval_batch)
    result = dict(dataset=a.dataset, mode=a.mode, backbone=getattr(a, 'backbone', 'unet'), interior=a.interior, updates=a.updates, batch=a.batch, lr=a.lr,
                  thresholds=dict(node=node_t, edge=edge_t, policy='train monitor only; candidates = 16-NN over retained nodes'),
                  train_monitor=evaluate(train_rows, node_t, edge_t), validation=evaluate(val_rows, node_t, edge_t),
                  validation_fixed_0p5=evaluate(fixed, 0.5, 0.5))
    (out_dir / name).write_text(json.dumps(result, indent=1))
    print('FINAL ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
