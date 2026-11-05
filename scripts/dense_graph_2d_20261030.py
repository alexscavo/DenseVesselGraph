#!/usr/bin/env python3
"""Native 2D port of the WS3 two-stage dense vessel-graph model (30 Oct 2026).

Faithful 2D analogue of GNBM scripts/dense_graph_20261030.py (3D), changing only what
2D requires.

Stage 1 (nodes): MONAI BasicUNet(spatial_dims=2, in_channels=3) at full 128^2 resolution
-> CenterNet node heatmap logit (ch0), 2 sub-pixel XY offsets (ch1-2), centreline logit
(ch3) and 32 feature channels (ch4-35). Nodes = heatmap local maxima (3x3 NMS) above a
threshold.
Stage 2 (edges): candidate pairs = each node's 16 nearest nodes (native units). For each
pair the features in a 24 (along) x 11 (across, +-12 px) corridor around the chord feed a
shared 2D CNN with two heads: an edge logit and the 6 interior controls of a clamped
cubic B-spline (K8), or polyline vertices / straight chord.

No Hungarian assignment. Candidate pairs are built on predicted peaks, a peak inherits
the GT node it is matched to (one-to-one) within the 2-native-unit evaluation tolerance,
and a pair is positive iff its matched GT nodes share a GT edge. Jittered GT-node pairs
are used alone during warm-up and alongside predicted-peak pairs afterwards. The
candidate rule (16-NN over retained nodes) is the same at training and inference.
Thresholds (node, then edge) are selected on the fixed training monitor subset only.

Data, splits, D4 augmentation, x/255-0.5 normalization and 128^2 crops are those of
scripts/train_native2d_long.py (StudyDataset, DATA, subset). Evaluation reuses
native2d.evaluation.positional / common_metrics and the long study's aggregate, so the
numbers are directly comparable with the native2D RelationFormer final-heldout.json.

Coordinates: pixel XY = normalized XY * 128 (x = column, pixel centres at integers);
metric/128 maps pixels to native units (pixels; micron for REAVER).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.spline import cubic_basis  # noqa: E402
from scripts.train_native2d_long import StudyDataset, DATA, SEED, read, subset, aggregate  # noqa: E402
from native2d.evaluation import positional, common_metrics, match_nodes, _boxes  # noqa: E402
from training.evaluation.metrics import detection_ap_ar, graph_betti_numbers  # noqa: E402

SIZE = 128
SIGMA = 1.5                # pixels
L, S = 24, 11              # corridor samples along / across the chord
GUIDE = 64                 # dense GT path resampled to 64 points per edge (3D cache 'guides')
NEIGHBOURS = 16
MAX_NEGATIVES = 256
MAX_PEAKS = 200
TOLERANCE = 2.0            # native units; label matching == evaluation tolerance
JITTER = 0.5               # native units (3D: 0.5 mm with a 2 mm tolerance)
FEAT = 4                   # first feature channel
NODE_THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.71, 0.05)]
EDGE_THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.91, 0.05)]
QUERIES = read(ROOT / 'experiments/native2d_mixed_20261028/plan.json')['common_queries']


# ----------------------------------------------------------------------------- data
def make_dataset(dataset, split, augment, preload=False):
    return StudyDataset(DATA / dataset / 'manifest.json', split, augment=augment, max_queries=QUERIES, preload=preload)


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch]


def resample(path, n):
    """Uniform arc-length resampling of a (P,2) polyline to n points."""
    path = np.asarray(path, float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if arc[-1] <= 1e-12:
        return np.repeat(path[:1], n, 0)
    t = np.linspace(0, arc[-1], n)
    return np.stack([np.interp(t, arc, path[:, k]) for k in range(2)], 1)


def dense_paths(target):
    off = target['path_offsets']
    return [target['full_paths'][a:b] * SIZE for a, b in zip(off[:-1], off[1:])]


def patch(target, device):
    """Per-patch tensors in pixel XY: nodes, edges, guides (E,64,2), metric (px -> native)."""
    assert tuple(np.asarray(target['shape_hw']).tolist()) == (SIZE, SIZE)
    nodes = torch.as_tensor(target['nodes'] * SIZE, dtype=torch.float32, device=device).reshape(-1, 2)
    edges = torch.as_tensor(target['edges'], dtype=torch.long, device=device).reshape(-1, 2)
    paths = dense_paths(target)
    guides = torch.as_tensor(np.stack([resample(p, GUIDE) for p in paths]) if paths else np.zeros((0, GUIDE, 2)),
                             dtype=torch.float32, device=device)
    metric = torch.as_tensor(target['metric'], dtype=torch.float32, device=device) / SIZE
    return nodes, edges, guides, metric


GRID = None


def targets(batch_targets, device):
    """Heatmap (peak at rounded pixel), XY offsets, peak mask and dilated centreline."""
    global GRID
    if GRID is None or GRID.device != torch.device(device):
        axis = torch.arange(SIZE, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        GRID = torch.stack([xx, yy], -1).view(-1, 2)  # row-major (y, x) flattening, XY values
    B = len(batch_targets)
    heat = torch.zeros(B, SIZE * SIZE, device=device)
    offset = torch.zeros(B, 2, SIZE, SIZE, device=device)
    peak = torch.zeros(B, SIZE, SIZE, dtype=torch.bool, device=device)
    line = torch.zeros(B, 1, SIZE, SIZE, device=device)
    for b, t in enumerate(batch_targets):
        v = torch.as_tensor(t['nodes'] * SIZE, dtype=torch.float32, device=device).reshape(-1, 2)
        if len(v):
            c = v.round().clamp(0, SIZE - 1)
            for chunk in torch.split(c, 64):
                d2 = ((GRID[:, None, :] - chunk[None]) ** 2).sum(-1)
                heat[b] = torch.maximum(heat[b], torch.exp(-d2 / (2 * SIGMA ** 2)).amax(1))
            i = c.long()
            peak[b, i[:, 1], i[:, 0]] = True
            offset[b, :, i[:, 1], i[:, 0]] = (v - c).T
        paths = dense_paths(t)
        if paths:
            j = torch.as_tensor(np.concatenate([resample(p, 256) for p in paths]), device=device).round().clamp(0, SIZE - 1).long()
            line[b, 0, j[:, 1], j[:, 0]] = 1.0
    line = F.max_pool2d(line, 3, 1, 1)
    return heat.view(B, 1, SIZE, SIZE), offset, peak, line


def focal(logits, heat):
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float()
    loss = -(pos * (1 - p) ** 2 * p.log() + (1 - pos) * (1 - heat) ** 4 * p ** 2 * (1 - p).log())
    return loss.sum() / pos.sum().clamp_min(1)


# ----------------------------------------------------------------------------- geometry
def frame(a, b):
    """2D local frame: d = b - a, unit tangent t, unit normal u = rot90(t)."""
    d = b - a
    t = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    u = torch.stack([-t[..., 1], t[..., 0]], -1)
    return d, t, u


def polyline(points, samples=64):
    P = points.shape[1]
    s = torch.linspace(0, P - 1, samples, device=points.device)
    i = s.floor().long().clamp(max=P - 2)
    w = (s - i)[None, :, None]
    return points[:, i] * (1 - w) + points[:, i + 1] * w


def to_mm(points, metric):
    return torch.einsum('epi,eji->epj', points, metric)


def chamfer(pred_mm, ref_mm):
    d = torch.cdist(polyline(pred_mm, 128), polyline(ref_mm, 128))
    return (d.amin(-1).mean(-1) + d.amin(-2).mean(-1)) / 2


# ----------------------------------------------------------------------------- model
class DenseGraph2D(nn.Module):
    def __init__(self, mode='spline', interior=6, radius=12.0):
        super().__init__()
        from monai.networks.nets import BasicUNet
        self.mode, self.m, self.radius = mode, interior, float(radius)
        self.unet = BasicUNet(spatial_dims=2, in_channels=3, out_channels=FEAT + 32, features=(32, 32, 64, 128, 256, 32))
        with torch.no_grad():
            self.unet.final_conv.bias[0] = -4.6
        self.corridor = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=(1, 2), padding=1), nn.GELU())
        width = 64 * 12 * 3 + 3 + 2 * 32 + 2
        self.edge_head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 1))
        self.geometry_head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 2 * interior))
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
        """Corridor + endpoint features for pairs a->b (pixel XY) of one patch."""
        d, t, u = frame(a, b)
        s = torch.linspace(-0.1, 1.1, L, device=a.device)
        o = torch.linspace(-self.radius, self.radius, S, device=a.device)
        p = a[:, None, None] + s[None, :, None, None] * d[:, None, None] + o[None, None, :, None] * u[:, None, None]
        grid = p / (SIZE - 1) * 2 - 1  # XY pixel centres -> (x=W, y=H) in [-1, 1]; no flip needed in 2D
        corridor = F.grid_sample(volume[None], grid.reshape(1, -1, S, 2), align_corners=True)
        corridor = corridor.view(volume.shape[0], len(a), L, S).transpose(0, 1)
        ends = torch.stack([a, b], 1)
        g = (ends / (SIZE - 1) * 2 - 1).view(1, -1, 1, 2)
        point = F.grid_sample(torch.cat([volume, heat], 0)[None], g, align_corners=True).view(volume.shape[0] + 1, len(a), 2)
        endpoint = point[:-1].permute(1, 2, 0).reshape(len(a), -1)
        return torch.cat([self.corridor(corridor).flatten(1), d.norm(dim=-1, keepdim=True) / SIZE, d / SIZE,
                          endpoint, point[-1]], -1)

    def controls(self, x, a, b):
        """[a, M interior points, b] in pixel XY (spline controls / polyline vertices)."""
        d, t, u = frame(a, b)
        local = self.geometry_head(x).view(len(a), self.m, 2) * self.radius
        interior = a[:, None] + self.anchor[None, :, None] * d[:, None] + local[..., :1] * t[:, None] + local[..., 1:] * u[:, None]
        return torch.cat([a[:, None], interior, b[:, None]], 1)

    def curves(self, x, a, b):
        if self.mode == 'chord':
            return torch.stack([a, b], 1)
        points = self.controls(x, a, b)
        return self.basis @ points if self.mode == 'spline' else polyline(points)


@torch.no_grad()
def peaks(out, threshold):
    heat = out[:, :1].float().sigmoid()
    keep = (heat == F.max_pool2d(heat, 3, 1, 1)) & (heat >= threshold)
    result = []
    for b in range(len(out)):
        idx = keep[b, 0].nonzero()  # (row, col)
        off = out[b, 1:3].float()[:, idx[:, 0], idx[:, 1]].T  # (dx, dy)
        score = heat[b, 0][keep[b, 0]]
        order = score.argsort(descending=True)
        result.append(((idx.flip(-1).float() + off).clamp(-0.5, SIZE - 0.5)[order], score[order]))
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


def match(pred_mm, gt_mm, tol=TOLERANCE):
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


def guide_for(pairs, mapping, gt_edges, guides, gt_px):
    """Dense reference (pixels) for positive pairs, oriented from the GT node matched to a."""
    index = {tuple(sorted(map(int, e))): k for k, e in enumerate(gt_edges.tolist())}
    refs = []
    for a, b in pairs.tolist():
        g = guides[index[tuple(sorted((mapping[a], mapping[b])))]]
        start = gt_px[mapping[a]]
        refs.append(g if (g[0] - start).norm() <= (g[-1] - start).norm() else g.flip(0))
    return torch.stack(refs) if refs else guides[:0]


def patch_terms(model, feats, heat_map, nodes_px, gt_px, gt_edges, guides, metric, use_pred, jitter_units, pixel_units):
    """Edge BCE + geometry chamfer for one patch's candidate set."""
    cand = nodes_px if use_pred else gt_px + torch.randn_like(gt_px) * (jitter_units / pixel_units)
    if len(cand) < 2:
        return None
    cand_mm = cand @ metric.T
    pairs = candidate_pairs(cand_mm)
    mapping = match(cand_mm, gt_px @ metric.T) if use_pred else {i: i for i in range(len(gt_px))}
    labels = pair_labels(pairs, mapping, gt_edges)
    negatives = torch.nonzero(labels == 0).squeeze(-1)
    if len(negatives) > MAX_NEGATIVES:
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
        ref = guide_for(pp, mapping, gt_edges, guides, gt_px)
        m = metric[None].expand(len(pp), 2, 2)
        geo = chamfer(to_mm(curve, m), to_mm(ref, m))
    return edge, geo, int(pos), len(labels)


def train_losses(model, images, batch_targets, device, use_pred, train_peak_threshold):
    heat_t, offset_t, peak_t, line_t = targets(batch_targets, device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
        out = model.forward_image(images.to(device))
    node_loss = focal(out[:, :1].float(), heat_t)
    m = peak_t.unsqueeze(1).expand_as(offset_t)
    off_loss = F.l1_loss(out[:, 1:3].float()[m], offset_t[m]) if m.any() else out.sum() * 0
    line_loss = F.binary_cross_entropy_with_logits(out[:, 3:4].float(), line_t, pos_weight=torch.tensor(5.0, device=device))
    feats, heat = out[:, FEAT:].float(), out[:, :1].float().sigmoid().detach()
    predicted = peaks(out.detach(), train_peak_threshold) if use_pred else [(None, None)] * len(images)
    edge_terms, geo_terms, stats = [], [], dict(pred_candidates=0, positives=0, pairs=0)
    for b, t in enumerate(batch_targets):
        gt_px, gt_edges, guides, metric = patch(t, device)
        for pred_mode in ([False, True] if use_pred else [False]):
            cand = predicted[b][0][:MAX_PEAKS] if pred_mode else None
            if pred_mode:
                stats['pred_candidates'] += len(cand)
            res = patch_terms(model, feats[b], heat[b], cand, gt_px, gt_edges, guides, metric, pred_mode, JITTER, float(metric[0].norm()))
            if res is None:
                continue
            edge_terms.append(res[0])
            if res[1] is not None:
                geo_terms.append(res[1])
            stats['positives'] += res[2]
            stats['pairs'] += res[3]
    edge_loss = torch.stack(edge_terms).mean() if edge_terms else out.sum() * 0
    geo_loss = torch.cat(geo_terms).mean() if geo_terms else out.sum() * 0
    return dict(node=node_loss, offset=off_loss, centerline=line_loss, edge=edge_loss, geometry=geo_loss), stats


# ----------------------------------------------------------------------------- inference / evaluation
RAW_THRESHOLD = min(NODE_THRESHOLDS)


@torch.no_grad()
def predict_out(model, out, batch_targets, rows, node_threshold, chunk=4096):
    """Per patch: raw peaks (>= 0.10) + all 16-NN pairs over peaks >= node_threshold."""
    feats = out[:, FEAT:].float()
    heat = out[:, :1].float().sigmoid()
    result = []
    for b, (p, score) in enumerate(peaks(out, RAW_THRESHOLD)):
        metric = torch.as_tensor(batch_targets[b]['metric'], dtype=torch.float32, device=out.device) / SIZE
        keep = torch.nonzero(score >= node_threshold).squeeze(-1)  # scores sorted descending: a prefix
        q = p[keep]
        pairs = candidate_pairs(q @ metric.T)
        probs = np.zeros(0)
        controls = np.zeros((0, model.m + 2, 2))
        if len(pairs):
            pr, ct = [], []
            for c in torch.split(pairs, chunk):
                x = model.pair_features(feats[b], heat[b], q[c[:, 0]], q[c[:, 1]])
                pr.append(model.edge_head(x).squeeze(-1).sigmoid())
                ct.append(model.controls(x, q[c[:, 0]], q[c[:, 1]]) if model.mode != 'chord' else torch.stack([q[c[:, 0]], q[c[:, 1]]], 1))
            probs = torch.cat(pr).cpu().numpy()
            controls = (torch.cat(ct) / SIZE).cpu().numpy()
        result.append(dict(sample_id=rows[b]['sample_id'], source_image=rows[b]['subject_id'], target=batch_targets[b],
                           built_at=node_threshold, nodes=(p / SIZE).cpu().numpy(), node_scores=score.cpu().numpy(),
                           pairs=pairs.cpu().numpy(), edge_scores=probs, controls=controls))
    return result


@torch.no_grad()
def predict(model, ds, ids, device, node_threshold, workers, batch=8):
    model.eval()
    rows = []
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, ids), batch_size=batch, num_workers=workers, collate_fn=collate)
    for images, batch_targets, meta in loader:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            out = model.forward_image(images.to(device))
        rows.extend(predict_out(model, out, batch_targets, meta, node_threshold))
    model.train()
    return rows


def decode(r, node_threshold, edge_threshold, mode):
    """Retained nodes/edges (+ curves) of one predicted patch, at given thresholds."""
    assert node_threshold >= r['built_at'] - 1e-9, 'pairs must be built at or below the node threshold'
    keep = np.nonzero(r['node_scores'] >= node_threshold)[0]
    remap = {int(k): i for i, k in enumerate(keep)}
    pairs = r['pairs']
    valid = np.isin(pairs, keep).all(1) if len(pairs) else np.zeros(0, bool)
    local = np.array([[remap[a], remap[b]] for a, b in pairs[valid]], int).reshape(-1, 2)
    scores = r['edge_scores'][valid] if len(local) else np.zeros(0)
    chosen = scores >= edge_threshold
    nodes = r['nodes'][keep]
    decoded = dict(nodes=nodes, edges=local[chosen], node_scores=r['node_scores'][keep], edge_scores=scores[chosen])
    if mode != 'chord' and chosen.any():
        ctl = r['controls'][valid][chosen].copy()
        ctl[:, 0], ctl[:, -1] = nodes[local[chosen][:, 0]], nodes[local[chosen][:, 1]]  # exact float endpoints
        if mode == 'spline':
            decoded['spline_controls'] = ctl
        else:
            decoded['edge_polylines'] = list(polyline(torch.as_tensor(ctl), 64).numpy())
    return decoded, local, scores


def crop_metrics(r, node_threshold, edge_threshold, mode, geometry):
    """Same per-crop record as train_native2d_study.metrics (minus RelationFormer-only ranking)."""
    gt = r['target']
    m = np.asarray(gt['metric'])
    gn, ge = gt['nodes'], np.asarray(gt['edges']).reshape(-1, 2)
    decoded, local, scores = decode(r, node_threshold, edge_threshold, mode)
    inserted = gt.get('inserted')  # explicit mask of recontracted targets; None: degree-based fallback
    result = {str(t): positional(decoded['nodes'] @ m.T, decoded['edges'], gn @ m.T, ge, t, inserted=inserted) for t in [1., 2., 4.]}
    raw = r['nodes']
    p0, p1 = graph_betti_numbers(len(decoded['nodes']), decoded['edges'])
    g0, g1 = graph_betti_numbers(len(gn), ge)
    mapping, _ = match_nodes(decoded['nodes'] @ m.T, gn @ m.T, TOLERANCE)
    gt_links = {tuple(sorted(e)) for e in ge.tolist()}
    candidate_links = {tuple(sorted((mapping[a], mapping[b]))) for a, b in local.tolist() if a in mapping and b in mapping}
    result.update(sample_id=r['sample_id'], source_image=r['source_image'], units=str(np.asarray(gt['units']).item()),
                  predicted_nodes=len(decoded['nodes']), target_nodes=len(gn), predicted_edges=len(decoded['edges']), target_edges=len(ge),
                  predicted_beta0=p0, target_beta0=g0, predicted_beta1=p1, target_beta1=g1, beta0_signed_error=p0 - g0, beta1_signed_error=p1 - g1,
                  raw_node_recall={str(t): positional(raw @ m.T, np.empty((0, 2), int), gn @ m.T, ge, t, inserted=inserted)['node']['recall'] for t in [1., 2., 4.]},
                  gt_edges_in_candidates=len(gt_links & candidate_links), candidate_pairs=len(local))
    if geometry:
        result['node_mAP'], result['node_mAR'] = detection_ap_ar(_boxes(raw), r['node_scores'], _boxes(gn), max_detections=max(1, len(raw)))
        result['edge_mAP'], result['edge_mAR'] = detection_ap_ar(_boxes(decoded['nodes'], local), scores, _boxes(gn, ge), max_detections=300)
        result['spline'] = common_metrics(decoded, gt)
        result['chord'] = common_metrics({k: v for k, v in decoded.items() if k not in ('spline_controls', 'edge_polylines')}, gt)
        assert result['spline']['common_branch_f1'] == result['chord']['common_branch_f1']
    else:
        result.update(node_mAP=np.nan, node_mAR=np.nan, edge_mAP=np.nan, edge_mAR=np.nan)
    return result


def summarize(crops, geometry):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # nanmean over the no-geometry sweep
        out = aggregate(crops)
    n_edges = sum(c['target_edges'] for c in crops)
    out['gt_edge_candidate_coverage_2'] = sum(c['gt_edges_in_candidates'] for c in crops) / n_edges if n_edges else None
    out['predicted_nodes_per_crop'] = float(np.mean([c['predicted_nodes'] for c in crops]))
    out['predicted_edges_per_crop'] = float(np.mean([c['predicted_edges'] for c in crops]))
    if geometry:
        for name in ('spline', 'chord'):  # pooled (micro) common-v2, as aggregate_common_vessels in 3D
            g = [c[name] for c in crops]
            tp, fp, fn = (sum(x[f'common_branch_{k}_total'] for x in g) for k in ('tp', 'fp', 'fn'))
            sp, np_ = sum(x['common_curve_supported_pred_samples_mean'] for x in g), sum(x['common_curve_pred_samples_mean'] for x in g)
            sg, ng = sum(x['common_curve_supported_gt_samples_mean'] for x in g), sum(x['common_curve_gt_samples_mean'] for x in g)
            prec, rec = (sp / np_ if np_ else 1.), (sg / ng if ng else 1.)
            out[name + '_pooled'] = dict(common_branch_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                                         common_curve_precision=prec, common_curve_recall=rec,
                                         common_curve_f1=2 * prec * rec / (prec + rec) if prec + rec else 0.)
    return out


METRIC_WORKERS = 1  # CPU processes for the (slow, exact) common-v2 curve metrics


def _crop_metrics(job):
    return crop_metrics(*job)


def evaluate(rows, node_threshold, edge_threshold, mode, geometry=True, keep_crops=False):
    jobs = [(r, node_threshold, edge_threshold, mode, geometry) for r in rows]
    if geometry and METRIC_WORKERS > 1 and len(rows) > 8:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(METRIC_WORKERS) as pool:  # children run numpy-only metric code
            crops = pool.map(_crop_metrics, jobs, chunksize=4)
    else:
        crops = [_crop_metrics(j) for j in jobs]
    result = dict(node_threshold=node_threshold, edge_threshold=edge_threshold, summary=summarize(crops, geometry))
    if keep_crops:
        result['samples'] = crops
    return result


def select_node_threshold(rows, mode):
    return max(NODE_THRESHOLDS, key=lambda t: evaluate(rows, t, 0.5, mode, geometry=False)['summary']['2.0']['node']['f1'])


def select_edge_threshold(rows, node, mode):
    return max(EDGE_THRESHOLDS, key=lambda t: evaluate(rows, node, t, mode, geometry=False)['summary']['2.0']['edge']['f1'])


def headline(result):
    s = result['summary']
    h = {t: (round(s[t]['node']['f1'], 4), round(s[t]['edge']['f1'], 4)) for t in ('1.0', '2.0', '4.0')}
    if 'spline' in s:
        h['K8_branch_curve_macro'] = (round(s['spline']['common_branch_f1'], 4), round(s['spline']['common_curve_f1'], 4))
        h['K8_branch_curve_pooled'] = (round(s['spline_pooled']['common_branch_f1'], 4), round(s['spline_pooled']['common_curve_f1'], 4))
    return h


def final_evaluation(model, a, monitor_ids, val_ids, train_eval, val_ds, device, out_dir, name='final.json'):
    t0 = time.time()
    node_t = select_node_threshold(predict(model, train_eval, monitor_ids, device, RAW_THRESHOLD, a.workers, a.eval_batch), a.mode)
    train_rows = predict(model, train_eval, monitor_ids, device, node_t, a.workers, a.eval_batch)
    edge_t = select_edge_threshold(train_rows, node_t, a.mode)
    train_fixed = predict(model, train_eval, monitor_ids, device, 0.5, a.workers, a.eval_batch)
    val_rows = predict(model, val_ds, val_ids, device, node_t, a.workers, a.eval_batch)
    val_fixed = val_rows if node_t == 0.5 else predict(model, val_ds, val_ids, device, 0.5, a.workers, a.eval_batch)
    result = dict(dataset=a.dataset, mode=a.mode, interior=a.interior, updates=a.updates, warmup_gt=a.warmup_gt, batch=a.batch, lr=a.lr,
                  units=str(np.asarray(val_rows[0]['target']['units']).item()) if val_rows else None,
                  train_crops=len(train_eval), monitor_crops=len(monitor_ids), validation_crops=len(val_ids),
                  thresholds=dict(node=node_t, edge=edge_t, policy='train monitor (subset() of training split) only; node by node F1@2, then edge by edge F1@2; candidates = 16-NN over retained nodes'),
                  train_monitor=evaluate(train_rows, node_t, edge_t, a.mode),
                  train_monitor_fixed_0p5=evaluate(train_fixed, 0.5, 0.5, a.mode))
    v_sel = evaluate(val_rows, node_t, edge_t, a.mode, keep_crops=True)
    v_fix = evaluate(val_fixed, 0.5, 0.5, a.mode, keep_crops=True)
    result['validation'] = {k: v for k, v in v_sel.items() if k != 'samples'}
    result['validation_fixed_0p5'] = {k: v for k, v in v_fix.items() if k != 'samples'}
    result['evaluation_seconds'] = time.time() - t0
    (out_dir / name.replace('.json', '-validation-crops.pkl')).write_bytes(pickle.dumps(dict(selected=v_sel['samples'], fixed_0p5=v_fix['samples'])))
    (out_dir / name).write_text(json.dumps(result, indent=1, default=float))
    if a.report_dir:
        a.report_dir.mkdir(parents=True, exist_ok=True)
        (a.report_dir / f'{a.dataset}-{out_dir.name}-{name}').write_text(json.dumps(result, indent=1, default=float))
    print('FINAL ' + json.dumps(dict(dataset=a.dataset, thresholds=result['thresholds'], train_monitor=headline(result['train_monitor']),
                                     train_monitor_fixed_0p5=headline(result['train_monitor_fixed_0p5']),
                                     validation=headline(result['validation']), validation_fixed_0p5=headline(result['validation_fixed_0p5']),
                                     seconds=result['evaluation_seconds'])), flush=True)
    return result


# ----------------------------------------------------------------------------- CPU checks
def cpu_checks(a):
    """Target rendering, oracle decode through the real evaluation path, tiny CPU train step."""
    torch.set_num_threads(2)
    report = {}
    for dataset in ('avwide', 'reaver', 'octasynth'):
        train_eval = make_dataset(dataset, 'train', augment=False)
        aug = make_dataset(dataset, 'train', augment=True)
        ids = subset(train_eval)[:a.check_crops]
        # 1. path orientation / D4 consistency of dense paths and targets
        bad_orient = bad_aug = 0
        for i in ids[:16]:
            for ds in (train_eval, aug):
                _, t, _ = ds[i]
                for k, (p, e) in enumerate(zip(dense_paths(t), t['edges'])):
                    ok = np.allclose(p[0], t['nodes'][e[0]] * SIZE, atol=1e-6) and np.allclose(p[-1], t['nodes'][e[1]] * SIZE, atol=1e-6)
                    if ds is train_eval:
                        bad_orient += not ok
                    else:
                        bad_aug += not ok
        # 2. target rendering + oracle decode
        images, tg, meta = collate([train_eval[i] for i in ids])
        heat, offset, peak, line = targets(tg, 'cpu')
        n_gt = sum(len(t['nodes']) for t in tg)
        unique_peaks = int(peak.sum())
        oracle = torch.zeros(len(ids), FEAT, SIZE, SIZE)
        oracle[:, :1] = torch.logit(heat.clamp(1e-6, 1 - 1e-6))
        oracle[:, 1:3] = offset
        rows = []
        for b, (p, score) in enumerate(peaks(oracle, 0.5)):
            t = tg[b]
            metric = torch.as_tensor(t['metric'], dtype=torch.float32) / SIZE
            gt_px, gt_edges, _, _ = patch(t, 'cpu')
            pairs = candidate_pairs(p @ metric.T)
            mapping = match(p @ metric.T, gt_px @ metric.T)
            labels = pair_labels(pairs, mapping, gt_edges).numpy() if len(pairs) else np.zeros(0)
            # oracle curves: GT K8 controls of the matched edge, oriented a->b
            ctl = np.zeros((len(pairs), 8, 2))
            index = {tuple(sorted(map(int, e))): k for k, e in enumerate(gt_edges.tolist())}
            for j, (u, v) in enumerate(pairs.tolist()):
                if labels[j]:
                    k = index[tuple(sorted((mapping[u], mapping[v])))]
                    c = t['controls'][k]
                    ctl[j] = c if mapping[u] == int(t['edges'][k][0]) else c[::-1]
                else:
                    ctl[j] = np.linspace(p[u].numpy() / SIZE, p[v].numpy() / SIZE, 8)
            rows.append(dict(sample_id=meta[b]['sample_id'], source_image=meta[b]['subject_id'], target=t, built_at=0.5,
                             nodes=(p / SIZE).numpy(), node_scores=np.ones(len(p)), pairs=pairs.numpy(), edge_scores=labels, controls=ctl))
        oracle_eval = evaluate(rows, 0.5, 0.5, 'spline')
        s = oracle_eval['summary']
        report[dataset] = dict(
            crops=len(ids), units=s['units'], gt_nodes=n_gt, distinct_peak_pixels=unique_peaks,
            heat_max=float(heat.max()), offset_abs_max=float(offset.abs().max()), centerline_fraction=float(line.mean()),
            dense_path_orientation_errors=bad_orient, augmented_path_orientation_errors=bad_aug,
            oracle={t: dict(node_f1=s[t]['node']['f1'], node_recall=s[t]['node']['recall'], edge_f1=s[t]['edge']['f1'],
                            edge_recall=s[t]['edge']['recall']) for t in ('1.0', '2.0', '4.0')},
            oracle_gt_edge_candidate_coverage_2=s['gt_edge_candidate_coverage_2'],
            oracle_K8_macro=s['spline'], oracle_chord_macro=s['chord'], oracle_K8_pooled=s['spline_pooled'], oracle_chord_pooled=s['chord_pooled'])
        print('CHECK ' + json.dumps({dataset: report[dataset]}, default=float), flush=True)
    # 3. tiny CPU forward/backward with both candidate sources + real predict/evaluate path
    torch.manual_seed(SEED)
    model = DenseGraph2D()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = make_dataset('avwide', 'train', augment=True)
    ids = [i for i, r in enumerate(ds.rows) if 4 <= r['nodes'] <= 20][:2]
    images, tg, meta = collate([ds[i] for i in ids])
    steps = []
    for step in range(3):
        losses, stats = train_losses(model, images, tg, 'cpu', use_pred=True, train_peak_threshold=0.0 if step == 0 else 0.2)
        loss = sum(losses.values())
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        opt.step()
        steps.append(dict({k: float(v) for k, v in losses.items()}, grad_norm=norm, **stats))
    print('CHECK tiny-train ' + json.dumps(steps), flush=True)
    with torch.no_grad():
        out = model.forward_image(images)
        out[:, :1] = 3.0 * torch.rand_like(out[:, :1]) - 1.0  # force many peaks through decode / metrics
        rows = predict_out(model, out, tg, meta, 0.5)
    ev = evaluate(rows, 0.5, 0.5, 'spline')
    report['tiny_cpu_train'] = steps
    report['tiny_predict_evaluate'] = dict(peaks=[len(r['nodes']) for r in rows], pairs=[len(r['pairs']) for r in rows], headline=headline(ev))
    print('CHECK tiny-eval ' + json.dumps(report['tiny_predict_evaluate']), flush=True)
    return report


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['avwide', 'reaver', 'octasynth'])
    p.add_argument('--mode', choices=['spline', 'polyline', 'chord'], default='spline')
    p.add_argument('--interior', type=int, default=6)
    p.add_argument('--updates', type=int, default=15000)
    p.add_argument('--warmup-gt', type=int, default=2000, help='Updates with GT-node candidates only')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--train-peak-threshold', type=float, default=0.2)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--eval-batch', type=int, default=16)
    p.add_argument('--output', type=Path)
    p.add_argument('--report-dir', type=Path, default=None, help='Also copy final json here (small, in the repo)')
    p.add_argument('--run-name', default=None)
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--cpu-checks', type=Path, default=None, help='Run CPU checks and write the report to this json')
    p.add_argument('--check-crops', type=int, default=64)
    p.add_argument('--metric-workers', type=int, default=1)
    a = p.parse_args()
    global METRIC_WORKERS
    METRIC_WORKERS = a.metric_workers
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if a.cpu_checks:
        report = cpu_checks(a)
        a.cpu_checks.parent.mkdir(parents=True, exist_ok=True)
        a.cpu_checks.write_text(json.dumps(report, indent=1, default=float))
        return
    assert a.dataset and a.output
    out_dir = a.output / a.dataset / (a.run_name or (a.mode + (str(a.interior) if a.mode != 'chord' else '')))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_eval = make_dataset(a.dataset, 'train', augment=False)
    val_ds = make_dataset(a.dataset, 'validation', augment=False)
    monitor_ids, val_ids = subset(train_eval), list(range(len(val_ds)))
    model = DenseGraph2D(a.mode, a.interior).to(device)
    if a.evaluate_only:
        model.load_state_dict(torch.load(out_dir / 'model.pt', map_location=device)['model'])
        final_evaluation(model, a, monitor_ids, val_ids, train_eval, val_ds, device, out_dir, name='final-reeval.json')
        return
    assert not (out_dir / 'model.pt').exists(), 'fresh run only; use --run-name or --evaluate-only'
    (out_dir / 'protocol.json').write_text(json.dumps(dict(
        args={k: str(v) for k, v in vars(a).items()}, seed=SEED, data_root=str(DATA), train_crops=len(train_eval),
        validation_crops=len(val_ids), monitor_ids=monitor_ids, monitor_sample_ids=[train_eval.rows[i]['sample_id'] for i in monitor_ids],
        sample_exposures=a.updates * a.batch, train_passes=a.updates * a.batch / len(train_eval), tolerance_native=TOLERANCE,
        jitter_native=JITTER, corridor=dict(along=L, across=S, half_width_px=12.0), neighbours=NEIGHBOURS, max_peaks=MAX_PEAKS,
        max_negatives=MAX_NEGATIVES, sigma_px=SIGMA, test_policy='no test loading or inference'), indent=1))
    train_ds = make_dataset(a.dataset, 'train', augment=True, preload=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.updates, pct_start=0.05)
    sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=a.updates * a.batch)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                         collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
    log = (out_dir / 'training.jsonl').open('a')
    start, step = time.time(), 0
    window = []
    for images, batch_targets, _ in loader:
        step += 1
        use_pred = step > a.warmup_gt
        losses, stats = train_losses(model, images, batch_targets, device, use_pred, a.train_peak_threshold)
        loss = sum(losses.values())
        assert torch.isfinite(loss), 'nonfinite loss'
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        window.append(dict({k: float(v) for k, v in losses.items()}, **stats))
        if step % 100 == 0 or step == a.updates or (a.updates <= 400 and step % 20 == 0):
            row = dict(step=step, seconds=time.time() - start, lr=sched.get_last_lr()[0], grad_norm=float(norm), predicted_candidates=use_pred,
                       **{k: float(np.mean([w[k] for w in window])) for k in window[0]})
            window = []
            log.write(json.dumps(row) + '\n')
            log.flush()
            print('STEP ' + json.dumps(row), flush=True)
    assert step == a.updates
    train_seconds = time.time() - start
    torch.save(dict(model=model.state_dict(), args={k: str(v) for k, v in vars(a).items()}, step=step, train_seconds=train_seconds), out_dir / 'model.pt')
    final_evaluation(model, a, monitor_ids, val_ids, train_eval, val_ds, device, out_dir)


if __name__ == '__main__':
    main()
