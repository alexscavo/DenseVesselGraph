#!/usr/bin/env python3
"""Two-stage dense model with centreline-traced connectivity and adaptive geometry (30 Oct 2026).

Same stage 1 as dense_graph_20261030 (U-Net heatmap/offset/centreline/features). Stage 2 changes:
- Candidates: geodesic Voronoi partition of the predicted centreline mask seeded at the
  nodes (GPU Bellman-Ford over 26-neighbours, voxel cost 1/(p+0.05)). Two nodes are a
  candidate iff their regions touch inside the mask, at any distance; regions block skip
  edges. The 16-NN pairs of dense_graph are kept as a fallback (straight chord path).
- Path: read back by steepest descent of the distance field from the contact voxels to both
  seeds, smoothed and resampled by arc length; exact sub-voxel nodes as endpoints.
- Corridor and geometry follow the path (24x11x11 tube in moving frames). Geometry is a
  clamped cubic B-spline whose interior controls are residuals around the path; K is chosen
  per edge by a fixed rule on the traced path (K8 / K16 / K24; chord fallback pairs K8).
Labels, losses, threshold selection and evaluation are unchanged from dense_graph_20261030.
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
sys.path.insert(0, str(ROOT / 'scripts'))
from models.spline import cubic_basis  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, collate, make_dataset, targets, focal  # noqa: E402
from oracle_geometry_20261030 import chamfer, to_mm  # noqa: E402
import dense_graph_20261030 as D  # noqa: E402

SIZE, L, S, SAMPLES = 64, 24, 11, 64
TAU = 0.5            # centreline mask threshold
SEED_RADIUS = 2      # voxels of mask forced around each node so it connects to the centreline
MAX_ITERS = 320      # geodesic propagation cap (longest BRAVA arc ~ 190 voxels)
INTERIOR = (6, 14, 22)  # K8 / K16 / K24
# Opt-in variants (2 Nov 2026); defaults reproduce the original traced model exactly.
#   soft:         no hard centreline mask; ink may cross any voxel at cost 1/(p+0.05)
#   knn_fallback: add 16-NN pairs (straight paths) to the traced candidates
#   gap_features: edge/geometry heads also read centreline-probability statistics along the path
# Opt-in (3 Nov 2026):
#   two_hop:      for a traced chain A-C-B whose ends do not touch, also propose A-B along the joined
#                 path A->C->B; the heads read a pass-through descriptor of C (flag, heat at C, bend at C)
#   passthrough:  node-level decision for two-hop chains (implies two_hop): a head scores whether C is a
#                 pass-through point; if so C, A-C and C-B are replaced by A-B, otherwise A-B is dropped.
#                 Two-hop pairs are never accepted as ordinary edges, so A-B/A-C/C-B cannot form a triangle.
# Opt-in (9 Nov 2026):
#   k_min_arc:    (K16, K24) minimum traced arc in mm before the arc/chord ratio may raise K. On short traces the ratio
#                 is dominated by voxel staircase / endpoint kinks (IXI chords < 5 mm: traced ratio 1.22 vs GT 1.00), so
#                 short straight edges got K16/K24. (0, 0) = the original rule; training default since 9 Nov is (8, 20).
#   arc_penalty:  training only. Weight of relu(arc_pred - 1.02 arc_gt - 0.1 mm) added to each positive pair's chamfer
#                 (mm). The chamfer is blind to sub-mm curls: short straight edges were drawn with arc/chord 3-5 and
#                 hundreds of degrees of turning (experiments/k_rule_check_20261109). TRAINING DEFAULT 0.1 since 9 Nov
#                 (--arc-penalty; arm traced_arcpen_r1109 removed the curls with no metric loss, arm_compare_20261109).
#                 The MODE default stays 0 so that runs saved before then are described as they were trained.
#   k_rule:       'ratio' = arc / arc-chord thresholds above (with k_min_arc); 'fit' = the fewest controls that represent
#                 the traced path: smooth it at k_fit_smooth_mm, least-squares fit K8 (then K16) with pinned endpoints, take
#                 the first K whose Hausdorff deviation from the smoothed path is <= k_fit_tol voxels, else K24.
#                 Uses only the traced path (no GT). 16-NN fallback pairs stay K8 under both rules.
MODE = dict(soft=False, knn_fallback=True, gap_features=False, two_hop=False, passthrough=False, k_min_arc=(0.0, 0.0),
            arc_penalty=0.0, k_rule='ratio', k_fit_tol=0.5, k_fit_smooth_mm=1.0, compress_tol=0.0)
# k_rule 'fixed24' = every pair (traced and fallback) uses the K24 head: one head learns from every edge, no rule.
# compress_tol (evaluation only, voxels): replace each predicted curve by the fewest-control (K8 / K16) endpoint-pinned
# least-squares spline that reproduces it within this Hausdorff tolerance; curves needing more keep their K24 shape.
# 0 = off. The chosen count is stored per pair as 'interior_compressed' (6 / 14 / 22).
# k_rule 'fit_length' = 'fit', with the original length floors (arc >= 15 mm -> at least K16, >= 35 mm -> K24): on long
# edges the traced path is smoother than the GT, so the fit alone under-selects (BRAVA >= 20 mm), and extra controls on
# long edges are cheap (wide spacing). The arc/chord ratio is not used.
K_MIN_ARC_DEFAULT = (8.0, 20.0)
MODE_DEFAULTS = dict(MODE)


def apply_saved_mode(final):
    """Set MODE exactly as recorded in a run's final.json (keys it lacks take the original defaults, so a run that
    predates a flag is evaluated as it was). Use instead of MODE.update when several runs share one process."""
    MODE.clear()
    MODE.update(MODE_DEFAULTS, **final.get('mode', {}))
    MODE['k_min_arc'] = tuple(MODE['k_min_arc'])
PROB_HOOK = None  # analysis only: f(prob, nodes, geometry) -> centreline probability used by predict()
GAP_FEATURES = 5
VIA_FEATURES = 3
MAX_VIA_DEGREE = 6  # skip pass-through pairs around nodes touching more regions than this
FEATURE_CHANNELS = 32  # default number of U-Net feature channels (outputs 5..36); --feature-channels since 9 Nov 2026
PASS_THRESHOLDS = [round(t, 2) for t in np.arange(0.3, 0.96, 0.05)]
OFFSETS = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
HALF = [o for o in OFFSETS if o > (0, 0, 0)]
BALL = [(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3) if i * i + j * j + k * k <= SEED_RADIUS ** 2]


def neighbours(x, fill, offsets=OFFSETS):
    """x [B,64,64,64] -> [B,len(offsets),64,64,64] with out[:,k,v] = x[v + offsets[k]]."""
    p = F.pad(x, (1, 1, 1, 1, 1, 1), value=fill)
    return torch.stack([p[:, 1 + a:1 + a + SIZE, 1 + b:1 + b + SIZE, 1 + c:1 + c + SIZE] for a, b, c in offsets], 1)


@torch.no_grad()
def geodesic_regions(prob, seeds):
    """Geodesic Voronoi labels/distances inside the centreline mask for each patch's seeds."""
    B, dev = prob.shape[0], prob.device
    mask = torch.ones_like(prob, dtype=torch.bool) if MODE['soft'] else prob >= TAU
    dist = torch.full_like(prob, float('inf'))
    label = torch.full_like(prob, -1.0)
    ball = torch.tensor(BALL, device=dev)
    for b, s in enumerate(seeds):
        if len(s) == 0:
            continue
        v = s.round().clamp(0, SIZE - 1).long()
        around = (v[:, None] + ball[None]).clamp(0, SIZE - 1).reshape(-1, 3)
        mask[b, around[:, 0], around[:, 1], around[:, 2]] = True
        dist[b, v[:, 0], v[:, 1], v[:, 2]] = 0.0
        label[b, v[:, 0], v[:, 1], v[:, 2]] = torch.arange(len(v), device=dev, dtype=prob.dtype)
    weight = torch.where(mask, 1.0 / (prob + 0.05), torch.full_like(prob, float('inf')))
    step = torch.tensor([float(np.linalg.norm(o)) for o in OFFSETS], device=dev).view(1, -1, 1, 1, 1)
    for it in range(MAX_ITERS):
        cand = neighbours(dist, float('inf')) + step * weight[:, None]
        best, arg = cand.min(1)
        update = (best < dist - 1e-6) & mask
        if it % 8 == 0 and not update.any():
            break
        new_label = neighbours(label, -1.0).gather(1, arg[:, None]).squeeze(1)
        dist = torch.where(update, best, dist)
        label = torch.where(update, new_label, label)
    return mask, dist, label, weight


@torch.no_grad()
def traced_pairs(prob, seeds):
    """Per patch: dict (i,j)->contact (voxel_i, voxel_j) for touching geodesic regions."""
    mask, dist, label, weight = geodesic_regions(prob, seeds)
    B, dev = prob.shape[0], prob.device
    out = [dict() for _ in range(B)]
    nl, nd, nm = neighbours(label, -1.0, HALF), neighbours(dist, float('inf'), HALF), neighbours(mask.float(), 0.0, HALF)
    steps = torch.tensor([float(np.linalg.norm(o)) for o in HALF], device=dev)
    half = torch.tensor(HALF, device=dev)
    keys, costs, ps, qs = [], [], [], []
    for k in range(len(HALF)):
        ok = mask & (nm[:, k] > 0) & (label >= 0) & (nl[:, k] >= 0) & (nl[:, k] != label) & torch.isfinite(dist) & torch.isfinite(nd[:, k])
        idx = ok.nonzero()
        if len(idx) == 0:
            continue
        b, x, y, z = idx.unbind(1)
        la, lb = label[b, x, y, z].long(), nl[:, k][b, x, y, z].long()
        p, q = idx[:, 1:], idx[:, 1:] + half[k]
        swap = (la > lb)[:, None]
        lo, hi = torch.minimum(la, lb), torch.maximum(la, lb)
        keys.append((b * 1024 + lo) * 1024 + hi)
        costs.append(dist[b, x, y, z] + nd[:, k][b, x, y, z] + steps[k] * weight[b, x, y, z])
        ps.append(torch.where(swap, q, p))  # ps: voxel owned by the lower node index
        qs.append(torch.where(swap, p, q))
    if not keys:
        return out, dist, label
    keys, costs, ps, qs = torch.cat(keys), torch.cat(costs), torch.cat(ps), torch.cat(qs)
    order = torch.argsort(costs)
    order = order[torch.argsort(keys[order], stable=True)]  # by key, then by cost
    keys, ps, qs, costs = keys[order], ps[order], qs[order], costs[order]
    first = torch.ones_like(keys, dtype=torch.bool)
    first[1:] = keys[1:] != keys[:-1]
    for key, p, q, c in zip(keys[first].tolist(), ps[first].tolist(), qs[first].tolist(), costs[first].tolist()):
        bb, rest = divmod(key, 1024 * 1024)
        out[bb][divmod(rest, 1024)] = (c, (tuple(p), tuple(q)))
    return out, dist, label


@torch.no_grad()
def descend(dist_b, label_b, start, target_label, steps=256):
    """Steepest descent of the distance field inside one region, from start voxels (P,3)."""
    dev = dist_b.device
    pos = start.clone()
    offs = torch.tensor(OFFSETS, device=dev)
    trace = [pos.clone()]
    for _ in range(steps):
        nb = (pos[:, None] + offs[None]).clamp(0, SIZE - 1)
        d = dist_b[nb[..., 0], nb[..., 1], nb[..., 2]]
        lab = label_b[nb[..., 0], nb[..., 1], nb[..., 2]]
        d = torch.where(lab == target_label[:, None], d, torch.full_like(d, float('inf')))
        best, arg = d.min(1)
        here = dist_b[pos[:, 0], pos[:, 1], pos[:, 2]]
        move = best < here
        if not move.any():
            break
        pos = torch.where(move[:, None], nb[torch.arange(len(pos), device=dev), arg], pos)
        trace.append(pos.clone())
    return torch.stack(trace, 1).float()  # (P,T,3)


def resample(path, n):
    """Arc-length resampling of polylines (P,T,3) -> (P,n,3)."""
    seg = (path[:, 1:] - path[:, :-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)
    total = cum[:, -1:].clamp_min(1e-6)
    t = torch.linspace(0, 1, n, device=path.device)[None] * total
    i = torch.searchsorted(cum, t.contiguous()).clamp(1, path.shape[1] - 1)
    c0, c1 = cum.gather(1, i - 1), cum.gather(1, i)
    w = ((t - c0) / (c1 - c0).clamp_min(1e-6)).clamp(0, 1)[..., None]
    p0 = path.gather(1, (i - 1)[..., None].expand(-1, -1, 3))
    p1 = path.gather(1, i[..., None].expand(-1, -1, 3))
    return p0 * (1 - w) + p1 * w


def smooth(path, k=5):
    pad = F.pad(path.transpose(1, 2), (k // 2, k // 2), mode='replicate')
    return F.avg_pool1d(pad, k, 1).transpose(1, 2)


def two_hop_pairs(contacts):
    """(A,B) -> C for every A-C, C-B contact pair whose ends do not touch (lowest joined cost wins)."""
    nbr = {}
    for (i, j), (cost, _) in contacts.items():
        nbr.setdefault(i, []).append((j, cost))
        nbr.setdefault(j, []).append((i, cost))
    best = {}
    for c, around in nbr.items():
        if len(around) > MAX_VIA_DEGREE:
            continue
        for x, (u, cu) in enumerate(around):
            for v, cv in around[x + 1:]:
                key = (min(u, v), max(u, v))
                if key in contacts:
                    continue
                if key not in best or cu + cv < best[key][1]:
                    best[key] = (c, cu + cv)
    return {k: c for k, (c, _) in best.items()}


def via_points(cand, pairs, hops):
    """(P,3) position of the pass-through node C, NaN for ordinary pairs."""
    via = torch.full((len(pairs), 3), float('nan'), device=cand.device)
    rows = [k for k, pr in enumerate(pairs.tolist()) if tuple(pr) in hops]
    if rows:
        via[rows] = cand[[hops[tuple(pairs[k].tolist())] for k in rows]]
    return via


def build_paths(cand, pairs, contacts, dist_b, label_b, hops=None):
    """Paths (P,SAMPLES,3) in voxels and a traced flag; chord paths for untraced pairs.
    Two-hop pairs (in hops) follow the traced A->C and C->B paths joined at C (flagged traced)."""
    P = len(pairs)
    a, b = cand[pairs[:, 0]], cand[pairs[:, 1]]
    s = torch.linspace(0, 1, SAMPLES, device=cand.device)[None, :, None]
    paths = a[:, None] * (1 - s) + b[:, None] * s
    traced = torch.zeros(P, dtype=torch.bool, device=cand.device)
    rows = [k for k, (i, j) in enumerate(pairs.tolist()) if (i, j) in contacts]
    if rows:
        ends = [contacts[tuple(pairs[k].tolist())][1] for k in rows]
        va = torch.tensor([e[0] for e in ends], device=cand.device)
        vb = torch.tensor([e[1] for e in ends], device=cand.device)
        ta = descend(dist_b, label_b, va, pairs[rows, 0].float())
        tb = descend(dist_b, label_b, vb, pairs[rows, 1].float())
        full = torch.cat([ta.flip(1), tb], 1)
        full[:, 0], full[:, -1] = a[rows], b[rows]  # exact sub-voxel endpoints
        p = smooth(resample(full, SAMPLES))
        p[:, 0], p[:, -1] = a[rows], b[rows]
        paths[rows] = p
        traced[rows] = True
    rows = [k for k, pr in enumerate(pairs.tolist()) if hops and tuple(pr) in hops and tuple(pr) not in contacts]
    if rows:
        ab = pairs[rows]
        c = torch.tensor([hops[tuple(pr)] for pr in ab.tolist()], device=cand.device)
        first = torch.stack([torch.minimum(ab[:, 0], c), torch.maximum(ab[:, 0], c)], 1)
        second = torch.stack([torch.minimum(c, ab[:, 1]), torch.maximum(c, ab[:, 1])], 1)
        p1, _ = build_paths(cand, first, contacts, dist_b, label_b)
        p2, _ = build_paths(cand, second, contacts, dist_b, label_b)
        p1 = torch.where((first[:, 0] == ab[:, 0])[:, None, None], p1, p1.flip(1))  # A -> C
        p2 = torch.where((second[:, 0] == c)[:, None, None], p2, p2.flip(1))  # C -> B
        p = smooth(resample(torch.cat([p1, p2[:, 1:]], 1), SAMPLES))
        p[:, 0], p[:, -1] = cand[ab[:, 0]], cand[ab[:, 1]]
        paths[rows] = p
        traced[rows] = True
    return paths, traced


def frames(path):
    """Moving frames (t,u,w) along paths (P,n,3) with a per-path helper axis (no flips)."""
    t = torch.cat([path[:, 1:2] - path[:, :1], path[:, 2:] - path[:, :-2], path[:, -1:] - path[:, -2:-1]], 1)
    t = F.normalize(t, dim=-1)
    chord = F.normalize(path[:, -1] - path[:, 0] + 1e-6, dim=-1)
    helper = torch.eye(3, device=path.device)[chord.abs().argmin(-1)][:, None].expand_as(t)
    u = torch.cross(t, helper, dim=-1)
    u = F.normalize(torch.where(u.norm(dim=-1, keepdim=True) < 1e-3, torch.cross(t, helper.roll(1, -1), dim=-1), u), dim=-1)
    return t, u, torch.cross(t, u, dim=-1)


def smooth_physical(paths_mm, scale_mm):
    """Moving average over ~scale_mm of arc (per-path window on the 64 arc-length samples), endpoints pinned."""
    step = (paths_mm[:, 1:] - paths_mm[:, :-1]).norm(dim=-1).mean(-1).clamp_min(1e-6)
    half = ((scale_mm / step).round().long().clamp(1, 31) // 2)  # window 2*half+1 (odd), same as smooth(path, k)
    n, pad = paths_mm.shape[1], 15
    padded = torch.cat([paths_mm[:, :1].expand(-1, pad, -1), paths_mm, paths_mm[:, -1:].expand(-1, pad, -1)], 1)  # replicate
    csum = F.pad(padded.cumsum(1), (0, 0, 1, 0))
    i = torch.arange(n, device=paths_mm.device)[None] + pad
    hi = (i + half[:, None] + 1)[..., None].expand(-1, -1, 3)
    lo = (i - half[:, None])[..., None].expand(-1, -1, 3)
    out = (csum.gather(1, hi) - csum.gather(1, lo)) / (2 * half + 1)[:, None, None].float()
    out[:, 0], out[:, -1] = paths_mm[:, 0], paths_mm[:, -1]
    return out


def point_polyline_distance(x, y):
    """(N,n,3) points to (N,m,3) polylines -> (N,n) distance to the nearest segment."""
    a, b = y[:, :-1], y[:, 1:]
    ab = b - a
    t = (((x[:, :, None] - a[:, None]) * ab[:, None]).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)[:, None]).clamp(0, 1)
    return (x[:, :, None] - (a[:, None] + t[..., None] * ab[:, None])).norm(dim=-1).min(-1).values


_FIT = {}


def fit_errors(paths_mm, controls):
    """Hausdorff deviation (mm) of the endpoint-pinned least-squares cubic B-spline with `controls` controls from each
    path (N,64,3), parameterised by sample index (the paths are arc-length samples), as data/spline_targets.fit_spline."""
    key = (controls, paths_mm.shape[1], str(paths_mm.device))
    if key not in _FIT:
        B = cubic_basis(controls, paths_mm.shape[1])[0].to(paths_mm.device)
        _FIT[key] = (B.float(), torch.linalg.pinv(B[:, 1:-1]).float())
    fitted = fit_spline_batch(paths_mm, controls)
    return torch.maximum(point_polyline_distance(fitted, paths_mm).max(-1).values,
                         point_polyline_distance(paths_mm, fitted).max(-1).values)


def fit_spline_batch(paths, controls):
    """Endpoint-pinned least-squares cubic B-spline with `controls` controls, sampled at the paths' sample count."""
    key = (controls, paths.shape[1], str(paths.device))
    if key not in _FIT:
        B = cubic_basis(controls, paths.shape[1])[0].to(paths.device)
        _FIT[key] = (B.float(), torch.linalg.pinv(B[:, 1:-1]).float())
    B, pinv = _FIT[key]
    ends = B[None, :, :1] * paths[:, :1] + B[None, :, -1:] * paths[:, -1:]
    inner = pinv[None] @ (paths - ends)
    return B[None] @ torch.cat([paths[:, :1], inner, paths[:, -1:]], 1)


def compress_curves(curves, tol, interior=None):
    """(P,64,3) predicted curves (voxels) -> (curves, interior counts): the fewest-control fit (K8, then K16) of each
    arc-length-resampled curve within Hausdorff tol voxels; otherwise the curve is kept as predicted. Never more
    controls than the curve already has (`interior`, 6 / 14 / 22; default 22): a K8 curve stays as it is."""
    q = resample(curves, curves.shape[1])
    m = interior.clone() if interior is not None else torch.full((len(curves),), 22, dtype=torch.long, device=curves.device)
    out, open_ = curves.clone(), m > 6
    for controls in (8, 16):
        open_ &= m > controls - 2
        if not open_.any():
            break
        idx = open_.nonzero().squeeze(1)
        fitted = fit_spline_batch(q[idx], controls)
        err = torch.maximum(point_polyline_distance(fitted, q[idx]).max(-1).values, point_polyline_distance(q[idx], fitted).max(-1).values)
        ok = err <= tol
        out[idx[ok]], m[idx[ok]] = fitted[ok], controls - 2
        open_[idx[ok]] = False
    return out, m


def fit_interior(paths_mm, tol_mm, smooth_mm):
    """Interior-control count (6/14/22) = fewest controls whose fit of the smoothed path is within tol_mm (N,)."""
    p = smooth_physical(paths_mm, smooth_mm)
    m = torch.full((len(p),), 22, dtype=torch.long, device=p.device)
    open_ = torch.ones(len(p), dtype=torch.bool, device=p.device)
    for controls in (8, 16):
        if not open_.any():
            break
        ok = torch.zeros_like(open_)
        ok[open_] = fit_errors(p[open_], controls) <= tol_mm[open_]
        m[ok] = controls - 2
        open_ &= ~ok
    return m


def choose_interior(paths_mm, traced, voxel_mm=None):
    """Interior controls per pair from the traced path. 'ratio': arc length, and arc/chord once the arc reaches
    MODE['k_min_arc']. 'fit': fewest controls that fit the smoothed traced path within MODE['k_fit_tol'] voxels
    (voxel_mm = mean voxel size of the patch, required)."""
    if MODE['k_rule'] == 'fixed24':
        return torch.full((len(paths_mm),), 22, dtype=torch.long, device=paths_mm.device)
    if MODE['k_rule'] in ('fit', 'fit_length'):
        assert voxel_mm is not None, 'the fit rule needs the voxel size'
        m = torch.full((len(paths_mm),), 6, dtype=torch.long, device=paths_mm.device)
        if traced.any():
            tol = torch.full((int(traced.sum()),), MODE['k_fit_tol'] * float(voxel_mm), device=paths_mm.device)
            m[traced] = fit_interior(paths_mm[traced], tol, MODE['k_fit_smooth_mm'])
        if MODE['k_rule'] == 'fit_length':
            arc = (paths_mm[:, 1:] - paths_mm[:, :-1]).norm(dim=-1).sum(-1)
            m[traced & (arc >= 15)] = m[traced & (arc >= 15)].clamp_min(14)
            m[traced & (arc >= 35)] = 22
        return m
    arc = (paths_mm[:, 1:] - paths_mm[:, :-1]).norm(dim=-1).sum(-1)
    ratio = arc / (paths_mm[:, -1] - paths_mm[:, 0]).norm(dim=-1).clamp_min(1e-3)
    min16, min24 = MODE['k_min_arc']
    m = torch.full_like(arc, 6, dtype=torch.long)
    m[traced & ((arc >= 15) | ((ratio >= 1.15) & (arc >= min16)))] = 14
    m[traced & ((arc >= 35) | ((ratio >= 1.5) & (arc >= min24)))] = 22
    return m


class TracedDenseGraph(D.DenseGraph):
    def __init__(self, radius=12.0, gap_features=False, two_hop=False, passthrough=False, backbone='unet', feature_channels=FEATURE_CHANNELS):
        super().__init__('spline', 6, radius, backbone, feature_channels)
        self.gap_features, self.two_hop = gap_features, two_hop or passthrough
        width = (self.edge_head[0].in_features + 1 + (GAP_FEATURES if gap_features else 0)
                 + (VIA_FEATURES if self.two_hop else 0))  # + arc length (+ gap statistics) (+ pass-through node)
        self.edge_head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 1))
        if passthrough:  # joined A->C->B path features + U-Net features at C
            self.pass_head = nn.Sequential(nn.Linear(width + self.fc, 256), nn.GELU(), nn.Linear(256, 1))
        del self.geometry_head  # replaced by the per-K heads below
        self.heads = nn.ModuleDict()
        for m in INTERIOR:
            head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 3 * m))
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.heads[str(m)] = head
            basis, greville = cubic_basis(m + 2, SAMPLES)
            self.register_buffer(f'basis{m}', basis)
            self.register_buffer(f'greville{m}', greville[1:-1])

    def path_features(self, volume, heat, paths, prob=None, via=None):
        a, b = paths[:, 0], paths[:, -1]
        p24 = resample(paths, L)
        t, u, w = frames(p24)
        o = torch.linspace(-self.radius, self.radius, S, device=paths.device)
        grid_pts = (p24[:, :, None, None] + o[None, None, :, None, None] * u[:, :, None, None] + o[None, None, None, :, None] * w[:, :, None, None])
        grid = (grid_pts / (SIZE - 1) * 2 - 1).flip(-1)
        corridor = F.grid_sample(volume[None], grid[None].reshape(1, -1, S, S, 3), align_corners=True)
        corridor = corridor.view(volume.shape[0], len(paths), L, S, S).transpose(0, 1)
        g = (torch.stack([a, b], 1) / (SIZE - 1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3)
        point = F.grid_sample(torch.cat([volume, heat], 0)[None], g, align_corners=True).view(volume.shape[0] + 1, len(paths), 2)
        d = b - a
        arc = (paths[:, 1:] - paths[:, :-1]).norm(dim=-1).sum(-1, keepdim=True)
        parts = [self.corridor(corridor).flatten(1), d.norm(dim=-1, keepdim=True) / SIZE, d / SIZE,
                 point[:-1].permute(1, 2, 0).reshape(len(paths), -1), point[-1], arc / SIZE]
        if self.gap_features:
            parts.append(gap_statistics(prob, paths))
        if self.two_hop:
            parts.append(via_features(heat, a, b, via))
        return torch.cat(parts, -1)

    def pass_logits(self, volume, x, via):
        """Pass-through logit for two-hop rows: x = joined-path features, via = (n,3) voxel position of C."""
        g = (via / (SIZE - 1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3)
        at_c = F.grid_sample(volume[None], g, align_corners=True).view(volume.shape[0], -1).T
        return self.pass_head(torch.cat([x, at_c], -1)).squeeze(-1)

    def path_curves(self, x, paths, interior, traced):
        """Adaptive-K B-spline around the path; residuals in path frames (12 vox chord / 4 vox traced)."""
        curves = torch.empty(len(paths), SAMPLES, 3, device=paths.device)
        for m in INTERIOR:
            sel = interior == m
            if not sel.any():
                continue
            g = getattr(self, f'greville{m}')
            base = paths[sel]
            idx = (g * (SAMPLES - 1)).round().long()
            t, u, w = frames(base)
            scale = torch.where(traced[sel], 4.0, self.radius)[:, None, None]
            res = self.heads[str(m)](x[sel]).view(-1, m, 3) * scale
            interior_pts = base[:, idx] + res[..., :1] * t[:, idx] + res[..., 1:2] * u[:, idx] + res[..., 2:] * w[:, idx]
            controls = torch.cat([base[:, :1], interior_pts, base[:, -1:]], 1)
            curves[sel] = getattr(self, f'basis{m}') @ controls
        return curves


def voxel_size(metric):
    """Mean voxel edge (mm) of a patch metric (3x3, voxel -> mm, columns = voxel axes)."""
    return float(metric.norm(dim=0).mean())


def via_features(heat, a, b, via):
    """Pass-through node descriptor: [is two-hop, heat at C, cos of the bend A->C->B]; zeros otherwise."""
    out = torch.zeros(len(a), VIA_FEATURES, device=a.device)
    if via is None:
        return out
    sel = ~torch.isnan(via[:, 0])
    if sel.any():
        c = via[sel]
        g = (c / (SIZE - 1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3)
        h = F.grid_sample(heat[None], g, align_corners=True).view(-1)
        bend = F.cosine_similarity(c - a[sel], b[sel] - c, dim=-1)
        out[sel] = torch.stack([torch.ones_like(h), h, bend], 1)
    return out


@torch.no_grad()
def gap_statistics(prob, paths):
    """Centreline probability along each path (64 samples): mean, min, fraction < 0.5,
    longest run < 0.5 (fraction of the path), log of the mean traversal cost 1/(p+0.05)."""
    g = (paths / (SIZE - 1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3)
    p = F.grid_sample(prob[None, None], g, align_corners=True).view(len(paths), -1)
    low = p < TAU
    run = torch.zeros(len(paths), device=p.device)
    best = torch.zeros_like(run)
    for j in range(p.shape[1]):
        run = (run + 1) * low[:, j]
        best = torch.maximum(best, run)
    return torch.stack([p.mean(1), p.min(1).values, low.float().mean(1), best / p.shape[1],
                        torch.log(1.0 / (p + 0.05)).mean(1)], 1)


def patch_candidates(cand, metric, contacts, hops=None):
    extra = set(hops or ())
    if not MODE['knn_fallback']:
        if not contacts and not extra:
            return torch.zeros(0, 2, dtype=torch.long, device=cand.device)
        return torch.tensor(sorted(set(contacts) | extra), dtype=torch.long, device=cand.device)
    knn = D.candidate_pairs(cand @ metric.T)
    keys = {tuple(p) for p in knn.tolist()} | set(contacts) | extra
    if not keys:
        return knn
    return torch.tensor(sorted(keys), dtype=torch.long, device=cand.device)


def patch_terms(model, feats, heat, cand, gt_vox, gt_edges, guides, metric, mapping, contacts, dist_b, label_b, prob_b=None):
    if len(cand) < 2:
        return None
    hops = two_hop_pairs(contacts) if MODE['two_hop'] else None
    pairs = patch_candidates(cand, metric, contacts, hops)
    if len(pairs) == 0:
        return None
    labels = D.pair_labels(pairs, mapping, gt_edges)
    hop = hop_rows(pairs, contacts, hops) if MODE['passthrough'] else torch.zeros_like(labels, dtype=torch.bool)
    negatives = torch.nonzero((labels == 0) & ~hop).squeeze(-1)  # chains are never subsampled
    if len(negatives) > D.MAX_NEGATIVES:
        keep = torch.cat([torch.nonzero((labels > 0) | hop).squeeze(-1), negatives[torch.randperm(len(negatives), device=negatives.device)[:D.MAX_NEGATIVES]]])
        pairs, labels, hop = pairs[keep], labels[keep], hop[keep]
    paths, traced = build_paths(cand, pairs, contacts, dist_b, label_b, hops)
    via = via_points(cand, pairs, hops) if hops else None
    x = model.path_features(feats, heat, paths, prob_b, via)
    ordinary = ~hop  # with passthrough, two-hop rows train the pass-through head (and geometry), not the edge head
    if ordinary.any():
        logits = model.edge_head(x[ordinary]).squeeze(-1)
        pos = labels[ordinary].sum()
        edge = F.binary_cross_entropy_with_logits(logits, labels[ordinary], pos_weight=((ordinary.sum() - pos) / pos.clamp_min(1)).clamp(max=20.0))
    else:
        edge = x.sum() * 0
    passthrough = None
    if hop.any():
        c = torch.tensor([hops[tuple(pr)] for pr in pairs[hop].tolist()], device=cand.device)
        spurious = torch.tensor([int(i) not in mapping for i in c.tolist()], device=cand.device, dtype=labels.dtype)
        target = labels[hop] * spurious  # C is a pass-through point iff C is spurious and A-B is a GT edge
        z = model.pass_logits(feats, x[hop], via[hop])
        npos = target.sum()
        passthrough = F.binary_cross_entropy_with_logits(z, target, pos_weight=((len(target) - npos) / npos.clamp_min(1)).clamp(max=20.0))
    geo = None
    positive = labels.bool()
    if positive.any():
        m = metric[None].expand(int(positive.sum()), 3, 3)
        interior = choose_interior(to_mm(paths[positive], m), traced[positive], voxel_size(metric))
        curve = model.path_curves(x[positive], paths[positive], interior, traced[positive])
        ref = D.guide_for(pairs[positive], mapping, gt_edges, guides, gt_vox)
        curve_mm, ref_mm = to_mm(curve, m), to_mm(ref, m)
        geo = chamfer(curve_mm, ref_mm)
        if MODE['arc_penalty']:
            arc_pred = (curve_mm[:, 1:] - curve_mm[:, :-1]).norm(dim=-1).sum(-1)
            arc_ref = (ref_mm[:, 1:] - ref_mm[:, :-1]).norm(dim=-1).sum(-1)
            geo = geo + MODE['arc_penalty'] * F.relu(arc_pred - 1.02 * arc_ref - 0.1)
    return edge, geo, traced, passthrough


def hop_rows(pairs, contacts, hops):
    """Rows of `pairs` that exist only as two-hop chains A-C-B."""
    return torch.tensor([bool(hops) and tuple(pr) in hops and tuple(pr) not in contacts for pr in pairs.tolist()],
                        dtype=torch.bool, device=pairs.device)


def contract(row, node_t, edge_t, tau):
    """Final graph of a passthrough row, encoded in its scores. Two-hop rows never count as ordinary edges;
    with tau set, chains A-C-B are visited by decreasing pass-through score and, if score >= tau and C has no
    accepted edge other than to A or B, C and its edges are removed and A-B (joined-path curve) is accepted."""
    if row.get('pass_scores') is None or row['edge_scores'] is None:
        return {**row, 'contracted': []}  # e.g. a patch with fewer than two detections
    ns, es = row['node_scores'].copy(), row['edge_scores'].copy()
    pairs, via, q = row['pairs'], row['via'], row['pass_scores']
    hop = via >= 0
    es[hop] = 0.0
    contracted = []
    if tau is not None:
        acc = (es >= edge_t) & (ns[pairs[:, 0]] >= node_t) & (ns[pairs[:, 1]] >= node_t)
        removed = set()
        for j in np.argsort(-np.where(hop, q, -1.0)):
            if not hop[j] or q[j] < tau:
                break
            a, b, c = int(pairs[j, 0]), int(pairs[j, 1]), int(via[j])
            if {a, b, c} & removed or min(ns[a], ns[b], ns[c]) < node_t:
                continue
            inc = np.nonzero(acc & ((pairs[:, 0] == c) | (pairs[:, 1] == c)))[0]
            if not set(pairs[inc].ravel().tolist()) - {c} <= {a, b}:
                continue
            es[inc], acc[inc] = 0.0, False
            ns[c] = -1.0
            removed.add(c)
            es[j], acc[j] = 1.0, True
            contracted.append((a, c, b))
    return {**row, 'node_scores': ns, 'edge_scores': es, 'contracted': contracted}


@torch.no_grad()
def predict(model, ds, device, node_threshold, workers, batch=8):
    model.eval()
    rows = []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=collate)
    offset = 0
    for images, nodes, edges, geometry in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model.forward_image(images.to(device))
        feats, heat, prob = out[:, 5:].float(), out[:, :1].float().sigmoid(), out[:, 4].float().sigmoid()
        if PROB_HOOK is not None:
            prob = PROB_HOOK(prob, nodes, geometry)
        found = D.peaks(out, node_threshold)
        contacts, dist, label = traced_pairs(prob, [p for p, _ in found])
        for b, (p, score) in enumerate(found):
            metric = geometry[b]['metric'].to(device) / SIZE
            row = dict(nodes=(p / SIZE).cpu().numpy(), node_scores=score.cpu().numpy(), pairs=np.zeros((0, 2), int),
                       edge_scores=None, curves=None, traced=None, interior=None, via=None)
            hops = two_hop_pairs(contacts[b]) if MODE['two_hop'] else None
            pairs = patch_candidates(p, metric, contacts[b], hops) if len(p) >= 2 else None
            if pairs is not None and len(pairs):
                paths, traced = build_paths(p, pairs, contacts[b], dist[b], label[b], hops)
                x = model.path_features(feats[b], heat[b], paths, prob[b], via_points(p, pairs, hops) if hops else None)
                hop = hop_rows(pairs, contacts[b], hops) if hops else torch.zeros(len(pairs), dtype=torch.bool, device=device)
                if hops or MODE['passthrough']:
                    row['via'] = np.where(hop.cpu().numpy(), [hops.get(tuple(pr), -1) if hops else -1 for pr in pairs.tolist()], -1)
                if MODE['passthrough']:
                    q = np.full(len(pairs), np.nan)
                    if hop.any():
                        via = via_points(p, pairs, hops)
                        q[hop.cpu().numpy()] = model.pass_logits(feats[b], x[hop], via[hop]).sigmoid().cpu().numpy()
                    row['pass_scores'] = q
                interior = choose_interior(to_mm(paths, metric[None].expand(len(paths), 3, 3)), traced, voxel_size(metric))
                curves = model.path_curves(x, paths, interior, traced)
                if MODE['compress_tol']:
                    curves, kept = compress_curves(curves, MODE['compress_tol'], interior)
                    row['interior_compressed'] = kept.cpu().numpy()
                row.update(pairs=pairs.cpu().numpy(), edge_scores=model.edge_head(x).squeeze(-1).sigmoid().cpu().numpy(),
                           curves=(curves / SIZE).cpu().numpy(), traced=traced.cpu().numpy(), interior=interior.cpu().numpy())
            record = ds.records[offset + b]
            row.update(sample_id=record.sample_id, cache=str(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz'))
            rows.append(row)
        offset += len(images)
    model.train()
    return rows


def length_breakdown(rows, node_t, edge_t):
    """GT edge outcome by arc length: recovered / endpoints missing / not candidate / rejected."""
    bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]
    tally = {f'{lo}-{hi if hi < 1e8 else "inf"}': dict(edges=0, recovered=0, endpoints_missing=0, not_candidate=0, rejected=0, traced_candidate=0) for lo, hi in bins}
    for r in rows:
        with np.load(r['cache']) as z:
            M, gn, ge = z['metric'], z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2)
            P, o = z['full_paths'] @ z['metric'].T, z['path_offsets']
        keep = np.nonzero(r['node_scores'] >= node_t)[0]
        m = D.match(torch.as_tensor(r['nodes'][keep] @ M.T), torch.as_tensor(gn))
        inv = {g: i for i, g in m.items()}
        score = {tuple(sorted(map(int, p))): s for p, s in zip(r['pairs'], r['edge_scores'])} if r['edge_scores'] is not None else {}
        traced = {tuple(sorted(map(int, p))): t for p, t in zip(r['pairs'], r['traced'])} if r.get('traced') is not None else {}
        for e, (a, b) in enumerate(ge):
            arc = np.linalg.norm(np.diff(P[o[e]:o[e + 1]], axis=0), axis=1).sum()
            name = next(f'{lo}-{hi if hi < 1e8 else "inf"}' for lo, hi in bins if lo <= arc < hi)
            t = tally[name]
            t['edges'] += 1
            if a not in inv or b not in inv:
                t['endpoints_missing'] += 1
                continue
            k = tuple(sorted((int(keep[inv[a]]), int(keep[inv[b]]))))
            if k not in score:
                t['not_candidate'] += 1
                continue
            t['traced_candidate'] += int(bool(traced.get(k, False)))
            t['recovered' if score[k] >= edge_t else 'rejected'] += 1
    return {k: {**v, **{f'{f}_fraction': v[f] / max(1, v['edges']) for f in ('recovered', 'endpoints_missing', 'not_candidate', 'rejected')}} for k, v in tally.items()}


def final_evaluation(model, a, monitor, val_ids, device, out_dir, name='final.json'):
    monitor_ds, val_ds = make_dataset(a.dataset, monitor, augment=False), make_dataset(a.dataset, val_ids, augment=False)
    plain = lambda rows: [contract(r, None, None, None) for r in rows]  # identity unless passthrough
    node_t = D.select_node_threshold(plain(predict(model, monitor_ds, device, min(D.NODE_THRESHOLDS), a.workers, a.eval_batch)))
    raw_train = predict(model, monitor_ds, device, node_t, a.workers, a.eval_batch)
    edge_t = D.select_edge_threshold(plain(raw_train), node_t)
    raw_val = predict(model, val_ds, device, node_t, a.workers, a.eval_batch)
    (out_dir / name.replace('.json', '-validation-rows.pkl')).write_bytes(pickle.dumps(raw_val))
    tau, passthrough = None, None
    if MODE['passthrough']:  # pass-through threshold on the train monitor, after node and edge thresholds
        f1 = {t: D.evaluate([contract(r, node_t, edge_t, t) for r in raw_train], node_t, edge_t, geometry=False)['edge_f1']
              for t in [None] + PASS_THRESHOLDS}
        tau = max(f1, key=lambda t: (f1[t], t is None))
        val_plain = plain(raw_val)
        passthrough = dict(threshold=tau, train_monitor_edge_f1_by_threshold={str(k): v for k, v in f1.items()},
                           validation_without_contraction=D.evaluate(val_plain, node_t, edge_t),
                           contracted_per_patch=float(np.mean([len(contract(r, node_t, edge_t, tau)['contracted']) for r in raw_val])))
    train_rows = [contract(r, node_t, edge_t, tau) for r in raw_train]
    val_rows = [contract(r, node_t, edge_t, tau) for r in raw_val]
    traced_share = float(np.mean(np.concatenate([r['traced'] for r in val_rows if r['traced'] is not None] or [np.zeros(0)])))
    interior = np.concatenate([r['interior'] for r in val_rows if r['interior'] is not None] or [np.zeros(0, int)])
    result = dict(dataset=a.dataset, model='traced+adaptive', mode=dict(MODE), offset_radius=getattr(a, 'offset_radius', 0.0), backbone=getattr(a, 'backbone', 'unet'),
                  feature_channels=getattr(a, 'feature_channels', FEATURE_CHANNELS),
                  updates=a.updates, batch=a.batch, lr=a.lr,
                  thresholds=dict(node=node_t, edge=edge_t, policy='train monitor only; candidates = geodesic contacts' + (' U 16-NN over retained nodes' if MODE['knn_fallback'] else ' only')),
                  train_monitor=D.evaluate(train_rows, node_t, edge_t), validation=D.evaluate(val_rows, node_t, edge_t),
                  validation_by_arc_length=length_breakdown(val_rows, node_t, edge_t),
                  validation_candidates=dict(traced_fraction=traced_share, interior_histogram={str(m): int((interior == m).sum()) for m in INTERIOR}),
                  passthrough=passthrough)
    (out_dir / name).write_text(json.dumps(result, indent=1))
    print('FINAL ' + json.dumps(result), flush=True)


def neighbourhood_offsets(nodes, device, radius):
    """Offset targets at every voxel within `radius` of each GT node's rounded voxel:
    target = GT position - voxel; a voxel near several nodes takes the nearest."""
    r = int(np.floor(radius))
    g = torch.arange(-r, r + 1, device=device)
    ball = torch.stack(torch.meshgrid(g, g, g, indexing='ij'), -1).view(-1, 3)
    ball = ball[(ball.float() ** 2).sum(-1) <= radius ** 2]
    B = len(nodes)
    offset = torch.zeros(B, 3, SIZE, SIZE, SIZE, device=device)
    mask = torch.zeros(B, SIZE, SIZE, SIZE, dtype=torch.bool, device=device)
    for b, n in enumerate(nodes):
        v = n.to(device) * SIZE
        if not len(v):
            continue
        vox = (v.round().long()[:, None] + ball[None]).view(-1, 3)
        src = v[:, None].expand(-1, len(ball), 3).reshape(-1, 3)
        ok = ((vox >= 0) & (vox < SIZE)).all(1)
        vox, src = vox[ok], src[ok]
        d = (src - vox.float()).norm(dim=-1)
        flat = (vox[:, 0] * SIZE + vox[:, 1]) * SIZE + vox[:, 2]
        best = torch.full((SIZE ** 3,), float('inf'), device=device).scatter_reduce(0, flat, d, 'amin')
        win = d <= best[flat]
        vox, src = vox[win], src[win]
        offset[b, :, vox[:, 0], vox[:, 1], vox[:, 2]] = (src - vox.float()).T
        mask[b, vox[:, 0], vox[:, 1], vox[:, 2]] = True
    return offset, mask


def warm_start(model, path, device):
    """Load every tensor whose name and shape match (U-Net, corridor CNN, ...); report the rest."""
    source = torch.load(path, map_location=device)['model']
    own = model.state_dict()
    loaded = {k: v for k, v in source.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(loaded, strict=False)
    return dict(source=str(path), loaded=len(loaded), fresh=sorted(set(own) - set(loaded)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=sorted(json.loads(PLAN.read_text())['datasets']), required=True)  # any plan entry
    p.add_argument('--updates', type=int, default=15000)
    p.add_argument('--warmup-gt', type=int, default=2000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--train-peak-threshold', type=float, default=0.2)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--eval-batch', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--init-from', type=Path, help='Warm start from a dense-model checkpoint (matching tensors)')
    p.add_argument('--run-name', default=None, help='Output subfolder name')
    p.add_argument('--soft-tracing', action='store_true', help='No hard centreline mask: cost 1/(p+0.05) everywhere')
    p.add_argument('--no-knn-fallback', action='store_true', help='Candidates = geodesic contacts only')
    p.add_argument('--gap-features', action='store_true', help='Edge/geometry heads read centreline statistics along the path')
    p.add_argument('--two-hop', action='store_true', help='Also propose A-B through a traced pass-through node C')
    p.add_argument('--passthrough', action='store_true', help='Node-level decision for two-hop chains (implies --two-hop)')
    p.add_argument('--backbone', choices=D.BACKBONES, default='unet',
                   help='Must match the --init-from checkpoint')
    p.add_argument('--offset-radius', type=float, default=1.75,
                   help='Supervise offsets at all voxels within this radius of each GT node; default 1.75 (3x3x3) since '
                        '3 Nov 2026, 0 = GT voxel only (all runs before traced_twohop_offset)')
    p.add_argument('--arc-penalty', type=float, default=0.1,
                   help='Weight of the excess-arc term relu(arc_pred - 1.02 arc_gt - 0.1 mm) in the geometry loss. '
                        'Default 0.1 since 9 Nov 2026; 0 = the original loss (all runs before traced_arcpen_r1109)')
    p.add_argument('--feature-channels', type=int, default=FEATURE_CHANNELS,
                   help='Shared feature channels F (must match the --init-from checkpoint); 32 = all runs before 9 Nov 2026')
    p.add_argument('--k-min-arc', type=float, nargs=2, default=None, metavar=('K16', 'K24'),
                   help='Minimum traced arc (mm) before arc/chord may raise K to K16 / K24. Training default 8 20 (since '
                        '9 Nov 2026); 0 0 = original rule. --evaluate-only defaults to the rule recorded in final.json '
                        '(original rule if none is recorded)')
    p.add_argument('--k-rule', choices=['ratio', 'fit', 'fit_length', 'fixed24'], default=None,
                   help="'ratio' (default): arc and arc/chord thresholds (+ --k-min-arc); 'fit': fewest controls whose "
                        "least-squares fit of the smoothed traced path is within --k-fit-tol voxels; 'fit_length': fit "
                        "with arc-length floors (>= 15 mm: K16, >= 35 mm: K24); 'fixed24': one K24 head for every pair")
    p.add_argument('--k-fit-tol', type=float, default=None, help='Fit rule tolerance in voxels (default 0.5)')
    p.add_argument('--k-fit-smooth-mm', type=float, default=None, help='Fit rule path smoothing scale in mm (default 1.0)')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--breakdown-rows', type=Path, help='Only compute the arc-length breakdown for saved rows (CPU)')
    p.add_argument('--breakdown-thresholds', type=float, nargs=2)
    a = p.parse_args()
    MODE.update(arc_penalty=a.arc_penalty)
    MODE.update(soft=a.soft_tracing, knn_fallback=not a.no_knn_fallback, gap_features=a.gap_features,
                two_hop=a.two_hop or a.passthrough, passthrough=a.passthrough)
    if a.breakdown_rows:
        rows = pickle.loads(a.breakdown_rows.read_bytes())
        print(json.dumps(length_breakdown(rows, *a.breakdown_thresholds), indent=1))
        return
    torch.manual_seed(364505)
    np.random.seed(364505)
    out_dir = a.output / a.dataset / (a.run_name or 'traced_adaptive')
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    if a.evaluate_only and (out_dir / 'final.json').exists():  # reproduce the K rule the model was evaluated with
        saved = json.loads((out_dir / 'final.json').read_text()).get('mode', {})
    if a.evaluate_only:  # the penalty only affects training: keep the value the model was trained with in final.json
        MODE['arc_penalty'] = saved.get('arc_penalty', 0.0)
    for key, arg, train_default in (('k_min_arc', a.k_min_arc, K_MIN_ARC_DEFAULT), ('k_rule', a.k_rule, 'ratio'),
                                    ('k_fit_tol', a.k_fit_tol, MODE_DEFAULTS['k_fit_tol']),
                                    ('k_fit_smooth_mm', a.k_fit_smooth_mm, MODE_DEFAULTS['k_fit_smooth_mm'])):
        if arg is not None:
            MODE[key] = arg
        elif a.evaluate_only:
            MODE[key] = saved.get(key, MODE_DEFAULTS[key])
        else:
            MODE[key] = train_default
    MODE['k_min_arc'] = tuple(MODE['k_min_arc'])
    plan = json.loads(PLAN.read_text())['datasets'][a.dataset]
    train_ids, val_ids, monitor = plan['train_ids'], plan['val_ids'], plan['monitor_train']
    if a.smoke:
        train_ids, val_ids, monitor = train_ids[:8], val_ids[:4], train_ids[:4]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TracedDenseGraph(gap_features=MODE['gap_features'], two_hop=MODE['two_hop'], passthrough=MODE['passthrough'],
                             backbone=a.backbone, feature_channels=a.feature_channels).to(device)
    if a.evaluate_only:
        model.load_state_dict(torch.load(out_dir / 'model.pt', map_location=device)['model'])
        final_evaluation(model, a, monitor, val_ids, device, out_dir, name='final-v2.json')
        return
    if a.init_from:
        (out_dir / 'warm-start.json').write_text(json.dumps(warm_start(model, a.init_from, device), indent=1))
    train_ds = make_dataset(a.dataset, train_ids, augment=True)
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
        if a.offset_radius > 0:
            offset_t, peak_t = neighbourhood_offsets(nodes, device, a.offset_radius)
        m = peak_t.unsqueeze(1).expand_as(offset_t)
        off_loss = F.l1_loss(out[:, 1:4].float()[m], offset_t[m]) if m.any() else out.sum() * 0
        line_loss = F.binary_cross_entropy_with_logits(out[:, 4:5].float(), line_t, pos_weight=torch.tensor(5.0, device=device))
        feats, heat, prob = out[:, 5:].float(), out[:, :1].float().sigmoid().detach(), out[:, 4].float().sigmoid().detach()
        sources = [('gt', None)] + ([('pred', D.peaks(out.detach(), a.train_peak_threshold))] if step > a.warmup_gt else [])
        edge_terms, geo_terms, traced_counts, pass_terms = [], [], [], []
        for kind, predicted in sources:
            cands, maps = [], []
            for b in range(len(images)):
                metric = geometry[b]['metric'].to(device) / SIZE
                gt_vox = nodes[b].to(device) * SIZE
                if kind == 'gt':
                    c = gt_vox + torch.randn_like(gt_vox) * (0.5 / float(metric[0].norm()))
                    mp = {i: i for i in range(len(gt_vox))}
                else:
                    c = predicted[b][0][:D.MAX_PEAKS]
                    mp = D.match(c @ metric.T, gt_vox @ metric.T)
                cands.append(c)
                maps.append(mp)
            contacts, dist, label = traced_pairs(prob, cands)
            for b in range(len(images)):
                metric = geometry[b]['metric'].to(device) / SIZE
                res = patch_terms(model, feats[b], heat[b], cands[b], nodes[b].to(device) * SIZE, edges[b].to(device),
                                  geometry[b]['guides'].to(device) * SIZE, metric, maps[b], contacts[b], dist[b], label[b], prob[b])
                if res is None:
                    continue
                edge_terms.append(res[0])
                if res[1] is not None:
                    geo_terms.append(res[1])
                traced_counts.append(float(res[2].float().mean()))
                if res[3] is not None:
                    pass_terms.append(res[3])
        edge_loss = torch.stack(edge_terms).mean() if edge_terms else out.sum() * 0
        pass_loss = torch.stack(pass_terms).mean() if pass_terms else out.sum() * 0
        geo_loss = torch.cat(geo_terms).mean() if geo_terms else out.sum() * 0
        loss = node_loss + off_loss + line_loss + edge_loss + geo_loss + pass_loss
        assert torch.isfinite(loss), 'nonfinite loss'
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or a.smoke:
            row = dict(step=step, seconds=time.time() - start, node=float(node_loss), centerline=float(line_loss), edge=float(edge_loss),
                       passthrough=float(pass_loss),
                       max_memory_gb=torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0.0,
                       geometry_mm=float(geo_loss), traced_pair_fraction=float(np.mean(traced_counts)) if traced_counts else 0.0,
                       predicted_candidates=step > a.warmup_gt)
            log.write(json.dumps(row) + '\n')
            log.flush()
            print('STEP ' + json.dumps(row), flush=True)
    torch.save(dict(model=model.state_dict(), args={k: str(v) for k, v in vars(a).items()}), out_dir / 'model.pt')
    final_evaluation(model, a, monitor, val_ids, device, out_dir)


if __name__ == '__main__':
    main()
