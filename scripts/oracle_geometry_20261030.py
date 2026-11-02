#!/usr/bin/env python3
"""WS4a oracle-topology edge-geometry learning (30 Oct 2026).

Question: given the GT graph (endpoints + connectivity), can edge shape be learned
from the image, and is a continuous cubic B-spline better than a straight chord and
than a polyline of degree-2 vertices with the same (or larger) output count?

All arms share the same U-Net, corridor sampler (features sampled in a tube around the
chord, so off-chord vessels are visible), network, loss and schedule; only the decoding
of the M predicted interior points differs: 'spline' = clamped cubic B-spline controls
(K = M + 2), 'polyline' = M degree-2 vertices. The chord and the cached GT K8 fit are
evaluated in every run. Reference = original dense path (cache 'guides'). Detection is
deliberately excluded (oracle topology); this is not a full graph-reconstruction result.
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
from node_heatmap_probe_20261030 import PLAN, collate, make_dataset  # noqa: E402

SIZE = 64
SAMPLES = 64
L, S = 24, 11  # corridor samples along / across the chord


def frame(a, b):
    d = b - a
    length = d.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    t = d / length
    helper = torch.eye(3, device=a.device)[t.abs().argmin(-1)]
    u = F.normalize(torch.cross(t, helper, dim=-1), dim=-1)
    return d, t, u, torch.cross(t, u, dim=-1)


def polyline(points, samples=SAMPLES):
    """Sample a polyline (E,P,3) at `samples` uniform parameter values."""
    P = points.shape[1]
    s = torch.linspace(0, P - 1, samples, device=points.device)
    i = s.floor().long().clamp(max=P - 2)
    w = (s - i)[None, :, None]
    return points[:, i] * (1 - w) + points[:, i + 1] * w


class Geometry(nn.Module):
    def __init__(self, mode, interior, radius):
        super().__init__()
        from monai.networks.nets import BasicUNet
        self.mode, self.m, self.radius = mode, interior, float(radius)
        self.unet = BasicUNet(spatial_dims=3, in_channels=1, out_channels=33, features=(32, 32, 64, 128, 256, 32))
        self.corridor = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv3d(64, 64, 3, stride=(2, 2, 2), padding=1), nn.GELU(),
            nn.Conv3d(64, 64, 3, stride=(1, 2, 2), padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Linear(64 * 12 * 3 * 3 + 4, 256), nn.GELU(), nn.Linear(256, 3 * interior))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if mode == 'spline':
            basis, greville = cubic_basis(interior + 2, SAMPLES)
            self.register_buffer('basis', basis)
            self.register_buffer('anchor', greville[1:-1])
        else:
            self.register_buffer('anchor', torch.arange(1, interior + 1) / (interior + 1))

    def sample(self, volume, a, b):
        """Corridor features (E,C,L,S,S) around chords a->b given in voxel coordinates."""
        d, t, u, w = frame(a, b)
        s = torch.linspace(-0.1, 1.1, L, device=a.device)
        o = torch.linspace(-self.radius, self.radius, S, device=a.device)
        p = (a[:, None, None, None] + s[None, :, None, None, None] * d[:, None, None, None]
             + o[None, None, :, None, None] * u[:, None, None, None] + o[None, None, None, :, None] * w[:, None, None, None])
        grid = (p / (SIZE - 1) * 2 - 1).flip(-1)  # (D,H,W) voxels -> (x=W,y=H,z=D) in [-1,1]
        return F.grid_sample(volume[None].expand(1, -1, -1, -1, -1), grid[None].reshape(1, -1, S, S, 3),
                             align_corners=True, padding_mode='zeros').view(volume.shape[0], len(a), L, S, S).transpose(0, 1)

    def curves(self, features, a, b, voxel_mm):
        """Predicted dense curves (E,SAMPLES,3) in voxel coordinates."""
        d, t, u, w = frame(a, b)
        x = self.corridor(features).flatten(1)
        length = d.norm(dim=-1, keepdim=True)
        x = torch.cat([x, length / SIZE, (d / SIZE)], -1)
        local = self.head(x).view(len(a), self.m, 3) * self.radius
        interior = (a[:, None] + self.anchor[None, :, None] * d[:, None]
                    + local[..., :1] * t[:, None] + local[..., 1:2] * u[:, None] + local[..., 2:] * w[:, None])
        points = torch.cat([a[:, None], interior, b[:, None]], 1)
        return self.basis @ points if self.mode == 'spline' else polyline(points)


def edge_batch(nodes, edges, geometry, device):
    """Flatten GT edges: endpoints (voxels), dense reference (voxels), metric, patch id."""
    A, B, ref, fit, metric, owner = [], [], [], [], [], []
    for k, (n, e, g) in enumerate(zip(nodes, edges, geometry)):
        if len(e) == 0:
            continue
        v = n.to(device) * SIZE
        guides = g['guides'].to(device) * SIZE
        a, b = v[e[:, 0]], v[e[:, 1]]
        flip = (guides[:, 0] - a).norm(dim=-1) > (guides[:, 0] - b).norm(dim=-1)
        guides = torch.where(flip[:, None, None], guides.flip(1), guides)
        keep = (b - a).norm(dim=-1) > 0.5
        A.append(a[keep]); B.append(b[keep]); ref.append(guides[keep])
        fit.append((g['curves'].to(device) * SIZE)[keep])
        metric.append((g['metric'].to(device) / SIZE).expand(int(keep.sum()), 3, 3))
        owner += [k] * int(keep.sum())
    if not A:
        return None
    return torch.cat(A), torch.cat(B), torch.cat(ref), torch.cat(fit), torch.cat(metric), torch.tensor(owner, device=device)


def to_mm(points, metric):
    return torch.einsum('epi,eji->epj', points, metric)


def distances(pred_mm, ref_mm):
    """Per-edge: mean symmetric, Hausdorff, coverage of reference at 0.5/1 mm."""
    dense = polyline(pred_mm, 256)
    d = torch.cdist(ref_mm, dense)
    ref_to_pred = d.amin(-1)
    pred_to_ref = torch.cdist(dense, polyline(ref_mm, 256)).amin(-1)
    return dict(mean=(ref_to_pred.mean(-1) + pred_to_ref.mean(-1)) / 2,
                hausdorff=torch.maximum(ref_to_pred.amax(-1), pred_to_ref.amax(-1)),
                cover05=(ref_to_pred <= 0.5).float().mean(-1), cover10=(ref_to_pred <= 1.0).float().mean(-1))


def chord_deviation(ref_mm):
    a, b = ref_mm[:, :1], ref_mm[:, -1:]
    t = F.normalize(b - a, dim=-1)
    r = ref_mm - a
    return (r - (r * t).sum(-1, keepdim=True) * t).norm(dim=-1).amax(-1)


def chamfer(pred_mm, ref_mm):
    d = torch.cdist(polyline(pred_mm, 128), polyline(ref_mm, 128))
    return (d.amin(-1).mean(-1) + d.amin(-2).mean(-1)) / 2


@torch.no_grad()
def evaluate(model, ds, device, batch, workers):
    model.eval()
    rows = {k: [] for k in ('sample', 'deviation', 'length')}
    metrics = {}
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=collate)
    offset = 0
    for images, nodes, edges, geometry in loader:
        eb = edge_batch(nodes, edges, geometry, device)
        if eb is not None:
            a, b, ref, fit, metric, owner = eb
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.unet(images.to(device))
            feats = out[:, :32].float()
            pred = torch.cat([model.curves(model.sample(feats[k], a[owner == k], b[owner == k]),
                                           a[owner == k], b[owner == k], None) for k in owner.unique()])
            ref_mm = to_mm(ref, metric)
            chord = torch.stack([a, b], 1)
            for name, curve in (('learned', pred), ('chord', chord), ('gt_k8_fit', fit)):
                for key, value in distances(to_mm(curve, metric), ref_mm).items():
                    metrics.setdefault(name + '_' + key, []).append(value.cpu())
            rows['deviation'].append(chord_deviation(ref_mm).cpu())
            rows['length'].append((ref_mm[:, 1:] - ref_mm[:, :-1]).norm(dim=-1).sum(-1).cpu())
            rows['sample'] += [ds.records[offset + int(k)].sample_id for k in owner.cpu()]
        offset += len(images)
    model.train()
    out = {k: torch.cat(v).numpy() for k, v in metrics.items()}
    out['deviation'] = torch.cat(rows['deviation']).numpy()
    out['length'] = torch.cat(rows['length']).numpy()
    out['sample'] = np.array(rows['sample'])
    return out


def summarize(e):
    bins = [(0, .5), (.5, 1), (1, 2), (2, 4), (4, 1e9)]
    result = dict(edges=int(len(e['deviation'])))
    for name in ('learned', 'chord', 'gt_k8_fit'):
        result[name] = {k: float(e[f'{name}_{k}'].mean()) for k in ('mean', 'hausdorff', 'cover05', 'cover10')}
        result[name]['by_chord_deviation_mm'] = {
            f'{lo}-{hi if hi < 1e8 else "inf"}': dict(edges=int(m.sum()), mean=float(e[f'{name}_mean'][m].mean()),
                                                     cover10=float(e[f'{name}_cover10'][m].mean()))
            for lo, hi in bins if (m := (e['deviation'] >= lo) & (e['deviation'] < hi)).any()}
    subjects = np.array([s.split('_')[1] for s in e['sample']])
    diff = e['learned_mean'] - e['chord_mean']
    ids = np.unique(subjects)
    per = np.array([diff[subjects == s].mean() for s in ids])
    rng = np.random.default_rng(261030)
    boot = [per[rng.integers(len(per), size=len(per))].mean() for _ in range(2000)]
    result['learned_minus_chord_mean_mm'] = dict(subject_mean=float(per.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], subjects=int(len(ids)))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['IXI', 'BRAVA', 'SyntheticMRI'], required=True)
    p.add_argument('--mode', choices=['spline', 'polyline'], required=True)
    p.add_argument('--interior', type=int, default=6, help='Predicted interior points (spline K = interior + 2)')
    p.add_argument('--radius', type=float, default=12.0, help='Corridor half-width in voxels')
    p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--eval-batch', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    torch.manual_seed(364505)
    np.random.seed(364505)
    name = f'{a.mode}{a.interior}'
    out_dir = a.output / a.dataset / name
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text())['datasets'][a.dataset]
    train_ids, val_ids, monitor = plan['train_ids'], plan['val_ids'], plan['monitor_train']
    if a.smoke:
        train_ids, val_ids, monitor = train_ids[:8], val_ids[:4], train_ids[:4]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = make_dataset(a.dataset, train_ids, augment=True)
    model = Geometry(a.mode, a.interior, a.radius).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.updates, pct_start=0.05)
    sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=a.updates * a.batch)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                         collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
    log = (out_dir / 'training.jsonl').open('a')
    start, step = time.time(), 0
    pos_weight = torch.tensor(5.0, device=device)
    for images, nodes, edges, geometry in loader:
        step += 1
        eb = edge_batch(nodes, edges, geometry, device)
        if eb is None:
            continue
        A, B, ref, fit, metric, owner = eb
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            out = model.unet(images.to(device))
        feats = out[:, :32].float()
        # Auxiliary centerline from the dense references keeps the features vessel-aware.
        line = torch.zeros_like(out[:, 32:33], dtype=torch.float32)
        j = F.interpolate(ref.transpose(1, 2), size=256, mode='linear', align_corners=True).transpose(1, 2).reshape(-1, 3).round().clamp(0, SIZE - 1).long()
        line[owner.repeat_interleave(256), 0, j[:, 0], j[:, 1], j[:, 2]] = 1.0
        aux = F.binary_cross_entropy_with_logits(out[:, 32:33].float(), F.max_pool3d(line, 3, 1, 1), pos_weight=pos_weight)
        pred = torch.cat([model.curves(model.sample(feats[k], A[owner == k], B[owner == k]), A[owner == k], B[owner == k], None)
                          for k in owner.unique()])
        ref = torch.cat([ref[owner == k] for k in owner.unique()])
        metric = torch.cat([metric[owner == k] for k in owner.unique()])
        geo = chamfer(to_mm(pred, metric), to_mm(ref, metric)).mean()
        loss = geo + 0.5 * aux
        assert torch.isfinite(loss), 'nonfinite loss'
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or a.smoke:
            row = dict(step=step, seconds=time.time() - start, chamfer_mm=float(geo), centerline=float(aux), edges=len(A))
            log.write(json.dumps(row) + '\n')
            log.flush()
            print('STEP ' + json.dumps(row), flush=True)
    torch.save(dict(model=model.state_dict(), args={k: str(v) for k, v in vars(a).items()}), out_dir / 'model.pt')
    result = dict(dataset=a.dataset, mode=a.mode, interior=a.interior, controls_or_vertices=a.interior + 2,
                  radius_voxels=a.radius, updates=a.updates, batch=a.batch, lr=a.lr)
    for split, ids in (('train_monitor', monitor), ('validation', val_ids)):
        e = evaluate(model, make_dataset(a.dataset, ids, augment=False), device, a.eval_batch, a.workers)
        np.savez_compressed(out_dir / f'{split}-edges.npz', **e)
        result[split] = summarize(e)
        print('EVAL ' + split + ' ' + json.dumps({k: result[split][k] for k in ('learned', 'chord', 'gt_k8_fit')}, default=str)[:600], flush=True)
    (out_dir / 'final.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
