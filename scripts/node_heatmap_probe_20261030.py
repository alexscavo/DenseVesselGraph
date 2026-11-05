#!/usr/bin/env python3
"""WS2 node-detectability probe (30 Oct 2026).

Question: are the unchanged contracted graph nodes of IXI/BRAVA detectable from the
image by a dense full-resolution detector, or is the stride-32 RelationFormer set
predictor the limit? A 3D U-Net predicts a CenterNet node heatmap, sub-voxel offsets
and an auxiliary centerline mask (from the K8 cache guides). Same cohorts, loader,
quarter-turn/noise augmentation and targets as the multidataset recipes. Thresholds are
selected on the training monitor only; validation is never used for selection.
Not a graph-reconstruction method: nodes only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.loaders.discovery import discover_synthetic_mri  # noqa: E402
from data.loaders.synthetic_mri import SyntheticMRIDataset  # noqa: E402

PLAN = ROOT / 'experiments/multidataset_continuous/plan.json'
SIZE = 64
SIGMA = 1.5  # voxels
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def records(dataset, ids):
    root = Path(json.loads(PLAN.read_text())['datasets'][dataset]['root'])
    by_id = {}
    for split in ('train', 'val'):
        for r in discover_synthetic_mri(root, split):
            by_id[r.sample_id] = r
    return [by_id[i] for i in ids]


def make_dataset(dataset, ids, augment):
    return SyntheticMRIDataset(records(dataset, ids), augment=augment, zoom_range=None,
                               spline_controls=8, foreground_mean=0.33335259556770325)


def collate(batch):
    images = torch.stack([b[0][0] for b in batch])
    return images, [b[2][0][:, :3] for b in batch], [b[3][0] for b in batch], [b[4][0] for b in batch]


GRID = None


def targets(nodes, geometry, device):
    """Heatmap (peak at rounded voxel), offsets, peak mask and centerline mask."""
    global GRID
    if GRID is None:
        axis = torch.arange(SIZE, device=device, dtype=torch.float32)
        GRID = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).view(-1, 3)
    B = len(nodes)
    heat = torch.zeros(B, SIZE ** 3, device=device)
    offset = torch.zeros(B, 3, SIZE, SIZE, SIZE, device=device)
    peak = torch.zeros(B, SIZE, SIZE, SIZE, dtype=torch.bool, device=device)
    line = torch.zeros(B, 1, SIZE, SIZE, SIZE, device=device)
    for b, (n, g) in enumerate(zip(nodes, geometry)):
        v = n.to(device) * SIZE
        if len(v):
            c = v.round().clamp(0, SIZE - 1)
            for chunk in torch.split(c, 32):
                d2 = ((GRID[:, None, :] - chunk[None]) ** 2).sum(-1)
                heat[b] = torch.maximum(heat[b], torch.exp(-d2 / (2 * SIGMA ** 2)).amax(1))
            i = c.long()
            peak[b, i[:, 0], i[:, 1], i[:, 2]] = True
            offset[b, :, i[:, 0], i[:, 1], i[:, 2]] = (v - c).T
        guides = g['guides'].to(device) * SIZE  # (E,64,3)
        if len(guides):
            dense = F.interpolate(guides.transpose(1, 2), size=256, mode='linear', align_corners=True).transpose(1, 2)
            j = dense.reshape(-1, 3).round().clamp(0, SIZE - 1).long()
            line[b, 0, j[:, 0], j[:, 1], j[:, 2]] = 1.0
    line = F.max_pool3d(line, 3, 1, 1)
    return heat.view(B, 1, SIZE, SIZE, SIZE), offset, peak, line


def focal(logits, heat):
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float()
    loss = -(pos * (1 - p) ** 2 * p.log() + (1 - pos) * (1 - heat) ** 4 * p ** 2 * (1 - p).log())
    return loss.sum() / pos.sum().clamp_min(1)


def losses(out, heat, offset, peak, line):
    node = focal(out[:, :1].float(), heat)
    m = peak.unsqueeze(1).expand_as(offset)
    off = F.l1_loss(out[:, 1:4].float()[m], offset[m]) if m.any() else out.sum() * 0
    cl = F.binary_cross_entropy_with_logits(out[:, 4:5].float(), line, pos_weight=torch.tensor(5.0, device=out.device))
    return node, off, cl


def node_types(n, e):
    v = n.numpy() * SIZE
    deg = np.zeros(len(v), int)
    for a, b in e.numpy():
        deg[a] += 1
        deg[b] += 1
    face = ((v <= -0.49) | (v >= SIZE - 0.51)).any(1)
    return np.where(face, 'boundary', np.where(deg >= 3, 'junction', np.where(deg == 1, 'terminal', 'other')))


def match(pred_mm, gt_mm, tol):
    if len(pred_mm) == 0 or len(gt_mm) == 0:
        return np.zeros(len(gt_mm), bool), 0
    d = np.linalg.norm(pred_mm[:, None] - gt_mm[None], axis=-1)
    r, c = linear_sum_assignment(np.where(d <= tol, d, 1e6))
    ok = d[r, c] <= tol
    hit = np.zeros(len(gt_mm), bool)
    hit[c[ok]] = True
    return hit, int(ok.sum())


@torch.no_grad()
def predict(model, ds, device, batch=8, workers=8):
    """Per patch: candidate peaks (normalized coords, scores >= min threshold) and GT."""
    model.eval()
    rows = []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=collate)
    for images, nodes, edges, geometry in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(images.to(device))
        heat = out[:, :1].float().sigmoid()
        keep = (heat == F.max_pool3d(heat, 3, 1, 1)) & (heat >= THRESHOLDS[0])
        for b in range(len(images)):
            idx = keep[b, 0].nonzero()
            score = heat[b, 0][keep[b, 0]]
            off = out[b, 1:4].float()[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
            coords = ((idx.float() + off) / SIZE).cpu().numpy()
            metric = geometry[b]['metric'].numpy()
            rows.append(dict(pred=coords @ metric.T, score=score.cpu().numpy(),
                             gt=nodes[b].numpy() @ metric.T, types=node_types(nodes[b], edges[b])))
    model.train()
    return rows


def score(rows, threshold, tol):
    tp = n_pred = n_gt = 0
    by_type = {}
    for r in rows:
        p = r['pred'][r['score'] >= threshold]
        hit, k = match(p, r['gt'], tol)
        tp += k
        n_pred += len(p)
        n_gt += len(r['gt'])
        for t in ('boundary', 'junction', 'terminal'):
            s = r['types'] == t
            a, b = by_type.get(t, (0, 0))
            by_type[t] = (a + int(hit[s].sum()), b + int(s.sum()))
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(threshold=threshold, tolerance_mm=tol, precision=precision, recall=recall, f1=f1,
                predicted_per_patch=n_pred / len(rows), gt_per_patch=n_gt / len(rows),
                recall_by_type={t: (a / b if b else None) for t, (a, b) in by_type.items()},
                count_by_type={t: b for t, (a, b) in by_type.items()})


def summarize(train_rows, val_rows):
    result = {}
    for tol in (1.0, 2.0):
        sweep = [score(train_rows, t, tol) for t in THRESHOLDS]
        best = max(sweep, key=lambda s: s['f1'])
        result[f'{tol:g}mm'] = dict(
            train_selected_threshold=best['threshold'], train=best,
            validation=score(val_rows, best['threshold'], tol),
            validation_at_0p5=score(val_rows, 0.5, tol),
            validation_raw_coverage_at_0p05=score(val_rows, 0.05, tol)['recall'])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=sorted(json.loads(PLAN.read_text())['datasets']), required=True)  # any plan entry
    p.add_argument('--updates', type=int, default=10000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--eval-every', type=int, default=2500)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--eval-batch', type=int, default=8)
    a = p.parse_args()
    torch.manual_seed(364505)
    np.random.seed(364505)
    from monai.networks.nets import BasicUNet
    out_dir = a.output / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text())['datasets'][a.dataset]
    train_ids, val_ids, monitor = plan['train_ids'], plan['val_ids'], plan['monitor_train']
    if a.smoke:
        train_ids, val_ids, monitor = train_ids[:8], val_ids[:8], train_ids[:8]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = make_dataset(a.dataset, train_ids, augment=True)
    monitor_ds = make_dataset(a.dataset, monitor, augment=False)
    val_ds = make_dataset(a.dataset, val_ids, augment=False)
    model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5,
                      features=(32, 32, 64, 128, 256, 32)).to(device)
    with torch.no_grad():  # CenterNet prior so initial heat is ~0.01
        model.final_conv.bias[0] = -4.6
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.updates, pct_start=0.05)
    sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=a.updates * a.batch)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                         collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
    log = (out_dir / 'training.jsonl').open('a')
    start = time.time()
    step = 0
    curve = []
    for images, nodes, edges, geometry in loader:
        heat, offset, peak, line = targets(nodes, geometry, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            out = model(images.to(device))
        node, off, cl = losses(out, heat, offset, peak, line)
        loss = node + off + cl
        assert torch.isfinite(loss), 'nonfinite loss'
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0 or a.smoke:
            row = dict(step=step, seconds=time.time() - start, node=float(node), offset=float(off), centerline=float(cl))
            log.write(json.dumps(row) + '\n')
            log.flush()
            print('STEP ' + json.dumps(row), flush=True)
        if step % a.eval_every == 0 or step == a.updates:
            final = step == a.updates
            val_rows = predict(model, val_ds if final else torch.utils.data.Subset(val_ds, range(min(128, len(val_ds)))), device, batch=a.eval_batch, workers=a.workers)
            summary = summarize(predict(model, monitor_ds, device, batch=a.eval_batch, workers=a.workers), val_rows)
            summary.update(step=step, seconds=time.time() - start,
                           validation_population='full' if final else 'first 128 validation patches')
            curve.append(summary)
            (out_dir / 'learning-curve.json').write_text(json.dumps(curve, indent=1))
            print('EVAL ' + json.dumps({k: (v['train']['f1'], v['validation']['f1']) for k, v in summary.items() if k.endswith('mm')}), flush=True)
            torch.save(dict(model=model.state_dict(), step=step, args=vars(a)), out_dir / 'model.pt')
    (out_dir / 'final.json').write_text(json.dumps(dict(
        dataset=a.dataset, updates=a.updates, batch=a.batch, lr=a.lr, sigma_voxels=SIGMA,
        train_patches=len(train_ids), validation_patches=len(val_ids), monitor_patches=len(monitor),
        threshold_policy='train-monitor F1-optimal per tolerance; validation never used for selection',
        result=curve[-1]), indent=1))


if __name__ == '__main__':
    main()
