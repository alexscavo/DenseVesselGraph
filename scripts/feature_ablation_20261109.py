#!/usr/bin/env python3
"""Test-time ablation of the 32 shared feature channels F (U-Net outputs 5..36) of a trained traced dense model (9 Nov 2026).

F feeds only stage 2 (path corridor and endpoint features). Heatmap, offsets, centreline and tracing do not read F, so
every variant keeps the same nodes, candidates and paths; only edge scores and curves change (checked: node counts are
asserted identical across variants).
F = final_conv(u1) is a 1x1x1 linear map of the last decoder output u1 (fea[5] = 32 channels for the U-Net), so a native
channel is an arbitrary basis direction. The variants therefore probe native channels AND principal directions.

Variants (all statistics and orderings from the TRAINING monitor; validation is only scored):
  none        the model as trained
  chan<c>     channel c replaced by its training mean (mean ablation: zeroing would be off-distribution)
  all_mean    every channel replaced by its mean: stage 2 sees only chord/arc geometry and the endpoint heat
  pca<k>      F projected onto its top-k principal directions (covariance of the corridor samples stage 2 reads),
              remaining directions set to the mean
  cum<k>      the k least important native channels mean-ablated together; order = single-channel edge-AP drop on
              the training monitor
Each variant is F' = A F + (I - A) mu, applied to the whole feature volume before stage 2.

Scores on validation, per variant:
  - edge ranking AP over candidate pairs among nodes >= 0.30 (threshold-free; all retained / both endpoints matched);
  - loop-aware node/edge/Branch/Curve F1 and |beta0|, |beta1| errors at the protocol's frozen global setting
    (V0: node 0.30, edge 0.75), with subject-paired bootstrap differences vs 'none' (protocol_report_20261106.paired);
  - edge F1 with the edge threshold re-selected on the training monitor for that variant (removes calibration shift).
Writes experiments/feature_ablation_20261109/<dataset>[_<run>].json.
usage: feature_ablation_20261109.py <dataset> [--run name] [--smoke]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402
from compare_dense_anchored_20261030 import KEYS, dense_patch  # noqa: E402
from decode_study_20261105 import as_row, decode, gt  # noqa: E402
from dense_ap_20261030 import ranking_ap  # noqa: E402
from diagnose_branch_relations import match_nodes  # noqa: E402
from ixi_query_diagnostics import betti, edge_accounting  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402
from protocol_report_20261106 import paired, pooled  # noqa: E402

RUNS = {'IXI-loops': 'traced_adaptive', 'TopBrainCTA-loops': 'traced_adaptive', 'TopBrainMRA-loops': 'traced_adaptive',
        'SyntheticMRI-loops': 'traced_adaptive', 'BRAVA': 'traced_offset'}  # the evaluation protocol's reference runs
NODE_T, EDGE_T = 0.30, 0.75
EDGE_GRID = [round(x, 2) for x in np.arange(0.5, 0.91, 0.05)]
PCA_K = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28]
CUM_K = [4, 8, 12, 16, 20, 24, 28]
FC = T.FEATURE_CHANNELS
ROWS = {}   # variant -> rows of the current chunk (read by forked workers)
STATS = {}  # corridor-sample moments collected by the hook


def corridor_hook(_module, inputs):
    if STATS.get('on'):
        x = inputs[0].detach().double().transpose(0, 1).reshape(FC, -1)  # (32, samples)
        STATS['n'] = STATS.get('n', 0) + x.shape[1]
        STATS['s'] = STATS.get('s', 0) + x.sum(1)
        STATS['ss'] = STATS.get('ss', 0) + x @ x.T


def affine(A, mu):
    """F' = A F + (I - A) mu as (A, bias) float32 tensors on mu's device."""
    eye = torch.eye(FC, dtype=torch.float64, device=mu.device)
    return A.float(), ((eye - A) @ mu).float()


def basic_variants(mu, cov):
    eye = torch.eye(FC, dtype=torch.float64, device=mu.device)
    v = {'none': None, 'all_mean': affine(eye * 0, mu)}
    for c in range(FC):
        A = eye.clone()
        A[c, c] = 0
        v[f'chan{c}'] = affine(A, mu)
    evals, evecs = torch.linalg.eigh(cov)
    U = evecs[:, evals.argsort(descending=True)]
    for k in PCA_K:
        v[f'pca{k}'] = affine(U[:, :k] @ U[:, :k].T, mu)
    return v


def cumulative_variants(mu, order):
    eye = torch.eye(FC, dtype=torch.float64, device=mu.device)
    v = {}
    for k in CUM_K:
        A = eye.clone()
        for c in order[:k]:
            A[c, c] = 0
        v[f'cum{k}'] = affine(A, mu)
    return v


@torch.no_grad()
def predict_variants(model, ds, variants, device, workers, keep_curves, batch=8):
    """T.predict at NODE_T, with stage 2 repeated for every feature variant on the same nodes, candidates and paths."""
    model.eval()
    rows = {v: [] for v in variants}
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=D.collate)
    offset = 0
    for images, nodes, edges, geometry in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model.forward_image(images.to(device))
        feats, heat, prob = out[:, 5:].float(), out[:, :1].float().sigmoid(), out[:, 4].float().sigmoid()
        found = D.peaks(out, NODE_T)
        contacts, dist, label = T.traced_pairs(prob, [p for p, _ in found])
        for b, (p, score) in enumerate(found):
            metric = geometry[b]['metric'].to(device) / D.SIZE
            record = ds.records[offset + b]
            base = dict(sample_id=record.sample_id, cache=str(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz'),
                        nodes=(p / D.SIZE).cpu().numpy(), node_scores=score.cpu().numpy(), pairs=np.zeros((0, 2), int),
                        edge_scores=None, curves=None, traced=None, interior=None)
            pairs = T.patch_candidates(p, metric, contacts[b]) if len(p) >= 2 else None
            if pairs is None or not len(pairs):
                for v in variants:
                    rows[v].append(dict(base))
                continue
            paths, traced = T.build_paths(p, pairs, contacts[b], dist[b], label[b])
            interior = T.choose_interior(T.to_mm(paths, metric[None].expand(len(paths), 3, 3)), traced, T.voxel_size(metric))
            base.update(pairs=pairs.cpu().numpy(), traced=traced.cpu().numpy(), interior=interior.cpu().numpy())
            for v, tf in variants.items():
                f = feats[b] if tf is None else torch.einsum('ij,jxyz->ixyz', tf[0], feats[b]) + tf[1][:, None, None, None]
                x = model.path_features(f, heat[b], paths, prob[b], None)
                row = dict(base, edge_scores=model.edge_head(x).squeeze(-1).sigmoid().cpu().numpy())
                if keep_curves:
                    row['curves'] = (model.path_curves(x, paths, interior, traced) / D.SIZE).cpu().numpy()
                rows[v].append(row)
        offset += len(images)
    return rows


def ap_parts(r):
    nodes_mm, gt_edges, _, _, m = gt(r['cache'])
    keep = np.flatnonzero(r['node_scores'] >= NODE_T)
    if r['edge_scores'] is None or len(keep) < 2:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0, bool)
    remap = {int(k): i for i, k in enumerate(keep)}
    valid = np.isin(r['pairs'], keep).all(1)
    sub, scores = r['pairs'][valid], r['edge_scores'][valid]
    local = np.array([[remap[int(a)], remap[int(b)]] for a, b in sub], int).reshape(-1, 2)
    mapping = match_nodes(r['nodes'][keep] @ m.T, nodes_mm, 2.0)
    _, _, labels, _, _ = edge_accounting(np.arange(len(keep)), local, scores, np.zeros(len(scores), bool), mapping, gt_edges, len(nodes_mm))
    cond = np.array([a in mapping and b in mapping for a, b in local], bool).reshape(-1)
    return scores, np.asarray(labels, bool), cond


def edge_counts(r, t_edge):
    nodes_mm, gt_edges, inserted, _, m = gt(r['cache'])
    nodes, pairs = decode(r, t_high=NODE_T, t_edge=t_edge)
    pred_mm = r['nodes'][nodes] @ m.T
    mapping = match_nodes(pred_mm, nodes_mm, 2.0) if len(pred_mm) and len(nodes_mm) else {}
    remap = {int(k): i for i, k in enumerate(nodes)}
    local = np.array([[remap[int(a)], remap[int(b)]] for a, b in r['pairs'][pairs]], int).reshape(-1, 2)
    row, *_ = edge_accounting(np.arange(len(nodes)), local, np.ones(len(local)), np.ones(len(local), bool), mapping,
                              gt_edges, len(nodes_mm), inserted)
    return row['tp'], row['fp'], row['fn']


def monitor_task(args):
    """Edge counts at every grid threshold (no geometry) + AP parts, one patch of one variant."""
    v, i = args
    r = ROWS[v][i]
    return v, i, [edge_counts(r, t) for t in EDGE_GRID], ap_parts(r)


def val_task(args):
    """Protocol V0 counts with geometry (as backbone_compare_20261109.per_patch), AP parts, edge counts at t_sel."""
    v, i, t_sel = args
    r = ROWS[v][i]
    nodes, pairs = decode(r, t_high=NODE_T, t_edge=EDGE_T)
    dr = as_row(r, nodes, pairs)
    c = dense_patch(dr, 0.5, 0.5)
    _, _, _, (g0, g1), _ = gt(r['cache'])
    p0, p1 = betti(len(dr['nodes']), dr['pairs'])
    return v, i, [c[k] for k in KEYS] + [abs(p0 - g0), abs(p1 - g1)], ap_parts(r), edge_counts(r, t_sel), len(r['nodes'])


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def ap_summary(parts):
    s = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)
    lab = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, bool)
    cond = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, bool)
    return dict(edge_ap_all_retained=ranking_ap(s, lab)['ap'], edge_ap_conditional=ranking_ap(s[cond], lab[cond])['ap'])


def score_monitor(model, dataset, ids, variants, device, workers):
    ROWS.clear()
    ROWS.update(predict_variants(model, make_dataset(dataset, ids, augment=False), variants, device, workers, keep_curves=False))
    n = len(ids)
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(monitor_task, [(v, i) for v in variants for i in range(n)], chunksize=8)
    out = {}
    for v in variants:
        mine = [r for r in res if r[0] == v]
        counts = np.asarray([r[2] for r in mine], float).sum(0)  # (grid, 3)
        f1s = [f1(*c) for c in counts]
        best = int(np.argmax(f1s))
        out[v] = dict(edge_f1_by_threshold=dict(zip(map(str, EDGE_GRID), f1s)), selected_edge_threshold=EDGE_GRID[best],
                      edge_f1_at_0p75=f1s[EDGE_GRID.index(EDGE_T)], **ap_summary([r[3] for r in mine]))
    ROWS.clear()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('--run', default=None)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--chunk', type=int, default=400)
    a = p.parse_args()
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    run = a.run or RUNS[a.dataset]
    scratch = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030' / a.dataset / run
    final = json.loads((scratch / 'final.json').read_text())
    T.apply_saved_mode(final)
    assert not T.MODE['two_hop'] and not T.MODE['passthrough'], 'two-hop / pass-through runs are not supported here'
    backbone = final.get('backbone', 'unet')
    device = 'cuda'
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], backbone=backbone).to(device)
    model.load_state_dict(torch.load(scratch / 'model.pt', map_location=device)['model'])
    model.eval()
    model.corridor.register_forward_pre_hook(corridor_hook)
    plan = json.loads(PLAN.read_text())['datasets'][a.dataset]
    monitor, val_ids = plan['monitor_train'], plan['val_ids']
    if a.smoke:
        monitor, val_ids, a.chunk = monitor[:16], val_ids[:24], 12
    t0 = time.time()

    # 1. Corridor-sample statistics of F on the training monitor (the unablated model).
    STATS.clear()
    STATS['on'] = True
    predict_variants(model, make_dataset(a.dataset, monitor, augment=False), {'none': None}, device, workers, keep_curves=False)
    STATS['on'] = False
    mu = STATS['s'] / STATS['n']
    cov = STATS['ss'] / STATS['n'] - torch.outer(mu, mu)
    evals = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0).cpu().numpy()
    w = model.unet.final_conv.weight.detach()[5:5 + FC].flatten(1).double()  # (32, fea5)
    w_sv = torch.linalg.svdvals(w).cpu().numpy()
    corr_w = model.corridor[0].weight.detach().double()  # (64, 32, 3, 3, 3)
    std = cov.diagonal().clamp_min(0).sqrt()
    stats = dict(corridor_samples=int(STATS['n']), channel_mean=mu.cpu().tolist(), channel_std=std.cpu().tolist(),
                 pca_explained_variance_ratio=(evals / evals.sum()).tolist(),
                 pca_cumulative=np.cumsum(evals / evals.sum()).tolist(),
                 participation_ratio=float(evals.sum() ** 2 / (evals ** 2).sum()),
                 decoder_width_fea5=int(w.shape[1]), final_conv_feature_singular_values=w_sv.tolist(),
                 corridor_conv_weight_norm_x_std=(corr_w.transpose(0, 1).flatten(1).norm(dim=1) * std).cpu().tolist())
    print('STATS', json.dumps({k: stats[k] for k in ('participation_ratio', 'decoder_width_fea5')}),
          'pca_cum', np.round(stats['pca_cumulative'][:16], 3).tolist(), f'{time.time() - t0:.0f}s', flush=True)

    # 2. Monitor: per-variant edge threshold and single-channel importance -> cumulative order.
    variants = basic_variants(mu, cov)
    mon = score_monitor(model, a.dataset, monitor, variants, device, workers)
    ref = mon['none']['edge_ap_conditional']
    drop = {c: ref - mon[f'chan{c}']['edge_ap_conditional'] for c in range(FC)}
    order = sorted(range(FC), key=lambda c: drop[c])  # least important first
    cum = cumulative_variants(mu, order)
    mon.update(score_monitor(model, a.dataset, monitor, cum, device, workers))
    variants.update(cum)
    print('MONITOR', json.dumps({v: [round(mon[v]['edge_ap_conditional'] or 0, 4), mon[v]['selected_edge_threshold']]
                                 for v in variants}), f'{time.time() - t0:.0f}s', flush=True)

    # 3. Validation, in chunks (rows with curves for ~50 variants do not fit in memory at once).
    v0 = {v: [] for v in variants}
    parts = {v: [] for v in variants}
    sel = {v: [] for v in variants}
    ids_all, n_nodes = [], {v: [] for v in variants}
    for start in range(0, len(val_ids), a.chunk):
        ids = val_ids[start:start + a.chunk]
        ROWS.clear()
        ROWS.update(predict_variants(model, make_dataset(a.dataset, ids, augment=False), variants, device, workers, keep_curves=True))
        ids_all += [r['sample_id'] for r in ROWS['none']]
        tasks = [(v, i, mon[v]['selected_edge_threshold']) for v in variants for i in range(len(ids))]
        with mp.get_context('fork').Pool(workers) as pool:
            res = pool.map(val_task, tasks, chunksize=8)
        for v, i, c, ap, ec, nn_ in res:  # pool.map keeps task order: variant-major, then patch
            v0[v].append(c)
            parts[v].append(ap)
            sel[v].append(ec)
            n_nodes[v].append(nn_)
        print(f'VAL chunk {start}-{start + len(ids)} / {len(val_ids)}  {time.time() - t0:.0f}s', flush=True)
    ROWS.clear()
    assert all(n_nodes[v] == n_nodes['none'] for v in variants), 'variants changed the node set'
    base = np.asarray(v0['none'], float)
    validation = {}
    for v in variants:
        arr = np.asarray(v0[v], float)
        resel = arr.copy()
        resel[:, [KEYS.index('tp'), KEYS.index('fp'), KEYS.index('fn')]] = np.asarray(sel[v], float)
        entry = dict(V0_global=pooled(arr), **ap_summary(parts[v]), edge_threshold_monitor=mon[v]['selected_edge_threshold'],
                     edge_f1_monitor_threshold=pooled(resel)['edge_f1'])
        if v != 'none':
            entry['V0_minus_none'] = paired(arr, base, ids_all)['difference']
            bres = base.copy()
            bres[:, [KEYS.index('tp'), KEYS.index('fp'), KEYS.index('fn')]] = np.asarray(sel['none'], float)
            entry['edge_f1_monitor_threshold_minus_none'] = paired(resel, bres, ids_all)['difference']['edge_f1']
        validation[v] = entry
    out = dict(dataset=a.dataset, run=run, backbone=backbone, mode=dict(T.MODE), smoke=a.smoke, node_threshold=NODE_T,
               edge_threshold_global=EDGE_T, monitor_patches=len(monitor), validation_patches=len(ids_all),
               stats=stats, channel_importance_monitor_ap_drop={str(c): drop[c] for c in range(FC)}, cumulative_order=order,
               monitor=mon, validation=validation, seconds=time.time() - t0)
    dst = ROOT / 'experiments/feature_ablation_20261109' / ('smoke' if a.smoke else '')
    dst.mkdir(parents=True, exist_ok=True)
    name = a.dataset + ('' if a.run is None else f'_{a.run}')
    (dst / f'{name}.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    for v in ['none', 'all_mean'] + [f'pca{k}' for k in PCA_K] + [f'cum{k}' for k in CUM_K]:
        e = validation[v]
        d = e.get('V0_minus_none', {})
        print('RESULT', v, json.dumps(dict(ap=round(e['edge_ap_conditional'] or 0, 4), edge_f1=round(e['V0_global']['edge_f1'], 4),
                                           curve_f1=round(e['V0_global']['curve_f1'], 4), edge_f1_resel=round(e['edge_f1_monitor_threshold'], 4),
                                           d_edge=d.get('edge_f1'), d_curve=d.get('curve_f1'))), flush=True)
    print(f'DONE {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
