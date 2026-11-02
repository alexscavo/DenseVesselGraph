#!/usr/bin/env python3
"""Stage-1 U-Net outputs of the traced dense model on one representative validation patch per dataset.

Columns: (1) image MIP + GT graph, (2) node heatmap MIP, (3) detected nodes = heatmap peaks
above the train-selected threshold, refined by the offsets, coloured matched / spurious against
GT within 2 mm, (4) zoom on one node: peak voxel, offset arrow, refined node, GT node,
(5) centreline-probability MIP with the 0.5 tracing contour and the GT centreline.
MIP = maximum-intensity projection along the first axis (D); in-plane axes are (H, W).
Patch choice follows the 31 Oct figure: 10-40 GT edges and edge F1 closest to the dataset value.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from node_heatmap_probe_20261030 import make_dataset, SIZE  # noqa: E402
from diagnose_branch_relations import match_nodes  # noqa: E402
from ixi_query_diagnostics import edge_accounting  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402

RUN = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
OUT = REPO / 'docs/research/figures/unet_outputs_20261102'
GREEN, ORANGE, RED = '#2ecc71', '#ff9f1c', '#e74c3c'


def representative(rows, nt, et, pooled):
    rng = np.random.default_rng(0)
    best = None
    for i in rng.permutation(len(rows))[:300]:
        r = rows[i]
        with np.load(r['cache']) as z:
            gn, ge, M = z['nodes'], z['edges'].reshape(-1, 2), z['metric']
        if not 10 <= len(ge) <= 40 or r['edge_scores'] is None:
            continue
        keep = np.nonzero(r['node_scores'] >= nt)[0]
        remap = {int(k): j for j, k in enumerate(keep)}
        valid = np.isin(r['pairs'], keep).all(1)
        sub, sc = r['pairs'][valid], r['edge_scores'][valid]
        local = np.array([[remap[a], remap[b]] for a, b in sub], int).reshape(-1, 2)
        mapping = match_nodes(r['nodes'][keep] @ M.T, gn @ M.T, 2.0)
        row, *_ = edge_accounting(np.arange(len(keep)), local, sc, sc >= et, mapping, ge, len(gn))
        f1 = 2 * row['tp'] / max(1, 2 * row['tp'] + row['fp'] + row['fn'])
        if best is None or abs(f1 - pooled) < best[0]:
            best = (abs(f1 - pooled), r['sample_id'])
    return best[1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['SyntheticMRI', 'IXI', 'BRAVA']
    fig, axes = plt.subplots(len(datasets), 5, figsize=(24, 5.2 * len(datasets)))
    for ri, d in enumerate(datasets):
        res = json.loads((RUN / d / 'traced_adaptive/final.json').read_text())
        nt, et = res['thresholds']['node'], res['thresholds']['edge']
        rows = pickle.loads((RUN / d / 'traced_adaptive/final-validation-rows.pkl').read_bytes())
        sid = representative(rows, nt, et, res['validation']['edge_f1'])
        ds = make_dataset(d, [sid], augment=False)
        sample = ds[0]
        img = sample[0][0].float()  # (1, 64, 64, 64)
        gt_nodes = sample[2][0][:, :3].numpy() * SIZE
        metric = sample[4][0]['metric'].numpy() / SIZE  # voxel -> mm
        guides = sample[4][0]['guides'].numpy() * SIZE
        model = T.TracedDenseGraph().to(device).eval()
        model.load_state_dict(torch.load(RUN / d / 'traced_adaptive/model.pt', map_location=device)['model'])
        with torch.no_grad():
            out = model.forward_image(img[None].to(device)).float()
        heat = out[0, 0].sigmoid()
        line = out[0, 4].sigmoid().cpu().numpy()
        keep = (heat == F.max_pool3d(heat[None, None], 3, 1, 1)[0, 0]) & (heat >= nt)
        idx = keep.nonzero()
        off = out[0, 1:4][:, idx[:, 0], idx[:, 1], idx[:, 2]].T
        voxel = idx.float().cpu().numpy()
        refined = (idx.float() + off).clamp(-0.5, SIZE - 0.5).cpu().numpy()
        heat = heat.cpu().numpy()
        mapping = match_nodes(refined @ metric.T, gt_nodes @ metric.T, 2.0)
        matched = np.array([i in mapping for i in range(len(refined))], bool)
        missed = np.array([j not in set(mapping.values()) for j in range(len(gt_nodes))], bool)

        im = img[0].numpy().max(0)
        fg = im[im > im.min()]
        lo, hi = np.percentile(fg, [40, 99.7]) if len(fg) else (im.min(), im.max())
        ax = axes[ri]
        # (1) image + GT graph
        ax[0].imshow(im, cmap='gray', vmin=lo, vmax=hi)
        for g in guides:
            ax[0].plot(g[:, 2], g[:, 1], '-', color=GREEN, lw=1.4)
        ax[0].plot(gt_nodes[:, 2], gt_nodes[:, 1], 'o', ms=4, color=GREEN, mec='k', mew=.4)
        ax[0].set_title(f'{d}: image (MIP) + GT graph\n{len(gt_nodes)} nodes, {len(guides)} edges', fontsize=10)
        # (2) heatmap
        h = ax[1].imshow(heat.max(0), cmap='magma', vmin=0, vmax=1)
        ax[1].plot(gt_nodes[:, 2], gt_nodes[:, 1], 'o', ms=6, mfc='none', mec=GREEN, mew=1.2)
        ax[1].set_title('Channel 0: node heatmap (MIP)\nGT nodes: green circles', fontsize=10)
        fig.colorbar(h, ax=ax[1], fraction=0.046, pad=0.02)
        # (3) detected nodes
        ax[2].imshow(im, cmap='gray', vmin=lo, vmax=hi)
        ax[2].plot(gt_nodes[missed, 2], gt_nodes[missed, 1], 'x', ms=7, color=GREEN, mew=1.6, label=f'missed GT node ({missed.sum()})')
        ax[2].plot(gt_nodes[~missed, 2], gt_nodes[~missed, 1], 'o', ms=8, mfc='none', mec=GREEN, mew=1.2, label=f'found GT node ({(~missed).sum()})')
        ax[2].plot(refined[matched, 2], refined[matched, 1], 'o', ms=4, color=ORANGE, mec='k', mew=.4, label=f'detected, matched ({matched.sum()})')
        ax[2].plot(refined[~matched, 2], refined[~matched, 1], 's', ms=4, color=RED, mec='k', mew=.4, label=f'detected, spurious ({(~matched).sum()})')
        ax[2].legend(loc='lower left', fontsize=7, framealpha=.85)
        ax[2].set_title(f'Nodes = heatmap peaks ≥ {nt:.2f} + offsets\n(2 mm matching to GT)', fontsize=10)
        # (4) zoom on one matched node: peak voxel, offset, refined position, GT
        if matched.any():
            k = int(np.nonzero(matched)[0][0])
            v, p, g = voxel[k], refined[k], gt_nodes[mapping[k]]
            r0, c0 = int(round(v[1])), int(round(v[2]))
            win = 4
            sl = heat[int(v[0])]  # the slice through the peak voxel, not a projection
            ax[3].imshow(sl, cmap='magma', vmin=0, vmax=1, extent=(-0.5, SIZE - 0.5, SIZE - 0.5, -0.5))
            for t in np.arange(-0.5, SIZE, 1):
                ax[3].axhline(t, color='w', lw=.3, alpha=.4)
                ax[3].axvline(t, color='w', lw=.3, alpha=.4)
            ax[3].plot(v[2], v[1], 's', ms=10, mfc='none', mec='w', mew=1.5, label='peak voxel (centre)')
            ax[3].annotate('', xy=(p[2], p[1]), xytext=(v[2], v[1]), arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
            ax[3].plot(p[2], p[1], 'o', ms=7, color=ORANGE, mec='k', label='refined = voxel + offset')
            ax[3].plot(g[2], g[1], 'P', ms=9, color=GREEN, mec='k', label='GT node')
            ax[3].set_xlim(c0 - win - .5, c0 + win + .5)
            ax[3].set_ylim(r0 + win + .5, r0 - win - .5)
            err_vox = np.linalg.norm(v - g) * metric[0, 0] if False else None
            d_peak = np.linalg.norm((v - g) @ metric.T)
            d_ref = np.linalg.norm((p - g) @ metric.T)
            ax[3].legend(loc='lower left', fontsize=7, framealpha=.85)
            ax[3].set_title(f'Zoom on one node (heatmap slice, 1 square = 1 voxel)\n'
                            f'offset ({off[k, 1].item():+.2f}, {off[k, 2].item():+.2f}) vox in-plane, '
                            f'{off[k, 0].item():+.2f} depth; error {d_peak:.2f} → {d_ref:.2f} mm', fontsize=9)
        # (5) centreline map
        c = ax[4].imshow(line.max(0), cmap='viridis', vmin=0, vmax=1)
        ax[4].contour(line.max(0), levels=[0.5], colors='w', linewidths=.8)
        for g in guides:
            ax[4].plot(g[:, 2], g[:, 1], '--', color=GREEN, lw=1)
        ax[4].set_title('Channel 4: centreline probability (MIP)\nwhite: 0.5 tracing contour; green dashed: GT centreline', fontsize=10)
        fig.colorbar(c, ax=ax[4], fraction=0.046, pad=0.02)
        for a in ax[[0, 1, 2, 4]]:
            a.set_xlim(-0.5, SIZE - 0.5)
            a.set_ylim(SIZE - 0.5, -0.5)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        print(d, sid, 'nodes', len(refined), 'matched', int(matched.sum()), 'gt', len(gt_nodes), flush=True)
    fig.suptitle('Stage-1 U-Net outputs of the traced dense model on validation patches (64³)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'unet_outputs_3d.png', dpi=110)
    print('saved', OUT / 'unet_outputs_3d.png')


if __name__ == '__main__':
    main()
