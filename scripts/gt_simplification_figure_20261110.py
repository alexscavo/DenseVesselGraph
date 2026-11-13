#!/usr/bin/env python3
"""Figure: original vs simplified GT graph on validation patches (10 Nov 2026, CPU, no pyvista).

One row per dataset, a random validation patch (seed 0, among patches with at least --min-nodes GT nodes; not
hand-picked). Left: original GT; nodes that the simplification removes or merges are orange. Right: simplified GT.
Background: maximum-intensity projection of the image patch along the first voxel axis. Colours are slots 1-2 of
the dataviz reference palette (blue / orange; identity is also in the legend and the marker shape).

usage: gt_simplification_figure_20261110.py <name>=<original root>=<simplified root> [...] [--rows 2]
Writes docs/research/figures/gt_simplification_20261110/gt_simplification.png
"""
import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, MUTED, SURFACE = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#fcfcfb'


def load(root, name):
    with np.load(root / 'val' / 'spline' / f'{name}.npz') as z:
        ins = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(z['nodes']), bool)
        return dict(nodes=z['nodes'] * 64, paths=[z['full_paths'][a:b] * 64 for a, b in zip(z['path_offsets'][:-1], z['path_offsets'][1:])],
                    inserted=ins, metric=z['metric'])


def draw(ax, image, g, removed=None):
    ax.imshow(image.max(0).T, cmap='gray', origin='lower', extent=(-0.5, 63.5, -0.5, 63.5), vmin=0, vmax=np.percentile(image, 99.8))
    for p in g['paths']:
        ax.plot(p[:, 1], p[:, 2], color=BLUE, lw=1.6, solid_capstyle='round', zorder=2)
    keep = ~g['inserted'] if removed is None else ~g['inserted'] & ~removed
    n = g['nodes']
    ax.scatter(n[keep, 1], n[keep, 2], s=34, color=BLUE, edgecolor='white', linewidth=1.0, zorder=3)
    if removed is not None and removed.any():
        r = removed & ~g['inserted']
        ax.scatter(n[r, 1], n[r, 2], s=40, marker='D', color=ORANGE, edgecolor='white', linewidth=1.0, zorder=4)
    ax.set_xlim(-0.5, 63.5), ax.set_ylim(-0.5, 63.5)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('datasets', nargs='+')
    p.add_argument('--rows', type=int, default=1, help='patches per dataset')
    p.add_argument('--min-nodes', type=int, default=14)
    a = p.parse_args()
    specs = [d.split('=') for d in a.datasets]
    n_rows = len(specs) * a.rows
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.2, 3.6 * n_rows), facecolor=SURFACE, squeeze=False)
    row = 0
    for name, orig, simp in specs:
        orig, simp = Path(orig), Path(simp)
        names = sorted(f.stem for f in (simp / 'val' / 'spline').glob('*.npz'))
        random.Random(0).shuffle(names)
        done = 0
        for sample in names:
            g0 = load(orig, sample)
            if (~g0['inserted']).sum() < a.min_nodes:
                continue
            g1 = load(simp, sample)
            image = nib.load(str(orig / 'val' / 'raw' / f'{sample}_data.nii.gz')).get_fdata()
            mm0, mm1 = g0['nodes'] / 64 @ g0['metric'].T, g1['nodes'] / 64 @ g1['metric'].T
            d = np.linalg.norm(mm0[:, None] - mm1[None], axis=-1).min(1) if len(mm1) else np.full(len(mm0), np.inf)
            removed = d > 1e-3
            draw(axes[row, 0], image, g0, removed)
            draw(axes[row, 1], image, g1)
            k0, k1 = int((~g0['inserted']).sum()), int((~g1['inserted']).sum())
            axes[row, 0].set_title(f'{name}, original: {k0} nodes, {int((removed & ~g0["inserted"]).sum())} removed or merged', fontsize=8, color=INK, loc='left')
            axes[row, 1].set_title(f'simplified at 2 mm: {k1} nodes', fontsize=8, color=INK, loc='left')
            axes[row, 0].set_ylabel(sample, fontsize=7, color=MUTED)
            row += 1
            done += 1
            if done == a.rows:
                break
    handles = [Line2D([], [], marker='o', ls='', color=BLUE, markeredgecolor='white', markersize=7, label='node kept'),
               Line2D([], [], marker='D', ls='', color=ORANGE, markeredgecolor='white', markersize=7, label='node removed or merged (edge < 2 mm)'),
               Line2D([], [], color=BLUE, lw=1.6, label='GT centreline')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / 'docs/research/figures/gt_simplification_20261110'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'gt_simplification.png', dpi=150, facecolor=SURFACE)
    print(out / 'gt_simplification.png')


if __name__ == '__main__':
    main()
