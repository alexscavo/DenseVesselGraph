#!/usr/bin/env python3
"""Figure for the size-matched backbone comparison (9 Nov 2026), from experiments/backbone_compare_20261109/*.json.

Rows: datasets; columns: node / edge / Branch / Curve F1 and |beta1| error at the frozen global decoding setting
(V0: node 0.30, edge 0.75). Points: arm minus U-Net with subject-paired 95% bootstrap CIs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'experiments/backbone_compare_20261109'
OUT = ROOT / 'docs/research/figures/backbones_20261109/backbone_size_matched.png'
DATASETS = [('IXI-loops', 'IXI'), ('BRAVA', 'BRAVA'), ('TopBrainCTA-loops', 'TopBrain CTA')]
METRICS = [('node_f1', 'Node F1'), ('edge_f1', 'Edge F1'), ('branch_f1', 'Branch F1'), ('curve_f1', 'Curve F1'), ('mae_beta1', '|β1| error (lower better)')]
ARMS = [('unet_wide', 'U-Net wide (15.6M)', '#2a78d6', 'o'), ('dsc3_slim', 'DSC3 slim (5.8M)', '#eb6834', 's'), ('dsc3', 'DSC3 (15.9M)', '#1baf7a', 'D')]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

fig, axes = plt.subplots(len(DATASETS), len(METRICS), figsize=(16, 7.2), facecolor='white')
for r, (key, name) in enumerate(DATASETS):
    res = json.loads((SRC / f'{key}.json').read_text())
    base = res['arms']['unet']['V0_global']
    for c, (m, title) in enumerate(METRICS):
        ax = axes[r, c]
        for i, (arm, label, color, marker) in enumerate(ARMS):
            d = res['minus_unet'][f'{arm} | V0_global'][m]
            y = len(ARMS) - 1 - i
            ax.plot(d['ci95'], [y, y], color=color, lw=2, solid_capstyle='round')
            ax.scatter([d['point']], [y], color=color, marker=marker, s=40, zorder=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels([a[1] for a in ARMS[::-1]] if c == 0 else [], fontsize=8.5, color=INK)
        ax.set_ylim(-0.6, len(ARMS) - 0.4)
        ax.tick_params(axis='x', labelsize=8, colors=MUTED, length=0)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        if r == 0:
            ax.set_title(title, fontsize=10, color=INK)
        ax.text(0.02, 0.96, f"U-Net {base[m]:.3f}" if m != 'mae_beta1' else f"U-Net {base[m]:.2f}", transform=ax.transAxes,
                fontsize=7.5, color=MUTED, va='top', bbox=dict(fc='white', ec='none', pad=1.5), zorder=5)
        if c == 0:
            ax.text(-0.75, 1.08, name, transform=ax.transAxes, fontsize=11, fontweight='bold', color=INK)
        if r == len(DATASETS) - 1:
            ax.set_xlabel('difference to U-Net (95% CI)', fontsize=8.5, color=MUTED)
fig.suptitle('Size-matched backbones vs U-Net (5.75M), frozen global decoding (node 0.30, edge 0.75), validation, single seed',
             fontsize=12, color=INK, x=0.08, ha='left')
fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.09, hspace=0.55, wspace=0.12)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(OUT)
