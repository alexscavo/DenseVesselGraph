#!/usr/bin/env python3
"""Backbone comparison figure: U-Net vs U-KAN (concat, add) vs DSCNet (4 Nov 2026).

Top: subject-paired difference to the U-Net (95% bootstrap CI over validation subjects) of the final
two-stage model (traced_adaptive recipe), at shared thresholds where the arms selected different ones
(shared-thresholds.json; hollow marker = approximate, the arm's node threshold had to be raised), else
at the common own thresholds (paired-*.json). Bottom: absolute validation values at each arm's own
train-monitor thresholds. CPU, saved results only; no selection on validation.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'experiments/dense_graph_20261030'
SCRATCH = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/dense_graph_20261030')
OUT = ROOT / 'docs/research/figures/backbones_20261104/backbone_comparison.png'
DATASETS = ('SyntheticMRI', 'IXI', 'BRAVA')
METRICS = (('node_f1', 'Node F1'), ('edge_f1', 'Edge F1'), ('curve_f1', 'Curve F1'), ('branch_f1', 'Branch F1'))
ARMS = (('ukan', 'U-KAN (concat)', '#2a78d6', 'o', '5.82M'),
        ('ukan_add', 'U-KAN (add)', '#eb6834', 's', '6.34M'),
        ('dsc3', 'DSCNet (K=3)', '#1baf7a', 'D', '15.9M'))
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def difference(dataset, arm):
    key = f'traced_adaptive_{arm}_minus_traced_adaptive'
    shared = json.loads((RES / 'shared-thresholds.json').read_text())[dataset][key]
    if shared.get('same_thresholds'):
        own = json.loads((RES / f'paired-traced_adaptive_{arm}-vs-traced_adaptive-{dataset}.json').read_text())
        return own[key], True
    return shared['difference'], bool(shared['exact'])


def main():
    fig = plt.figure(figsize=(13, 9.2), facecolor='white')
    grid = fig.add_gridspec(4, 4, height_ratios=(1, 1, 1, 1.15), hspace=0.55, wspace=0.12)
    for r, dataset in enumerate(DATASETS):
        for c, (metric, title) in enumerate(METRICS):
            ax = fig.add_subplot(grid[r, c])
            for i, (arm, name, color, marker, _) in enumerate(ARMS):
                diff, exact = difference(dataset, arm)
                d = diff[metric]
                y = len(ARMS) - 1 - i
                lo, hi = d['ci95']
                ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle='round', zorder=2)
                ax.scatter([d['point']], [y], s=46, marker=marker, zorder=3, linewidths=1.6,
                           facecolors=color if exact else 'white', edgecolors=color)
            ax.axvline(0, color=MUTED, lw=1, zorder=1)
            ax.set_xlim(-0.06, 0.06)
            ax.set_xticks([-0.04, -0.02, 0, 0.02, 0.04])
            ax.set_xticklabels(['-0.04', '', '0', '', '+0.04'])
            ax.set_ylim(-0.6, len(ARMS) - 0.4)
            ax.set_yticks(range(len(ARMS)))
            ax.set_yticklabels([a[1] for a in ARMS[::-1]] if c == 0 else [], fontsize=8.5, color=INK)
            ax.tick_params(axis='x', labelsize=8, colors=MUTED, length=0)
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='x', color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(GRID)
            if r == 0:
                ax.set_title(title, fontsize=10.5, color=INK, pad=6)
            if c == 0:
                ax.text(-0.62, 1.12, dataset, transform=ax.transAxes, fontsize=11, fontweight='bold', color=INK)
            if r == len(DATASETS) - 1:
                ax.set_xlabel('difference to U-Net (95% CI)', fontsize=8.5, color=MUTED)

    # absolute values at own thresholds
    ax = fig.add_subplot(grid[3, :])
    ax.axis('off')
    header = ['Backbone', 'Params'] + [f'{d}\nNode / Edge / Curve / Branch' for d in DATASETS]
    rows = []
    for arm, name, _, _, params in (('', 'U-Net (baseline)', None, None, '5.75M'),) + ARMS:
        row = [name, params]
        for dataset in DATASETS:
            run = 'traced_adaptive' + ('_' + arm if arm else '')
            v = json.loads((SCRATCH / dataset / run / 'final.json').read_text())['validation']
            row.append(' / '.join(f'{v[m]:.3f}' for m in ('node_f1', 'edge_f1', 'common_curve_f1', 'common_branch_f1')))
            row[-1] += f'   (nt {v["node_threshold"]:.2f})'
        rows.append(row)
    table = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='center',
                     colWidths=(0.13, 0.07, 0.26, 0.26, 0.26))
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0:
            cell.set_text_props(color=INK, fontweight='bold')
            cell.set_height(cell.get_height() * 1.5)
        else:
            cell.set_text_props(color=INK)
    ax.set_title('Absolute validation values, each arm at its own train-monitor thresholds (nt = node threshold)',
                 fontsize=9.5, color=MUTED, pad=-8, y=1.16)

    fig.suptitle('Backbone comparison, final two-stage model (traced + adaptive-K), validation',
                 fontsize=13, color=INK, x=0.07, ha='left', y=0.985)
    fig.text(0.07, 0.948,
             'Subject-paired differences to the U-Net at shared thresholds. Filled marker: exact. Hollow: approximate '
             '(arm node threshold raised; tracing not repeated).\nSingle seed per arm; same recipe, from scratch. '
             'DSCNet has about 3x the U-Net parameters (no same-size control yet). DSCNet K=9 still training.',
             fontsize=8.5, color=MUTED, ha='left', va='top')
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.03)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(OUT)


if __name__ == '__main__':
    main()
