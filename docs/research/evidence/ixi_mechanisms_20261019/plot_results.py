"""Render the reviewed cache-derived tables; run from this evidence directory."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
rows = [r for r in csv.DictReader((root/'comparison.csv').open()) if float(r['tolerance_mm']) == 2.]
index = {(r['experiment'], r['policy']): r for r in rows}
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)


def bars(ax, experiment, policies, labels, columns):
    x = np.arange(len(policies)); width = .8/len(columns)
    for j, (column, label) in enumerate(columns):
        values = [float(index[experiment, p][column]) for p in policies]
        ax.bar(x+(j-(len(columns)-1)/2)*width, values, width, label=label)
    ax.set_xticks(x, labels); ax.set_ylim(0, .92 if experiment == 'A' else .65)
    ax.legend(frameon=False); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)


bars(axes[0, 0], 'A', ['normal', 'previous_oracle', 'raw_oracle'], ['Normal', 'Deletion oracle', 'All-query oracle'],
     [('node_recall', 'Node recall'), ('f1', 'Edge F1'), ('common_curve_f1', 'Curve F1')])
axes[0, 0].set_title('A: more recovered nodes, small edge gain')
bars(axes[0, 1], 'C', ['neither_local_residual', 'coordinates_only', 'classification_only', 'both'],
     ['Neither', 'Coordinates', 'Classification', 'Both'],
     [('node_recall', 'Node recall'), ('f1', 'Edge F1'), ('common_curve_f1', 'Curve F1')])
axes[0, 1].set_title('C: coordinate residual supplies most of the benefit')

arms = ['original', 'focal_control', 'regularized', 'hard_unmatched', 'hard_matched']
labels = ['Original', 'Focal', 'CE', 'CE + U hard', 'CE + M hard']
ranking = json.loads((root/'ranking-review.json').read_text())
for split, offset, color in [('train', -.18, '#9aa0a6'), ('val', .18, '#2864b4')]:
    values = [next(r['conditional_mm_ap'] for r in ranking if r['arm'] == a and r['split'] == split) for a in arms]
    axes[1, 0].bar(np.arange(5)+offset, values, .36, color=color, label=split)
axes[1, 0].set_xticks(np.arange(5), labels, rotation=10); axes[1, 0].set_ylim(0, .75)
axes[1, 0].set_title('D: training ranking gains fail to transfer')
axes[1, 0].set_ylabel('Conditional relation AP'); axes[1, 0].legend(frameon=False)
axes[1, 0].grid(axis='y', alpha=.2); axes[1, 0].set_axisbelow(True)

for a, label in zip(arms, labels):
    r = index['A', 'normal'] if a == 'original' else index['D_'+a, a]
    x, y = float(r['tp']), float(r['fp'])
    axes[1, 1].scatter(x, y, s=65)
    offset = (6, -18) if a == 'hard_unmatched' else (6, 13) if a == 'hard_matched' else (6, 7)
    axes[1, 1].annotate(label, (x, y), xytext=offset, textcoords='offset points')
axes[1, 1].set_xlim(80, 680); axes[1, 1].set_ylim(0, 4900)
axes[1, 1].set_xlabel('Correct predicted edges'); axes[1, 1].set_ylabel('False predicted edges')
axes[1, 1].set_title('D: recall gains also add many false connections')
axes[1, 1].grid(alpha=.2)
fig.suptitle('IXI: 128 validation patches, 26 subjects; positional matching at 2 mm\nCurve F1 retains the independent common-v2 geometry protocol', fontsize=13)
fig.savefig(root/'overview.png', dpi=170)
print(root/'overview.png')
