#!/usr/bin/env python3
"""Figures of the stitched full-volume graphs (12 Nov 2026, CPU, no pyvista).

Input: the dumps of stitch_volume_20261112.py --dump ($SCRATCH/experiments/gnbm/stitching_20261112/<entry>-<patient>.pkl).
Per subject, in docs/research/figures/stitching_20261112/:
  <entry>-<patient>.png   static: the GT volume graph (left; centreline found / missed by the stitched prediction) and
                          the stitched prediction (right; centreline on / off the GT), projected along the thinnest
                          axis over the maximum-intensity projection of the image; faint lines = the planes where
                          patch cores meet (the stitching planes); triangles = cut ends that found no partner.
  <entry>-<patient>.html  interactive 3D (plotly.js from cdnjs, data embedded): GT graph, stitched prediction, stitched
                          GT patches, unmatched cut ends and the core planes as legend toggles; a table of the scores.
Colours: slots 1-3 of the dataviz reference palette (blue, orange, aqua), light and dark steps.
usage: stitch_volume_figures_20261112.py <entry> [<entry> ...]
"""
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DUMP = Path(os.environ['SCRATCH']) / 'experiments/gnbm/stitching_20261112'
OUT = ROOT / 'docs/research/figures/stitching_20261112'
BLUE, ORANGE, AQUA, INK, MUTED, SURFACE = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#fcfcfb'
NAMES = {'IXI-simp2': 'IXI', 'BRAVA-simp2': 'BRAVA', 'TopBrainCTA-simp2': 'TopBrain CTA'}


def dense(p, step=0.5):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.r_[0., np.cumsum(seg)]
    if cum[-1] <= 0:
        return p[:1]
    t = np.linspace(0, cum[-1], max(2, int(np.ceil(cum[-1] / step)) + 1))
    return np.stack([np.interp(t, cum, p[:, k]) for k in range(3)], 1)


def segments(edges, other_tree, Minv, axes):
    """2D segments of the edges (voxel coordinates on `axes`) and a flag: within 1.5 mm of the other graph."""
    segs, flags = [], []
    for _, _, p in edges:
        q = dense(p)
        ok = other_tree.query(q)[0] <= 1.5 if other_tree is not None else np.zeros(len(q), bool)
        v = (q @ Minv.T)[:, axes]
        segs += [v[i:i + 2] for i in range(len(v) - 1)]
        flags += list(ok[:-1] & ok[1:])
    return segs, np.asarray(flags, bool)


def f1(s, a, b, tp):
    return 2 * s[tp] / max(1, s[a] + s[b])


def static(d, name):
    import nibabel as nib
    A, Minv = d['A'], np.linalg.inv(d['A'])
    image = nib.load(d['raw']).get_fdata()
    spacing = np.linalg.norm(A, axis=0)
    k = int(np.argmin(np.asarray(image.shape) * spacing))
    axes = [i for i in range(3) if i != k]
    mip = image.max(k)
    (gn, ge), (pn, pe) = d['gt_volume'], d['pred']
    tree_g = cKDTree(np.concatenate([dense(p) for _, _, p in ge]))
    tree_p = cKDTree(np.concatenate([dense(p) for _, _, p in pe])) if pe else None
    s = d['pred_score']
    aspect = spacing[axes[0]] / spacing[axes[1]]
    w = 6.0
    h = w * (image.shape[axes[0]] * spacing[axes[0]]) / (image.shape[axes[1]] * spacing[axes[1]])
    fig, ax = plt.subplots(1, 2, figsize=(2 * w + 0.4, h + 1.4), facecolor=SURFACE)
    planes = [[(st[i] + 64 + st[i + 1]) / 2 for i in range(len(st) - 1)] for st in d['grid']]
    for a, (edges, nodes, tree, title) in zip(ax, (
            (ge, gn, tree_p, f"GT volume graph: {len(gn)} nodes, {len(ge)} edges\n{100 * s['cl_gt_ok'] / max(1, s['cl_gt']):.0f}% of its centreline is in the stitched prediction"),
            (pe, pn, tree_g, f"Stitched prediction: {len(pn)} nodes, {len(pe)} edges; node F1 {f1(s, 'nodes', 'gt_nodes', 'node_tp'):.2f}, edge F1 {f1(s, 'edges', 'gt_edges', 'edge_tp'):.2f}\n"
                             f"{100 * s['cl_pred_ok'] / max(1, s['cl_pred']):.0f}% of its centreline is on the GT"))):
        a.imshow(mip, cmap='gray', origin='lower', vmin=np.percentile(mip, 5), vmax=np.percentile(mip, 99.7), aspect=aspect, alpha=0.75)
        for v in planes[axes[0]]:
            a.axhline(v, color='white', lw=0.4, alpha=0.35)
        for v in planes[axes[1]]:
            a.axvline(v, color='white', lw=0.4, alpha=0.35)
        segs, ok = segments(edges, tree, Minv, axes)
        swap = [seg[:, ::-1] for seg in segs]  # imshow: x = second axis, y = first axis
        a.add_collection(LineCollection([sg for sg, f in zip(swap, ok) if f], colors=BLUE, linewidths=1.4, capstyle='round'))
        a.add_collection(LineCollection([sg for sg, f in zip(swap, ok) if not f], colors=ORANGE, linewidths=1.4, capstyle='round'))
        nv = (nodes @ Minv.T)[:, axes]
        a.scatter(nv[:, 1], nv[:, 0], s=7, color='white', edgecolor=INK, linewidth=0.4, zorder=3)
        a.set_title(title, fontsize=8.5, color=INK, loc='left')
        a.set_xticks([]), a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    oc = (d['pred_open_cuts'] @ Minv.T)[:, axes] if len(d['pred_open_cuts']) else np.zeros((0, 2))
    ax[1].scatter(oc[:, 1], oc[:, 0], s=22, marker='^', color=AQUA, edgecolor='white', linewidth=0.5, zorder=4)
    handles = [Line2D([], [], color=BLUE, lw=1.8, label='left: GT found by the prediction / right: prediction on the GT (within 1.5 mm)'),
               Line2D([], [], color=ORANGE, lw=1.8, label='left: GT missed / right: prediction off the GT'),
               Line2D([], [], marker='^', ls='', color=AQUA, markeredgecolor='white', markersize=7, label=f'cut end without a partner ({len(oc)})'),
               Line2D([], [], marker='o', ls='', color='white', markeredgecolor=INK, markersize=4, label='node'),
               Line2D([], [], color='#9a9a96', lw=0.8, label='planes where patch cores meet')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.suptitle(f"{name}, validation subject {d['patient']}: full volume from stitched 64³ patches ({d['run']})", fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    path = OUT / f"{d['entry']}-{d['patient']}.png"
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def lines(edges, centre):
    xs = []
    for _, _, p in edges:
        q = np.round(dense(p, 1.0) - centre, 2)
        xs += q.tolist() + [[None, None, None]]
    return list(map(list, zip(*xs))) if xs else [[], [], []]


def interactive(d, name):
    A = d['A']
    (gn, ge), (pn, pe), (sn, se) = d['gt_volume'], d['pred'], d['gt']
    centre = gn.mean(0)
    pts = lambda n: np.round(n - centre, 2).T.tolist() if len(n) else [[], [], []]  # noqa: E731
    planes = [[(st[i] + 64 + st[i + 1]) / 2 for i in range(len(st) - 1)] for st in d['grid']]
    lo = [min(st) for st in d['grid']]
    hi = [max(st) + 64 for st in d['grid']]
    box = []
    for k in range(3):  # each core plane drawn as its rectangle outline
        i, j = [x for x in range(3) if x != k]
        for v in planes[k]:
            corners = []
            for a, b in ((lo[i], lo[j]), (hi[i], lo[j]), (hi[i], hi[j]), (lo[i], hi[j]), (lo[i], lo[j])):
                c = [0.0, 0.0, 0.0]
                c[k], c[i], c[j] = v, a, b
                corners.append(c)
            box += (np.round(np.asarray(corners) @ A.T - centre, 2)).tolist() + [[None, None, None]]
    s, o = d['pred_score'], d['gt_score']
    row = lambda label, x: [label, x['nodes'], x['edges'], f"{f1(x, 'nodes', 'gt_nodes', 'node_tp'):.3f}", f"{f1(x, 'edges', 'gt_edges', 'edge_tp'):.3f}",  # noqa: E731
                            f"{x['cl_pred_ok'] / max(1, x['cl_pred']):.3f}", f"{x['cl_gt_ok'] / max(1, x['cl_gt']):.3f}", x['beta0'], x['beta1']]
    table = [['GT volume graph', len(gn), len(ge), '', '', '', '', s['gt_beta0'], s['gt_beta1']],
             row('Stitched GT patches (test of the stitching alone)', o), row('Stitched prediction', s)]
    data = dict(gt=lines(ge, centre), gt_nodes=pts(gn), pred=lines(pe, centre), pred_nodes=pts(pn), stitched_gt=lines(se, centre),
                cuts=pts(d['pred_open_cuts']), planes=list(map(list, zip(*box))) if box else [[], [], []], table=table)
    html = TEMPLATE.replace('__TITLE__', f'{name} {d["patient"]} stitched volume').replace(
        '__HEAD__', f'{name}, validation subject {d["patient"]}: full-volume graph stitched from 64³ patches').replace(
        '__SUB__', f'Model {d["run"]} at its own thresholds (node {d["thresholds"]["node"]}, edge {d["thresholds"]["edge"]}). Each patch keeps its core; edges are cut at the '
                   'planes where cores meet and the cut points of neighbouring patches are matched one-to-one within 2 mm. Coordinates in mm. '
                   'Click a legend entry to show or hide it; drag to rotate.').replace('__DATA__', json.dumps(data, separators=(',', ':')))
    path = OUT / f"{d['entry']}-{d['patient']}.html"
    path.write_text(html)
    return path


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light; --surface: #fcfcfb; --text: #0b0b0b; --text2: #52514e; --rule: #dcdbd6; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --neutral: #9a9a96; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --surface: #1a1a19; --text: #f0efec; --text2: #c3c2b7; --rule: #3a3a37; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neutral: #77766f; } }
body { margin: 0; padding: 16px; background: var(--surface); color: var(--text); font: 14px/1.45 system-ui, sans-serif; }
h1 { font-size: 17px; margin: 0 0 4px; } p { margin: 0 0 10px; color: var(--text2); max-width: 1000px; }
#plot { width: 100%; height: 72vh; min-height: 420px; }
table { border-collapse: collapse; margin-top: 12px; font-size: 13px; } th, td { padding: 4px 10px; border-bottom: 1px solid var(--rule); text-align: right; }
th:first-child, td:first-child { text-align: left; } th { color: var(--text2); font-weight: 600; }
.wrap { overflow-x: auto; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
</head><body>
<h1>__HEAD__</h1><p>__SUB__</p>
<div id="plot"></div>
<div class="wrap"><table id="scores"><thead><tr><th>Graph</th><th>Nodes</th><th>Edges</th><th>Node F1</th><th>Edge F1</th><th>Centreline precision</th><th>Centreline recall</th><th>Components</th><th>Loops</th></tr></thead><tbody></tbody></table></div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function draw() {
  const c = {s1: css('--s1'), s2: css('--s2'), s3: css('--s3'), n: css('--neutral'), text: css('--text'), text2: css('--text2'), surface: css('--surface'), rule: css('--rule')};
  const line = (name, a, color, width, visible) => ({type: 'scatter3d', mode: 'lines', name, x: a[0], y: a[1], z: a[2], line: {color, width}, visible, connectgaps: false,
    hovertemplate: name + '<br>%{x:.1f}, %{y:.1f}, %{z:.1f} mm<extra></extra>'});
  const dots = (name, a, color, size, symbol, visible) => ({type: 'scatter3d', mode: 'markers', name, x: a[0], y: a[1], z: a[2], marker: {color, size, symbol, line: {color: c.surface, width: 1}}, visible,
    hovertemplate: name + '<br>%{x:.1f}, %{y:.1f}, %{z:.1f} mm<extra></extra>'});
  const traces = [
    line('GT volume graph', D.gt, c.s1, 4, true), dots('GT nodes', D.gt_nodes, c.s1, 3, 'circle', 'legendonly'),
    line('Stitched prediction', D.pred, c.s2, 4, true), dots('Predicted nodes', D.pred_nodes, c.s2, 3, 'circle', 'legendonly'),
    dots('Cut ends without a partner', D.cuts, c.s3, 4, 'diamond', true),
    line('Stitched GT patches', D.stitched_gt, c.s3, 2, 'legendonly'),
    line('Planes where patch cores meet', D.planes, c.n, 1, 'legendonly')];
  const axis = {color: c.text2, gridcolor: c.rule, zerolinecolor: c.rule, backgroundcolor: c.surface, showbackground: true, title: {text: 'mm'}};
  Plotly.react('plot', traces, {paper_bgcolor: c.surface, font: {color: c.text, family: 'system-ui, sans-serif', size: 12}, margin: {l: 0, r: 0, t: 0, b: 0},
    legend: {orientation: 'h', y: 1.02, yanchor: 'bottom', x: 0, font: {color: c.text}}, scene: {aspectmode: 'data', xaxis: axis, yaxis: axis, zaxis: axis}}, {responsive: true, displaylogo: false});
}
document.querySelector('#scores tbody').innerHTML = D.table.map(r => '<tr>' + r.map(v => '<td>' + v + '</td>').join('') + '</tr>').join('');
draw();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script></body></html>
'''


def main(entries):
    OUT.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        for f in sorted(DUMP.glob(f'{entry}-*.pkl')):
            d = pickle.loads(f.read_bytes())
            name = NAMES.get(entry, entry)
            print(static(d, name), interactive(d, name), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
