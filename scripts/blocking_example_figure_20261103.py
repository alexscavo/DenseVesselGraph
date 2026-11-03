#!/usr/bin/env python3
"""Worked examples of traced-candidate failures for a GT edge A-B whose endpoints are both detected (3 Nov 2026).

traced_adaptive, validation. One row per dataset: a third detection C lies on the vessel between A and
B (within 1.5 mm of the GT path), so the geodesic regions of A and B never touch; A-C and C-B
touch instead. Tracing is re-run on CPU from the saved detections. Panels: (1) image
MIP with the GT edge and detections, (2) centreline probability with the geodesic regions of the
relevant nodes, (3) the traced candidate paths around A and B with their edge scores.
Also prints, for each example, which region owns the voxels along the GT path.
Projection: along the axis on which the GT path is thinnest; panels are cropped around the edge.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from node_heatmap_probe_20261030 import make_dataset, SIZE  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402

D = T.D
RUN = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
OUT = REPO / 'docs/research/figures/unet_outputs_20261102'
NEAR_MM, END_MM = 1.5, 2.0
COLOURS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#d62728'}


def find_case(rows, nt, et, want_intermediate):
    """Shortest-to-read case: a GT edge >= 20 mm, both endpoints found, pair not traced."""
    best = None
    for r in rows:
        with np.load(r['cache']) as z:
            M, gn, ge = z['metric'], z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2)
            P, o = z['full_paths'] @ z['metric'].T, z['path_offsets']
        if r['traced'] is None:
            continue
        pred = r['nodes'] @ M.T
        inv = {g: i for i, g in D.match(torch.as_tensor(pred), torch.as_tensor(gn)).items()}
        traced = {tuple(sorted(map(int, p))): bool(t) for p, t in zip(r['pairs'], r['traced'])}
        for e, (a, b) in enumerate(ge):
            if a not in inv or b not in inv:
                continue
            path = P[o[e]:o[e + 1]]
            along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
            k = tuple(sorted((inv[a], inv[b])))
            score = {tuple(sorted(map(int, q))): v for q, v in zip(r['pairs'], r['edge_scores'])}
            if not 20 <= along[-1] <= 45 or traced.get(k, False) or score.get(k, 0.0) >= et:
                continue  # want: not traced AND missed
            inner = path[(along > END_MM) & (along < along[-1] - END_MM)]
            d = np.linalg.norm(pred[:, None] - inner[None], axis=-1).min(1)
            d[[inv[a], inv[b]]] = np.inf
            near = np.nonzero(d < NEAR_MM)[0]
            if bool(len(near)) != want_intermediate or (want_intermediate and len(near) != 1):
                continue
            c_matched = bool(near[0] in inv.values()) if len(near) else False
            rank = (c_matched, abs(len(pred) - 12))  # prefer a spurious C, and a readable patch
            if best is None or rank < best[0]:
                best = (rank, r['sample_id'], e, r['nodes'] * SIZE, inv[a], inv[b], int(near[0]) if len(near) else None, c_matched)
    return best


def panel_case(axes, d, case, nt, et):
    _, sid, e, seeds, ia, ib, ic, c_matched = case
    model = T.TracedDenseGraph().eval()
    model.load_state_dict(torch.load(RUN / d / 'traced_adaptive/model.pt', map_location='cpu')['model'])
    s = make_dataset(d, [sid], augment=False)[0]
    metric = s[4][0]['metric'] / SIZE
    with torch.no_grad():
        out = model.forward_image(s[0][0].float()[None]).float()
        prob = out[:, 4].sigmoid()
        p = torch.as_tensor(seeds)  # the saved (GPU, bf16) detections, so indices match the evaluation
        contacts, dist, label = T.traced_pairs(prob, [p])
        pairs = T.patch_candidates(p, metric, contacts[0])
        paths, traced = T.build_paths(p, pairs, contacts[0], dist[0], label[0])
        x = model.path_features(out[0, 5:], out[0, :1].sigmoid(), paths)
        edge_scores = model.edge_head(x).squeeze(-1).sigmoid()
    rec = make_dataset(d, [sid], augment=False).records[0]
    with np.load(rec.graph.parent.parent / 'spline' / f'{sid}.npz') as z:
        gp = z['full_paths'][z['path_offsets'][e]:z['path_offsets'][e + 1]] * SIZE
    pv, lab, pr = p.numpy(), label[0].numpy().astype(int), prob[0].numpy()
    print('RETRACED', d, sid, 'A-B contact' if tuple(sorted((ia, ib))) in contacts[0] else 'no A-B contact',
          'C spurious' if not c_matched else 'C matched to a GT node', flush=True)
    axis = int(np.argmin(np.ptp(gp, 0)))
    keep_axes = [k for k in range(3) if k != axis]
    lo = np.clip(np.floor(gp.min(0) - 8), 0, SIZE - 1).astype(int)
    hi = np.clip(np.ceil(gp.max(0) + 8), 0, SIZE - 1).astype(int)
    sl = [slice(lo[k], hi[k] + 1) for k in range(3)]
    def proj(v):
        return v[:, keep_axes[1]], v[:, keep_axes[0]]
    img = s[0][0][0].numpy()[tuple(sl)].max(axis)
    names = {'A': ia, 'B': ib} | ({'C': ic} if ic is not None else {})
    # region ownership along the GT path
    gi = np.clip(np.round(gp), 0, SIZE - 1).astype(int)
    own = lab[gi[:, 0], gi[:, 1], gi[:, 2]]
    inv_names = {v: k for k, v in names.items()}
    tally = {}
    for v in own:
        k = inv_names.get(v, 'outside mask' if v < 0 else 'other node')
        tally[k] = tally.get(k, 0) + 1
    others = sorted({int(v) for v in own if v >= 0 and v not in inv_names})
    print('OWNERSHIP', d, sid, {k: round(v / len(own), 3) for k, v in tally.items()}, 'other labels', others, flush=True)
    for k, v in zip('DEFG', others[:4]):
        names[k] = v
        COLOURS.setdefault(k, ['#9467bd', '#8c564b', '#e377c2', '#17becf'][len(COLOURS) % 4])
    extent = (lo[keep_axes[1]] - .5, hi[keep_axes[1]] + .5, hi[keep_axes[0]] + .5, lo[keep_axes[0]] - .5)
    # (1) image
    a0 = axes[0]
    fg = img[img > img.min()]
    a0.imshow(img, cmap='gray', extent=extent, vmin=np.percentile(fg, 30) if len(fg) else None, vmax=np.percentile(fg, 99.7) if len(fg) else None)
    a0.plot(*proj(gp), '-', color='#2ecc71', lw=2, label='GT edge A–B')
    for k, i in names.items():
        a0.plot(*proj(pv[i:i + 1]), 'o', ms=9, color=COLOURS[k], mec='k')
        a0.annotate(k, proj(pv[i:i + 1]), xytext=(5, 5), textcoords='offset points', color=COLOURS[k], fontsize=12, weight='bold')
    rest = [i for i in range(len(pv)) if i not in names.values()]
    a0.plot(*proj(pv[rest]), 's', ms=4, color='w', mec='k', label='other detections')
    a0.set_xlim(extent[0], extent[1]); a0.set_ylim(extent[2], extent[3])
    a0.legend(loc='lower left', fontsize=7)
    arc = np.linalg.norm(np.diff(gp @ metric.numpy().T, axis=0), axis=1).sum()
    a0.set_title(f'{d} {sid}\nGT edge A–B, {arc:.0f} mm; detections (MIP)', fontsize=9)
    # (2) centreline + regions
    a1 = axes[1]
    a1.imshow(pr[tuple(sl)].max(axis), cmap='Greys', vmin=0, vmax=1.5, extent=extent)
    for k, i in names.items():
        vox = np.argwhere(lab == i)
        vox = vox[((vox >= lo) & (vox <= hi)).all(1)]
        a1.scatter(*proj(vox), s=14, color=COLOURS[k], alpha=.35, lw=0, marker='s')
        a1.plot(*proj(pv[i:i + 1]), 'o', ms=9, color=COLOURS[k], mec='k')
        a1.annotate(k, proj(pv[i:i + 1]), xytext=(5, 5), textcoords='offset points', color=COLOURS[k], fontsize=12, weight='bold')
    a1.plot(*proj(gp), '--', color='#2ecc71', lw=1.2)
    a1.set_xlim(extent[0], extent[1]); a1.set_ylim(extent[2], extent[3])
    share = ', '.join(f'{k} {v / len(own):.0%}' for k, v in sorted(tally.items(), key=lambda t: -t[1]))
    a1.set_title(f'Centreline p (grey) + geodesic regions (colours)\nGT path voxels owned by: {share}', fontsize=9)
    # (3) candidates touching A or B
    a2 = axes[2]
    a2.imshow(img, cmap='gray', extent=extent, alpha=.5)
    a2.plot(*proj(gp), '-', color='#2ecc71', lw=5, alpha=.35, label='GT edge A–B (missed)')
    pl = pairs.numpy()
    for j, (u, v) in enumerate(pl):
        if not ({u, v} & {ia, ib}):
            continue
        sc = float(edge_scores[j])
        style = '-' if traced[j] else ':'
        col = 'tab:red' if sc < et else 'tab:blue'
        q = paths[j].numpy()
        a2.plot(*proj(q), style, color=col, lw=1.5)
        mid = q[len(q) // 2:len(q) // 2 + 1]
        lab_uv = ''.join(k for k, i in names.items() if i in (u, v)) or '·'
        if {u, v} <= set(names.values()):
            a2.annotate(f'{lab_uv} {sc:.2f}{"" if traced[j] else " (straight)"}', proj(mid), fontsize=8, color=col)
    for k, i in names.items():
        a2.plot(*proj(pv[i:i + 1]), 'o', ms=9, color=COLOURS[k], mec='k')
        a2.annotate(k, proj(pv[i:i + 1]), xytext=(5, 5), textcoords='offset points', color=COLOURS[k], fontsize=12, weight='bold')
    ab = [j for j, (u, v) in enumerate(pl) if {u, v} == {ia, ib}]
    note = f'A–B: straight fallback, score {float(edge_scores[ab[0]]):.2f}' if ab else 'A–B: not a candidate'
    a2.set_xlim(extent[0], extent[1]); a2.set_ylim(extent[2], extent[3])
    a2.set_title(f'Candidates at A or B (solid = traced, dotted = straight 16-NN)\nblue accepted, red rejected at edge threshold {et:.2f}; {note}', fontsize=9)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    return dict(sample=sid, intermediate_is_matched=c_matched, ownership=tally, ab=note)


def main():
    torch.set_num_threads(8)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    info = {}
    for row, (d, want) in enumerate([('SyntheticMRI', True), ('IXI', True), ('BRAVA', True)]):
        res = json.loads((RUN / d / 'traced_adaptive/final.json').read_text())
        nt, et = res['thresholds']['node'], res['thresholds']['edge']
        rows = pickle.loads((RUN / d / 'traced_adaptive/final-validation-rows.pkl').read_bytes())
        case = find_case(rows, nt, et, want)
        del rows
        print('CASE', d, case, flush=True)
        info[d] = panel_case(axes[row], d, case, nt, et)
    fig.suptitle('In-between detection C blocks the traced candidate A–B (traced_adaptive, validation)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'tracing_blocking_examples.png', dpi=110)
    print('INFO', json.dumps(info, default=str), flush=True)


if __name__ == '__main__':
    main()
