#!/usr/bin/env python3
"""Figures for §4.12–4.13 (9 Nov 2026), drawn from saved validation predictions (no model is run).

1. curve_examples.png: a RANDOM sample (seed 0, not hand-picked) of short GT edges (chord 1–5 mm) accepted as true
   positives by both the reference model and the arc-penalty model; GT dense path vs each model's curve, projected on
   the GT path's best-fit plane, in mm.
2. patch_overlay.png: one validation patch per dataset, image MIP with every accepted curve of the reference and of
   the arc-penalty model (compressed to the fewest controls within 0.5 voxel), GT paths underneath.
3. feature_ablation.png: Edge F1 change vs number of principal feature directions kept (experiments/feature_ablation).
Each run's own train-selected thresholds (final.json) decide which edges are accepted.
usage: curve_shape_figures_20261109.py   (CPU; writes docs/research/figures/curve_shape_20261109/)
"""
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.spline_targets import fit_spline, resample  # noqa: E402

RUNS = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
OUT = ROOT / 'docs/research/figures/curve_shape_20261109'
SETS = {'IXI-loops': 'traced_adaptive', 'BRAVA': 'traced_offset'}
ARM = 'traced_arcpen_r1109'
SIZE = 64
ORANGE, BLUE, INK = '#B4481F', '#2B66A3', '#14213D'


def load(dataset, run):
    d = RUNS / dataset / run
    final = json.loads((d / 'final.json').read_text())
    rows = pickle.loads((d / 'final-validation-rows.pkl').read_bytes())
    return {r['sample_id']: r for r in rows}, final['thresholds']['node'], final['thresholds']['edge']


def match(a, b, tol=2.0):
    if not len(a) or not len(b):
        return {}
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    r, c = linear_sum_assignment(np.where(d <= tol, d, 1e6))
    return {int(i): int(j) for i, j in zip(r, c) if d[i, j] <= tol}


def accepted_tp(row, nt, et):
    """GT edge index -> curve (mm) for accepted true-positive pairs."""
    if row['edge_scores'] is None:
        return {}
    z = np.load(row['cache'])
    M = z['metric']
    keep = row['node_scores'] >= nt
    mp = match(row['nodes'] @ M.T, z['nodes'] @ M.T)
    mp = {i: j for i, j in mp.items() if keep[i]}
    ge = {tuple(sorted(map(int, e))): k for k, e in enumerate(z['edges'].reshape(-1, 2))}
    out = {}
    for k in np.flatnonzero(row['edge_scores'] >= et):
        a, b = map(int, row['pairs'][k])
        e = ge.get(tuple(sorted((mp.get(a, -1), mp.get(b, -2))))) if a in mp and b in mp else None
        if e is not None:
            c = np.concatenate([row['nodes'][a][None], row['curves'][k][1:-1], row['nodes'][b][None]]) @ M.T
            out[e] = c
    return out


def gt_path(cache, e):
    z = np.load(cache)
    o = z['path_offsets']
    return z['full_paths'][o[e]:o[e + 1]] @ z['metric'].T


def arc(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def plane(p):
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c)
    return c, vt[:2]


def compress(c_mm, voxel_mm, tol_vox=0.5):
    """Fewest-control fit (K8, K16) within tol, as compress_curves in the model (mm here)."""
    q = resample(c_mm, 64)
    for k in (8, 16):
        f = fit_spline(q, k, 64)[1]
        d = np.linalg.norm(f[:, None] - q[None], axis=-1)
        if max(d.min(1).max(), d.min(0).max()) <= tol_vox * voxel_mm:
            return f, k
    return c_mm, 24


def examples():
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.4))
    rng = np.random.default_rng(0)
    for row_i, (ds, ref) in enumerate(SETS.items()):
        R, rn, re_ = load(ds, ref)
        A, an, ae = load(ds, ARM)
        pool = []
        for sid in sorted(set(R) & set(A)):
            tr, ta = accepted_tp(R[sid], rn, re_), accepted_tp(A[sid], an, ae)
            for e in sorted(set(tr) & set(ta)):
                g = gt_path(R[sid]['cache'], e)
                if 1 <= np.linalg.norm(g[-1] - g[0]) < 5:
                    pool.append((sid, e, g, tr[e], ta[e]))
            if len(pool) > 4000:
                break
        pick = rng.choice(len(pool), 5, replace=False)
        for col, i in enumerate(pick):
            sid, e, g, cr, ca = pool[i]
            ax = axes[row_i, col]
            c, B = plane(np.vstack([g, cr, ca]))
            for p, colr, lw, ls, lab in ((g, INK, 3, '--', 'GT path'), (cr, ORANGE, 2.5, '-', 'reference'), (ca, BLUE, 2.5, '-', 'arc penalty')):
                xy = (p - c) @ B.T
                ax.plot(xy[:, 0], xy[:, 1], ls, color=colr, lw=lw, label=lab)
            ends = (np.stack([g[0], g[-1]]) - c) @ B.T
            ax.plot(ends[:, 0], ends[:, 1], 'o', color=INK, ms=7)
            ch = np.linalg.norm(g[-1] - g[0])
            tort = lambda p: arc(p) / max(np.linalg.norm(p[-1] - p[0]), 1e-6)  # noqa: E731  (each curve's own chord)
            ax.set_title(f'{ds.split("-")[0]} · GT chord {ch:.1f} mm\narc/chord: ref {tort(cr):.2f} · penalty {tort(ca):.2f} · GT {tort(g):.2f}', fontsize=12)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=10)
            ax.set_xlabel('mm', fontsize=10)
        print(ds, 'pool of short TP edges in both models:', len(pool), flush=True)
    fig.suptitle('Random sample (seed 0) of short true edges (GT chord 1–5 mm), validation: reference vs arc-length penalty.\n'
                 'Curves start at the predicted nodes (matched to the GT nodes within 2 mm); the GT path at the GT nodes.', fontsize=14)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / 'curve_examples.png', dpi=110)
    plt.close(fig)


def overlay():
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    for row_i, (ds, ref) in enumerate(SETS.items()):
        R, rn, re_ = load(ds, ref)
        A, an, ae = load(ds, ARM)
        # a typical busy patch: the 75th percentile of accepted reference edges among patches present in both
        cand = []
        for sid in sorted(set(R) & set(A)):
            r = R[sid]
            if r['edge_scores'] is not None:
                cand.append((int((r['edge_scores'] >= re_).sum()), sid))
        cand.sort()
        sid = cand[int(0.75 * (len(cand) - 1))][1]
        cache = Path(R[sid]['cache'])
        z = np.load(cache)
        M = z['metric']
        voxel = float(np.linalg.norm(M, axis=0).mean()) / SIZE  # mm per voxel
        img = np.asarray(nib.load(cache.parent.parent / 'raw' / f'{sid}_data.nii.gz').dataobj, float)
        proj = 'MIP' if ds.startswith('IXI') else 'mean projection'  # BRAVA background saturates a MIP
        mip = img.max(0) if proj == 'MIP' else img.mean(0)
        for col, (rows, nt, et, name, colr) in enumerate(((R, rn, re_, 'reference', ORANGE), (A, an, ae, 'arc penalty, compressed', BLUE))):
            ax = axes[row_i, col]
            ax.imshow(mip, cmap='gray', origin='upper', vmin=np.percentile(mip, 1 if proj != 'MIP' else 50), vmax=np.percentile(mip, 99.8))
            o = z['path_offsets']
            for e in range(len(z['edges'].reshape(-1, 2))):
                p = z['full_paths'][o[e]:o[e + 1]] * SIZE
                ax.plot(p[:, 2], p[:, 1], '--', color='#9FE3A0', lw=1.5)
            r = rows[sid]
            keep = r['node_scores'] >= nt
            ks = []
            for k in np.flatnonzero(r['edge_scores'] >= et):
                a, b = map(int, r['pairs'][k])
                if not (keep[a] and keep[b]):
                    continue
                c = np.concatenate([r['nodes'][a][None], r['curves'][k][1:-1], r['nodes'][b][None]]) * SIZE
                if col == 1:
                    cm, kk = compress(c @ (M / SIZE).T, voxel)
                    c = cm @ np.linalg.inv(M / SIZE).T
                    ks.append(kk)
                ax.plot(c[:, 2], c[:, 1], '-', color=colr, lw=2)
            n = r['nodes'][keep] * SIZE
            ax.plot(n[:, 2], n[:, 1], 'o', color='#FFFDF8', mec=INK, ms=6)
            extra = f' · mean controls {np.mean(ks):.1f}' if ks else ''
            ax.set_title(f'{ds.split("-")[0]} {sid} ({proj}): {name}{extra}', fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('Validation patches (projection along D): accepted curves (solid), GT paths (green dashed), nodes (circles)', fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / 'patch_overlay.png', dpi=100)
    plt.close(fig)


def ablation():
    ks = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    names = {'SyntheticMRI-loops': 'SyntheticMRI', 'IXI-loops': 'IXI', 'BRAVA': 'BRAVA', 'TopBrainMRA-loops': 'TopBrain MRA', 'TopBrainCTA-loops': 'TopBrain CTA'}
    for (ds, name), col, mk in zip(names.items(), ['#6B7280', BLUE, ORANGE, '#4A7F5A', INK], 'osD^v'):
        v = json.loads((ROOT / f'experiments/feature_ablation_20261109/{ds}.json').read_text())['validation']
        y = [v[f'pca{k}']['V0_minus_none']['edge_f1']['point'] if k < 32 else 0.0 for k in ks]
        ax.plot(ks, y, marker=mk, color=col, lw=2, label=name)
    ax.axhline(-0.01, color='#B4481F', lw=1, ls=':', label='protocol margin −0.01')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('principal feature directions kept (of 32)', fontsize=12)
    ax.set_ylabel('Edge F1 change vs full model', fontsize=12)
    ax.set_ylim(-0.12, 0.01)
    ax.set_xticks(ks)
    ax.legend(fontsize=11, ncol=2)
    ax.set_title('Test-time feature ablation (frozen global decoding, validation)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'feature_ablation.png', dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    ablation()
    examples()
    overlay()
    print('SAVED', sorted(p.name for p in OUT.iterdir()), flush=True)
