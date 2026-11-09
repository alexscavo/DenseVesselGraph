#!/usr/bin/env python3
"""QC of the TubeTK Skwerer preparation (5 Nov 2026): per subject, axial / coronal / sagittal MIPs of the skull-stripped
MRA with the clipped Skwerer graph (cyan), the Skwerer parts cut by the brain mask (yellow) and the Bullitt centreline
not covered by Skwerer within 1 mm (red). Subjects: Normal-008 (duplicate subtree removed), the two train/val subjects
with the largest missing Bullitt fraction, and one tall-grid (Normal-096..109) train/val subject. Test subjects are
not shown. usage: tubetk_stripping_qc_20261105.py [dataset_dir] [subject ...]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

O = Path(sys.argv[1] if len(sys.argv) > 1 else '/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/TubeTK_skwerer_20261105')
OUT = Path(__file__).resolve().parents[1] / 'docs/research/figures/tubetk_20261105'
OUT.mkdir(parents=True, exist_ok=True)
prep = json.loads((O / 'preparation.json').read_text())
rows = {r['subject']: r for r in prep['per_subject']}
if len(sys.argv) > 2:
    chosen = sys.argv[2:]
else:
    dev = [r for r in prep['per_subject'] if r['split'] != 'test']
    worst = [r['subject'] for r in sorted([r for r in dev if r['has_bullitt'] and r['subject'] != 'sub-Normal008'],
                                          key=lambda r: -r['missing_bullitt_fraction'])[:2]]
    tall = [r['subject'] for r in dev if int(r['subject'][-3:]) >= 96][:1]
    chosen = (['sub-Normal008'] if rows['sub-Normal008']['split'] != 'test' else []) + worst + tall
fig, axes = plt.subplots(len(chosen), 3, figsize=(18, 6.2 * len(chosen)), facecolor='black',
                         gridspec_kw=dict(width_ratios=[1, 1, 1]))
axes = np.atleast_2d(axes)
for i, subject in enumerate(chosen):
    r = rows[subject]; folder = O / 'volumes' / subject
    img = nib.load(folder / 'brain_skullstripped.nii.gz'); a = np.asanyarray(img.dataobj).astype(np.float32)
    zoom = img.header.get_zooms()[:3]
    with np.load(folder / 'qc_points.npz') as q:
        kept, cut, miss = q['skwerer_kept_vox'], q['skwerer_removed_vox'], q['bullitt_missing_vox']
    lo, hi = np.percentile(a[a > 0], (50, 99.9))
    for k, (axis, name) in enumerate([(2, 'axial'), (1, 'coronal'), (0, 'sagittal')]):
        keep = [d for d in range(3) if d != axis]
        ax = axes[i, k]
        ax.imshow(np.clip((a.max(axis) - lo) / (hi - lo), 0, 1).T, cmap='gray', origin='lower',
                  aspect=zoom[keep[1]] / zoom[keep[0]])
        for pts, colour, size, alpha in ((kept[::3], '#00e5ff', 0.15, 0.35), (cut, '#ffd600', 0.6, 0.8), (miss, '#ff1744', 0.6, 0.9)):
            if len(pts):
                ax.scatter(pts[:, keep[0]], pts[:, keep[1]], s=size, c=colour, alpha=alpha, linewidths=0)
        ax.set_xlim(-0.5, a.shape[keep[0]] - 0.5); ax.set_ylim(-0.5, a.shape[keep[1]] - 0.5); ax.axis('off')
        extra = (f", Bullitt missing {r['missing_bullitt_fraction']:.1%} ({r['missing_bullitt_mm']:.0f} mm; r>=2 mm: {r['missing_radius_ge_2mm_mm']:.0f} mm)"
                 if r['has_bullitt'] else ', no Bullitt tree')
        ax.set_title(f"{subject} ({r['split']}) {name} MIP: kept {r['kept_length_fraction']:.1%} of Skwerer length{extra if k == 0 else ''}",
                     color='white', fontsize=9)
fig.suptitle('Skull-stripped MRA (desc-brain_MRA). Cyan: Skwerer GT after clipping to desc-brain_mask; yellow: Skwerer cut by the mask; '
             'red: Bullitt centreline (inside mask) with no Skwerer within 1 mm', color='white', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.985))
path = OUT / 'tubetk_stripping_qc.png'
fig.savefig(path, dpi=100, facecolor='black'); print(path, chosen, flush=True)
