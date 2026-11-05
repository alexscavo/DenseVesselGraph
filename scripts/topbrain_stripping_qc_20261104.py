#!/usr/bin/env python3
"""QC of the skull-stripped TopBrain volumes (4 Nov 2026): per case, three views of the original image with the
brain-mask outline (yellow) and the TopCoW tight mask (cyan), the stripped image, and the vessel labels kept (white)
vs removed by the mask (red). Mid slices through the mask's centre, plus axial MIP."""
import sys, json
from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import ndimage
O = Path(sys.argv[1] if len(sys.argv) > 1 else '/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/TopBrain_stripped_20261104')
TC = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/TopCoW/TopCoW_preprocessed')
OUT = Path(__file__).resolve().parents[1] / 'docs/research/figures/topbrain_stripping_20261104' / O.name
OUT.mkdir(parents=True, exist_ok=True)
prep = json.loads((O / 'preparation.json').read_text())
def onto(mask_img, img):
    o = nib.orientations; t = o.ornt_transform(o.io_orientation(mask_img.affine), o.io_orientation(img.affine))
    return np.asanyarray(mask_img.as_reoriented(t).dataobj) > 0
def win(x, mod):
    lo, hi = ((-100, 400) if mod == 'ct' else np.percentile(x[x > 0], (1, 99.7)))
    return np.clip((x - lo) / (hi - lo), 0, 1)
for mod in ('mr', 'ct'):
    vols = [v for v in prep['volumes'] if v['modality'] == mod][::6][:4]
    fig, axes = plt.subplots(len(vols), 6, figsize=(21, 3.7 * len(vols)), facecolor='black')
    for r, v in enumerate(vols):
        img = nib.load(v['source_image']); a = np.asanyarray(img.dataobj).astype(np.float32)
        s = np.asanyarray(nib.load(v['image']).dataobj).astype(np.float32)
        m = np.asanyarray(nib.load(v['mask']).dataobj) > 0
        tag = {'mr': 'MRA', 'ct': 'CTA'}[mod]; sub = v['topcow_subject']
        tight = onto(nib.load(TC / sub / tag.lower() / f'{sub}_space-{tag}_desc-braintight_mask.nii.gz'), img)
        lab = np.asanyarray(nib.load(v['source_label']).dataobj) > 0
        c = [int(x) for x in ndimage.center_of_mass(m)]
        zoom = img.header.get_zooms()[:3]
        views = [('axial', lambda x: x[:, :, c[2]].T, zoom[1] / zoom[0]), ('coronal', lambda x: x[:, c[1], :].T, zoom[2] / zoom[0]),
                 ('sagittal', lambda x: x[c[0], :, :].T, zoom[2] / zoom[1])]
        for k, (name, cut, asp) in enumerate(views):
            ax = axes[r, k]; ax.imshow(win(cut(a), mod), cmap='gray', origin='lower', aspect=asp)
            ax.contour(cut(m.astype(float)), [0.5], colors='yellow', linewidths=0.8)
            ax.contour(cut(tight.astype(float)), [0.5], colors='cyan', linewidths=0.6)
            ax.set_title(f"{v['subject']} {name}: original, mask (yellow), tight (cyan)", color='white', fontsize=8)
        ax = axes[r, 3]; ax.imshow(win(views[1][1](s), mod), cmap='gray', origin='lower', aspect=views[1][2]); ax.set_title('stripped, coronal', color='white', fontsize=8)
        ax = axes[r, 4]; ax.imshow(win(s.max(2).T if mod == 'mr' else np.where(m, a, -1024).max(2).T, mod), cmap='gray', origin='lower', aspect=zoom[1] / zoom[0]); ax.set_title('stripped, axial MIP', color='white', fontsize=8)
        ax = axes[r, 5]; kept = (lab & m).max(1).T; lost = (lab & ~m).max(1).T
        rgb = np.zeros(kept.shape + (3,)); rgb[kept] = 1; rgb[lost & ~kept] = (1, 0.2, 0.2)
        ax.imshow(rgb, origin='lower', aspect=zoom[2] / zoom[0]); ax.set_title(f"labels coronal MIP: kept white, removed red ({v['vessel_kept']:.0%} kept)", color='white', fontsize=8)
        for ax in axes[r]: ax.axis('off')
    fig.tight_layout(); fig.savefig(OUT / f'stripping_qc_{mod}.png', dpi=110, facecolor='black'); print(OUT / f'stripping_qc_{mod}.png', flush=True)
