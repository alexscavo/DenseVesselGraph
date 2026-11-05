#!/usr/bin/env python3
"""Mask options for stripping TopBrain (4 Nov 2026): per volume, the share of vessel voxels each candidate mask
keeps and, for CTA, the share of mask voxels above 500 HU (bone proxy). Candidates: TopCoW tight mask (HD-BET /
TotalSegmentator), tight dilated 2 and 4 mm, and TopCoW's widened mask (used in the stripped variant)."""
import glob, json, sys
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_topbrain_stripped_20261104 import TOPBRAIN, TOPCOW, onto, label_names  # noqa: E402
OUT = Path(__file__).resolve().parents[1] / 'experiments/topbrain_preparation_20261104/mask_options.json'
rows = []
ONLY = sys.argv[1] if len(sys.argv) > 1 else None  # optional modality filter: mr or ct
if ONLY: OUT = OUT.with_name(f'mask_options_{ONLY}.json')
for f in sorted(glob.glob(str(TOPBRAIN / f"images*_topbrain_{ONLY or '*'}/*.nii.gz"))):
    f = Path(f); mod, num = f.name[7:9], f.name[10:13]; tag = {'mr': 'MRA', 'ct': 'CTA'}[mod]
    sub = f'sub-topcow{mod}{num}'; stem = TOPCOW / 'TopCoW_preprocessed' / sub / tag.lower() / sub
    img = nib.load(f); a = np.asanyarray(img.dataobj)
    lab = np.asanyarray(nib.load(glob.glob(str(TOPBRAIN / f'labels*_topbrain_{mod}/topcow_{mod}_{num}.nii.gz'))[0]).dataobj)
    L = lab > 0; names = label_names(mod)
    tight = onto(nib.load(f'{stem}_space-{tag}_desc-braintight_mask.nii.gz'), img)
    wide = onto(nib.load(f'{stem}_space-{tag}_desc-brain_mask.nii.gz'), img)
    z = np.array(img.header.get_zooms()[:3]); masks = dict(tight=tight, wide=wide)
    for mm in (2, 4):
        r = np.ceil(mm / z).astype(int); g = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
        ball = ((g * z[:, None, None, None]) ** 2).sum(0) <= mm ** 2
        masks[f'tight+{mm}mm'] = ndimage.binary_dilation(tight, ball)
    row = dict(subject=f.stem[:13], modality=mod)
    for k, m in masks.items():
        row[k] = dict(kept=round(float((L & m).sum() / L.sum()), 4),
                      cow_kept=round(float((np.isin(lab, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]) & m).sum() / max(1, np.isin(lab, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]).sum())), 4))
        if mod == 'ct': row[k]['bone_share'] = round(float((a[m] > 500).mean()), 4)
        row[k]['lost_labels'] = {names.get(int(v), str(v)): round(float(((lab == v) & ~m).sum() / (lab == v).sum()), 2)
                                 for v in np.unique(lab[L]) if ((lab == v) & ~m).sum() / (lab == v).sum() > 0.2}
    rows.append(row); print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'lost_labels'}) for k, v in row.items()}), flush=True)
OUT.write_text(json.dumps(rows, indent=1) + '\n')
