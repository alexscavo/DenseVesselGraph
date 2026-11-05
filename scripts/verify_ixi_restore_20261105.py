#!/usr/bin/env python3
"""Verify restored IXI images (5 Nov 2026) against patches cut from the originals in September.

For each IXI test subject (the 21 whose images were purged and restored): the restored image must have its label's
grid (shape, affine), and every surviving September test patch in the original K8 cache
(IXI/transfer_splines_k8_20261015, raw patches = normalized crops of the original image) must equal the same crop
of the restored image after the same normalization (generate_synthetic_mri_dataset.normalize_vessel_image).
Read-only."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.generate_synthetic_mri_dataset import normalize_vessel_image  # noqa: E402

I = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI')
K = I / 'transfer_splines_k8_20261015'
rows = list(csv.DictReader((I / 'vascular_graph_sources/optimal_radius_0p75x_c095_vedo_original/source_manifest.csv').open()))
test = {r['patient_id']: r for r in rows if r['patch_split'] == 'test'}
patches = defaultdict(list)
for r in csv.DictReader((K / 'patch_index.csv').open()):
    if r['split'] == 'test':
        patches[r['patient_id']].append(r)
report, total, identical = {}, 0, 0
for pid, r in sorted(test.items()):
    image, label = nib.load(r['image']), nib.load(r['label'])
    grid = image.shape == label.shape and np.allclose(image.affine, label.affine, atol=1e-4)
    norm, threshold = normalize_vessel_image(np.asanyarray(image.dataobj))
    same = 0
    for p in patches[pid]:
        s = [int(p['start_' + a]) for a in 'dhw']
        crop = norm[s[0]:s[0] + 64, s[1]:s[1] + 64, s[2]:s[2] + 64].astype(np.float32)
        old = np.asanyarray(nib.load(K / 'test/raw' / f"{p['sample_id']}_data.nii.gz").dataobj)
        same += int(crop.shape == old.shape and np.array_equal(crop, old))
    total += len(patches[pid])
    identical += same
    report[Path(r['image']).name] = dict(patient_id=pid, grid_matches_label=bool(grid), patches=len(patches[pid]), identical=same)
    print(Path(r['image']).name, report[Path(r['image']).name], flush=True)
summary = dict(n_subjects=len(test), all_grids_match=all(v['grid_matches_label'] for v in report.values()),
               patches=total, identical=identical, subjects=report)
(REPO / 'experiments/loop_subdivision_20261105/ixi_restore_verification.json').write_text(json.dumps(summary, indent=1) + '\n')
print('VERIFY', json.dumps({k: v for k, v in summary.items() if k != 'subjects'}), flush=True)
