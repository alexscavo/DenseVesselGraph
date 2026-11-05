#!/usr/bin/env python3
"""TopBrain (batches 1+2) as full, unstripped volumes with the complete labels (4 Nov 2026).

Supersedes the skull-stripped variant (prepare_topbrain_stripped_20261104.py): TopCoW's widened brain mask
holds skull-base bone (CTA) and extracranial soft tissue (MRA), and any brain mask drops 24-45% of the TopBrain
annotation (cervical ICA/VA, ECA branches, sinuses). Here nothing is masked, so the GT graph covers every
annotated vessel and matches the image.

Writes below --output (images and labels are referenced in place, never copied or modified):
- manifest.csv: extraction manifest, one row per volume; dataset topbrain_full_{mr,ct};
- patient_split_{mr,ct}.csv: one subject-level split shared by both modalities (TopCoW patient number, seed 42,
  15/5/5), identical to the stripped variant's; patient ids as prepare_real_graph_patch_sources forms them;
- preparation.json: provenance (sha256 of every image and label) and the split.
The official TopBrain 'Te' CTAs are an artificial split (TopBrain note.txt), so they are pooled.
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_topbrain_stripped_20261104 import DATA, SPLIT_SEED, SPLIT_SIZES, TOPBRAIN, patient_split, sha256  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DATA / 'TopBrain_full_20261104')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    volumes = []
    for image_path in sorted(glob.glob(str(TOPBRAIN / 'images*_topbrain_*/*.nii.gz'))):
        image_path = Path(image_path)
        modality, number = image_path.name[7:9], image_path.name[10:13]
        label_path = Path(glob.glob(str(TOPBRAIN / f'labels*_topbrain_{modality}/topcow_{modality}_{number}.nii.gz'))[0])
        image, label = nib.load(image_path), nib.load(label_path)
        if image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-4):
            raise ValueError(f'{image_path.name}: image/label grid mismatch')
        volumes.append(dict(modality=modality, patient=number, subject=f'topcow_{modality}_{number}',
                            image=str(image_path), image_sha256=sha256(image_path),
                            label=str(label_path), label_sha256=sha256(label_path),
                            shape=list(image.shape), spacing=np.sqrt((label.affine[:3, :3] ** 2).sum(0)).tolist()))
    by_mod = {m: sorted([v for v in volumes if v['modality'] == m], key=lambda v: v['patient']) for m in ('mr', 'ct')}
    patients = sorted({v['patient'] for v in volumes})
    assert all({v['patient'] for v in by_mod[m]} == set(patients) for m in by_mod), 'MRA/CTA patients differ'
    split = patient_split(patients)
    rows = []
    for m in ('mr', 'ct'):
        for v in by_mod[m]:
            v['index'], v['split'] = len(rows), split[v['patient']]
            sp = v['spacing']
            rows.append(dict(index=v['index'], dataset=f'topbrain_full_{m}', modality=m, split='train', subject=v['subject'],
                             image=v['image'], label=v['label'], shape_x=v['shape'][0], shape_y=v['shape'][1], shape_z=v['shape'][2],
                             spacing_x=f'{sp[0]:.8g}', spacing_y=f'{sp[1]:.8g}', spacing_z=f'{sp[2]:.8g}'))
    with open(a.output / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for m in ('mr', 'ct'):
        with open(a.output / f'patient_split_{m}.csv', 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(('patient_id', 'split'))
            for v in by_mod[m]:
                w.writerow((f"{v['index']:06d}", v['split']))
    (a.output / 'preparation.json').write_text(json.dumps(dict(
        created='2026-11-04', script='scripts/prepare_topbrain_full_20261104.py',
        policy=dict(stripping='none (full field of view)', labels='complete TopBrain labels, unmodified',
                    split=dict(level='patient (TopCoW number), shared by MRA and CTA', seed=SPLIT_SEED, sizes=SPLIT_SIZES,
                               note='official TopBrain Te CTAs are an artificial split and are pooled'),
                    intensity=dict(mr='legacy MAD clip (as IXI)', ct='fixed window [-100, 400] HU')),
        split=split, volumes=volumes), indent=1) + '\n')
    print(len(rows), 'volumes;', {s: sum(1 for x in split.values() if x == s) for s in SPLIT_SIZES}, 'patients')


if __name__ == '__main__':
    main()
