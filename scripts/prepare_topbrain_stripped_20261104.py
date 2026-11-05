#!/usr/bin/env python3
"""Skull-stripped TopBrain volumes with labels clipped to the same brain mask (4 Nov 2026).

Production choice (5 Nov 2026, user): --mask braintight --output .../TopBrain_tight_20261105. The widened mask
(default, TopBrain_stripped_20261104) holds skull-base bone on CTA and sinus/soft tissue on MRA: superseded.

TopBrain (batches 1+2) reuses the TopCoW images: topcow_{mr,ct}_NNN is TopCoW subject
sub-topcow{mr,ct}NNN, whose preprocessed folder holds a QC'd brain mask (HD-BET for MRA, widened
below the brain and dilated 2 mm; TotalSegmentator for CTA, widened below the brain, not dilated).
The masks are on TopCoW's RAS-flipped grid; they are reoriented onto the TopBrain (LPS) grid and
the affines are checked to agree. The TopBrain image must equal the TopCoW raw image voxel for voxel.

Outputs (originals untouched), below --output:
- images_{mr,ct}/topcow_{mod}_NNN_0000.nii.gz: image inside the mask; outside 0 (MRA) or -1024 HU
  (CTA, values below -1024 clamped to -1024 as in TopCoW);
- labels_{mr,ct}/topcow_{mod}_NNN.nii.gz: the TopBrain multi-class label, set to 0 outside the
  same mask, so the GT graph only holds vessels the stripped image shows;
- masks_{mr,ct}/topcow_{mod}_NNN_brainmask.nii.gz;
- manifest.csv (extraction manifest, one row per volume; dataset topbrain_stripped_{mr,ct}),
  patient_split_{mr,ct}.csv (one subject-level split shared by both modalities, keyed by the
  TopCoW patient number, so a patient's MRA and CTA never fall in different splits),
  preparation.json (provenance, sha256s, per-volume and per-label kept fractions, intensities).
The official TopBrain 'Te' CTAs are an artificial split (TopBrain note.txt), so they are pooled.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import median_abs_deviation

DATA = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets')
TOPBRAIN, TOPCOW = DATA / 'TopBrain', DATA / 'TopCoW'
SPLIT_SEED = 42
SPLIT_SIZES = dict(train=15, val=5, test=5)  # 25 patients; 3 validation subjects (70/15/15) is too few for paired CIs


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def label_names(modality):
    names = {}
    for line in (TOPBRAIN / f'itksnap_labelmap_txt/labelmap_topbrain_{modality}.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            names[int(line.split()[0])] = line.split('"')[1]
    return names


def onto(mask_image, image):
    """Reorient mask_image onto image's voxel grid (pure axis flips/permutations only) and verify."""
    o = nib.orientations
    transform = o.ornt_transform(o.io_orientation(mask_image.affine), o.io_orientation(image.affine))
    moved = mask_image.as_reoriented(transform)
    if moved.shape != image.shape or not np.allclose(moved.affine, image.affine, atol=1e-3):
        raise ValueError('mask grid does not match image grid after reorientation')
    return np.asanyarray(moved.dataobj) > 0


def save_like(array, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(array.dtype)
    header.set_xyzt_units('mm')  # TopBrain label headers say 'unknown'; their spacing is the image's, in mm
    out = nib.Nifti1Image(array, reference.affine, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp.nii.gz')
    nib.save(out, str(tmp))
    os.replace(tmp, path)


def process(job):
    image_path, output = Path(job[0]), Path(job[1])
    which = job[2] if len(job) > 2 else 'brain'  # TopCoW mask: 'brain' (widened) or 'braintight'
    modality, number = image_path.name[7:9], image_path.name[10:13]
    tag = {'mr': 'MRA', 'ct': 'CTA'}[modality]
    subject = f'sub-topcow{modality}{number}'
    stem = TOPCOW / 'TopCoW_preprocessed' / subject / tag.lower() / subject
    label_path = Path(glob.glob(str(TOPBRAIN / f'labels*_topbrain_{modality}/topcow_{modality}_{number}.nii.gz'))[0])
    raw_path = Path(glob.glob(str(TOPCOW / f'TopCoW_raw/images*/{image_path.name}'))[0])
    mask_path = Path(f'{stem}_space-{tag}_desc-{which}_mask.nii.gz')

    image, label_image = nib.load(image_path), nib.load(label_path)
    data = np.asanyarray(image.dataobj)
    raw = nib.load(raw_path)
    if raw.shape != image.shape or not np.allclose(raw.affine, image.affine, atol=1e-4) \
            or not np.array_equal(np.asanyarray(raw.dataobj), data):
        raise ValueError(f'{image_path.name}: TopBrain image differs from TopCoW raw {raw_path}')
    if label_image.shape != image.shape or not np.allclose(label_image.affine, image.affine, atol=1e-4):
        raise ValueError(f'{image_path.name}: label grid differs from image grid')
    mask = onto(nib.load(mask_path), image)
    label = np.asanyarray(label_image.dataobj)

    if modality == 'ct':
        stripped = np.where(mask, np.maximum(data, -1024), -1024).astype(np.int16)
    else:
        if data.min() < 0:
            raise ValueError(f'{image_path.name}: negative MRA intensities')
        stripped = np.where(mask, data, 0).astype(data.dtype)
    clipped = np.where(mask, label, 0).astype(label.dtype)

    name = f'topcow_{modality}_{number}'
    out_image = output / f'images_{modality}' / f'{name}_0000.nii.gz'
    out_label = output / f'labels_{modality}' / f'{name}.nii.gz'
    out_mask = output / f'masks_{modality}' / f'{name}_brainmask.nii.gz'
    save_like(stripped, image, out_image)
    save_like(clipped, label_image, out_label)
    save_like(mask.astype(np.uint8), image, out_mask)

    names = label_names(modality)
    vessel = label > 0
    per_label = {}
    for value in np.unique(label[vessel]):
        n = int((label == value).sum())
        per_label[names.get(int(value), str(value))] = dict(voxels=n, kept=round(float(((label == value) & mask).sum() / n), 4))
    inside = stripped[mask].astype(np.float32)
    kept = vessel & mask
    intensity = dict(vessel_percentiles={p: float(v) for p, v in zip((1, 5, 50, 95, 99), np.percentile(data[kept], (1, 5, 50, 95, 99)))},
                     tissue_percentiles={p: float(v) for p, v in zip((1, 5, 50, 95, 99), np.percentile(data[mask & ~vessel], (1, 5, 50, 95, 99)))})
    if modality == 'mr':  # what the legacy MAD clip of generate_synthetic_mri_dataset would do on the stripped volume
        integer = stripped.astype(np.int32).ravel()
        threshold = float(median_abs_deviation(integer, scale='normal') * 4.0 + np.median(integer))
        legacy = dict(threshold=threshold, fallback=not threshold > 0)
        if legacy['fallback']:
            legacy['threshold'] = float(np.percentile(stripped.astype(np.float32), 99.5))
        legacy['vessel_saturated_fraction'] = float((data[kept] >= legacy['threshold']).mean())
        legacy['tissue_median_normalized'] = float(np.median(data[mask & ~vessel]) / legacy['threshold'])
        intensity['legacy_normalization'] = legacy
    return dict(
        modality=modality, patient=number, subject=name, topcow_subject=subject,
        image=str(out_image), label=str(out_label), mask=str(out_mask),
        source_image=str(image_path), source_image_sha256=sha256(image_path),
        source_label=str(label_path), source_label_sha256=sha256(label_path),
        brain_mask=str(mask_path), brain_mask_sha256=sha256(mask_path), topcow_raw=str(raw_path),
        shape=list(image.shape), spacing_mm=[float(z) for z in image.header.get_zooms()[:3]],
        mask_fraction=round(float(mask.mean()), 4), inside_min=float(inside.min()), inside_max=float(inside.max()),
        vessel_voxels=int(vessel.sum()), vessel_kept=round(float(kept.sum() / vessel.sum()), 4),
        labels=per_label, intensity=intensity)


def patient_split(patients):
    ranked = sorted(patients, key=lambda p: hashlib.sha256(f'{SPLIT_SEED}:topbrain:{p}'.encode()).hexdigest())
    assert len(ranked) == sum(SPLIT_SIZES.values()), (len(ranked), SPLIT_SIZES)
    out, i = {}, 0
    for split, n in SPLIT_SIZES.items():
        for p in ranked[i:i + n]:
            out[p] = split
        i += n
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DATA / 'TopBrain_stripped_20261104')
    p.add_argument('--mask', choices=('brain', 'braintight'), default='brain',
                   help="TopCoW mask: 'brain' (widened below the brain; holds skull-base bone on CTA) or 'braintight' "
                        "(HD-BET / TotalSegmentator brain only)")
    p.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')) // 2 or 1)
    p.add_argument('--smoke', action='store_true', help='one MRA and one CTA only; no manifest or split')
    a = p.parse_args()
    images = sorted(glob.glob(str(TOPBRAIN / 'images*_topbrain_*/*.nii.gz')))
    if a.smoke:
        for image in (images[0], images[-1]):
            r = process((image, str(a.output), a.mask))
            print(json.dumps({k: v for k, v in r.items() if k != 'labels'}), flush=True)
            print('labels kept < 0.95:', {k: v for k, v in r['labels'].items() if v['kept'] < 0.95}, flush=True)
        return
    jobs = [(i, str(a.output), a.mask) for i in images]
    with Pool(a.workers) as pool:
        records = pool.map(process, jobs, chunksize=1)
    by_mod = {m: sorted([r for r in records if r['modality'] == m], key=lambda r: r['patient']) for m in ('mr', 'ct')}
    patients = sorted({r['patient'] for r in records})
    assert all({r['patient'] for r in by_mod[m]} == set(patients) for m in by_mod), 'MRA/CTA patients differ'
    split = patient_split(patients)

    rows = []
    for m in ('mr', 'ct'):
        for r in by_mod[m]:
            r['index'] = len(rows)
            r['split'] = split[r['patient']]
            sp = np.sqrt((nib.load(r['label']).affine[:3, :3] ** 2).sum(0))
            rows.append(dict(index=r['index'], dataset=f"topbrain_{'tight' if a.mask == 'braintight' else 'stripped'}_{m}", modality=m, split='train', subject=r['subject'],
                             image=r['image'], label=r['label'], shape_x=r['shape'][0], shape_y=r['shape'][1], shape_z=r['shape'][2],
                             spacing_x=f'{sp[0]:.8g}', spacing_y=f'{sp[1]:.8g}', spacing_z=f'{sp[2]:.8g}'))
    with open(a.output / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for m in ('mr', 'ct'):  # patient ids as prepare_real_graph_patch_sources forms them: f'{manifest index:06d}'
        with open(a.output / f'patient_split_{m}.csv', 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(('patient_id', 'split'))
            for r in by_mod[m]:
                w.writerow((f"{r['index']:06d}", r['split']))
    summary = dict(
        created='2026-11-04', script='scripts/prepare_topbrain_stripped_20261104.py',
        policy=dict(stripping=f'TopCoW desc-{a.mask}_mask reoriented onto the TopBrain grid', outside=dict(mr=0, ct=-1024),
                    labels='TopBrain labels set to 0 outside the same mask; originals untouched',
                    split=dict(level='patient (TopCoW number), shared by MRA and CTA', seed=SPLIT_SEED, sizes=SPLIT_SIZES,
                               note='official TopBrain Te CTAs are an artificial split and are pooled')),
        split=split, volumes=records)
    (a.output / 'preparation.json').write_text(json.dumps(summary, indent=1) + '\n')
    for m in ('mr', 'ct'):
        kept = [r['vessel_kept'] for r in by_mod[m]]
        print(m, 'volumes', len(kept), 'vessel kept min/median/max', min(kept), float(np.median(kept)), max(kept), flush=True)


if __name__ == '__main__':
    main()
