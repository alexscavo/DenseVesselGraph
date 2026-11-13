#!/usr/bin/env python3
"""Run a trained traced model on EVERY grid position of full volumes, for stitching (12 Nov 2026, GPU).

The saved validation rows of a run only cover the stored patches (grid positions that contain GT graph). Stitching
a whole volume needs a prediction everywhere, and the zero-overlap / thin-overlap tilings need other grids. Here the
raw volume is normalised exactly as scripts/generate_synthetic_mri_dataset.py does (whole volume first, then crops),
cropped at every position of a grid, and fed through T.predict (no augmentation).

Grids (patch starts per axis):
  overlap  the dataset's own grid, endpoint_axis_starts(size, 64, maximum_stride), ALL positions;
  thin     the same function with maximum stride 64 (minimal overlap);
  tile0    zero overlap: starts 0, 64, 128, ...; the normalised volume is zero-padded at the far end to a multiple of 64.

Checks (the run stops on any failure):
  crop     for up to 3 stored patches of every subject, the crop of the normalised volume at the stored start equals
           the stored patch file (max abs difference < 1e-5);
  dataset  once per run of this script: the tensor fed to the model and the metric equal what
           make_dataset(entry, [sample_id], augment=False) yields (max abs difference < 1e-5);
  rows     (diagnostic, not a stop) overlap grid, validation, the run's own node threshold: the predicted nodes at the
           stored validation starts against the run's final-validation-rows.pkl.

usage: predict_volume_grid_20261112.py <entry> <run> [--split val|train] [--subjects N] [--grids overlap thin tile0]
       [--node-thresholds t ...] [--out DIR]
Writes <out>/<grid>/<patient>-node<t>.pkl (rows with extra keys start, patient), <out>/<grid>/meta/<patient>.json and
<out>/<grid>/meta.json (all patients of the grid).
"""
import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_traced_20261030 as T  # noqa: E402
from node_heatmap_probe_20261030 import PLAN  # noqa: E402
from scripts.generate_synthetic_mri_dataset import endpoint_axis_starts, normalize_vessel_image, normalize_window  # noqa: E402

SIZE = 64
FOREGROUND_MEAN = 0.33335259556770325  # as node_heatmap_probe_20261030.make_dataset
RUNS = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
OUT = Path(os.environ['SCRATCH']) / 'experiments/gnbm/stitching_20261112/rows'
TOL = 1e-5


def normalise(raw_image, normalization):
    """The whole-volume normalisation of the generator for this cache's recorded normalisation."""
    raw = np.asanyarray(raw_image.dataobj)
    if normalization == 'legacy_mad_clip_v1':
        return normalize_vessel_image(raw)[0]
    m = re.fullmatch(r'window_(.+)_(.+)_v1', normalization)
    if m:
        return normalize_window(raw, (float(m.group(1)), float(m.group(2))))[0]
    raise ValueError(f'unknown normalization {normalization!r}')


def grid_starts(shape, grid, maximum_stride):
    if grid == 'overlap':
        return [endpoint_axis_starts(s, SIZE, m) for s, m in zip(shape, maximum_stride)]
    if grid == 'thin':
        return [endpoint_axis_starts(s, SIZE, SIZE) for s in shape]
    if grid == 'tile0':
        return [list(range(0, int(math.ceil(s / SIZE)) * SIZE, SIZE)) for s in shape]
    raise ValueError(grid)


class GridPatches(torch.utils.data.Dataset):
    """What T.predict needs: items as SyntheticMRIDataset (image minus the foreground mean, geometry['metric']) and
    records with sample_id / graph. There is no GT here: nodes and edges are empty."""

    def __init__(self, volume, starts, metric, patient):
        self.volume, self.starts, self.patient = volume, [tuple(int(x) for x in s) for s in starts], patient
        self.metric = torch.as_tensor(np.asarray(metric, np.float32))
        nowhere = Path('/nonexistent/vtp/none.vtp')  # T.predict derives a 'cache' path from it; overwritten below
        self.records = [SimpleNamespace(sample_id=f'{patient}_{d:03d}_{h:03d}_{w:03d}', graph=nowhere) for d, h, w in self.starts]

    def __len__(self):
        return len(self.starts)

    def image(self, i):
        d, h, w = self.starts[i]
        crop = np.ascontiguousarray(self.volume[d:d + SIZE, h:h + SIZE, w:w + SIZE])
        assert crop.shape == (SIZE,) * 3, (self.patient, self.starts[i], crop.shape)
        return torch.from_numpy(crop).float().unsqueeze(0) - FOREGROUND_MEAN  # same ops, same order as the loader

    def __getitem__(self, i):
        return ([self.image(i)], [None], [torch.zeros(0, 3)], [torch.zeros(0, 2, dtype=torch.long)], [{'metric': self.metric}], [1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entry')
    ap.add_argument('run')
    ap.add_argument('--split', choices=('val', 'train'), default='val')
    ap.add_argument('--subjects', type=int, default=0, help='first N patients of the split (sorted); 0 = all')
    ap.add_argument('--grids', nargs='+', default=['overlap'], choices=('overlap', 'thin', 'tile0'))
    ap.add_argument('--node-thresholds', type=float, nargs='+', help="default: the run's own node threshold")
    ap.add_argument('--out', type=Path)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--no-dataset-check', action='store_true', help='skip the make_dataset comparison (needs pyvista)')
    a = ap.parse_args()
    out = a.out or OUT / f'{a.entry}-{a.run}'
    plan = json.loads(PLAN.read_text())['datasets'][a.entry]
    simplified = Path(plan['root'])
    original = Path(json.loads((simplified / 'simplify_manifest.json').read_text())['source'])
    gen = json.loads((original / 'generation_config.json').read_text())
    source_root, maximum_stride, normalization = Path(gen['source_root']), gen['maximum_stride'], gen['normalization']
    assert list(gen['crop_size']) == [SIZE] * 3 and list(gen['pad']) == [0, 0, 0], (gen['crop_size'], gen['pad'])
    index = list(csv.DictReader((original / 'patch_index.csv').open()))
    by_sample = {r['sample_id']: r for r in index}
    ids = plan['val_ids' if a.split == 'val' else 'train_ids']
    plan_ids = defaultdict(list)
    for i in ids:
        plan_ids[by_sample[i]['patient_id']].append(i)
    patients = sorted(plan_ids)[:a.subjects] if a.subjects else sorted(plan_ids)
    stored = defaultdict(list)
    for r in index:
        stored[r['patient_id']].append(r)

    run_dir = RUNS / a.entry / a.run
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(final)
    own = final['thresholds']
    thresholds = [round(t, 4) for t in (a.node_thresholds or [own['node']])]
    device = 'cuda'
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], two_hop=T.MODE['two_hop'], passthrough=T.MODE['passthrough'],
                               backbone=final.get('backbone', 'unet'), feature_channels=final.get('feature_channels', 32)).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device)['model'])
    model.eval()
    print(f'{a.entry} {a.run}: split {a.split}, {len(patients)} patients {patients}, grids {a.grids}, node thresholds {thresholds}, '
          f'own thresholds {own["node"]} / {own["edge"]}, normalization {normalization}, maximum stride {maximum_stride}', flush=True)
    saved_val = None
    if a.split == 'val' and 'overlap' in a.grids and own['node'] in thresholds:
        saved_val = {r['sample_id']: r for r in pickle.loads((run_dir / 'final-validation-rows.pkl').read_bytes())}
    dataset_checked = a.no_dataset_check
    for patient in patients:
        t0 = time.time()
        raw_image = nib.load(str(source_root / 'raw' / f'{patient}.nii.gz'))
        affine = raw_image.affine
        A = affine[:3, :3]
        metric = (A @ np.diag([SIZE] * 3)).astype(np.float32)
        volume = normalise(raw_image, normalization)
        shape = tuple(int(s) for s in volume.shape)
        # ---- check 1: crops of the normalised volume equal the stored patches
        mine = sorted(stored[patient], key=lambda r: int(r['patch_index']))
        picks = [mine[k] for k in sorted({0, len(mine) // 2, len(mine) - 1})]
        for r in picks:
            st = tuple(int(r[k]) for k in ('start_d', 'start_h', 'start_w'))
            ref = np.asanyarray(nib.load(str(original / r['split'] / 'raw' / f"{r['sample_id']}_data.nii.gz")).dataobj)
            crop = volume[st[0]:st[0] + SIZE, st[1]:st[1] + SIZE, st[2]:st[2] + SIZE]
            diff = float(np.abs(crop.astype(np.float64) - ref.astype(np.float64)).max())
            print(f'CHECK crop {a.entry} {patient} {r["sample_id"]} start {st}: max abs diff {diff:.3g}', flush=True)
            if not diff < TOL:
                raise SystemExit(f'CHECK FAILED crop {a.entry} {patient} {r["sample_id"]}: max abs diff {diff}')
        # ---- check 1b: metric equals the cache metric (simplified cache, which the loader reads)
        for sid in plan_ids[patient][:3]:
            with np.load(simplified / by_sample[sid]['split'] / 'spline' / f'{sid}.npz') as z:
                diff = float(np.abs(z['metric'].astype(np.float64) - metric).max())
            print(f'CHECK metric {a.entry} {patient} {sid}: max abs diff {diff:.3g}', flush=True)
            if not diff < TOL:
                raise SystemExit(f'CHECK FAILED metric {a.entry} {patient} {sid}: max abs diff {diff}')
        # ---- check 2 (once): the tensor fed to the model equals what the training loader yields
        if not dataset_checked:
            from node_heatmap_probe_20261030 import make_dataset
            sids = plan_ids[patient][:3]
            ref_ds = make_dataset(a.entry, sids, augment=False)
            starts = [tuple(int(by_sample[s][k]) for k in ('start_d', 'start_h', 'start_w')) for s in sids]
            ds = GridPatches(volume, starts, metric, patient)
            for k, sid in enumerate(sids):
                item = ref_ds[k]
                d_img = float((item[0][0] - ds[k][0][0]).abs().max())
                d_met = float((item[4][0]['metric'].double() - ds.metric.double()).abs().max())
                print(f'CHECK dataset {a.entry} {sid}: image max abs diff {d_img:.3g} (shape {tuple(item[0][0].shape)} vs {tuple(ds[k][0][0].shape)}), '
                      f'metric max abs diff {d_met:.3g}', flush=True)
                if not (d_img < TOL and d_met < TOL and item[0][0].shape == ds[k][0][0].shape):
                    raise SystemExit(f'CHECK FAILED dataset {a.entry} {sid}: image {d_img}, metric {d_met}')
            dataset_checked = True
        # ---- inference on every grid position
        for grid in a.grids:
            starts_axis = grid_starts(shape, grid, maximum_stride)
            vol = volume
            if grid == 'tile0':
                padded = tuple(s[-1] + SIZE for s in starts_axis)
                vol = np.zeros(padded, np.float32)
                vol[:shape[0], :shape[1], :shape[2]] = volume
            starts = [(d, h, w) for d in starts_axis[0] for h in starts_axis[1] for w in starts_axis[2]]
            ds = GridPatches(vol, starts, metric, patient)
            gdir = out / grid
            (gdir / 'meta').mkdir(parents=True, exist_ok=True)
            for t in thresholds:
                t1 = time.time()
                rows = T.predict(model, ds, device, t, a.workers, a.batch)
                for row, st in zip(rows, ds.starts):
                    row.update(start=st, patient=patient, cache=None)
                path = gdir / f'{patient}-node{t:g}.pkl'
                tmp = path.with_suffix(f'.tmp{os.getpid()}')
                tmp.write_bytes(pickle.dumps(rows))
                os.replace(tmp, path)
                n_nodes = sum(int((r['node_scores'] >= t).sum()) for r in rows)
                print(f'PRED {a.entry} {patient} {grid} node {t:g}: {len(rows)} positions, {n_nodes} nodes, {time.time() - t1:.0f} s', flush=True)
                if grid == 'overlap' and saved_val is not None and t == own['node']:
                    rowcheck(a.entry, patient, rows, plan_ids[patient], by_sample, saved_val)
            mfile = gdir / 'meta' / f'{patient}.json'
            prev = json.loads(mfile.read_text()) if mfile.exists() else {}
            meta = dict(patient=patient, split=a.split, grid=grid, shape=list(shape), A=A.tolist(), starts=starts_axis,
                        node_thresholds=sorted(set(prev.get('node_thresholds', [])) | set(thresholds)), run_thresholds=own,
                        normalization=normalization, maximum_stride=maximum_stride, positions=len(starts),
                        padded_shape=list(vol.shape))
            mfile.write_text(json.dumps(meta, indent=1, default=float) + '\n')
            merged = {json.loads(f.read_text())['patient']: json.loads(f.read_text()) for f in sorted((gdir / 'meta').glob('*.json'))}
            tmp = gdir / f'meta.json.tmp{os.getpid()}'
            tmp.write_text(json.dumps(dict(entry=a.entry, run=a.run, grid=grid, patients=merged), indent=1, default=float) + '\n')
            os.replace(tmp, gdir / 'meta.json')
        print(f'{patient}: shape {shape}, {time.time() - t0:.0f} s', flush=True)


def rowcheck(entry, patient, rows, sids, by_sample, saved):
    """Diagnostic: rows at the stored validation starts against the run's saved validation rows."""
    at = {r['start']: r for r in rows}
    same, n, worst, count_diff = 0, 0, 0.0, []
    for sid in sids:
        if sid not in saved:
            continue
        st = tuple(int(by_sample[sid][k]) for k in ('start_d', 'start_h', 'start_w'))
        mine, ref = at[st], saved[sid]
        n += 1
        count_diff.append(len(mine['nodes']) - len(ref['nodes']))
        if len(mine['nodes']) == len(ref['nodes']):
            same += 1
            if len(ref['nodes']):
                a_ = np.sort(mine['nodes'] * SIZE, axis=0)
                b_ = np.sort(ref['nodes'] * SIZE, axis=0)
                worst = max(worst, float(np.abs(a_ - b_).max()))
    print(f'ROWCHECK {entry} {patient}: {n} stored validation patches, {same} with the same node count as final-validation-rows '
          f'(node-count differences: sum |d| {int(np.abs(count_diff).sum()) if count_diff else 0}), max sorted-coordinate difference '
          f'where counts agree {worst:.3g} voxels', flush=True)


if __name__ == '__main__':
    main()
