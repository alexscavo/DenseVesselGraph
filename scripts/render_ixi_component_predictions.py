#!/usr/bin/env python3
"""Render saved final predictions on one pre-existing fixed validation example."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from medpy.io import load
import numpy as np

WORK = Path(__file__).resolve().parents[1]
META = WORK / 'experiments/ixi_local_components_20261026'
MAIN = WORK.parent / 'Graph-Native-Betti-Matching-Representation'
OUT = MAIN / 'docs/research/evidence/ixi_local_components_20261028'
SAMPLE = 'sample_000001_0009'
ARMS = ('A-xyz-fresh', 'A-cls')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def render(volume, panels, path):
    fig, axes = plt.subplots(len(panels), 3, figsize=(14, 4.35 * len(panels)),
                             squeeze=False, facecolor='white')
    low, high = np.quantile(volume, [.03, .995])
    assert high > low
    projections = [(0, 2, 1, 'D', 'W', 'H'), (1, 2, 0, 'H', 'W', 'D'),
                   (2, 1, 0, 'W', 'H', 'D')]
    shape = np.array(volume.shape)
    for row, panel in enumerate(panels):
        for col, (axis, x, y, name, xlabel, ylabel) in enumerate(projections):
            ax = axes[row, col]
            ax.imshow(volume.max(axis=axis), cmap='gray', origin='lower',
                      vmin=low, vmax=high, interpolation='nearest')
            for curve in panel['curves']:
                points = np.asarray(curve) * shape
                ax.plot(points[:, x], points[:, y], color=panel['color'],
                        lw=1.5, alpha=.95)
            nodes = panel['nodes'] * shape
            ax.scatter(nodes[:, x], nodes[:, y], s=16, c=panel['color'],
                       edgecolors='#172127', linewidths=.35, zorder=5)
            ax.set(xlim=(-.5, shape[x] - .5), ylim=(-.5, shape[y] - .5),
                   xlabel=f'{xlabel} (input voxels)', ylabel=f'{ylabel} (input voxels)',
                   title=f"{panel['label']}\nProjection along {name}")
    fig.suptitle(f'IXI validation: {SAMPLE} | final checkpoint: 5,500 updates',
                 fontsize=15, y=.997)
    fig.legend(handles=[Line2D([0], [0], color='#ffba49', marker='o', label='GT graph / dense reference'),
                        Line2D([0], [0], color='#00d6c6', marker='o', label='Saved predicted nodes / K8 curves')],
               loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .035, 1, .975))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    plan = read(META / 'plan.json')
    root = Path(plan['output_root'])
    assert SAMPLE in plan['dataset']['val_ids']
    OUT.mkdir(parents=True, exist_ok=True)
    data = Path(plan['dataset']['root']) / 'val'
    image_path = data / 'raw' / f'{SAMPLE}_data.nii.gz'
    target_path = data / 'spline' / f'{SAMPLE}.npz'
    # Use the same MedPy image reader as data/loaders/io.py. No segmentation input.
    raw, _ = load(str(image_path))
    config = read(root / 'runs' / ARMS[0] / 'config.json')
    mean = config['data']['datasets']['synthetic_mri']['foreground_mean']
    volume = np.asarray(raw, dtype=np.float32) - float(mean)
    assert volume.shape == (64, 64, 64) and np.isfinite(volume).all()
    with np.load(target_path, allow_pickle=False) as cache:
        gt = {k: cache[k].copy() for k in cache.files}
    paths = [gt['full_paths'][a:b] for a, b in zip(gt['path_offsets'][:-1], gt['path_offsets'][1:])]
    panels = [dict(label=f"GT: {len(gt['nodes'])} nodes, {len(gt['edges'])} edges",
                   nodes=gt['nodes'], curves=paths, color='#ffba49')]
    metadata = dict(sample=SAMPLE, split='val', subject=plan['dataset']['subjects'][SAMPLE.split('_')[1]],
                    selection='First existing fixed validation visualization; no score-based selection',
                    checkpoint_step=5500, input_image=str(image_path), image_sha256=sha(image_path),
                    foreground_mean_subtracted=float(mean), target=str(target_path), target_sha256=sha(target_path),
                    no_segmentation_overlay=True, inference_or_training=False, arms={})
    for arm in ARMS:
        folder = root / 'runs' / arm
        assert read(folder / 'complete.json')['primary_step'] == 5500
        assert read(folder / 'config.json')['data']['datasets']['synthetic_mri']['foreground_mean'] == mean
        source = folder / 'step-5500-reconstruction/validation-000000/predictions.json'
        predictions = read(source)
        pred = next(p for p in predictions if p.get('source_sample_id', p['sample_id']) == SAMPLE)
        del predictions
        nodes = np.asarray(pred['nodes_dhw'])
        edges = np.asarray(pred['edges'], dtype=int).reshape(-1, 2)
        curves = np.asarray(pred['spline_curves'])
        controls = np.asarray(pred['spline_controls'])
        queries = np.asarray(pred['query_ids'], dtype=int)
        metric = np.asarray(pred['spline_metric'])
        assert len(curves) == len(edges) and len(nodes) == len(queries)
        assert np.allclose(curves[:, [0, -1]], nodes[edges], atol=1e-6)
        assert np.allclose(controls[:, [0, -1]], nodes[edges], atol=1e-6)
        assert np.allclose(metric, gt['metric'])
        with zipfile.ZipFile(folder / 'step-5500/caches.zip') as archive:
            with np.load(io.BytesIO(archive.read(SAMPLE + '.npz'))) as cache:
                assert np.allclose(nodes @ metric.T, cache['physical'][queries], atol=1e-5)
                assert np.allclose(pred['node_scores'], cache['scores'][queries], atol=1e-6)
                assert np.array_equal(queries, np.flatnonzero(cache['scores'] > .5))
        row = next(r for r in read(folder / 'step-5500/rows.json')
                   if r['sample'] == SAMPLE and r['split'] == 'val' and r['tolerance'] == '2.0')
        assert len(nodes) == row['nodes'] and len(edges) == row['edges']
        panels.append(dict(label=f'{arm}: {len(nodes)} nodes, {len(edges)} edges',
                           nodes=nodes, curves=curves, color='#00d6c6'))
        metadata['arms'][arm] = dict(predictions=str(source), predictions_sha256=sha(source),
                                    saved_graph=pred, physical_2mm_counts=row,
                                    final_raw_cache_coordinate_and_score_parity=True)
    render(volume, panels, OUT / f'{SAMPLE}-comparison.png')
    for arm, panel in zip(ARMS, panels[1:]):
        render(volume, [panels[0], panel], OUT / f'{SAMPLE}-{arm}.png')
    (OUT / f'{SAMPLE}-provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(OUT / f'{SAMPLE}-comparison.png', flush=True)


if __name__ == '__main__':
    main()
