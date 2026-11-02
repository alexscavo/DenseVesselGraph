"""Render one real D07 training patch through its frozen augmentation loader."""
import base64
import hashlib
import html
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO.parent / 'gnbm-brava-learning-diagnosis-20261026'
sys.path.insert(0, str(FROZEN))
from data.brava_representation_study import RepresentationDataset
from data.loaders.common import SamplePaths
from data.loaders.synthetic_mri import SyntheticMRIDataset
from data.augmentations import rotate_coordinates, rotate_volume
from data.spline_augmentation import rotate_spline_geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecordIntensity:
    """Observe the existing MONAI transform without changing its draws."""
    def __init__(self, transform):
        self.transform = transform

    def set_random_state(self, seed):
        self.seed = seed
        self.transform.set_random_state(seed=seed)

    def __call__(self, image):
        self.before = image.detach().clone()
        result = self.transform(image)
        self.noise_applied = bool(self.transform.transforms[0]._do_transform)
        return result


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    torch.set_num_threads(2)
    output = REPO / 'docs/research/evidence/brava_supervision_20261027'
    output.mkdir(parents=True, exist_ok=True)
    plan_path = REPO / 'experiments/brava_augmentation_20261026/plan.json'
    plan = json.loads(plan_path.read_text())
    sample = plan['populations']['representative_train'][0]
    subject = plan['subjects'][sample.split('_')[1]]
    assert sample in plan['ids']['train']
    print('Rendering actual training patch', subject, sample, flush=True)
    root = Path(plan['dataset']) / 'train'
    record = SamplePaths(root/'raw'/f'{sample}_data.nii.gz',
                         root/'seg'/f'{sample}_seg.nii.gz',
                         root/'vtp'/f'{sample}_graph.vtp', sample)
    cache_path = (REPO.parent / 'Graph-Native-Betti-Matching-Representation-BRAVA-20261026'
                  / 'experiments/brava_representation_20261026/targets.pkl')
    with cache_path.open('rb') as stream:
        cache = pickle.load(stream)[sample]['K8']
    source_hashes = json.loads((REPO/'experiments/brava_augmentation_20261026/frozen-source-sha256.json').read_text())
    checked_sources = {}
    for name in ['data/brava_representation_study.py', 'data/augmentations.py',
                 'data/spline_augmentation.py', 'data/loaders/synthetic_mri.py']:
        checked_sources[name] = sha(FROZEN/name)
        assert checked_sources[name] == source_hashes[name], name
    with np.load(root/'spline'/f'{sample}.npz') as disk:
        for name in ['nodes', 'edges', 'controls', 'curves', 'guides', 'metric',
                     'full_paths', 'path_offsets', 'boundary']:
            np.testing.assert_array_equal(cache[name], disk[name])
    base = SyntheticMRIDataset([record], augment=False, zoom_range=None,
                              spline_controls=8,
                              foreground_mean=plan['config']['data']['datasets']['synthetic_mri']['foreground_mean'])
    ds = RepresentationDataset(base, {sample: {'K8': cache}}, 'K8', augment=True)
    ds.augment = False
    clean = ds[0]
    observer = RecordIntensity(ds.intensity)
    ds.intensity = observer
    ds.augment = True
    seed = 261039
    expected_rng = random.Random(seed)
    turns = tuple(expected_rng.randint(0, 3) for _ in range(3))
    expected_noise_seed = expected_rng.randrange(2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    augmented = ds[0]
    assert observer.seed == expected_noise_seed
    image, aug_image = clean[0][0], augmented[0][0]
    shape = tuple(image.shape[-3:])
    expected_image = rotate_volume(image, turns)
    torch.testing.assert_close(observer.before, expected_image, rtol=0, atol=0)
    expected_nodes = rotate_coordinates(clean[2][0], turns, shape, allow_patch_faces=True)
    torch.testing.assert_close(augmented[2][0], expected_nodes, rtol=0, atol=0)
    expected_geometry = rotate_spline_geometry(clean[4][0], turns, shape)
    for name in ['controls', 'curves', 'guides', 'metric']:
        torch.testing.assert_close(augmented[4][0][name], expected_geometry[name], rtol=0, atol=0)
    torch.testing.assert_close(clean[3][0], augmented[3][0], rtol=0, atol=0)
    # An independent voxel-index formula checks the coordinate transform against
    # the actual torch.rot90 image operation, including the size-minus-one offset.
    def independently_rotated_voxels(points):
        voxel = points.numpy().astype(float) * np.array(shape)
        current_shape = list(shape)
        for k, (a, b) in zip(turns, [(0, 1), (0, 2), (1, 2)]):
            for _ in range(k):
                old_a, old_b = voxel[..., a].copy(), voxel[..., b].copy()
                voxel[..., a], voxel[..., b] = current_shape[b]-1-old_b, old_a
                current_shape[a], current_shape[b] = current_shape[b], current_shape[a]
        return voxel
    rotation_error_vox = float(np.abs(independently_rotated_voxels(clean[2][0])-augmented[2][0].numpy()*64).max())
    geometry_errors = {key: float(np.abs(independently_rotated_voxels(clean[4][0][key])-
                                        augmented[4][0][key].numpy()*64).max())
                       for key in ['controls', 'curves', 'guides']}
    assert max([rotation_error_vox]+list(geometry_errors.values())) < 1e-5
    torch.testing.assert_close(augmented[4][0]['controls'][:, [0,-1]],
                               augmented[2][0][augmented[3][0]], atol=2e-6, rtol=0)
    print('Image/graph rotation verified; building overlays', flush=True)
    geometry = clean[4][0]
    np.testing.assert_allclose(cache['controls'][:, [0, -1]], cache['nodes'][cache['edges']], atol=2e-6)
    dense = torch.from_numpy(cache['full_paths'].copy()).float()
    aug_dense = rotate_coordinates(dense, turns, shape, allow_patch_faces=True)
    variants = []
    for label, item, paths in [('Clean', clean, dense), ('Augmented', augmented, aug_dense)]:
        variants.append(dict(label=label, image=item[0][0][0].detach().cpu().numpy(),
                             nodes=item[2][0].numpy()*64,
                             curves=item[4][0]['curves'].numpy()*64,
                             controls=item[4][0]['controls'].numpy()*64,
                             paths=paths.numpy()*64))
    lo, hi = np.percentile(variants[0]['image'], [0.5, 99.7])
    offsets = cache['path_offsets']
    colors = dict(dense='#ff9c35', spline='#00d5f0', nodes='#ffe066', controls='#d488ff')
    axis_names = ['D', 'H', 'W']
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor='#101720')
    for row, data in enumerate(variants):
        for axis in range(3):
            ax = axes[row, axis]
            b, c = [v for v in range(3) if v != axis]
            ax.imshow(data['image'].max(axis=axis).T, origin='lower', cmap='gray',
                      vmin=lo, vmax=hi, extent=(-.5, 63.5, -.5, 63.5), interpolation='nearest')
            for start, end in zip(offsets[:-1], offsets[1:]):
                path = data['paths'][start:end]
                ax.plot(path[:, b], path[:, c], color=colors['dense'], lw=2.0, alpha=.85)
            for curve in data['curves']:
                ax.plot(curve[:, b], curve[:, c], color=colors['spline'], lw=1.0)
            ax.scatter(data['nodes'][:, b], data['nodes'][:, c], c=colors['nodes'],
                       s=18, linewidths=.4, edgecolors='#1a1a1a', zorder=4)
            ax.set(xlim=(-.5, 63.5), ylim=(-.5, 63.5), xlabel=f'{axis_names[b]} (voxel)',
                   ylabel=f'{axis_names[c]} (voxel)',
                   title=f"{data['label']} · projection along {axis_names[axis]}")
            ax.tick_params(colors='#cbd5e1', labelsize=8)
            ax.xaxis.label.set_color('#cbd5e1'); ax.yaxis.label.set_color('#cbd5e1')
            ax.title.set_color('white')
    fig.suptitle(f'BRAVA {subject} · {sample} · actual K8 training supervision\n'
                 f'64³ patch · {len(cache["nodes"])} graph nodes · {len(cache["edges"])} edges · '
                 f'quarter-turn draws {turns} in DH / DW / HW', color='white', fontsize=14, y=.995)
    legend = [Line2D([0], [0], color=colors['dense'], lw=3, label='Annotated SWC paths'),
              Line2D([0], [0], color=colors['spline'], lw=2, label='K8 spline targets'),
              Line2D([0], [0], color=colors['nodes'], marker='o', lw=0, label='Graph nodes')]
    fig.legend(handles=legend, loc='lower center', ncol=3, facecolor='#1c2633',
               edgecolor='none', labelcolor='white', bbox_to_anchor=(.5, .012))
    fig.tight_layout(rect=(0, .06, 1, .925))
    png = output/'brava_clean_augmented_overlay.png'
    fig.savefig(png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Interactive 3D views add image slices to disambiguate depth collapsed by MIP.
    view = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                         subplot_titles=['Clean image + training graph', 'Augmented image + training graph'])
    anchor = cache['full_paths'][len(cache['full_paths'])//2]*64
    aug_anchor = rotate_coordinates(torch.tensor(anchor[None]/64).float(), turns, shape,
                                    allow_patch_faces=True).numpy()[0]*64
    for col, (data, center) in enumerate(zip(variants, [anchor, aug_anchor]), 1):
        for axis in range(3):
            index = int(np.clip(np.rint(center[axis]), 0, 63))
            b, c = [v for v in range(3) if v != axis]
            u, v = np.meshgrid(np.arange(64), np.arange(64), indexing='ij')
            xyz = [None]*3; xyz[axis] = np.full_like(u, index); xyz[b] = u; xyz[c] = v
            texture = np.take(data['image'], index, axis=axis)
            view.add_trace(go.Surface(x=xyz[0], y=xyz[1], z=xyz[2], surfacecolor=texture,
                           colorscale='Gray', cmin=float(lo), cmax=float(hi), showscale=False,
                           opacity=.72, lighting=dict(ambient=1, diffuse=0, specular=0),
                           name=f'{data["label"]}: MRI {axis_names[axis]}={index}',
                           showlegend=True, legendgroup=f'raw{col}', hoverinfo='skip'), row=1, col=col)
        def joined(curves):
            return np.concatenate([np.vstack([p, np.full((1, 3), np.nan)]) for p in curves])
        groups = [('Annotated SWC', joined([data['paths'][a:b] for a,b in zip(offsets[:-1], offsets[1:])]), colors['dense'], 7),
                  ('K8 spline target', joined(data['curves']), colors['spline'], 4)]
        for label, points, color, width in groups:
            view.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='lines',
                           name=f'{data["label"]}: {label}', line=dict(color=color,width=width)), row=1, col=col)
        p = data['nodes']
        view.add_trace(go.Scatter3d(x=p[:,0], y=p[:,1], z=p[:,2], mode='markers',
                       name=f'{data["label"]}: graph nodes', marker=dict(color=colors['nodes'],size=3),
                       text=[f'Node {i}' for i in range(len(p))], hovertemplate='%{text}<extra></extra>'), row=1, col=col)
        # Show the eight controls of the longest annotated edge on demand.
        lengths = [np.linalg.norm(np.diff(cache['full_paths'][a:b]@cache['metric'].T, axis=0),axis=1).sum()
                   for a,b in zip(offsets[:-1], offsets[1:])]
        edge = int(np.argmax(lengths)); p = data['controls'][edge]
        view.add_trace(go.Scatter3d(x=p[:,0], y=p[:,1], z=p[:,2], mode='lines+markers',
                       visible='legendonly', name=f'{data["label"]}: 8 controls, edge {edge}',
                       line=dict(color=colors['controls'], width=3, dash='dash'),
                       marker=dict(color=colors['controls'],size=4),
                       text=[f'P{i}' for i in range(8)]), row=1, col=col)
    for scene in ['scene', 'scene2']:
        view.layout[scene].update(aspectmode='cube', xaxis=dict(title='D (voxel)',range=[-.5,63.5]),
                                 yaxis=dict(title='H (voxel)',range=[-.5,63.5]),
                                 zaxis=dict(title='W (voxel)',range=[-.5,63.5]))
    view.update_layout(template='plotly_dark',height=720,margin=dict(l=0,r=0,b=0,t=35),
                       legend=dict(orientation='h',y=-.05),paper_bgcolor='#101720')
    metadata = dict(sample=sample, subject=subject, split='train',
                    selection='First patch in the fixed representative training monitor; no predictions used.',
                    representation='Original K8 cubic B-spline, 8 controls including 2 shared endpoints',
                    graph_nodes=len(cache['nodes']), graph_edges=len(cache['edges']),
                    input_shape=list(image.shape), seed=seed, quarter_turns=list(turns),
                    rotation_planes=['DH','DW','HW'], intensity_seed=observer.seed,
                    gaussian_noise_applied=observer.noise_applied,
                    measured_noise_std=float((aug_image.detach()-observer.before).std()),
                    node_rotation_max_error_vox=rotation_error_vox,
                    independent_geometry_rotation_max_error_vox=geometry_errors,
                    pre_noise_image_exact=True, nodes_and_geometry_exact=True, adjacency_unchanged=True,
                    target_cache_equals_saved_K8=True, source_hashes_verified=checked_sources,
                    target_cache=str(cache_path), target_file=str(root/'spline'/f'{sample}.npz'),
                    target_file_sha256=sha(root/'spline'/f'{sample}.npz'), image=str(record.image),
                    image_sha256=sha(record.image), plan=str(plan_path), plan_sha256=sha(plan_path),
                    display='Identical linear grayscale window in all panels; no registration, smoothing or target refitting.',
                    limitations='MIPs collapse depth. Native image slices in the interactive view retain depth. SWC coverage is incomplete; brain masks are not vessel labels.')
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    payload = base64.b64encode(png.read_bytes()).decode()
    body = view.to_html(full_html=False,include_plotlyjs=True,config=dict(displaylogo=False,scrollZoom=True))
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BRAVA K8 — actual augmented training overlay</title><style>
body{{margin:0;background:#101720;color:#e2e8f0;font:16px system-ui}}header,details{{padding:18px 24px}}h1{{font-size:25px}}p{{line-height:1.5}}img{{width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#00d5f0}}</style>
<header><h1>BRAVA {subject}: clean and augmented training supervision</h1>
<p>Actual training patch <b>{sample}</b>: 64³ voxels, {len(cache['nodes'])} graph nodes and {len(cache['edges'])} edges.
<b>Orange:</b> annotated SWC paths. <b>Cyan:</b> saved K8 spline targets. <b>Yellow:</b> shared graph nodes.</p>
<p>The existing training loader drew {turns} quarter turns, in order DH / DW / HW. Gaussian noise applied: <b>{observer.noise_applied}</b>.
Image and target received the same rotation; pre-noise image, graph nodes, spline curves and controls passed exact replay checks. Adjacency is unchanged.
The raw MRA is the actual normalized, mean-centered model input; all panels use the same display window.</p>
<p>The top row below is clean, the bottom row augmented. These maximum-intensity projections collapse depth; use the interactive 3D image slices below to inspect depth.</p></header>
<img alt="Three orthogonal clean versus augmented image and graph overlays" src="data:image/png;base64,{payload}">
<header><p>Drag to rotate, scroll to zoom, and click legend entries to hide image planes or either graph overlay. Purple control polygons are optional: they are not vessel centerlines. P0 and P7 are graph endpoints; the six internal controls shape each cubic spline. These are supervision targets, not model predictions.</p></header>
{body}<header><p>Annotation gaps are preserved. Untraced image vessels do not imply image/graph augmentation misalignment. Brain masks are not vessel masks. No registration or refitting was performed for these displays.</p></header>
<details><summary>Exact source and verification record</summary><pre>{html.escape(json.dumps(metadata,indent=2))}</pre></details></html>'''
    (output/'brava_augmentation_overlay.html').write_text(document)
    print(json.dumps(metadata,indent=2))
    print('PNG',png)
    print('HTML',output/'brava_augmentation_overlay.html')


if __name__ == '__main__':
    main()
