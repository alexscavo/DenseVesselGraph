#!/usr/bin/env python3
"""Portable, linked-camera 3D views of existing GT and final IXI predictions."""
import copy
import hashlib
import html
import json
from pathlib import Path

from medpy.io import load
import numpy as np
import plotly.graph_objects as go

WORK = Path(__file__).resolve().parents[1]
MAIN = WORK.parent / 'Graph-Native-Betti-Matching-Representation'
OUT = MAIN / 'docs/research/evidence/ixi_local_components_20261028'
SAMPLE = 'sample_000001_0009'
CAMERA = dict(eye=dict(x=1.45, y=1.45, z=1.1), up=dict(x=0, y=0, z=1),
              center=dict(x=0, y=0, z=0))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def joined(curves):
    return np.concatenate([np.vstack([p, np.full((1, 3), np.nan)]) for p in curves])


def graph(figure, nodes, curves, scene, name, color, *, scores=None,
          queries=None, overlay=False, showlegend=True):
    visible = False if overlay else True
    group = 'overlay' if overlay else ('gt' if name == 'Ground truth' else 'prediction')
    points = joined(curves)
    curve_index = len(figure.data)
    figure.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='lines',
        line=dict(color=color, width=3 if overlay else 6,
                  dash='dot' if overlay else 'solid'), opacity=.55 if overlay else 1,
        name=name + ' edges', legendgroup=group, showlegend=showlegend and not overlay,
        scene=scene, visible=visible, hoverinfo='skip', connectgaps=False))
    text = [f'GT node {i}' for i in range(len(nodes))] if scores is None else [
        f'Query {int(q)}<br>confidence {float(s):.4f}' for q, s in zip(queries, scores)]
    node_index = len(figure.data)
    figure.add_trace(go.Scatter3d(
        x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode='markers',
        marker=dict(color=color, size=3.2 if overlay else 4.4, opacity=.45 if overlay else 1,
                    line=dict(color='#233545', width=.5)),
        name=name + ' nodes', text=text, legendgroup=group, showlegend=False,
        hovertemplate='%{text}<br>X %{x:.2f}, Y %{y:.2f}, Z %{z:.2f} mm<extra></extra>',
        scene=scene, visible=visible))
    return [curve_index, node_index]


def build(panels, volume, metric, center, bounds, gt, path):
    shape = np.array(volume.shape)
    physical = lambda p: np.asarray(p) @ metric.T - center
    n = len(panels)
    scene_names = ['scene' if i == 0 else f'scene{i+1}' for i in range(n)]
    figure = go.Figure()
    mri_indices, overlay_indices = [], []
    low, high = np.quantile(volume, [.03, .995])
    assert high > low
    layout = dict(margin=dict(l=5, r=5, t=52, b=5), height=660,
                  paper_bgcolor='white', plot_bgcolor='white',
                  legend=dict(orientation='h', x=.5, xanchor='center', y=1.1,
                              groupclick='togglegroup'), annotations=[])
    gap = .012
    width = (1 - gap * (n - 1)) / n
    for i, (panel, scene) in enumerate(zip(panels, scene_names)):
        x0, x1 = i * (width + gap), i * (width + gap) + width
        layout[scene] = dict(domain=dict(x=[x0, x1], y=[0, 1]), aspectmode='data',
            camera=copy.deepcopy(CAMERA), dragmode='orbit', bgcolor='#f8fafc',
            xaxis=dict(title='X (mm)', range=bounds[0], backgroundcolor='#f8fafc'),
            yaxis=dict(title='Y (mm)', range=bounds[1], backgroundcolor='#f8fafc'),
            zaxis=dict(title='Z (mm)', range=bounds[2], backgroundcolor='#f8fafc'))
        layout['annotations'].append(dict(x=(x0+x1)/2, y=1.035, xref='paper', yref='paper',
            text=f"<b>{panel['label']}</b><br>{len(panel['nodes'])} nodes, {len(panel['curves'])} edges",
            showarrow=False, font=dict(size=15, color='#172b3a')))
        for axis in range(3):
            other = [j for j in range(3) if j != axis]
            u, v = np.meshgrid(np.arange(shape[other[0]]), np.arange(shape[other[1]]), indexing='ij')
            voxel = np.empty(u.shape + (3,), dtype=float)
            voxel[..., axis] = shape[axis] // 2
            voxel[..., other[0]] = u
            voxel[..., other[1]] = v
            xyz = physical(voxel / shape)
            mri_indices.append(len(figure.data))
            figure.add_trace(go.Surface(
                x=xyz[..., 0], y=xyz[..., 1], z=xyz[..., 2],
                surfacecolor=np.take(volume, shape[axis] // 2, axis=axis),
                colorscale='Gray', cmin=float(low), cmax=float(high), showscale=False,
                opacity=.18, lighting=dict(ambient=1, diffuse=0, specular=0),
                name='Raw MRI slices', showlegend=False, scene=scene, hoverinfo='skip'))
        graph(figure, physical(panel['nodes']), [physical(p) for p in panel['curves']],
              scene, 'Ground truth' if panel['is_gt'] else 'Prediction',
              '#f4a52c' if panel['is_gt'] else '#00bfae',
              scores=panel.get('scores'), queries=panel.get('queries'),
              showlegend=panel['is_gt'] or i == 1)
        if not panel['is_gt']:
            overlay_indices.extend(graph(figure, physical(gt['nodes']),
                [physical(p) for p in gt['curves']], scene, 'GT overlay', '#f4a52c',
                overlay=True, showlegend=False))
    figure.update_layout(**layout)
    assert len(mri_indices) == 3 * len(panels)
    figure.to_json()  # Validate the complete Plotly schema before writing HTML.
    script = r'''
const gd = document.getElementById('{plot_id}');
const scenes = __SCENES__;
const mri = __MRI__;
const overlays = __OVERLAYS__;
const initialCamera = __CAMERA__;
const linked = document.getElementById('linked');
let busy = false;
let lastScene = scenes[0];
function changeCameras(updates) {
  busy = true;
  Promise.resolve(Plotly.relayout(gd, updates)).finally(() => { busy = false; });
}
gd.on('plotly_relayout', (event) => {
  if (busy) return;
  const source = scenes.find(s => event[s + '.camera']);
  if (!source) return;
  lastScene = source;
  if (!linked.checked) return;
  const updates = {};
  scenes.forEach(s => { if (s !== source) updates[s + '.camera'] = event[source + '.camera']; });
  changeCameras(updates);
});
document.getElementById('mri').addEventListener('change', function() {
  Plotly.restyle(gd, {visible: this.checked}, mri);
});
document.getElementById('overlay').addEventListener('change', function() {
  Plotly.restyle(gd, {visible: this.checked}, overlays);
});
document.getElementById('reset').addEventListener('click', () => {
  const updates = {};
  scenes.forEach(s => { updates[s + '.camera'] = initialCamera; });
  changeCameras(updates);
});
linked.addEventListener('change', function() {
  if (!this.checked) return;
  const updates = {};
  scenes.forEach(s => { updates[s + '.camera'] = gd.layout[lastScene].camera; });
  changeCameras(updates);
});
'''
    for key, value in [('__SCENES__', scene_names), ('__MRI__', mri_indices),
                       ('__OVERLAYS__', overlay_indices), ('__CAMERA__', CAMERA)]:
        script = script.replace(key, json.dumps(value))
    title = 'GT + ' + ' + '.join(p['label'] for p in panels if not p['is_gt'])
    view = figure.to_html(full_html=False, include_plotlyjs=True, div_id='ixi3d',
        config=dict(responsive=True, scrollZoom=True, displaylogo=False), post_script=script)
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ — IXI final predictions</title>
<style>body{margin:0;padding:12px 16px;background:white;color:#172b3a;font:15px system-ui,sans-serif}
h1{font-size:21px;margin:0 0 6px}p{margin:5px 0;color:#42576a}.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin:14px 0 4px}
label{cursor:pointer}button{border:1px solid #aabcc9;border-radius:5px;background:#f4f8fa;padding:6px 12px;cursor:pointer}
.plotwrap{min-width:780px}footer{font-size:13px;color:#576d7f;padding-top:6px}
.orange{color:#b87300}.cyan{color:#008e82}</style></head><body>
<h1>__TITLE__</h1><p>IXI_000001 · sample_000001_0009 · validation · final 5,500-update checkpoints</p>
<p><b class="orange">Orange: GT graph / dense reference paths.</b> <b class="cyan">Cyan: saved predicted nodes and K8 curves.</b></p>
<div class="controls"><label><input id="linked" type="checkbox" checked> Link rotations and zoom</label>
<label><input id="mri" type="checkbox" checked> Raw MRI slices</label>
<label><input id="overlay" type="checkbox"> Overlay GT on predictions</label>
<button id="reset" type="button">Reset view</button></div><div class="plotwrap">__VIEW__</div>
<footer>Drag to rotate; scroll to zoom; hover over nodes for IDs/confidence. All panels share the same physical scale and patch-centred millimetre coordinates. MRI slices use one shared display window. The HTML works offline.</footer>
</body></html>'''.replace('__TITLE__', html.escape(title)).replace('__VIEW__', view)
    path.write_text(page)
    return dict(file=str(path), sha256=sha(path), panels=[p['label'] for p in panels],
                scene_names=scene_names, trace_count=len(figure.data),
                mri_traces=mri_indices, gt_overlay_traces=overlay_indices,
                interactive_script=script.replace('{plot_id}', 'ixi3d'))


def main():
    source = OUT / f'{SAMPLE}-provenance.json'
    prior = json.loads(source.read_text())
    assert prior['checkpoint_step'] == 5500 and prior['split'] == 'val'
    assert sha(prior['input_image']) == prior['image_sha256']
    assert sha(prior['target']) == prior['target_sha256']
    raw, _ = load(prior['input_image'])
    volume = np.asarray(raw, dtype=np.float32) - prior['foreground_mean_subtracted']
    assert volume.shape == (64, 64, 64)
    with np.load(prior['target'], allow_pickle=False) as target:
        metric = target['metric'].copy()
        curves = [target['full_paths'][a:b].copy() for a, b in
                  zip(target['path_offsets'][:-1], target['path_offsets'][1:])]
        gt = dict(label='Ground truth', nodes=target['nodes'].copy(), curves=curves, is_gt=True)
    panels = [gt]
    for arm in ('A-xyz-fresh', 'A-cls'):
        p = prior['arms'][arm]
        assert sha(p['predictions']) == p['predictions_sha256']
        record = p['saved_graph']
        nodes = np.asarray(record['nodes_dhw'])
        edges = np.asarray(record['edges'], dtype=int)
        curves = np.asarray(record['spline_curves'])
        assert np.allclose(metric, record['spline_metric'])
        assert np.allclose(curves[:, [0, -1]], nodes[edges], atol=1e-6)
        panels.append(dict(label=arm, nodes=nodes, curves=curves, is_gt=False,
                           scores=record['node_scores'], queries=record['query_ids']))
    center = ((np.array(volume.shape)-1)/2/np.array(volume.shape)) @ metric.T
    cloud = np.concatenate([np.asarray(p['nodes']) @ metric.T - center for p in panels] +
                           [np.concatenate(p['curves']) @ metric.T - center for p in panels])
    corners = np.array(np.meshgrid(*[[-.5/64, 63.5/64]]*3)).reshape(3, -1).T @ metric.T - center
    cloud = np.concatenate((cloud, corners))
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    pad = np.maximum((high-low)*.025, .4)
    bounds = [[float(a), float(b)] for a, b in zip(low-pad, high+pad)]
    outputs = [build(panels, volume, metric, center, bounds, gt,
                     OUT / f'{SAMPLE}-3d-comparison.html')]
    for panel in panels[1:]:
        outputs.append(build([gt, panel], volume, metric, center, bounds, gt,
                             OUT / f'{SAMPLE}-{panel["label"]}-3d.html'))
    metadata = dict(sample=SAMPLE, subject=prior['subject'], split='val', checkpoint_step=5500,
        source_preview_provenance=str(source), source_sha256=sha(source),
        renderer=str(Path(__file__).resolve()), renderer_sha256=sha(Path(__file__).resolve()),
        coordinate_transform='normalized DHW @ metric.T minus common physical patch center',
        metric=metric.tolist(), center_mm=center.tolist(), identical_scene_bounds_mm=bounds,
        graph_counts={p['label']:dict(nodes=len(p['nodes']), edges=len(p['curves'])) for p in panels},
        no_new_inference_or_training=True, self_contained=True, outputs=outputs)
    (OUT / f'{SAMPLE}-3d-provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps([dict(file=p['file'], bytes=Path(p['file']).stat().st_size) for p in outputs], indent=2))


if __name__ == '__main__':
    main()
