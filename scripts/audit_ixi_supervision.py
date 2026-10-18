#!/usr/bin/env python3
"""Bounded validation-only image/graph audit. No inference or target modification."""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, label
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_vessel_core import records, reference
from metrics.vessel_graph import VesselGraph
from metrics.representation_geometry import sample_polyline_by_arclength


def sampled(graph):
    return np.concatenate([sample_polyline_by_arclength(c, .25) for c in graph.curves_mm]) if graph.curves_mm else np.empty((0, 3))


def support(points, raw, mask, metric):
    """Physical distance to foreground voxel centres; outside points stay outside."""
    shape = np.array(raw.shape)
    vox = points @ np.linalg.inv(metric).T * shape
    inside = np.all((vox >= -.5 - 1e-5) & (vox <= shape - .5 + 1e-5), axis=1)
    query_vox = np.clip(vox, 0, shape - 1)
    fg = np.argwhere(mask) / shape @ metric.T
    distance = cKDTree(fg).query(points)[0] if len(fg) else np.full(len(points), np.inf)
    occupancy = map_coordinates(mask.astype(float), query_vox.T, order=0, mode='nearest') * inside if len(points) else []
    intensity = map_coordinates(raw, query_vox.T, order=1, mode='nearest') if len(points) else []
    intensity = np.where(inside, intensity, np.nan)
    return dict(n=len(points), in_bounds=float(np.mean(inside)) if len(points) else None,
                inside_mask=float(np.mean(occupancy)) if len(points) else None,
                within_1mm=float(np.mean((distance <= 1) & inside)) if len(points) else None,
                median_distance_mm=float(np.median(distance)) if len(points) and len(fg) else None,
                mean_intensity=float(np.nanmean(intensity)) if np.any(inside) else None), vox


def write_view(out, name, raw, mask, gt_vox, pred_vox, gt_nodes, pred_nodes):
    lo, hi = np.quantile(raw, [.01, .995])
    scaled = np.uint8(np.clip((raw-lo)/max(hi-lo, 1e-8), 0, 1)*255)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    slices = []
    for axis in range(3):
        k = int(np.argmax(mask.sum(axis=tuple(i for i in range(3) if i != axis))))
        slices.append(k)
        a, b = [i for i in range(3) if i != axis]
        for row in range(2):
            ax = axes[row, axis]
            ax.imshow(np.take(scaled, k, axis=axis).T, origin='lower', cmap='gray', vmin=0, vmax=255)
            if row:
                m = np.take(mask, k, axis=axis).T
                if m.any() and not m.all(): ax.contour(m, levels=[.5], colors=['cyan'], linewidths=.5)
                for pts, color, size in ((gt_vox,'lime',2),(pred_vox,'magenta',2),(gt_nodes,'lime',35),(pred_nodes,'magenta',35)):
                    q = pts[np.abs(pts[:,axis]-k)<=.75]
                    if len(q): ax.scatter(q[:,a], q[:,b], s=size, c=color, marker='o' if size==2 else 'x')
            ax.set_title(f'Axis {axis}, slice {k}' + (': overlays' if row else ': raw'))
    fig.suptitle(name + ' | cyan mask; green GT; magenta prediction; crosses nodes')
    fig.tight_layout(); fig.savefig(out/(name+'.png'), dpi=120); plt.close(fig)
    return dict(name=name, shape=list(raw.shape), raw=base64.b64encode(scaled.tobytes()).decode(),
                mask=base64.b64encode(mask.astype(np.uint8).tobytes()).decode(),
                gt=gt_vox.round(2).tolist(), pred=pred_vox.round(2).tolist(),
                gn=gt_nodes.round(2).tolist(), pn=pred_nodes.round(2).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--prediction-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=16)
    a=p.parse_args()
    if a.samples<1: p.error('--samples must be positive')
    ids=sorted(x.stem for x in (a.dataset/'val/spline').glob('*.npz'))
    # One patch per filename-derived subject, chosen without inspecting predictions.
    groups={}
    for name in ids: groups.setdefault(name.rsplit('_',1)[0], []).append(name)
    rank=lambda x: hashlib.sha256(('ixi-audit-20261018:'+x).encode()).hexdigest()
    selected=sorted(min(groups[s], key=rank) for s in sorted(groups,key=rank)[:a.samples])
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[]; views=[]; rng=np.random.default_rng(181026)
    for rec in records(a.prediction_dir/'predictions.json'):
        name=rec['source_sample_id']
        if name not in selected: continue
        raw=nib.load(a.dataset/'val/raw'/f'{name}_data.nii.gz').get_fdata(dtype=np.float32)
        mask=nib.load(a.dataset/'val/seg'/f'{name}_seg.nii.gz').get_fdata(dtype=np.float32)>.5
        assert raw.shape==mask.shape and np.isfinite(raw).all() and mask.any()
        gt,metric=reference(a.dataset/'val/spline'/f'{name}.npz')
        pred=VesselGraph.from_prediction(rec,metric)
        shape=np.array(raw.shape); spacing=np.linalg.norm(metric/shape,axis=0)
        unit_axes=metric / np.linalg.norm(metric, axis=0)
        gram=unit_axes.T@unit_axes
        assert np.allclose(gram,np.eye(3),atol=1e-6), 'EDT requires orthogonal voxel axes'
        gpoints,ppoints=sampled(gt),sampled(pred)
        row=dict(sample_id=name,foreground_fraction=float(mask.mean()),foreground_intensity=float(raw[mask].mean()),background_intensity=float(raw[~mask].mean()))
        coords={}
        for key,points in dict(gt_curve=gpoints,pred_curve=ppoints,gt_node=gt.nodes_mm,pred_node=pred.nodes_mm,
                               random=rng.uniform(0,shape-1,(4096,3))/shape@metric.T).items():
            values,coords[key]=support(points,raw,mask,metric)
            row.update({key+'_'+k:v for k,v in values.items()})
        # Thickness-aware coverage is a review flag, not anatomical completeness.
        radius=distance_transform_edt(mask,sampling=spacing)
        local_radius=map_coordinates(radius,coords['gt_curve'].T,order=1,mode='nearest')
        fg_vox=np.argwhere(mask); fg_mm=fg_vox/shape@metric.T
        if len(gpoints):
            d,j=cKDTree(gpoints).query(fg_mm)
            explained=d<=local_radius[j]+max(spacing)
        else: explained=np.zeros(len(fg_vox),bool)
        row['mask_unexplained_fraction_heuristic']=float(np.mean(~explained))
        components,n=label(mask)
        labels_at_fg=components[tuple(fg_vox.T)]
        sizes=np.bincount(labels_at_fg,minlength=n+1)
        covered=np.bincount(labels_at_fg[explained],minlength=n+1)
        row['mask_components_ge8']=int(np.sum(sizes[1:]>=8))
        row['unexplained_components_ge8']=int(np.sum((sizes[1:]>=8)&(covered[1:]==0)))
        rows.append(row)
        views.append(write_view(a.output,name,raw,mask,coords['gt_curve'],coords['pred_curve'],coords['gt_node'],coords['pred_node']))
    assert sorted(r['sample_id'] for r in rows)==selected, 'Missing or duplicate predictions'
    with (a.output/'per-patch.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    relation=json.loads((a.prediction_dir/'relation-diagnostic.json').read_text())
    decomposition={}
    for t,r in relation.items():
        total=r['supported_target_edges']; recovered=r['recoverable_edges']; correct=r['correct_direct_edges']
        decomposition[t]=dict(supported_edges=total,missing_endpoint=total-recovered,
                             recovered_endpoints_missing_relation=recovered-correct,correct_direct=correct,
                             node_recall=r['node_recall'],conditional_relation_recall=r['conditional_direct_recall'])
        assert sum(decomposition[t][k] for k in ('missing_endpoint','recovered_endpoints_missing_relation','correct_direct'))==total
    result=dict(protocol='ixi-supervision-audit-v1',dataset=str(a.dataset),prediction_dir=str(a.prediction_dir),
                selected_ids=selected,sampling='SHA256 deterministic subject ranking; one ranked patch per filename-derived subject; validation only',
                physical_coordinates='cached metric; normalized coordinates times image shape; voxel-face bounds [-0.5,size-0.5]; 0.25mm curve sampling',
                means={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if k!='sample_id' and any(r[k] is not None for r in rows)},
                full_validation_decomposition=decomposition,
                caveats=['Mask support is not anatomical accuracy; no segmentation head is evaluated.',
                         'Thickness-aware unexplained foreground is heuristic, using nearest guide radius plus one largest voxel spacing.',
                         'Image means are descriptive, not a foreground classification score.',
                         'Macro means over selected patches; relation decomposition reuses full validation.',
                         'Nodes and relations are coupled; decomposition is accounting, not a causal oracle experiment.'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.output/'full-validation-relations.json').write_text(json.dumps(relation,indent=2)+'\n')
    html='''<!doctype html><meta charset="utf-8"><title>IXI supervision audit</title><style>body{background:#171717;color:white;font:16px sans-serif}canvas{width:512px;height:512px;image-rendering:pixelated}label{margin:12px}</style><h2>IXI validation: raw image and graph supervision</h2><p>Cyan: mask · green: reference · magenta: prediction · crosses: nodes. Slice overlays show points within 0.75 voxel of slice. References are uncertain. Toggle overlays to inspect raw evidence.</p><select id="sample"></select><select id="axis"><option>0</option><option>1</option><option>2</option></select><input id="slice" type="range" min="0" max="63" value="32"><span id="num"></span><br><label><input id="mask" type="checkbox" checked>Mask</label><label><input id="gt" type="checkbox" checked>GT</label><label><input id="pred" type="checkbox" checked>Prediction</label><br><canvas id="view" width="512" height="512"></canvas><script>const data=DATA;const $=id=>document.getElementById(id);data.forEach((d,i)=>{$('sample').add(new Option(d.name,i));d.r=Uint8Array.from(atob(d.raw),c=>c.charCodeAt(0));d.m=Uint8Array.from(atob(d.mask),c=>c.charCodeAt(0));});function draw(){const d=data[+$('sample').value],a=+$('axis').value,k=+$('slice').value,axes=[0,1,2].filter(v=>v!==a),b=axes[0],c=axes[1],ctx=$('view').getContext('2d');$('slice').max=d.shape[a]-1;$('num').textContent=k;const w=512/d.shape[b],h=512/d.shape[c];for(let x=0;x<d.shape[b];x++)for(let y=0;y<d.shape[c];y++){let p=[0,0,0];p[a]=k;p[b]=x;p[c]=y;let i=(p[0]*d.shape[1]+p[1])*d.shape[2]+p[2],v=d.r[i];ctx.fillStyle=$('mask').checked&&d.m[i]?`rgb(${v*.6},${Math.min(255,v*.6+100)},${Math.min(255,v*.6+100)})`:`rgb(${v},${v},${v})`;ctx.fillRect(x*w,(d.shape[c]-1-y)*h,w,h);}for(const [key,node,color,toggle] of [['gt',false,'lime','gt'],['pred',false,'magenta','pred'],['gn',true,'lime','gt'],['pn',true,'magenta','pred']]){if(!$(toggle).checked)continue;ctx.fillStyle=color;ctx.strokeStyle=color;for(const p of d[key]){if(Math.abs(p[a]-k)>.75)continue;let x=(p[b]+.5)*w,y=(d.shape[c]-.5-p[c])*h;if(node){ctx.beginPath();ctx.moveTo(x-5,y);ctx.lineTo(x+5,y);ctx.moveTo(x,y-5);ctx.lineTo(x,y+5);ctx.stroke();}else ctx.fillRect(x-1,y-1,2,2);}}}document.querySelectorAll('input,select').forEach(e=>e.oninput=draw);draw();</script>'''
    (a.output/'viewer.html').write_text(html.replace('DATA',json.dumps(views,separators=(',',':'))))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
