#!/usr/bin/env python3
"""Interactive 3D view of one TopBrain (tight-mask) ground-truth patch per modality (5 Nov 2026).

One standalone HTML per modality (plotly.js from cdnjs; surfaces computed in the browser), from the K8
transfer-spline cache: the vessel-segmentation mesh, the GT graph (nodes: junction/end vs patch-boundary;
edges as their fitted 8-control B-spline curves, or as straight chords), the dense extracted centreline the
graph was contracted from, and an adjustable image isosurface. The patch is a validation patch (never test)
whose branch count is at the 75th percentile of the validation split, so it is busy but typical.
Axes are voxel indices (D, H, W), aspect ratio from the physical spacing.
usage: topbrain_patch_3d_20261105.py [dataset_dir]                       (TopBrain, unchanged behaviour)
       topbrain_patch_3d_20261105.py --dataset tubetk [--root dataset_dir] (TubeTK Skwerer, 5 Nov 2026: the seg
       slot is the rasterized GT tree, and the page says so)
"""
import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASETS = {
    'topbrain': dict(root='/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/TopBrain_tight_20261105',
                     cache='transfer_splines_k8_20261105', out='topbrain_3d_20261105', mods=('mr', 'ct'),
                     names={'mr': 'TopBrain MRA', 'ct': 'TopBrain CTA'}, file='topbrain_{mod}_patch_3d.html',
                     note='Skull-stripped with the tight TopCoW brain mask; labels clipped to the same mask.',
                     seg_name='Vessel segmentation', seg_label='Vessel segmentation (mesh)'),
    'tubetk': dict(root='/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/TubeTK_skwerer_20261105',
                   cache='transfer_splines_k8_loops_20261105', out='tubetk_20261105', mods=('',),
                   names={'': 'TubeTK MRA (Skwerer GT)'}, file='tubetk_patch_3d.html',
                   note='Skull-stripped with desc-brain_mask; Skwerer tree clipped to the same mask. The pink surface is the '
                        'seg slot = the Skwerer tree rasterized by the TubeTK pipeline (a rendering of this GT), '
                        'NOT an independent vessel segmentation.',
                   seg_name='Seg slot (rasterized GT tree)', seg_label='Seg slot: rasterized GT tree (mesh)'),
}
SIZE = 64


def b64(volume):
    return base64.b64encode(np.ascontiguousarray(volume, dtype=np.uint8).tobytes()).decode()


def lines(paths):
    xs, ys, zs = [], [], []
    for p in paths:
        for c, a in zip((xs, ys, zs), np.asarray(p).T):
            c.extend(np.round(a, 2).tolist() + [None])
    return dict(x=xs, y=ys, z=zs)


def points(p):
    p = np.asarray(p).reshape(-1, 3)
    return dict(x=np.round(p[:, 0], 2).tolist(), y=np.round(p[:, 1], 2).tolist(), z=np.round(p[:, 2], 2).tolist())


def build(mod, ROOT, preset):
    CACHE = ROOT / preset['cache']; OUT = REPO / 'docs/research/figures' / preset['out']
    root = CACHE / mod if mod else CACHE
    rows = [r for r in csv.DictReader((root / 'patch_index.csv').open()) if r['split'] == 'val']
    rows.sort(key=lambda r: int(r['branch_count']))
    row = rows[int(0.75 * (len(rows) - 1))]
    sid = row['sample_id']
    raw = np.asanyarray(nib.load(root / 'val/raw' / f'{sid}_data.nii.gz').dataobj).astype(np.float32)
    seg = np.asanyarray(nib.load(root / 'val/seg' / f'{sid}_seg.nii.gz').dataobj) > 0
    with np.load(root / 'val/spline' / f'{sid}.npz') as z:
        nodes, edges = z['nodes'] * SIZE, z['edges'].reshape(-1, 2)
        curves = z['curves'] * SIZE
        paths = [z['full_paths'][z['path_offsets'][e]:z['path_offsets'][e + 1]] * SIZE for e in range(len(edges))]
        boundary, metric = z['boundary'].astype(bool), z['metric']
    spacing = np.linalg.norm(metric, axis=0) / SIZE
    degree = np.bincount(edges.ravel(), minlength=len(nodes)) if len(edges) else np.zeros(len(nodes), int)
    junction = (degree >= 3) & ~boundary
    end = (degree < 3) & ~boundary
    chords = [nodes[[a, b]] for a, b in edges]
    lo, hi = np.percentile(raw, [1, 99.9])
    img8 = np.clip((raw - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    on = np.concatenate(paths) if paths else np.zeros((0, 3))
    vi = np.clip(np.round(on), 0, SIZE - 1).astype(int)
    init = float(np.percentile(img8[vi[:, 0], vi[:, 1], vi[:, 2]], 25)) if len(vi) else 160.0
    name = preset['names'][mod]
    data = dict(title=name, note=preset['note'], seg_name=preset['seg_name'], seg_label=preset['seg_label'], sample=sid, patient=row['patient_id'], size=SIZE, spacing=spacing.round(4).tolist(),
                image_threshold=round(init, 1), seg=b64(seg * 255), image=b64(img8),
                curves=lines(curves), chords=lines(chords), centreline=lines(paths),
                junctions=points(nodes[junction]), ends=points(nodes[end]), boundary=points(nodes[boundary]),
                summary=dict(nodes=len(nodes), edges=len(edges), junctions=int(junction.sum()), ends=int(end.sum()),
                             boundary=int(boundary.sum()), seg_voxels=int(seg.sum()),
                             centreline_mm=round(float(sum(np.linalg.norm(np.diff(p * spacing, axis=0), axis=1).sum() for p in paths)), 1)))
    html = TEMPLATE.replace('__TITLE__', f'{name} patch 3D').replace('__DATA__', json.dumps(data, separators=(',', ':')))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / preset['file'].format(mod=mod)
    path.write_text(html)
    print('SAVED', path, f'{path.stat().st_size / 1e6:.1f} MB', sid, json.dumps(data['summary']), flush=True)


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
:root{--bg:#fafafa;--fg:#1d1d1f;--muted:#666;--panel:#fff;--line:#ddd}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#ececec;--muted:#9a9a9a;--panel:#1e1e21;--line:#333}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 0 4px}
.sub{color:var(--muted);font-size:12.5px}
.wrap{display:flex;flex-wrap:wrap}
#controls{width:280px;padding:12px 16px;box-sizing:border-box}
#plot{flex:1;min-width:320px;height:calc(100vh - 70px)}
fieldset{border:1px solid var(--line);border-radius:6px;margin:0 0 10px;padding:6px 10px;background:var(--panel)}
legend{font-size:12px;color:var(--muted)}
label{display:flex;align-items:center;gap:6px;margin:3px 0;cursor:pointer}
.sw{width:12px;height:12px;border-radius:2px;display:inline-block}
input[type=range]{width:100%}
table{border-collapse:collapse;font-size:12.5px}td{padding:1px 6px 1px 0}
</style></head><body>
<header><h1 id="title"></h1><div class="sub" id="sub"></div></header>
<div class="wrap"><div id="controls"></div><div id="plot"></div></div>
<script>
const DATA = __DATA__;
const N = DATA.size;
function decode(b){const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a;}
const X=new Float32Array(N*N*N),Y=new Float32Array(N*N*N),Z=new Float32Array(N*N*N);
for(let i=0,n=0;i<N;i++)for(let j=0;j<N;j++)for(let k=0;k<N;k++,n++){X[n]=i;Y[n]=j;Z[n]=k;}
function iso(name,vals,lo,color,opacity,visible){return {type:'isosurface',name,x:X,y:Y,z:Z,value:vals,isomin:lo,isomax:255,
  surface:{count:1},caps:{x:{show:false},y:{show:false},z:{show:false}},colorscale:[[0,color],[1,color]],showscale:false,
  opacity,visible,hoverinfo:'skip',lighting:{ambient:0.6,diffuse:0.7,specular:0.2}};}
function line(name,d,color,width,dash,visible){return {type:'scatter3d',mode:'lines',name,...d,line:{color,width,dash},visible,hoverinfo:'name'};}
function pts(name,d,color,size,symbol,visible){return {type:'scatter3d',mode:'markers',name,...d,marker:{color,size,symbol,line:{color:'#000',width:1}},visible};}
const traces=[], items=[];
function add(t,group,label,color){items.push({i:traces.length,group,label,color,on:t.visible!==false});traces.push(t);}
add(line('GT edge (B-spline)',DATA.curves,'#1f77b4',6,'solid',true),'Ground-truth graph','Edges as B-spline curves','#1f77b4');
add(line('GT edge (chord)',DATA.chords,'#7f7f7f',3,'dash',false),'Ground-truth graph','Edges as straight chords','#7f7f7f');
add(pts('Junction',DATA.junctions,'#ff9f1c',4,'circle',true),'Ground-truth graph','Junction nodes','#ff9f1c');
add(pts('End',DATA.ends,'#2ca02c',4,'diamond',true),'Ground-truth graph','End nodes','#2ca02c');
add(pts('Patch-boundary anchor',DATA.boundary,'#9467bd',3,'square',true),'Ground-truth graph','Patch-boundary anchors','#9467bd');
add(line('Extracted centreline',DATA.centreline,'#111111',2,'solid',true),'Centreline','Dense extracted centreline','#111111');
add(iso(DATA.seg_name,decode(DATA.seg),128,'#e8a0a0',0.30,true),'Volumes',DATA.seg_label,'#e8a0a0');
const imgIdx=traces.length;
add(iso('Image',decode(DATA.image),DATA.image_threshold,'#9aa4b4',0.2,false),'Volumes','Image isosurface','#9aa4b4');
const sp=DATA.spacing, ext=sp.map(s=>s*N), mx=Math.max(...ext);
const ax=(t)=>({title:t,range:[-0.5,N-0.5],showbackground:false});
const layout={margin:{l:0,r:0,t:0,b:0},showlegend:false,paper_bgcolor:'rgba(0,0,0,0)',
  font:{color:getComputedStyle(document.body).color},
  scene:{xaxis:ax('D (voxel)'),yaxis:ax('H (voxel)'),zaxis:ax('W (voxel)'),aspectmode:'manual',
         aspectratio:{x:ext[0]/mx,y:ext[1]/mx,z:ext[2]/mx}}};
Plotly.newPlot('plot',traces,layout,{responsive:true,displaylogo:false});
const s=DATA.summary;
document.getElementById('title').textContent=`${DATA.title}: ground truth on validation patch ${DATA.sample}`;
document.getElementById('sub').textContent=`${DATA.note} Voxel spacing ${sp.map(v=>v.toFixed(3)).join(' × ')} mm.`;
const c=document.getElementById('controls');
let html='';
for(const g of ['Ground-truth graph','Centreline','Volumes']){
  html+=`<fieldset><legend>${g}</legend>`;
  for(const it of items.filter(x=>x.group===g))
    html+=`<label><input type="checkbox" data-i="${it.i}" ${it.on?'checked':''}><span class="sw" style="background:${it.color}"></span>${it.label}</label>`;
  if(g==='Volumes') html+=`<div style="margin-top:6px;font-size:12px">Image threshold <span id="tv">${DATA.image_threshold}</span>
    <input type="range" id="thr" min="20" max="250" step="1" value="${DATA.image_threshold}"></div>`;
  html+='</fieldset>';
}
html+=`<fieldset><legend>This patch</legend><table>
<tr><td>Nodes / edges</td><td>${s.nodes} / ${s.edges}</td></tr>
<tr><td>Junction / end / boundary</td><td>${s.junctions} / ${s.ends} / ${s.boundary}</td></tr>
<tr><td>Centreline length</td><td>${s.centreline_mm} mm</td></tr>
<tr><td>Segmentation voxels</td><td>${s.seg_voxels}</td></tr></table>
<div class="sub" style="margin-top:6px">Drag to rotate, scroll to zoom. Nodes are the contracted graph: degree-2 points are absorbed into the B-spline edges.</div></fieldset>`;
c.innerHTML=html;
c.querySelectorAll('input[type=checkbox]').forEach(b=>b.addEventListener('change',e=>Plotly.restyle('plot',{visible:e.target.checked},[+e.target.dataset.i])));
const thr=document.getElementById('thr');
thr.addEventListener('change',e=>{document.getElementById('tv').textContent=e.target.value;Plotly.restyle('plot',{isomin:+e.target.value},[imgIdx]);});
thr.addEventListener('input',e=>{document.getElementById('tv').textContent=e.target.value;});
</script></body></html>
'''

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root_positional', nargs='?', help='dataset dir (TopBrain default usage)')
    ap.add_argument('--dataset', choices=sorted(DATASETS), default='topbrain')
    ap.add_argument('--root')
    a = ap.parse_args()
    preset = DATASETS[a.dataset]
    root = Path(a.root or a.root_positional or preset['root'])
    for m in preset['mods']:
        build(m, root, preset)
