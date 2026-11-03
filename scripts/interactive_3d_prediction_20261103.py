#!/usr/bin/env python3
"""Interactive 3D view of one traced_adaptive validation prediction per dataset (3 Nov 2026).

Same patches as docs/research/figures/unet_outputs_20261102/unet_outputs_3d.png. One standalone
HTML per dataset (plotly.js from cdnjs; surfaces are computed in the browser):
  - predicted graph at the train-selected thresholds: accepted edges as predicted curves,
    blue = correct (both endpoints matched to the endpoints of a GT edge, 2 mm), red = false;
    detected nodes orange = matched, red = spurious
  - GT graph: dense GT paths (green; dashed = missed), GT nodes
  - vessel segmentation surface (SyntheticMRI, IXI); BRAVA has no vessel segmentation (its seg
    slot is the brain mask), so only the image surface is shown there
  - image isosurface with an adjustable threshold; predicted centreline p = 0.5 surface
Axes are voxel indices (D, H, W) with the aspect ratio set from the physical spacing.
"""
import base64
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from node_heatmap_probe_20261030 import make_dataset, SIZE  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402
from visualize_unet_outputs_20261102 import representative  # noqa: E402

D = T.D
RUN = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
OUT = REPO / 'docs/research/figures/interactive_3d_20261103'
VESSEL_SEG = {'SyntheticMRI': True, 'IXI': True, 'BRAVA': False}


def b64(volume):
    return base64.b64encode(np.ascontiguousarray(volume, dtype=np.uint8).tobytes()).decode()


def lines(paths):
    """List of (N,3) voxel paths -> x, y, z lists with None separators."""
    xs, ys, zs = [], [], []
    for p in paths:
        for c, a in zip((xs, ys, zs), p.T):
            c.extend(np.round(a, 2).tolist() + [None])
    return dict(x=xs, y=ys, z=zs)


def points(p):
    return dict(x=np.round(p[:, 0], 2).tolist(), y=np.round(p[:, 1], 2).tolist(), z=np.round(p[:, 2], 2).tolist())


def build(d):
    res = json.loads((RUN / d / 'traced_adaptive/final.json').read_text())
    nt, et = res['thresholds']['node'], res['thresholds']['edge']
    rows = pickle.loads((RUN / d / 'traced_adaptive/final-validation-rows.pkl').read_bytes())
    sid = representative(rows, nt, et, res['validation']['edge_f1'])
    r = next(x for x in rows if x['sample_id'] == sid)
    del rows
    s = make_dataset(d, [sid], augment=False)[0]
    model = T.TracedDenseGraph().eval()
    model.load_state_dict(torch.load(RUN / d / 'traced_adaptive/model.pt', map_location='cpu')['model'])
    with torch.no_grad():
        prob = model.forward_image(s[0][0].float()[None])[0, 4].float().sigmoid().numpy()
    image = s[0][0][0].numpy()
    seg = s[1][0][0].numpy() + 0.5
    with np.load(r['cache']) as z:
        M, gn, ge = z['metric'], z['nodes'] * SIZE, z['edges'].reshape(-1, 2)
        gp = [z['full_paths'][z['path_offsets'][e]:z['path_offsets'][e + 1]] * SIZE for e in range(len(ge))]
    spacing = np.linalg.norm(M, axis=0) / SIZE  # mm per voxel along D, H, W
    pv = r['nodes'] * SIZE
    m = D.match(torch.as_tensor(pv @ M.T / SIZE), torch.as_tensor(gn @ M.T / SIZE))
    gt_set = {tuple(sorted(map(int, e))) for e in ge}
    acc = [] if r['edge_scores'] is None else np.nonzero(r['edge_scores'] >= et)[0]
    tp, fp, hit = [], [], set()
    for j in acc:
        u, v = map(int, r['pairs'][j])
        curve = r['curves'][j] * SIZE
        key = tuple(sorted((m[u], m[v]))) if u in m and v in m else None
        if key in gt_set and key not in hit:
            tp.append(curve)
            hit.add(key)
        else:
            fp.append(curve)
    found_gt = [tuple(sorted(map(int, e))) in hit for e in ge]
    matched = np.array([i in m for i in range(len(pv))], bool)
    n_tp, n_fp, n_fn = len(tp), len(fp), len(ge) - len(hit)
    edge_f1 = 2 * n_tp / max(1, 2 * n_tp + n_fp + n_fn)
    node_tp = int(matched.sum())
    node_f1 = 2 * node_tp / max(1, len(pv) + len(gn))
    # display volumes: image rescaled on foreground percentiles; vessel threshold initialised
    # at the 25th percentile of the image on the GT centreline
    fg = image[image > image.min()]
    lo, hi = np.percentile(fg, [1, 99.9]) if len(fg) else (image.min(), image.max())
    img8 = np.clip((image - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    on = np.concatenate(gp) if gp else np.zeros((0, 3))
    vi = np.clip(np.round(on), 0, SIZE - 1).astype(int)
    init = float(np.percentile(img8[vi[:, 0], vi[:, 1], vi[:, 2]], 25)) if len(vi) else 160.0
    data = dict(
        dataset=d, sample=sid, size=SIZE, spacing=spacing.round(4).tolist(), vessel_seg=VESSEL_SEG[d],
        thresholds=dict(node=nt, edge=et), image_threshold=round(init, 1),
        summary=dict(gt_nodes=len(gn), gt_edges=len(ge), detected=len(pv), matched=node_tp, accepted_edges=len(acc),
                     correct_edges=n_tp, false_edges=n_fp, missed_edges=n_fn, node_f1=round(node_f1, 3), edge_f1=round(edge_f1, 3)),
        image=b64(img8), prob=b64(np.clip(prob * 255, 0, 255)), seg=b64((seg > 0.5) * 255) if VESSEL_SEG[d] else None,
        gt_found=lines([p for p, f in zip(gp, found_gt) if f]), gt_missed=lines([p for p, f in zip(gp, found_gt) if not f]),
        gt_nodes=points(gn), pred_tp=lines(tp), pred_fp=lines(fp),
        pred_matched=points(pv[matched]), pred_spurious=points(pv[~matched]))
    html = TEMPLATE.replace('__TITLE__', f'{d} prediction 3D').replace('__DATA__', json.dumps(data, separators=(',', ':')))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'prediction_3d_{d}.html'
    path.write_text(html)
    print('SAVED', path, 'seg voxels', int((seg > 0.5).sum()) if VESSEL_SEG[d] else None, f'{path.stat().st_size / 1e6:.1f} MB', json.dumps(data['summary']), flush=True)


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
#controls{width:270px;padding:12px 16px;box-sizing:border-box}
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
add(line('GT edge (recovered)',DATA.gt_found,'#2ca02c',5,'solid',true),'Ground truth','GT edges, recovered','#2ca02c');
add(line('GT edge (missed)',DATA.gt_missed,'#2ca02c',4,'dash',true),'Ground truth','GT edges, missed (dashed)','#2ca02c');
add(pts('GT node',DATA.gt_nodes,'#2ca02c',3,'circle',true),'Ground truth','GT nodes','#2ca02c');
add(line('Predicted edge, correct',DATA.pred_tp,'#1f77b4',6,'solid',true),'Prediction','Edges, correct','#1f77b4');
add(line('Predicted edge, false',DATA.pred_fp,'#d62728',6,'solid',true),'Prediction','Edges, false','#d62728');
add(pts('Detected node, matched',DATA.pred_matched,'#ff9f1c',4,'diamond',true),'Prediction','Nodes, matched','#ff9f1c');
add(pts('Detected node, spurious',DATA.pred_spurious,'#d62728',4,'square',true),'Prediction','Nodes, spurious','#d62728');
if(DATA.seg) add(iso('Vessel segmentation',decode(DATA.seg),128,'#e8a0a0',0.30,true),'Volumes','Vessel segmentation (mesh)','#e8a0a0');
const imgIdx=traces.length;
add(iso('Image',decode(DATA.image),DATA.image_threshold,'#9aa4b4',0.22,!DATA.seg),'Volumes','Image isosurface','#9aa4b4');
add(iso('Predicted centreline p=0.5',decode(DATA.prob),128,'#3fb6c9',0.25,false),'Volumes','Predicted centreline (p = 0.5)','#3fb6c9');
const sp=DATA.spacing, ext=sp.map(s=>s*N), mx=Math.max(...ext);
const ax=(t)=>({title:t,range:[-0.5,N-0.5],showbackground:false});
const layout={margin:{l:0,r:0,t:0,b:0},showlegend:false,paper_bgcolor:'rgba(0,0,0,0)',
  font:{color:getComputedStyle(document.body).color},
  scene:{xaxis:ax('D (voxel)'),yaxis:ax('H (voxel)'),zaxis:ax('W (voxel)'),aspectmode:'manual',
         aspectratio:{x:ext[0]/mx,y:ext[1]/mx,z:ext[2]/mx}}};
Plotly.newPlot('plot',traces,layout,{responsive:true,displaylogo:false});
const s=DATA.summary;
document.getElementById('title').textContent=`${DATA.dataset}: dense + traced prediction on validation patch ${DATA.sample}`;
document.getElementById('sub').textContent=`Thresholds (train-selected): node ${DATA.thresholds.node}, edge ${DATA.thresholds.edge}. Voxel spacing ${sp.map(v=>v.toFixed(3)).join(' × ')} mm.`+
  (DATA.vessel_seg?'':' BRAVA has no vessel segmentation (manual SWC traces only): image surface + GT graph shown instead.');
const c=document.getElementById('controls');
let html='';
for(const g of ['Prediction','Ground truth','Volumes']){
  html+=`<fieldset><legend>${g}</legend>`;
  for(const it of items.filter(x=>x.group===g))
    html+=`<label><input type="checkbox" data-i="${it.i}" ${it.on?'checked':''}><span class="sw" style="background:${it.color}"></span>${it.label}</label>`;
  if(g==='Volumes') html+=`<div style="margin-top:6px;font-size:12px">Image threshold <span id="tv">${DATA.image_threshold}</span>
    <input type="range" id="thr" min="20" max="250" step="1" value="${DATA.image_threshold}"></div>`;
  html+='</fieldset>';
}
html+=`<fieldset><legend>This patch</legend><table>
<tr><td>GT nodes / edges</td><td>${s.gt_nodes} / ${s.gt_edges}</td></tr>
<tr><td>Detected nodes (matched)</td><td>${s.detected} (${s.matched})</td></tr>
<tr><td>Accepted edges</td><td>${s.accepted_edges}</td></tr>
<tr><td>Correct / false / missed</td><td>${s.correct_edges} / ${s.false_edges} / ${s.missed_edges}</td></tr>
<tr><td>Node F1 / edge F1</td><td>${s.node_f1} / ${s.edge_f1}</td></tr></table>
<div class="sub" style="margin-top:6px">Drag to rotate, scroll to zoom. Edge correct = both endpoints matched (2 mm) to the endpoints of one GT edge.</div></fieldset>`;
c.innerHTML=html;
c.querySelectorAll('input[type=checkbox]').forEach(b=>b.addEventListener('change',e=>Plotly.restyle('plot',{visible:e.target.checked},[+e.target.dataset.i])));
const thr=document.getElementById('thr');
thr.addEventListener('change',e=>{document.getElementById('tv').textContent=e.target.value;Plotly.restyle('plot',{isomin:+e.target.value},[imgIdx]);});
thr.addEventListener('input',e=>{document.getElementById('tv').textContent=e.target.value;});
</script></body></html>
'''


def main():
    torch.set_num_threads(8)
    for d in ['SyntheticMRI', 'IXI', 'BRAVA']:
        build(d)


if __name__ == '__main__':
    main()
