#!/usr/bin/env python3
"""Offline 3D Brava image/SWC explorer, independent of preparation completion."""
import argparse
import base64
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes
from plotly.offline import get_plotlyjs
from scripts.prepare_brava import read_swc,swc_to_voxel,overlay
from scripts.generate_synthetic_mri_dataset import endpoint_grid_positions


def case(row,stride):
    image=nib.load(row['image']);raw=np.asarray(image.dataobj,dtype=np.float32)
    mask=np.asarray(nib.load(row['brain_mask']).dataobj)>0
    skull=np.where(mask,raw,0)
    a,edges=read_swc(row['primary_swc']);q=swc_to_voxel(a[:,2:5],row['shape'],row['spacing'],reflect_x=row.get('reflect_x',False))
    degree=np.bincount(edges.ravel(),minlength=len(a))
    down=skull[::stride,::stride,::stride]
    low,high=np.percentile(skull[mask],[1,99.5]);high=max(high,low+1.)
    pixels=np.rint(np.clip((down-low)/(high-low),0,1)*255).astype('uint8')
    vertices,faces,_,_=marching_cubes(mask.astype('float32'),.5,step_size=4)
    grid=endpoint_grid_positions(row['shape'],(64,64,64),(40,40,40))
    starts=np.array(grid)
    counts=[int(((q>=s-.5)&(q<=s+63.5)).all(1).sum()) for s in starts]
    start=starts[int(np.argmax(counts))].tolist()
    diagnostics=[]
    for flip in itertools.product([False,True],repeat=3):
        coords=np.where(flip,np.array(row['shape'])-1-q,q)
        v=map_coordinates(raw,coords.T,order=1,mode='constant',cval=0)
        b=map_coordinates(mask.astype('uint8'),coords.T,order=0,mode='constant',cval=0)
        diagnostics.append(dict(relative_flip=list(flip),median=float(np.median(v)),mean=float(v.mean()),brain_fraction=float(b.mean())))
    current=diagnostics[0];best=max(diagnostics,key=lambda r:r['median'])
    inside=float(((q>=-.5)&(q<=np.array(row['shape'])-.5)).all(1).mean())
    passed=inside>=.99 and current['brain_fraction']>=.95 and current['median']>=.95*best['median']
    data=dict(subject=row['subject'],shape=row['shape'],spacing=row['spacing'],affine=row['affine'],
        source_image=row['image'],brain_mask=row['brain_mask'],swc=row['primary_swc'],
        nodes_voxel=np.round(q,6).tolist(),swc_ids=a[:,0].astype(int).tolist(),edges=edges.tolist(),degrees=degree.tolist(),
        mesh=dict(vertices_voxel=np.round(vertices,3).tolist(),faces=faces.tolist()),
        image=dict(shape=list(pixels.shape),stride=stride,pixels=base64.b64encode(pixels.tobytes()).decode()),
        patch_grid=grid,initial_patch=start,alignment_passed=passed,inside_fraction=inside,
        alignment_diagnostics=diagnostics,
        note='Source-data preview; skull stripping computed in memory. No completed patch dataset or model predictions are implied. Reflection control changes the display only.')
    return data,skull,q


HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Brava · image and graph explorer</title><style>
body{margin:0;font:14px system-ui,sans-serif;background:#f5f7fa;color:#172a3a}header{padding:16px 22px;background:#fff;border-bottom:1px solid #d8e0e7}h1{font-size:23px;margin:0 0 5px}p{margin:5px 0;max-width:1100px;line-height:1.4}.layout{display:grid;grid-template-columns:300px 1fr;height:calc(100vh - 150px);min-height:580px}aside{padding:16px;overflow:auto;background:#fff;border-right:1px solid #d8e0e7}#plot{min-width:0;height:100%}label{display:block;margin:12px 0 5px}select,button{padding:7px;border:1px solid #bcc9d5;border-radius:5px;background:#fff}select{width:100%}input[type=range]{width:100%}input[type=checkbox]{margin-right:7px}.status{padding:10px;border-radius:5px;margin:12px 0;background:#e7f5ed}.status.bad{background:#fff0d9;color:#713c00}small{line-height:1.45;display:block;color:#526777}.buttons{display:flex;gap:8px;margin:12px 0}details{margin-top:16px}pre{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere}.warning{color:#824600;font-weight:600}.line{border-top:1px solid #e0e6ed;margin:16px 0}@media(max-width:800px){.layout{display:block;height:auto}#plot{height:70vh}aside{border-right:none}}
</style><script>__PLOTLY__</script></head><body>
<header><h1>Brava · image and graph explorer</h1><p>Rotate, zoom and inspect the provided SWC traces against the skull-stripped MRA. The transparent shell is the <b>brain mask</b>, not a vessel segmentation. The MRI preview is downsampled; graph coordinates remain at full resolution.</p><p class="warning">Preparation is incomplete: the audit failed on BG0014. This viewer uses the original source files and does not imply that training patches are ready.</p></header>
<div class="layout"><aside>
<label for="subject">Subject</label><select id="subject"></select><div id="status" class="status"></div>
<label><input type="checkbox" id="brain" checked>Brain-mask surface</label>
<label><input type="checkbox" id="slices" checked>MRI slice planes</label>
<label><input type="checkbox" id="lines" checked>SWC graph lines</label>
<label><input type="checkbox" id="anchors" checked>Junctions and endpoints</label>
<label><input type="checkbox" id="allnodes">All SWC samples</label>
<label><input type="checkbox" id="box" checked>Candidate 64³ patch</label>
<div class="line"></div><b>Move MRI slices (array voxel axes)</b><div id="sliders"></div>
<div class="buttons"><button id="full">Whole brain</button><button id="focus">Focus patch</button></div>
<label><input type="checkbox" id="mirror">Diagnostic left–right reflection of traces</label>
<small>This reflection changes only the display, not the image, source SWC or dataset. BG0014's reflected traces have stronger image support, but this is not an accepted alignment correction.</small>
<p id="patchinfo"></p><small>The yellow box follows the nearest location on IXI's endpoint-distributed 64³ / maximum-stride-40 grid. It is a candidate patch, not an exported training sample.</small>
<details><summary>Sources and alignment measurements</summary><pre id="metadata"></pre></details>
</aside><div id="plot"></div></div>
<script>
const cases=__DATA__, subject=document.getElementById('subject'),plot=document.getElementById('plot');
let current=null,volume=null,coordinates=[],activePatch=[],focus=false,creating=false;
const el=id=>document.getElementById(id),on=id=>el(id).checked;
const world=(p)=>current.affine.slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);
function lineTrace(nodes,edges,name,color,width){let x=[],y=[],z=[];for(const [a,b] of edges){for(const i of [a,b]){const p=nodes[i];x.push(p[0]);y.push(p[1]);z.push(p[2]);}x.push(null);y.push(null);z.push(null);}return {type:'scatter3d',mode:'lines',x,y,z,name,line:{color,width},hoverinfo:'skip'};}
function markerTrace(ids,name,color,size){let pts=ids.map(i=>coordinates[i]);return {type:'scatter3d',mode:'markers',x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),z:pts.map(p=>p[2]),name,marker:{size,color},customdata:ids.map(i=>[current.swc_ids[i],current.degrees[i]]),hovertemplate:'SWC ID %{customdata[0]} · degree %{customdata[1]}<br>%{x:.2f}, %{y:.2f}, %{z:.2f} mm<extra>'+name+'</extra>'};}
function slices(){let out=[],sz=current.image.shape,stride=current.image.stride;
 for(let axis=0;axis<3;axis++){let other=[0,1,2].filter(a=>a!==axis),idx=Math.min(sz[axis]-1,Math.round(+el('slice'+axis).value/stride)),x=[],y=[],z=[],surfacecolor=[];
  for(let i=0;i<sz[other[0]];i++){let xx=[],yy=[],zz=[],vv=[];for(let j=0;j<sz[other[1]];j++){let p=[0,0,0];p[axis]=idx;p[other[0]]=i;p[other[1]]=j;let w=world(p.map(v=>v*stride));xx.push(w[0]);yy.push(w[1]);zz.push(w[2]);vv.push(volume[(p[0]*sz[1]+p[1])*sz[2]+p[2]]);}x.push(xx);y.push(yy);z.push(zz);surfacecolor.push(vv);}
  out.push({type:'surface',x,y,z,surfacecolor,cmin:0,cmax:255,colorscale:'Gray',showscale:false,opacity:.68,name:'MRI axis '+axis,showlegend:true,visible:on('slices'),hovertemplate:'MRI slice · physical mm<br>%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>'});
 }return out;}
function cube(){let s=activePatch.map(v=>v-.5),p=[];for(let i=0;i<8;i++)p.push(world(s.map((v,a)=>v+((i>>a)&1)*64)));let edges=[];for(let i=0;i<8;i++)for(let a=0;a<3;a++)if(!((i>>a)&1))edges.push([i,i|(1<<a)]);return {points:p,trace:lineTrace(p,edges,'Candidate 64³ patch','#e6b525',5)};}
function patchAtSlices(){let center=[0,1,2].map(a=>+el('slice'+a).value);activePatch=current.patch_grid.reduce((best,p)=>{let d=p.reduce((sum,v,a)=>sum+(v+31.5-center[a])**2,0);return !best||d<best.d?{p,d}:best;},null).p;el('patchinfo').textContent='Patch start (array voxels): ['+activePatch.join(', ')+']';}
function draw(reset=false){if(!current||creating)return;patchAtSlices();let q=current.nodes_voxel.map(p=>on('mirror')?[current.shape[0]-1-p[0],p[1],p[2]]:p);coordinates=q.map(world);
 let v=current.mesh.vertices_voxel.map(world),f=current.mesh.faces;
 let traces=[{type:'mesh3d',x:v.map(p=>p[0]),y:v.map(p=>p[1]),z:v.map(p=>p[2]),i:f.map(t=>t[0]),j:f.map(t=>t[1]),k:f.map(t=>t[2]),name:'Brain mask (not vessels)',color:'#a4b8cc',opacity:.10,hoverinfo:'skip',visible:on('brain'),showlegend:true}];
 traces.push(...slices());let lines=lineTrace(coordinates,current.edges,'SWC centerlines','#0bbfa4',4);lines.visible=on('lines');traces.push(lines);
 for(const [test,name,color,size] of [[d=>d>=3,'Junctions','#ff9b32',4],[d=>d===1,'Endpoints','#4796ff',3],[d=>d===0,'Isolated samples','#ed5db4',4]]){let ids=current.degrees.map((d,i)=>test(d)?i:-1).filter(i=>i>=0);if(ids.length){let t=markerTrace(ids,name,color,size);t.visible=on('anchors');traces.push(t);}}
 let all=markerTrace(current.swc_ids.map((_,i)=>i),'All SWC samples','#d5f7f0',2);all.visible=on('allnodes');traces.push(all);let c=cube();c.trace.visible=on('box');traces.push(c.trace);
 let scene={aspectmode:'data',xaxis:{title:'World x (mm)'},yaxis:{title:'World y (mm)'},zaxis:{title:'World z (mm)'},bgcolor:'#172231'};
 if(focus){for(let [a,key] of ['xaxis','yaxis','zaxis'].entries()){let vals=c.points.map(p=>p[a]);scene[key].range=[Math.min(...vals)-2,Math.max(...vals)+2];}}
 let layout={title:{text:current.subject+' · '+current.nodes_voxel.length+' SWC samples / '+current.edges.length+' connections'+(on('mirror')?' · DIAGNOSTIC REFLECTION':''),font:{size:17,color:'#eaf1f8'}},scene,margin:{l:0,r:0,b:0,t:50},paper_bgcolor:'#172231',font:{color:'#dae7f3'},legend:{x:0,y:1,font:{size:11}},uirevision:current.subject+(reset?Date.now():'')};
 Plotly.react(plot,traces,layout,{responsive:true,displaylogo:false,scrollZoom:true});}
function choose(){creating=true;current=cases.find(c=>c.subject===subject.value);focus=false;el('mirror').checked=false;volume=Uint8Array.from(atob(current.image.pixels),c=>c.charCodeAt(0));el('sliders').innerHTML='';
 for(let axis=0;axis<3;axis++){let label=document.createElement('label');label.innerHTML='Axis '+axis+': <span id="value'+axis+'"></span>';let slider=document.createElement('input');slider.type='range';slider.id='slice'+axis;slider.min=0;slider.max=current.shape[axis]-1;slider.step=current.image.stride;slider.value=current.initial_patch[axis]+31.5;slider.oninput=()=>{el('value'+axis).textContent=slider.value;};slider.onchange=()=>draw();label.appendChild(slider);el('sliders').appendChild(label);el('value'+axis).textContent=slider.value;}
 let pass=current.alignment_passed;el('status').className='status'+(pass?'':' bad');el('status').textContent=pass?'Source alignment checks pass for this example.':'Alignment audit FAILS for this example. Inspect before using as training GT.';
 el('metadata').textContent=JSON.stringify({image:current.source_image,brain_mask:current.brain_mask,SWC:current.swc,shape:current.shape,spacing_mm:current.spacing,SWC_units_per_voxel:.62,inside_fraction:current.inside_fraction,display_stride:current.image.stride,alignment:current.alignment_diagnostics,note:current.note},null,2);creating=false;draw(true);}
for(const c of cases){let o=document.createElement('option');o.value=c.subject;o.textContent=c.subject+(c.alignment_passed?' · passing case':' · alignment failure');subject.appendChild(o);}subject.onchange=choose;
for(const id of ['brain','slices','lines','anchors','allnodes','box','mirror'])el(id).onchange=()=>draw();el('full').onclick=()=>{focus=false;draw(true);};el('focus').onclick=()=>{focus=true;draw(true);};choose();
</script></body></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--subjects',nargs='+',default=['BG0001','BG0006','BG0014']);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=3)
    a=p.parse_args();rows=json.loads((a.preparation/'inventory.json').read_text())['subjects'];lookup={r['subject']:r for r in rows}
    payload=[];provenance=[];a.output.parent.mkdir(parents=True,exist_ok=True)
    for subject in a.subjects:
        data,skull,q=case(lookup[subject],a.stride);payload.append(data)
        provenance.append({k:v for k,v in data.items() if k not in ['nodes_voxel','swc_ids','edges','degrees','mesh','image','patch_grid']})
        overlay(a.output.with_name(subject+'-source-preview.png'),skull,q)
        if not data['alignment_passed']:
            reflected=q.copy();reflected[:,0]=data['shape'][0]-1-reflected[:,0]
            overlay(a.output.with_name(subject+'-diagnostic-x-reflection.png'),skull,reflected)
        print(subject,'alignment_passed',data['alignment_passed'],flush=True)
    document=HTML.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(payload,separators=(',',':')))
    a.output.write_text(document);a.output.with_suffix('.json').write_text(json.dumps(dict(preparation=str(a.preparation),subjects=provenance),indent=2)+'\n')
    print(a.output,'bytes',a.output.stat().st_size)


if __name__=='__main__':main()
