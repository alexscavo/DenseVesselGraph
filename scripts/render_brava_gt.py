"""Standalone graph-only viewer of validated BRAVA development annotations."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import plotly.graph_objects as go
from scripts.audit_synthetic_mri_grid import SourceGraph


def render(prepared,subject,output):
    sources=list(csv.DictReader((prepared/'sources/source_manifest.csv').open()))
    row=next(r for r in sources if r['subject']==subject)
    assert row['patch_split']!='test','Use a development subject; keep the test set closed'
    marker=json.loads((prepared/'volumes'/subject/'complete.json').read_text())
    assert marker['completed']
    base=prepared/'volumes'/subject/'graphs'/marker['primary_variant']
    fig=go.Figure();stats=[];groups=[]
    for name in ['junction_only','adaptive','dense']:
        g=SourceGraph.from_directory(base/name);points=g.node_positions
        lookup={int(v):i for i,v in enumerate(g.node_ids)}
        edges=np.array([[lookup[a],lookup[b]] for a,b in g.edges],int).reshape(-1,2)
        degree=np.bincount(edges.ravel(),minlength=len(points))
        # Draw original annotated polylines, never chords across sparse anchors.
        curves=np.concatenate([np.vstack([line,np.full((1,3),np.nan)]) for line in g.edge_polylines])
        start=len(fig.data);visible=name=='junction_only'
        fig.add_trace(go.Scatter3d(x=curves[:,0],y=curves[:,1],z=curves[:,2],mode='lines',name='Annotated centerlines',
            line=dict(color='#31c9b0',width=3),hoverinfo='skip',visible=visible))
        for mask,label,color,size in [(degree>=3,'Junctions','#ffad46',4),(degree==1,'Endpoints','#75adff',3),
                                     (degree==0,'Isolated nodes','#ff71b6',4),(degree==2,'Degree-2 samples','#d6f9f1',2)]:
            ids=np.flatnonzero(mask);p=points[ids]
            state='legendonly' if label=='Degree-2 samples' and visible else visible
            fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',name=label,visible=state,
                marker=dict(color=color,size=size),customdata=np.column_stack([g.node_ids[ids],degree[ids]]),
                hovertemplate='Node %{customdata[0]} · degree %{customdata[1]}<br>%{x:.2f}, %{y:.2f}, %{z:.2f} mm<extra>'+label+'</extra>'))
        groups.append(list(range(start,len(fig.data))))
        stats.append(dict(representation=name,nodes=len(points),edges=len(edges),junctions=int((degree>=3).sum()),endpoints=int((degree==1).sum())))
    buttons=[]
    for name,group in zip(['Junction anchors','Curvature-adaptive anchors','Dense SWC'],groups):
        visibility=[False]*len(fig.data)
        for i in group:visibility[i]='legendonly' if i==group[-1] else True
        buttons.append(dict(label=name,method='update',args=[dict(visible=visibility)]))
    fig.update_layout(template='plotly_dark',margin=dict(l=0,r=0,b=0,t=65),height=820,
        updatemenus=[dict(buttons=buttons,x=0,y=1.08,xanchor='left')],
        scene=dict(aspectmode='data',xaxis_title='World x (mm)',yaxis_title='World y (mm)',zaxis_title='World z (mm)'),
        legend=dict(x=0,y=.98),uirevision=subject)
    body=fig.to_html(full_html=False,include_plotlyjs=True,config=dict(scrollZoom=True,displaylogo=False,responsive=True))
    summary=' · '.join(f"{r['representation']}: {r['nodes']:,} nodes / {r['edges']:,} edges" for r in stats)
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>BRAVA GT — SUBJECT</title><style>body{margin:0;background:#111827;color:#e5e7eb;font:15px system-ui}header{padding:18px 24px}h1{margin:0 0 8px;font-size:24px}p{margin:7px 0;line-height:1.45}small{color:#acb8c8}details{margin:10px 24px 24px;overflow-wrap:anywhere}</style></head><body><header><h1>BRAVA SUBJECT · manually annotated ground-truth centerlines</h1><p>Drag to rotate · scroll to zoom · right-drag to pan · click legend entries to toggle points. Use the dropdown to change the graph anchors.</p><p>Sparse edges follow their original dense SWC paths. These are annotations, not model predictions or a vessel segmentation. Traces can be incomplete.</p><small>SUMMARY</small></header>BODY<details><summary>Provenance</summary><pre>METADATA</pre></details></body></html>'''
    metadata=dict(subject=subject,split=row['patch_split'],graph_root=str(base),primary_variant=marker['primary_variant'],
        coordinate_system='Prepared physical world coordinates in millimetres; existing SWC corrections retained',
        reflected_x=marker.get('reflect_x',False),representations=stats,meshes_or_segmentations_loaded=False)
    document=document.replace('SUBJECT',html.escape(subject)).replace('SUMMARY',html.escape(summary)).replace('BODY',body).replace('METADATA',html.escape(json.dumps(metadata,indent=2)))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(document)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output);print(json.dumps(metadata,indent=2));print('Bytes:',output.stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation',type=Path,default=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057'))
    p.add_argument('--subject',default='BG0001');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.preparation,a.subject,a.output)
