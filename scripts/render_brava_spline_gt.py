"""View actual BRAVA training-cache spline targets without refitting geometry."""
import hashlib
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import BSpline


def joined(curves):
    return np.concatenate([np.vstack([c,np.full((1,3),np.nan)]) for c in curves])


def main():
    plan=json.loads(Path('experiments/brava_graph_study/plan.json').read_text())
    root=Path(plan['dataset']);fig=go.Figure();metadata=[];groups=[]
    training_subject=plan['subjects_by_id'][plan['ids']['train'][0]]
    for split,subject in [('train',training_subject),('val','BG0001')]:
        candidates=[]
        for name in plan['ids'][split]:
            if plan['subjects_by_id'][name]!=subject:continue
            path=root/split/'spline'/(name+'.npz')
            with np.load(path) as d:candidates.append((len(d['edges']),name,path))
        for _,name,path in sorted(candidates,reverse=True)[:4]:
            with np.load(path) as d:gt={k:d[k] for k in d.files}
            metric=gt['metric'];controls=gt['controls'];n=controls.shape[1]
            knots=np.r_[np.zeros(4),np.arange(1,n-3)/(n-3),np.ones(4)]
            curves=[BSpline(knots,c,3)(np.linspace(0,1,257))@metric.T for c in controls]
            replay=np.array([BSpline(knots,c,3)(np.linspace(0,1,64)) for c in controls])
            np.testing.assert_allclose(replay,gt['curves'],atol=2e-6,rtol=2e-6)
            np.testing.assert_allclose(controls[:,[0,-1]],gt['nodes'][gt['edges']],atol=2e-6)
            guides=[gt['full_paths'][a:b]@metric.T for a,b in zip(gt['path_offsets'][:-1],gt['path_offsets'][1:])]
            line=joined(curves);original=joined(guides);cp=joined(controls@metric.T);nodes=gt['nodes']@metric.T
            start=len(fig.data);active=start==0
            # Contrast in both colour and width keeps coincident paths identifiable.
            fig.add_trace(go.Scatter3d(x=original[:,0],y=original[:,1],z=original[:,2],mode='lines',name='Dense annotated centerline · orange',visible=active,line=dict(color='#FF9F1C',width=9),hovertemplate='%{x:.2f}, %{y:.2f}, %{z:.2f} mm<extra>Original dense centerline</extra>'))
            fig.add_trace(go.Scatter3d(x=line[:,0],y=line[:,1],z=line[:,2],mode='lines',name='Continuous spline GT · cyan',visible=active,line=dict(color='#19E6FF',width=4),hovertemplate='%{x:.2f}, %{y:.2f}, %{z:.2f} mm<extra>Actual spline target</extra>'))
            fig.add_trace(go.Scatter3d(x=nodes[:,0],y=nodes[:,1],z=nodes[:,2],mode='markers',name='Shared graph anchors · white',visible=active,marker=dict(color='#FFFFFF',size=3),text=[f'GT node {i}' for i in range(len(nodes))],hovertemplate='%{text}<br>%{x:.2f}, %{y:.2f}, %{z:.2f} mm<extra></extra>'))
            fig.add_trace(go.Scatter3d(x=cp[:,0],y=cp[:,1],z=cp[:,2],mode='lines+markers',name='Spline control polygon · purple (optional)',visible='legendonly' if active else False,line=dict(color='#CE93FF',width=2,dash='dash'),marker=dict(size=3,color='#CE93FF'),hoverinfo='name'))
            groups.append((f'{split} · {subject} · {name} · {len(gt["edges"])} edges',list(range(start,len(fig.data)))))
            metadata.append(dict(split=split,subject=subject,sample_id=name,cache=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),nodes=len(nodes),edges=len(controls),controls_per_edge=n,
                cached_curve_replay_max_error_mm=float(np.linalg.norm((replay-gt['curves'])@metric.T,axis=-1).max())))
    buttons=[]
    for label,indices in groups:
        vis=[False]*len(fig.data)
        for i in indices:vis[i]=True if i<indices[0]+3 else 'legendonly'
        buttons.append(dict(label=label,method='update',args=[dict(visible=vis),dict(title=dict(text=label))]))
    fig.update_layout(template='plotly_dark',height=790,title=groups[0][0],margin=dict(l=0,r=0,b=0,t=100),
        updatemenus=[dict(buttons=buttons,x=0,y=1.12,xanchor='left')],legend=dict(x=0,y=.98),
        scene=dict(aspectmode='data',xaxis_title='Patch-relative x (mm)',yaxis_title='Patch-relative y (mm)',zaxis_title='Patch-relative z (mm)'))
    body=fig.to_html(full_html=False,include_plotlyjs=True,config=dict(scrollZoom=True,displaylogo=False,responsive=True))
    intro='''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>BRAVA — actual continuous GT targets</title><style>body{margin:0;background:#111827;color:#e5e7eb;font:15px system-ui}header{padding:18px 24px}h1{font-size:24px;margin:0 0 10px}p{line-height:1.4;margin:7px 0}details{padding:15px 24px;overflow-wrap:anywhere}</style></head><body><header><h1>BRAVA · continuous ground-truth targets supplied to the model</h1><p><b style="color:#19E6FF">Cyan, thin: actual continuous spline GT.</b> <b style="color:#FF9F1C">Orange, thick: original dense annotated centerline.</b> White dots: shared graph anchors. Both geometries are visible by default; orange can form a border where they coincide. The spline uses 8 controls per edge. No curves were refitted or smoothed for this viewer.</p><p>Rotate by dragging, scroll to zoom, right-drag to pan. Choose a patch using the dropdown. Click either geometry in the legend to hide/show it independently and inspect any differences. Purple control polygons are optional and hidden by default. Dashed control polygons are not the spline curves.</p><p>These are real 64³ model patches, not an independently fitted whole-brain graph. Four patches with the most GT edges are shown for one training subject and one validation subject; no test data are used. Supervision uses spline controls/curves, while original SWC paths remain the geometric evaluation reference.</p></header>'''
    output=Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/BRAVA_spline_dense_overlay.html')
    output.write_text(intro+body+'<details><summary>Exact target-cache provenance</summary><pre>'+html.escape(json.dumps(metadata,indent=2))+'</pre></details></body></html>')
    output.with_name('BRAVA_continuous_GT.html').write_text(output.read_text())
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output);print('Patches',len(metadata),'bytes',output.stat().st_size,'max replay error mm',max(r['cached_curve_replay_max_error_mm'] for r in metadata));print('Subjects:',training_subject,'BG0001')

if __name__=='__main__':main()
