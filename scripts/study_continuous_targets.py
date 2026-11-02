"""GT representation study. Original dense paths are the only reference."""
import argparse,csv,gzip,json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.continuous_representation import KS,TOLERANCES,fit_options,choose,split_path,compressed_curve
from scripts.analyze_spline_target_capacity import measure,write,spline_polyline


def worker(root,task,limit=None):
    plan=json.loads((root/'plan.json').read_text());spec=plan['tasks'][task];source=plan['datasets'][spec['dataset']]
    assert hashlib.sha256((Path(source['root'])/'manifest.json').read_bytes()).hexdigest()==source['manifest_sha256']
    out=root/'gt';out.mkdir(exist_ok=True);started=time.monotonic();count=0
    with gzip.open(out/f'{task:02d}.csv.gz','wt',newline='') as handle:
        writer=None
        for sample in spec['ids'][:limit]:
            with np.load(Path(source['root'])/spec['split']/'spline'/(sample+'.npz')) as f:
                metric=f['metric'].astype(float);paths=f['full_paths'].astype(float)@metric.T
                offsets=f['path_offsets'];nodes=len(f['nodes']);edges=len(f['edges'])
                for edge,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
                    path=paths[a:b];options=fit_options(path,f['controls'][edge].astype(float)@metric.T)
                    scores={k:measure(c,path) for k,c in options.items()}
                    common=dict(dataset=spec['dataset'],split=spec['split'],subject=source['subjects'][sample.split('_')[1]],sample_id=sample,edge=edge,base_nodes=nodes,base_edges=edges,
                        length_mm=scores[8]['length_mm'],chord_mm=scores[8]['chord_mm'],tortuosity=scores[8]['tortuosity'],turn_radians=scores[8]['total_turn_radians'])
                    rows=[]
                    def row(method,k,m,tol=0.,segments=1,passed=True,extra=0,fits=0):
                        return dict(common,method=method,tolerance_mm=tol,controls=k,segments=segments,extra_nodes=extra,extra_edges=extra,
                            # Unique coordinate vectors include graph anchors separately: endpoints are shared.
                            internal_vectors=k-2*segments+(segments-1 if extra==0 else 0),passed=passed,fits=fits,
                            mean_mm=m['symmetric_mean_mm'],p95_mm=max(m['spline_to_dense_p95_mm'],m['dense_to_spline_p95_mm']),
                            max_mm=m['hausdorff_sampled_mm'],upper_mm=m['hausdorff_upper_mm'],forward_max_mm=m['spline_to_dense_max_sampled_mm'],backward_max_mm=m['dense_to_spline_max_sampled_mm'])
                    for k in KS:rows.append(row('fixed_'+str(k),k,scores[k]))
                    for tol in TOLERANCES:
                        k,passed,m=choose(options,path,tol,scores);rows.append(row('oracle_K',k,m,tol,passed=passed))
                        cc,k=compressed_curve(options[24],tol);cm=measure(cc,path);rows.append(row('compress_GT24',k,cm,tol,passed=cm['hausdorff_upper_mm']<=tol))
                        pieces=split_path(path,tol);ms=pieces['scores'];weights=np.array([x['length_mm'] for x in ms]);weights/=weights.sum()
                        # Maximum is bounded segment-wise; mean/P95 here are explicitly segment summaries.
                        m={key:max(x[key] for x in ms) for key in ['hausdorff_sampled_mm','hausdorff_upper_mm','spline_to_dense_max_sampled_mm','dense_to_spline_max_sampled_mm','spline_to_dense_p95_mm','dense_to_spline_p95_mm']}
                        m['symmetric_mean_mm']=float(sum(w*x['symmetric_mean_mm'] for w,x in zip(weights,ms)))
                        n=len(ms)
                        rows.append(row('piecewise_inside_edge',8*n,m,tol,n,pieces['passed'],0,pieces['fits']))
                        rows.append(row('split_graph',8*n,m,tol,n,pieces['passed'],n-1,pieces['fits']))
                    for r in rows:
                        if writer is None:writer=csv.DictWriter(handle,fieldnames=list(r));writer.writeheader()
                        writer.writerow(r)
                    count+=1
            print(task,sample,count,flush=True)
    write(out/f'{task:02d}.json',dict(completed=True,patches=len(spec['ids'][:limit]),edges=count,seconds=time.monotonic()-started))


def collect(root,publish_results=True):
    import pandas as pd
    plan=json.loads((root/'plan.json').read_text());parts=[]
    for i,t in enumerate(plan['tasks']):
        s=json.loads((root/'gt'/f'{i:02d}.json').read_text());assert s['completed'] and s['patches']==len(t['ids'])
        parts.append(pd.read_csv(root/'gt'/f'{i:02d}.csv.gz'))
    d=pd.concat(parts,ignore_index=True);out=root/'results';out.mkdir(exist_ok=True)
    keys=['dataset','split','method','tolerance_mm'];rows=[]
    for key,g in d.groupby(keys):
        r=dict(zip(keys,key));r.update(branches=len(g),mean_mm=g.mean_mm.mean(),p95_branch_max_mm=g.max_mm.quantile(.95),worst_max_mm=g.max_mm.max(),controls_per_branch=g.controls.mean(),extra_nodes_per_branch=g.extra_nodes.mean(),failed_fraction=1-g.passed.mean(),fraction_extra_controls=(g.controls>8).mean(),fraction_split=(g.segments>1).mean())
        for tol in TOLERANCES:r['within_'+str(tol)]=float((g.upper_mm<=tol).mean())
        rows.append(r)
    aggregate=pd.DataFrame(rows);aggregate.to_csv(out/'gt-aggregate.csv',index=False)
    sub=d.groupby(keys+['subject']).agg(mean_mm=('mean_mm','mean'),max_mm=('max_mm','max'),controls=('controls','mean'),extra_nodes=('extra_nodes','mean')).reset_index();sub.to_csv(out/'gt-subjects.csv',index=False)
    patch=d.groupby(keys+['sample_id']).agg(base_nodes=('base_nodes','first'),base_edges=('base_edges','first'),extra_nodes=('extra_nodes','sum'),controls=('controls','sum'),internal_vectors=('internal_vectors','sum')).reset_index()
    patch['nodes']=patch.base_nodes+patch.extra_nodes;patch['edges']=patch.base_edges+patch.extra_nodes
    patch['coordinate_vectors']=patch.nodes+patch.internal_vectors;patch['fits_256_queries']=patch.nodes<=256
    patch['coordinate_bytes_float32']=12*patch.coordinate_vectors;patch['edge_index_bytes_int64']=16*patch.edges
    patch.to_csv(out/'gt-patches.csv.gz',index=False)
    for col,bins in [('length_mm',[0,10,25,50,100,np.inf]),('tortuosity',[1,1.05,1.2,1.5,2,np.inf]),('turn_radians',[0,1,3,6,12,np.inf])]:
        d['bin']=pd.cut(d[col],bins,include_lowest=True)
        d.groupby(keys+['bin'],observed=True).agg(branches=('edge','size'),mean_mm=('mean_mm','mean'),p95_max=('max_mm',lambda x:x.quantile(.95)),controls=('controls','mean')).reset_index().to_csv(out/('gt-by-'+col+'.csv'),index=False)
    lines=['## New GT results','','All distances use original dense paths. Piecewise mean is segment-arc-weighted; its P95 field is the maximum segment P95, not a pooled percentile. Explicit splitting and internal segments share the same curves; only graph-node costs differ.','',aggregate.round(5).to_markdown(index=False),'','### Patch complexity','',patch.groupby(keys).agg(mean_nodes=('nodes','mean'),max_nodes=('nodes','max'),mean_controls=('controls','mean'),mean_coordinate_vectors=('coordinate_vectors','mean'),query_coverage=('fits_256_queries','mean')).round(4).to_markdown(),'']
    (out/'gt-report.md').write_text('\n'.join(lines))
    # Train-only correctness gate, never use validation to select a representation.
    train=aggregate[(aggregate.dataset=='BRAVA')&(aggregate.split=='train')]
    a=float(train[train.method=='fixed_8'].mean_mm.iloc[0]);b=float(train[train.method=='fixed_24'].mean_mm.iloc[0])
    write(out/'gt-gate.json',dict(completed=True,larger_fixed_mean_improves=b<a,train_K8_mean_mm=a,train_K24_mean_mm=b,full_model_submission_authorized=False))
    overlays(root)
    if publish_results:publish(root)


def overlays(root):
    import plotly.graph_objects as go
    plan=json.loads((root/'plan.json').read_text());base=Path(plan['datasets']['BRAVA']['root'])
    fig=go.Figure();buttons=[]
    for edge in [14,25,18]:
        with np.load(base/'train/spline/sample_000002_0222.npz') as f:
            a,b=f['path_offsets'][edge:edge+2];path=f['full_paths'][a:b].astype(float)@f['metric'].T
        start=len(fig.data);visible=start==0
        curves=[('Dense centerline','#ff9f1c',path,9)]
        options=fit_options(path)
        curves += [('K='+str(k),color,spline_polyline(options[k]),4) for k,color in [(8,'#19e6ff'),(16,'#ff69c8'),(24,'#92ed50')]]
        pieces=split_path(path,.5);curves.append(('Piecewise K8 · tolerance 0.5 mm','#ffffff',np.concatenate([spline_polyline(c) for c in pieces['controls']]),4))
        for name,color,c,width in curves:fig.add_trace(go.Scatter3d(x=c[:,0],y=c[:,1],z=c[:,2],name=name,mode='lines',line=dict(color=color,width=width),visible=visible))
        joins=np.array([c[-1] for c in pieces['controls'][:-1]]).reshape(-1,3)
        fig.add_trace(go.Scatter3d(x=joins[:,0],y=joins[:,1],z=joins[:,2],mode='markers',name='Internal joins / optional graph nodes',marker=dict(color='white',size=6,symbol='diamond'),visible=visible))
        buttons.append((edge,list(range(start,len(fig.data)))))
    fig.update_layout(template='plotly_dark',scene=dict(aspectmode='data',xaxis_title='x mm',yaxis_title='y mm',zaxis_title='z mm'),title='BRAVA sample_000002_0222 · original dense paths and GT fits',updatemenus=[dict(buttons=[dict(label='Edge '+str(e),method='update',args=[dict(visible=[i in ids for i in range(len(fig.data))])]) for e,ids in buttons])])
    fig.write_html(root/'results/overlays.html',include_plotlyjs=True)


def publish(root):
    import shutil
    evidence=Path('docs/research/evidence/continuous_representation');evidence.mkdir(parents=True,exist_ok=True)
    for p in (root/'results').glob('*'):
        if p.is_file():shutil.copy2(p,evidence/p.name)
    doc=Path('docs/research/continuous_representation_comparison.md');s=doc.read_text();marker='\n<!-- GENERATED RESULTS -->\n'
    s=s.split(marker)[0]+marker
    for name in ['gt-report.md','learned-report.md']:
        p=root/'results'/name
        if p.exists():s+='\n'+p.read_text()
    doc.write_text(s)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','collect']);p.add_argument('--root',type=Path,required=True);p.add_argument('--task',type=int);p.add_argument('--limit',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.mode=='worker':worker(a.root,a.task,a.limit)
    else:collect(a.root,not a.smoke)
