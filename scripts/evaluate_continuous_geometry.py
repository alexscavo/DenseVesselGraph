"""Cached learned geometry, train-only selector probe and common graph metrics."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.continuous_representation import fit_options,choose,compressed_curve,curve_features
from scripts.analyze_spline_target_capacity import measure,write,spline_polyline
from data.spline_targets import fit_spline


def collect(root,publish_results=True):
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels
    from scripts.evaluate_brava_study import patch_metrics,aggregate
    from scripts.study_continuous_targets import publish
    plan=json.loads((root/'plan.json').read_text());out=root/'results';out.mkdir(exist_ok=True)
    manifests={}
    for k in [8,16,24]:
        assert json.loads((root/'predictions'/str(k)/'complete.json').read_text())['completed']
        manifests[k]=json.loads((root/'predictions'/str(k)/'records.json').read_text())
    assert all([(r['sample_id'],r['split']) for r in manifests[k]]==[(r['sample_id'],r['split']) for r in manifests[8]] for k in [16,24])
    # A small selector predicts a compression K from predicted geometry only.
    features=[];labels=[];unmet=0
    for record in manifests[24]:
        if record['split']!='train':continue
        with np.load(record['cache']) as c,np.load(record['gt']) as gt:
            metric=gt['metric'].astype(float)
            for control,edge in zip(c['matched_controls'],c['matched_edges']):
                a,b=gt['path_offsets'][edge:edge+2];path=gt['full_paths'][a:b].astype(float)@metric.T
                options=fit_options(path);k,ok,_=choose(options,path,.5);unmet+=int(not ok)
                features.append(curve_features(control@metric.T));labels.append(k)
    classifier=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=221039) if len(set(labels))>1 else DummyClassifier(strategy='most_frequent'))
    classifier.fit(features,labels)
    import joblib
    joblib.dump(classifier,out/'control-selector.joblib')
    write(out/'selector-protocol.json',dict(features='Predicted K24 curve only; no GT-derived inference inputs',training_examples=len(labels),training_oracle_tolerance_failures=unmet,classes=sorted(set(labels)),tolerance_mm=.5,selector_type='separately trained predicted-curve compression classifier; not a K-specific neural projection head'))
    rows=[];graph_rows=[];common_rows=[]
    for index,record in enumerate(manifests[8]):
        with np.load(record['gt']) as f:gt={k:f[k] for k in f.files}
        metric=gt['metric'].astype(float);source={}
        for k in [8,16,24]:
            with np.load(manifests[k][index]['cache']) as f:source[k]={n:f[n] for n in f.files}
        for k in [16,24]:
            for key in ['nodes','edges','query_ids','candidate_edges','matched_edges','matched_reverse']:
                np.testing.assert_array_equal(source[8][key],source[k][key],err_msg='Frozen population drift: '+key)
            np.testing.assert_allclose(source[8]['candidate_scores'],source[k]['candidate_scores'],rtol=0,atol=0)
        meta={n:record[n] for n in ['sample_id','subject','split']}
        original=Path(plan['original_predictions'])/(record['sample_id']+'.npz')
        if record['split']=='val' and original.exists():
            with np.load(original) as f:old={n:f[n] for n in f.files}
            np.testing.assert_allclose(old['nodes'],source[8]['nodes'],atol=1e-6,rtol=1e-6)
            np.testing.assert_array_equal(old['edges'],source[8]['edges'])
            patch,_=patch_metrics(old,gt,record['gt'])
            graph_rows.extend(dict(meta,method='original_pretrained_K8',**r) for r in patch)
        for k,c in source.items():
            patch,_=patch_metrics(c,gt,record['gt'])
            graph_rows.extend(dict(meta,method='K'+str(k),**r) for r in patch)
            for control,edge,reverse in zip(c['matched_controls'],c['matched_edges'],c['matched_reverse']):
                a,b=gt['path_offsets'][edge:edge+2];path=gt['full_paths'][a:b].astype(float)@metric.T
                if reverse:path=path[::-1]
                physical=control.astype(float)@metric.T
                alternatives={'K'+str(k):physical}
                if k==24:
                    alternatives['compress24_0.5']=compressed_curve(physical,.5)[0]
                    selected=int(classifier.predict([curve_features(physical)])[0]);alternatives['learned_K_compression']=physical if selected==24 else fit_spline(spline_polyline(physical),selected)[0]
                    # Oracle K selects a GT label but still outputs a refit of the predicted curve.
                    oracle,ok,_=choose(fit_options(path),path,.5)
                    alternatives['oracle_K_compression']=physical if oracle==24 else fit_spline(spline_polyline(physical),oracle)[0]
                for name,cc in alternatives.items():
                    m=measure(cc,path);endpoint=float(np.linalg.norm(cc[[0,-1]]-path[[0,-1]],axis=1).max())
                    kk=len(cc);knots=np.r_[np.zeros(4),np.arange(1,kk-3)/(kk-3),np.ones(4)];g=np.array([knots[j+1:j+4].mean() for j in range(kk)])[:,None]
                    neutral=cc-((1-g)*cc[0]+g*cc[-1])+((1-g)*path[0]+g*path[-1])
                    mn=measure(neutral,path)
                    rows.append(dict(meta,method=name,edge=int(edge),controls=len(cc),endpoint_max_mm=endpoint,mean_mm=m['symmetric_mean_mm'],max_mm=m['hausdorff_sampled_mm'],endpoint_neutral_mean_mm=mn['symmetric_mean_mm'],length_mm=m['length_mm']))
        # Deployable postprocessing on actual decoded edges: no GT label selection.
        for method in ['compress24_0.5','learned_K_compression']:
            c=source[24];controls=[]
            for cc in c['spline_controls']:
                physical=cc.astype(float)@metric.T
                if method=='compress24_0.5':converted=compressed_curve(physical,.5)[0]
                else:
                    k=int(classifier.predict([curve_features(physical)])[0]);converted=physical if k==24 else fit_spline(spline_polyline(physical),k)[0]
                controls.append(converted@np.linalg.inv(metric).T)
            common=evaluate_common_vessels(dict(nodes=c['nodes'],edges=c['edges'],spline_controls=controls),record['gt'])
            common_rows.append(dict(meta,method=method,controls=sum(map(len,controls)),**common))
        if index%64==0:print(index,len(manifests[8]),flush=True)
    d=pd.DataFrame(rows);d.to_csv(out/'learned-branches.csv.gz',index=False)
    pd.DataFrame(graph_rows).to_csv(out/'learned-graphs.csv.gz',index=False);pd.DataFrame(common_rows).to_csv(out/'compressed-graphs.csv.gz',index=False)
    summaries=[]
    for (split,method),g in d.groupby(['split','method']):
        eligible=g[g.endpoint_max_mm<=1]
        summaries.append(dict(split=split,method=method,branches=len(g),mean_mm=g.mean_mm.mean(),max_p95_mm=g.max_mm.quantile(.95),controls=g.controls.mean(),endpoint_neutral_mean_mm=g.endpoint_neutral_mean_mm.mean(),localized_branches=len(eligible),localized_mean_mm=eligible.mean_mm.mean()))
    pd.DataFrame(summaries).to_csv(out/'learned-aggregate.csv',index=False)
    subjects=d.groupby(['split','subject','method']).agg(mean_mm=('mean_mm','mean'),neutral_mm=('endpoint_neutral_mean_mm','mean')).reset_index();subjects.to_csv(out/'learned-subjects.csv',index=False)
    intervals=[];rng=np.random.default_rng(221039);v=subjects[subjects.split=='val'].pivot(index='subject',columns='method',values='neutral_mm')
    for method in v.columns:
        delta=(v[method]-v['K8']).dropna().to_numpy();draw=rng.choice(delta,(1000,len(delta)),replace=True).mean(1)
        intervals.append(dict(method=method,paired_subjects=len(delta),difference=float(delta.mean()),low=float(np.quantile(draw,.025)),high=float(np.quantile(draw,.975))))
    write(out/'learned-bootstrap.json',intervals)
    aggregates=[]
    for split in ['train','val']:
        for method in ['K8','K16','K24','original_pretrained_K8']:
            for tol in [1,2]:
                g=[r for r in graph_rows if r['split']==split and r['method']==method and r['geometry']=='spline' and r['tolerance_mm']==tol]
                if g:aggregates.append(dict(split=split,method=method,tolerance_mm=tol,**aggregate(g)))
    from scripts.evaluate_brava_study import clean
    write(out/'learned-graph-aggregate.json',clean(aggregates))
    for split in ['train','val']:
        for method in ['compress24_0.5','learned_K_compression']:
            g=[r for r in common_rows if r['split']==split and r['method']==method]
            if g:write(out/(split+'-'+method+'-common.json'),clean(aggregate_common_vessels([{key:value for key,value in row.items() if key.startswith('common_')} for row in g])))
    # Keep graph results visible in the report, including the unchanged detector limitations.
    graph_table=[dict(split=r['split'],method=r['method'],tolerance_mm=r['tolerance_mm'],node_recall=r['node_recall'],edge_f1=r['f1'],common_branch_f1=r['common_branch_f1'],common_curve_f1=r['common_curve_f1'],component_mae=r['beta0_mae'],cycle_mae=r['beta1_mae']) for r in aggregates]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for k in [8,16,24]:
        history=root/'heads'/str(k)/'history.json'
        if history.exists():
            h=json.loads(history.read_text());ax.plot([r['step'] for r in h],[r['loss'] for r in h],label='K'+str(k))
    ax.set(xlabel='Matched updates',ylabel='Weighted residual-control loss');ax.legend();fig.tight_layout();fig.savefig(out/'training-curves.png',dpi=150);plt.close(fig)
    budget=json.loads((root/'budget.json').read_text())
    lines=['## New learned geometry results','','The detector is frozen and prediction/candidate parity is checked exactly. Hungarian GT-positive diagnostics include misplaced endpoints; the endpoint-neutral diagnostic is evaluation-only. It does not repair inference. B2 here is a separately trained **compression selector**, not a learned bank of K-specific spline heads. Explicit graph splitting and a learned internal-piecewise generator remain pending.','',pd.DataFrame(summaries).round(5).to_markdown(index=False),'','### Paired validation-subject differences in endpoint-neutral mean error versus K8','',pd.DataFrame(intervals).round(5).to_markdown(index=False),'','Graph metrics, per-subject tables and complete branch-level files are saved alongside this report. No improvement in connectivity is possible from a frozen geometry head alone.']
    lines+=['','### Actual training budget','','```json',json.dumps(budget,indent=2),'```','','### Graph reconstruction (frozen nodes and relations)','',pd.DataFrame(graph_table).round(5).to_markdown(index=False),'','![Training curves](evidence/continuous_representation/training-curves.png)']
    (out/'learned-report.md').write_text('\n'.join(lines))
    if publish_results:publish(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();collect(a.root,not a.smoke)
