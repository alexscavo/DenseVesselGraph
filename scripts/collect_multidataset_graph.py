"""Collect cached development results, paired subjects and stage readiness."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.analyze_spline_target_capacity import write


def collect(root):
    plan=json.loads((root/'plan.json').read_text());out=root/'results';out.mkdir(exist_ok=True);results=[];geometry=[];decisions=[];intervals=[]
    for arm in plan['arms']:
        base=root/'evaluation'/arm['name'];assert json.loads((base/'complete.json').read_text())['completed']
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            for row in json.loads((folder/'summary.json').read_text()):results.append(dict(dataset=arm['dataset'],arm=arm['name'],step=int(folder.name),**row))
            g=pd.read_csv(folder/'geometry.csv.gz');g['dataset']=arm['dataset'];g['arm']=arm['name'];g['step']=int(folder.name);geometry.append(g)
            # Uncertainty reflects subject sampling only, never training-seed variation.
            patch=pd.read_csv(folder/'patches.csv.gz');patch=patch[(patch.split=='val')&(patch.policy=='default')&(patch.tolerance_mm==2.)]
            sums=patch.groupby('subject')[['tp','fp','fn','common_curve_supported_pred_samples','common_curve_pred_samples','common_curve_supported_gt_samples','common_curve_gt_samples']].sum().to_numpy()
            rng=np.random.default_rng(223039);samples=sums[rng.integers(len(sums),size=(1000,len(sums)))].sum(1)
            edge=2*samples[:,0]/np.maximum(1,2*samples[:,0]+samples[:,1]+samples[:,2]);p=samples[:,3]/np.maximum(1,samples[:,4]);r=samples[:,5]/np.maximum(1,samples[:,6]);curve=2*p*r/np.maximum(1e-20,p+r)
            intervals.append(dict(dataset=arm['dataset'],arm=arm['name'],step=int(folder.name),subjects=len(sums),edge_f1_low=float(np.quantile(edge,.025)),edge_f1_high=float(np.quantile(edge,.975)),curve_f1_low=float(np.quantile(curve,.025)),curve_f1_high=float(np.quantile(curve,.975))))
    frame=pd.DataFrame(results);frame.to_csv(out/'all-results.csv',index=False);geo=pd.concat(geometry,ignore_index=True);geo.to_csv(out/'geometry-edges.csv.gz',index=False);pd.DataFrame(intervals).to_csv(out/'subject-bootstrap.csv',index=False)
    gs=geo.groupby(['dataset','arm','step','split']).agg(branches=('edge','size'),mean_mm=('mean_mm','mean'),max_p95_mm=('max_mm',lambda x:x.quantile(.95)),neutral_mm=('neutral_mean_mm','mean'),controls=('controls','mean')).reset_index();gs.to_csv(out/'geometry-summary.csv',index=False)
    geo['length_group']=pd.cut(geo.length_mm,[0,10,25,50,100,np.inf],include_lowest=True)
    geo['endpoints_within_1mm']=geo.endpoint_max_mm<=1.;geo['K8_difficult']=geo.K8_fit_sampled_max_mm>.5
    geo.groupby(['dataset','arm','split','length_group','endpoints_within_1mm','K8_difficult'],observed=True).agg(branches=('edge','size'),mean_mm=('mean_mm','mean'),neutral_mm=('neutral_mean_mm','mean')).reset_index().to_csv(out/'geometry-stratified.csv',index=False)
    # Only BRAVA validation chooses the recovery winner. Primary matched-final table remains separate.
    choices=frame[(frame.dataset=='BRAVA')&(frame.split=='val')&(frame.policy=='default')&(frame.tolerance_mm==2.)]
    best=choices.loc[choices.f1.idxmax()];decision=dict(dataset='BRAVA',arm=best['arm'],step=int(best['step']),validation_edge_f1=float(best['f1']),validation_node_recall=float(best['node_recall']),representation_stage_ready=bool(best['f1']>.04 and best['node_recall']>.25),gate='Predefined feasibility gate: edge F1>0.04 and node recall2mm>0.25; review paired comparisons and annotation limitations before further submission.',automatic_followup_training=False)
    write(out/'brava-detector-selection.json',decision)
    # Paired differences between representation arms on identical subjects/populations.
    paired=[]
    for dataset in ['IXI','SyntheticMRI']:
        patches={}
        for arm in [a for a in plan['arms'] if a['dataset']==dataset]:
            file=root/'evaluation'/arm['name']/f'{plan["geometry_updates"]:06d}'/'patches.csv.gz';d=pd.read_csv(file);d=d[(d.split=='val')&(d.policy=='default')&(d.tolerance_mm==2.)]
            patches[arm['name']]=d
        baseline=patches[dataset+'_K8'];subjects=sorted(baseline.subject.unique());keys=['common_curve_supported_pred_samples','common_curve_pred_samples','common_curve_supported_gt_samples','common_curve_gt_samples'];bt=baseline.groupby('subject')[keys].sum().loc[subjects].to_numpy()
        def score(v):
            p=v[...,0]/np.maximum(1,v[...,1]);r=v[...,2]/np.maximum(1,v[...,3]);return 2*p*r/np.maximum(1e-20,p+r)
        rng=np.random.default_rng(223039);samples=rng.integers(len(subjects),size=(1000,len(subjects)))
        for name,d in patches.items():
            assert d.sample_id.tolist()==baseline.sample_id.tolist()
            for col in ['node_tp','nodes','gt_nodes','tp','fp','fn']:np.testing.assert_array_equal(d[col].to_numpy(),baseline[col].to_numpy())
            table=d.groupby('subject')[keys].sum().loc[subjects].to_numpy();delta=score(table[samples].sum(1))-score(bt[samples].sum(1))
            paired.append(dict(dataset=dataset,arm=name,reference=dataset+'_K8',curve_f1_difference=float(score(table.sum(0))-score(bt.sum(0))),low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975))))
    pd.DataFrame(paired).to_csv(out/'paired-representations.csv',index=False)
    primary=frame[(frame.split=='val')&(frame.policy=='default')&(frame.tolerance_mm==2.)]
    primary=primary[(primary.dataset!='BRAVA')|(primary.step==plan['detector_updates'])]
    cols=['dataset','arm','step','node_recall','raw_query_recall','f1','conditional_relation_ap','conditional_recall','common_branch_f1','common_curve_f1','beta0_mae','beta1_mae']
    lines=['## Completed initial batch','','Matched-final development results; no test evaluation. Frozen geometry arms have identical node/relation results by design. BRAVA recovery arms train the full detector.','',primary[cols].round(5).to_markdown(index=False),'','### Learned geometry','',gs.round(5).to_markdown(index=False),'','### Paired subject Curve F1 differences','',pd.DataFrame(paired).round(5).to_markdown(index=False),'','### BRAVA recovery selection','',json.dumps(decision,indent=2),'','A feasibility gate is not a scientific claim of superiority. Inspect best-versus-final learning curves, independent seed limitations, the selector confusion matrices and long-branch errors. Explicit splitting, recovered-BRAVA geometry learning, full end-to-end representations and node/relation ablations remain separate pending stages.']
    (out/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in plan['arms']:
        h=json.loads((root/'runs'/arm['name']/'history.json').read_text());ax=axes[0 if arm['kind']=='detector' else 1];ax.plot([x['step'] for x in h],[x['loss'] for x in h],label=arm['name'],alpha=.8)
    for ax in axes:ax.set(xlabel='Updates',ylabel='Training loss');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'training-curves.png',dpi=170);plt.close(fig)
    render_examples(root,plan,geo,out,decision)
    evidence=Path('docs/research/evidence/multidataset_continuous');evidence.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ['report.md','all-results.csv','geometry-summary.csv','subject-bootstrap.csv','paired-representations.csv','brava-detector-selection.json','training-curves.png','geometry-stratified.csv','BRAVA-example.html','IXI-example.html','SyntheticMRI-example.html']:shutil.copy2(out/name,evidence/name)
    doc=Path('docs/research/multidataset_continuous_graph_experiments.md');s=doc.read_text().split('\n<!-- INITIAL RESULTS -->')[0];doc.write_text(s+'\n<!-- INITIAL RESULTS -->\n'+(out/'report.md').read_text())
    write(out/'collection-complete.json',dict(completed=True,training_arms=len(plan['arms']),all_evaluations_complete=True,report=str(out/'report.md'),note='Numerical results and all figures/evidence published successfully'))


def render_examples(root,plan,geo,out,decision):
    import plotly.graph_objects as go
    for dataset in ['BRAVA','IXI','SyntheticMRI']:
        selected=decision['arm'] if dataset=='BRAVA' else dataset+'_piece_C1'
        candidates=geo[(geo.dataset==dataset)&(geo.split=='val')&(geo.arm==selected)]
        if dataset=='BRAVA':candidates=candidates[candidates.step==decision['step']]
        case=candidates.sort_values('length_mm',ascending=False).iloc[0];sample=case['sample_id'];edge=int(case['edge']);step=int(case['step'])
        source=Path(plan['datasets'][dataset]['root'])/'val/spline'/(sample+'.npz')
        with np.load(source) as f:
            a,b=f['path_offsets'][edge:edge+2];metric=f['metric'];path=f['full_paths'][a:b]@metric.T
        fig=go.Figure();fig.add_trace(go.Scatter3d(x=path[:,0],y=path[:,1],z=path[:,2],mode='lines',name='Dense annotation',line=dict(color='#ff9f1c',width=9)))
        methods=[selected] if dataset=='BRAVA' else [dataset+'_'+m for m in ['K8','K24','adaptive','piece_C0','piece_C1']]
        for name,color in zip(methods,['#19e6ff','#92ed50','#ff69c8','#ffffff','#bd9aff']):
            file=root/'predictions'/name/f'{step:06d}'/(sample+'.npz')
            with np.load(file) as c:
                indices=np.where(c['matched_edges']==edge)[0]
                if not len(indices):continue
                i=int(indices[0]);curve=c['matched_curves'][i]@metric.T
                if c['matched_reverse'][i]:curve=curve[::-1]
            fig.add_trace(go.Scatter3d(x=curve[:,0],y=curve[:,1],z=curve[:,2],mode='lines',name=name,line=dict(color=color,width=4)))
        fig.update_layout(template='plotly_dark',title=dataset+' validation · '+sample+' · step '+str(step)+' · GT edge '+str(edge)+' · Hungarian-positive diagnostic, not necessarily a decoded edge',scene=dict(aspectmode='data',xaxis_title='x mm',yaxis_title='y mm',zaxis_title='z mm'))
        fig.write_html(out/(dataset+'-example.html'),include_plotlyjs=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();collect(a.root)
