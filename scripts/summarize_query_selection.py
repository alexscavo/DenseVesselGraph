"""Count-controlled, subject-paired N08 evaluation. No policy changes here."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scripts.run_query_selection import OUT
from scripts.analyze_graph_errors import aggregate,write,read,csvwrite

def summarize(df):
    r=aggregate(df.to_dict('records'));r.update(short_edge_recall=r['short_edges_tp']/r['gt_short_edges'] if r['gt_short_edges'] else None,close_junction_recall=r['close_junctions_recovered']/r['gt_close_junctions'] if r['gt_close_junctions'] else None,mean_pred_degree_max=r['pred_degree_max']/r['patches'])
    return r

def intervals(a,b):
    a=a.sort_values('sample');b=b.sort_values('sample');assert a['sample'].tolist()==b['sample'].tolist()
    fields=['node_tp','nodes','gt_nodes','edge_tp','edges','gt_edges','beta0_absolute_error','beta1_absolute_error','raw_node_tp','fp_gt_shortcut','fp_near_nonadjacent','anchors_gained','anchors_lost']
    frames=[d.groupby('subject')[fields].sum().sort_index() for d in [a,b]];assert frames[0].index.equals(frames[1].index)
    v=np.stack([d.to_numpy(float) for d in frames]);subjects=len(frames[0]);rng=np.random.default_rng(261026);indices=rng.integers(subjects,size=(2000,subjects));samples=v[:,indices].sum(2);total=v.sum(1)
    def metric(x,name):
        f={key:x[...,i] for i,key in enumerate(fields)}
        if name=='edge_f1':return 2*f['edge_tp']/np.maximum(f['edges']+f['gt_edges'],1)
        if name=='node_f1':return 2*f['node_tp']/np.maximum(f['nodes']+f['gt_nodes'],1)
        if name=='node_recall':return f['node_tp']/np.maximum(f['gt_nodes'],1)
        if name=='edge_fp':return f['edges']-f['edge_tp']
        return f[name]
    result={}
    for name in ['edge_f1','node_f1','node_recall','edge_tp','edge_fp','beta0_absolute_error','beta1_absolute_error','fp_gt_shortcut','fp_near_nonadjacent']:
        d=metric(samples,name)[1]-metric(samples,name)[0];point=metric(total,name)[1]-metric(total,name)[0]
        result[name]=dict(delta=float(point),ci95=np.quantile(d,[.025,.975]).tolist(),draws=2000,subjects=subjects)
    return result

def shared_ap_ci(labels,base,policy,subject):
    subjects,si=np.unique(subject,return_inverse=True);orders=[]
    for p in [base,policy]:
        order=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[order])),len(p)-1];orders.append((order,ends))
    rng=np.random.default_rng(261026);deltas=[]
    if not labels.any():return dict(delta=None,ci95=None,subjects=len(subjects),draws=0)
    for _ in range(500):
        count=np.bincount(rng.integers(len(subjects),size=len(subjects)),minlength=len(subjects));weights=count[si];values=[]
        for order,ends in orders:
            tp=np.cumsum(weights[order]*labels[order]);n=np.cumsum(weights[order]);den=tp[-1]
            if den==0:break
            recall=tp[ends]/den;precision=tp[ends]/np.maximum(n[ends],1);values.append(float(np.dot(np.diff(np.r_[0,recall]),precision)))
        if len(values)==2:deltas.append(values[1]-values[0])
    return dict(delta=float(average_precision_score(labels,policy)-average_precision_score(labels,base)),ci95=np.quantile(deltas,[.025,.975]).tolist(),subjects=len(subjects),draws=len(deltas))

def main():
    assert read(OUT/'complete.json')['completed'];choice=read(OUT/'training-selection.json');df=pd.read_csv(OUT/'per-patch.csv.gz');rows=[];paired=[];ranking=[];per_subject=[]
    for key,x in df.groupby(['seed','split','tolerance','method']):
        meta=dict(seed=int(key[0]),split=key[1],tolerance=key[2],method=key[3]);rows.append(dict(meta,**summarize(x)))
        for subject,y in x.groupby('subject'):per_subject.append(dict(meta,subject=subject,**summarize(y)))
    actual=['protected_nms','naive_global','guarded_global','local_profile']
    for seed in [364505,92741]:
        for split in ['train','val']:
            for tol in [1.,2.]:
                subset=df[(df.seed==seed)&(df.split==split)&(df.tolerance==tol)]
                for method in actual:
                    b=subset[subset.method==method]
                    for control in ['normal','confidence_nms',method+'_count_confidence',method+'_count_spatial']:
                        a=subset[subset.method==control]
                        if control.startswith(method+'_count_'):assert np.array_equal(a.sort_values('sample')['nodes'].to_numpy(),b.sort_values('sample')['nodes'].to_numpy())
                        paired.append(dict(seed=seed,split=split,tolerance=tol,method=method,control=control,**intervals(a,b)))
    for m in read(OUT/'ranking-manifest.json'):
        with np.load(m['path']) as f:
            r={k:m[k] for k in ['seed','split','tolerance','method']}
            for kind in ['all_retained','conditional']:
                y=f[kind+'_labels'];p=f[kind+'_scores'];r[kind+'_ap']=float(average_precision_score(y,p)) if y.any() else None;r[kind+'_candidates']=len(y);r[kind+'_positives']=int(y.sum());r[kind+'_prevalence']=float(y.mean()) if len(y) else None
            for ref in ['normal','nms']:
                y=f['shared_'+ref+'_labels'];base=f['shared_'+ref+'_base'];policy=f['shared_'+ref+'_policy'];r['shared_'+ref+'_candidates']=len(y);r['shared_'+ref+'_positives']=int(y.sum());r['shared_'+ref+'_base_ap']=float(average_precision_score(y,base)) if y.any() else None;r['shared_'+ref+'_policy_ap']=float(average_precision_score(y,policy)) if y.any() else None
                if m['method'] in actual and m['tolerance']==2.:r['shared_'+ref+'_difference']=shared_ap_ci(y,base,policy,f['shared_'+ref+'_subject'])
            ranking.append(r)
    primary=choice['primary'];decisions=[]
    for seed in [364505,92741]:
        conditions=[r for r in paired if r['seed']==seed and r['split']=='val' and r['tolerance']==2 and r['method']==primary and r['control'].startswith(primary+'_count_')]
        assert len(conditions)==2
        rank=next(r for r in ranking if r['seed']==seed and r['split']=='val' and r['tolerance']==2 and r['method']==primary)
        gate=all(r['edge_f1']['ci95'][0]>0 and r['node_recall']['delta']>=-.01 and r['edge_tp']['delta']>=0 for r in conditions) and rank['shared_nms_difference']['delta'] is not None and rank['shared_nms_difference']['delta']>0
        decisions.append(dict(seed=seed,primary=primary,passed=gate,count_controls=[r['control'] for r in conditions],shared_anchor_ranking_delta=rank['shared_nms_difference']['delta']))
    result=dict(completed=True,cpu_only=True,primary=primary,training_selection=choice,summary=rows,paired=paired,ranking=ranking,decision=dict(consistent_count_controlled_benefit=all(r['passed'] for r in decisions),seeds=decisions,stop_further_search=True),uncertainty='2000 subject draws for graph metrics; 500 exact AP draws; seed261026; reused validation exploratory')
    write(OUT/'summary.json',result);csvwrite(OUT/'aggregate.csv',rows);csvwrite(OUT/'per-subject.csv',per_subject);write(OUT/'paired-differences.json',paired)
    print('Primary:',primary,'decision:',result['decision'],flush=True)
    for r in rows:
        if r['split']=='val' and r['tolerance']==2 and r['method'] in ['normal','confidence_nms']+actual:print(r['seed'],r['method'],r['node_recall'],r['edge_f1'],r['edge_tp'],r['edge_fp'],r['beta1_signed_error_mean'],flush=True)
if __name__=='__main__':main()
