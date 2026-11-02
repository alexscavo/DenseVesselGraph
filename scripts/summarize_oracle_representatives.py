"""Subject-paired summaries of fixed-cache representative diagnostics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score
from scripts.oracle_representatives import OUT
from scripts.analyze_graph_errors import write,aggregate,csvwrite

def table(name):return pd.read_csv(OUT/(name+'.csv.gz'))
def mean(x):return float(np.mean(x)) if len(x) else None
def auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
def ap(y,p):return float(average_precision_score(y,p)) if np.any(y) else None

def paired_ci(frame,field,draws=2000):
    g=frame.groupby('subject')[field].agg(['sum','count']);v=g.to_numpy();rng=np.random.default_rng(261026)
    x=v[rng.integers(len(v),size=(draws,len(v)))].sum(1);z=x[:,0]/np.maximum(x[:,1],1)
    return dict(mean=float(v[:,0].sum()/v[:,1].sum()),ci95=np.quantile(z,[.025,.975]).tolist(),subjects=len(v))

def ap_ci(frame,left,right):
    """Exact AP with tie grouping, paired whole-subject weights (500 draws)."""
    subjects=sorted(frame.subject.unique());si=pd.Categorical(frame.subject,categories=subjects).codes
    y=frame.label.to_numpy();items=[]
    for col in [left,right]:
        p=frame[col].to_numpy();order=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[order])),len(p)-1]
        items.append((order,ends))
    rng=np.random.default_rng(261026);deltas=[]
    for j in range(500):
        counts=np.bincount(rng.integers(len(subjects),size=len(subjects)),minlength=len(subjects));weight=counts[si];vals=[]
        for order,ends in items:
            w=weight[order];positives=np.cumsum(w*y[order]);total=np.cumsum(w)
            den=positives[-1]
            if not den:break
            recall=positives[ends]/den;precision=positives[ends]/np.maximum(total[ends],1)
            vals.append(np.dot(np.diff(np.r_[0,recall]),precision))
        if len(vals)==2:deltas.append(vals[1]-vals[0])
    return dict(delta=ap(y,frame[right])-ap(y,frame[left]),ci95=np.quantile(deltas,[.025,.975]).tolist(),draws=len(deltas),subjects=len(subjects))

def graph_ci(frame,a,b):
    values=[]
    for condition in [a,b]:
        d=frame[frame.condition==condition].groupby('subject')[['edge_tp','edges','gt_edges']].sum()
        values.append(d)
    assert values[0].index.equals(values[1].index)
    v=np.stack([x.to_numpy() for x in values]);rng=np.random.default_rng(261026)
    indices=rng.integers(v.shape[1],size=(2000,v.shape[1]));draw=v[:,indices].sum(2)
    f1=2*draw[:,:,0]/(draw[:,:,1]+draw[:,:,2]);delta=f1[1]-f1[0]
    return dict(ci95=np.quantile(delta,[.025,.975]).tolist(),mean_delta=float(2*v[1,:,0].sum()/(v[1,:,1:3].sum())-2*v[0,:,0].sum()/v[0,:,1:3].sum()))

def oracle_ap_ci(frame):
    subjects=sorted(frame.subject.unique());items=[]
    for cond in ['nearest_shared','confidence_shared']:
        x=frame[frame.condition==cond];p=x.score.to_numpy();order=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[order])),len(p)-1]
        items.append((x.label.to_numpy()[order],pd.Categorical(x.subject,categories=subjects).codes[order],ends))
    rng=np.random.default_rng(261026);deltas=[]
    for _ in range(500):
        counts=np.bincount(rng.integers(len(subjects),size=len(subjects)),minlength=len(subjects));vals=[]
        for y,si,ends in items:
            w=counts[si];positives=np.cumsum(w*y);total=np.cumsum(w)
            if positives[-1]==0:break
            recall=positives[ends]/positives[-1];precision=positives[ends]/np.maximum(total[ends],1);vals.append(float(np.dot(np.diff(np.r_[0,recall]),precision)))
        if len(vals)==2:deltas.append(vals[1]-vals[0])
    a=frame[frame.condition=='nearest_shared'];b=frame[frame.condition=='confidence_shared']
    # Matched candidate identities imply equal labels/prevalence per patch and subject.
    assert a.groupby('subject').label.agg(['count','sum']).equals(b.groupby('subject').label.agg(['count','sum']))
    return dict(delta=ap(b.label,b.score)-ap(a.label,a.score),ci95=np.quantile(deltas,[.025,.975]).tolist(),draws=len(deltas),subjects=len(subjects))

def main():
    rep=table('representatives');pair=table('stable-neighbor-pairs');swap=table('single-swaps');gr=table('graphs');gp=table('graph-pairs');inter=table('interactions')
    result=dict(protocol=dict(training_patches=128,validation_patches=480,training_monitor_subjects=int(gr[gr.split=='train'].subject.nunique()),validation_subjects=int(gr[gr.split=='val'].subject.nunique()),step=12000,node_threshold=.5,edge_threshold=.5,graph_bootstrap_draws=2000,ap_bootstrap_draws=500,bootstrap_seed=261026),representatives=[],graphs=[],ranking=[],oracle_ranking_differences=[],neighbors=[],swaps=[],interactions=[],differences=[],margins=[])
    group=['seed','split','tolerance'];marginrows=[]
    for key,df in rep.groupby(group):
        seed,split,tol=key;changed=df[df.changed];meta=dict(seed=int(seed),split=split,tolerance=tol)
        row=dict(meta,common_vertices=len(df),changed=len(changed),changed_fraction=len(changed)/len(df),gt_type_changed=changed.groupby(['degree','boundary']).size().reset_index(name='count').to_dict('records'))
        if len(changed):
            d=changed.confidence_error_mm-changed.nearest_error_mm
            changed=changed.assign(error_delta=d)
            row.update(localization_delta=paired_ci(changed,'error_delta'),confidence_more_distant_fraction=mean(d>1e-6),more_distant_by_025mm=mean(d>.25),more_distant_by_05mm=mean(d>.5),alternative_distance_mean=mean(changed.alternative_separation_mm),raw_candidates_mean=mean(changed.raw_candidates),degree_mean=mean(changed.degree))
            for name in ['nearest','confidence']:
                for field in ['error_mm','confidence','foreground','soft_support','foreground_distance_mm','in_volume','hungarian_assigned','hungarian_same_gt','near_gt_count','below_threshold']:
                    row[name+'_'+field]=mean(changed[name+'_'+field])
                row[name+'_error_mm_quantiles']=np.quantile(changed[name+'_error_mm'],[.5,.95,1]).tolist()
            row['nearest_only_gt']=int(gr[(gr.seed==seed)&(gr.split==split)&(gr.tolerance==tol)&(gr.condition=='nearest')].only_nearest.sum())
            row['confidence_only_gt']=int(gr[(gr.seed==seed)&(gr.split==split)&(gr.tolerance==tol)&(gr.condition=='confidence')].only_confidence.sum())
        result['representatives'].append(row)
    for key,df in gr.groupby(group+['condition']):
        seed,split,tol,condition=key;v=aggregate(df.to_dict('records'))
        result['graphs'].append(dict(seed=int(seed),split=split,tolerance=tol,condition=condition,**v))
    for key,df in gp.groupby(group+['condition']):
        seed,split,tol,condition=key
        result['ranking'].append(dict(seed=int(seed),split=split,tolerance=tol,condition=condition,candidates=len(df),positives=int(df.label.sum()),prevalence=mean(df.label),ap=ap(df.label,df.score),auc=auc(df.label,df.score)))
    for key,df in gp[gp.condition.isin(['nearest_shared','confidence_shared'])].groupby(group):result['oracle_ranking_differences'].append(dict(seed=int(key[0]),split=key[1],tolerance=key[2],**oracle_ap_ci(df)))
    for key,df in pair.groupby(group):
        seed,split,tol=key;meta=dict(seed=int(seed),split=split,tolerance=tol)
        row=dict(meta,pairs=len(df),positives=int(df.label.sum()),nearest_ap=ap(df.label,df.nearest_score),confidence_ap=ap(df.label,df.confidence_score),nearest_auc=auc(df.label,df.nearest_score),confidence_auc=auc(df.label,df.confidence_score))
        row['ap_difference']=ap_ci(df,'nearest_score','confidence_score')
        row['categories']=[dict(category=cat,count=len(x),nearest_score=mean(x.nearest_score),confidence_score=mean(x.confidence_score),nearest_above_threshold=mean(x.nearest_score>.5),confidence_above_threshold=mean(x.confidence_score>.5)) for cat,x in df.groupby('category')]
        result['neighbors'].append(row)
        for rkey,x in df.groupby(['sample','subject','gt_vertex']):
            if not x.label.any() or not (~x.label.astype(bool)).any():continue
            positives=x[x.label==1]
            for negtype in ['all','near_negative']:
                negatives=x[x.label==0] if negtype=='all' else x[x.category=='near_negative']
                if len(negatives)==0:continue
                mn=positives.nearest_score.mean()-negatives.nearest_score.mean();mc=positives.confidence_score.mean()-negatives.confidence_score.mean()
                marginrows.append(dict(meta,sample=rkey[0],subject=rkey[1],gt_vertex=rkey[2],negative_type=negtype,nearest_margin=float(mn),confidence_margin=float(mc),margin_delta=float(mc-mn)))
    for key,df in swap.groupby(group):
        result['swaps'].append(dict(seed=int(key[0]),split=key[1],tolerance=key[2],valid_swaps=len(df),nearest_tp=int(df.nearest_tp.sum()),confidence_tp=int(df.confidence_tp.sum()),nearest_fp=int(df.nearest_fp.sum()),confidence_fp=int(df.confidence_fp.sum()),tp_delta=int((df.confidence_tp-df.nearest_tp).sum()),fp_delta=int((df.confidence_fp-df.nearest_fp).sum())))
    for key,df in inter.groupby(group):
        valid=df[df.score_interaction.notna()]
        result['interactions'].append(dict(seed=int(key[0]),split=key[1],tolerance=key[2],pairs=len(df),positives=int(df.label.sum()),invalid_single_swap_collisions=len(df)-len(valid),score_interaction_mean=mean(valid.score_interaction),score_interaction_abs_mean=mean(abs(valid.score_interaction)),decision_interaction_nonzero_fraction=mean(valid.decision_interaction!=0),tp_delta=int((((df.both_score>.5)-(df.nearest_score>.5).astype(int))*df.label).sum()),fp_delta=int((((df.both_score>.5)-(df.nearest_score>.5).astype(int))*(1-df.label)).sum())))
    for key,df in gr.groupby(group):
        row=dict(seed=int(key[0]),split=key[1],tolerance=key[2])
        for a,b in [('nearest','confidence'),('nearest_shared','confidence_shared'),('normal','confidence_nms_1mm'),('matched_count_confidence','confidence_nms_1mm')]:row[b+'_minus_'+a]=graph_ci(df,a,b)
        result['differences'].append(row)
    margin=pd.DataFrame(marginrows);csvwrite(OUT/'margins.csv.gz',marginrows)
    for key,df in margin.groupby(group+['negative_type']):
        meta=dict(seed=int(key[0]),split=key[1],tolerance=key[2],negative_type=key[3]);row=dict(meta,count=len(df),nearest_margin=mean(df.nearest_margin),confidence_margin=mean(df.confidence_margin),delta=paired_ci(df,'margin_delta'))
        joined=df.merge(rep,on=group+['sample','subject','gt_vertex']);d=joined.confidence_error_mm-joined.nearest_error_mm
        row['margin_delta_vs_localization_delta_spearman']=float(spearmanr(d,joined.margin_delta).statistic) if len(joined)>2 else None
        result['margins'].append(row)
    if (OUT/'embedding-replay-summary.json').exists() and __import__('json').loads((OUT/'embedding-replay-summary.json').read_text())['pairs']>0:
        result['embedding']=[];result['embedding_representatives']=[]
        emb=table('embedding-intervention-pairs');alts=table('embedding-alternatives')
        for key,df in emb.groupby(['seed','split']):
            row=dict(seed=int(key[0]),split=key[1],pairs=len(df),positives=int(df.label.sum()),scores={})
            for name in ['nearest','confidence','class_axis_only','orthogonal_only']:
                p=df[name+'_score'];row['scores'][name]=dict(ap=ap(df.label,p),auc=auc(df.label,p),true_mean=mean(p[df.label==1]),false_mean=mean(p[df.label==0]),tp=int(((p>.5)&(df.label==1)).sum()),fp=int(((p>.5)&(df.label==0)).sum()))
            row['neighbor_cosine']=[dict(label=int(l),nearest=mean(x.nearest_neighbor_cosine),confidence=mean(x.confidence_neighbor_cosine)) for l,x in df.groupby('label')];result['embedding'].append(row)
        for key,df in alts.groupby(['seed','split']):result['embedding_representatives'].append(dict(seed=int(key[0]),split=key[1],count=len(df),cosine=mean(df.cosine),nearest_norm=mean(df.nearest_norm),confidence_norm=mean(df.confidence_norm),class_axis_delta_energy_fraction=mean(df.class_axis_delta_energy_fraction)))
    write(OUT/'summary.json',result);csvwrite(OUT/'aggregate-graphs.csv',result['graphs']);csvwrite(OUT/'representative-summary.csv',result['representatives'])
    print('summary saved',flush=True)
if __name__=='__main__':main()
