"""CPU R07 paired evaluation; operating points are selected on training only."""
import sys,io,json,zipfile,hashlib,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_curve
from scripts.analyze_graph_errors import read,write,csvwrite,geometry,topo
from scripts.node_verifier_metrics import association
from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels
OUT=ROOT/'experiments/r07_matched_supervision_20261027'
ORIGINAL=ROOT.parent/'Graph-Native-Betti-Matching-Representation'


def candidate_context(c,gt,tol):
    ids=np.flatnonzero(c['scores']>.5)
    local=association(c['physical'][ids],c['truth'],tol)
    mapping={int(ids[q]):int(g) for q,g in local.items()}
    target=np.full(len(c['scores']),-1,int)
    for q,g in mapping.items():target[q]=g
    pairs=c['pairs'];a,b=pairs.T
    active=np.isin(pairs,ids).all(1);mm=(target[a]>=0)&(target[b]>=0)
    labels=np.array([x in mapping and y in mapping and tuple(sorted((mapping[x],mapping[y]))) in gt['links'] for x,y in pairs])
    distance=np.full(len(pairs),np.inf)
    distance[mm]=np.linalg.norm(c['truth'][target[a[mm]]]-c['truth'][target[b[mm]]],axis=-1)
    near=active&mm&~labels&(distance<5.)
    inverse={g:q for q,g in mapping.items()};pair_index={tuple(sorted(e)):i for i,e in enumerate(pairs.tolist())}
    positive_indices=np.array([pair_index[tuple(sorted((inverse[a],inverse[b])))] if a in inverse and b in inverse else -1 for a,b in gt['edges']],int)
    hungarian=np.full(len(c['scores']),-1,int);hungarian[c['hungarian_source']]=c['hungarian_target']
    hungarian_mask=(hungarian[a]>=0)&(hungarian[b]>=0)
    hungarian_labels=np.array([hungarian[x]>=0 and hungarian[y]>=0 and tuple(sorted((int(hungarian[x]),int(hungarian[y])))) in gt['links'] for x,y in pairs])
    return dict(ids=ids,mapping=mapping,active=active,mm=mm,labels=labels,near=near,tolerance=tol,
                hungarian_mask=hungarian_mask,hungarian_labels=hungarian_labels,
                positive_indices=positive_indices,raw_tp=len(association(c['physical'],c['truth'],tol)))


def calibration(y,scores,gt_edges,target_precision):
    order=np.argsort(-scores,kind='stable');s=scores[order];labels=y[order]
    if not len(s):return dict(max_f1=.5,baseline_precision=.5)
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.cumsum(labels)[ends];n=ends+1;precision=tp/n;f1=2*tp/(n+gt_edges)
    # Move one representable score step down before storing a Python float.
    # A float64 nextafter step can round back onto a float32 score in decoding.
    thresholds=np.nextafter(s[ends],np.asarray(-np.inf,dtype=s.dtype)).astype(float)
    best=np.lexsort((-thresholds,-f1))[0]
    eligible=np.flatnonzero(precision>=target_precision)
    selected=eligible[np.lexsort((-thresholds[eligible],-precision[eligible],-tp[eligible]))[0]] if len(eligible) else None
    return dict(max_f1=float(thresholds[best]),baseline_precision=float(thresholds[selected]) if selected is not None else 1.)


def graph_metrics(c,gt,context,probability,threshold,arc_cut):
    ids=context['ids'];chosen=context['active']&(probability>threshold);labels=context['labels']
    edges=c['pairs'][chosen];lookup={int(q):i for i,q in enumerate(ids)}
    remapped=np.array([[lookup[a],lookup[b]] for a,b in edges],int).reshape(-1,2)
    b0,b1=topo(len(ids),remapped);g0,g1=topo(len(c['truth']),gt['edges'])
    pi=context['positive_indices'];eligible=pi>=0;correct=np.zeros(len(pi),bool);correct[eligible]=probability[pi[eligible]]>threshold
    arc=np.array([r['arc_mm'] for r in gt['geometry']]);long=arc>=arc_cut
    mapping=context['mapping'];dist=np.linalg.norm(c['physical'][ids,None]-c['truth'][None],axis=-1)
    duplicates=sum(q not in mapping and (dist[i]<=context['tolerance']).any() for i,q in enumerate(ids))
    row=dict(patches=1,nodes=len(ids),gt_nodes=len(c['truth']),node_tp=len(mapping),raw_node_tp=context['raw_tp'],
             edges=int(chosen.sum()),gt_edges=len(gt['edges']),edge_tp=int((chosen&labels).sum()),
             eligible_edges=int(eligible.sum()),duplicates=int(duplicates),
             near_candidates=int(context['near'].sum()),near_false_edges=int((chosen&context['near']).sum()),
             false_unmatched_edges=int((chosen&~context['mm']).sum()),
             gt_long_edges=int(long.sum()),eligible_long_edges=int((eligible&long).sum()),long_tp=int((correct&long).sum()),
             nodes_bias=len(ids)-len(c['truth']),edges_bias=int(chosen.sum())-len(gt['edges']),
             beta0_bias=b0-g0,beta1_bias=b1-g1,beta0_error=abs(b0-g0),beta1_error=abs(b1-g1))
    return row,chosen


def aggregate(rows):
    fields=[k for k,v in rows[0].items() if isinstance(v,(int,float,np.number)) and k not in ['seed','step','tolerance','threshold']]
    r={k:sum(row[k] for row in rows) for k in fields}
    ratio=lambda a,b:r[a]/r[b] if r[b] else None
    r.update(node_precision=ratio('node_tp','nodes'),node_recall=ratio('node_tp','gt_nodes'),
             raw_node_recall=ratio('raw_node_tp','gt_nodes'),node_f1=2*r['node_tp']/(r['nodes']+r['gt_nodes']),
             edge_precision=ratio('edge_tp','edges'),edge_recall=ratio('edge_tp','gt_edges'),
             edge_f1=2*r['edge_tp']/(r['edges']+r['gt_edges']),edge_fp=r['edges']-r['edge_tp'],edge_fn=r['gt_edges']-r['edge_tp'],
             conditional_recall=ratio('edge_tp','eligible_edges'),near_fpr=ratio('near_false_edges','near_candidates'),
             long_recall=ratio('long_tp','gt_long_edges'),conditional_long_recall=ratio('long_tp','eligible_long_edges'))
    for k in ['nodes_bias','edges_bias','beta0_bias','beta1_bias','beta0_error','beta1_error']:r[k+'_per_patch']=r[k]/r['patches']
    return r


def paired_graph_ci(a,b,draws=1000):
    assert [r['sample'] for r in a]==[r['sample'] for r in b]
    subjects=sorted({r['subject'] for r in a});fields=['edge_tp','edges','gt_edges','near_false_edges','near_candidates','long_tp','gt_long_edges']
    matrices=np.array([[[sum(r[f] for r in rows if r['subject']==s) for f in fields] for s in subjects] for rows in [a,b]],float)
    rng=np.random.default_rng(271026);sample=matrices[:,rng.integers(len(subjects),size=(draws,len(subjects)))].sum(2)
    total=matrices.sum(1)
    metrics={'edge_f1':lambda x:2*x[...,0]/np.maximum(x[...,1]+x[...,2],1),
             'edge_tp':lambda x:x[...,0],'edge_fp':lambda x:x[...,1]-x[...,0],
             'near_fpr':lambda x:x[...,3]/np.maximum(x[...,4],1),
             'long_recall':lambda x:x[...,5]/np.maximum(x[...,6],1)}
    return {k:dict(delta=float(f(total)[1]-f(total)[0]),ci95=np.quantile(f(sample)[1]-f(sample)[0],[.025,.975]).tolist(),subjects=len(subjects),draws=draws) for k,f in metrics.items()}


def paired_ap_ci(labels,base,policy,subject):
    subjects,si=np.unique(subject,return_inverse=True);orders=[]
    for p in [base,policy]:
        order=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[order])),len(p)-1];orders.append((order,ends))
    if not labels.any():return dict(delta=None,ci95=None,subjects=len(subjects),draws=0)
    rng=np.random.default_rng(271026);deltas=[]
    for _ in range(500):
        count=np.bincount(rng.integers(len(subjects),size=len(subjects)),minlength=len(subjects));weights=count[si];values=[]
        for order,ends in orders:
            tp=np.cumsum(weights[order]*labels[order]);n=np.cumsum(weights[order]);den=tp[-1]
            if den==0:break
            values.append(float(np.dot(np.diff(np.r_[0,tp[ends]/den]),tp[ends]/np.maximum(n[ends],1))))
        if len(values)==2:deltas.append(values[1]-values[0])
    return dict(delta=float(average_precision_score(labels,policy)-average_precision_score(labels,base)),ci95=np.quantile(deltas,[.025,.975]).tolist(),subjects=len(subjects),draws=len(deltas))


def paired_curve_ci(a,b):
    assert [r['sample'] for r in a]==[r['sample'] for r in b]
    subjects=sorted({r['subject'] for r in a})
    fields=['common_curve_pred_samples','common_curve_gt_samples','common_curve_supported_pred_samples',
            'common_curve_supported_gt_samples','common_branch_tp','common_branch_fp','common_branch_fn']
    values=np.array([[[sum(r[k] for r in rows if r['subject']==s) for k in fields] for s in subjects] for rows in [a,b]],float)
    rng=np.random.default_rng(271026);samples=values[:,rng.integers(len(subjects),size=(1000,len(subjects)))].sum(2);total=values.sum(1)
    def curve(v):
        p=np.divide(v[...,2],v[...,0],out=np.ones_like(v[...,2]),where=v[...,0]>0)
        r=np.divide(v[...,3],v[...,1],out=np.ones_like(v[...,3]),where=v[...,1]>0)
        return np.divide(2*p*r,p+r,out=np.zeros_like(p),where=(p+r)>0)
    def branch(v):
        denominator=2*v[...,4]+v[...,5]+v[...,6]
        return np.divide(2*v[...,4],denominator,out=np.ones_like(denominator),where=denominator>0)
    return {name:dict(delta=float(f(total)[1]-f(total)[0]),ci95=np.quantile(f(samples)[1]-f(samples)[0],[.025,.975]).tolist(),subjects=len(subjects),draws=1000) for name,f in [('curve_f1',curve),('branch_f1',branch)]}


def load(seed,smoke=False):
    p=read(OUT/'plan.json');records={r['sample']:r for r in p['records'][str(seed)]};result=[]
    with zipfile.ZipFile(OUT/'large'/str(seed)/('smoke-features.zip' if smoke else 'features.zip')) as z:
        for name in z.namelist():
            with np.load(io.BytesIO(z.read(name))) as f:c={k:f[k] for k in f.files if k!='tokens'}
            c['record']=records[name.removesuffix('.npz')];c['gt']=geometry(c['record']['gt'])
            c['context']={tol:candidate_context(c,c['gt'],tol) for tol in [1.,2.]}
            result.append(c)
    return result


def smoke():
    data=load(364505,True);rows=[];curves=[]
    with zipfile.ZipFile(OUT/'large/364505/smoke/S0/scores-0003.zip') as z:
        for c in data:
            score=np.load(io.BytesIO(z.read(c['record']['sample']+'.npz')))['relations']
            for tol in [1.,2.]:
                row,chosen=graph_metrics(c,c['gt'],c['context'][tol],score,.5,10.)
                row.update(sample=c['record']['sample'],subject=c['record']['subject']);rows.append(row)
            curves.append(dict(sample=c['record']['sample'],subject=c['record']['subject'],**curve_metrics(c,chosen)))
    assert len(data)==8
    a=aggregate(rows);b=aggregate_common_vessels([{k:v for k,v in row.items() if k.startswith(('common_','native_','representation_'))} for row in curves])
    assert all(np.isfinite(a[k]) for k in ['node_f1','edge_f1'])
    assert paired_graph_ci(rows,rows,20)['edge_f1']['ci95']==[0.,0.]
    assert paired_curve_ci(curves,curves)['curve_f1']['ci95']==[0.,0.]
    assert np.isfinite(b['common_curve_f1'])
    write(OUT/'collector-gpu-smoke.json',dict(passed=True,real_patches=8,graph_curve_aggregation_and_identical_pairing=True,
          note='Execution check after three updates only; not a scientific result.'))


def curve_metrics(c,chosen):
    ids=c['context'][2.]['ids'];lookup={int(q):i for i,q in enumerate(ids)};selected=np.flatnonzero(chosen)
    controls={int(i):v for i,v in zip(c['curve_pair_indices'],c['curve_controls'])}
    prediction=dict(nodes=c['physical'][ids]@np.linalg.inv(c['metric'].T),
                    edges=np.array([[lookup[a],lookup[b]] for a,b in c['pairs'][chosen]],int).reshape(-1,2),
                    spline_controls=[controls[int(i)] for i in selected],spline_family='bspline')
    return evaluate_common_vessels(prediction,c['record']['gt'])


def run():
    p=read(OUT/'plan.json');allrows=[];summaries=[];ranks=[];curve_rows=[];thresholds=[];differences=[];learning=[]
    (OUT/'ranking').mkdir(exist_ok=True)
    for seed in p['seeds']:
        for arm in ['S0','S1']:assert read(OUT/'large'/str(seed)/'runs'/arm/'complete.json')['completed']
        complete=[read(OUT/'large'/str(seed)/'runs'/arm/'complete.json') for arm in ['S0','S1']]
        assert complete[0]['order_sha256']==complete[1]['order_sha256']
        assert complete[0]['effective_budget_sha256']==complete[1]['effective_budget_sha256']
        data=load(seed);training=[c for c in data if c['record']['split']=='train']
        arc_cut=float(np.quantile([r['arc_mm'] for c in training for r in c['gt']['geometry']],.75))
        base_y=np.concatenate([c['context'][2.]['labels'][c['context'][2.]['active']] for c in training])
        base_s=np.concatenate([c['relations'][c['context'][2.]['active']] for c in training])
        precision_target=float(base_y[base_s>.5].mean())
        methods=[('original',0)]+[(a,step) for a in ['S0','S1'] for step in [100,300,1000]]
        score_sets={};curve_cache={}
        for arm,step in methods:
            if arm=='original':scores={c['record']['sample']:c['relations'] for c in data}
            else:
                with zipfile.ZipFile(OUT/'large'/str(seed)/'runs'/arm/f'scores-{step:04d}.zip') as z:
                    scores={c['record']['sample']:np.load(io.BytesIO(z.read(c['record']['sample']+'.npz')))['relations'] for c in data}
            score_sets[arm,step]=scores
            train_scores=np.concatenate([scores[c['record']['sample']][c['context'][2.]['active']] for c in training])
            cuts=dict(fixed=.5,**calibration(base_y,train_scores,sum(len(c['gt']['edges']) for c in training),precision_target))
            achieved={name:float(base_y[train_scores>cut].mean()) if (train_scores>cut).any() else None for name,cut in cuts.items()}
            thresholds.append(dict(seed=seed,arm=arm,step=step,policies=cuts,baseline_training_precision=precision_target,
                                   achieved_training_precision=achieved,precision_target_feasible=achieved['baseline_precision'] is not None and achieved['baseline_precision']>=precision_target,training_only=True))
            # One ranking population per checkpoint; candidate IDs/labels stay fixed.
            for split in ['train','val']:
                selected=[c for c in data if c['record']['split']==split]
                for tol in [1.,2.]:
                    row=dict(seed=seed,arm=arm,step=step,split=split,tolerance=tol)
                    for kind in ['conditional','all_retained','hungarian_pairs']:
                        masks=[c['context'][tol]['hungarian_mask'] if kind=='hungarian_pairs' else c['context'][tol]['active'] & (c['context'][tol]['mm'] if kind=='conditional' else True) for c in selected]
                        label_key='hungarian_labels' if kind=='hungarian_pairs' else 'labels'
                        y=np.concatenate([c['context'][tol][label_key][mask] for c,mask in zip(selected,masks)])
                        s=np.concatenate([scores[c['record']['sample']][mask] for c,mask in zip(selected,masks)])
                        subject=np.concatenate([np.repeat(c['record']['subject'],mask.sum()) for c,mask in zip(selected,masks)])
                        row[kind+'_ap']=float(average_precision_score(y,s)) if y.any() else None
                        row[kind+'_candidates']=len(y);row[kind+'_positives']=int(y.sum())
                        path=OUT/'ranking'/f'{seed}-{arm}-{step}-{split}-{tol:g}-{kind}.npz'
                        np.savez_compressed(path,labels=y,scores=s,subject=subject)
                        prec,rec,_=precision_recall_curve(y,s)
                        # Clearly diagnostic validation PR operating points, never
                        # applied to graph decoding or used to select checkpoints.
                        base_prob=np.concatenate([c['relations'][mask] for c,mask in zip(selected,masks)])
                        base_prec=float(y[base_prob>.5].mean()) if (base_prob>.5).any() else 1.
                        base_rec=float(y[base_prob>.5].sum()/y.sum()) if y.any() else 0.
                        at_precision=rec[prec>=base_prec]
                        at_recall=prec[rec>=base_rec]
                        row[kind+'_diagnostic_recall_at_original_precision']=float(at_precision.max()) if len(at_precision) else None
                        row[kind+'_diagnostic_precision_at_original_recall']=float(at_recall.max()) if len(at_recall) else None
                    ranks.append(row)
            for policy,cutoff in cuts.items():
                if step in [100,300] and policy!='fixed':continue
                for c in data:
                    rec=c['record'];prob=scores[rec['sample']]
                    for tol in [1.,2.]:
                        r,chosen=graph_metrics(c,c['gt'],c['context'][tol],prob,cutoff,arc_cut)
                        r.update(seed=seed,arm=arm,step=step,split=rec['split'],sample=rec['sample'],subject=rec['subject'],tolerance=tol,policy=policy,threshold=cutoff)
                        allrows.append(r)
                    if step in [0,1000]:
                        key=(rec['sample'],tuple(np.flatnonzero(chosen)))
                        if key not in curve_cache:curve_cache[key]=curve_metrics(c,chosen)
                        curve_rows.append(dict(seed=seed,arm=arm,step=step,split=rec['split'],sample=rec['sample'],subject=rec['subject'],policy=policy,**curve_cache[key]))
            print('evaluated',seed,arm,step,flush=True)
        # Exact step-zero score parity is a second gate after extraction parity.
        for arm in ['S0','S1']:
            with zipfile.ZipFile(OUT/'large'/str(seed)/'runs'/arm/'scores-0000.zip') as z:
                for c in data:
                    s=np.load(io.BytesIO(z.read(c['record']['sample']+'.npz')))['relations']
                    np.testing.assert_allclose(s,c['relations'],atol=1e-5,rtol=0)
            learning.append(dict(seed=seed,arm=arm,history=read(OUT/'large'/str(seed)/'runs'/arm/'history.json')))
        for split in ['train','val']:
            for tol in [1.,2.]:
                for control,control_step in [('S0',1000),('original',0)]:
                    for policy in ['fixed','max_f1','baseline_precision']:
                        rr=[[r for r in allrows if (r['seed'],r['arm'],r['step'],r['split'],r['tolerance'],r['policy'])==(seed,arm,step,split,tol,policy)] for arm,step in [(control,control_step),('S1',1000)]]
                        differences.append(dict(seed=seed,split=split,tolerance=tol,policy=policy,comparison='S1 minus '+control,**paired_graph_ci(*rr)))
                    for kind in ['conditional','all_retained','hungarian_pairs']:
                        arrays=[]
                        for arm,step in [(control,control_step),('S1',1000)]:
                            with np.load(OUT/'ranking'/f'{seed}-{arm}-{step}-{split}-{tol:g}-{kind}.npz') as f:arrays.append({k:f[k] for k in f.files})
                        np.testing.assert_array_equal(arrays[0]['labels'],arrays[1]['labels']);np.testing.assert_array_equal(arrays[0]['subject'],arrays[1]['subject'])
                        differences.append(dict(seed=seed,split=split,tolerance=tol,population=kind,comparison='S1 minus '+control+' AP',**paired_ap_ci(arrays[0]['labels'],arrays[0]['scores'],arrays[1]['scores'],arrays[0]['subject'])))
            for control,control_step in [('S0',1000),('original',0)]:
                for policy in ['fixed','max_f1','baseline_precision']:
                    rr=[[r for r in curve_rows if (r['seed'],r['arm'],r['step'],r['split'],r['policy'])==(seed,arm,step,split,policy)] for arm,step in [(control,control_step),('S1',1000)]]
                    differences.append(dict(seed=seed,split=split,policy=policy,comparison='S1 minus '+control+' common-v2',**paired_curve_ci(*rr)))
    keys=['seed','arm','step','split','tolerance','policy'];groups={tuple(r[k] for k in keys) for r in allrows}
    per_subject=[]
    for group in sorted(groups):
        rr=[r for r in allrows if tuple(r[k] for k in keys)==group];s=dict(zip(keys,group));s.update(aggregate(rr))
        curves=[r for r in curve_rows if all(r[k]==s[k] for k in ['seed','arm','step','split','policy'])]
        if curves:s.update(aggregate_common_vessels([{k:v for k,v in r.items() if k.startswith(('common_','native_','representation_'))} for r in curves]))
        summaries.append(s)
        for subject in sorted({r['subject'] for r in rr}):per_subject.append(dict(zip(keys,group),subject=subject,**aggregate([r for r in rr if r['subject']==subject])))
    write(OUT/'summary.json',dict(completed=True,summary=summaries,ranking=ranks,differences=differences,thresholds=thresholds,learning=learning,
          node_predictions_identical=True,plan=p,stage3_unfrozen_not_launched=True))
    csvwrite(OUT/'per-patch.csv.gz',allrows);csvwrite(OUT/'per-subject.csv',per_subject);csvwrite(OUT/'aggregate.csv',summaries)
    csvwrite(OUT/'common-v2-per-patch.csv.gz',curve_rows);write(OUT/'thresholds.json',thresholds);write(OUT/'paired-differences.json',differences)
    plot(read(OUT/'summary.json'));publish(read(OUT/'summary.json'));print('R07 completed',flush=True)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for row,seed in enumerate([364505,92741]):
        for arm,color in [('S0','tab:blue'),('S1','tab:orange')]:
            for split,style in [('train','-'),('val','--')]:
                r=sorted([v for v in result['ranking'] if v['seed']==seed and v['arm']==arm and v['split']==split and v['tolerance']==2.],key=lambda x:x['step'])
                axes[row,0].plot([v['step'] for v in r],[v['conditional_ap'] for v in r],style,marker='o',color=color,label=arm+'/'+split)
                s=sorted([v for v in result['summary'] if v['seed']==seed and v['arm']==arm and v['split']==split and v['tolerance']==2. and v['policy']=='fixed'],key=lambda x:x['step'])
                axes[row,1].plot([v['step'] for v in s],[v['edge_f1'] for v in s],style,marker='o',color=color)
            history=next(v['history'] for v in result['learning'] if v['seed']==seed and v['arm']==arm)
            axes[row,2].plot([v['step'] for v in history],[v['loss'] for v in history],color=color,label=arm)
        for j,label in enumerate(['Conditional relation AP','Positional edge F1, 2mm','Optimization CE × original edge weight']):
            axes[row,j].set_title(f'{seed}: {label}');axes[row,j].set_xlabel('Relation-only updates');axes[row,j].grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,2].legend(fontsize=8)
    fig.savefig(OUT/'learning-curves.png',dpi=150);plt.close(fig)


def publish(result):
    s=result['summary'];ranking=result['ranking'];lines=['## R07 — modern SyntheticMRI matched-pair negative redistribution — 27 October 2026','',
      'Completed frozen-detector relation-only diagnostic. N08 remained negative/mixed and no query-selection mechanism was changed. Same two A-xyz step-12,000 sources; 128 training-monitor patches/32 subjects and 480 validation patches/13 disjoint subjects per seed. One 1,000-update relation-only continuation per policy and source; fixed early checkpoints 100/300/1,000. No test data, unmatched mining, image adapter or unfrozen confirmation.','',
      'The existing 9,999-pair cap already includes every Hungarian-matched negative on these patches. S1 tests relative negative multiplicity, not missing binary exposure: preserve all positive slots and exact per-graph/batch post-ratio-balancing negative counts; replace at most 25% of negative slots with copies of reliable <5-mm nonadjacent pairs, bounded at 4x their S0 multiplicity. Reliability: each Hungarian query within 2 mm of its own nearest GT anchor, with at least 0.5-mm margin to the next GT anchor. All GT-based gates are training supervision/offline evaluation only. Existing CE, edge weight 6, ratio 0.15/tolerance 0.02 and random endpoint orientation are unchanged.','',
      '| Seed | Scorer | Step | Validation policy | Edge F1 2mm | TP / FP / FN | Conditional AP 2mm | Near-negative FPR | Long-edge recall | Branch / Curve F1 | β1 MAE |',
      '|---|---|---:|---|---:|---|---:|---:|---:|---|---:|']
    for r in s:
        if r['split']!='val' or r['tolerance']!=2. or r['step'] not in [0,1000]:continue
        ap=next(v['conditional_ap'] for v in ranking if (v['seed'],v['arm'],v['step'],v['split'],v['tolerance'])==(r['seed'],r['arm'],r['step'],'val',2.))
        lines.append(f'|{r["seed"]}|{r["arm"]}|{r["step"]}|{r["policy"]}|{r["edge_f1"]:.5f}|{r["edge_tp"]}/{r["edge_fp"]}/{r["edge_fn"]}|{ap:.5f}|{r["near_fpr"]:.4f}|{r["long_recall"]:.4f}|{r["common_branch_f1"]:.4f}/{r["common_curve_f1"]:.4f}|{r["beta1_error_per_patch"]:.4f}|')
    gates=[]
    for seed in [364505,92741]:
        ap=next(v for v in result['differences'] if v.get('population')=='conditional' and v['seed']==seed and v['split']=='val' and v['tolerance']==2. and v['comparison']=='S1 minus S0 AP')
        edge=next(v for v in result['differences'] if v.get('policy')=='fixed' and v['seed']==seed and v['split']=='val' and v['tolerance']==2. and v['comparison']=='S1 minus S0')
        base_ap=next(v for v in result['differences'] if v.get('population')=='conditional' and v['seed']==seed and v['split']=='val' and v['tolerance']==2. and v['comparison']=='S1 minus original AP')
        base_edge=next(v for v in result['differences'] if v.get('policy')=='fixed' and v['seed']==seed and v['split']=='val' and v['tolerance']==2. and v['comparison']=='S1 minus original')
        source_rows=[next(v for v in s if (v['seed'],v['arm'],v['step'],v['split'],v['tolerance'],v['policy'])==(seed,arm,step,'val',2.,'fixed')) for arm,step in [('original',0),('S0',1000),('S1',1000)]]
        gate=(ap['ci95'] is not None and ap['ci95'][0]>0 and edge['edge_f1']['ci95'][0]>0 and
              edge['edge_tp']['delta']>=0 and edge['edge_fp']['delta']<=0 and edge['near_fpr']['delta']<0 and edge['long_recall']['delta']>=0 and
              base_ap['delta']>0 and base_edge['edge_f1']['delta']>0 and base_edge['edge_tp']['delta']>=0 and
              base_edge['edge_fp']['delta']<=0 and base_edge['near_fpr']['delta']<0 and base_edge['long_recall']['delta']>=0 and
              source_rows[2]['beta1_error_per_patch']<=min(v['beta1_error_per_patch'] for v in source_rows[:2]))
        gates.append(gate)
        lines+=['',f'Seed {seed}, S1−S0 at fixed cutoff: Edge F1 {edge["edge_f1"]}; TP {edge["edge_tp"]}; FP {edge["edge_fp"]}; conditional AP {ap}.']
    conclusion=('S1 passes the predefined two-seed ranking/F1/TP gate; a matched unfrozen confirmation may be discussed, but is not launched.' if all(gates) else 'S1 does not pass the predefined two-seed ranking/F1/TP gate. Do not promote the sampler or launch unfrozen/R08 training automatically.')
    lines+=['',conclusion,'',
      'Interpretation: current sampling does not omit these matched negatives. This study tests a bounded change in their relative loss contribution. The prior attribution assigned about 90.7–90.8% of missed GT edges to absent recovered endpoint pairs; that historical budget still bounds relation-only recovery here because node proposals remain fixed. This experiment cannot causally establish that query representations are the sole remaining problem. Potential false-negative flags in the audit reflect conflicting spatial identities, not proof of incorrect annotations.','',
      'Only 18 reliable close negatives occur in the 128 training snapshots per seed (17 patches), so this is a limited-data intervention. Frozen clean eval-mode features do not reproduce augmented train-mode query distributions. Two continuation seeds share source history; development validation is reused. Whole-subject intervals do not quantify source-training uncertainty. Reported same-population AP and PR diagnostics must not be compared across differing candidate prevalence. Long edges use the training arc-length upper-quartile boundary.','',
      'Operating points: fixed 0.5, training-max positional F1 and training-selected threshold meeting the original training precision; thresholds are applied unchanged to validation. Validation-only PR comparisons at the original precision/recall are explicitly diagnostic. Original node threshold remains 0.5. Common-v2 uses 1-mm anchor/geometry support with 0.25-mm arc sampling, independently of 1/2-mm positional matching. Curves use the unchanged frozen K8/CNN geometry for every retained pair.','',
      'Detailed early learning curves, training/validation AP, 1/2-mm positional metrics, pair counts/prevalence, per-subject results, graph biases, loss/multiplicity histories and whole-subject uncertainty are saved in `../gnbm-r07-matched-supervision-20261027/experiments/r07_matched_supervision_20261027`. Checkpoint/cache/source identities and accepted jobs are in `plan.json`, `code-identities.json` and `submission.json`.','',
      'Retrieve from projects: `python3 gnbm-r07-matched-supervision-20261027/scripts/show_r07.py`. No subsequent experiment is automatically authorized.']
    block='\n'.join(lines)+'\n';(OUT/'report.md').write_text(block)
    report=ORIGINAL/'docs/research/metrics/8-ixi-relation-matching-calibration.md'
    registry=ORIGINAL/'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md'
    with (OUT/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker='<!-- R07 COLLECTED 20261027 -->';before=report.read_text()
        if marker not in before:report.write_text(before.rstrip()+'\n\n'+marker+'\n'+block)
        old=registry.read_text();rows=old.splitlines();found=0
        for i,row in enumerate(rows):
            if row.startswith('| **R07** |'):
                rows[i]='| **R07** | Modern SyntheticMRI reliable matched-negative redistribution; original CE/ratio and fixed effective pair budget; two frozen A-xyz seeds | **DONE — '+('measured' if all(gates) else 'negative/mixed')+'** | '+conclusion+' All matched negatives already enter S0; S1 tests bounded multiplicity, not missing exposure. | [R07 collected report](../docs/research/metrics/8-ixi-relation-matching-calibration.md#r07--modern-syntheticmri-matched-pair-negative-redistribution--27-october-2026) |';found+=1
            elif row.startswith('| Did we test *modern reliable near-nonadjacent matched-negative sampling*'):
                rows[i]='| Did we test *modern reliable near-nonadjacent matched-negative sampling* with edge CE? | ✅ | R07 completed as a bounded two-seed fixed-detector multiplicity comparison; every matched negative already enters S0. '+conclusion+' | R07 |'
            elif row.startswith('| **R07/R08 —'):
                rows[i]='| **R07/R08 — modern reliable matched/unmatched negative supervision** | **R07 DONE**, collected modern matched-pair comparison. **R08 NOT RUN**; no unmatched supervision launched. | '+conclusion+' |'
            elif row.startswith('### Q3 — R07:'):
                rows[i]='### Q3 — R07: reliable nearby nonadjacent matched-pair supervision — DONE, collected 27 October 2026; original scope below retained for provenance'
        assert found==1
        if marker not in old:rows+=['',marker,conclusion+' No R08 or unfrozen confirmation launched. See existing relation-supervision report.']
        # Preserve unrelated concurrent updates; defer if a file changed while read.
        assert registry.read_text()==old,'Concurrent registry edit; saved report.md remains available for integration'
        registry.write_text('\n'.join(rows)+'\n')
    write(OUT/'publication.json',dict(completed=True,report=str(report),registry=str(registry),passed_gate=all(gates)))
    claim=ROOT.parent/'experiment_gpu_coordination/r07-matched-supervision-20261027.json'
    c=read(claim);c.update(state='completed; collector appended results',concurrent_h100=0);write(claim,c)


if __name__=='__main__':run()
