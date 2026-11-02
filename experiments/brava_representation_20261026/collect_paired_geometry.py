"""Supplementary subject summaries and GT-conditioned geometry uncertainty."""
from pathlib import Path
import json, random
ROOT=Path(__file__).resolve().parent
rows=[];geometry=[]
for group in ['original','split']:
    folder=ROOT/'evaluation'/group/'004000'
    rows+=json.loads((folder/'rows.json').read_text())
    geometry+=json.loads((folder/'geometry.json').read_text())
subjects=sorted({r['subject'] for r in rows if r['split']=='val'})
per_subject={}
div=lambda a,b:a/b if b else 0
for arm in ['K8','K24','C1','split_K8']:
    per_subject[arm]={}
    for subject in subjects:
        rr=[r for r in rows if r['arm']==arm and r['subject']==subject and r['split']=='val' and r['tolerance']=='2.0']
        total=lambda key:sum(r[key] for r in rr)
        p=div(total('common_curve_supported_pred_samples'),total('common_curve_pred_samples'))
        q=div(total('common_curve_supported_gt_samples'),total('common_curve_gt_samples'))
        per_subject[arm][subject]=dict(patches=len(rr),raw_original_anchor_recall=div(total('raw_original_anchor_tp'),total('original_anchors')),original_anchor_recall=div(total('filtered_original_anchor_tp'),total('original_anchors')),common_curve_f1=div(2*p*q,p+q),common_branch_f1=div(2*total('common_branch_tp'),2*total('common_branch_tp')+total('common_branch_fp')+total('common_branch_fn')),cycles_mae=div(total('beta1_error'),len(rr)))
(ROOT/'per-subject-results.json').write_text(json.dumps(per_subject,indent=2)+'\n')
rng=random.Random(261039)
draws=[rng.choices(subjects,k=len(subjects)) for _ in range(1000)]
def quantile(x,p):
    x=sorted(x);v=(len(x)-1)*p;i=int(v)
    return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(v-i)
result={}
for arm in ['K24','C1','split_K8']:
    result[arm]={}
    for label in ['all','difficult']:
        pair={}
        for name in ['K8',arm]:
            rr=[r for r in geometry if r['arm']==name and r['split']=='val' and (label=='all' or r['difficult'])]
            pair[name]={s:[r for r in rr if r['subject']==s] for s in subjects}
        keys=lambda name:{(r['sample'],r['parent']) for rr in pair[name].values() for r in rr}
        assert keys('K8')==keys(arm)
        metrics={}
        for metric in ['mean_mm','neutral_mean_mm','oracle_endpoint_mean_mm']:
            sums={name:{s:(sum(r[metric] for r in rr),len(rr)) for s,rr in pair[name].items()} for name in pair}
            def delta(ss):
                means={name:sum(d[s][0] for s in ss)/sum(d[s][1] for s in ss) for name,d in sums.items()}
                return means[arm]-means['K8']
            values=[delta(ss) for ss in draws]
            metrics[metric]=dict(delta=delta(subjects),ci95=[quantile(values,.025),quantile(values,.975)])
        result[arm][label]=dict(paired_parent_branches=len(keys(arm)),metrics=metrics)
(ROOT/'geometry-paired-uncertainty.json').write_text(json.dumps(dict(subjects=subjects,draws=1000,seed=261039,rng='Python random.Random; separate from primary NumPy common-metric bootstrap',comparisons=result),indent=2)+'\n')
