"""Paired subject summaries for the raw-MRA/binary vessel-input diagnostic."""
import argparse
import json
from pathlib import Path


def collect(root,smoke=False):
    import csv
    import shutil
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from scripts.ixi_cached_mechanisms import save,table
    from scripts.ixi_query_diagnostics import summarize
    from scripts.ixi_segmentation_oracle import ARMS
    from metrics.common_vessel_metrics import aggregate_common_vessels
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evaluation=root/('smoke-evaluation' if smoke else 'evaluation');curves=[];raw={};native={};selected={};states=[]
    for arm in ARMS:
        folders=sorted((evaluation/arm).iterdir());ranking=[]
        if not smoke:states.append(json.loads((root/'runs'/arm/'summary.json').read_text()))
        for folder in folders:
            summary=json.loads((folder/'summary.json').read_text());assert summary['completed']
            for r in summary['comparison']:curves.append(dict(arm=arm,step=summary['step'],**r))
            val=next(r for r in summary['comparison'] if r['split']=='val' and r['tolerance_mm']==2)
            ranking.append((val['common_branch_f1'],val['common_curve_f1'],val['f1'] or 0.,-summary['step'],folder))
        selected[arm]=dict(step=int(max(ranking)[4].name),rule='validation common Branch F1, then Curve F1, then positional edge F1; earlier update on ties')
        raw[arm]=json.loads((folders[-1]/'per-patch.json').read_text())
        native[arm]=json.loads((folders[-1]/'native-summary.json').read_text())
    if states:
        for key in ['initial_model_sha256','updates','batch_order_sha256']:assert len({s[key] for s in states})==1
        assert states[0]['trainable_names']==states[1]['trainable_names']
    keys=lambda rr:[(r['split'],r['sample_id'],r['tolerance_mm']) for r in rr]
    assert keys(raw['raw_mra'])==keys(raw['vessel_mask'])
    subjects=sorted({r['subject'] for r in raw['raw_mra'] if r['split']=='val'})
    rng=np.random.default_rng(191032);draws=rng.choice(subjects,size=(40 if smoke else 1000,len(subjects)))
    def aggregate(rr):
        s=summarize(rr);g=aggregate_common_vessels([{k:v for k,v in r.items() if k.startswith('common_')} for r in rr])
        matched=sum(r['retained_matched'] for r in rr)
        result={k:s[k] for k in ['node_precision','node_recall','node_f1','precision','recall','f1','conditional_recall','beta0_mae','beta1_mae']}
        result.update(branch_f1=g['common_branch_f1'],curve_f1=g['common_curve_f1'],
            duplicates_per_recovered_gt=sum(r['retained_duplicates'] for r in rr)/matched if matched else None,
            high_confidence_duplicates_per_patch=float(np.mean([r['retained_high_confidence_duplicates'] for r in rr])),
            raw_query_recall=sum(r['raw_matched'] for r in rr)/sum(r['gt_nodes'] for r in rr),
            node_count_bias=float(np.mean([r['node_count_bias'] for r in rr])),node_count_mae=float(np.mean([r['node_count_absolute_error'] for r in rr])),
            unmatched_false_edges_per_patch=float(np.mean([r['MU_fp']+r['UU_fp'] for r in rr])))
        return {k:float(v) if v is not None else np.nan for k,v in result.items()}
    intervals=[];by_subject=[]
    for tol in [1,2]:
        arms={a:{s:[r for r in raw[a] if r['split']=='val' and r['subject']==s and r['tolerance_mm']==tol] for s in subjects} for a in ARMS}
        for arm in ARMS:
            for s,rr in arms[arm].items():by_subject.append(dict(arm=arm,subject=s,tolerance_mm=tol,**aggregate(rr)))
        boot={a:[aggregate([r for s in draw for r in arms[a][s]]) for draw in draws] for a in ARMS}
        point={a:aggregate([r for rr in arms[a].values() for r in rr]) for a in ARMS}
        for metric in point['raw_mra']:
            delta=np.array([b[metric]-a[metric] for a,b in zip(boot['raw_mra'],boot['vessel_mask'])]);valid=delta[np.isfinite(delta)]
            intervals.append(dict(tolerance_mm=tol,metric=metric,raw=point['raw_mra'][metric],mask=point['vessel_mask'][metric],
                delta_mask_minus_raw=point['vessel_mask'][metric]-point['raw_mra'][metric],
                delta_95pct=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,subjects=len(subjects),resamples=len(draws)))
    def safe(v):
        from scripts.evaluate_ixi_segmentation_oracle import safe as clean
        return clean(v)
    output=root/('smoke-results' if smoke else 'results');output.mkdir(exist_ok=True)
    table(output/'learning-curves.csv',curves);table(output/'per-subject.csv',safe(by_subject));save(output/'paired-bootstrap.json',safe(intervals));save(output/'checkpoint-selection.json',selected)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for arm in ARMS:
        for split,style in [('train','--'),('val','-')]:
            rr=[r for r in curves if r['arm']==arm and r['split']==split and r['tolerance_mm']==2]
            for ax,key,label in zip(axes,['node_recall','f1','common_curve_f1'],['Node recall (2 mm)','Positional edge F1 (2 mm)','Common-v2 Curve F1']):
                ax.plot([r['step'] for r in rr],[r[key] for r in rr],style,label=arm+'/'+split);ax.set(xlabel='Update',ylabel=label)
    axes[0].legend(fontsize=7);fig.savefig(output/'learning-curves.png',dpi=160);plt.close(fig)
    audit=json.loads((root/'audit/manifest.json').read_text())
    lines=['## Completed results','',f'Paired optimization population: {audit["counts"]}; subjects: {audit["subject_counts"]}. Evaluation uses the fixed training monitoring subset and complete paired validation manifest.','',
        '|Input|Final validation node mAP|Node mAR|Edge mAP|Edge mAR|SMD|','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        n=next(r for r in native[arm] if r['split']=='val' and r['subject'] is None)
        lines.append('|'+arm+'|'+'|'.join('NA' if n[k] is None else f'{n[k]:.4f}' for k in ['node_mAP','node_mAR','edge_mAP','edge_mAR','smd'])+'|')
    lines+=['','|Input|Node precision|Node recall|Edge F1|Conditional relation recall|Branch F1|Curve F1|Component MAE|Cycle MAE|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rr=[r for r in raw[arm] if r['split']=='val' and r['tolerance_mm']==2];s=aggregate(rr)
        lines.append('|'+arm+'|'+'|'.join('NA' if not np.isfinite(s[k]) else f'{s[k]:.4f}' for k in ['node_precision','node_recall','f1','conditional_recall','branch_f1','curve_f1','beta0_mae','beta1_mae'])+'|')
    lines+=['','### Interpretation','',
        'This is reconstruction from supplied masks that generated the graph references. It measures appearance removal under correlated supervision, not performance from independently annotated anatomical labels or a deployable raw-image pipeline.',
        'Use paired-bootstrap.json to assess localization/recovery, duplicate multiplicity, conditional connectivity and false cycles separately. Node-only gains with persistently weak conditional relations implicate downstream graph reasoning; gains accompanied by duplicates or loops do not establish faithful reconstruction. Little gain in a short run does not establish an architectural impossibility.',
        'Primary comparison is the matched final update. checkpoint-selection.json records a validation-only best checkpoint per arm; it is secondary and need not select the same update. Evaluation folders retain raw fixed-cutoff metrics and final training-calibrated policies. No test labels were used.',
        'Original physical matching and common-v2 definitions are retained. Native box AP/AR and normalized straight-edge SMD have different meanings from positional connectivity and continuous-curve metrics. The pretrained spline head was frozen in this stable node/relation training configuration, so Curve F1 also reflects that geometry decoder constraint.',
        'The common SyntheticMRI initializer already learned raw-image features. There is no prior IXI exposure, but it remains a modality pretraining bias; both arms train their CNN/BN state. Any absence of convergence or severe generalization gap calls for a bounded convergence check before rejecting segmentation input.',
        f'Outputs: `{root}`. Detailed native/physical per-patch and per-subject tables, localization.csv, uniqueness.csv, thresholds.json and final calibration/ tables are retained.','']
    (output/'results.md').write_text('\n'.join(lines)+'\n');save(output/'summary.json',safe(dict(completed=True,smoke=smoke,paired_bootstrap=intervals,selection=selected,native=native)))
    if not smoke:
        evidence=Path('docs/research/evidence/ixi_segmentation_input_20261020');evidence.mkdir(parents=True,exist_ok=True)
        for p in output.iterdir():shutil.copy2(p,evidence/p.name)
        for p in (root/'audit').glob('*.png'):shutil.copy2(p,evidence/p.name)
        for name in ['manifest.json','alignment.csv']:shutil.copy2(root/'audit'/name,evidence/('audit-'+name))
        shutil.copy2(root/'benchmark/budget.json',evidence/'budget.json')
        for arm in ARMS:
            dest=evidence/arm;dest.mkdir(exist_ok=True);last=sorted((evaluation/arm).iterdir())[-1]
            for name in ['summary.json','history.json','resolved-config.json']:shutil.copy2(root/'runs'/arm/name,dest/name)
            for name in ['native-summary.json','localization.csv','uniqueness.csv','thresholds.json','per-subject.csv']:shutil.copy2(last/name,dest/name)
            shutil.copy2(last/'calibration/comparison.csv',dest/'train-calibrated-comparison.csv')
        doc=Path('docs/research/metrics/11-ixi-segmentation-input-oracle.md')
        doc.write_text(doc.read_text().split('## Completed results')[0]+(output/'results.md').read_text()+'\n![Learning curves](../evidence/ixi_segmentation_input_20261020/learning-curves.png)\n')
    print(output/'results.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--show',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root=a.root or Path(json.loads(Path('experiments/ixi_segmentation_input/submission.json').read_text())['output_root'])
    if a.show:
        path=root/'results/results.md'
        if not path.exists():raise SystemExit(f'Results pending; no waiting. Expected {path}')
        print(path.read_text())
    else:collect(root,a.smoke)
