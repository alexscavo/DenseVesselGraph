"""Collect existing BRAVA Stage 1 metrics; --show never reruns computation."""
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
    from scripts.evaluate_brava_study import aggregate,clean
    from scripts.brava_graph_study import ARMS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gt_analysis=None if smoke else json.loads((root/'representations/summary.json').read_text())
    if gt_analysis is not None:assert gt_analysis['completed'] and not gt_analysis['test_evaluated']
    plan=json.loads((root/'plan.json').read_text());names=list(plan['checkpoints'])+ARMS
    summaries={};raw={};curves=[];selected={};states=[]
    evaluation=root/('smoke-evaluation' if smoke else 'evaluation')
    for name in names:
        folders=sorted((evaluation/name).iterdir());choices=[]
        for folder in folders:
            m=json.loads((folder/'summary.json').read_text());assert m['completed'] and m['same_graph_verified']
            for row in m['comparison']:curves.append(dict(model=name,checkpoint=folder.name,**row))
            if name in ARMS:
                v=next(r for r in m['comparison'] if r['split']=='val' and r['tolerance_mm']==2 and r['geometry']=='spline')
                choices.append((v['common_branch_f1'],v['common_curve_f1'],v['f1'] or 0.,-int(folder.name),folder.name))
        if choices:selected[name]=dict(checkpoint=max(choices)[4],rule='full BRAVA validation Branch F1, Curve F1, positional edge F1; earlier on ties')
        last=folders[-1];summaries[name]=json.loads((last/'summary.json').read_text());raw[name]=json.loads((last/'per-patch.json').read_text())
        if name in ARMS and not smoke:states.append(json.loads((root/'runs'/name/'summary.json').read_text()))
    if states:
        for key in ['common_initial_state','order_sha256','updates']:assert states[0][key]==states[1][key]
    subjects=sorted({r['subject'] for r in raw['standard'] if r['split']=='val'});rng=np.random.default_rng(211039)
    draws=rng.choice(subjects,size=(30 if smoke else 1000,len(subjects)))
    intervals=[]
    metrics=['node_precision','node_recall','f1','conditional_recall','common_branch_f1','common_curve_f1','beta0_mae','beta1_mae','duplicates_per_recovered_gt']
    for tol in [1,2]:
        lookup={name:{s:[r for r in rr if r['split']=='val' and r['subject']==s and r['geometry']=='spline' and r['tolerance_mm']==tol] for s in subjects} for name,rr in raw.items()}
        boot={name:[aggregate([r for s in draw for r in groups[s]]) for draw in draws] for name,groups in lookup.items()}
        for metric in metrics:
            delta=np.array([b[metric]-a[metric] if a[metric] is not None and b[metric] is not None else np.nan for a,b in zip(boot['standard'],boot['coordinate_refinement'])])
            good=delta[np.isfinite(delta)]
            intervals.append(dict(comparison='coordinate_refinement minus standard',tolerance_mm=tol,metric=metric,
                subjects=len(subjects),resamples=len(draws),delta_95pct=np.quantile(good,[.025,.975]).tolist() if len(good) else None))
        for name in names:
            delta=[]
            for draw in draws:
                rr=[r for s in draw for r in raw[name] if r['split']=='val' and r['subject']==s and r['tolerance_mm']==tol and r['geometry']=='chord']
                chord=aggregate(rr)['common_curve_f1']
                i=len(delta);delta.append(boot[name][i]['common_curve_f1']-chord)
            intervals.append(dict(comparison=name+': same-graph spline minus chord',tolerance_mm=tol,metric='common_curve_f1',subjects=len(subjects),
                resamples=len(draws),delta_95pct=np.quantile(delta,[.025,.975]).tolist()))
    out=root/('smoke-results' if smoke else 'results');out.mkdir(exist_ok=True)
    table(out/'learning-curves.csv',clean(curves));save(out/'paired-bootstrap.json',clean(intervals));save(out/'checkpoint-selection.json',selected)
    lines=['## Completed Stage 1 results','',
        'All scores are relative to available manual annotations. Reference-unmatched predictions are not confirmed anatomical false positives. Test subjects were not evaluated. Primary trained comparison uses matched final updates; secondary checkpoints are selected on the full eight-subject validation split.','',
        '|Model|Queries|Node recall 2 mm|Edge F1 2 mm|Branch F1|Curve F1 spline|Curve F1 same-graph chord|Component MAE|Cycle MAE|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        m=summaries[name];r=next(r for r in m['comparison'] if r['split']=='val' and r['tolerance_mm']==2 and r['geometry']=='spline')
        chord=next(r for r in m['comparison'] if r['split']=='val' and r['tolerance_mm']==2 and r['geometry']=='chord')
        pred=root/('smoke-predictions' if smoke else 'predictions')/name/m['checkpoint']/'summary.json';q=json.loads(pred.read_text())['object_queries']
        vals=[r['node_recall'],r['f1'],r['common_branch_f1'],r['common_curve_f1'],chord['common_curve_f1'],r['beta0_mae'],r['beta1_mae']]
        lines.append('|'+name+'|'+str(q)+'|'+'|'.join('NA' if v is None else f'{v:.4f}' for v in vals)+'|')
    lines+=['','|Model|Recovered connections at 2 mm|Chord error mm|Spline error mm|','|---|---:|---:|---:|']
    for name,m in summaries.items():
        r=next(r for r in m['same_connection_geometry'] if r['split']=='val' and r['tolerance_mm']==2 and r['subject'] is None)
        lines.append('|'+name+'|'+str(r['connections'])+'|'+'|'.join('NA' if r[k] is None else f'{r[k]:.4f}' for k in ['chord_error_mm','spline_error_mm'])+'|')
    if gt_analysis is not None:
        lines+=['','### GT representation analysis (not learned predictions)','','All geometry is evaluated against original dense development centerlines. Spline rows are oracle fits, not model predictions; fewer anchors alone do not demonstrate learned query efficiency.','','|Split|Representation|Mean nodes|Mean edges|Branch F1|Curve F1|Guide distance mm|Component error|Cycle error|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in gt_analysis['comparisons']:
            values=[r[k] for k in ['mean_object_nodes','mean_edges','common_branch_f1','common_curve_f1','common_guide_distance_mm','common_betti0_abs_error_mean','common_betti1_abs_error_mean']]
            lines.append('|'+r['split']+'|'+r['representation']+'|'+'|'.join('NA' if v is None else f'{v:.4f}' for v in values)+'|')
    lines+=['','### Paired subject-bootstrap uncertainty','','95% percentile intervals;1,000 resamples of eight subjects, seed211039. Values are differences on the0–1 scale. These intervals do not measure training-seed uncertainty.','','|Comparison|Metric (2 mm positional protocol)|95% interval|','|---|---|---|']
    for r in intervals:
        if r['tolerance_mm']!=2:continue
        ci=r['delta_95pct'];label='NA (undefined in eligible resamples)' if ci is None else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]'
        lines.append('|'+r['comparison']+'|'+r['metric']+'|'+label+'|')
    lines+=['','### Recovery and interpretation','','The full-volume analysis originally failed because the source reader deduplicated a coincident SWC parent/child pair to a one-point path. The analysis now explicitly writes both coincident endpoints, retaining the annotated edge without inventing spatial extent, changing common-v2 metrics, or modifying training data. Completed subject analyses are checkpointed for safe CPU-only resumption. Original failure1374051 and cancelled collector1374060 remain in accounting; recovery job IDs are in `experiments/brava_graph_study/recovery.json`.',
        'Both scratch arms completed23,500 updates on8,768 training patches, with all1,796 validation patches evaluated. IXI zero-shot is the strongest measured starting point; its differing pretraining history prevents a controlled causal claim about initialization. Scratch training yields too few recovered connections to establish whether learned spline geometry or refinement is beneficial. Refinement increases raw-query coverage but reduces filtered node recall in this pilot. These runs do not establish convergence. No new inference, training or Stage2 experiment was needed for reporting recovery.']
    recovery=Path('experiments/brava_graph_study/recovery.json')
    if not smoke and recovery.exists():
        jobs=json.loads(recovery.read_text())['jobs']
        lines+=['','Recovery jobs: '+jobs['representations']['id']+' (CPU GT analysis) → '+jobs['collection']['id']+' (CPU collection, afterok). No training or inference rerun.']
    lines+=['','### Interpretation boundaries','',

        'Same-graph geometry comparisons share identical nodes, edges and scores; differences in Curve F1/guide distance isolate geometry. Branch connectivity must be unchanged. Recovered-connection distance tables quantify spline/chord error on exactly the same direct GT connections, without GT input features or GT endpoint replacement.',
        'The two scratch-trained arms isolate the enabled coordinate refinement module, including its added parameters; logits receive no local residual. Both use the full same training population and matched updates. Their initial shared weights and outputs are checked, but these single-seed development results do not establish convergence or robustness across seeds.',
        'Zero-shot models differ in source training history and may have different query capacities; they are starting points, not an isolated causal comparison of source domains. Source straight-versus-spline checkpoints are independently trained current-compatible controls, not a verified rerun of the exact original GrEVe checkpoint.',
        'Inspect per-subject errors, duplicates, conditional recall, native detection and same-connection geometry alongside coverage. Better SMD/Curve F1 or fewer cycles alone cannot establish correct anatomy. Native SMD samples straight edges; continuous geometry is measured separately. No source-transfer, separate straight-training or query-budget arm was automatically launched.','',f'Output root: `{root}`.','']
    (out/'results.md').write_text('\n'.join(lines)+'\n');save(out/'summary.json',clean(dict(completed=True,results=summaries,paired_bootstrap=intervals,selected=selected)))
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for name in ARMS:
        for split,style in [('train','--'),('val','-')]:
            rr=[r for r in curves if r['model']==name and r['split']==split and r['geometry']=='spline' and r['tolerance_mm']==2]
            for ax,key in zip(axes,['node_recall','f1','common_curve_f1']):ax.plot([int(r['checkpoint']) for r in rr],[r[key] for r in rr],style,label=name+'/'+split);ax.set(xlabel='Update',ylabel=key)
    axes[0].legend(fontsize=6);fig.savefig(out/'learning-curves.png',dpi=150);plt.close(fig)
    if not smoke:
        evidence=Path('docs/research/evidence/brava_stage1_20261021');evidence.mkdir(parents=True,exist_ok=True)
        for path in out.iterdir():shutil.copy2(path,evidence/path.name)
        for name in ['audit/summary.json','audit/volume-representations.csv','representations/summary.json','representations/per-subject.csv','benchmark/budget.json']:
            p=root/name;shutil.copy2(p,evidence/('-'.join(p.parts[-2:])))
        doc=Path('docs/research/BRAVA_experiment_plan.md');doc.write_text(doc.read_text().split('## Completed Stage 1 results')[0]+(out/'results.md').read_text())
        marker='## BRAVA Stage 1 recovery — 22 October 2026'
        appendix=marker+'\n\nThis dated update supersedes the preliminary BRAVA completion review and earlier preparation-only status. Training and inference were reused; recovered CPU representation analysis and final aggregation completed.\n\n'+(out/'results.md').read_text()
        for name in ['docs/research/ALL_EXPERIMENTS_AND_RESULTS.md','docs/research/EXPERIMENT_REVIEW_20261021.md']:
            review=Path(name);review.write_text(review.read_text().split(marker)[0].rstrip()+'\n\n'+appendix)
        (evidence/'completion-review.md').write_text(appendix)
        status=evidence/'completion-review.json'
        previous=json.loads(status.read_text()) if status.exists() else {}
        previous.update(complete_model_evaluations=True,final_collector_complete=True,GT_representation_analysis_complete=True,results_path=str(out/'results.md'))
        status.write_text(json.dumps(previous,indent=2)+'\n')

    print(out/'results.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--show',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root=a.root or Path(json.loads(Path('experiments/brava_graph_study/submission.json').read_text())['output_root'])
    if a.show:
        path=root/'results/results.md'
        if not path.exists():raise SystemExit(f'Results pending; no waiting. Expected {path}')
        print(path.read_text())
    else:collect(root,a.smoke)
