"""Collect paired subject uncertainty and print completed IXI node-focal results."""
import argparse
import json
from pathlib import Path


def collect(root, smoke=False):
    import csv
    import shutil
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from scripts.ixi_node_focal_ablation import ARMS
    from scripts.ixi_cached_mechanisms import table, save
    from scripts.ixi_query_diagnostics import summarize
    from metrics.common_vessel_metrics import aggregate_common_vessels
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = root/('smoke-evaluation' if smoke else 'evaluation'); rows = {}; curves = []; final = {}; unique = []
    states = []
    for arm in ARMS:
        if not smoke: states.append(json.loads((root/'runs'/arm/'summary.json').read_text()))
        folders = sorted((base/arm).iterdir())
        for folder in folders:
            summary = json.loads((folder/'summary.json').read_text()); assert summary['completed']
            for r in summary['comparison']: curves.append(dict(arm=arm, step=summary['step'], **r))
            with (folder/'uniqueness.csv').open() as f:
                unique.extend(dict(arm=arm,step=summary['step'],**r) for r in csv.DictReader(f))
        last = folders[-1]; final[arm] = json.loads((last/'summary.json').read_text())
        rows[arm] = json.loads((last/'per-patch.json').read_text())
    if states:
        for key in ['updates','batch_order_sha256','initial_model_sha256']:
            assert len({s[key] for s in states}) == 1
        assert all(s['trainable_names'] == states[0]['trainable_names'] for s in states)
    ids = [(r['split'],r['sample_id'],r['tolerance_mm']) for r in rows[ARMS[0]]]
    assert all([(r['split'],r['sample_id'],r['tolerance_mm']) for r in rows[a]] == ids for a in ARMS)
    subjects = sorted({r['subject'] for r in rows[ARMS[0]] if r['split'] == 'val'})
    draws = np.random.default_rng(191032).choice(subjects, size=(40 if smoke else 1000,len(subjects)))
    def metrics(rr):
        s = summarize(rr); geo = aggregate_common_vessels([{k:v for k,v in r.items() if k.startswith('common_')} for r in rr])
        matched = sum(r['retained_matched'] for r in rr)
        auc = [r['raw_correct_vs_duplicate_auc'] for r in rr if r['raw_correct_vs_duplicate_auc'] is not None]
        result = dict(node_recall=s['node_recall'],node_precision=s['node_precision'],edge_f1=s['f1'],
            branch_f1=geo['common_branch_f1'],curve_f1=geo['common_curve_f1'],
            duplicates_per_recovered_gt=sum(r['retained_duplicates'] for r in rr)/matched if matched else np.nan,
            high_confidence_duplicates_per_patch=float(np.mean([r['retained_high_confidence_duplicates'] for r in rr])),
            node_count_mae=float(np.mean([r['node_count_absolute_error'] for r in rr])),
            mean_patch_correct_duplicate_auc=float(np.mean(auc)) if auc else np.nan,
            beta0_mae=s['beta0_mae'],beta1_mae=s['beta1_mae'])
        return {k: np.nan if v is None else v for k,v in result.items()}
    intervals = []
    for tolerance in [1,2]:
        distributions = {}
        for arm in ARMS:
            by_subject = {s:[r for r in rows[arm] if r['split'] == 'val' and r['tolerance_mm'] == tolerance and r['subject'] == s] for s in subjects}
            distributions[arm] = [metrics([r for s in draw for r in by_subject[s]]) for draw in draws]
        for arm, reference in [('ce_equal','ce_weighted'),('focal_equal','ce_equal'),('focal_equal','ce_weighted')]:
            for metric in distributions[arm][0]:
                values = np.array([a[metric]-b[metric] for a,b in zip(distributions[arm],distributions[reference])],float)
                values = values[np.isfinite(values)]
                intervals.append(dict(arm=arm,reference=reference,tolerance_mm=tolerance,metric=metric,
                    delta_95pct=np.quantile(values,[.025,.975]).tolist() if len(values) else None,subjects=len(subjects),resamples=len(draws)))
    output = root/('smoke-results' if smoke else 'results'); output.mkdir(exist_ok=True)
    table(output/'learning-curves.csv',curves); table(output/'uniqueness.csv',unique); save(output/'bootstrap.json',intervals)
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for arm in ARMS:
        for split, style in [('train','--'),('val','-')]:
            rr = [r for r in curves if r['arm'] == arm and r['split'] == split and r['tolerance_mm'] == 2]
            axes[0].plot([r['step'] for r in rr],[r['node_recall'] for r in rr],style,label=arm+'/'+split)
            axes[1].plot([r['step'] for r in rr],[r['common_curve_f1'] for r in rr],style,label=arm+'/'+split)
    axes[0].set(xlabel='Update',ylabel='Positional node recall (2 mm)');axes[1].set(xlabel='Update',ylabel='Common-v2 Curve F1')
    axes[0].legend(fontsize=6);fig.savefig(output/'learning-curves.png',dpi=160);plt.close(fig)
    lines = ['## Completed results','', '|Arm|Population|Node precision|Node recall|Edge F1|Branch F1|Curve F1|Component MAE|Cycle MAE|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for r in final[arm]['comparison']:
            if r['tolerance_mm'] == 2:
                lines.append('|'+arm+'|'+r['split']+'|'+'|'.join('NA' if r[k] is None else f'{r[k]:.4f}' for k in ['node_precision','node_recall','f1','common_branch_f1','common_curve_f1','beta0_mae','beta1_mae'])+'|')
    lines += ['', 'Paired subject intervals are in `bootstrap.json`; uniqueness and all intermediate training/validation metrics are in the companion CSVs. A vs B isolates class weighting, B vs C isolates focal modulation. Spatial correct-versus-duplicate AUC is diagnostic, not an inference feature. Pooled raw AUC is reported separately from the bootstrap mean per-patch AUC.', '',
        'This is a short continuation pilot conditioned on a mature focal-edge checkpoint and a common transition to edge CE/ratio upsampling. Inspect learning-curve trends before drawing convergence conclusions. The node loss can change Hungarian assignments and hence which queries represent supervised endpoints; the matching algorithm and edge candidate policy remain identical. Do not interpret higher node AP/recall as improved uniqueness unless duplicate multiplicity and graph false edges also support it. No result here is a test-set estimate.', '',
        'Next decision: if B improves over A while C does not improve over B, prioritize weighting rather than focal modulation. If C improves uniqueness over B without losing legitimate-node recall, confirm with a longer matched-budget study. If gains occur only on training subjects or curves have not stabilized, investigate generalization or extend the learning-curve budget before rejecting focal loss. No edge-loss or architecture sweep is automatically warranted.', '', f'Output: `{root}`.','']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    save(output/'summary.json',dict(completed=True,smoke=smoke,results=final,bootstrap=intervals))
    if not smoke:
        evidence = Path('docs/research/evidence/ixi_node_focal_20261020'); evidence.mkdir(parents=True,exist_ok=True)
        for p in output.iterdir(): shutil.copy2(p,evidence/p.name)
        shutil.copy2(root/'benchmark/budget.json',evidence/'budget.json')
        for arm in ARMS:
            target=evidence/arm;target.mkdir(exist_ok=True)
            for name in ['summary.json','history.json','resolved-config.json']: shutil.copy2(root/'runs'/arm/name,target/name)
            last=sorted((base/arm).iterdir())[-1]
            for name in ['per-subject.csv','uniqueness.csv']:shutil.copy2(last/name,target/name)
        doc=Path('docs/research/metrics/10-ixi-node-focal-ablation.md')
        doc.write_text(doc.read_text().split('## Completed results')[0]+(output/'results.md').read_text()+'\n![Learning curves](../evidence/ixi_node_focal_20261020/learning-curves.png)\n')
        with Path('docs/research/IXI_graph_extraction_diagnostic_roadmap.md').open('a') as f:f.write('\nCompleted the controlled IXI node class-weight/focal continuation; see [report 10](metrics/10-ixi-node-focal-ablation.md). Interpret learning curves and uniqueness before extending loss studies; no new edge sweep was included.\n')
    print(output/'results.md')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--show',action='store_true');a=p.parse_args()
    root=a.root or Path(json.loads(Path('experiments/ixi_node_focal_ablation/submission.json').read_text())['output_root'])
    if a.show:
        path=root/'results/results.md'
        if not path.exists():raise SystemExit(f'Results pending; no waiting. Expected {path}')
        print(path.read_text())
    else:collect(root,a.smoke)
