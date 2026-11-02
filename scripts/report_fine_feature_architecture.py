"""Collect completed baseline/pilots, with paired subject uncertainty."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.report_multiscale_pretrained import paired,selected

ROOT=Path('experiments/fine_feature_architecture')
DOC=Path('docs/research/fine_feature_architecture_study.md')
ARMS=['B0','A','A-control','C']


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def contribution(arm,mode,normal_rows):
    directory=ROOT/'runs'/arm/('final-'+mode);rr=read(directory/'rows.json');ids={r['sample'] for r in rr};normal=[r for r in normal_rows if r['sample'] in ids]
    changes=[]
    for rec in read(directory/'records.json'):
        with np.load(rec['cache']) as b,np.load(ROOT/'runs'/arm/'final'/(rec['sample']+'.npz')) as a:
            assert np.array_equal(a['pairs'],b['pairs'])
            distance=np.linalg.norm(a['physical']-b['physical'],axis=-1);retained=a['scores']>.5
            changes.append(dict(displacement_mm=float(distance.mean()),retained_displacement_mm=float(distance[retained].mean()),probability_change=float(np.abs(a['scores']-b['scores']).mean()),relation_change=float(np.abs(a['relations']-b['relations']).mean()),retention_flips=int((retained!=(b['scores']>.5)).sum())))
    return dict(paired={t:paired(selected(normal,tol=t),selected(rr,tol=t)) for t in ['1.0','2.0']},prediction_change={k:float(np.mean([x[k] for x in changes])) for k in changes[0]},summary=read(directory/'summary.json'),reconstruction=read(ROOT/'runs'/arm/('reconstruction-'+mode)/'summary.json'))


def collect(partial=False):
    result=dict(plan=read(ROOT/'plan.json'),jobs=read(ROOT/'jobs.json'),split_audit=read(ROOT/'split-audit.json'))
    names=['historical','baseline']+ARMS;rows={};summary={};reconstruction={}
    for name in names:
        d=ROOT/'runs'/name/('selected' if name in ['historical','baseline'] else 'final')
        if (d/'summary.json').exists():
            summary[name]=read(d/'summary.json');rows[name]=read(d/'rows.json')
        p=ROOT/'runs'/name/'reconstruction'/'summary.json'
        if p.exists():reconstruction[name]=read(p)
    result.update(summary=summary,reconstruction=reconstruction)
    p=ROOT/'baseline-gate.json';result['baseline_gate']=read(p) if p.exists() else None
    p=ROOT/'runs'/'baseline'/'history.jsonl';result['baseline_history']=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    if 'baseline' in rows and 'historical' in rows:
        result['baseline_minus_historical']={t:paired(selected(rows['historical'],tol=t),selected(rows['baseline'],tol=t)) for t in ['1.0','2.0']}
    completed=all((ROOT/'runs'/a/'complete.json').exists() for a in ARMS)
    if completed:
        result['pilot_complete']={a:read(ROOT/'runs'/a/'complete.json') for a in ARMS}
        hashes={v['batch_sha256'] for v in result['pilot_complete'].values()};assert len(hashes)==1,'Pilot data/augmentation mismatch'
        configs={a:read(ROOT/'runs'/a/'config.json') for a in ARMS}
        for c in configs.values():
            c['experiment']['name']='matched';c['model']['node_refinement']={'enabled':False};c['model']['decoder']['feature_levels']=1;c['model']['decoder'].pop('fine_feature_stage',None)
        assert all(c==configs['B0'] for c in configs.values()),'Non-architecture configuration mismatch'
        result['step_zero_parity']={}
        initial_records=read(ROOT/'runs'/'B0'/'step-0000'/'records.json')
        for arm in ARMS[1:]:
            differences=[]
            for rec in initial_records:
                with np.load(rec['cache']) as a,np.load(ROOT/'runs'/arm/'step-0000'/(rec['sample']+'.npz')) as b:
                    for key in ['pairs','truth','gt_edges']:assert np.array_equal(a[key],b[key]),(arm,key)
                    differences.append(dict(coordinate_mm_max=float(np.linalg.norm(a['physical']-b['physical'],axis=-1).max()),object_probability_max=float(np.abs(a['scores']-b['scores']).max()),relation_probability_max=float(np.abs(a['relations']-b['relations']).max()),retention_flips=int(((a['scores']>.5)!=(b['scores']>.5)).sum())))
            result['step_zero_parity'][arm]=dict(patches=len(differences),maxima={k:max(d[k] for d in differences) for k in differences[0]},total_retention_flips=sum(d['retention_flips'] for d in differences))
            if arm in ['A','A-control']:assert all(v==0 for v in result['step_zero_parity'][arm]['maxima'].values())
        result['paired']={f'{b}-minus-{a}':{t:paired(selected(rows[a],tol=t),selected(rows[b],tol=t)) for t in ['1.0','2.0']} for a,b in [('baseline','B0'),('B0','A'),('A-control','A'),('B0','A-control'),('B0','C')]}
        result['interventions']={a:{mode:contribution(a,mode,rows[a]) for mode in (['zero','roll'] if a=='A' else ['blocked','roll'])} for a in ['A','C']}
        result['training']={a:read(ROOT/'runs'/a/'training.json') for a in ARMS}
        result['pilot_monitors']={a:{step:read(ROOT/'runs'/a/f'step-{step:04d}'/'summary.json') for step in [0,1000,2000]} for a in ARMS}
    result['pilots_completed']=completed
    if not partial:assert completed or result['baseline_gate'] is not None and not result['baseline_gate']['passed'],'Study still running'
    write(ROOT/'analysis.json',result);plot(result);render(result)


def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path('docs/research/evidence/fine_feature_architecture');out.mkdir(parents=True,exist_ok=True)
    history=[x for x in r['baseline_history'] if 'evaluation' in x]
    if history:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,key,title in zip(axes,['node_f1','edge_f1'],['Node F1 at 2 mm','Edge F1 at 2 mm']):
            for split in ['train','val']:
                ax.plot([h['epoch'] for h in history],[h['evaluation']['metrics'][split]['2.0'][key] for h in history],label=split,marker='.',markersize=4)
            if 'historical' in r['summary']:ax.axhline(r['summary']['historical']['metrics']['val']['2.0'][key],color='gray',linestyle=':',label='Historical validation')
            ax.set(xlabel='Scratch training epoch',ylabel='F1',title=title,ylim=(0,1));ax.grid(alpha=.2);ax.legend()
        fig.savefig(out/'baseline.png',dpi=180);fig.savefig(out/'baseline.pdf');plt.close(fig)
    if not r['pilots_completed']:return
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    names=['baseline']+ARMS
    for ax,key,title in zip(axes[0],['node_f1','edge_f1'],['Node F1 at 2 mm','Edge F1 at 2 mm']):
        values=[r['summary'][a]['metrics']['val']['2.0'][key] for a in names]
        ax.bar(names,values,color=['gray','tab:blue','tab:orange','tab:green','tab:red']);ax.set(title=title,ylabel='F1',ylim=(0,1));ax.grid(axis='y',alpha=.2)
    comparisons=['B0-minus-baseline','A-minus-B0','A-minus-A-control','C-minus-B0'];ax=axes[1,0]
    for i,c in enumerate(comparisons):
        v=r['paired'][c]['2.0']['metrics']['edge_f1'];lo,hi=v['ci95'];d=v['delta'];ax.errorbar(100*d,i,xerr=[[100*max(0,d-lo)],[100*max(0,hi-d)]],fmt='o',capsize=3,color='tab:blue')
    ax.set(yticks=range(len(comparisons)),yticklabels=comparisons,xlabel='Edge F1 difference, percentage points',title='Paired subject 95% intervals');ax.axvline(0,color='gray',linestyle=':');ax.grid(axis='x',alpha=.2)
    interventions=[('A','zero'),('A','roll'),('C','blocked'),('C','roll')];ax=axes[1,1]
    for i,(arm,mode) in enumerate(interventions):
        v=r['interventions'][arm][mode]['paired']['2.0']['metrics']['edge_f1'];lo,hi=v['ci95'];d=v['delta'];ax.errorbar(100*d,i,xerr=[[100*max(0,d-lo)],[100*max(0,hi-d)]],fmt='o',capsize=3,color='tab:orange')
    ax.set(yticks=range(4),yticklabels=[f'{a}: {m}' for a,m in interventions],xlabel='Intervention − normal edge F1, points',title='Does removing image information hurt?');ax.axvline(0,color='gray',linestyle=':');ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'pilots.png',dpi=180);fig.savefig(out/'pilots.pdf');plt.close(fig)


def render(r):
    lines=['## Measured results','']
    gate=r['baseline_gate']
    if gate is None:
        lines+=['Baseline training is still in progress; final quality-gate and architectural conclusions are pending.','']
    else:
        lines += [f"Baseline gate: **{'PASS' if gate['passed'] else 'FAIL'}**, selected epoch {gate['selected_epoch']}. Checkpoint: `{gate['selected_checkpoint']}`.",'', '| Gate, 2 mm | Value | Required | Passed |','|---|---:|---:|---|']
        for key,passed in gate['checks'].items():
            threshold=r['plan']['quality_gate']['hungarian_mm_max' if key=='hungarian_mm' else key]
            lines.append(f"| {key} | {gate['metrics'][key]:.4f} | {'≤' if key=='hungarian_mm' else '≥'} {threshold} | {passed} |")
        lines+=['']
    for tol in ['1.0','2.0']:
        lines += [f'### Physical node/edge metrics at {tol} mm','', '| Model | Raw recall | Filtered recall | Node precision | Node F1 | Edge F1 | Conditional edge recall | Hungarian mm |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,v in r['summary'].items():
            s=v['metrics']['val'][tol];lines.append('| '+name+' | '+' | '.join(f'{s[k]:.4f}' for k in ['raw_recall','node_recall','node_precision','node_f1','edge_f1','conditional_recall','hungarian_mm'])+' |')
        lines+=['']
    lines+=['### Existing native and common-v2 reconstruction protocols','', '| Model | Native node F1 | Native edge F1 | Branch F1 | Curve F1 | Same-endpoint chord Curve F1 |','|---|---:|---:|---:|---:|---:|']
    for name,s in r['reconstruction'].items():lines.append('| '+name+' | '+' | '.join(f'{s[k]:.4f}' for k in ['native_node_f1','native_edge_f1','common_branch_f1','common_curve_f1','same_endpoint_chord_common_curve_f1'])+' |')
    lines+=['','Historical is evaluated afresh on the identical 480 validation patches with the same code and thresholds. Its different training population, initializer history and loss recipe preclude attributing differences solely to node focal. Native F1 and physical one-to-one F1 are different protocols.','',
        '### Training/validation and confidence discrimination','',
        '| Model / split | Node F1, 2 mm | Edge F1, 2 mm | Duplicate fraction | Correct-vs-false AUROC | Correct-vs-duplicate AUROC | Reliable-label ECE | Component MAE | Cycle MAE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:f'{x:.4f}' if x is not None else 'n/a'
    for name,v in r['summary'].items():
        for split in ['train','val']:
            s=v['metrics'][split]['2.0'];d=v['discrimination'][split]
            vals=[s['node_f1'],s['edge_f1'],s['duplicate_rate'],d['auc_correct_false'],d['auc_correct_duplicate'],d['ece'],s['beta0_mae'],s['beta1_mae']]
            lines.append(f'| {name} / {split} | '+' | '.join(fmt(x) for x in vals)+' |')
    lines+=['','Confidence categories use the raw-query maximum-cardinality/minimum-distance association at 2 mm; duplicates are surplus candidates within tolerance. ECE uses the existing conservative spatial-label mask and is annotation-relative. It is not calibrated on validation. Training evaluation is the fixed 128-patch, augmentation-free monitor.','']
    if 'baseline_minus_historical' in r:
        lines+=['### Paired subject uncertainty','', 'Differences below are fractions (multiply by 100 for percentage points). Intervals resample the 13 subjects, preserving patch pairing, for 2,000 bootstrap draws. They do not include training-seed uncertainty.','', '| Comparison, 2 mm | Node F1 delta [95% CI] | Edge F1 delta [95% CI] | Raw recall delta [95% CI] |','|---|---:|---:|---:|']
        comparisons={'baseline-minus-historical':r['baseline_minus_historical'],**r.get('paired',{})}
        for name,p in comparisons.items():
            vals=[p['2.0']['metrics'][k] for k in ['node_f1','edge_f1','raw_recall']]
            lines.append('| '+name+' | '+' | '.join(f"{x['delta']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for x in vals)+' |')
        lines+=['']
    if r['pilots_completed']:
        lines+=['### Initial prediction correspondence','', 'Comparison with B0 on the same 32 training and 64 validation patches, before any pilot updates. Query-pair ordering and ground-truth nodes/edges match exactly. A and A-control are exactly prediction-identical. C uses its initial warm-in offset.','', '| Arm | Maximum query displacement, mm | Maximum object-probability change | Maximum relation-probability change | Total retention flips |','|---|---:|---:|---:|---:|']
        for arm,x in r['step_zero_parity'].items():
            v=x['maxima'];lines.append(f"| {arm} | {v['coordinate_mm_max']:.6g} | {v['object_probability_max']:.6g} | {v['relation_probability_max']:.6g} | {x['total_retention_flips']} |")
        lines+=['','### Paired localization results at 1 mm','', '| Comparison | Node F1 delta [95% CI] | Edge F1 delta [95% CI] | Raw recall delta [95% CI] |','|---|---:|---:|---:|']
        for name in ['A-minus-B0','A-minus-A-control','C-minus-B0']:
            vals=[r['paired'][name]['1.0']['metrics'][k] for k in ['node_f1','edge_f1','raw_recall']]
            lines.append('| '+name+' | '+' | '.join(f"{x['delta']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for x in vals)+' |')
        lines+=['']
        lines+=['### Actual image-information contribution','', 'Deltas are intervention minus normal inference on the same 128 validation patches. Negative F1 means removing/shifting the information hurts. These interventions are diagnostic distribution shifts, so interpret them together with the separately trained A-control.','', '| Model / intervention | Node F1 delta [95% CI], 2 mm | Edge F1 delta [95% CI], 2 mm | Mean query shift, mm | Object probability change | Relation probability change |','|---|---:|---:|---:|---:|---:|']
        for arm,modes in r['interventions'].items():
            for mode,v in modes.items():
                intervals=[v['paired']['2.0']['metrics'][k] for k in ['node_f1','edge_f1']];s=v['prediction_change']
                lines.append(f'| {arm} / {mode} | '+' | '.join(f"{x['delta']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for x in intervals)+' | '+' | '.join(f'{s[k]:.4g}' for k in ['displacement_mm','probability_change','relation_change'])+' |')
        lines+=['','| C attention layer | Fine mass | Fine inside-center mass |','|---|---:|---:|']
        for name,v in r['summary']['C']['attention'].items():lines.append(f"| {name} | {v['mass'][1]:.5f} | {v['inside_mass'][1]:.5f} |")
        lines+=['','All four SHA256 hashes of images and node-coordinate targets over every training batch match. Edge and spline targets were not independently hashed; their dataset, loader, augmentation and random seeds are shared. Resolved configurations match after removing the declared architectural settings. Final pilot checkpoints are predeclared update 3000, not picked by validation.','']
    lines+=['### Baseline learning trajectory','', '| Epoch | Update | Mean training objective | Validation node F1, 2 mm | Validation edge F1, 2 mm | Raw recall, 2 mm |','|---|---:|---:|---:|---:|---:|']
    for h in r['baseline_history']:
        if 'evaluation' in h:
            s=h['evaluation']['metrics']['val']['2.0'];lines.append(f"| {h['epoch']} | {h['step']} | {h['mean_train_loss']:.4f} | {s['node_f1']:.4f} | {s['edge_f1']:.4f} | {s['raw_recall']:.4f} |")
    if any('evaluation' in h for h in r['baseline_history']):lines+=['','![Baseline learning trajectory](evidence/fine_feature_architecture/baseline.png)','']
    if r['pilots_completed']:lines+=['','![Matched pilots and feature interventions](evidence/fine_feature_architecture/pilots.png)','']
    lines+=['','Exact numeric evidence and all 1/2-mm bootstrap differences: `experiments/fine_feature_architecture/analysis.json`. Bootstrap seed231123; training seed364505.','']
    text=DOC.read_text();start='<!-- RESULTS -->';end='<!-- END RESULTS -->'
    if start not in text:text+='\n'+start+'\n'+end+'\n'
    before,rest=text.split(start,1);_,after=rest.split(end,1);DOC.write_text(before+start+'\n'+'\n'.join(lines)+'\n'+end+after)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');collect(p.parse_args().partial)
