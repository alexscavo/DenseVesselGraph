"""Paired subject uncertainty and intervention report for the bounded pilot."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.node_verifier_metrics import aggregate

ROOT=Path('experiments/multiscale_pretrained')
METRICS=['raw_recall','node_recall','node_precision','node_f1','edge_recall','edge_f1',
         'conditional_recall','hungarian_mm','nearest_mm','beta0_mae','beta1_mae']


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def stats(rows):
    r=aggregate([{k:v for k,v in row.items() if k!='conditional_ap'} for row in rows])
    r.update(raw_recall=r['raw_tp']/r['gt_nodes'],hungarian_mm=r['hungarian_mm_sum']/r['hungarian_n'],nearest_mm=r['nearest_mm_sum']/r['gt_nodes'])
    return r


def selected(rows,split='val',tol='2.0'):
    return [r for r in rows if r['split']==split and r['tolerance']==tol]


def paired(a,b,draws=2000):
    """B minus A; resample whole subjects, maintaining the patch pairing."""
    a={r['sample']:r for r in a};b={r['sample']:r for r in b}
    assert a.keys()==b.keys()
    assert all(a[k]['subject']==b[k]['subject'] for k in a)
    subjects=sorted({r['subject'] for r in a.values()})
    # Aggregate additive sufficient statistics once per subject.
    def grouped(rows):
        result=[]
        for s in subjects:
            rr=[r for r in rows.values() if r['subject']==s]
            result.append({k:sum(r[k] for r in rr) for k,v in rr[0].items() if isinstance(v,(int,float)) and k!='conditional_ap'})
        return result
    ga,gb=grouped(a),grouped(b);sa,sb=stats(list(a.values())),stats(list(b.values()))
    samples={m:[] for m in METRICS};rng=np.random.default_rng(231123)
    for indices in rng.integers(len(subjects),size=(draws,len(subjects))):
        aa,bb=stats([ga[i] for i in indices]),stats([gb[i] for i in indices])
        for m in METRICS:samples[m].append(bb[m]-aa[m])
    return dict(subjects=len(subjects),patches=len(a),draws=draws,metrics={m:dict(delta=sb[m]-sa[m],ci95=np.quantile(samples[m],[.025,.975]).tolist()) for m in METRICS})


def collect():
    result=dict(complete={a:read(ROOT/'runs'/a/'complete.json') for a in ['B0','B1']},correctness=read(ROOT/'gpu-correctness.json'),tiny=read(ROOT/'tiny-fits.json'))
    configs={a:read(ROOT/'runs'/a/'config.json') for a in ['B0','B1']}
    assert configs['B0']['model']['decoder']['feature_levels']==1
    assert configs['B1']['model']['decoder']['feature_levels']==2
    configs['B1']['model']['decoder']['feature_levels']=1
    assert configs['B0']==configs['B1'],'Unmatched configuration'
    proofs={a:read(ROOT/'runs'/a/'batch-proof.json') for a in ['B0','B1']}
    assert proofs['B0']==proofs['B1'],'Training inputs do not match'
    result['matched_batches']=proofs['B0'];result['metrics']={};result['learning_curve']={};result['attention']={}
    initial=[]
    for rec in read(ROOT/'runs'/'B0'/'predictions-0000-normal'/'records.json'):
        with np.load(rec['cache']) as a,np.load(ROOT/'runs'/'B1'/'predictions-0000-normal'/(rec['sample']+'.npz')) as b:
            assert np.array_equal(a['pairs'],b['pairs'])
            initial.append(dict(nodes_mm_max=float(np.linalg.norm(a['physical']-b['physical'],axis=-1).max()),object_probability_max=float(np.abs(a['scores']-b['scores']).max()),relation_probability_max=float(np.abs(a['relations']-b['relations']).max()),retained_node_flips=int(((a['scores']>.5)!=(b['scores']>.5)).sum())))
    result['step_zero_population_parity']=dict(patches=len(initial),maxima={k:max(x[k] for x in initial) for k in initial[0]},total_retained_node_flips=sum(x['retained_node_flips'] for x in initial))
    rows={}
    for arm in ['B0','B1']:
        path=ROOT/'runs'/arm/'predictions-2000-normal';rows[arm]=read(path/'rows.json')
        result['metrics'][arm]=read(path/'summary.json')['metrics']
        result['attention'][arm]=read(path/'summary.json')['attention']
        reference=read(ROOT/'runs'/arm/'predictions-0000-normal'/'rows.json');ids={r['sample'] for r in reference}
        result['learning_curve'][arm]={}
        for step in [0,250,1000,2000]:
            rr=read(ROOT/'runs'/arm/f'predictions-{step:04d}-normal'/'rows.json');rr=[r for r in rr if r['sample'] in ids]
            result['learning_curve'][arm][step]={split:{tol:stats(selected(rr,split,tol)) for tol in ['1.0','2.0']} for split in ['train','val']}
    result['paired_B1_minus_B0']={tol:paired(selected(rows['B0'],tol=tol),selected(rows['B1'],tol=tol)) for tol in ['1.0','2.0']}
    result['interventions']={}
    for mode in ['blocked','roll']:
        path=ROOT/'runs'/'B1'/f'predictions-2000-{mode}';rr=read(path/'rows.json');ids={r['sample'] for r in rr};normal=[r for r in rows['B1'] if r['sample'] in ids]
        records=read(path/'records.json');changes=[]
        for record in records:
            with np.load(record['cache']) as b,np.load(ROOT/'runs'/'B1'/'predictions-2000-normal'/(record['sample']+'.npz')) as a:
                changes.append(dict(node_mm=float(np.linalg.norm(a['physical']-b['physical'],axis=-1).mean()),score=float(np.abs(a['scores']-b['scores']).mean()),retained_flips=int(((a['scores']>.5)!=(b['scores']>.5)).sum())))
        result['interventions'][mode]=dict(paired={tol:paired(selected(normal,tol=tol),selected(rr,tol=tol)) for tol in ['1.0','2.0']},metrics=read(path/'summary.json')['metrics'],prediction_changes={k:float(np.mean([r[k] for r in changes])) for k in changes[0]})
    result['training']={a:read(ROOT/'runs'/a/'training.json') for a in ['B0','B1']}
    write(ROOT/'analysis.json',result)
    plot(result)
    render(result)


def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    steps=[0,250,1000,2000]
    for ax,key,title in zip(axes[0],['node_f1','edge_f1'],['Node F1 at 2 mm','Edge F1 at 2 mm']):
        for arm in ['B0','B1']:
            ax.plot(steps,[r['learning_curve'][arm][s]['val']['2.0'][key] for s in steps],marker='o',label=arm)
        ax.set(title=title+' — fixed 64 validation patches',xlabel='Optimizer updates',ylabel='F1');ax.legend();ax.grid(alpha=.2)
    h=r['training']['B1'];steps=[v['step'] for v in h]
    for kind in ['encoder','decoder']:
        axes[1,0].plot(steps,[np.mean([x['mass'][1] for n,x in v['attention'].items() if n.startswith('decoder.'+kind+'.')]) for v in h],marker='o',label=kind)
    axes[1,0].axhline(1e-4,color='gray',linestyle=':',label='Initial fine prior')
    axes[1,0].set(title='B1 fine attention — training minibatches',xlabel='Optimizer updates',ylabel='Mean attention mass',yscale='log');axes[1,0].legend();axes[1,0].grid(alpha=.2)
    axes[1,1].plot(steps,[v['gradients']['fine_input_proj.'] for v in h],marker='o',color='tab:orange')
    axes[1,1].set(title='B1 fine projection — training minibatches',xlabel='Optimizer updates',ylabel='Gradient L2 norm',yscale='log');axes[1,1].grid(alpha=.2)
    out=Path('docs/research/evidence/multiscale_pretrained');out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'pilot.png',dpi=180);fig.savefig(out/'pilot.pdf');plt.close(fig)


def render(r):
    lines=['## Completed bounded pilot','',
        'Both arms used identical sample schedules and first augmented batches (SHA256 checked). All comparisons below use the predeclared update-2,000 checkpoint. Intervals are paired 95% percentile bootstrap intervals from 2,000 resamples of the 13 validation subjects, not independent patch resamples. They describe subject variation for one training seed.','',
        '| Validation metric | B0, 1 mm | B1, 1 mm | B0, 2 mm | B1, 2 mm |',
        '|---|---:|---:|---:|---:|']
    for key,label in [('raw_recall','Raw one-to-one recall'),('node_recall','Filtered recall'),('node_precision','Node precision'),('node_f1','Node F1'),('edge_recall','Edge recall'),('edge_f1','Edge F1'),('conditional_recall','Endpoint-conditional edge recall'),('mean_patch_conditional_ap','Mean patch conditional AP'),('duplicate_rate','Duplicate fraction'),('hungarian_mm','Hungarian distance, mm'),('nearest_mm','Nearest proposal distance, mm'),('beta0_mae','Component-count MAE'),('beta1_mae','Cycle-count MAE')]:
        vals=[r['metrics'][arm]['val'][tol][key] for tol in ['1.0','2.0'] for arm in ['B0','B1']]
        lines.append('| '+label+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
    lines+=['','| B1 − B0, validation | 1 mm delta [95% CI] | 2 mm delta [95% CI] |','|---|---:|---:|']
    for key in METRICS:
        vals=[r['paired_B1_minus_B0'][tol]['metrics'][key] for tol in ['1.0','2.0']]
        lines.append('| '+key+' | '+' | '.join(f"{x['delta']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for x in vals)+' |')
    lines+=['','### Learning and actual fine-feature contribution','',
        'Learning curves below use the same 64 validation patches at every step; these are distinct from the final full-validation table.','',
        '| Arm / update | Raw recall, 2 mm | Filtered recall | Node F1 | Edge F1 | Hungarian mm |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ['B0','B1']:
        for step in [0,250,1000,2000]:
            s=r['learning_curve'][arm][step]['val']['2.0'];lines.append(f'| {arm} / {step} | '+' | '.join(f'{s[k]:.4f}' for k in ['raw_recall','node_recall','node_f1','edge_f1','hungarian_mm'])+' |')
    lines+=['','| Final split, 2 mm | Raw recall | Filtered recall | Node F1 | Edge F1 |','|---|---:|---:|---:|---:|']
    for arm in ['B0','B1']:
        for split in ['train','val']:
            s=r['metrics'][arm][split]['2.0'];lines.append(f'| {arm} / {split} | '+' | '.join(f'{s[k]:.4f}' for k in ['raw_recall','node_recall','node_f1','edge_f1'])+' |')
    lines+=['','Interventions use the identical 128-patch validation subset. Deltas are intervention minus normal B1; negative recall/F1 means the normal fine pathway helps. Rolling changes fine-map spatial correspondence but retains its values. Blocking renormalizes all attention onto the coarse level.','',
        '| Intervention | Node F1 delta [95% CI], 2 mm | Edge F1 delta [95% CI], 2 mm | Mean query displacement, mm | Mean absolute object-probability change |',
        '|---|---:|---:|---:|---:|']
    for mode,x in r['interventions'].items():
        intervals=[x['paired']['2.0']['metrics'][k] for k in ['node_f1','edge_f1']]
        lines.append('| '+mode+' | '+' | '.join(f"{v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for v in intervals)+f" | {x['prediction_changes']['node_mm']:.3g} | {x['prediction_changes']['score']:.3g} |")
    lines+=['','Final fine attention mass and mass with sampling centers inside the feature volume are averaged over the final 128 training + 480 validation snapshots. Inside-center mass is a sampling diagnostic, not exact interpolation support.','',
        '| Attention module | Fine mass | Fine inside-center mass |','|---|---:|---:|']
    for name,x in r['attention']['B1'].items():lines.append(f"| {name} | {x['mass'][1]:.6f} | {x['inside_mass'][1]:.6f} |")
    lines+=['','### Correctness, tiny fits and resources','']
    c=r['correctness'];lines.append(f"Compiled CUDA/reference forward max error: {c['cuda_reference_max_error']:.3g}; gradient max errors: {c['cuda_gradient_max_errors']}. {c['migration']['copied_tensors']} compatible tensors were copied bitwise.")
    for key,x in c['parity'].items():lines.append(f"Step-zero {key}: fine-disabled max error {x['blocked_max_error']:.3g}; active-fine max error {x['active_max_error']:.3g}"+(' in normalized coordinates.' if key=='pred_nodes' else ' in logits.'))
    p=r['step_zero_population_parity'];v=p['maxima']
    lines+=['',f"Across the larger {p['patches']}-patch step-zero population, active B1 versus B0 has maximum query displacement {v['nodes_mm_max']:.4f} mm, object-probability difference {v['object_probability_max']:.5f}, and relation-probability difference {v['relation_probability_max']:.5f}. Two of 11,520 candidate retention decisions change. This is near parity, not bitwise equality with the active new path.",'']
    for arm in ['B0','B1']:
        s=r['tiny'][arm][-1]['metrics'];x=r['complete'][arm]
        lines.append(f"{arm}: eight-patch fit after 512 updates: node recall {s['node_recall']:.4f}, edge F1 {s['edge_f1']:.4f}; pilot job {x['job_id']}, {x['updates']} updates, {x['seconds']/60:.2f} min including evaluation, peak allocated GPU memory {x['peak_gpu_gib']:.2f} GiB.")
    lines+=['','![Matched validation learning curves and fine-path diagnostics](evidence/multiscale_pretrained/pilot.png)','',
        'Full numeric evidence, all bootstrap metrics and per-step gradient norms: `experiments/multiscale_pretrained/analysis.json`.','']
    doc=Path('docs/research/multiscale_pretrained_feasibility.md');text=doc.read_text();start=text.index('<!-- RESULTS -->');end=text.index('<!-- END RESULTS -->')
    doc.write_text(text[:start]+'<!-- RESULTS -->\n'+'\n'.join(lines)+'\n'+text[end:])


if __name__=='__main__':collect()
