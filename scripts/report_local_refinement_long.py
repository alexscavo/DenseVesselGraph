"""CPU analysis of the fixed-budget, paired two-seed local-refinement study."""
import argparse
import io
import json
import zipfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.node_verifier_metrics import measure,aggregate,select
from scripts.report_multiscale_pretrained import selected,stats,METRICS
from metrics.common_vessel_metrics import aggregate_common_vessels

ROOT=Path('experiments/local_refinement_long')
DOC=Path('docs/research/local_refinement_long_study.md')
ARMS=['B0','A','A-xyz','A-control'];SEEDS=[364505,92741]
COMPARISONS=[('B0','A'),('B0','A-xyz'),('A','A-xyz'),('A-control','A'),('A-control','A-xyz')]


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def run(seed,arm):return ROOT/str(seed)/'runs'/arm


def load_cache(path):
    path=Path(path)
    if path.exists():return np.load(path)
    with zipfile.ZipFile(path.parent/'caches.zip') as archive:
        return np.load(io.BytesIO(archive.read(path.name)))

def crossed(a,b,common=False):
    """Paired whole-subject draws shared across seeds; seed draws remain paired."""
    seeds=sorted(a);subjects=sorted({r['subject'] for r in a[seeds[0]]})
    fields=['common_branch_f1','common_curve_f1','common_anchor_f1'] if common else METRICS
    fn=aggregate_common_vessels if common else stats
    grouped=[];observed=[]
    for seed in seeds:
        aa={r['sample']:r for r in a[seed]};bb={r['sample']:r for r in b[seed]};assert aa.keys()==bb.keys()
        assert all(aa[k]['subject']==bb[k]['subject'] for k in aa)
        assert sorted({r['subject'] for r in aa.values()})==subjects
        sides=[]
        for rr in [aa,bb]:
            groups=[]
            for subject in subjects:
                rows=[r for r in rr.values() if r['subject']==subject]
                groups.append({k:sum(r[k] for r in rows) for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in ['conditional_ap','common_guide_distance_mm','common_tortuosity_error']})
            sides.append(groups)
        grouped.append(sides);sa,sb=fn(sides[0]),fn(sides[1]);observed.append([sb[k]-sa[k] for k in fields])
    rng=np.random.default_rng(251026);draws=np.empty((2000,len(seeds),len(fields)))
    for d,idx in enumerate(rng.integers(len(subjects),size=(2000,len(subjects)))):
        for j,(aa,bb) in enumerate(grouped):
            sa,sb=fn([aa[i] for i in idx]),fn([bb[i] for i in idx]);draws[d,j]=[sb[k]-sa[k] for k in fields]
    seed_indices=rng.integers(len(seeds),size=(2000,len(seeds)))
    joint=np.stack([draws[i,idx].mean(0) for i,idx in enumerate(seed_indices)])
    observed=np.asarray(observed)
    return dict(subjects=len(subjects),draws=2000,metrics={k:dict(mean_delta=float(observed[:,i].mean()),seed_deltas={str(s):float(observed[j,i]) for j,s in enumerate(seeds)},seed_ci95={str(s):np.quantile(draws[:,j,i],[.025,.975]).tolist() for j,s in enumerate(seeds)},subject_ci95=np.quantile(draws[:,:,i].mean(1),[.025,.975]).tolist(),crossed_ci95=np.quantile(joint[:,i],[.025,.975]).tolist()) for i,k in enumerate(fields)})


def caches(directory):
    out=[]
    for rec in read(directory/'records.json'):
        with load_cache(rec['cache']) as data:c={k:data[k] for k in data.files}
        out.append((rec,c))
    return out


def operating(seed):
    path=ROOT/str(seed)/'operating.json'
    if path.exists():return read(path)
    base=caches(run(seed,'B0')/'step-12000');counts={r['sample']:int((c['scores']>.5).sum()) for r,c in base}
    thresholds=np.unique(np.r_[0.,np.linspace(.05,.95,37),.5,1.]).tolist();result={};count_rows={}
    for arm in ARMS:
        cached=base if arm=='B0' else caches(run(seed,arm)/'step-12000');curves={};matched={};calibration={}
        for tol in [1.,2.]:
            key=str(tol);curves[key]={};memo={}
            # Identical top-K sets recur over thresholds; memoize per sample/count.
            def evaluate_threshold(threshold,split):
                rows=[]
                for rec,c in cached:
                    if rec['split']!=split:continue
                    ids=np.sort(select(c,c['scores'],threshold=threshold));mkey=(rec['sample'],len(ids))
                    if mkey not in memo:memo[mkey]=measure(c,ids,tol)
                    rows.append(memo[mkey])
                return aggregate(rows)
            for split in ['train','val']:
                curves[key][split]=[dict(threshold=t,**evaluate_threshold(t,split)) for t in thresholds]
            rr=[]
            original={r['sample']:r for r in selected(read(run(seed,arm)/'step-12000'/'rows.json'),tol=key)}
            for rec,c in cached:
                if rec['split']!='val':continue
                ids=np.sort(select(c,c['scores'],count=counts[rec['sample']]));r=measure(c,ids,tol)
                for k in ['raw_tp','hungarian_mm_sum','hungarian_n','nearest_mm_sum']:r[k]=original[rec['sample']][k]
                rr.append(dict(r,sample=rec['sample'],subject=rec['subject'],split='val',tolerance=key))
            matched[key]=stats(rr);count_rows[(arm,key)]=rr
            calibration[key]={}
            for target in [.5,.7,.8,.9]:
                candidates=[x for x in curves[key]['train'] if x['nodes'] and x['node_precision']>=target]
                chosen=max(candidates,key=lambda x:(x['node_recall'],-x['threshold'])) if candidates else None
                calibration[key][str(target)]=None if chosen is None else dict(threshold=chosen['threshold'],train=chosen,val=next(x for x in curves[key]['val'] if x['threshold']==chosen['threshold']))
        result[arm]=dict(curves=curves,matched_count=matched,train_calibrated=calibration)
        print('operating',seed,arm,flush=True)
    for arm in ARMS:
        result[arm]['matched_precision']={}
        for tol in ['1.0','2.0']:
            target=next(x['node_precision'] for x in result['B0']['curves'][tol]['val'] if x['threshold']==.5)
            curve=result[arm]['curves'][tol]['val'];points=[]
            for lo,hi in zip(curve,curve[1:]):
                pl,ph=lo['node_precision'],hi['node_precision']
                if not lo['nodes'] or not hi['nodes'] or pl==ph or not min(pl,ph)<=target<=max(pl,ph):continue
                # Interpolate additive counts, not F1 ratios. Solve the mixture
                # weight so expected TP / expected retained count equals target.
                w=(target*lo['nodes']-lo['node_tp'])/((hi['node_tp']-lo['node_tp'])-target*(hi['nodes']-lo['nodes']))
                counts={k:float(lo[k]*(1-w)+hi[k]*w) for k in ['nodes','gt_nodes','node_tp','duplicates','patches','edges','gt_edges','edge_tp','eligible','beta0_error','beta1_error']}
                point=aggregate([counts]);assert abs(point['node_precision']-target)<1e-8
                point.update(threshold_bracket=[lo['threshold'],hi['threshold']],upper_weight=w);points.append(point)
            for x in curve:
                if x['nodes'] and abs(x['node_precision']-target)<1e-12:points.append({k:x[k] for k in ['node_precision','node_recall','node_f1','edge_precision','edge_recall','edge_f1','beta0_mae','beta1_mae']})
            result[arm]['matched_precision'][tol]=dict(target_precision=target,interpolated=max(points,key=lambda x:x['node_recall']) if points else None)
    write(path,result)
    write(ROOT/str(seed)/'matched-count-rows.json',{arm:{tol:count_rows[(arm,tol)] for tol in ['1.0','2.0']} for arm in ARMS})
    return result


def consistency(seed,arm):
    root=run(seed,arm);out={}
    for folder in sorted(root.glob('reconstruction-*')):
        records=read(folder/'validation-000000'/'predictions.json');maximum=0.;n_edges=0
        for r in records:
            nodes=np.asarray(r['nodes_dhw']);edges=np.asarray(r['edges'],int).reshape(-1,2)
            assert len(set(r['query_ids']))==len(nodes)
            if not len(edges):continue
            assert edges.min()>=0 and edges.max()<len(nodes)
            endpoints=nodes[edges]
            for field in ['spline_controls','spline_curves']:
                geometry=np.asarray(r[field]);assert len(geometry)==len(edges)
                maximum=max(maximum,float(np.max(np.abs(geometry[:,[0,-1]]-endpoints))))
            n_edges+=len(edges)
        assert maximum<1e-6,(seed,arm,folder,maximum)
        out[folder.name]=dict(patches=len(records),edges=n_edges,max_normalized_endpoint_error=maximum)
    return out


def collect():
    plan=read(ROOT/'plan.json');result=dict(plan=plan,jobs=read(ROOT/'jobs.json'),smoke=read(ROOT/'smoke.json'),runs={},paired={},interventions={},operating={},consistency={})
    rows={};commons={};matched={};hashes=[]
    for seed in SEEDS:
        key=str(seed);result['runs'][key]={};rows[seed]={};commons[seed]={};seed_hashes=[];configs=[]
        for arm in ARMS:
            root=run(seed,arm);done=read(root/'complete.json');assert done['completed'] and done['steps']==12000
            seed_hashes.append(done['batch_sha256']);cfg=read(root/'config.json');cfg['experiment']['name']='matched';cfg['model']['node_refinement']={'enabled':False};configs.append(cfg)
            trajectory={str(s):read(root/f'step-{s}'/'summary.json') for s in plan['checkpoints']}
            result['runs'][key][arm]=dict(complete=done,trajectory=trajectory,final=trajectory['12000'],selected=trajectory[str(done['selected_step'])],reconstruction=read(root/'reconstruction-final'/'summary.json'),selected_reconstruction=read(root/('reconstruction-final' if done['selected_step']==12000 else 'reconstruction-selected')/'summary.json'))
            rows[seed][arm]={phase:read(root/f'step-{step}'/'rows.json') for phase,step in [('final',12000),('selected',done['selected_step'])]}
            commons[seed][arm]=[dict(r,sample=r['sample_id'],subject=plan['dataset']['subjects'][r['sample_id'].split('_')[1]]) for r in read(root/'reconstruction-final'/'validation-000000'/'common-v2-per-patch.json')]
            result['consistency'][f'{seed}/{arm}']=consistency(seed,arm)
        assert len(set(seed_hashes))==1 and all(c==configs[0] for c in configs)
        hashes.append(seed_hashes[0]);result['operating'][key]=operating(seed);matched[seed]=read(ROOT/key/'matched-count-rows.json')
    assert hashes[0]!=hashes[1],'Seeds must actually change the training inputs'
    for a,b in COMPARISONS:
        name=f'{b}-minus-{a}';result['paired'][name]={}
        for phase in ['final','selected']:
            result['paired'][name][phase]={tol:crossed({s:selected(rows[s][a][phase],tol=tol) for s in SEEDS},{s:selected(rows[s][b][phase],tol=tol) for s in SEEDS}) for tol in ['1.0','2.0']}
        result['paired'][name]['matched_count']={tol:crossed({s:matched[s][a][tol] for s in SEEDS},{s:matched[s][b][tol] for s in SEEDS}) for tol in ['1.0','2.0']}
        result['paired'][name]['common']=crossed({s:commons[s][a] for s in SEEDS},{s:commons[s][b] for s in SEEDS},common=True)
    for arm in ['A','A-xyz']:
        result['interventions'][arm]={}
        for mode in ['zero','roll']+(['no-confidence','no-coordinate'] if arm=='A' else []):
            normal={};other={};changes={};reconstruction={}
            for seed in SEEDS:
                root=run(seed,arm);other[seed]=read(root/('final-'+mode)/'rows.json');ids={r['sample'] for r in other[seed]};normal[seed]=[r for r in rows[seed][arm]['final'] if r['sample'] in ids]
                shifts=[]
                for rec,c in caches(root/('final-'+mode)):
                    with load_cache(root/'step-12000'/(rec['sample']+'.npz')) as orig:
                        assert np.array_equal(c['pairs'],orig['pairs']) and np.array_equal(c['truth'],orig['truth'])
                        rel=float(np.abs(c['relations']-orig['relations']).max());assert rel==0,(arm,mode,rel)
                        mask=orig['scores']>.5;dist=np.linalg.norm(c['physical']-orig['physical'],axis=-1)
                        prob=float(np.abs(c['scores']-orig['scores']).mean())
                        if arm=='A-xyz':assert prob==0
                        if mode=='no-confidence':assert dist.max()==0
                        shifts.append(dict(retained_displacement_mm=float(dist[mask].mean()),probability_change=prob,relation_max_change=rel,retention_flips=int(((c['scores']>.5)!=mask).sum())))
                changes[str(seed)]={k:float(np.mean([x[k] for x in shifts])) for k in shifts[0]}
                reconstruction[str(seed)]=read(root/('reconstruction-'+mode)/'summary.json')
            result['interventions'][arm][mode]=dict(paired={tol:crossed({s:selected(normal[s],tol=tol) for s in SEEDS},{s:selected(other[s],tol=tol) for s in SEEDS}) for tol in ['1.0','2.0']},changes=changes,reconstruction=reconstruction)
    write(ROOT/'analysis.json',result);render(result);plot(result)


def interval(x):return f"{x['mean_delta']:+.4f} [{x['crossed_ci95'][0]:+.4f}, {x['crossed_ci95'][1]:+.4f}]"


def render(r):
    lines=['## Completed measurements','', 'Primary comparison: fixed update12000. Tables retain both fine-tuning seeds; they share one source baseline. Bootstrap intervals resample paired subjects and seeds (2000 draws, seed251026); two seeds cannot characterize a population of training initializations reliably. Per-seed paired-subject intervals and fixed-seed subject intervals are also retained in analysis.json.','']
    for tol in ['1.0','2.0']:
        lines += [f'### Physical validation metrics at {tol} mm','', '| Seed | Arm | Raw recall | Recall | Precision | Node F1 | Edge F1 | Hungarian mm |','|---|---|---:|---:|---:|---:|---:|---:|']
        for seed in SEEDS:
            for arm in ARMS:
                s=r['runs'][str(seed)][arm]['final']['metrics']['val'][tol]
                lines.append(f'| {seed} | {arm} | '+' | '.join(f'{s[k]:.4f}' for k in ['raw_recall','node_recall','node_precision','node_f1','edge_f1','hungarian_mm'])+' |')
        lines+=['']
    lines+=['### Paired final differences','', '| Comparison | Tolerance | Node F1 delta [95% interval] | Edge F1 delta [95% interval] | Per-seed edge deltas |','|---|---|---:|---:|---|']
    for name,phases in r['paired'].items():
        for tol in ['1.0','2.0']:
            x=phases['final'][tol]['metrics'];lines.append(f'| {name} | {tol} mm | {interval(x["node_f1"])} | {interval(x["edge_f1"])} | '+', '.join(f'{s}: {v:+.4f}' for s,v in x['edge_f1']['seed_deltas'].items())+' |')
    lines+=['','### Graph reconstruction and training/validation behavior','', '| Seed | Arm | Train/val node F1, 2 mm | Train/val edge F1, 2 mm | Branch F1 | Curve F1 | Chord Curve F1 | Components MAE | Cycles MAE | Duplicate fraction |','|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for arm in ARMS:
            x=r['runs'][str(seed)][arm];v=x['final']['metrics']['val']['2.0'];t=x['final']['metrics']['train']['2.0'];g=x['reconstruction']
            lines.append(f'| {seed} | {arm} | {t["node_f1"]:.4f}/{v["node_f1"]:.4f} | {t["edge_f1"]:.4f}/{v["edge_f1"]:.4f} | '+' | '.join(f'{z:.4f}' for z in [g['common_branch_f1'],g['common_curve_f1'],g['same_endpoint_chord_common_curve_f1'],v['beta0_mae'],v['beta1_mae'],v['duplicate_rate']])+' |')
    lines+=['','| Comparison | Branch F1 delta [95% interval] | Curve F1 delta [95% interval] |','|---|---:|---:|']
    for name,x in r['paired'].items():
        m=x['common']['metrics'];lines.append(f'| {name} | {interval(m["common_branch_f1"])} | {interval(m["common_curve_f1"])} |')
    lines+=['','### Confidence and conditional relations','', '| Seed | Arm | Correct/false AUROC | Correct/duplicate AUROC | ECE | Edge precision, 2 mm | Edge recall, 2 mm | Conditional edge recall |','|---|---|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for arm in ARMS:
            x=r['runs'][str(seed)][arm]['final'];d=x['discrimination']['val'];m=x['metrics']['val']['2.0'];lines.append(f'| {seed} | {arm} | '+' | '.join(f'{v:.4f}' for v in [d['auc_correct_false'],d['auc_correct_duplicate'],d['ece'],m['edge_precision'],m['edge_recall'],m['conditional_recall']])+' |')
    lines+=['','### Matched-count comparisons','', 'Each patch retains the same number as its paired B0 prediction at threshold0.5. No GT count is used. Query rankings differ; the relation head and its scores are unchanged.','', '| Comparison | Tolerance | Node F1 delta [95% interval] | Edge F1 delta [95% interval] |','|---|---|---:|---:|']
    for name,x in r['paired'].items():
        for tol in ['1.0','2.0']:
            m=x['matched_count'][tol]['metrics'];lines.append(f'| {name} | {tol} | {interval(m["node_f1"])} | {interval(m["edge_f1"])} |')
    lines+=['','### Matched-precision validation curves','', 'Descriptive interpolation of additive counts between adjacent thresholds at each seed\'s B0 precision (0.5 threshold), choosing the highest interpolated recall if multiple brackets exist. The mixture weight solves expected TP / expected retained count = target precision, then F1 is recomputed. These use validation labels and are not deployable calibrated inference results; interpolated graph metrics summarize a threshold mixture, not a single reconstructed graph.','', '| Seed | Tolerance | Arm | Target precision | Recall | Edge F1 |','|---|---|---|---:|---:|---:|']
    for seed in SEEDS:
        for tol in ['1.0','2.0']:
            for arm in ARMS:
                x=r['operating'][str(seed)][arm]['matched_precision'][tol];v=x['interpolated']
                lines.append(f'| {seed} | {tol} | {arm} | {x["target_precision"]:.4f} | '+('unattainable | n/a |' if v is None else f'{v["node_recall"]:.4f} | {v["edge_f1"]:.4f} |'))
    lines+=['','### Train-calibrated operating points','', 'Thresholds are chosen on128 training-monitor patches at target precision0.5 (1 mm) or0.8 (2 mm) and then applied unchanged to validation. Achieved validation precision need not match the target. All four targets0.5/0.7/0.8/0.9, their train scores and thresholds are in operating.json.','', '| Seed | Tolerance | Arm | Threshold | Val precision | Val recall | Val edge F1 |','|---|---|---|---:|---:|---:|---:|']
    for seed in SEEDS:
        for tol,target in [('1.0','0.5'),('2.0','0.8')]:
            for arm in ARMS:
                x=r['operating'][str(seed)][arm]['train_calibrated'][tol][target]
                lines.append(f'| {seed} | {tol} | {arm} | '+('unattainable | n/a | n/a | n/a |' if x is None else f'{x["threshold"]:.3f} | {x["val"]["node_precision"]:.4f} | {x["val"]["node_recall"]:.4f} | {x["val"]["edge_f1"]:.4f} |'))
    lines+=['','### Image and correction interventions','', 'Intervention minus normal on the same128 validation patches. All raw relation probabilities remain exactly unchanged. Image intervention scores for coordinate-only A are exactly unchanged too.','', '| Arm | Intervention | Node F1 delta, 2 mm | Edge F1 delta, 2 mm | Mean retained-node displacement, mm |','|---|---|---:|---:|---:|']
    for arm,modes in r['interventions'].items():
        for mode,x in modes.items():
            m=x['paired']['2.0']['metrics'];d=np.mean([v['retained_displacement_mm'] for v in x['changes'].values()]);lines.append(f'| {arm} | {mode} | {interval(m["node_f1"])} | {interval(m["edge_f1"])} | {d:.4f} |')
    lines+=['','### Predefined secondary checkpoint selection','', '| Seed | Arm | Selected update | Node F1, 2 mm | Edge F1, 2 mm |','|---|---|---:|---:|---:|']
    for seed in SEEDS:
        for arm in ARMS:
            x=r['runs'][str(seed)][arm];m=x['selected']['metrics']['val']['2.0'];lines.append(f'| {seed} | {arm} | {x["complete"]["selected_step"]} | {m["node_f1"]:.4f} | {m["edge_f1"]:.4f} |')
    lines+=['','Secondary paired comparisons are in analysis.json; primary conclusions use equal final budgets. Every saved reconstruction passed edge-index and spline/curve endpoint checks.','', '![Matched training trajectories](evidence/local_refinement_long/trajectories.png)','', 'Exact numeric evidence: experiments/local_refinement_long/analysis.json; per-seed operating.json includes threshold curves and calibration.','']
    text=DOC.read_text();before,rest=text.split('<!-- RESULTS -->',1);_,after=rest.split('<!-- END RESULTS -->',1);DOC.write_text(before+'<!-- RESULTS -->\n'+'\n'.join(lines)+'\n<!-- END RESULTS -->'+after)


def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');steps=r['plan']['checkpoints']
    colors=dict(zip(ARMS,['tab:blue','tab:orange','tab:green','gray']))
    for row,seed in enumerate(SEEDS):
        for col,key in enumerate(['node_f1','edge_f1']):
            ax=axes[row,col]
            for arm in ARMS:
                trajectory=r['runs'][str(seed)][arm]['trajectory'];ax.plot(steps,[trajectory[str(s)]['metrics']['val']['2.0'][key] for s in steps],marker='o',color=colors[arm],label=arm)
            ax.set(title=f'Seed {seed}: {key}, 2 mm',xlabel='Optimizer updates',ylabel='F1');ax.grid(alpha=.2);ax.legend()
    out=Path('docs/research/evidence/local_refinement_long');out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'trajectories.png',dpi=180);fig.savefig(out/'trajectories.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--operating-seed',type=int);a=p.parse_args()
    if a.operating_seed is not None:operating(a.operating_seed)
    else:collect()
