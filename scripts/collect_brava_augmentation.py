"""CPU paired analysis using the unchanged completed-diagnosis metric functions."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
DIAG=REPO.parent/'gnbm-brava-learning-diagnosis-20261026'
sys.path.insert(0,str(DIAG))
import numpy as np
from sklearn.metrics import average_precision_score
from scripts.brava_learning_audit import read,write
from scripts.collect_brava_learning_diagnosis import load,metrics,summarize,paired_difference
from scripts.node_verifier_metrics import association,discrimination,components

OUT=REPO/'experiments/brava_augmentation_20261026'


def topology(c):
    ids=np.flatnonzero(c['scores']>.5);remap={int(q):i for i,q in enumerate(ids)}
    pairs=c['pairs'];keep=np.isin(pairs,ids).all(1)&(c['relations']>.5)
    edges=np.asarray([[remap[int(a)],remap[int(b)]] for a,b in pairs[keep]],int).reshape(-1,2)
    a,b=components(len(ids),edges);ga,gb=components(len(c['truth']),c['gt_edges'])
    return dict(pred_components=a,gt_components=ga,pred_cycles=b,gt_cycles=gb,
                signed_nodes=len(ids)-len(c['truth']),signed_edges=len(edges)-len(c['gt_edges']),
                signed_components=a-ga,signed_cycles=b-gb)


def confidence(caches):
    out=discrimination(caches,[c['scores'] for c in caches]);groups={k:[] for k in ['all','correct','duplicate','false']}
    for c in caches:
        m=association(c['physical'],c['truth'],2.)
        d=np.linalg.norm(c['physical'][:,None]-c['truth'][None],axis=-1)
        kind=np.full(len(c['scores']),2);kind[d.min(1)<=2]=1;kind[list(m)]=0
        groups['all'].extend(c['scores'])
        for k,name in enumerate(['correct','duplicate','false']):groups[name].extend(c['scores'][kind==k])
    out['quantiles']={k:dict(n=len(v),q05_q25_q50_q75_q95=np.quantile(v,[.05,.25,.5,.75,.95]).tolist() if len(v) else None) for k,v in groups.items()}
    return out


def main():
    p=read(OUT/'plan.json')
    assert all(read(OUT/'runs'/arm/'complete.json')['completed'] for arm in p['arms'])
    initials=[read(OUT/'runs'/arm/'initial.json') for arm in p['arms']]
    assert initials[0]==initials[1]
    complete=[read(OUT/'runs'/arm/'complete.json') for arm in p['arms']]
    assert complete[0]['patch_order_sha256']==complete[1]['patch_order_sha256']==read(OUT/'preflight.json')['patch_order_sha256']
    rows=[];summaries=[];allc={};assignments={};conf=[];losses=[]
    for arm in p['arms']:
        for step in [1000,2000,4000]:
            for pop,ids in p['populations'].items():
                caches=[];rp=[];conditional={1.:[],2.:[]};retained={1.:[],2.:[]}
                path=OUT/'predictions'/arm/str(step)/(pop+'.zip')
                names=[]
                for sample,c in load(path):
                    names.append(sample);caches.append(c);topo=topology(c)
                    assignments[arm,step,pop,sample]=dict(zip(c['hungarian_target'].tolist(),c['hungarian_source'].tolist()))
                    for tol in [1.,2.]:
                        r,cond,ret=metrics(c,tol);r.update(topo,arm=arm,step=step,population=pop,sample=sample,
                                 subject=p['subjects'][sample.split('_')[1]],tolerance=tol)
                        rows.append(r);rp.append(r);conditional[tol].append(cond);retained[tol].append(ret)
                assert names==ids,(arm,step,pop)
                allc[arm,step,pop]=caches
                conf.append(dict(arm=arm,step=step,population=pop,**confidence(caches)))
                for tol in [1.,2.]:
                    rr=[r for r in rp if r['tolerance']==tol];s=dict(arm=arm,step=step,population=pop,tolerance=tol,**summarize(rr))
                    for label,data in [('conditional',conditional[tol]),('all_retained',retained[tol])]:
                        y=np.concatenate([v[0] for v in data]);score=np.concatenate([v[1] for v in data])
                        s.update({label+'_ap':float(average_precision_score(y,score)) if y.any() else None,
                                  label+'_candidates':len(y),label+'_positives':int(y.sum())})
                    for k in ['signed_nodes','signed_edges','signed_components','signed_cycles']:
                        s[k+'_per_patch']=s[k]/len(ids)
                    summaries.append(s)
                records=read(path.with_name(pop+'-losses.json'));n=sum(r['batch_size'] for r in records)
                mean={k:sum(r['losses'][k]*r['batch_size'] for r in records)/n for k in records[0]['losses']}
                losses.append(dict(arm=arm,step=step,population=pop,clean_evaluation_mean_losses=mean,records=records))
    differences=[];churn=[];reconstruction=[]
    for step in [1000,2000,4000]:
        for pop in p['populations']:
            for tol in [1.,2.]:
                selected=[[r for r in rows if r['arm']==arm and r['step']==step and r['population']==pop and r['tolerance']==tol] for arm in p['arms']]
                differences.append(dict(step=step,population=pop,tolerance=tol,direction='off minus on',
                  **{f:paired_difference(*selected,f) for f in ['node_f1','edge_f1','raw_recall']}))
            for sample,a,b in zip(p['populations'][pop],allc['on',step,pop],allc['off',step,pop]):
                np.testing.assert_array_equal(a['truth'],b['truth']);np.testing.assert_array_equal(a['gt_edges'],b['gt_edges'])
                da=np.linalg.norm(a['physical'][:,None]-a['truth'][None],axis=-1).min(0)
                db=np.linalg.norm(b['physical'][:,None]-b['truth'][None],axis=-1).min(0)
                reconstruction.append(dict(step=step,population=pop,sample=sample,subject=p['subjects'][sample.split('_')[1]],
                  off_minus_on_nearest_gt_distance_mm=float((db-da).mean()),
                  changed_confidence_decisions=int(((a['scores']>.5)!=(b['scores']>.5)).sum()),
                  on=topology(a),off=topology(b)))
    for arm in p['arms']:
        for pop,ids in p['populations'].items():
            for a,b in [(1000,2000),(2000,4000)]:
                changes=[]
                for sample in ids:
                    x=assignments[arm,a,pop,sample];y=assignments[arm,b,pop,sample]
                    changes.extend(x[g]!=y[g] for g in x)
                churn.append(dict(arm=arm,population=pop,from_step=a,to_step=b,
                                  churn=float(np.mean(changes)),gt_assignments=len(changes)))
    per_subject=[]
    for s in summaries:
        rr=[r for r in rows if all(r[k]==s[k] for k in ['arm','step','population','tolerance'])]
        for subj in sorted({r['subject'] for r in rr}):
            per_subject.append(dict(**{k:s[k] for k in ['arm','step','population','tolerance']},subject=subj,
                               **summarize([r for r in rr if r['subject']==subj])))
    result=dict(completed=True,summaries=summaries,differences=differences,confidence=conf,
                assignment_churn=churn,losses=losses,checkpoint_provenance=[read(f) for f in sorted((OUT/'runs').glob('*/checkpoint-*.json'))],
                protocol=p,initial_state_equal=True,training_runs=complete,
                uncertainty='1000 whole-subject paired bootstrap draws, seed 261026; one training seed; development cohorts only.')
    result['batch_parity']=[dict(arm=arm,population=pop,**read(OUT/'predictions'/arm/'4000'/(pop+'-batch-parity.json')))
                            for arm in p['arms'] for pop in p['populations']]
    for checkpoint in result['checkpoint_provenance']:
        path=Path(checkpoint['path']);checkpoint.update(canonical_path=str(path.resolve()),bytes=path.stat().st_size)
    write(OUT/'analysis.json',result);write(OUT/'per-patch.json',rows);write(OUT/'per-subject.json',per_subject)
    write(OUT/'reconstruction-differences.json',reconstruction)
    plot_curves(result)
    publish(result)
    claim=REPO.parent/'experiment_gpu_coordination/brava-augmentation-20261026.json'
    c=read(claim);c.update(state='completed; results appended',concurrent_h100=0,analysis=str(OUT/'analysis.json'))
    write(claim,c)
    print('Collected and published augmentation comparison',flush=True)


def plot_curves(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    fields=[('raw_recall','Raw node recall, 2 mm'),('node_recall','Filtered node recall, 2 mm'),
            ('node_f1','Node F1, 2 mm'),('edge_f1','Edge F1, 2 mm'),
            ('hungarian_mm','Hungarian distance (mm)')]
    for ax,(field,label) in zip(axes.flat,fields):
        for arm,color in [('on','tab:blue'),('off','tab:orange')]:
            for pop,style in [('representative_train','-'),('validation','--')]:
                rr=sorted([r for r in s['summaries'] if r['arm']==arm and r['population']==pop and r['tolerance']==2.],key=lambda r:r['step'])
                ax.plot([r['step'] for r in rr],[r[field] for r in rr],style,marker='o',color=color,label=arm+' / '+('train' if pop=='representative_train' else 'validation'))
        ax.set_title(label);ax.set_xticks([1000,2000,4000]);ax.set_xlabel('Updates');ax.grid(alpha=.2)
    ax=axes.flat[-1]
    for arm,color in [('on','tab:blue'),('off','tab:orange')]:
        for pop,style in [('representative_train','-'),('validation','--')]:
            rr=sorted([r for r in s['losses'] if r['arm']==arm and r['population']==pop],key=lambda r:r['step'])
            ax.plot([r['step'] for r in rr],[r['clean_evaluation_mean_losses']['total'] for r in rr],style,marker='o',color=color)
    ax.set_title('Clean evaluation total loss');ax.set_xticks([1000,2000,4000]);ax.set_xlabel('Updates');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('BRAVA K8 augmentation on/off: fixed cohorts, one seed; annotation-relative metrics')
    fig.savefig(OUT/'learning-curves.png',dpi=160);plt.close(fig)


def publish(s):
    def get(arm,step,pop,tol=2.):return next(r for r in s['summaries'] if (r['arm'],r['step'],r['population'],r['tolerance'])==(arm,step,pop,tol))
    def fmt(x):return 'undefined' if x is None else f'{x:.4f}'
    train=[get(a,4000,'representative_train') for a in ['on','off']]
    val=[get(a,4000,'validation') for a in ['on','off']]
    dt=train[1]['node_f1']-train[0]['node_f1'];dv=val[1]['node_f1']-val[0]['node_f1']
    de=train[1]['edge_f1']-train[0]['edge_f1'];ve=val[1]['edge_f1']-val[0]['edge_f1']
    if dt>0 and de>0:
        interpretation='Disabling augmentation improves representative training reconstruction on node and edge F1.'
        vd=next(r for r in s['differences'] if r['step']==4000 and r['population']=='validation' and r['tolerance']==2.)
        if dv>0 and ve>0:
            interpretation+=' Validation node F1 improves modestly, but the edge-F1 gain is uncertain across subjects (its paired 95% interval includes zero).'
            if vd['edge_f1']['ci95'][0]>0:
                interpretation=interpretation.replace('but the edge-F1 gain is uncertain across subjects (its paired 95% interval includes zero).','and edge F1 also improves across these development subjects.')
        else:interpretation+=' The training gain does not consistently transfer to both validation metrics.'
        if val[1]['raw_recall']<val[0]['raw_recall'] and val[1]['hungarian_mm']>val[0]['hungarian_mm']:
            interpretation+=' Validation raw 2-mm localization recall decreases and Hungarian distance worsens. The much larger train gain, declining clean training loss and worse final validation loss are consistent with increased memorization/overfitting; this is not a broadly improved or recovered detector.'
        interpretation+=' The matched comparison supports an optimization cost of this augmentation regime under the fixed budget, while one seed and reused development subjects limit the generalization claim.'
    elif dt<=0 and de<=0:
        interpretation='Disabling augmentation does not improve representative training node or edge F1; these measurements do not support augmentation as a sufficient explanation of the weak fit.'
    else:interpretation='The training effect is mixed between node and edge F1; disabling augmentation is not a consistent reconstruction improvement.'
    poor=all(r['edge_f1']<.2 for r in train)
    if poor:
        interpretation+=' Connectivity remains weak in both training arms and on validation. Recommended follow-up, not launched: separately isolate exposure/training budget with a predefined schedule, or assignment/optimization with fixed exposure; change one factor at a time and retain the generalization gap as an explicit endpoint.'
    text=['## D07: controlled original-K8 augmentation on/off — 27 October 2026','',interpretation,'',
      f'Final off−on train node/edge F1: {dt:+.4f}/{de:+.4f}; validation: {dv:+.4f}/{ve:+.4f}. These are annotation-relative graph errors against incomplete SWC traces. Brain masks are not vessel masks.','',
      '**Matched protocol.** Fresh K8-only control and treatment use the identical clean SyntheticMRI source, 1,248 training patches/39 subjects, fixed seed 261039, batch eight, patch order, fully trainable active detector/relation parameters, node focal gamma 2, established edge CE, AdamW and 4,000-update warmup/cosine schedule. Backbone LR 3.5e-5, other parameters 7e-5, 100-step warmup, 30% floor. `graph_only` spline route; no K24/C1/split/refinement/relation variants. Saved BN, clean inputs, batch eight and unchanged 0.5 node/edge thresholds are used for evaluation at exactly 1k/2k/4k. All targets, matching and physical metric functions are unchanged.','',
      'The historical augmentation-on run included auxiliary geometry heads, so whole-run exact parity was not established and it was not reused as the causal control. Both new arms remove those heads. Existing augmentation is three independent quarter-turn draws and Gaussian noise (probability 0.35, maximum standard deviation 0.015). The off loader draws and discards that transform before returning the clean image/target, preserving Python/NumPy/Torch RNG state for stochastic loss sampling. Smoke checks verified exact clean inputs and identical post-loader RNG, finite backward passes, and zero geometry-to-detector gradients. Initial parameter hashes match, and both arms use the identical deterministic patch stream recorded in `training-patch-stream.jsonl.gz` (32,000 draws; 25 or 26 per patch).','',
      f'Source checkpoint: `{s["protocol"]["source"]}`; SHA256 `{s["protocol"]["source_sha256"]}`. Full checkpoint paths/hashes: `experiments/brava_augmentation_20261026/analysis.json` → `checkpoint_provenance`. Source snapshot, target/manifest hashes, plans, Slurm jobs, losses and all prediction caches are isolated under `experiments/brava_augmentation_20261026`.','',
      '| Updates | Population | Aug | Raw R1 / R2 | Filtered R1 / R2 | Node / edge F1 (2mm) | Hungarian mm | Relation recall / AP (2mm) |',
      '|---:|---|---|---|---|---|---:|---|']
    job=read(OUT/'submission.json')
    parity=s['batch_parity'];pr=[r for condition in parity for r in condition['rows']]
    maxdrift={k:max(r['max_errors'][k] for r in pr) for k in ['physical','scores','relations']}
    execution=f'**Execution and evaluation recovery.** Smoke {job["smoke"]} passed. Both training tasks in array {job["training_array"]} reached 4,000 updates and saved net/optimizer, then exited with the final batch-parity assertion; initial collector {job.get("collector_initial")} was cancelled. Inference-only job {job.get("evaluation_recovery")} recovered the final saved checkpoints with **zero optimizer updates**, followed by CPU collector {job["collector"]}. Failed logs and the original checker source are preserved. Complete image-stream hashes were not written before that failure; patch order is reconstructed exactly from the frozen deterministic runner and saved plan, rather than represented as a recovered live hash. Primary evaluation stays at batch eight. The batch-one/eight audit covers the first eight fixed patches per population per arm: maximum coordinate drift {maxdrift["physical"]:.6f} mm, node-score drift {maxdrift["scores"]:.6g}, relation-score drift {maxdrift["relations"]:.6g}; node/relation 0.5-threshold crossings across this audit: {sum(r["node_threshold_crossings"] for r in pr)}/{sum(r["relation_threshold_crossings"] for r in pr)}. This is measured numerical drift, not exact batch-size invariance.'
    text[6:6]=[execution,'']
    for step in [1000,2000,4000]:
        for pop in s['protocol']['populations']:
            for arm in ['on','off']:
                a=get(arm,step,pop,1.);b=get(arm,step,pop)
                text.append(f'|{step}|{pop}|{arm}|{a["raw_recall"]:.4f} / {b["raw_recall"]:.4f}|{a["node_recall"]:.4f} / {b["node_recall"]:.4f}|{b["node_f1"]:.4f} / {b["edge_f1"]:.4f}|{b["hungarian_mm"]:.3f}|{b["conditional_recall"]:.4f} / {fmt(b["conditional_ap"])}|')
    text+=['','Confidence quantiles, correct-versus-duplicate ranking, canonical AP, eligible-relation denominators/AP, components/cycles, signed node/edge/component/cycle counts, subject bootstrap differences and assignment churn are retained in `analysis.json`. `per-patch.json`, `per-subject.json` and `reconstruction-differences.json` preserve paired denominators and nearest-GT displacement differences. Conditional relation recall/AP depend on localized endpoints and may have small positive denominators.','',
      '| Population (4k, off−on) | Node F1 difference [95% subject CI] | Edge F1 difference [95% subject CI] |','|---|---|---|']
    for pop in s['protocol']['populations']:
        d=next(r for r in s['differences'] if r['step']==4000 and r['population']==pop and r['tolerance']==2.)
        n,e=d['node_f1'],d['edge_f1']
        text.append(f'|{pop}|{n["delta"]:+.4f} [{n["ci95"][0]:+.4f}, {n["ci95"][1]:+.4f}]|{e["delta"]:+.4f} [{e["ci95"][0]:+.4f}, {e["ci95"][1]:+.4f}]|')
    text+=['','Intervals use 1,000 paired whole-subject bootstrap draws. They quantify development-subject uncertainty, not training-seed uncertainty.','',
      '| Population / aug (4k) | β0 / β1 MAE | Signed nodes / edges per patch | Signed β0 / β1 per patch |','|---|---|---|---|']
    for pop in s['protocol']['populations']:
        for arm in ['on','off']:
            b=get(arm,4000,pop)
            text.append(f'|{pop} / {arm}|{b["beta0_mae"]:.3f} / {b["beta1_mae"]:.3f}|{b["signed_nodes_per_patch"]:+.3f} / {b["signed_edges_per_patch"]:+.3f}|{b["signed_components_per_patch"]:+.3f} / {b["signed_cycles_per_patch"]:+.3f}|')
    def loss(arm,pop):return next(r['clean_evaluation_mean_losses'] for r in s['losses'] if (r['arm'],r['step'],r['population'])==(arm,4000,pop))
    lt=[loss(a,'representative_train') for a in ['on','off']];lv=[loss(a,'validation') for a in ['on','off']]
    text+=['',f'**Fit versus generalization:** clean representative-training total loss falls {lt[0]["total"]:.3f}→{lt[1]["total"]:.3f} with augmentation off, while validation total loss rises {lv[0]["total"]:.3f}→{lv[1]["total"]:.3f} and validation edge CE rises {lv[0]["edges"]:.3f}→{lv[1]["edges"]:.3f}. Validation raw 2-mm recall changes {val[0]["raw_recall"]:.4f}→{val[1]["raw_recall"]:.4f}, despite improved confidence-filtered recall {val[0]["node_recall"]:.4f}→{val[1]["node_recall"]:.4f}. Validation predicted edges increase {int(val[0]["edges"])}→{int(val[1]["edges"])}, true edges {int(val[0]["edge_tp"])}→{int(val[1]["edge_tp"])}, with cycle MAE {val[0]["beta1_mae"]:.3f}→{val[1]["beta1_mae"]:.3f}. Thus the modest validation operating-point gain does not establish improved anatomical reconstruction or justify a general recommendation to disable augmentation.','',
      '| Population / aug (4k) | Correct-versus-duplicate AUC | Canonical node AP | Assignment churn 2k→4k |','|---|---:|---:|---:|']
    for pop in ['representative_train','validation']:
        for arm in ['on','off']:
            c=next(r for r in s['confidence'] if (r['arm'],r['step'],r['population'])==(arm,4000,pop))
            ch=next(r for r in s['assignment_churn'] if (r['arm'],r['from_step'],r['to_step'],r['population'])==(arm,2000,4000,pop))
            text.append(f'|{pop} / {arm}|{fmt(c["auc_correct_duplicate"])}|{c["canonical_ap"]:.4f}|{ch["churn"]:.4f}|')
    text+=['','Lower assignment churn does not itself prove better localization; the validation off arm has more stable assignments but larger Hungarian distance. Correct/duplicate labels and confidence quantiles use the established 2-mm spatial association.','']
    text+=['','![Matched BRAVA augmentation learning curves](../../experiments/brava_augmentation_20261026/learning-curves.png)','',
      'Losses on optimization minibatches are logged every 10 updates in each run’s `training.json`; the on/off optimization inputs differ by design. Clean representative-training/validation/tiny losses at each checkpoint use the identical evaluation code and batching and are in `analysis.json` → `losses`; these are the matched fit/generalization comparisons.','',
      'The unchanged 16-patch subset is an additional diagnostic evaluated by both cohort-trained models; it is not a new tiny-set training arm. Earlier tiny-set overfit had 256 draws/patch versus about 25.64 here, with a different LR schedule and augmentation. It does not prove that the larger experiment merely needs more epochs. No test subjects were inspected, no threshold/checkpoint was selected from validation, and no follow-up was launched. Previous negative results remain unchanged.','']
    block='\n'.join(text);write(OUT/'report-section.json',dict(section=block))
    (OUT/'report.md').write_text(block)
    target=REPO/'docs/research/BRAVA_experiment_plan.md';marker='BRAVA AUGMENTATION COMPARISON 20261026'
    import fcntl
    with (OUT/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);old=target.read_text();begin='<!-- BEGIN '+marker+' -->'
        end='<!-- END '+marker+' -->';replacement=begin+'\n'+block+'\n'+end
        if begin not in old:target.write_text(old.rstrip()+'\n\n'+replacement+'\n')
        else:
            start=old.index(begin);stop=old.index(end,start)+len(end)
            target.write_text(old[:start]+replacement+old[stop:])
    write(OUT/'publication.json',dict(report=str(target),completed=True,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    update_registry(s,interpretation,dt,dv,de,ve)


def update_registry(s,interpretation,dt,dv,de,ve):
    import fcntl
    registry=REPO/'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md'
    job=read(OUT/'submission.json')
    summary=f'4k off−on train node/edge F1 {dt:+.4f}/{de:+.4f}; validation {dv:+.4f}/{ve:+.4f}. '+interpretation
    brief=f'4k off−on train node/edge F1 {dt:+.4f}/{de:+.4f}; validation {dv:+.4f}/{ve:+.4f}. Large fit gain; validation edge CI includes zero, raw 2-mm recall falls and loss/cycles worsen. Memorization is plausible; no recovered detector or follow-up launched.'
    evidence='[D07 report](../docs/research/BRAVA_experiment_plan.md#d07-controlled-original-k8-augmentation-onoff--27-october-2026); [artifacts](../experiments/brava_augmentation_20261026/analysis.json)'
    row='| **D07** | BRAVA original-K8 fully trainable augmentation off vs matched fresh augmentation on; same clean source/1,248 patches/seed/order/4k updates | **DONE — measured** | '+brief+' | '+evidence+' |'
    with registry.with_suffix('.md.d07.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);old=registry.read_text();lines=old.splitlines();found=0
        for i,line in enumerate(lines):
            if line.startswith('| **D07** |'):lines[i]=row;found+=1
            elif line.startswith('### Q2 — D07:'):lines[i]='### Q2 — D07: BRAVA K8 augmentation-on/off training ablation — DONE, measured 27 October 2026'
            elif line.startswith('- **BRAVA:**'):
                lines[i]='- **BRAVA:** D06 remains a completed negative BN diagnosis. **D07 is now DONE:** '+brief+' See the existing BRAVA report.'
            elif line.startswith('| Is the weak BRAVA detector due to BN or poor larger-cohort optimization?'):
                lines[i]='| Is the weak BRAVA detector due to BN or poor larger-cohort optimization? | ⚠️ | D06 found no BN rescue; D07 measured augmentation on/off: '+brief+' | D06–D07 |'
        assert found==1
        marker='<!-- BEGIN D07 COLLECTED 20261027 -->'
        appendix=[marker,'### D07 collected outcome — 27 October 2026','',summary,'',
              f'GPU smoke `{job["smoke"]}` passed; matched training array `{job["training_array"]}` reached 4k in both arms, then failed the final numerical batch-parity assertion. Initial collector `{job.get("collector_initial")}` was cancelled. Inference-only recovery `{job.get("evaluation_recovery")}` used the saved 4k weights with zero optimizer updates; CPU collection `{job["collector"]}` produced fixed 1k/2k/4k metrics. Failed logs/checker source remain preserved. Source `{s["protocol"]["source"]}`, SHA256 `{s["protocol"]["source_sha256"]}`. Exact six checkpoint hashes and paths are in `experiments/brava_augmentation_20261026/analysis.json`. Fresh K8-only controls were used because historical auxiliary-head whole-run parity was not established. Seed 261039, batch eight, 1,248 train/256 validation patches, 39/8 subjects, fixed target/matcher/loss/schedule; saved-BN batch1/8 numerical drift is reported. No new architecture, test evaluation or threshold selection.','',
              'Decision gate: distinguish improved optimized-on training fit from validation improvement and memorization. Both arms receive approximately 25.64 draws/patch; earlier 16-patch overfit used different exposure/schedules and cannot prove duration alone is the cause. Incomplete SWC errors are annotation-relative; brain masks are not vessel labels. Any exposure/assignment follow-up remains a recommendation only.','',
              evidence,'<!-- END D07 COLLECTED 20261027 -->']
        if marker not in old:lines.extend(['']+appendix+[''])
        else:
            start=lines.index(marker);stop=lines.index('<!-- END D07 COLLECTED 20261027 -->',start)+1
            lines[start:stop]=appendix
        registry.write_text('\n'.join(lines)+'\n')
    write(OUT/'registry-publication.json',dict(path=str(registry),completed=True,sha256=hashlib.sha256(registry.read_bytes()).hexdigest()))


if __name__=='__main__':main()
