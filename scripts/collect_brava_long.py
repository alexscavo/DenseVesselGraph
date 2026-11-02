"""Collect paired milestone graphs, uncertainty, curves, costs and existing reports."""
import argparse
import fcntl
import io
import json
import multiprocessing as mp
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.brava_long_experiment import OUT,MAIN,ARMS,MILESTONES,read,write,sha
from scripts.brava_long_evaluation import aggregate


def common_task(item):
    from metrics.common_vessel_metrics import evaluate_common_vessels
    record,prediction=item
    return dict(sample=record['sample'],split=record['split'],subject=record['subject'],
                val_monitor=record['val_monitor'],**evaluate_common_vessels(prediction,record['gt']))


def common_inputs(records):
    for path in sorted({r['archive'] for r in records}):
        with zipfile.ZipFile(path) as archive:
            for r in records:
                if r['archive']!=path:continue
                with np.load(io.BytesIO(archive.read(r['entry']))) as data:
                    yield r,{k:data[k].copy() for k in ['nodes','edges','spline_controls']}


def collect_epoch(epoch):
    from metrics.common_vessel_metrics import aggregate_common_vessels
    result=dict(epoch=epoch,arms={},paired_subject_intervals={})
    for arm in ARMS:
        prediction=OUT/'large'/'predictions'/arm/f'epoch-{epoch:03d}'
        p=read(prediction/'summary.json')
        assert p['completed'] and p['full_validation']
        marker=OUT/'analysis'/f'{arm}-{epoch:03d}-common.json'
        if marker.exists():commons=read(marker)
        else:
            commons=[]
            with mp.get_context('fork').Pool(4) as pool:
                for i,r in enumerate(pool.imap(common_task,common_inputs(p['records']),chunksize=8),1):
                    commons.append(r)
                    if i%256==0:print('common metrics',arm,epoch,i,'/',len(p['records']),flush=True)
            write(marker,commons)
        summary=dict(metrics=p['metrics'],losses=p['losses'],common={})
        for group in ['train','val','val_monitor']:
            rr=[r for r in commons if r['split']==group or (group=='val_monitor' and r['val_monitor'])]
            summary['common'][group]=aggregate_common_vessels([{k:v for k,v in r.items() if k not in ['sample','split','subject','val_monitor']} for r in rr])
        summary['per_subject']={}
        for subject in sorted({r['subject'] for r in p['rows']}):
            rr=[r for r in p['rows'] if r['subject']==subject and r['tolerance_mm']==2.]
            cc=[r for r in commons if r['subject']==subject]
            summary['per_subject'][subject]=dict(aggregate(rr),**aggregate_common_vessels(
                [{k:v for k,v in r.items() if k not in ['sample','split','subject','val_monitor']} for r in cc]))
        result['arms'][arm]=summary
    a=read(OUT/'large'/'predictions'/'scratch'/f'epoch-{epoch:03d}'/'summary.json')
    b=read(OUT/'large'/'predictions'/'synthetic'/f'epoch-{epoch:03d}'/'summary.json')
    assert [(r['split'],r['sample'],r['gt']) for r in a['records']]==[(r['split'],r['sample'],r['gt']) for r in b['records']]
    # All complete epochs must consume exactly the same images/graph targets.
    ha=read(OUT/'large'/'runs'/'scratch'/'history.json')
    hb=read(OUT/'large'/'runs'/'synthetic'/'history.json')
    da={r['epoch']:r for r in ha};db={r['epoch']:r for r in hb}
    for e in range(1,epoch+1):
        assert da[e]['full_augmented_batch_chain_sha256']==db[e]['full_augmented_batch_chain_sha256'],('augmentation stream mismatch',e)
        assert da[e]['patch_order_sha256']==db[e]['patch_order_sha256']
        assert da[e]['patches']==db[e]['patches']==8768
    rng=np.random.default_rng(271027)
    subjects=read(OUT/'plan.json')['subject_groups']['val']
    for tolerance in [1.,2.]:
        subject_rows={arm:{s:[r for r in p['rows'] if r['split']=='val' and r['subject']==s and r['tolerance_mm']==tolerance]
                           for s in subjects} for arm,p in [('scratch',a),('synthetic',b)]}
        draws={k:[] for k in ['node_f1','f1','raw_node_recall','conditional_recall']}
        for _ in range(1000):
            chosen=rng.choice(subjects,len(subjects),replace=True)
            values={arm:aggregate([r for s in chosen for r in subject_rows[arm][s]]) for arm in ARMS}
            for k in draws:
                if all(values[arm][k] is not None for arm in ARMS):draws[k].append(values['synthetic'][k]-values['scratch'][k])
        result['paired_subject_intervals'][str(tolerance)]={k:list(np.quantile(v,[.025,.975])) if v else None for k,v in draws.items()}
    result.update(completed=True,identical_full_augmentation_streams=True,patches_per_epoch=8768,
                  bootstrap='1000 paired whole-subject resamples; 8 reused val subjects; no training-seed uncertainty')
    write(OUT/'analysis'/f'epoch-{epoch:03d}.json',result)
    return result


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    panels=[('node_f1','Node F1 at 2 mm'),('f1','Edge F1 at 2 mm'),('raw_node_recall','Raw node recall at 2 mm'),
            ('hungarian_mean_mm','Hungarian distance (mm)'),('conditional_recall','Conditional relation recall'),('total','Clean loss')]
    for arm,color in [('scratch','#cc6677'),('synthetic','#4477aa')]:
        sources=sorted((OUT/'large'/'predictions'/arm).glob('epoch-*/summary.json'))
        snapshots=sorted([read(p) for p in sources],key=lambda x:x['epoch'])
        for group,style in [('train','-'),('val_monitor','--')]:
            for ax,(key,title) in zip(axes.flat,panels):
                values=[]
                for p in snapshots:
                    if key=='total':value=p['losses'].get('train' if group=='train' else 'val',{}).get('total')
                    else:value=p['metrics'].get(group,{}).get('2.0',{}).get(key)
                    # Full-val losses have a different population: omit those
                    # from the fixed-val-monitor loss curve rather than mix them.
                    if key=='total' and group=='val_monitor' and p['full_validation']:continue
                    if value is not None:values.append((p['epoch'],value))
                if values:
                    ax.plot(*zip(*values),style,color=color,label=f'{arm} / {group}',linewidth=1.5)
                ax.set(title=title,xlabel='Full BRAVA epochs');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.tight_layout()
    fig.savefig(OUT/'learning-curves.png',dpi=160);plt.close(fig)


def publish():
    plan=read(OUT/'plan.json');selected=read(OUT/'batch-selection.json') if (OUT/'batch-selection.json').exists() else None
    submission=read(OUT/'submission.json') if (OUT/'submission.json').exists() else {}
    results=[read(OUT/'analysis'/f'epoch-{e:03d}.json') for e in MILESTONES if (OUT/'analysis'/f'epoch-{e:03d}.json').exists()]
    lines=['## D08: full-cohort BRAVA long-training initialization comparison — 27 October 2026','',
        '**Status: '+('completed through epoch 300; interpretation pending review.' if results and results[-1]['epoch']==300 else 'controlled protocol implemented; correctness/launch status below; results pending or partial.')+'**','',
        'This is distinct from the older full-data 20–40-pass weighted-node-CE/edge-focal runs and the modern 1,248-patch/4k-update pilots. It compares scratch against compatible SyntheticMRI initialization under one fixed modern recipe. It does not isolate every difference from those historical studies.','',
        f'Population: **8,768 training patches / 39 subjects**, full **1,796 validation patches / eight disjoint subjects**. Fixed representative train160 and val256 monitors are reused for learning curves. No test images/targets are loaded. Shared 256 queries, original K8 cubic B-splines, two relation tokens, single-level detector and existing `graph_only` route; no node refiner, token adapter, extra loss or unmatched supervision.','',
        'Node softmax focal gamma 2 / equal weights; matched-edge CE / equal weights / ratio upsampling 0.15; all established graph/spline coefficients and physical metrics retained. Both arms use the existing three-plane quarter turns plus Gaussian noise p=.35 and std 0–.015, no zoom. Augmentation is deterministically seeded per epoch/sample. Every epoch visits every patch once; complete augmented input/target stream hashes must match between arms.','',
        'Shared AdamW: LR 7e-5, encoder 3.5e-5, weight decay 1e-4, betas (.9,.999), eps 1e-8. Five full epochs of linear warmup, followed by cosine to 1e-6 over **300 full epochs**. Optimizer/global-step/LR continuity is preserved through 25-epoch Slurm chunks. No early stopping or validation-based LR choice. Float32; no AMP/accumulation.','',
        f'Source: `{plan["source"]["path"]}`; SHA256 `{plan["source"]["sha256"]}`. Selected by source-only SyntheticMRI validation edge F1 (**{plan["source"]["source_validation_edge_f1_2mm"]:.6f}**) among the documented compatible B0 continuations; source parent scored 0.5881. Refined A/A-xyz variants require architecture changes and are excluded. Preserve 120 learned object-query rows; move the two learned relation rows from 120/121 to 256/257; retain the 136 added rows from the identical seeded scratch model. Every other source tensor must transfer exactly. Fresh target AdamW for both arms.','']
    if selected:
        lines += [f'CPU audit and GPU smoke passed. Selected **batch {selected["batch_size"]}**, four data workers, **{selected["updates_per_epoch"]} updates/epoch**, **{selected["total_updates"]:,} target updates/arm**. Benchmark estimate: **{selected["estimated_training_gpu_hours_per_arm"]:.2f} training GPU-hours/arm**, excluding evaluation/startup/queue. Measured costs are recorded separately. Benchmark optimizer updates are discarded.','',
                  '| Arm | Batch | Seconds/update | Patches/s | Peak reserved GiB |','|---|---:|---:|---:|---:|']
        lines += [f'|{r["arm"]}|{r["batch_size"]}|{r["seconds_per_update"]:.4f}|{r["patches_per_second"]:.2f}|{r["peak_reserved_bytes"]/2**30:.2f}|' for r in selected['rows']]
        lines += ['']
    lines += ['Predefined milestone evaluations: **25, 50, 100, 150, 200, 300 epochs**. Primary comparison is epoch 300 at unchanged .5 node/edge thresholds. Saved-BN, clean evaluation batch eight. Positional 1/2-mm metrics and common-v2 (1-mm anchors/curve support, .25-mm sampling) remain separate. Report node/edge TP/FP/FN/F1, raw recall, localization/Hungarian distances, conditional connectivity/AP and prevalence, duplicate/count biases, components/cycles, Branch/Curve F1, losses and paired subject uncertainty.','']
    if results:
        lines += ['| Epoch | Arm | Train node / edge F1 (2mm) | Full val node / edge F1 | Val raw recall | Val conditional recall | Branch / Curve F1 |','|---:|---|---|---|---:|---:|---|']
        for result in results:
            for arm in ARMS:
                p=result['arms'][arm];t=p['metrics']['train']['2.0'];v=p['metrics']['val']['2.0'];c=p['common']['val']
                fmt=lambda x:'NA' if x is None else f'{x:.5f}'
                lines.append(f'|{result["epoch"]}|{arm}|{fmt(t["node_f1"])} / {fmt(t["f1"])}|{fmt(v["node_f1"])} / {fmt(v["f1"])}|{fmt(v["raw_node_recall"])}|{fmt(v["conditional_recall"])}|{fmt(c["common_branch_f1"])} / {fmt(c["common_curve_f1"])}|')
        lines += ['', 'Results are descriptive pending review of the complete trajectory. Paired intervals resample eight whole validation subjects and do not measure training-seed uncertainty. All graph discrepancies are relative to incomplete SWC annotations. Brain masks are not vessel labels.','',
                  '![BRAVA long-training fixed-monitor learning curves](../../../gnbm-brava-long-20261027/experiments/brava_long_20261027/learning-curves.png)','']
    lines += ['Interpretation predefined before training: increasing train **and** validation recovery supports a benefit from additional optimization/exposure under this recipe; strong train fit with persistently weak validation supports a generalization bottleneck; weak fit in both retains optimization/assignment/supervision alternatives. Tiny-set overfit does not prove the cohort needs only more epochs. Initialization gains are conditional on this shared schedule; pretraining adds source-data compute. No architecture/loss follow-up launches automatically.','',
              f'Isolated worktree: `{ROOT}`; artifacts: `{OUT}` with large checkpoints/caches at `{plan["large_root"]}`. `plan.json`, `cpu-preflight.json`, `gpu-smoke.json`, `batch-selection.json`, frozen code hashes/archive, per-epoch histories/input hashes, immutable checkpoint identities, prediction ZIPs, per-patch/subject metrics and Slurm accounting preserve provenance.','',
              'Submission record:','```json',json.dumps(submission,indent=2),'```','']
    section='\n'.join(lines)
    write(OUT/'report-publication.json',dict(report=str(MAIN/'docs/research/BRAVA_experiment_plan.md'),
         registry=str(MAIN/'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md'),completed_milestones=[r['epoch'] for r in results]))
    (OUT/'report.md').write_text(section)
    begin='<!-- BEGIN BRAVA LONG 20261027 -->';end='<!-- END BRAVA LONG 20261027 -->'
    coord=ROOT.parent/'experiment_gpu_coordination'
    with (coord/'brava-long-docs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=MAIN/'docs/research/BRAVA_experiment_plan.md';text=path.read_text();block=begin+'\n'+section+'\n'+end
        if begin in text:
            start=text.index(begin);finish=text.index(end,start)+len(end);text=text[:start]+block+text[finish:]
        else:text+='\n\n'+block+'\n'
        path.write_text(text)
        registry=MAIN/'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md';text=registry.read_text()
        status='DONE — measured' if results and results[-1]['epoch']==300 else 'SUBMITTED — verify'
        desc=f'Full train8768/39 subjects; 256 queries; scratch vs compatible SyntheticMRI B0; node focal2/edge CE ratio .15; fixed300 epochs. Collected milestones: {[r["epoch"] for r in results]}. No test access.'
        row=f'| **D08** | BRAVA full-cohort long-training initialization comparison | **{status}** | {desc} | [Existing BRAVA report](../docs/research/BRAVA_experiment_plan.md#d08-full-cohort-brava-long-training-initialization-comparison--27-october-2026) |'
        old=[line for line in text.splitlines() if line.startswith('| **D08** |')]
        if old:text=text.replace(old[0],row)
        else:
            anchor=next(line for line in text.splitlines() if line.startswith('| **D07** |'))
            text=text.replace(anchor,anchor+'\n'+row)
        registry.write_text(text)


def main(epoch=None):
    if epoch is not None:
        collect_epoch(epoch);plots()
    if (OUT/'submission.json').exists():
        p=read(OUT/'submission.json');jobs=[p.get('preflight'),p.get('smoke')]+p.get('training_arrays',[])+p.get('collectors',[])
        jobs=[j for j in jobs if j]
        if jobs:
            text=subprocess.check_output(['sacct','-j',','.join(jobs),'--format=JobID,State,ExitCode,Start,End,Elapsed,AllocTRES%60','--parsable2'],text=True)
            (OUT/'slurm-accounting.psv').write_text(text)
    publish()
    if epoch==300:
        claim=ROOT.parent/'experiment_gpu_coordination/brava-long-20261027.json'
        p=read(claim);p.update(state='completed; both arms and all milestone metrics collected',concurrent_h100=0);write(claim,p)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int);a=parser.parse_args();main(a.epoch)
