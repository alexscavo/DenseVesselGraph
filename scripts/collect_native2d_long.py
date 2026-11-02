#!/usr/bin/env python3
"""Read-only learning-curve collection; no inference or new training submission."""
import csv
import fcntl
import json
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native2d_mixed_20261028'
DATA=Path(json.loads((OUT/'plan.json').read_text())['data_root'])
PROJECTS=ROOT.parent.parent if ROOT.parent.name=='GNBM' else ROOT.parent
MAIN=PROJECTS/'Graph-Native-Betti-Matching-Representation'

def read(path):return json.loads(path.read_text()) if path.exists() else None

def main():
    plan=read(OUT/'plan.json');submission=read(OUT/'submission.json');gate=read(OUT/'gates.json');records=[]
    accounting={}
    if submission:
        ids=list(submission.get('runs',{}).values())+[submission['gates']]
        try:
            raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobIDRaw,State,ExitCode,Elapsed,AllocTRES'],text=True,timeout=20)
            for line in raw.splitlines():
                fields=line.split('|')
                if len(fields)>=5:accounting[fields[0]]=dict(state=fields[1],exit_code=fields[2],elapsed=fields[3],allocated_tres=fields[4])
            (OUT/'slurm-accounting.json').write_text(json.dumps(accounting,indent=2)+'\n')
        except (OSError,subprocess.SubprocessError):
            accounting=read(OUT/'slurm-accounting.json') or {}
    body=['## Native2D-04 — mixed plant/vessel pretraining and longer vessel training, 28 October 2026','',
        'User clarified that pretraining must mix plants and vessels. The superseded pure-plant gate/launcher1670808/1670809/1670810 were cancelled while pending; no pure-plant training occurred. Use50 mixed-pretraining epochs, then100 vessel-only epochs, with unchanged confidence cutoffs0.5. Previous pilot artifacts and the superseded protocol archive are retained.', '']
    body.append('Protocol: [native2d_long_20261028.md](../docs/research/native2d_long_20261028.md). Queued/running states are not completed experimental results.')
    readiness=read(OUT/'readiness-progress.json')
    if readiness and not submission:body.append(f"Preparation job `{readiness['preparation_job']}`; conditional CPU continuation `{readiness['conditional_continuation_job']}`. Ten CPU tests and native cache-loader parity passed. Dataset-wide preparation and source topology audits precede GPU gates and real training.")
    if submission:body.append(f"GPU gate job `{submission['gates']}`; conditional training launcher `{submission.get('train_launcher','pending')}`. The launcher submits training only after all dataset/GPU gates pass.")
    if plan:
        body.append(f"Full native ResNet101/512D,3+3 layers,5 relation tokens, common{plan['common_queries']} object queries from contracted training GT distributions. Each vessel stage:100 exact passes, batch4, identical exposure/augmentation/optimizer schedule within each B0/A-xyz pair.")
        if 'plants' in plan['sources']:body.append('For each vessel dataset, pretrain one B0 on plants plus ONLY that target training split. Use the existing50:50 domain-balanced sampling with replacement and a twice-source epoch (51,734 draws), batch32, for50 epochs. The final mixed checkpoint for that target initializes both B0/A-xyz; A-xyz starts with zero coordinate residuals. Fresh optimizer/scheduler for100 vessel-only epochs, batch4; no plant replay. All pretraining geometry and target-validation distances use pixels; REAVER fine-tuning/final evaluation restores its original micron calibration. Pretraining validation/selection uses target validation only, never pooled plant/vessel metrics.')
        body.extend(['','| Dataset | Train / validation crops | Train maximum nodes | Units |','|---|---:|---:|---|'])
        for dataset,a in plan['sources'].items():body.append(f"| {dataset} | {a['samples']['train']} / {a['samples']['validation']} | {a['training_maximum_nodes']} | {read(DATA/dataset/'manifest.json')['units']} |")
    if gate:
        body.append('\nGPU gates passed: initialization/gradient/strict shared-weight transfer checks, fixed1,200-update full-model two-crop overfit per cohort and production-batch throughput. Gate weights are never reused for training.')
        body.extend(['','| Dataset / benchmark | Batch | Measured images/s | Estimated optimizer hours/run |','|---|---:|---:|---:|'])
        for name,b in gate['benchmark'].items():body.append(f"| {name} | {b['batch']} | {b['images_per_second']:.2f} | {b['estimated_training_hours']:.2f} |")
    if not plan:
        body.append('')
        for dataset in ['avwide','reaver','octasynth','plants']:
            audit=read(DATA/dataset/'audit.json')
            if audit:body.append(f"Prepared {dataset}: {audit['samples']['train']} training /{audit['samples']['validation']} validation crops; maximum training GT nodes{audit['training_maximum_nodes']}; units{read(DATA/dataset/'manifest.json')['units']}.")
            else:body.append(dataset+': preparation incomplete; no long-run results exist yet.')
    body.extend(['','| Run | Slurm | State | Epoch | Node mAP | Edge mAP | Node F1 @2 units | Edge F1 @2 units |','|---|---|---|---:|---:|---:|---:|---:|'])
    jobs=submission.get('runs',{}) if submission else {}
    names=[name for d in ['avwide','reaver','octasynth'] for name in [d+'_mixedpretrain_50',*[d+'_'+init+'_'+v for init in ['scratch','mixed'] for v in ['B0','Axyz']]]]
    for name in names:
        directory=OUT/'runs'/name;status=read(directory/'status.json');protocol=read(directory/'protocol.json');final=read(directory/'final-heldout.json')
        failure=read(OUT/(name+'-failure.json'))
        state='complete' if final and status and status['state']=='complete' else ('failed' if failure else (status or {}).get('state','queued' if name in jobs else 'not submitted'))
        job_state=accounting.get(jobs.get(name),{}).get('state','')
        if job_state.startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','DEPENDENCY')):state=job_state.lower()
        elif job_state=='RUNNING' and not status:state='starting / caching'
        summary=final['summary'] if final else None
        records.append(dict(run=name,job=jobs.get(name),state=state,slurm=accounting.get(jobs.get(name)),status=status,protocol=protocol,final_summary=summary,failure=failure))
        if summary:body.append(f"| {name} | {jobs.get(name,'—')} | {state} | {status['epoch']} | {summary['node_mAP_crop_macro']:.4f} | {summary['edge_mAP_crop_macro']:.4f} | {summary['2.0']['node']['f1']:.4f} | {summary['2.0']['edge']['f1']:.4f} |")
        else:body.append(f"| {name} | {jobs.get(name,'—')} | {state} | {(status or {}).get('epoch','—')} | — | — | — | — |")
    for dataset in ['avwide','reaver','octasynth','pretrain_avwide','pretrain_reaver','pretrain_octasynth']:
        fig,axes=plt.subplots(1,4,figsize=(15,3.4));any_curve=False
        for r in records:
            if dataset.startswith('pretrain_'):
                if r['run']!=dataset.removeprefix('pretrain_')+'_mixedpretrain_50':continue
            elif not r['run'].startswith(dataset+'_') or '_mixedpretrain_' in r['run']:continue
            directory=OUT/'runs'/r['run'];curve=read(directory/'learning-curve.json') or []
            if curve:
                any_curve=True
                for ax,key in zip(axes[:2],['node_mAP_crop_macro','edge_mAP_crop_macro']):ax.plot([x['epoch'] for x in curve],[x['validation'][key] for x in curve],label=r['run'])
                axes[2].plot([x['epoch'] for x in curve],[x['validation']['2.0']['edge']['f1'] for x in curve],label=r['run'])
            if (directory/'training.jsonl').exists():
                rows=[json.loads(x) for x in (directory/'training.jsonl').read_text().splitlines()]
                axes[3].plot([x['epoch'] for x in rows],[x['losses']['total'] for x in rows],label=r['run']);any_curve=True
        for ax,title in zip(axes,['Node mAP','Edge mAP','Edge F1 @2 native units','Training loss']):
            ax.set_xlabel('Epoch');ax.set_title(title);ax.grid(alpha=.25)
            if ax.lines:ax.legend(fontsize=6)
        fig.suptitle(dataset+' — confidence thresholds0.5; interim subset, final full validation');fig.tight_layout()
        if any_curve:fig.savefig(OUT/('curves-'+dataset+'.png'),dpi=140)
        plt.close(fig)
    # Paired controls must agree on image exposure and initial shared weights.
    for dataset in ['avwide','reaver','octasynth']:
        for initial in ['scratch','mixed']:
            pair=[r for r in records if r['run'] in [dataset+'_'+initial+'_B0',dataset+'_'+initial+'_Axyz'] and r['protocol']]
            if len(pair)==2:
                a,b=(r['protocol'] for r in pair);assert a['exposure_sha256']==b['exposure_sha256'] and a['initialization_sha256']==b['initialization_sha256']
    body.extend(['','Primary comparisons use the final fixed-epoch checkpoints on complete validation populations; selected checkpoints are secondary. Selection uses fixed interim source-image groups, excludes initialization and resolves trained ties toward later epochs. Pixel and micron metrics are reported separately. Raw versus filtered recall, conditional relation populations/AP, duplicates, components/cycles and same-graph K8/chord geometry remain in per-run validation JSON. Mixed-pretraining validation reports target-vessel detection/graph metrics in pixels. Plant graphs supply sparse source supervision; plant validation is not pooled into target metrics.',
        '', 'No test arrays or predictions are used. AV-WIDE remains17 training /6 validation images with unknown patient independence; image07 excluded. REAVER is the separately verified diaphragm60x4/2-image cohort in microns. OCTA uses official source-image membership and regenerated128² crops with coordinate/crop-boundary QC. Plant parent IDs are disjoint across official train/val/test membership.',
        '', 'Mixed pretraining adds vessel exposures as well as biological data. Relative to scratch100, a gain cannot isolate a biological-pretraining causal effect; no extra vessel-only-pretraining control or sweep is launched. Realized per-domain draws and source/target memberships are recorded in protocol.json and exposure.npy.',
        '', f'Artifacts: `{OUT}`; validated datasets reused read-only from `{DATA}`. Source/config hashes: `source.zip`, `code-identities.json`; epoch losses and full state checkpoints: `runs/<name>/`; final metrics: `final-heldout.json`; plots: `curves-<dataset>.png` and separate pretraining curves. Stop at50 mixed /100 vessel epochs; exceptions stop jobs without automatic retries, new sweeps or budget extensions.'])
    text='\n'.join(body)+'\n';(OUT/'report.md').write_text(text);(OUT/'results.json').write_text(json.dumps(dict(plan=plan,gates=gate,submission=submission,runs=records),indent=2)+'\n')
    marker='<!-- NATIVE2D LONG LIVE 20261028 -->';end='<!-- END NATIVE2D LONG LIVE 20261028 -->'
    for name in ['research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md','research/image-to-graph-vessel.md','research/experiments.md']:
        path=MAIN/name
        with Path(str(path)+'.native2d.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);old=path.read_text();block=marker+'\n'+text+end
            if marker in old:a,b=old.split(marker,1);_,c=b.split(end,1);old=a+block+c
            else:old=old.rstrip()+'\n\n'+block+'\n'
            path.write_text(old)
    eligible=[r for r in records if r['run'] in jobs]
    terminal=('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')
    finished=bool(eligible) and all((r.get('slurm') or {}).get('state','').startswith(terminal) for r in eligible)
    gate_failed=bool(submission) and not jobs and accounting.get(submission['gates'],{}).get('state','').startswith(terminal[1:])
    if submission and (finished or gate_failed):
        p=PROJECTS/'experiment_gpu_coordination/native2d-mixed-20261028.json';claim=read(p)
        if claim:
            claim.update(state='completed eligible fixed-epoch runs' if eligible and all(r['state']=='complete' for r in eligible) else 'stopped; inspect failed/cancelled jobs and saved evidence',concurrent_h100=0,reserved_after_dependency=0)
            p.write_text(json.dumps(claim,indent=2)+'\n')
    print(text)

if __name__=='__main__':main()
