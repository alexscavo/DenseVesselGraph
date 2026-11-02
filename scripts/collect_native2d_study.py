#!/usr/bin/env python3
"""Collect measured outcomes without inference, retries or new GPU submission."""
import fcntl
import hashlib
import json
import subprocess
import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('NATIVE2D_STUDY_DIR',str(ROOT/'experiments/native2d_20261028')))
MAIN=ROOT.parent/'Graph-Native-Betti-Matching-Representation'
REVISION=OUT.name.endswith('targetfix')
START='<!-- NATIVE2D TARGETFIT LIVE 20261028 -->' if REVISION else '<!-- NATIVE2D HELDOUT LIVE 20261028 -->'
END='<!-- END NATIVE2D TARGETFIT LIVE 20261028 -->' if REVISION else '<!-- END NATIVE2D HELDOUT LIVE 20261028 -->'

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def additional_summary(held):
    if not held:return None
    rows=held['samples'];summary=held['summary'];positive=[r for r in rows if r['target_edges']>0]
    n=sum(r['target_nodes'] for r in rows)
    for t in ['1.0','2.0','4.0']:
        summary[t]['raw_node_recall_micro']=sum(round(r['raw_node_recall'][t]*r['target_nodes']) for r in rows)/n if n else None
    summary['target_edge_positive_crops']=len(positive)
    summary['empty_edge_target_crops']=len(rows)-len(positive)
    for geometry in ['spline','chord']:
        summary[geometry+'_edge_positive_crop_macro']={k:sum(r[geometry][k] for r in positive)/len(positive) if positive else None for k in ['common_branch_f1','common_curve_f1']}
    return held

def update(path,body):
    with Path(str(path)+'.native2d.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        text=path.read_text() if path.exists() else ''
        block=START+'\n'+body+'\n'+END
        if START in text:
            a,b=text.split(START,1);_,c=b.split(END,1);text=a+block+c
        else:text=text.rstrip()+'\n\n'+block+'\n'
        path.write_text(text)

def main():
    gate=read(OUT/'gates.json');audit=read(OUT/'av-wide-audit.json');submission=read(OUT/'submission.json')
    accounting=read(OUT/'slurm-accounting.json')
    record=dict(gates=gate,audit=audit,submission=submission,slurm_accounting=accounting,arms={})
    body=['## Native2D-03: corrected-target held-out AV-WIDE study — 28 October 2026' if REVISION else '## Native2D-02: held-out original-width AV-WIDE study — 28 October 2026','',
        'Authorized bounded B0 versus A-xyz pilot, following Native2D-01. All measurements below refer to image-disjoint validation, never test data. Patient independence is unverified. Distances are pixels.', '']
    if audit:body.append(f"Cohort: 17 train / 6 validation images; six reserved test images have header-only inspection. Image07 excluded for its diagonal self-loop. {audit['samples']['train']} train / {audit['samples']['validation']} validation 128² tiles; {audit['queries']} object queries selected from contracted training GT. All spatial tiles retained, including empty graphs.")
    finding=read(OUT/'target-fit-failure.json')
    if finding:body.append('**Target-quality limitation:** the original irregular-sampling K8 least-squares fit produces pathological controls (one verified128² tile reaches56,890px), causing shared loss spikes. This pilot is preserved but does not support a reliable geometry conclusion. A separately recorded single corrected-target pair uses uniform arc-length samples of unchanged dense paths; no graph labels, image splits, architecture or loss policy change.')
    fit=read(OUT/'target-fit-audit.json')
    if fit:body.append(f"Correction: uniform64 arc-length samples of the same polyline before K8 fitting. All graph/image/split/reference-path tensors are exactly unchanged. Refit {fit['branches']} branches; control max absolute coordinate {fit['old_max_abs_control_pixel']:.1f}→{fit['new_max_abs_control_pixel']:.1f}px, uniform design condition number{fit['uniform_design_condition_number']:.2f}. One fresh fixed-budget pair only; original checkpoints are not resumed.")
    if gate:
        body.append(f"Full ResNet101/512D, 3 encoder + 3 decoder layers, 5 relation tokens: gates {'PASSED' if gate['passed'] else 'FAILED'}; {gate['full_architecture_optimizer_updates']} optimizer updates on two 128² training tiles before the experiment. Shared initialization and graph-only gradient checks passed. Budget fixed to {gate['updates']} updates per arm, batch4, no automatic extension.")
        body.append('Throughput (existing PyTorch deformable attention): '+', '.join(f"{r['arm']} {r['seconds_per_update']:.4f}s/update, {r['images_per_second']:.2f} images/s" for r in gate['benchmark'])+'.')
    if submission:body.append('Slurm: '+', '.join(f'{k} `{v}`' for k,v in submission.items() if isinstance(v,str) and v.isdigit())+'.')
    if accounting and accounting.get('B0',{}).get('state')=='TIMEOUT':
        body.append('Scheduler outcome: A-xyz completed normally. B0 completed all6,000 updates and all506-crop final metrics in about436s, then its redundant initialization-checkpoint curve evaluation hit the1h45 Slurm limit. Its job state is **TIMEOUT**, despite complete primary final-checkpoint evidence. The initialization artifact is preserved as `B0/excluded-initialization.pt`; the trained-checkpoint index was repaired from the genuine final checkpoint/metrics, without retraining or new inference. No budget extension or threshold tuning was performed.')
    body.extend(['','Primary comparison: **final fixed-budget checkpoints**, all validation tiles. Branch/Curve columns use GT edge-positive crops; all-crop aggregates are also retained in JSON, where empty/empty graphs can receive perfect scores.',
        '', '| Arm | State | Final update | Node F1 @2px | Edge F1 @2px | Branch F1 @1px | K8 Curve F1 @1px | Chord Curve F1 @1px |',
        '|---|---|---:|---:|---:|---:|---:|---:|'])
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for arm in ['B0','Axyz']:
        directory=OUT/arm;status=read(directory/'status.json');held=additional_summary(read(directory/'best-heldout.json'));failure=read(OUT/(arm+'-failure.json'))
        final=additional_summary(read(directory/f"validation-{gate['updates']:05d}.json")) if gate else None
        curve=read(directory/'learning-curve.json') or []
        record['arms'][arm]=dict(status=status,selected_heldout=held,final_heldout=final,failure=failure)
        state='complete' if final else ('failed' if failure else (status or {}).get('state','not started'))
        if final:
            s=final['summary'];body.append(f"| {arm} | {state} | {final['step']} | {s['2.0']['node']['f1']:.4f} | {s['2.0']['edge']['f1']:.4f} | {s['spline_edge_positive_crop_macro']['common_branch_f1']:.4f} | {s['spline_edge_positive_crop_macro']['common_curve_f1']:.4f} | {s['chord_edge_positive_crop_macro']['common_curve_f1']:.4f} |")
        else:body.append(f'| {arm} | {state} | — | — | — | — | — | — |')
        if curve:
            for ax,kind in zip(axes[:2],['node','edge']):
                ax.plot([r['step'] for r in curve],[r['validation']['2.0'][kind]['f1'] for r in curve],marker='o',label=arm)
                ax.set_ylabel(kind+' micro F1 @2px')
            train=[]
            if (directory/'training.jsonl').exists():
                train=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
            if train:axes[2].plot([r['step'] for r in train],[r['losses']['total'] for r in train],label=arm)
    for ax in axes:
        ax.set_xlabel('optimizer update');ax.grid(alpha=.25)
        if ax.lines:ax.legend()
    axes[2].set_ylabel('training total loss');fig.tight_layout();fig.savefig(OUT/'learning-curves.png',dpi=150);plt.close(fig)
    if all(record['arms'][a]['final_heldout'] is not None for a in ['B0','Axyz']):
        body.extend(['','| Arm | Node mAP | Node mAR | Edge mAP | Edge mAR | Raw node recall @2px, micro | Filtered node recall @2px |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for arm in ['B0','Axyz']:
            s=record['arms'][arm]['final_heldout']['summary']
            body.append(f"| {arm} | {s['node_mAP_crop_macro']:.4f} | {s['node_mAR_crop_macro']:.4f} | {s['edge_mAP_crop_macro']:.4f} | {s['edge_mAR_crop_macro']:.4f} | {s['2.0']['raw_node_recall_micro']:.4f} | {s['2.0']['node']['recall']:.4f} |")
        if all(sum(r['predicted_nodes'] for r in record['arms'][a]['final_heldout']['samples'])==0 for a in ['B0','Axyz']):
            body.append('\n**Negative bounded pilot:** both final checkpoints retain zero nodes/edges at the predeclared0.5 confidence thresholds, missing7,613 GT nodes and6,308 GT edges. Conditional relation performance is undefined because no filtered endpoint pairs remain; zero duplicates are a consequence of empty predictions. There is no evidence of A-xyz benefit or spline superiority. The8 edge-empty target crops inflate all-crop Branch/Curve macro F1 to0.01581; on498 edge-positive crops these metrics are exactly zero. Pixel-space raw coverage and unthresholded AP remain visible, and no validation-driven operating-point change is made. Stop at the fixed budget; no automatic extension.')
    body.extend(['','Intermediate validation uses 24 fixed spatial tiles (4/image); final and selected trained checkpoints are evaluated on every validation tile. Curve points at final therefore use a larger population; do not interpret that discontinuity as a training change. Selection uses the same 24 tiles throughout, excludes initialization, and resolves trained ties toward later updates. Best and final outcomes are both saved.',
        '', 'Detection AP/AR uses unthresholded query scores, IoU .50:.05:.95, node limit equal to query capacity and edge top300; reported AP is crop macro. Physical node/edge F1 is micro over crops at 1/2/4px, including crop-boundary anchors. Branch/Curve F1 is crop macro at 1px, .25px sampling. Spline/chord use identical inferred nodes/edges; topology cannot change. Raw/filtered recall, relation AP populations, duplicates and Betti errors are retained per crop. No subject-level uncertainty claim, full-image stitching, independent manual topology or pretraining benefit is established.',
        '', f'Evidence and checkpoints: `{OUT}`. [Protocol and dataset readiness](../docs/research/native2d_heldout_20261028.md). Learning curves: `learning-curves.png`; full metrics: `B0/best-heldout.json`, `Axyz/best-heldout.json`; final metrics: `<arm>/validation-<budget>.json`; weights/optimizer: `<arm>/best.pt`, `<arm>/latest.pt`. Failed gates and jobs are reported without automatic resubmission.'])
    text='\n\n'.join(body[:0])+ '\n'.join(body)
    (OUT/'report.md').write_text(text+'\n')
    (OUT/'results.json').write_text(json.dumps(record,indent=2,allow_nan=True)+'\n')
    for path in [MAIN/'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md',MAIN/'research/image-to-graph-vessel.md',MAIN/'research/experiments.md']:
        update(path,text)
    hashes=[]
    for arm in ['B0','Axyz']:
        s=read(OUT/arm/'status.json')
        if s:hashes.append((s['exposure_sha256'],s['initialization_sha256']))
    assert len(set(hashes))<=1,'Paired initialization/exposure mismatch'
    coord=ROOT.parent/'experiment_gpu_coordination'/('native2d-targetfit-20261028.json' if REVISION else 'native2d-20261028.json')
    claim=read(coord)
    if claim and '--live' not in sys.argv:
        complete=all(record['arms'][a]['final_heldout'] is not None for a in ['B0','Axyz'])
        claim.update(state='completed held-out pilot' if complete else 'collected; inspect gates and arm status',
            concurrent_h100=0,reserved_after_dependency=0,completed=complete)
        coord.write_text(json.dumps(claim,indent=2)+'\n')
    print(text)

if __name__=='__main__':main()
