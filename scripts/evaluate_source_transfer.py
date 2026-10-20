#!/usr/bin/env python3
"""Matched held-out evaluation and existing-validation comparison for source transfer."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def collect(plan, split):
    rows=[]
    for arm,run in plan['runs'].items():
        run=Path(run)
        if split=='validation':
            folders=list((run/'best-full-validation').glob('validation-*'))
            assert len(folders)==1,(arm,folders)
            folder=folders[0]
        else:
            result_root=Path(plan.get('existing_results',{}).get(arm,str(Path(plan['output_root'])/arm)))
            summary=json.loads((result_root/'summary.json').read_text())
            assert summary['sample_ids_sha256']==plan['test_ids_sha256']
            folder=Path(summary['metrics_directory'])
        native=json.loads((folder/'summary.json').read_text())
        common=json.loads((folder/'common-v2-summary.json').read_text())
        core=json.loads((folder/'vessel-core-summary.json').read_text())
        assert native['samples']==(1952 if split=='validation' else 13071)
        row=dict(model=arm,samples=native['samples'])
        row.update({k:native[k] for k in ('node_mAP','node_mAR','edge_mAP','edge_mAR','node_f1','edge_f1','beta0_absolute_error','beta1_absolute_error')})
        row.update({k:common[k] for k in ('common_branch_f1','common_curve_f1','common_guide_distance_mm','same_endpoint_chord_common_curve_f1')})
        row.update({'core_'+k:v for k,v in core.items() if isinstance(v,(int,float))})
        rows.append(row)
    dest=Path(plan['report_root']);dest.mkdir(parents=True,exist_ok=True)
    with (dest/f'{split}-comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    columns=['model','node_mAP','node_mAR','edge_mAP','edge_mAR','node_f1','edge_f1','common_branch_f1','common_curve_f1','beta0_absolute_error','beta1_absolute_error']
    text=f'# Source transfer: {split} comparison\n\nValidation-selected best checkpoints; same SyntheticMRI population and evaluation protocol. '
    text+='Graph-only pretraining followed by straight-edge or detached residual B-spline fine-tuning. Evaluation uses saved BatchNorm statistics, without recalibration. '
    text+='Single seed; no statistical superiority claim. Guides are uncertain extracted references.\n\n'
    if split=='validation':text+='These are validation results used for model selection, not held-out test results.\n\n'
    text+='| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'
    best={k:(min if k.endswith('_absolute_error') else max)(r[k] for r in rows) for k in columns if k!='model'}
    for row in rows:
        cells=[row['model']]
        for k in columns[1:]:
            value=f'{row[k]:.4f}'
            cells.append('**'+value+'**' if row[k]==best[k] else value)
        text+='| '+' | '.join(cells)+' |\n'
    text+='\n**Bold** marks the best value per column (higher AP/AR/F1; lower Betti errors).\n'
    (dest/f'{split}-comparison.md').write_text(text)
    print(text)

def evaluate(plan, arm, smoke):
    from data.loaders import build_evaluation_loader
    from models import build_model
    from models.checkpoint import load_checkpoint_file
    from scripts.train_spline_weight_pilot import geometry,write_json,seed
    from scripts.train_relation_distance_pilot import prediction_diagnostics
    run=Path(plan['runs'][arm]);assert (run/'COMPLETE').is_file()
    checkpoint=load_checkpoint_file(run/'best.pt');cfg=checkpoint['config']
    cfg['data']['train_augmentation']=False;cfg['tracking']['enabled']=False
    assert not cfg['domain_adaptation']['enabled']
    ids=json.loads(Path(plan['population_manifest']).read_text())['test_ids']
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==plan['test_ids_sha256']
    loader=build_evaluation_loader(cfg,dataset_name='synthetic_mri',split='test',sample_ids=ids[:2] if smoke else ids)
    assert [r.sample_id for r in loader.dataset.records]==(ids[:2] if smoke else ids)
    assert cfg['data']['datasets']['synthetic_mri']['root']==plan['dataset_root']
    assert cfg['evaluation']==plan['evaluation_protocol']
    seed(plan['seed']);model=build_model(cfg).cuda();model.load_state_dict(checkpoint['net']);model.eval()
    out=Path(plan['output_root'])/(('smoke-' if smoke else '')+arm);out.mkdir(parents=True,exist_ok=False)
    metrics=geometry(model,loader,cfg,out,checkpoint['steps'],None,checkpoint['epoch'],True,split='test')
    folder=out/f"test-{checkpoint['steps']:06d}"
    baseline=json.loads(Path(plan['baseline_diagnostic']).read_text())
    prediction_diagnostics(folder/'predictions.json',Path(plan['dataset_root']),'test',baseline,core=True)
    if plan.get('duplicate_nodes'):
        from scripts.limited_data_support import duplicate_diagnostics
        duplicate_diagnostics(folder/'predictions.json',Path(plan['dataset_root']),'test')
    write_json(out/'summary.json',dict(checkpoint=str(run/'best.pt'),epoch=checkpoint['epoch'],samples=len(loader.dataset),sample_ids_sha256=plan['test_ids_sha256'] if not smoke else None,metrics_directory=str(folder),metrics=metrics))
    print(json.dumps(metrics))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('experiments/leaf_pretraining/evaluation/plan.json'))
    p.add_argument('--arm');p.add_argument('--smoke',action='store_true');p.add_argument('--collect',choices=['validation','test'])
    a=p.parse_args();plan=json.loads(a.config.read_text())
    if a.collect:collect(plan,a.collect)
    else:
        assert a.arm in plan['runs'];evaluate(plan,a.arm,a.smoke)
