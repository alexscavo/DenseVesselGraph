#!/usr/bin/env python3
"""Prepare exact limited-data membership or collect the locked paired comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path('experiments/limited_botanical_transfer')
REPORT = Path('docs/research/pretraining/limited-data-botanical-transfer.md')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subject(name):
    match = re.fullmatch(r'sample_(\d+)_(\d+)', name)
    assert match, name
    return str(int(match[1]))


def prepare(policy):
    import yaml
    assert not (HERE/'plan.yaml').exists(), 'Existing paired experiment; do not resample or overwrite it'
    previous = yaml.safe_load(Path('configs/leaf_generalization_da_b32.yaml').read_text())
    population_path = Path(previous['population_manifest'])
    population = json.loads(population_path.read_text())
    base = yaml.safe_load(Path(previous['base_config']).read_text())
    mri = Path(base['data']['datasets']['synthetic_mri']['root'])
    leaf = Path(previous['plant_root'])
    selected = sorted(sorted(population['train_ids'], key=lambda name:
        (hashlib.sha256(f"{previous['seed']}\0train\0{name}".encode()).digest(), name))[:200])
    assert len(set(selected)) == 200
    expected = {'train': population['train_ids'], 'val': population['validation_ids'], 'test': population['test_ids']}
    subjects = {}
    # Verify on-disk memberships using directory sets, including all required sidecars.
    for split, ids in expected.items():
        assert {p.name.removesuffix('_data.nii.gz') for p in (mri/split/'raw').glob('*_data.nii.gz')} == set(ids)
        for folder, suffix in [('seg','_seg.nii.gz'), ('vtp','_graph.vtp'), ('spline','.npz')]:
            assert {p.name.removesuffix(suffix) for p in (mri/split/folder).glob('*'+suffix)} == set(ids), (split,folder)
        subjects[split] = sorted({subject(name) for name in ids},key=int)
    for left,right in [('train','val'),('train','test'),('val','test')]:
        assert not set(subjects[left]) & set(subjects[right]), 'Subject leakage'
    with (mri/'patch_index.csv').open() as stream:
        index = {row['sample_id']: row for row in csv.DictReader(stream)}
    assert all(str(int(index[name]['patient_id'])) == subject(name) and index[name]['split'] == split
               for split, ids in expected.items() for name in ids)
    leaf_manifest = json.loads((leaf/'manifest.json').read_text())
    leaf_ids = sorted(row['sample_id'] for row in leaf_manifest['accepted']['train'])
    for folder,suffix in [('raw','_data.png'),('seg','_seg.png'),('vtp','_graph.vtp'),('spline','.npz')]:
        assert {p.name.removesuffix(suffix) for p in (leaf/'train'/folder).glob('*'+suffix)} == set(leaf_ids), folder
    assert (leaf/'READY').is_file()
    # Leaf IDs are split-local (some names are reused for different PNGs).
    # Only train is sampled; no source validation enters model selection.
    reused_leaf_names = set(leaf_ids) & {r['sample_id'] for r in leaf_manifest['accepted']['val']}
    # Derive the fixed step budgets from the completed reference run summaries.
    reference = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/plant_generalization_20261014_da_b32_w1_restart/runs')
    counts = {}
    for arm in ('pretrain_vessels','pretrain_mixed','vessels_spline','mixed_spline'):
        counts[arm] = json.loads((reference/arm/'summary.json').read_text())['steps']
    assert counts['pretrain_vessels'] == counts['pretrain_mixed']
    assert counts['vessels_spline'] == counts['mixed_spline']
    plan = dict(previous)
    plan.update(output_root='/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/limited_botanical_200_20261020',
        train_samples=200, fixed_update_budget=True, early_stopping=False,
        tracking_group='limited-botanical-200-20261020',
        population_manifest=str(HERE/'population-manifest.json'), evaluation_membership_run=str(HERE))
    plan['domain_adaptation'] = dict(plan['domain_adaptation'], enabled=policy=='mixed_only')
    plan['domain_policy'] = policy
    plan['pretrain_draws_per_epoch'] = counts['pretrain_vessels']//plan['pretrain_epochs']*plan['batch_size']
    plan['finetune_draws_per_epoch'] = counts['vessels_spline']//plan['finetune_epochs']*plan['batch_size']
    assert plan['pretrain_draws_per_epoch']*plan['pretrain_epochs']//32 == counts['pretrain_vessels']
    assert plan['finetune_draws_per_epoch']*plan['finetune_epochs']//32 == counts['vessels_spline']
    # Original leaf embedding, with its established 3D placement/rotation/flips.
    plan['source_augmentation'] = {}
    metadata = dict(train_ids=selected, validation_ids=population['validation_ids'], test_ids=population['test_ids'],
        counts=dict(train=200,val=len(population['validation_ids']),test=len(population['test_ids'])),
        seed=plan['seed'], source_population_manifest=str(population_path),source_population_sha256=sha(population_path),
        source_counts=population['counts'], selected_percentage=100*len(selected)/len(population['train_ids']),
        original_training_patches=len(population['train_ids']),
        selected_training_subjects=sorted({subject(name) for name in selected},key=int), split_subjects=subjects,
        grouping='patient_id in original patch_index.csv; verified against the subject token of every patch ID',
        leaf_root=str(leaf), leaf_training_count=len(leaf_ids),leaf_manifest_sha256=sha(leaf/'manifest.json'),
        leaf_training_ids=leaf_ids,reference_steps=counts,
        leaf_train_val_reused_filenames=len(reused_leaf_names),
        selection='Existing seeded_random SHA256(seed\\0train\\0sample_id) rank, first 200; membership then sorted')
    save(HERE/'population-manifest.json',metadata)
    save(HERE/'plan.yaml',plan)  # JSON is valid YAML, read by the existing runner.
    membership = json.loads((Path(previous['evaluation_membership_run'])/'evaluation-sample-ids.json').read_text())
    assert set(membership['val']) <= set(population['validation_ids'])
    save(HERE/'evaluation-sample-ids.json',dict(train=selected,val=membership['val']))
    (HERE/'selected-synthetic-mri-ids.txt').write_text('\n'.join(selected)+'\n')
    evaluation = json.loads(Path('experiments/combined_pretraining/evaluation-plan.json').read_text())
    evaluation.update(runs={name:str(Path(plan['output_root'])/'runs'/arm) for name,arm in
                           [('vessel_only','vessels_spline'),('leaves_vessels','mixed_spline')]},
        population_manifest=str(HERE/'population-manifest.json'),
        output_root=str(Path(plan['output_root'])/'test'),report_root=str(HERE/'results'),duplicate_nodes=True)
    evaluation.pop('existing_results',None)
    save(HERE/'evaluation-plan.json',evaluation)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(f'''# Limited-data SyntheticMRI: botanical versus vessel-only pretraining

## Hypothesis and locked comparison

Does leaf diversity improve complete vessel graph reconstruction when only 200 annotated vascular patches are available, under equal optimizer-update and example-processing budgets? One shared seed ({plan['seed']}); a promising difference will need replication. The reference full-data study favored MRI-only. These runs test a different annotation budget, and their scores must not be treated as a seed-controlled contrast with that earlier study.

A: selected 200 MRI patches for pretraining and fine-tuning. B: the identical 200 MRI patches plus {len(leaf_ids):,} leaf training patches for pretraining, then the same 200 MRI patches for fine-tuning. No other sources. The subset is {metadata['selected_percentage']:.2f}% of the actual {len(population['train_ids']):,}-patch reference training population.

## Data provenance and membership

MRI root: `{mri}`. Leaf root: `{leaf}`. Original PNG/seg/VTP leaf data use the established synthetic 2D-to-3D embedding, rotations/flips and degree-two contraction; the newer nonplanar/textured variant is not enabled. Leaf source validation ({len(leaf_manifest['accepted']['val'])} patches) is not used in training or checkpoint selection.

All required MRI raw/seg/VTP/spline and leaf raw/seg/VTP/spline memberships were verified against their manifests. The MRI `patient_id` grouping was checked for every patch: no subject occurs in more than one split. The 200 selected patches cover {len(metadata['selected_training_subjects'])} training volumes. No validation or test MRI enters training. Selection uses the existing seeded-random ranking rather than graph quality or outcomes.

Exact IDs, subject groups, source manifest hashes and leaf IDs: `experiments/limited_botanical_transfer/population-manifest.json`. Plain IDs: `experiments/limited_botanical_transfer/selected-synthetic-mri-ids.txt`.

## Stages, initialization and losses

Reuse the documented full-data protocol: graph-only pretraining followed by joint detector training with a detached residual B-spline geometry branch (8 controls, local image features, spline weight {plan['spline_weight']}). The spline parameters are frozen during graph-only pretraining. Both fine-tuning branches initialize spline parameters identically from a fresh shared seed, and transfer their own validation-selected detector. A common immutable scratch initializer is generated by verification; both pretraining arms load it and record the same detector/spline SHA256. No previously trained weights are loaded as a scratch initializer. Reference `base_config` supplies settings only.

Domain policy: **{policy}**. `none` disables domain adaptation in both arms and gives identical model/loss definitions. `mixed_only` retains the historical domain loss weight 1.0 only for B and therefore tests the transfer recipe with a different auxiliary objective; it is not a data-only ablation.

Same Hungarian matcher, graph losses, AdamW (LR {plan['learning_rate']}, weight decay 1e-4), batch 32, polynomial power 0.9, no early stopping. MRI augmentation: quarter-turn 3D rotations and Gaussian noise probability 0.35 / maximum SD 0.015; no zoom. Train BatchNorm statistics during updates; use saved statistics at evaluation. The whole detector trains in both stages; only spline gradients are detached from detector tokens/endpoints in fine-tuning.

## Fixed budgets and exposure accounting

Reference summaries establish {counts['pretrain_vessels']:,} pretraining updates and {counts['vessels_spline']:,} fine-tuning updates. Each arm uses exactly those counts. Fifty scheduler units of 1,563 updates and 100 units of 313 updates preserve the old schedule and validation cadence. Warmup is 1,563 / 313 updates. These are reference-sized scheduler units, not passes through the 200-patch dataset.

Every batch has 32 examples: {counts['pretrain_vessels']*32:,} pretraining presentations and {counts['vessels_spline']*32:,} fine-tuning presentations per arm. The reference run's partial last batches are padded by extra replacement draws (800 extra presentations across pretraining, 1,600 across fine-tuning), preserving optimizer counts while matching full batch sizes between arms.

A uses replacement draws of the 200 MRI patches. B reuses the existing weighted replacement sampler: expected 50% leaves / 50% MRI (individual MRI weight = {len(leaf_ids)}/200). Thus A has {counts['pretrain_vessels']*32:,} MRI pretraining presentations; B expects {counts['pretrain_vessels']*16:,} MRI and {counts['pretrain_vessels']*16:,} leaf presentations. Fine-tuning has {counts['vessels_spline']*32:,} MRI presentations in each arm. **Target exposure is not matched.** Exact realized domain totals and unique IDs are recorded from successfully optimized batches, excluding worker prefetch, in each run's `exposure.json` and checkpoints. Sampler RNG and counters survive continuations. Learning curves are `history.jsonl`, `training-losses.jsonl` and offline W&B logs.

## Evaluation and selection

Same existing {len(membership['val'])}-patch subject-disjoint validation monitor every five scheduler units; select maximum common Branch F1, then native edge F1, as before. Evaluate both final and validation-selected checkpoints on all {len(population['validation_ids']):,} validation patches. Independent test ({len(population['test_ids']):,} patches) uses validation-selected checkpoints only, unchanged locked thresholds, saved BatchNorm statistics and existing native/common/core evaluation. No test-driven selection.

Report native node/edge precision, recall, F1, mAP/mAR, common Branch/Curve F1, Betti component/cycle errors, SMD, predicted node/edge counts and duplicate nodes. Duplicate definition: after existing unique node matching at 2 mm, unmatched predictions within 2 mm of an already recovered GT node; reported separately from native box-based matching. Geometry is measured against extracted uncertain guides. Monitor train graph/geometry on exactly the selected 200 MRI patches, without augmentation, so training/validation divergence can be inspected.

## Submission and collection

Pending submission. Full chains are conditional on a GPU forward/backward smoke with actual batch 32. Existing H100 chains are dependencies; the two new arms form two serial lanes. Checkpoints, predictions and large outputs: `{plan['output_root']}`.

Collect without rerunning inference:

```bash
python3 scripts/limited_botanical_transfer.py collect
```

The dependent collector appends validation/test tables, actual exposure totals, selected checkpoints and a cautious single-seed interpretation here. Machine-readable outputs are under `experiments/limited_botanical_transfer/results/`.

## Selected SyntheticMRI training patch IDs

```text
{chr(10).join(selected)}
```
''')
    print(json.dumps({k:metadata[k] for k in ['original_training_patches','selected_percentage','leaf_training_count','reference_steps']},indent=2))


def collect():
    plan = json.loads((HERE/'plan.yaml').read_text())
    evaluation = json.loads((HERE/'evaluation-plan.json').read_text())
    root = Path(plan['output_root'])
    rows = []; exposures = {}
    for label, run in evaluation['runs'].items():
        run = Path(run)
        fine = json.loads((run/'exposure.json').read_text())
        pre = root/'runs'/('pretrain_vessels' if label=='vessel_only' else 'pretrain_mixed')
        before = json.loads((pre/'exposure.json').read_text())
        pre_summary = json.loads((pre/'summary.json').read_text())
        exposures[label] = dict(pretraining={k:v for k,v in before.items() if k!='encountered_ids'},
            finetuning={k:v for k,v in fine.items() if k!='encountered_ids'},
            transferred_pretraining_best_epoch=pre_summary['best_epoch'],
            transferred_pretraining_best_update=pre_summary['best_epoch']*1563,
            total_synthetic_mri_presentations=before['presentations']['synthetic_mri']+fine['presentations']['synthetic_mri'])
        assert before['optimizer_updates']==78150 and fine['optimizer_updates']==31300
        for split in ('validation','test'):
            if split=='validation':
                folders=list((run/'best-full-validation').glob('validation-*')); assert len(folders)==1
                folder=folders[0]
                checkpoint=json.loads((run/'summary.json').read_text())
                selected_epoch=checkpoint['best_epoch'];selected_step=selected_epoch*313
            else:
                summary=json.loads((Path(evaluation['output_root'])/label/'summary.json').read_text())
                folder=Path(summary['metrics_directory']); selected_epoch=summary['epoch'];selected_step=selected_epoch*313
            native=json.loads((folder/'summary.json').read_text())
            common=json.loads((folder/'common-v2-summary.json').read_text())
            duplicates=json.loads((folder/'duplicate-nodes.json').read_text())
            assert native['samples']==(1952 if split=='validation' else 13071)
            row=dict(model=label,split=split,selected_epoch=selected_epoch,selected_update=selected_step)
            keys=('samples','node_precision','node_recall','node_f1','edge_precision','edge_recall','edge_f1',
                  'node_mAP','node_mAR','edge_mAP','edge_mAR','beta0_absolute_error','beta1_absolute_error',
                  'smd','predicted_nodes','predicted_edges','target_nodes','target_edges')
            row.update({k:native[k] for k in keys})
            row.update({k:common[k] for k in ('common_branch_f1','common_curve_f1','common_curve_precision','common_curve_recall')})
            row.update({k:duplicates[k] for k in ('duplicates_per_patch','duplicate_fraction')})
            rows.append(row)
    for stage in ('pretraining','finetuning'):
        assert exposures['vessel_only'][stage]['samples_processed']==exposures['leaves_vessels'][stage]['samples_processed']
    dest=HERE/'results';save(dest/'summary.json',dict(rows=rows,exposures=exposures))
    with (dest/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text='\n## Completed comparison\n\n'
    for split in ('validation','test'):
        text+=f'### {split.capitalize()}\n\n| Model | Node F1 | Edge F1 | Branch F1 | Curve F1 | Component MAE | Cycle MAE | SMD | Duplicate fraction |\n|---|---|---|---|---|---|---|---|---|\n'
        selected=[r for r in rows if r['split']==split]
        for r in selected:
            text+='| '+r['model']+' | '+' | '.join(f"{r[k]:.4f}" for k in ('node_f1','edge_f1','common_branch_f1','common_curve_f1','beta0_absolute_error','beta1_absolute_error','smd','duplicate_fraction'))+' |\n'
        a=next(r for r in selected if r['model']=='vessel_only');b=next(r for r in selected if r['model']=='leaves_vessels')
        text+=f"\nMixed minus vessel-only: edge F1 {b['edge_f1']-a['edge_f1']:+.4f}, Branch F1 {b['common_branch_f1']-a['common_branch_f1']:+.4f}, Curve F1 {b['common_curve_f1']-a['common_curve_f1']:+.4f}. Connectivity and geometry must be interpreted together with component/cycle errors and counts.\n"
        connectivity='improved' if all(b[k]>a[k] for k in ('edge_f1','common_branch_f1')) else 'did not consistently improve'
        geometry='improved' if b['common_curve_f1']>a['common_curve_f1'] and b['smd']<a['smd'] else 'did not consistently improve'
        text+=f'\nIn this seed, leaf pretraining {connectivity} connectivity by edge/Branch F1 and {geometry} geometry by Curve F1/SMD.\n'
    text+='\nThis single seed measures the two recipes under equal optimization and example-processing budgets, with unequal vascular exposure. Differences do not establish statistical superiority or distinguish generalization from optimization by themselves; compare the saved train/validation trajectories. The earlier full-data MRI-only advantage concerns a different annotation budget.\n\n### Actual presentations and unique coverage\n\n```json\n'+json.dumps(exposures,indent=2)+'\n```\n'
    text+='\n### Validation-selected checkpoints\n\n'+ '\n'.join(f"- {r['model']}: epoch/scheduler unit {r['selected_epoch']}, fine-tuning update {r['selected_update']}." for r in rows if r['split']=='validation')+'\n'
    text+='\n### Final training versus validation monitor\n\n'
    for label,run in evaluation['runs'].items():
        history=[json.loads(line) for line in (Path(run)/'history.jsonl').read_text().splitlines()]
        last=history[-1]
        assert last['train'] is not None
        text+=f"- {label}, final update {last['step']}: train/validation edge F1 {last['train']['native_edge_f1']:.4f}/{last['validation']['native_edge_f1']:.4f}, Branch F1 {last['train']['common_branch_f1']:.4f}/{last['validation']['common_branch_f1']:.4f}. These use 200 training and 128 monitored validation patches, rather than the full-validation/test populations above.\n"
    body=REPORT.read_text().split('\n## Completed comparison')[0]
    REPORT.write_text(body+text);(dest/'comparison.md').write_text(text.lstrip())
    print(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','collect'])
    parser.add_argument('--domain-policy',choices=['none','mixed_only'])
    args=parser.parse_args()
    if args.mode=='prepare':
        assert args.domain_policy, 'Choose and record the domain objective before preparing the comparison'
        prepare(args.domain_policy)
    else:collect()
