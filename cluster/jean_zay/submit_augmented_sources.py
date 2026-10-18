#!/usr/bin/env python3
"""Three augmented source mixes, paired fine-tunings, tests, and matched controls."""
import argparse
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--submit',action='store_true')
a=p.parse_args()
directory=Path('experiments/source_augmentation/training_20261018')
directory.mkdir(exist_ok=True)
record=directory/'submission.json'
assert not record.exists(), 'Submission already recorded; inspect before resubmitting'
sources=dict(plants='plant',leaves='leaf',combined='combined')
baseline=json.loads(Path('experiments/combined_pretraining/evaluation-plan.json').read_text())
plan=dict(baseline)
plan['runs']=dict(baseline['runs'])
plan['existing_results']={arm:str(Path(baseline['output_root'])/arm) for arm in baseline['runs']}
plan['output_root']='/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/augmented_source_tests_20261018'
plan['report_root']='docs/research/pretraining/augmented-transfer-results'
configs={}
for source,stem in sources.items():
    config=f'configs/{stem}_generalization_da_b32_source_aug.yaml'
    # These fixed reviewed plans use plain top-level scalar paths.
    values=dict(line.split(': ',1) for line in Path(config).read_text().splitlines()
                if ': ' in line and not line.startswith((' ','#')))
    assert (Path(values['plant_root'])/'READY').is_file()
    assert (Path(values['plant_root']).parent/'plants-smoke/READY').is_file()
    assert not (Path(values['output_root'])/'runs').exists(), 'Never reuse previous output'
    assert values['population_manifest']==baseline['population_manifest']
    assert values['pretrain_epochs']=='50' and values['finetune_epochs']=='100'
    assert values['batch_size']=='32' and values['early_stopping']=='false'
    configs[source]=config
    for rep in ('straight','spline'):
        plan['runs'][source+'_aug_'+rep]=str(Path(values['output_root'])/'runs'/('mixed_'+rep))
for arm,path in plan['existing_results'].items():
    result=json.loads((Path(path)/'summary.json').read_text())
    assert result['sample_ids_sha256']==plan['test_ids_sha256']
    assert (Path(plan['runs'][arm])/'COMPLETE').is_file()
evaluation=directory/'evaluation-plan.json'
evaluation.write_text(json.dumps(plan,indent=2)+'\n')
state=dict(status='prepared',jobs={},configs=configs,evaluation_plan=str(evaluation),
           unchanged_controls=list(baseline['runs']),seed=baseline['seed'],
           comparison='Original versus nonplanar/textured-soft biological source; same target data and stage policy')
if not a.submit:
    print('Preflight passed: 3 fresh pretrainings, 6 fine-tunings/tests, 8 existing controls.')
    raise SystemExit(0)
record.write_text(json.dumps(state,indent=2)+'\n')

def submit(name,args):
    command=['sbatch','--parsable','--kill-on-invalid-dep=yes',*args]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    state['jobs'][name]=dict(id=job,command=command)
    record.write_text(json.dumps(state,indent=2)+'\n')
    print(name,job,flush=True)
    return job

script='cluster/jean_zay/plant_generalization.slurm'
tests=[]
for source,config in configs.items():
    export=f'--export=ALL,GENERALIZATION_CONFIG={config},SOURCE_AUGMENTATION_CHECKS=1'
    verify=submit(source+'_verify',[export,'--job-name=aug-'+source+'-verify','--time=01:00:00',script,'verify'])
    p25=submit(source+'_pretrain25',[export,'--job-name=aug-'+source+'-pretrain','--time=12:00:00','--array=0',f'--dependency=afterok:{verify}',script,'pretrain','25'])
    p50=submit(source+'_pretrain50',[export,'--job-name=aug-'+source+'-pretrain','--time=12:00:00','--array=0',f'--dependency=aftercorr:{p25}',script,'pretrain','50'])
    f50=submit(source+'_finetune50',[export,'--job-name=aug-'+source+'-finetune','--time=06:00:00','--array=0-1',f'--dependency=afterok:{p50}',script,'finetune','50'])
    f100=submit(source+'_finetune100',[export,'--job-name=aug-'+source+'-finetune','--time=06:00:00','--array=0-1',f'--dependency=aftercorr:{f50}',script,'finetune','100'])
    test_export=f'--export=ALL,EVALUATION_PLAN={evaluation},EVALUATION_ARMS={source}_aug_straight:{source}_aug_spline'
    tests.append(submit(source+'_test',[test_export,'--job-name=aug-'+source+'-test','--time=03:00:00','--array=0-1',f'--dependency=aftercorr:{f100}','cluster/jean_zay/combined_transfer_test.slurm','test']))
submit('comparison',[f'--export=ALL,EVALUATION_PLAN={evaluation},COLLECT_VALIDATION=1',
    '--dependency=afterok:'+':'.join(tests),'cluster/jean_zay/collect_combined_transfer_test.slurm'])
state['status']='queued'
record.write_text(json.dumps(state,indent=2)+'\n')
