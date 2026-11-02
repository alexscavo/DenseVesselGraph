"""Prepare/submit only BRAVA Stage 1; Stage 2 remains a documented design."""
import argparse
from datetime import datetime
import getpass
import hashlib
import json
from pathlib import Path
import subprocess


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def spread(ids,count):
    groups={}
    for name in ids:groups.setdefault(name.rsplit('_',1)[0],[]).append(name)
    result=[]
    for i in range(max(map(len,groups.values()))):
        for names in groups.values():
            if i<len(names):result.append(names[i])
            if len(result)==count:return result
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true');a=parser.parse_args()
    directory=Path('experiments/brava_graph_study');directory.mkdir(exist_ok=True);record=directory/'submission.json';planfile=directory/'plan.json'
    assert not record.exists(), 'Existing BRAVA Stage 1 submission; do not duplicate it'
    if planfile.exists():plan=json.loads(planfile.read_text())
    else:
        dataset=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057/transfer_splines_k8')
        manifest=json.loads((dataset/'manifest.json').read_text())
        sources=json.loads((dataset.parent/'sources/source_manifest.json').read_text())['subjects']
        subjects={r['patient_id']:r['subject'] for r in sources}
        pilot=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_query_diagnostics_20261019/pilot-1291329-local_image/manifest.json')
        checkpoint_manifest=json.loads(pilot.read_text())
        reference=json.loads(Path('experiments/combined_pretraining/evaluation-plan.json').read_text())
        paths=dict(synthetic_spline=str(Path(reference['runs']['mri_only_spline'])/'best.pt'),
            synthetic_straight=str(Path(reference['runs']['mri_only_straight'])/'best.pt'),ixi_spline=checkpoint_manifest['checkpoint'])
        ids=dict(train=manifest['train_ids'],val=manifest['validation_ids'],test=manifest['test_ids'])
        plan=dict(dataset=str(dataset),manifest_sha256=digest(dataset/'manifest.json'),ids=ids,
            subjects_by_id={name:subjects[name.split('_')[1]] for names in ids.values() for name in names},
            train_monitor_ids=spread(ids['train'],96),smoke_ids=dict(train=spread(ids['train'],4),val=spread(ids['val'],8)),
            training_config=checkpoint_manifest['config'],configuration_source=str(pilot),configuration_source_sha256=digest(pilot),
            checkpoints={name:dict(path=p,sha256=digest(p)) for name,p in paths.items()},
            output_root='/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/brava_graph_study_'+datetime.now().strftime('%Y%m%d-%H%M%S'),
            seed=211039,training_initializer='fresh random; no source pretraining',
            arms=['standard','coordinate_refinement'],stage2_training_submitted=False)
        planfile.write_text(json.dumps(plan,indent=2)+'\n')
    if not a.submit:print(planfile);print(plan['output_root']);return
    existing=subprocess.check_output(['squeue','-u',getpass.getuser(),'-h','-o','%A|%P|%f|%b'],text=True)
    gpu=sorted({r.split('|')[0].strip() for r in existing.splitlines() if 'h100' in r.lower() or 'gpu_p6' in r},key=int)
    root=plan['output_root'];state=dict(output_root=root,jobs={},existing_gpu_jobs=gpu,plan=str(planfile),stage2_submitted=False)
    files=['scripts/brava_graph_study.py','scripts/audit_brava_study.py','scripts/evaluate_brava_study.py','scripts/collect_brava_study.py',
        'models/node_refinement.py','models/relationformer.py','cluster/jean_zay/brava_study_gpu.slurm','cluster/jean_zay/brava_study_cpu.slurm',
        'tests/test_brava_graph_study.py','experiments/brava_graph_study/checkpoint-inventory.json','experiments/brava_graph_study/stage2-plan.json']
    state['source_sha256']={p:digest(p) for p in files}
    def submit(name,kind,mode,dependency=None,time='02:00:00',array=None):
        command=['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+time]
        if dependency:command+=['--dependency='+dependency]
        if array:command+=['--array='+array]
        command+=['cluster/jean_zay/brava_study_'+kind+'.slurm',mode,root]
        job=subprocess.check_output(command,text=True).strip().split(';')[0];assert job.isdigit()
        state['jobs'][name]=dict(id=job,dependency=dependency,walltime=time,array=array,command=command)
        record.write_text(json.dumps(state,indent=2)+'\n');print(name,job,dependency,flush=True);return job
    audit=submit('dataset_audit','cpu','audit',time='02:00:00')
    submit('GT_representation_analysis','cpu','representations','afterok:'+audit,time='04:00:00')
    smoke=submit('gpu_smoke','gpu','smoke','afterok:'+audit+(',afterany:'+':'.join(gpu) if gpu else ''),time='00:45:00')
    tiny=submit('cpu_smoke','cpu','smoke','afterok:'+smoke,time='00:45:00')
    benchmark=submit('throughput_benchmark','gpu','benchmark','afterok:'+tiny,time='00:30:00')
    zero=submit('zero_shot_prediction','gpu','zero','afterok:'+benchmark,time='01:30:00',array='0-2%2')
    ze=submit('zero_shot_evaluation','cpu','zero','aftercorr:'+zero,time='04:00:00',array='0-2%3')
    train=submit('training','gpu','train','afterok:'+zero,time='06:00:00',array='0-1%2')
    pred=submit('trained_prediction','gpu','predict','afterok:'+train,time='03:00:00',array='0-1%2')
    ev=submit('trained_evaluation','cpu','evaluate','aftercorr:'+pred,time='08:00:00',array='0-1%2')
    submit('collection','cpu','collect',f'afterok:{ze}:{ev}:'+state['jobs']['GT_representation_analysis']['id'],time='02:00:00')
    state['submitted']=True;record.write_text(json.dumps(state,indent=2)+'\n')
    with Path('docs/research/BRAVA_experiment_plan.md').open('a') as f:
        f.write('\n## Accepted Stage 1 jobs\n\nOutput: `'+root+'`.\n\n|Stage|Job|Dependency|Wall time|\n|---|---|---|---|\n')
        for name,j in state['jobs'].items():f.write(f'|{name}|{j["id"]}|{j["dependency"]}|{j["walltime"]}|\n')
        f.write('\nGPU stages wait for all existing H100 jobs: '+(', '.join(gpu) or 'none')+'. No Stage 2 training was submitted.\n')
    print('Retrieve: python3 scripts/collect_brava_study.py --show')


if __name__=='__main__':main()
