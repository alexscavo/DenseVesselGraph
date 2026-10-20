#!/usr/bin/env python3
"""Submit one paired fixed-step comparison after existing H100 work drains."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path('experiments/limited_botanical_transfer')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--cpu-check-job',required=True)
    args=parser.parse_args()
    assert args.cpu_check_job.isdigit()
    record=HERE/'submission.json'
    assert not record.exists(), 'Already submitted: preserve provenance and avoid duplicate runs'
    plan=json.loads((HERE/'plan.yaml').read_text())
    assert plan['domain_policy'] in ('none','mixed_only')
    rows=subprocess.check_output(['squeue','-u',getpass.getuser(),'-h','-o','%A|%P|%f|%b'],text=True)
    previous=sorted({row.split('|')[0].strip() for row in rows.splitlines()
                     if 'gpu_p6' in row or 'h100' in row.lower()},key=int)
    state=dict(plan=str(HERE/'plan.yaml'),existing_h100_jobs=previous,
        cpu_correctness_job=args.cpu_check_job,
        concurrency='Wait for every currently active/pending H100 job; smoke uses one GPU, subsequent arrays use at most two. Whole-array stage barriers prevent overlap.',
        jobs={},source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path('scripts/train_plant_generalization.py'),Path('scripts/limited_data_support.py'),
             Path('scripts/limited_botanical_transfer.py'),HERE/'plan.yaml',HERE/'population-manifest.json']})
    def submit(name,script,arguments,*,time,array=None,dependency=None,exports=None,cpu=False):
        command=['sbatch','--parsable','--kill-on-invalid-dep=yes','--job-name=limited-'+name,'--time='+time]
        if array:command+=['--array='+array]
        if dependency:command+=['--dependency='+dependency]
        if exports:command+=['--export=ALL,'+','.join(k+'='+v for k,v in exports.items())]
        command += [script]+arguments
        job=subprocess.check_output(command,text=True).strip().split(';')[0] if args.submit else str(9900000+len(state['jobs']))
        assert job.isdigit()
        state['jobs'][name]=dict(id=job,dependency=dependency,command=command)
        if args.submit:record.write_text(json.dumps(state,indent=2)+'\n')
        print(name,job,dependency or 'none',flush=True)
        return job
    training='cluster/jean_zay/plant_generalization.slurm'
    exports=dict(GENERALIZATION_CONFIG=str(HERE/'plan.yaml'),LIMITED_DATA_VERIFY='1')
    smoke=submit('smoke',training,['verify'],time='00:30:00',
        dependency='afterok:'+args.cpu_check_job+(',afterany:'+':'.join(previous) if previous else ''),exports=exports)
    pre25=submit('pre25',training,['pretrain','25'],time='12:00:00',array='0-1%2',dependency='afterok:'+smoke,exports=exports)
    pre50=submit('pre50',training,['pretrain','50'],time='12:00:00',array='0-1%2',dependency='afterok:'+pre25,exports=exports)
    fine50=submit('fine50',training,['finetune','50'],time='06:00:00',array='1,3%2',dependency='afterok:'+pre50,exports=exports)
    fine100=submit('fine100',training,['finetune','100'],time='06:00:00',array='1,3%2',dependency='afterok:'+fine50,exports=exports)
    test=submit('test','cluster/jean_zay/combined_transfer_test.slurm',['test'],time='03:00:00',array='0-1%2',dependency='afterok:'+fine100,
        exports=dict(EVALUATION_PLAN=str(HERE/'evaluation-plan.json'),EVALUATION_ARMS='vessel_only:leaves_vessels',LIMITED_DATA_VERIFY='1'))
    submit('collect','cluster/jean_zay/collect_limited_botanical.slurm',[],time='00:10:00',dependency='afterok:'+test)
    if args.submit:
        report=Path('docs/research/pretraining/limited-data-botanical-transfer.md')
        text='\n## Accepted jobs\n\n| Stage | Job ID | Dependency |\n|---|---|---|\n'
        text+=f'| CPU real-data forward/backward check | {args.cpu_check_job} | none |\n'
        for name,job in state['jobs'].items():text+=f"| {name} | {job['id']} | {job['dependency'] or 'none'} |\n"
        text+='\nH100 jobs already in queue at submission: '+', '.join(previous)+'. The shared smoke waits for all of them; the paired stage arrays use at most two H100s. Full training cannot execute if smoke fails.\n'
        report.write_text(report.read_text()+text)


if __name__=='__main__':main()
