"""Submit smoke first, then explicitly launch the verified 300-epoch pair."""
import argparse
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.brava_long_experiment import OUT,read,write,sha
COORD=ROOT.parent/'experiment_gpu_coordination'


def submit(options,script,*args):
    value=subprocess.check_output(['sbatch','--parsable','--kill-on-invalid-dep=yes',*options,script,*map(str,args)],text=True,cwd=ROOT).strip().split(';')[0]
    assert value.isdigit(),value
    return value


def gpu_queue():
    value=subprocess.check_output(['squeue','-h','-u','upz25mj','-o','%i|%T|%b|%j'],text=True)
    ids=[]
    for line in value.splitlines():
        fields=line.split('|')
        if 'gpu:' in fields[2]:ids.append(fields[0].split('_')[0])
    return sorted(set(ids)),value


def freeze():
    names=set(read(OUT/'inherited-source-sha256.json'))
    names.update(['scripts/brava_long_experiment.py','scripts/brava_long_evaluation.py',
                  'scripts/collect_brava_long.py','cluster/jean_zay/submit_brava_long.py',
                  'cluster/jean_zay/brava_long_cpu.slurm','cluster/jean_zay/brava_long_gpu.slurm',
                  'tests/test_brava_long.py'])
    hashes={name:sha(ROOT/name) for name in sorted(names)}
    write(OUT/'frozen-source-sha256.json',hashes)
    with zipfile.ZipFile(OUT/'frozen-source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:archive.write(ROOT/name,name)
    return sha(OUT/'frozen-source-sha256.json')


def main(mode,preflight,recover_smoke=False):
    claim=COORD/'brava-long-20261027.json'
    with (COORD/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        record=read(OUT/'submission.json') if (OUT/'submission.json').exists() else {'preflight':preflight}
        if mode=='smoke':
            assert read(OUT/'cpu-preflight.json')['passed']
            if 'smoke' in record:
                assert recover_smoke and not record.get('training_arrays'), 'Never duplicate active/executed long training'
                previous=record['smoke']
                status=subprocess.check_output(['sacct','-j',previous,'-X','-n','-P','-o','State,ExitCode'],text=True).strip()
                assert status.split('|')[0].split()[0] in ['CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY'],status
                write(OUT/f'submission-smoke-{previous}.json',record)
                for name in ['frozen-source.zip','frozen-source-sha256.json']:
                    old=OUT/name;shutil.copy2(old,old.with_name(old.stem+'-'+previous+old.suffix))
                record.setdefault('previous_smokes',[]).append(dict(job=previous,status=status,
                    reason='Shorten multiprocessing temporary socket path; correct aggregation of one-off gradient diagnostics. No long-training updates or scientific settings changed.'))
            source=freeze()
            ids,queue=gpu_queue()
            options=['--time=00:25:00','--job-name=brava-long-smoke','--qos=qos_gpu_h100-dev']
            if ids:options+=['--dependency=afterany:'+':'.join(ids)]
            job=submit(options,'cluster/jean_zay/brava_long_gpu.slurm','smoke')
            record.update(smoke=job,smoke_prior_gpu_jobs=ids,smoke_queue=queue,frozen_source_manifest_sha256=source)
            state=read(claim);state.update(state='CPU checks passed; one H100 smoke/benchmark submitted',concurrent_h100=1,release_after=job,jobs=record)
            write(claim,state)
        else:
            assert read(OUT/'cpu-preflight.json')['passed']
            selection=read(OUT/'gpu-smoke.json');assert selection['passed']
            assert read(OUT/'evaluation-smoke.json')['passed']
            assert not record.get('training_arrays'),'Long comparison already submitted'
            accounting=subprocess.check_output(['sacct','-j',record['smoke'],'-X','-n','-P','-o','State,ExitCode'],text=True).strip()
            assert accounting.startswith('COMPLETED|0:0'),accounting
            assert sha(OUT/'frozen-source-sha256.json')==record['frozen_source_manifest_sha256']
            for name,value in read(OUT/'frozen-source-sha256.json').items():
                assert sha(ROOT/name)==value, f'Code changed after GPU gate: {name}'
            ids,queue=gpu_queue()
            # Every chunk is 25 full epochs. Dependencies form a barrier across
            # the pair; never more than two of this study's H100s can run.
            seconds=8768*25/selection['slowest_patches_per_second']*1.5+1200
            minutes=max(60,math.ceil(seconds/600)*10)
            assert minutes<=20*60,'Measured 25-epoch chunk exceeds allowed 20h walltime'
            wall=f'{minutes//60:02d}:{minutes%60:02d}:00'
            record.update(training_arrays=[],arrays_by_end_epoch={},collectors=[],collector_epochs={},
                          global_h100_limit=2,training_prior_gpu_jobs=ids,training_queue=queue,
                          chunk_walltime=wall,selected_batch=selection['batch_size'])
            previous=None
            for epoch in range(25,301,25):
                options=['--array=0-1%2','--time='+wall,'--job-name=fbrava-'+str(epoch)]
                if previous:options+=['--dependency=afterok:'+previous]
                elif ids:options+=['--dependency=afterany:'+':'.join(ids)]
                array=submit(options,'cluster/jean_zay/brava_long_gpu.slurm','train',epoch)
                record['training_arrays'].append(array);record['arrays_by_end_epoch'][str(epoch)]=array
                write(OUT/'submission.json',record)
                if epoch in [25,50,100,150,200,300]:
                    collector=submit(['--dependency=afterok:'+array,'--job-name=brava-long-report-'+str(epoch)],
                        'cluster/jean_zay/brava_long_cpu.slurm','collect',epoch)
                    record['collectors'].append(collector);record['collector_epochs'][str(epoch)]=collector
                    write(OUT/'submission.json',record)
                previous=array
            state=read(claim);state.update(state='300-epoch scratch/pretrained pair submitted after passed CPU/GPU/evaluation gates',
                concurrent_h100=2,release_after=previous,jobs=record)
            write(claim,state)
        write(OUT/'submission.json',record)
        print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','train']);parser.add_argument('--preflight',default='1636355')
    parser.add_argument('--recover-smoke',action='store_true')
    args=parser.parse_args();main(args.mode,args.preflight,args.recover_smoke)
