"""Submit one gated R07 batch under the shared global GPU coordination lock."""
import json,fcntl,subprocess,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/r07_matched_supervision_20261027'
COORD=ROOT.parent/'experiment_gpu_coordination'


def submit(options,*args):
    value=subprocess.check_output(['sbatch','--parsable',*options,*args],text=True,cwd=ROOT).strip().split(';')[0]
    assert value.isdigit(),value
    return value


def main():
    assert json.loads((OUT/'cpu-smoke.json').read_text())['passed']
    assert json.loads((OUT/'supervision-audit.json').read_text())['completed']
    with (COORD/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not (OUT/'submission.json').exists(),'Do not duplicate submissions'
        queue=subprocess.check_output(['squeue','-h','-u','upz25mj','-o','%i|%T|%b|%j'],text=True)
        gpu=[]
        for line in queue.splitlines():
            fields=line.split('|')
            if 'r07-' in fields[-1]:raise RuntimeError('An R07 job already exists')
            if 'gpu:' in fields[2]:gpu.append(fields[0].split('_')[0])
        gpu=sorted(set(gpu));options=['--time=00:10:00','--kill-on-invalid-dep=yes']
        if gpu:options+=['--dependency=afterany:'+':'.join(gpu)]
        smoke=submit(options,'cluster/jean_zay/r07_gpu.slurm','smoke')
        record=dict(smoke=smoke,prior_gpu_jobs=gpu,smoke_dependency='afterany:'+':'.join(gpu) if gpu else None,
                    queue_at_submission=queue,global_h100_limit=2)
        (OUT/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        train=submit(['--array=0-1%2','--dependency=afterok:'+smoke,'--kill-on-invalid-dep=yes'],
                     'cluster/jean_zay/r07_gpu.slurm','run')
        record.update(training_array=train,training_dependency='afterok:'+smoke,
                      seed_tasks={'0':364505,'1':92741},arms_per_seed=['S0','S1'])
        (OUT/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        collector=submit(['--dependency=afterok:'+train,'--kill-on-invalid-dep=yes'],'cluster/jean_zay/r07_cpu.slurm')
        record.update(collector=collector,collector_dependency='afterok:'+train)
        (OUT/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        claim=COORD/'r07-matched-supervision-20261027.json';c=json.loads(claim.read_text())
        c.update(state='submitted, GPU smoke gates two-seed frozen relation comparison',concurrent_h100=2,release_after=train,jobs=record)
        claim.write_text(json.dumps(c,indent=2)+'\n')
        for job in [smoke,train,collector]:
            subprocess.check_call(['scontrol','show','job',job],stdout=subprocess.DEVNULL)
        print(json.dumps(record,indent=2))

if __name__=='__main__':main()
