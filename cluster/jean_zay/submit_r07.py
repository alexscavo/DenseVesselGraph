"""Submit one gated R07 batch under the shared global GPU coordination lock."""
import json,fcntl,subprocess,hashlib,zipfile,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/r07_matched_supervision_20261027'
COORD=ROOT.parent/'experiment_gpu_coordination'


def submit(options,*args):
    value=subprocess.check_output(['sbatch','--parsable',*options,*args],text=True,cwd=ROOT).strip().split(';')[0]
    assert value.isdigit(),value
    return value


def main(recover=False):
    assert json.loads((OUT/'cpu-smoke.json').read_text())['passed']
    assert json.loads((OUT/'supervision-audit.json').read_text())['completed']
    assert json.loads((OUT/'checkpoint-loading-smoke.json').read_text())['passed']
    with (COORD/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        previous=None
        if (OUT/'submission.json').exists():
            assert recover,'Do not duplicate submissions; recovery must be explicit'
            previous=json.loads((OUT/'submission.json').read_text())
            accounting=subprocess.check_output(['sacct','-j',','.join(previous[k] for k in ['smoke','training_array','collector']),'-X','-n','-P',
                                                '-o','JobID,State,Elapsed,ExitCode'],text=True)
            rows=[r.split('|') for r in accounting.splitlines() if r.strip()]
            assert len(rows)>=3 and all(r[1].split()[0] in ['FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY'] for r in rows),accounting
            training=[r for r in rows if r[0].startswith(previous['training_array'])]
            assert training and all(r[1].startswith('CANCELLED') and r[2]=='00:00:00' for r in training),'Never duplicate executed training'
            archive=OUT/('submission-failed-'+previous['smoke']+'.json')
            if not archive.exists():archive.write_text(json.dumps(previous,indent=2)+'\n')
            (OUT/'recovery-accounting.psv').write_text(accounting)
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
        if previous:
            record.update(replaces=previous,recovery='Pin frozen model import before audit helper path mutation; both full source checkpoints strictly loaded on CPU; protocol unchanged.')
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

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recover',action='store_true');args=parser.parse_args();main(args.recover)
