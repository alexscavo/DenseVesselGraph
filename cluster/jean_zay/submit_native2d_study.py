"""Submit exactly two controlled arms after passed gates under the global lock."""
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/native2d_20261028'
COORD=ROOT.parent/'experiment_gpu_coordination'

def command(args):
    return subprocess.check_output(args,text=True,cwd=ROOT,timeout=45).strip()

def submit(args):
    value=command(['sbatch','--parsable',*args]).split(';')[0]
    assert value.isdigit(),value
    return value

def main():
    gates=json.loads((OUT/'gates.json').read_text());assert gates['passed']
    assert json.loads((OUT/'cpu-checks.json').read_text())['passed']
    assert json.loads((OUT/'av-wide-audit.json').read_text())['passed']
    record=json.loads((OUT/'submission.json').read_text())
    assert not record.get('full_training_submitted'),'No duplicate submission'
    identities=json.loads((OUT/'code-identities.json').read_text())
    assert hashlib.sha256((ROOT/'scripts/train_native2d_study.py').read_bytes()).hexdigest()==identities['scripts/train_native2d_study.py']
    with (COORD/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=command(['squeue','-h','-u','upz25mj','-o','%i|%T|%b|%j'])
        prior=sorted(set(row.split('|')[0].split('_')[0] for row in queue.splitlines() if 'gpu:' in row.split('|')[2]),key=int)
        # Wait for every preexisting GPU job; this includes already reserved
        # chains, and prevents exceeding two when the gate is still COMPLETING.
        after=sorted(set([*prior,record['gates']]),key=int)
        dependency='afterok:'+record['gates']
        others=[j for j in after if j!=record['gates']]
        if others:dependency+=',afterany:'+':'.join(others)
        record.update(queue_at_submission=queue,prior_gpu_jobs=prior,dependency=dependency,
            full_training_submitted=True,updates=gates['updates'],peak_concurrent_h100=2,global_h100_limit=2)
        # Save each returned job before the next mutation to prevent duplicates
        # if scheduler submission fails partway through.
        for arm in ['B0','Axyz']:
            job=submit(['--dependency='+dependency,'--kill-on-invalid-dep=yes','--job-name=native2d-'+arm,
                'cluster/jean_zay/native2d_study_gpu.slurm',arm])
            record[arm]=job;(OUT/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        collector=submit(['--dependency=afterany:'+record['B0']+':'+record['Axyz'],'--kill-on-invalid-dep=yes',
            'cluster/jean_zay/native2d_study_collect.slurm'])
        record['collector']=collector;(OUT/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        claim=json.loads((COORD/'native2d-20261028.json').read_text())
        claim.update(state='paired original-width 6000-update training submitted',concurrent_h100=2,
            reserved_after_dependency=2,release_after=collector,jobs=record)
        (COORD/'native2d-20261028.json').write_text(json.dumps(claim,indent=2)+'\n')
        print(json.dumps(record,indent=2))

if __name__=='__main__':main()
