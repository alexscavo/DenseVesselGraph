"""One target-fit correction: gated fresh B0/A-xyz, serialized in one GPU slot."""
import fcntl
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/native2d_20261028_targetfix'
COORD=ROOT.parent/'experiment_gpu_coordination'

def command(args):return subprocess.check_output(args,text=True,cwd=ROOT,timeout=45).strip()
def submit(args):
    job=command(['sbatch','--parsable',*args]).split(';')[0];assert job.isdigit();return job

def main():
    phase=sys.argv[1];assert phase in ['gates','train']
    assert json.loads((OUT/'target-fit-audit.json').read_text())['passed']
    assert json.loads((OUT/'cpu-checks.json').read_text())['passed']
    record=json.loads((OUT/'submission.json').read_text()) if (OUT/'submission.json').exists() else dict(phase='target-fit-correction',global_h100_limit=2,native2d_peak_concurrent_h100=1)
    if phase=='gates':assert 'gates' not in record
    else:
        assert json.loads((OUT/'gates.json').read_text())['passed'] and 'B0' not in record
        assert json.loads((OUT/'gates.json').read_text())['updates']==6000
    with (COORD/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=command(['squeue','-h','-u','upz25mj','-o','%i|%T|%b|%j'])
        gpu=[r.split('|') for r in queue.splitlines() if 'gpu:' in r.split('|')[2]]
        # The independently verified IXI chain is single-GPU with array %1.
        # Any new/unknown reservation instead serializes our job behind all of it.
        unknown=[r[0].split('_')[0] for r in gpu if r[0].split('_')[0] not in ['1653858','1653859','1653860',record.get('gates')]]
        prior=sorted(set(r[0].split('_')[0] for r in gpu if r[0].split('_')[0]!=record.get('gates')),key=int) if unknown else []
        record['queue_at_'+phase]=queue
        if phase=='gates':
            files=[*ROOT.glob('native2d/*.py'),*ROOT.glob('training/losses/*.py'),*ROOT.glob('metrics/*.py'),
                ROOT/'models/matcher.py',ROOT/'models/spline.py',ROOT/'training/evaluation/metrics.py',
                ROOT/'configs/base.yaml',ROOT/'configs/native2d.yaml',ROOT/'scripts/train_native2d_study.py',
                ROOT/'scripts/prepare_native2d_fit_revision.py',ROOT/'scripts/collect_native2d_study.py',
                ROOT/'tests/test_native2d_fitting.py',Path(__file__).resolve(),
                ROOT/'cluster/jean_zay/native2d_fit_revision_gpu.slurm',ROOT/'cluster/jean_zay/native2d_fit_revision_collect.slurm']
            with zipfile.ZipFile(OUT/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in files:z.write(p,str(p.relative_to(ROOT)))
            (OUT/'code-identities.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
            options=['--dependency=afterany:'+':'.join(prior)] if prior else []
            record['gates']=submit([*options,'--job-name=native2d-fitfix-gates','--time=00:15:00','cluster/jean_zay/native2d_fit_revision_gpu.slurm','gates'])
        else:
            identities=json.loads((OUT/'code-identities.json').read_text())
            assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in identities.items())
            dependency='afterok:'+record['gates']
            if prior:dependency+=',afterany:'+':'.join(prior)
            record['B0']=submit(['--dependency='+dependency,'--kill-on-invalid-dep=yes','--job-name=native2d-fitfix-B0','cluster/jean_zay/native2d_fit_revision_gpu.slurm','B0'])
            (OUT/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
            record['Axyz']=submit(['--dependency=afterany:'+record['B0'],'--kill-on-invalid-dep=yes','--job-name=native2d-fitfix-Axyz','cluster/jean_zay/native2d_fit_revision_gpu.slurm','Axyz'])
            record['collector']=submit(['--dependency=afterany:'+record['B0']+':'+record['Axyz'],'--kill-on-invalid-dep=yes','cluster/jean_zay/native2d_fit_revision_collect.slurm'])
            record.update(full_training_submitted=True,updates=6000)
        (OUT/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        (COORD/'native2d-targetfit-20261028.json').write_text(json.dumps(dict(owner='native2D single target-fit correction',worktree=str(ROOT),output=str(OUT),state=phase+' submitted',concurrent_h100=1,reserved_after_dependency=1,global_limit=2,release_after=record.get('Axyz',record['gates']),jobs=record),indent=2)+'\n')
        print(json.dumps(record,indent=2))

if __name__=='__main__':main()
