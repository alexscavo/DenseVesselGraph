"""One bounded GPU check, serialized after existing work under the common lock."""
import fcntl
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/native2d_20261027'
COORD=ROOT.parent/'experiment_gpu_coordination'


def command(argv):
    return subprocess.check_output(argv,text=True,cwd=ROOT,timeout=45).strip()


def submit(args):
    job=command(['sbatch','--parsable',*args]).split(';')[0]
    if not job.isdigit():raise RuntimeError(job)
    return job


def main():
    checks=json.loads((OUT/'cpu-checks.json').read_text())
    assert checks['passed']
    assert (OUT/'dataset-audit.json').exists() and (OUT/'manifest.json').exists()
    if (OUT/'submission.json').exists():raise RuntimeError('Submission already recorded; do not duplicate smoke')
    files=sorted(Path('native2d').glob('*.py'))+[Path(p) for p in [
        'scripts/check_native2d.py','scripts/collect_native2d.py','scripts/prepare_native2d_smoke.py',
        'scripts/show_native2d.py','tests/test_native2d.py','configs/native2d.yaml','configs/base.yaml',
        'models/__init__.py','models/matcher.py','models/spline.py','training/losses/criterion.py',
        'training/losses/focal.py','metrics/common_vessel_metrics.py','metrics/representation_geometry.py',
        'metrics/branch_connectivity.py','training/evaluation/metrics.py',
        'cluster/jean_zay/native2d_smoke_gpu.slurm','cluster/jean_zay/native2d_collect_cpu.slurm',
        'cluster/jean_zay/submit_native2d_smoke.py']]
    with zipfile.ZipFile(OUT/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(ROOT/path,str(path))
    (OUT/'code-identities.json').write_text(json.dumps({str(f):hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
    with (COORD/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=command(['squeue','-h','-u','upz25mj','-o','%i|%T|%b|%j'])
        gpu=[]
        for row in queue.splitlines():
            fields=row.split('|')
            if 'native2d-' in fields[-1]:raise RuntimeError('Existing native2d job; no duplicate submission')
            if 'gpu:' in fields[2]:gpu.append(fields[0].split('_')[0])
        # Include terminal jobs of already reserved multi-stage GPU chains.
        for claim in COORD.glob('*.json'):
            data=json.loads(claim.read_text())
            terminal=str(data.get('release_after',''))
            if terminal.isdigit() and any(terminal==x for x in gpu):gpu.append(terminal)
        prior=sorted(set(gpu),key=int)
        options=['--kill-on-invalid-dep=yes']
        dependency='afterany:'+':'.join(prior) if prior else None
        if dependency:options+=['--dependency='+dependency]
        job=submit([*options,'cluster/jean_zay/native2d_smoke_gpu.slurm'])
        record=dict(gpu=job,gpu_dependency=dependency,prior_gpu_jobs=prior,queue_at_submission=queue,
            maximum_simultaneous_h100=1,global_h100_limit=2,walltime='00:20:00',
            tiny_updates=1000,tiny_training_crops=2,validation_development_crops=2,
            full_architecture_optimizer_updates=0,full_training_submitted=False)
        (OUT/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        collector=submit(['--dependency=afterany:'+job,'--kill-on-invalid-dep=yes','cluster/jean_zay/native2d_collect_cpu.slurm'])
        record.update(collector=collector,collector_dependency='afterany:'+job)
        (OUT/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        (COORD/'native2d-20261027.json').write_text(json.dumps(dict(owner='native 2D correctness port',worktree=str(ROOT),
            output=str(OUT),state='bounded smoke queued after existing GPU chains',concurrent_h100=0,
            reserved_after_dependency=1,global_limit=2,release_after=job,jobs=record),indent=2)+'\n')
        for submitted in [job,collector]:command(['scontrol','show','job',submitted])
        print(json.dumps(record,indent=2))


if __name__=='__main__':main()
