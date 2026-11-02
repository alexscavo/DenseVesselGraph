"""Freeze a controlled study and serialize all jobs into at most two GPU lanes."""
import fcntl
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROJECTS=ROOT.parent.parent if ROOT.parent.name=='GNBM' else ROOT.parent
OUT=ROOT/'experiments/native2d_mixed_20261028'
COORD=PROJECTS/'experiment_gpu_coordination'

def read(p):return json.loads(p.read_text())
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def command(args):return subprocess.check_output(args,text=True,cwd=ROOT,timeout=45).strip()
def submit(args):
    value=command(['sbatch','--parsable',*args]).split(';')[0];assert value.isdigit();return value

def main():
    phase=sys.argv[1];assert phase in ['gates','train']
    plan=read(OUT/'plan.json');assert {'avwide','reaver','octasynth','plants'}<=set(plan['sources'])
    data=Path(plan['data_root'])
    assert all(a['passed'] for a in plan['sources'].values())
    assert '\nOK\n' in (OUT/'cpu-tests-v2.log').read_text()
    assert read(OUT/'cpu-checks.json')['passed']
    assert read(OUT/'source-topology-audit.json')['passed']
    assert read(OUT/'mixed-cpu-checks.json')['passed']
    p=OUT/'submission.json';record=read(p) if p.exists() else dict(global_h100_limit=2,runs={},collectors={})
    with (COORD/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=command(['squeue','-h','-u','upz25mj','-o','%i|%T|%b|%j'])
        ours=set(record['runs'].values())|{record.get('gates','')}
        prior=sorted({r.split('|')[0].split('_')[0] for r in queue.splitlines() if 'gpu:' in r.split('|')[2] and r.split('|')[0].split('_')[0] not in ours},key=int)
        record['queue_at_'+phase]=queue
        if phase=='gates':
            assert 'gates' not in record
            files=set([*ROOT.glob('native2d/*.py'),*ROOT.glob('training/losses/*.py'),*ROOT.glob('metrics/*.py'),*ROOT.glob('models/**/*.py'),
                *ROOT.glob('configs/*.py'),ROOT/'configs/base.yaml',ROOT/'configs/native2d.yaml',ROOT/'training/evaluation/metrics.py',
                *[ROOT/('scripts/'+n+'.py') for n in ['train_native2d_study','train_native2d_long','prepare_native2d_long','audit_native2d_long','collect_native2d_long']],
                ROOT/'tests/test_native2d_mixed.py',
                ROOT/'tests/test_native2d.py',ROOT/'tests/test_native2d_fitting.py',Path(__file__).resolve(),
                *ROOT.glob('cluster/jean_zay/native2d_long*.slurm'),*OUT.glob('planned-configs/*.yaml')])
            with zipfile.ZipFile(OUT/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for f in sorted(files):z.write(f,str(f.relative_to(ROOT)))
            write(OUT/'code-identities.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)})
            write(OUT/'data-identities.json',{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [OUT/'plan.json',OUT/'source-topology-audit.json',*data.glob('*/manifest.json'),*data.glob('*/audit.json'),*data.glob('*/targets.zip')]})
            options=['--dependency=afterany:'+':'.join(prior)] if prior else []
            record['gates']=submit([*options,'--job-name=n2d-mixed-gates','--time=00:25:00','cluster/jean_zay/native2d_long_gpu.slurm','--gates'])
            write(p,record)
            record['train_launcher']=submit(['--dependency=afterok:'+record['gates'],'--kill-on-invalid-dep=yes','cluster/jean_zay/native2d_long_stage.slurm','train'])
            record['gate_collector']=submit(['--dependency=afterany:'+record['gates'],'--kill-on-invalid-dep=yes','cluster/jean_zay/native2d_long_collect.slurm'])
        else:
            gate=read(OUT/'gates.json');assert gate['passed']
            for filename,digest in read(OUT/'code-identities.json').items():assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest,filename
            for filename,digest in read(OUT/'data-identities.json').items():assert hashlib.sha256((OUT/filename).read_bytes()).hexdigest()==digest,filename
            # Walltime includes full held-out evaluation/checkpoint IO, not just updates.
            assert all(b['estimated_training_hours']<10 for b in gate['benchmark'].values()),'Declared 14h walltime needs a reviewed budget'
            # Obtain AV scratch results early while AV mixed pretraining runs.
            # The final OCTA pair can use both slots after its shared pretrain.
            lanes=[['avwide_scratch_B0','avwide_scratch_Axyz','reaver_scratch_B0','reaver_scratch_Axyz',
                    'reaver_mixedpretrain_50','reaver_mixed_B0','reaver_mixed_Axyz',
                    'octasynth_mixedpretrain_50','octasynth_mixed_B0'],
                   ['avwide_mixedpretrain_50','avwide_mixed_B0','avwide_mixed_Axyz',
                    'octasynth_scratch_B0','octasynth_scratch_Axyz','octasynth_mixed_Axyz']]
            record['lanes']=lanes
            for lane in lanes:
                previous=None
                for name in lane:
                    if name in record['runs']:previous=record['runs'][name];continue
                    if '_mixedpretrain_' in name:
                        dataset=name.split('_')[0];initial='scratch';variant='B0';stage_options=['--pretrain-mixed']
                    else:dataset,initial,variant=name.split('_');stage_options=[]
                    required=[previous or record['gates']]
                    if initial=='mixed':required.append(record['runs'][dataset+'_mixedpretrain_50'])
                    dependency='afterok:'+':'.join(dict.fromkeys(required))
                    if not previous and prior:dependency+=',afterany:'+':'.join(prior)
                    job=submit(['--dependency='+dependency,'--kill-on-invalid-dep=yes','--job-name=n2d-'+name,
                        'cluster/jean_zay/native2d_long_gpu.slurm','--dataset',dataset,'--initialization',initial,'--variant',variant,*stage_options])
                    record['runs'][name]=job;previous=job;write(p,record)
                    record['collectors'][name]=submit(['--dependency=afterany:'+job,'--kill-on-invalid-dep=yes','cluster/jean_zay/native2d_long_collect.slurm']);write(p,record)
            record['full_training_submitted']=True
        write(p,record)
        write(COORD/'native2d-mixed-20261028.json',dict(owner='native2D fixed50 mixed plant/vessel and fixed100 vessel study',worktree=str(ROOT),output=str(OUT),state=phase+' submitted',
            concurrent_h100=1 if phase=='gates' else 2,reserved_after_dependency=1 if phase=='gates' else 2,global_limit=2,
            terminal_gpu_jobs=[record['runs'][lane[-1]] for lane in record['lanes']] if record.get('lanes') else [record['gates']],jobs=record))
        print(json.dumps(record,indent=2))

if __name__=='__main__':main()
