"""Submit one isolated, dependency-gated IXI batch; no polling."""
import fcntl,json,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];META=ROOT/'experiments/ixi_local_components_20261026'
SHARED=ROOT.parent/'experiment_gpu_coordination';SHARED.mkdir(exist_ok=True)
def cmd(*a):return subprocess.check_output(a,cwd=ROOT,text=True).strip()
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
with (SHARED/'submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if '--recover' in sys.argv:
  prior=json.loads((META/'jobs.json').read_text())
  ids=[v['id'] for v in prior['jobs'].values()]
  active=cmd('squeue','-h','-j',','.join(ids),'-o','%i|%T')
  assert not active,'Prior chain has active jobs; preserve them'
  history=META/('jobs.initial-'+prior['jobs']['smoke']['id']+'.json')
  assert not history.exists(),'Recovery already recorded'
  (META/'jobs.json').rename(history)
 assert not (META/'jobs.json').exists(),'Do not duplicate an existing submission'
 # Includes RUNNING and PENDING allocations, across all same-user studies.
 queue=cmd('squeue','-h','-u','upz25mj','-o','%i|%j|%T|%b|%P')
 previous=[]
 for line in queue.splitlines():
  fields=line.split('|')
  if 'gpu' in fields[3].lower() or 'gpu' in fields[4].lower():previous.append(fields[0].split('_')[0])
 previous=sorted(set(previous));jobs=dict(submitted_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),worktree=str(ROOT),branch='research/ixi-local-components-20261026',max_h100=1,shared_global_limit=2,prior_gpu_jobs=previous,queue_at_submission=queue,jobs={})
 claim=SHARED/'ixi-local-components-20261026.json'
 write(claim,dict(owner='IXI local CNN components',state='submitting',worktree=str(ROOT),concurrent_h100=1,global_limit=2,note='Reserve at most one other H100 globally while this chain is active; check all Slurm allocations, not per-agent arrays.'))
 def submit(name,script,*opts,args=()):
  job=cmd('sbatch','--parsable',*opts,str(ROOT/script),*args).split(';')[0];assert job.isdigit(),job
  jobs['jobs'][name]=dict(id=job,options=list(opts),args=list(args));write(META/'jobs.json',jobs);write(claim,dict(state='accepted_partial',concurrent_h100=1,global_limit=2,worktree=str(ROOT),jobs=jobs['jobs']));return job
 dep=['--dependency=afterany:'+':'.join(previous)] if previous else []
 smoke=submit('smoke','cluster/jean_zay/ixi_local_components_gpu.slurm','--time=00:30:00',*dep,args=('smoke',))
 zero=submit('zero_shot','cluster/jean_zay/ixi_local_components_gpu.slurm','--time=01:15:00','--dependency=afterok:'+smoke,'--kill-on-invalid-dep=yes',args=('zero',))
 train=submit('training','cluster/jean_zay/ixi_local_components_gpu.slurm','--array=0-5%1','--dependency=afterok:'+zero,'--kill-on-invalid-dep=yes',args=('train',))
 collect=submit('collector','cluster/jean_zay/ixi_local_components_cpu.slurm','--dependency=afterok:'+train,'--kill-on-invalid-dep=yes')
 jobs['all_accepted']=True;write(META/'jobs.json',jobs);write(claim,dict(state='accepted',concurrent_h100=1,global_limit=2,worktree=str(ROOT),release_after=collect,jobs=jobs['jobs']))
 print(json.dumps(jobs,indent=2))
