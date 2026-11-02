"""No training: queue tiny BN diagnostics behind all active agents' GPU work."""
import fcntl,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];META=ROOT/'experiments/cheap_diagnostics_20261026';SHARED=ROOT.parent/'experiment_gpu_coordination';SHARED.mkdir(exist_ok=True)
def command(*args):return subprocess.check_output(args,cwd=ROOT,text=True).strip()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
with (SHARED/'submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not (META/'jobs.json').exists(),'Existing submission must not be duplicated'
 queue=command('squeue','-h','-u','upz25mj','-o','%i|%j|%T|%b|%P');ids=[]
 for line in queue.splitlines():
  parts=line.split('|')
  if 'gpu' in parts[3].lower() or 'gpu' in parts[4].lower():ids.append(parts[0].split('_')[0])
 ids=sorted(set(ids));dependency=['--dependency=afterany:'+':'.join(ids)] if ids else []
 jobs=dict(submitted_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),worktree=str(ROOT),prior_gpu_jobs=ids,queue_at_submission=queue,concurrent_h100=1,global_limit=2,jobs={})
 gpu=command('sbatch','--parsable','--array=0-1%1',*dependency,str(ROOT/'cluster/jean_zay/cheap_bn_gpu.slurm')).split(';')[0];assert gpu.isdigit();jobs['jobs']['bn']=dict(id=gpu,dependencies=ids);write(META/'jobs.json',jobs)
 report=command('sbatch','--parsable','--dependency=afterok:'+gpu,'--kill-on-invalid-dep=yes',str(ROOT/'cluster/jean_zay/cheap_bn_cpu.slurm')).split(';')[0];assert report.isdigit();jobs['jobs']['collector']=dict(id=report,dependency=gpu);jobs['accepted']=True;write(META/'jobs.json',jobs)
 write(SHARED/'cheap-diagnostics-20261026.json',dict(state='accepted',worktree=str(ROOT),concurrent_h100=1,global_limit=2,prior_gpu_jobs=ids,release_after=report,jobs=jobs['jobs'],report_ownership=['docs/research/spline_learning/spline-collapse-diagnosis.md','docs/research/metrics/9-ixi-image-aware-relation-learning.md'],roadmap='Append only if unchanged since audit; no future automatic roadmap edits'))
 print(json.dumps(jobs,indent=2))
