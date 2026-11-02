"""One serial BRAVA H100 chain, coordinated with the independent IXI agent."""
import fcntl,json,subprocess,time,sys,re
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];ROOT=REPO/'experiments/brava_representation_20261026';SHARED=REPO.parent/'experiment_gpu_coordination'
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def cmd(*x):return subprocess.check_output(x,cwd=REPO,text=True).strip()
with (SHARED/'submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert not (ROOT/'jobs.json').exists()
 other=[json.loads(p.read_text()) for p in SHARED.glob('*.json') if p.name!='brava-representation-20261026.json']
 active=[p for p in other if p.get('state') not in ['complete','released','cancelled']]
 known={j['id'] for p in active for j in p.get('jobs',{}).values()}
 queue=cmd('squeue','-h','-u','upz25mj','-o','%i|%T|%b|%P');unknown=[]
 for line in queue.splitlines():
  fields=line.split('|');job=fields[0].split('_')[0]
  if ('gpu' in fields[2].lower() or 'gpu' in fields[3].lower()) and job not in known:unknown.append(job)
 # Two registered studies can share the other slot only when Slurm proves
 # that every queued downstream GPU job waits for all queued upstream jobs.
 serialized=[]
 if sum(p.get('concurrent_h100',0) for p in active)>1:
  assert len(active)==2 and all(p.get('concurrent_h100')==1 for p in active),active
  gpu_ids={line.split('|')[0].split('_')[0] for line in queue.splitlines() if 'gpu' in line.split('|')[2].lower() or 'gpu' in line.split('|')[3].lower()}
  for upstream,downstream in [active,active[::-1]]:
   up={j['id'] for j in upstream.get('jobs',{}).values()} & gpu_ids
   down={j['id'] for j in downstream.get('jobs',{}).values()} & gpu_ids
   if not up or not down:
    serialized=[upstream['worktree'],downstream['worktree']];break
   if not up.issubset(set(downstream.get('prior_gpu_jobs',[]))):continue
   states=[cmd('scontrol','show','job',j) for j in down]
   if all('JobState=PENDING' in state and all(re.search(r'afterany:'+re.escape(j)+r'(?:_|\()',state) for j in up) for state in states):
    serialized=[upstream['worktree'],downstream['worktree']];break
  assert serialized,('Other studies do not have a verified serial GPU dependency',active)
 jobs=dict(other_studies_verified_serial=serialized,cpu_targets=sys.argv[1],queue_at_submission=queue,other_claims=active,unknown_gpu_prerequisites=sorted(set(unknown)),max_study_h100=1,max_global_h100=2,submitted_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),jobs={})
 def submit(name,script,args,dependencies):
  options=['--dependency=afterok:'+':'.join(dependencies),'--kill-on-invalid-dep=yes']
  if name.startswith('tiny_'):options.append('--time=00:30:00')
  if name=='tiny_original' and unknown:options[0]+= ',afterany:'+':'.join(sorted(set(unknown)))
  job=cmd('sbatch','--parsable',*options,str(REPO/script),*args).split(';')[0];assert job.isdigit()
  jobs['jobs'][name]=dict(id=job,args=args,options=options);write(ROOT/'jobs.json',jobs)
  write(SHARED/'brava-representation-20261026.json',dict(owner='BRAVA representations',state='accepted_partial',concurrent_h100=1,global_limit=2,worktree=str(REPO),jobs=jobs['jobs']))
  return job
 gpu='cluster/jean_zay/brava_representation_study_gpu.slurm';cpu='cluster/jean_zay/brava_representation_study_cpu.slurm'
 first=submit('tiny_original',gpu,['tiny','--group','original'],[sys.argv[1]])
 second=submit('tiny_split',gpu,['tiny','--group','split'],[first])
 original=submit('train_original',gpu,['train','--group','original'],[second])
 split=submit('train_split',gpu,['train','--group','split'],[original])
 for group,dep in [('original',original),('split',split)]:
  for step in [1000,2000,4000]:submit(f'evaluate_{group}_{step}',cpu,['evaluate','--group',group,'--step',str(step)],[dep])
 jobs['all_accepted']=True;write(ROOT/'jobs.json',jobs)
 write(SHARED/'brava-representation-20261026.json',dict(owner='BRAVA representations',state='accepted',concurrent_h100=1,global_limit=2,worktree=str(REPO),release_after=split,jobs=jobs['jobs']))
 print(json.dumps(jobs,indent=2))
