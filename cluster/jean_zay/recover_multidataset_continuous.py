"""Resubmit blocked stages after the snapshot orientation import repair."""
import hashlib,json,subprocess
from pathlib import Path
base=Path('experiments/multidataset_continuous');record=base/'submission.json';recovery=base/'recovery.json'
assert not recovery.exists(),'Recovery recorded; inspect rather than duplicate'
state=json.loads(record.read_text());root=Path(state['output_root']);assert json.loads((root/'audit.json').read_text())['completed']
state['previous_jobs']=dict(state['jobs']);plan=json.loads((root/'plan.json').read_text());n=len(plan['arms']);jobs={}
def submit(name,resource,mode,dep=None,array=None,time='01:00:00'):
 cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+time]
 if dep:cmd+=['--dependency='+dep]
 if array:cmd+=['--array='+array]
 cmd+=['cluster/jean_zay/multidataset_continuous_'+resource+'.slurm',mode,str(root)]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
 item=dict(id=job,dependency=dep,array=array,walltime=time);jobs[name]=item;state['jobs'][name]=item
 record.write_text(json.dumps(state,indent=2)+'\n');recovery.write_text(json.dumps(dict(jobs=jobs,reason='Missing snapshot canonical_reverse import; completed audit reused',output_root=str(root)),indent=2)+'\n');print(name,job,dep or '',flush=True);return job
smoke=submit('GPU forward backward and collector smoke','gpu','smoke')
train=submit('Detector recovery and geometry training plus predictions','gpu','train','afterok:'+smoke,array=f'0-{n-1}%2',time='05:00:00')
eval=submit('Per-arm CPU evaluation','cpu','evaluate','aftercorr:'+train,array=f'0-{n-1}%4',time='08:00:00')
collect=submit('Aggregate report','cpu','collect','afterok:'+eval)
for name in ['scripts/multidataset_graph_evaluation.py','scripts/multidataset_graph_training.py','tests/test_multidataset_graph.py']:state['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
record.write_text(json.dumps(state,indent=2)+'\n')
p=Path('docs/research/multidataset_continuous_graph_experiments.md')
with p.open('a') as f:
 f.write('\n## Smoke recovery —22 October2026\n\nCluster audit1451641 completed. GPU smoke1451642 passed the seven original tests and the first detector forward/backward updates, then failed because snapshot lacked its `canonical_reverse` import. Main training never started; downstream jobs were cancelled by their dependencies. The missing import was added and the actual snapshot-global orientation lookup was verified on CPU. A regression now exercises that dependency. Retry smoke folders include the job ID, preserving the failed evidence. Data, objectives and budgets are unchanged.\n\n|Replacement stage|Job|Dependency|\n|---|---|---|\n')
 for name,s in jobs.items():f.write(f'|{name}|{s["id"]}|{s["dependency"] or "none"}|\n')
print('Retrieve: python3 scripts/show_multidataset_graph.py')
