"""Submit one bounded staged batch; maximum two simultaneous H100 allocations."""
import hashlib,json,subprocess
from pathlib import Path
base=Path('experiments/multidataset_continuous');record=base/'submission.json'
assert not record.exists(),'Existing batch recorded; inspect before resubmitting'
assert json.loads((base/'local-audit/audit.json').read_text())['completed'],'Local audit must pass first'
plan=json.loads((base/'plan.json').read_text());root=Path(plan['output_root']);root.mkdir(parents=True,exist_ok=False)
(root/'plan.json').write_text((base/'plan.json').read_text());state=dict(output_root=str(root),jobs={},arms=plan['arms'],maximum_H100_concurrency=2,end_to_end_representation_submitted=False,source_sha256={})
files=['models/adaptive_geometry.py','data/adaptive_split_targets.py','metrics/common_vessel_metrics.py','scripts/multidataset_graph_training.py','scripts/multidataset_graph_evaluation.py','scripts/audit_multidataset_graph.py','scripts/collect_multidataset_graph.py','tests/test_adaptive_geometry.py','tests/test_multidataset_graph.py','cluster/jean_zay/multidataset_continuous_gpu.slurm','cluster/jean_zay/multidataset_continuous_cpu.slurm']
for name in files:state['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
def submit(name,resource,mode,dependency=None,array=None,time='01:00:00'):
 command=['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+time]
 if dependency:command+=['--dependency='+dependency]
 if array:command+=['--array='+array]
 command+=['cluster/jean_zay/multidataset_continuous_'+resource+'.slurm',mode,str(root)]
 job=subprocess.check_output(command,text=True).strip().split(';')[0];assert job.isdigit()
 state['jobs'][name]=dict(id=job,dependency=dependency,array=array,walltime=time)
 record.write_text(json.dumps(state,indent=2)+'\n');print(name,job,dependency or '',flush=True);return job
n=len(plan['arms']);audit=submit('Data and checkpoint audit','cpu','audit')
smoke=submit('GPU forward backward and collector smoke','gpu','smoke','afterok:'+audit,time='01:00:00')
train=submit('Detector recovery and geometry training plus predictions','gpu','train','afterok:'+smoke,array=f'0-{n-1}%2',time='05:00:00')
eval=submit('Per-arm CPU evaluation','cpu','evaluate','aftercorr:'+train,array=f'0-{n-1}%4',time='08:00:00')
collect=submit('Aggregate report','cpu','collect','afterok:'+eval,time='01:00:00')
doc=Path('docs/research/multidataset_continuous_graph_experiments.md')
with doc.open('a') as f:
 f.write('\n## Accepted initial batch\n\n|Stage|Job|Dependency|Array|Wall allocation|\n|---|---|---|---|---|\n')
 for name,s in state['jobs'].items():f.write(f'|{name}|{s["id"]}|{s["dependency"] or "none"}|{s["array"] or "none"}|{s["walltime"]}|\n')
 f.write('\nTraining-array mapping:\n\n|Task|Arm|Dataset|Updates|\n|---|---|---|---:|\n')
 for i,arm in enumerate(plan['arms']):f.write(f'|{i}|{arm["name"]}|{arm["dataset"]}|{plan["detector_updates"] if arm["kind"]=="detector" else plan["geometry_updates"]}|\n')
 f.write('\nGPU smoke checks and actual throughputs are pending their scheduled job. Training starts only after all smoke checks pass. No further model stages are automatically submitted.\n')
print('Retrieve: python3 scripts/show_multidataset_graph.py')
