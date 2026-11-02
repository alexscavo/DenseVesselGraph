"""One bounded GT/head batch; end-to-end detector training is deliberately absent."""
import hashlib,json,subprocess
from pathlib import Path
base=Path('experiments/continuous_representation');record=base/'submission.json'
assert not record.exists(),'Existing submission; inspect instead of duplicating'
plan=json.loads((base/'plan.json').read_text());root=Path(plan['output_root']);root.mkdir(parents=True,exist_ok=False)
(root/'plan.json').write_text((base/'plan.json').read_text())
state=dict(output_root=str(root),jobs={},end_to_end_submitted=False,source_sha256={})
for p in [*Path('scripts').glob('*continuous*.py'),*Path('cluster/jean_zay').glob('*continuous*'),Path('tests/test_continuous_representation.py')]:state['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
def submit(name,resource,mode,dependency=None,array=None,time='02:00:00',cpus=None):
 cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+time]
 if dependency:cmd+=['--dependency=afterok:'+dependency]
 if array:cmd+=['--array='+array]
 if cpus:cmd+=['--cpus-per-task='+str(cpus)]
 cmd+=['cluster/jean_zay/continuous_representation_'+resource+'.slurm',mode,str(root)]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
 state['jobs'][name]=dict(id=job,dependency='afterok:'+dependency if dependency else None,array=array,walltime=time)
 record.write_text(json.dumps(state,indent=2)+'\n');print(name,job,flush=True);return job
gt=submit('GT representations','cpu','gt',array='0-'+str(len(plan['tasks'])-1)+'%4',time='06:00:00',cpus=1)
col=submit('GT collection','cpu','collect',gt,time='02:00:00',cpus=4)
smoke=submit('GPU correctness and benchmark','gpu','benchmark',col,time='01:00:00')
train=submit('Frozen geometry heads K8 K16 K24','gpu','train',smoke,array='0-2%2',time='02:00:00')
pred=submit('Frozen predictions','gpu','predict',train,array='0-2%2',time='02:00:00')
final=submit('Learned metrics and adaptive selectors','cpu','learned',pred,time='06:00:00',cpus=4)
doc=Path('docs/research/continuous_representation_comparison.md')
with doc.open('a') as f:
 f.write('\n### Accepted submission\n\n|Stage|Job|Dependency|Array|Allocation|\n|---|---|---|---|---|\n')
 for name,s in state['jobs'].items():f.write(f'|{name}|{s["id"]}|{s["dependency"] or "none"}|{s["array"] or "none"}|{s["walltime"]}|\n')
 f.write('\nAll stages use afterok prerequisites; GPU concurrency is at most two. No full detector training submitted. GPU correctness and actual update budget remain pending the gated benchmark.\n')
print('Retrieve: python3 scripts/show_continuous_representation.py')
