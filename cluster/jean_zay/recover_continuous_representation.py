"""Resume only blocked neural stages after variable-K metric accounting repair."""
import hashlib,json,subprocess
from pathlib import Path
base=Path('experiments/continuous_representation');record=base/'submission.json';recovery=base/'recovery.json'
assert not recovery.exists(),'Recovery already recorded; do not duplicate'
state=json.loads(record.read_text());root=Path(state['output_root']);assert json.loads((root/'results/gt-gate.json').read_text())['completed']
state['previous_jobs']=dict(state['jobs']);new={}
def submit(name,mode,resource,dep=None,array=None,time='02:00:00'):
 cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+time]
 if dep:cmd+=['--dependency=afterok:'+dep]
 if array:cmd+=['--array='+array]
 cmd+=['cluster/jean_zay/continuous_representation_'+resource+'.slurm',mode,str(root)]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
 new[name]=dict(id=job,dependency='afterok:'+dep if dep else None,array=array,walltime=time)
 state['jobs'][name]=new[name];record.write_text(json.dumps(state,indent=2)+'\n');recovery.write_text(json.dumps(dict(jobs=new,output_root=str(root)),indent=2)+'\n');print(name,job,flush=True);return job
s=submit('GPU correctness and benchmark','benchmark','gpu',time='01:00:00')
t=submit('Frozen geometry heads K8 K16 K24','train','gpu',s,'0-2%2')
p=submit('Frozen predictions','predict','gpu',t,'0-2%2')
e=submit('Learned metrics and adaptive selectors','learned','cpu',p,time='06:00:00')
state['recovery_reason']='Original smoke1434270 passed training/prediction; collector failed counting variable-K controls via rectangular np.asarray. Count is now summed per edge; common-v2 geometry definitions unchanged.'
for path in ['metrics/common_vessel_metrics.py','scripts/evaluate_continuous_geometry.py','scripts/train_continuous_geometry.py','tests/test_adaptive_curve_metrics.py','cluster/jean_zay/continuous_representation_gpu.slurm']:
 state['source_sha256'][path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
record.write_text(json.dumps(state,indent=2)+'\n')
doc=Path('docs/research/continuous_representation_comparison.md');text=doc.read_text();marker='\n<!-- GENERATED RESULTS -->\n';parts=text.split(marker,1)
lines=['','## Recovery of the geometry smoke collector','','All18 GT tasks and collector1434269 completed. GPU smoke1434270 passed the five original tests,36 short head-training updates and all six-patch prediction caches, then failed in common-v2 representation-control bookkeeping. A rectangular array conversion assumed equal K on every edge. The fix sums each edge\'s internal control count; geometric/topological metric definitions and existing uniform-K counts are unchanged. The cached replay also exposed string metadata being passed into the numeric metric aggregator; it now receives only common metric fields. A focused regression checks ragged and fixed-K counts and metric parity. Saved smoke outputs were replayed on CPU. No GT work or full training is repeated. The replacement benchmark uses a new smoke directory, preserving failed-run evidence, and saves timing before collection.','','|Stage|Replacement job|Dependency|','|---|---|---|']
for name,v in new.items():lines.append(f'|{name}|{v["id"]}|{v["dependency"] or "none"}|')
doc.write_text(parts[0]+'\n'.join(lines)+'\n'+(marker+parts[1] if len(parts)>1 else ''))
print('Retrieve: python3 scripts/show_continuous_representation.py')
