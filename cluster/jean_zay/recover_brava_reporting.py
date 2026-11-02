"""Recover only BRAVA CPU representation analysis and result collection."""
import hashlib
import json
from pathlib import Path
import subprocess

record=Path('experiments/brava_graph_study/recovery.json')
assert not record.exists(), 'Recovery already recorded; do not submit duplicates'
root=json.loads(Path('experiments/brava_graph_study/submission.json').read_text())['output_root']
state=dict(output_root=root,reason='Coincident SWC endpoints deduplicated to a one-point reference path',
           training_or_inference_rerun=False,jobs={},source_sha256={})
for name in ['scripts/audit_brava_study.py','scripts/collect_brava_study.py','tests/test_brava_reference_geometry.py']:
    state['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
for name,mode,wall,dependency in [('representations','representations','04:00:00',None),('collection','collect','02:00:00','representations')]:
    cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+wall]
    dep=None if dependency is None else 'afterok:'+state['jobs'][dependency]['id']
    if dep:cmd+=['--dependency='+dep]
    cmd+=['cluster/jean_zay/brava_study_cpu.slurm',mode,root]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
    state['jobs'][name]=dict(id=job,dependency=dep,walltime=wall)
    record.write_text(json.dumps(state,indent=2)+'\n');print(name,job,dep,flush=True)
marker='## BRAVA Stage 1 recovery — 22 October 2026'
section=marker+'\n\nThe one-point full-volume reference path was traced to a coincident SWC parent/child pair. The evaluation adapter now writes both coincident endpoints while retaining the original link and geometry. The unchanged common-v2 metric accepts it; a focused regression test and replay of the failing BH0009 edge pass. No source data, model predictions, checkpoints or training were changed.\n\n'
section+='CPU recovery jobs: '+state['jobs']['representations']['id']+' (GT representation analysis,4h allocation) → '+state['jobs']['collection']['id']+' (collector,2h allocation; afterok dependency). Results are pending. The collector will replace this section with model/representation tables and paired subject-bootstrap intervals in both complete review files.\n\nRetrieve without rerunning computation: `python3 scripts/collect_brava_study.py --show`.\n'
for name in ['docs/research/ALL_EXPERIMENTS_AND_RESULTS.md','docs/research/EXPERIMENT_REVIEW_20261021.md']:
    p=Path(name);p.write_text(p.read_text().split(marker)[0].rstrip()+'\n\n'+section)
p=Path('docs/research/BRAVA_experiment_plan.md');p.write_text(p.read_text()+'\n\n'+section)
print('Retrieve: python3 scripts/collect_brava_study.py --show')
