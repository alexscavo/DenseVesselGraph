"""Submit only CPU cache analysis and collection, preserving existing experiments."""
from pathlib import Path
import hashlib
import json
import subprocess

base=Path('experiments/spline_target_capacity');record=base/'submission.json'
assert not record.exists(),'Existing capacity submission; do not duplicate'
plan=json.loads((base/'plan.json').read_text());root=Path(plan['output_root'])
root.mkdir(parents=True,exist_ok=False);(root/'plan.json').write_text((base/'plan.json').read_text())
state=dict(output_root=str(root),jobs={},GPU_jobs_submitted=False,source_sha256={})
for name in ['scripts/analyze_spline_target_capacity.py','tests/test_spline_target_capacity.py','data/spline_targets.py','metrics/representation_geometry.py','cluster/jean_zay/spline_target_capacity.slurm']:
    state['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
for stage,mode,array in [('analysis','worker','0-7%4'),('collection','collect',None)]:
    command=['sbatch','--parsable','--kill-on-invalid-dep=yes']
    dep=None
    if array:command+=['--array='+array]
    else:dep='afterok:'+state['jobs']['analysis']['id'];command+=['--dependency='+dep,'--cpus-per-task=4']
    command+=['cluster/jean_zay/spline_target_capacity.slurm',mode,str(root)]
    job=subprocess.check_output(command,text=True).strip().split(';')[0];assert job.isdigit()
    state['jobs'][stage]=dict(id=job,dependency=dep,array=array,walltime='02:00:00')
    record.write_text(json.dumps(state,indent=2)+'\n');print(stage,job,dep,flush=True)
doc=Path('docs/research/spline_learning/ixi-brava-control-capacity.md');doc.parent.mkdir(parents=True,exist_ok=True)
doc.write_text('''# IXI and BRAVA — adequacy of 8-control spline targets

Status: CPU analysis submitted; full-population results pending. No GPU inference, training, graph regeneration or test-target reads.

## Locked protocol

- All prepared IXI development patches:10,000 train /2,649 validation;123 /26 disjoint subjects.
- All prepared BRAVA development patches:8,768 train /1,796 validation;39 /8 disjoint subjects.
- Evaluate actual cached cubic8-control targets against their own original dense branch paths in physical millimetres, preserving patch, node and edge conventions.
- Report directed mean/P95/maximum distances, sampled symmetric Hausdorff and continuous-error bounds, arc-length ratios, tortuosity/turning angle, coverage and same-edge Curve F1 at0.5/1/2 mm.
- Sample arc length at0.1 mm plus all vertices for maxima; spline tessellation error bounded by0.01 mm. Continuous maximum lies between sampled maximum−0.01 and sampled maximum+0.06 mm.
- Report per-edge, per-patch and per-subject results, length/tortuosity strata,1,000 subject-bootstrap intervals, and overlap/annotation limitations. Patches are not independent anatomical branches.
- Diagnostic refits with8/12/16 controls:64 worst-error branches plus up to64 seeded, one-per-subject branches per dataset/split, and every edge in requested BRAVA sample_000002_0222. Same production least-squares fitting routine and257 fitting samples. This deliberately enriched subset does not estimate population prevalence.

## Focused verification and preliminary example

Three synthetic checks passed: straight identity, missed-bend shortcut, and overshoot/tessellation-bound behaviour. A reduced64-patch worker/collector smoke exercises both datasets and splits without publishing smoke results. Cached curve replay is checked against the controls in physical coordinates.

The requested BRAVA example is already a64³ training patch. Its longest branch (edge14) has82.106 mm dense arc length, sampled maximum error2.930 mm, mean symmetric error0.933 mm,60.51% dense-path coverage at1 mm and spline/dense arc-length ratio0.8881. This is one example; dataset-wide prevalence is not yet known.

## Execution

'''+f'CPU array: `{state["jobs"]["analysis"]["id"]}` (`0-7%4`,one CPU/task,2h allocation); collector: `{state["jobs"]["collection"]["id"]}` (`afterok`,4 CPUs,2h allocation). Large output: `{root}`.\n\nThe collector replaces this pending report with complete tables, uncertainty, diagnostic fits and figures, copies compact evidence to `docs/research/evidence/spline_target_capacity_20261022/`, and appends the results to `docs/research/ALL_EXPERIMENTS_AND_RESULTS.md`. Full per-edge CSVs remain on SCRATCH.\n\nRetrieve without rerunning analysis:\n\n```bash\npython3 scripts/analyze_spline_target_capacity.py show\n```\n')
print('Retrieve: python3 scripts/analyze_spline_target_capacity.py show')
