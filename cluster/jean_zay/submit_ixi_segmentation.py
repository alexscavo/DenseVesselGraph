"""Prepare or submit the paired mask-input diagnostic, without duplicate jobs."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submit',action='store_true');a=p.parse_args()
    directory=Path('experiments/ixi_segmentation_input');directory.mkdir(exist_ok=True)
    record=directory/'submission.json';planfile=directory/'plan.json'
    if record.exists():raise FileExistsError(f'Existing submission {record}; refusing duplicate jobs')
    if planfile.exists():plan=json.loads(planfile.read_text())
    else:
        shared=json.loads(Path('experiments/ixi_image_relation_learning/plan.json').read_text())
        initializer=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/local_controlled_unfreeze_20261010/graph_only/model.pt')
        plan={k:shared[k] for k in ['ids','previous_ids','sites','subjects','dataset','dataset_manifest_sha256']}
        plan.update(output_root='/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_segmentation_input_'+datetime.now().strftime('%Y%m%d-%H%M%S'),
            initializer=str(initializer),initializer_sha256=hashlib.sha256(initializer.read_bytes()).hexdigest(),
            seed=201021,arms=['raw_mra','vessel_mask'],batch_size=8,training_walltime='02:00:00',
            budget_rule='min(30000, floor(4500/(1.2*slowest_seconds_per_update)/100)*100); require >=2000',
            provenance_category=2,graph_source='Supplied labelsTr; original segmentation annotation method unknown',
            trainable='All detector parameters except existing frozen spline head',
            initializer_transfer='120 source object queries preserved; 136 shared seeded queries; two source relation tokens moved to indices 256:258; fresh identity node-refinement')
        planfile.write_text(json.dumps(plan,indent=2)+'\n')
    if not a.submit:
        print(planfile);print(plan['output_root']);return
    root=plan['output_root']
    files=['scripts/ixi_segmentation_oracle.py','scripts/evaluate_ixi_segmentation_oracle.py','scripts/collect_ixi_segmentation_oracle.py',
        'tests/test_ixi_segmentation_oracle.py','data/loaders/synthetic_mri.py','data/loaders/mixed.py','configs/loader.py',
        'cluster/jean_zay/ixi_segmentation_gpu.slurm','cluster/jean_zay/ixi_segmentation_cpu.slurm']
    state=dict(output_root=root,plan=plan,jobs={},source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files})
    def submit(name,kind,mode,dep=None,time='02:00:00',array=None):
        cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+time]
        if dep:cmd+=['--dependency='+dep]
        if array:cmd+=['--array='+array]
        cmd+=['cluster/jean_zay/ixi_segmentation_'+kind+'.slurm',mode,root]
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
        state['jobs'][name]=dict(id=job,dependency=dep,array=array,walltime=time,command=cmd)
        record.write_text(json.dumps(state,indent=2)+'\n');print(name,job,dep,flush=True);return job
    audit=submit('alignment_audit','cpu','audit',time='01:30:00')
    node=json.loads(Path('experiments/ixi_node_focal_ablation/submission.json').read_text())
    smoke=submit('gpu_smoke','gpu','smoke','afterok:'+audit+',afterany:'+node['jobs']['prediction']['id'],time='00:45:00')
    tiny=submit('cpu_smoke_collection','cpu','smoke','afterok:'+smoke,time='00:45:00')
    bench=submit('benchmark','gpu','benchmark','afterok:'+tiny,time='00:30:00')
    train=submit('training','gpu','train','afterok:'+bench,array='0-1%2')
    pred=submit('prediction','gpu','predict','afterok:'+train,time='02:00:00',array='0-1%2')
    evaluate=submit('evaluation','cpu','evaluate','aftercorr:'+pred,time='06:00:00',array='0-1%2')
    submit('collection','cpu','collect','afterok:'+evaluate,time='02:00:00')
    state['submitted']=True;record.write_text(json.dumps(state,indent=2)+'\n')
    with Path('docs/research/metrics/11-ixi-segmentation-input-oracle.md').open('a') as f:
        f.write('\n## Accepted submissions\n\nOutput root: `'+root+'`.\n\n|Stage|Job|Dependency|Wall time|\n|---|---|---|---|\n')
        for name,j in state['jobs'].items():f.write(f'|{name}|{j["id"]}|{j["dependency"]}|{j["walltime"]}|\n')
    print('Retrieve: python3 scripts/collect_ixi_segmentation_oracle.py --show')


if __name__=='__main__':main()
