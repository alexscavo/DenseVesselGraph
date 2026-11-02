#!/usr/bin/env python3
"""Submit the missing three-arm IXI node-loss continuation; never an edge sweep."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submit',action='store_true');a=p.parse_args()
    directory=Path('experiments/ixi_node_focal_ablation');record=directory/'submission.json'
    if record.exists():raise FileExistsError(f'Existing submission: {record}; no duplicate jobs')
    image=json.loads(Path('experiments/ixi_image_relation_learning/submission.json').read_text())
    shared=image['plan'];root='/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_node_focal_'+datetime.now().strftime('%Y%m%d-%H%M%S')
    refinement=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_node_refinement_20261018/local_image/trained-heads.pt')
    plan=dict(output_root=root,ids=shared['ids'],sites=shared['sites'],subjects=shared['subjects'],
        evaluation_ids=dict(train=shared['previous_ids']['train'],val=shared['ids']['val']),
        dataset=shared['dataset'],dataset_manifest_sha256=shared['dataset_manifest_sha256'],checkpoint=shared['checkpoint'],
        checkpoint_sha256=shared['checkpoint_sha256'],refinement=str(refinement),refinement_sha256=hashlib.sha256(refinement.read_bytes()).hexdigest(),
        arms=['ce_weighted','ce_equal','focal_equal'],seed=201021,batch_size=8,learning_rate=7e-5,scheduler='constant',
        training_target_seconds=3300,benchmark_warmup=3,benchmark_timed_updates=32,
        budget_rule='min(20000, floor(3300 / (1.2 * slowest_seconds_per_update) / 100) * 100); require >=200',
        training_walltime='01:30:00',edge_ce_ratio=.15,edge_ratio_tolerance=.02,
        trainable='all detector, transformer, node-refinement and relation parameters; spline head frozen',
        limitation='Mature IXI initializer trained with edge focal; all arms share edge CE transition and optimizer restart. Learning-curve pilot, not convergence from scratch.')
    files=['scripts/ixi_node_focal_ablation.py','scripts/evaluate_ixi_node_focal.py','scripts/collect_ixi_node_focal.py',
        'tests/test_ixi_node_focal_ablation.py','cluster/jean_zay/ixi_node_focal_gpu.slurm','cluster/jean_zay/ixi_node_focal_cpu.slurm']
    state=dict(output_root=root,plan=plan,jobs={},source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files})
    if a.submit:directory.mkdir(parents=True,exist_ok=True);(directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def submit(name,kind,mode,dep,time='01:30:00',array=None):
        cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+time,'--dependency='+dep]
        if array:cmd+=['--array='+array]
        cmd+=['cluster/jean_zay/ixi_node_focal_'+kind+'.slurm',mode,root]
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0] if a.submit else str(9100000+len(state['jobs']))
        assert job.isdigit();state['jobs'][name]=dict(id=job,dependency=dep,array=array,command=cmd)
        if a.submit:record.write_text(json.dumps(state,indent=2)+'\n')
        print(name,job,dep,flush=True);return job
    # Serialize GPU batches, preserving the existing image experiment and global two-H100 cap.
    smoke=submit('gpu_smoke','gpu','smoke','afterany:'+image['jobs']['training']['id'],time='00:45:00')
    tiny=submit('cpu_smoke_collection','cpu','smoke','afterok:'+smoke,time='00:30:00')
    benchmark=submit('benchmark','gpu','benchmark','afterok:'+tiny,time='00:30:00')
    train=submit('training','gpu','train','afterok:'+benchmark,array='0-2%2')
    prediction=submit('prediction','gpu','predict','afterok:'+train,time='01:30:00',array='0-2%2')
    evaluation=submit('evaluation','cpu','evaluate','aftercorr:'+prediction,time='04:00:00',array='0-2%2')
    submit('collection','cpu','collect','afterok:'+evaluation,time='02:00:00')
    if a.submit:
        state['submitted']=True;record.write_text(json.dumps(state,indent=2)+'\n')
        with Path('docs/research/metrics/10-ixi-node-focal-ablation.md').open('a') as f:
            f.write('\n## Accepted submission\n\nOutput: `'+root+'`.\n\n|Stage|Job|Dependency|\n|---|---|---|\n')
            for name,j in state['jobs'].items():f.write(f'|{name}|{j["id"]}|{j["dependency"]}|\n')
    print('Retrieve: python3 scripts/collect_ixi_node_focal.py --show')


if __name__=='__main__':main()
