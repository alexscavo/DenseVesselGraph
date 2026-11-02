"""Recover image-study report tables only, then collect after all writers stop."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submit',action='store_true');a=p.parse_args()
    folder=Path('experiments/ixi_image_relation_learning');record=folder/'reporting-recovery.json'
    if record.exists():raise FileExistsError(f'Recovery already recorded: {record}')
    original=json.loads((folder/'submission.json').read_text());root=original['output_root']
    dep='afterany:'+original['jobs']['original_evaluation']['id']+':'+original['jobs']['adapter_evaluation']['id']
    state=dict(output_root=root,jobs={},inference=False,training=False,
        cause='Empty degree histogram [] coerces to float64 during addition into int64 accumulator',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in map(Path,[
            'scripts/evaluate_ixi_image_relations.py','scripts/collect_ixi_image_relations.py',
            'cluster/jean_zay/ixi_image_relations_cpu.slurm','tests/test_image_relation_reporting.py'])})
    def submit(name,mode,dependency,time,array=None):
        command=['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+time,'--dependency='+dependency]
        if array:command+=['--array='+array]
        command+=['cluster/jean_zay/ixi_image_relations_cpu.slurm',mode,root]
        job=subprocess.check_output(command,text=True).strip().split(';')[0] if a.submit else str(9300000+len(state['jobs']))
        assert job.isdigit();state['jobs'][name]=dict(id=job,dependency=dependency,array=array,command=command)
        if a.submit:record.write_text(json.dumps(state,indent=2)+'\n')
        print(name,job,dependency,flush=True);return job
    reporting=submit('saved_table_recovery','finish-reporting',dep,'00:30:00','0-4%2')
    submit('collection','collect','afterok:'+reporting,'02:00:00')
    if a.submit:
        state['submitted']=True;record.write_text(json.dumps(state,indent=2)+'\n')
        with Path('docs/research/metrics/9-ixi-image-aware-relation-learning.md').open('a') as f:
            f.write('\n## CPU reporting recovery\n\nEmpty-graph degree histograms caused NumPy float-to-integer addition failures after the primary metric tables were saved. Explicit integer conversion fixes the arithmetic. Two targeted regressions pass; real saved-table replay was verified locally. Recovery checks all expected sample/tolerance/policy rows, reads existing scores and graph/length tables, and completes reporting without model inference, graph-metric recomputation or training. A separate reporting completion marker now gates collection. Original jobs and result tables are preserved.\n\n')
            for name,j in state['jobs'].items():f.write(f'- {name}: `{j["id"]}`, `{j["dependency"]}`.\n')
            f.write('\nRecovery waits for the original evaluation array to finish before touching its output folders. Retrieve: `python3 scripts/collect_ixi_image_relations.py --show`.\n')
    print('Retrieve: python3 scripts/collect_ixi_image_relations.py --show')


if __name__=='__main__':main()
