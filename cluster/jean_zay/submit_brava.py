#!/usr/bin/env python3
"""Submit a once-only CPU Brava graph/skull-stripping/patch preparation chain."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true')
    parser.add_argument('--retry-failed',action='store_true',help='Reuse the failed audit output root after verifying all old jobs are terminal')
    parser.add_argument('--alignment-preflight',type=Path,help='Completed local all-subject/variant audit evidence')
    args=parser.parse_args()
    directory=Path('experiments/brava_preparation');record=directory/'submission.json'
    previous=None
    if record.exists():
        if not args.retry_failed:raise FileExistsError(f'Existing exact preparation: {record}; no duplicate jobs')
        previous=json.loads(record.read_text())
        ids=','.join(j['id'] for j in previous['jobs'].values())
        states=subprocess.check_output(['sacct','-X','-n','-j',ids,'--format=State','-P'],text=True)
        terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
        found=[s.strip().split('|')[0].split()[0].rstrip('+') for s in states.splitlines() if s.strip()]
        assert found and all(s in terminal for s in found), ('Old jobs not terminal; no duplicate retry',found)
        assert not (Path(previous['output_root'])/'audit/complete.json').exists(), 'Retry only the failed initial audit; do not repeat a successful preparation'
        if args.submit:
            archive=directory/('submission-'+previous['jobs']['alignment_and_native_exports']['id']+'.json')
            assert not archive.exists(), archive
            archive.write_text(json.dumps(previous,indent=2)+'\n')
    source=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava')
    count=len(list((source/'brains').glob('*.nii.gz')));assert count==56
    root=previous['output_root'] if previous else str(source/('prepared_swc_'+datetime.now().strftime('%Y%m%d-%H%M%S')))
    state=dict(output_root=root,source=str(source),jobs={},image_subjects=count,
        policy=dict(skull_strip='image * (supplied brain mask > 0)',vessel_segmentations=False,
            graphs='provided SWC parent links; dense, junction-only and radius-adaptive; preserve original guides',
            primary='original except malformed BH0027 original: use complete colorcoded counterpart',
            patch_size=[64,64,64],maximum_stride=[40,40,40],pad=[0,0,0],
            split='subject-disjoint 70/15/15 seed42; all trace variants share the same subject',
            targets='shared junction/end/boundary anchors, 8 B-spline controls, 256-query capacity; all compatible patches',
            seg_slot='supplied brain mask only, for legacy loader compatibility; not vessel GT'),
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['scripts/prepare_brava.py','cluster/jean_zay/prepare_brava.slurm',
            'docs/research/evidence/brava_preparation/orientation-corrections.json']})
    if previous:state['retry_of']=previous['jobs']['alignment_and_native_exports']['id']
    if args.alignment_preflight:
        checked=json.loads(args.alignment_preflight.read_text())
        assert checked['completed'] and len(checked['inventory']['subjects'])==count
        assert len(checked['checks'])==sum(len(r['valid_swcs']) for r in checked['inventory']['subjects'])
        assert checked['orientation_corrections_sha256']==state['source_sha256']['docs/research/evidence/brava_preparation/orientation-corrections.json']
        state['alignment_preflight']=dict(path=str(args.alignment_preflight),sha256=hashlib.sha256(args.alignment_preflight.read_bytes()).hexdigest(),subjects=count,valid_variants=len(checked['checks']))
    if args.submit:directory.mkdir(parents=True,exist_ok=True)
    def submit(name,mode,dependency=None,time='01:00:00',array=None):
        command=['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+time]
        if dependency:command+=['--dependency=afterok:'+dependency]
        if array:command+=['--array='+array]
        command+=['cluster/jean_zay/prepare_brava.slurm',mode,root]
        job=subprocess.check_output(command,text=True).strip().split(';')[0] if args.submit else str(9200000+len(state['jobs']))
        assert job.isdigit();state['jobs'][name]=dict(id=job,dependency=dependency,array=array,command=command)
        if args.submit:record.write_text(json.dumps(state,indent=2)+'\n')
        print(name,job,'afterok:'+dependency if dependency else 'none',flush=True);return job
    audit=submit('alignment_and_native_exports','audit')
    smoke=submit('real_graph_and_patch_smoke','smoke',audit)
    graphs=submit('full_volume_graphs_and_skull_stripping','subject',smoke,time='00:30:00',array=f'0-{count-1}%4')
    staging=submit('subject_split_and_staging','stage',graphs,time='00:15:00')
    patches=submit('ixi_grid_patches','patches',staging,time='04:00:00')
    targets=submit('model_compatible_spline_targets','targets',patches,time='04:00:00')
    submit('collection','collect',targets,time='00:30:00')
    if args.submit:
        state['submitted']=True;record.write_text(json.dumps(state,indent=2)+'\n')
        with Path('docs/research/BRAVA_preparation.md').open('a') as f:
            f.write('\n## Accepted jobs\n\nOutput: `'+root+'`.\n\n|Stage|Job ID|afterok|\n|---|---|---|\n')
            for name,j in state['jobs'].items():f.write(f'|{name}|{j["id"]}|{j["dependency"] or "none"}|\n')
            f.write('\nRetrieve without waiting or rerunning preparation:\n\n```bash\npython3 scripts/prepare_brava.py show --output '+root+'\n```\n')
    print('Output',root)
    print('Retrieve: python3 scripts/prepare_brava.py show --output '+root)


if __name__=='__main__':main()
