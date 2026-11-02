#!/usr/bin/env python3
"""Submit the bounded cache-based A–D batch and its automatic collectors; never poll."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    record = Path('experiments/ixi_mechanisms/submission.json')
    if args.submit and record.exists():
        raise FileExistsError(f'Existing batch record: {record}')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root = f'/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_mechanisms_{stamp}'
    state = dict(output_root=root, train_patches=256, validation_patches=128, test_evaluation=False, jobs={},
                 git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     sorted(list(Path('scripts').glob('*ixi*mechanism*.py'))+list(Path('cluster/jean_zay').glob('*ixi*mechanism*'))+
                            [Path(p) for p in ['tests/test_ixi_cached_mechanisms.py', 'training/evaluation/inference.py',
                             'training/losses/criterion.py', 'training/losses/focal.py', 'models/matcher.py', 'models/spline.py',
                             'models/local_spline_geometry.py', 'metrics/common_vessel_metrics.py', 'scripts/ixi_query_diagnostics.py']])})
    if args.submit:
        record.parent.mkdir(parents=True, exist_ok=True)
        Path('experiments/spline_learning/logs').mkdir(parents=True, exist_ok=True)
    def submit(name, wrapper, mode, dependencies=None, array=None, time='02:00:00'):
        command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', '--time='+time]
        if dependencies:
            command += ['--dependency='+dependencies]
        if array:
            command += ['--array='+array]
        command += [f'cluster/jean_zay/{wrapper}', mode, root]
        if args.submit:
            job = subprocess.check_output(command, text=True).strip().split(';')[0]
            assert job.isdigit(), job
        else:
            job = str(9000000+len(state['jobs']))
        state['jobs'][name] = dict(id=job, dependency=dependencies, array=array, command=command)
        if args.submit:
            record.write_text(json.dumps(state, indent=2)+'\n')
        print(name, job, ' '.join(command), flush=True)
        return job
    cpu = 'ixi_mechanisms_cpu.slurm'; gpu = 'ixi_mechanisms_relation.slurm'
    smoke = submit('cpu_smoke', cpu, 'smoke', time='00:30:00')
    prepare = submit('features', cpu, 'prepare', 'afterok:'+smoke)
    abc = submit('ABC', cpu, 'abc', 'afterok:'+prepare, '0-2%3')
    dsmoke = submit('relation_smoke', gpu, 'smoke', 'afterok:'+smoke, time='00:30:00')
    training = submit('D', gpu, 'pilot', 'afterok:'+dsmoke, '0-3%2')
    evaluation = submit('D_evaluation', cpu, 'relation', f'aftercorr:{training},afterok:{prepare}', '0-3%2')
    submit('collection', cpu, 'collect', f'afterok:{abc}:{evaluation}', time='00:30:00')
    if args.submit:
        state['submitted'] = True
        record.write_text(json.dumps(state, indent=2)+'\n')
    print('Later: python3 scripts/collect_ixi_mechanisms.py --show')


if __name__ == '__main__':
    main()
