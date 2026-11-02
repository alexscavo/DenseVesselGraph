#!/usr/bin/env python3
"""Submit the cache-only CPU matching/calibration batch; never wait or poll."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--submit', action='store_true'); args = p.parse_args()
    record = Path('experiments/ixi_relation_matching_calibration/submission.json')
    if args.submit and record.exists():
        raise FileExistsError(record)
    root = '/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_relation_matching_calibration_'+datetime.now().strftime('%Y%m%d-%H%M%S')
    sources = ['scripts/ixi_relation_matching_calibration.py', 'scripts/collect_ixi_relation_matching_calibration.py',
               'scripts/ixi_cached_mechanisms.py', 'scripts/ixi_query_diagnostics.py', 'models/matcher.py',
               'models/spline.py', 'models/local_spline_geometry.py', 'metrics/common_vessel_metrics.py',
               'tests/test_ixi_relation_matching_calibration.py', 'cluster/jean_zay/ixi_relation_matching_calibration.slurm']
    state = dict(output_root=root, cpu_only=True, training=False, train_patches=256, validation_patches=128, jobs={},
                 source_sha256={s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources})
    def submit(name, mode, dependency=None, array=None, time='02:00:00'):
        command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', '--time='+time]
        if dependency:
            command += ['--dependency='+dependency]
        if array:
            command += ['--array='+array]
        command += ['cluster/jean_zay/ixi_relation_matching_calibration.slurm', mode, root]
        job = subprocess.check_output(command, text=True).strip().split(';')[0] if args.submit else str(9000000+len(state['jobs']))
        assert job.isdigit(), job
        state['jobs'][name] = dict(id=job, dependency=dependency, array=array, command=command)
        if args.submit:
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps(state, indent=2)+'\n')
        print(name, job, ' '.join(command), flush=True)
        return job
    smoke = submit('smoke', 'smoke', time='00:30:00')
    preparation = submit('preparation', 'prepare', 'afterok:'+smoke)
    analyses = submit('analyses', 'analyse', 'afterok:'+preparation, '0-5%3')
    submit('collection', 'collect', 'afterok:'+analyses)
    if args.submit:
        state['submitted'] = True; record.write_text(json.dumps(state, indent=2)+'\n')
        doc = Path('docs/research/metrics/8-ixi-relation-matching-calibration.md')
        with doc.open('a') as f:
            f.write('\n## Accepted submission\n\nOutput root: `'+root+'`.\n\n|Stage|ID|Dependency|\n|---|---|---|\n')
            for name, j in state['jobs'].items():
                f.write(f'|{name}|{j["id"]}|{j["dependency"] or "none"}|\n')
            f.write('\nArray indices: 0 Hungarian/query-selection analysis; 1 original; 2 focal; 3 regularized; 4 unmatched-hard; 5 matched-hard.\n')
    print('Retrieve: python3 scripts/collect_ixi_relation_matching_calibration.py --show')


if __name__ == '__main__':
    main()
