#!/usr/bin/env python3
"""Submit the bounded P0 comparison only after a reviewed, successful smoke."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-job', required=True, type=int)
    p.add_argument('--submit', action='store_true')
    a = p.parse_args()
    smoke = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_query_diagnostics_20261019')/f'smoke-{a.smoke_job}-local_image'
    summary = json.loads((smoke/'summary.json').read_text())
    reproduction = json.loads((smoke/'reproduction.json').read_text())
    assert summary['cache_decode_parity'] and summary['frozen_state_unchanged']
    assert reproduction['checked'] == 4 and reproduction['exact_query_ids_and_edge_sets']
    record = Path('experiments/ixi_query_diagnostics/submission.json')
    if record.exists():
        raise FileExistsError(f'Existing submission record: {record}; inspect it before submitting again')
    state = dict(smoke_job=a.smoke_job, smoke=str(smoke), mode='pilot',
                 train_patches=256, validation_patches=128, model_updates=0, jobs={})
    if a.submit:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(state, indent=2)+'\n')
    for arm in ('parent', 'baseline', 'local_image'):
        command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', f'--dependency=afterok:{a.smoke_job}',
                   '--time=02:00:00', 'cluster/jean_zay/ixi_query_diagnostics.slurm', 'pilot', arm]
        if a.submit:
            job = subprocess.check_output(command, text=True).strip().split(';')[0]
            state['jobs'][arm] = dict(id=job, command=command,
                output=f'/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_query_diagnostics_20261019/pilot-{job}-{arm}')
            record.write_text(json.dumps(state, indent=2)+'\n')
            print(arm, job, flush=True)
        else:
            print(' '.join(command))


if __name__ == '__main__':
    main()
