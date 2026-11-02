"""Submit only prerequisitely valid L08 jobs under the shared GPU lock."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.node_first_curriculum import META, MAIN, plan, read


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def cmd(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot', 'full'], default='pilot')
    args = parser.parse_args()
    p = plan()
    stage = args.stage
    root = Path(p['output_root'])
    record = META / (stage + '-jobs.json')
    assert not record.exists(), 'Preserve existing submission; do not duplicate it'
    if stage == 'full':
        assert read(root / 'pilot/decision.json')['full_allowed'], 'Pilot did not meet promotion gate'
        assert (root / 'pilot/analysis.json').exists(), 'Verified pilot analysis missing'
        assert read(root / 'cpu-smoke.json')['completed'] and read(root / 'smoke.json')['completed']
    shared = MAIN.parent / 'experiment_gpu_coordination'
    shared.mkdir(exist_ok=True)
    (META / 'logs').mkdir(exist_ok=True)
    # Freeze source identities separately from all random/trained weights.
    sources = [path for directory in ['models', 'training', 'data', 'configs', 'metrics', 'scripts', 'tests']
               for path in (ROOT / directory).rglob('*')
               if path.is_file() and path.suffix in ['.py', '.yaml', '.cu', '.cuh', '.cpp', '.h']]
    sources += [ROOT / 'cluster/jean_zay' / name for name in
                ['submit_node_first_curriculum.py', 'node_first_gpu.slurm', 'node_first_cpu.slurm']]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    if stage == 'full':
        assert hashes == read(META / 'pilot-source-sha256.json'), 'Source changed after pilot; require review'
    write(META / (stage + '-source-sha256.json'), hashes)
    with zipfile.ZipFile(META / (stage + '-source.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(ROOT)))
        archive.write(META / 'plan.json', str((META / 'plan.json').relative_to(ROOT)))
    gpu_time = '03:00:00'
    if stage == 'full':
        evidence = read(root / 'pilot/analysis.json')
        seconds = max(r['seconds'] for seed in evidence['results'].values() for r in seed['completion'])
        estimate = max(seconds * 5 * 1.2 + 1200,
                       read(root / 'smoke.json')['full_train_seconds_estimate'] * 1.3 + 1800)
        hours = math.ceil(estimate / 1800) / 2
        assert hours <= 12, 'Estimated full run exceeds bounded12h wall allocation'
        gpu_time = f'{int(hours):02d}:{int(hours % 1 * 60):02d}:00'
    with (shared / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not record.exists(), 'Another submitter already recorded this stage'
        queue = cmd('squeue', '-h', '-u', 'upz25mj', '-o', '%i|%j|%T|%b|%P')
        prior = sorted({line.split('|')[0].split('_')[0] for line in queue.splitlines()
                        if 'gpu' in line.split('|')[3].lower() or 'gpu' in line.split('|')[4].lower()})
        jobs = dict(stage=stage, submitted_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    worktree=str(ROOT), initializers='random step0; no pretrained graph weights',
                    max_h100=2, shared_global_limit=2, queue_at_submission=queue,
                    prior_gpu_jobs=prior, updates=p['stages'][stage]['updates'],
                    seeds=p['stages'][stage]['seeds'], training_walltime=gpu_time, jobs={})
        claim = shared / 'node-first-curriculum-20261028.json'
        write(claim, dict(owner='L08 scratch node-first curriculum', stage=stage, state='submitting',
                          concurrent_h100=2, global_limit=2, worktree=str(ROOT),
                          note='At most two single-GPU arms; wait for all earlier GPU jobs. Other agents must wait or coordinate under submission.lock.'))
        def submit(name, script, options=(), arguments=()):
            job = cmd('sbatch', '--parsable', *options, str(ROOT / script), *arguments).split(';')[0]
            assert job.isdigit(), job
            jobs['jobs'][name] = dict(id=job, options=list(options), arguments=list(arguments))
            write(record, jobs)
            write(claim, dict(owner='L08 scratch node-first curriculum', stage=stage,
                              state='accepted_partial', concurrent_h100=2, global_limit=2,
                              worktree=str(ROOT), jobs=jobs['jobs']))
            return job
        previous = ['--dependency=afterany:' + ':'.join(prior)] if prior else []
        if stage == 'pilot':
            smoke = submit('gpu_smoke', 'cluster/jean_zay/node_first_gpu.slurm',
                           ['--time=00:40:00', *previous], ['smoke'])
            cpu = submit('cpu_smoke', 'cluster/jean_zay/node_first_cpu.slurm',
                         ['--dependency=afterok:' + smoke, '--kill-on-invalid-dep=yes', '--time=00:40:00'], ['smoke'])
            previous = ['--dependency=afterok:' + cpu, '--kill-on-invalid-dep=yes']
        count = 2 * len(p['stages'][stage]['seeds'])
        training = submit('training', 'cluster/jean_zay/node_first_gpu.slurm',
                          [f'--array=0-{count-1}%2', '--time=' + gpu_time, *previous],
                          ['train', '--stage', stage])
        evaluation = submit('evaluation', 'cluster/jean_zay/node_first_cpu.slurm',
                            [f'--array=0-{count-1}%2', '--dependency=aftercorr:' + training,
                             '--kill-on-invalid-dep=yes'], ['arm', '--stage', stage])
        collection_args = ['collect', '--stage', stage]
        if stage == 'pilot':
            collection_args += ['--auto-promote']
        collection = submit('collector', 'cluster/jean_zay/node_first_cpu.slurm',
                            ['--dependency=afterok:' + evaluation, '--kill-on-invalid-dep=yes'], collection_args)
        jobs['all_accepted'] = True
        write(record, jobs)
        write(claim, dict(owner='L08 scratch node-first curriculum', stage=stage, state='accepted',
                          concurrent_h100=2, global_limit=2, worktree=str(ROOT),
                          release_after=collection, jobs=jobs['jobs'],
                          note='Two single-GPU training arms maximum; all dependencies recorded; no polling.'))
    # Fresh-read append only; preserve other agents' ongoing research edits.
    marker = f'<!-- L08 SUBMISSION {stage} 20261028 -->'
    text = f'\n\n{marker}\n### L08 {stage} accepted —28October2026\n\n'
    text += ('Same random initialization, modern B0,120object/2relation tokens, focal-node gamma2/equal weights and matched CE ratio0.15. '
             f'{jobs["updates"]} matched updates;GPU wall{gpu_time} per arm; seeds{jobs["seeds"]}. '
             'Jobs: `' + ' → '.join(k + ':' + v['id'] for k, v in jobs['jobs'].items()) + '`. '
             f'Shared H100 ceiling2; this array%2 waits for preceding GPU jobs{prior}. '
             'No scientific result claimed before evaluation. Full100epoch/two-seed confirmation is automatic only after the predefined positive pilot gate; negative/inconclusive does not expand. '
             f'Retrieval: `python3 {ROOT}/scripts/show_node_first_curriculum.py`.\n')
    with (META / 'publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for document in [MAIN / 'docs/research/fine_feature_architecture_study.md',
                         MAIN / 'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md']:
            before = document.read_text()
            assert marker not in before
            after = before + text
            if 'REGISTRY' in document.name:
                lines = after.splitlines()
                for index, line in enumerate(lines):
                    if line.startswith('| **L08** |'):
                        lines[index] = f'| **L08** | Scratch B0 joint versus node-first edge curriculum | **{stage.upper()} SUBMITTED — verify** | Same random weights; node-only10%, cosine edge ramp10–20%, full joint thereafter. Matched120query B0 focal-node/edge-CE; no pretrained detector. | [Scratch study](../docs/research/fine_feature_architecture_study.md); `{record}` |'
                after = '\n'.join(lines) + '\n'
            assert document.read_text() == before, 'Concurrent document edit; job records already preserved'
            document.write_text(after)
    print(json.dumps(jobs, indent=2))


if __name__ == '__main__':
    main()
