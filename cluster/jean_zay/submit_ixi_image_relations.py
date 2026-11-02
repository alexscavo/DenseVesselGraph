#!/usr/bin/env python3
"""Create the fixed expanded manifest and submit one bounded image-adapter batch."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

SEED = 201020


def balanced_selection(candidates, preserved, count, sites):
    assert set(preserved) <= set(candidates)
    rank = lambda n: hashlib.sha256((str(SEED)+n).encode()).hexdigest()
    groups = defaultdict(list)
    for name in candidates:
        if name not in set(preserved): groups[name.rsplit('_', 1)[0]].append(name)
    for subject in groups: groups[subject].sort(key=rank, reverse=True)
    selected = list(preserved); per_subject = Counter(n.rsplit('_', 1)[0] for n in selected); per_site = Counter(sites[n] for n in selected)
    subjects_per_site = Counter(sites[v[0]] for v in groups.values() if v)
    while len(selected) < count:
        available = [s for s, v in groups.items() if v]
        assert available, 'Not enough official patches'
        subject = min(available, key=lambda s: (per_subject[s], per_site[sites[groups[s][-1]]]/subjects_per_site[sites[groups[s][-1]]], rank(s)))
        name = groups[subject].pop(); selected.append(name); per_subject[subject] += 1; per_site[sites[name]] += 1
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--submit', action='store_true'); args = parser.parse_args()
    directory = Path('experiments/ixi_image_relation_learning'); record = directory/'submission.json'
    if record.exists(): raise FileExistsError(f'Existing exact experiment: {record}; no duplicate batch submitted')
    previous = json.loads(Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_query_diagnostics_20261019/pilot-1291329-local_image/manifest.json').read_text())
    dataset = Path(previous['dataset']); official = json.loads((dataset/'manifest.json').read_text())
    generation = json.loads((dataset/'generation_config.json').read_text())
    with (Path(generation['source_root'])/'source_manifest.csv').open() as f: sources = {r['patient_id']: r for r in csv.DictReader(f)}
    candidates = dict(train=official['train_ids'], val=official['validation_ids'])
    assert not {n.rsplit('_', 1)[0] for n in candidates['train']} & {n.rsplit('_', 1)[0] for n in candidates['val']}
    sites = {n: next((s for s in ['HH', 'Guys', 'IOP'] if '-'+s in sources[n.split('_')[1]]['subject']), 'unknown') for names in candidates.values() for n in names}
    ids = {s: balanced_selection(names, previous['ids'][s], 2048 if s == 'train' else 512, sites) for s, names in candidates.items()}
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root = '/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_image_relation_learning_'+stamp
    plan = dict(output_root=root, ids=ids, sites={n: sites[n] for names in ids.values() for n in names},
        subjects={s: len({n.rsplit('_', 1)[0] for n in names}) for s, names in ids.items()},
        site_counts={s: dict(Counter(sites[n] for n in names)) for s, names in ids.items()},
        previous_ids=previous['ids'], checkpoint=previous['checkpoint'], checkpoint_sha256=previous['checkpoint_sha256'],
        adapter_sha256=previous['adapter_sha256'], dataset=str(dataset), dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
        seed=SEED, batch_size=8, learning_rate=7e-5, arms=['token_geometry', 'endpoints', 'chord', 'corridor'],
        image_projection=8, endpoint_offsets_vox=[-4, 0, 4], longitudinal_samples=16, corridor_offsets_mm=[-4, 0, 4],
        benchmark_updates_per_arm=200, benchmark_warmup=8, training_target_seconds=3300, throughput_safety_factor=1.15,
        budget_rule='floor(3300 / (1.15 * slowest_seconds_per_update) / 100) * 100, maximum 100000; require at least 2000',
        checkpoints=[1000, 2000, 'common_final'], training_walltime='01:30:00',
        estimated_shared_logical_bytes=2560*(64*32**3*4+258*552*4+32640*2*4),
        estimate_4096_1024_bytes=5120*(64*32**3*4+258*552*4+32640*2*4), larger_design_submitted=False)
    sources_files = ['models/image_relation_adapter.py','scripts/ixi_image_relation_learning.py','scripts/evaluate_ixi_image_relations.py',
                     'scripts/collect_ixi_image_relations.py','tests/test_image_relation_adapter.py',
                     'cluster/jean_zay/ixi_image_relations_gpu.slurm','cluster/jean_zay/ixi_image_relations_cpu.slurm']
    state = dict(output_root=root, jobs={}, plan=plan, source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources_files})
    if args.submit:
        directory.mkdir(parents=True, exist_ok=True); (directory/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    def submit(name, script, mode, dependency=None, array=None, time='01:30:00'):
        cmd = ['sbatch','--parsable','--kill-on-invalid-dep=yes','--time='+time]
        if dependency: cmd += ['--dependency='+dependency]
        if array: cmd += ['--array='+array]
        cmd += ['cluster/jean_zay/'+script, mode, root]
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0] if args.submit else str(9000000+len(state['jobs']))
        assert job.isdigit()
        state['jobs'][name] = dict(id=job, dependency=dependency, array=array, command=cmd)
        if args.submit: record.write_text(json.dumps(state, indent=2)+'\n')
        print(name, job, ' '.join(cmd), flush=True); return job
    gpu = 'ixi_image_relations_gpu.slurm'; cpu = 'ixi_image_relations_cpu.slurm'
    gs = submit('gpu_smoke', gpu, 'smoke', time='00:30:00')
    cs = submit('cpu_smoke_collection', cpu, 'smoke', 'afterok:'+gs, time='00:30:00')
    cache = submit('shared_cache', gpu, 'prepare', 'afterok:'+cs)
    bench = submit('throughput_benchmark', gpu, 'benchmark', 'afterok:'+cache, time='00:30:00')
    training = submit('training', gpu, 'train', 'afterok:'+bench, '0-3%2')
    baseline = submit('original_evaluation', cpu, 'baseline', 'afterok:'+cache, time='04:00:00')
    evaluation = submit('adapter_evaluation', cpu, 'evaluate', 'aftercorr:'+training, '0-3%2', time='04:00:00')
    submit('collection', cpu, 'collect', f'afterok:{baseline}:{evaluation}', time='02:00:00')
    if args.submit:
        state['submitted'] = True; record.write_text(json.dumps(state, indent=2)+'\n')
        with Path('docs/research/metrics/9-ixi-image-aware-relation-learning.md').open('a') as f:
            f.write('\n## Accepted submission\n\nOutput root: `'+root+'`.\n\n|Stage|ID|Dependency|\n|---|---|---|\n')
            for name,j in state['jobs'].items(): f.write(f'|{name}|{j["id"]}|{j["dependency"] or "none"}|\n')
            f.write('\nActual common updates will be written by the queued benchmark to `benchmark/budget.json`; no training begins until it succeeds.\n')
    print('Manifest counts', {s: len(v) for s,v in ids.items()}, 'subjects',plan['subjects'])
    print('Retrieve: python3 scripts/collect_ixi_image_relations.py --show')


if __name__ == '__main__': main()
