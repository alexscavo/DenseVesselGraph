#!/usr/bin/env python3
"""Evaluate head-only D scores on CPUs, or collect the complete bounded batch."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate_relation(source, output):
    import numpy as np
    import torch
    from scripts.ixi_cached_mechanisms import read_cache, heads, target, decode, measure, summarize_run
    from scripts.ixi_cached_mechanisms import manifest
    torch.set_num_threads(1)
    training = json.loads((source/'summary.json').read_text())
    assert training['completed']
    output.mkdir(parents=True, exist_ok=False)
    _, spline = heads(); rows, strata = [], []
    for split, names in training['ids'].items():
        for name in names:
            c = read_cache('local_image', split, name)
            with np.load(source/split/(name+'.npz')) as f:
                c['pair_logits'] = f['pair_logits']
            c['edge_scores'] = torch.tensor(c['pair_logits']).softmax(-1)[:, 1].numpy()
            threshold = manifest()['config']['evaluation']['edge_threshold']
            c['edge_selected'] = c['pair_logits'].argmax(-1) == 1 if threshold is None else c['edge_scores'] > threshold
            graph = decode(c, spline); path, gt, mask = target(split, name)
            for tolerance in (1., 2.):
                r, lengths = measure(c, graph, path, gt, mask, tolerance)
                key = dict(split=split, sample_id=name, subject=name.rsplit('_', 1)[0], tolerance_mm=tolerance, policy=training['arm'])
                rows.append(dict(**key, **r)); strata.extend(dict(**key, **v) for v in lengths)
    summarize_run(output, rows, strata, dict(experiment='D', training=training))


def collect(root):
    import csv
    import numpy as np
    from scripts.ixi_cached_mechanisms import aggregate, save, table, CACHES
    from scripts.train_ixi_relation_mechanisms import ARMS
    directories = {k: root/k for k in ['A', 'B', 'C']}
    directories.update({'D_'+a: root/'D-eval'/a for a in ARMS})
    missing = [str(p/'summary.json') for p in directories.values() if not (p/'summary.json').exists()]
    if missing:
        raise SystemExit('Results not complete; no waiting. Missing:\n'+'\n'.join(missing))
    summaries = {k: json.loads((p/'summary.json').read_text()) for k, p in directories.items()}
    assert all(s['completed'] for s in summaries.values())
    d = [summaries['D_'+a]['training'] for a in ARMS]
    assert len({s['initial_head_sha256'] for s in d}) == 1
    assert len({s['batch_order_sha256'] for s in d}) == 1
    assert all(s['ids'] == summaries['A']['ids'] == summaries['B']['ids'] == summaries['C']['ids'] for s in d)
    rows = []
    for key, s in summaries.items():
        rows.extend(dict(experiment=key, **r) for r in s.get('comparison', []) if r['split'] == 'val')
    table(root/'comparison.csv', rows)
    intervals = []
    for experiment, baseline in [('A', 'normal'), ('C', 'neither_local_residual'), ('D', 'regularized')]:
        if experiment == 'D':
            rr = sum([json.loads((root/'D-eval'/a/'per-patch.json').read_text()) for a in ARMS], [])
        else:
            rr = json.loads((root/experiment/'per-patch.json').read_text())
        for tolerance in (1., 2.):
            selected = [r for r in rr if r['split'] == 'val' and r['tolerance_mm'] == tolerance]
            policies = sorted({r['policy'] for r in selected}); subjects = sorted({r['subject'] for r in selected})
            lookup = {(p, s): [r for r in selected if r['policy'] == p and r['subject'] == s] for p in policies for s in subjects}
            # Resample subjects, retain all their patches; paired across policies.
            draws = np.random.default_rng(191032).integers(len(subjects), size=(1000, len(subjects)))
            for policy in policies:
                if policy == baseline:
                    continue
                delta = {k: [] for k in ['f1', 'node_recall', 'common_branch_f1', 'common_curve_f1']}
                for draw in draws:
                    before = aggregate(sum([lookup[baseline, subjects[i]] for i in draw], []))
                    after = aggregate(sum([lookup[policy, subjects[i]] for i in draw], []))
                    for k in delta:
                        delta[k].append(after[k]-before[k])
                intervals.append(dict(experiment=experiment, tolerance_mm=tolerance, baseline=baseline, policy=policy,
                    subjects=len(subjects), resamples=1000, delta_95pct={k: np.quantile(v, [.025, .975]).tolist() for k, v in delta.items()}))
    # Same frozen training-defined arc-length bins for all graph experiments.
    with (root/'A'/'positive-lengths.csv').open() as f:
        lengths = [float(r['length_mm']) for r in csv.DictReader(f) if r['split'] == 'train' and r['policy'] == 'normal' and float(r['tolerance_mm']) == 2.]
    cuts = np.quantile(lengths, [.25, .5, .75]); length_rows = []
    for experiment, directory in directories.items():
        if experiment == 'B':
            continue
        with (directory/'positive-lengths.csv').open() as f:
            records = list(csv.DictReader(f))
        for split, tol, policy in sorted({(r['split'], r['tolerance_mm'], r['policy']) for r in records}):
            for quartile in range(4):
                group = [r for r in records if (r['split'], r['tolerance_mm'], r['policy']) == (split, tol, policy)
                         and np.searchsorted(cuts, float(r['length_mm'])) == quartile]
                length_rows.append(dict(experiment=experiment, split=split, tolerance_mm=tol, policy=policy, quartile=quartile+1,
                    targets=len(group), recovered=sum(r['recovered'] == 'True' for r in group), correct=sum(r['selected'] == 'True' for r in group)))
    table(root/'length-strata.csv', length_rows)
    save(root/'bootstrap.json', intervals)
    save(root/'summary.json', dict(completed=True, results=summaries, bootstrap=intervals, arc_length_quartiles_mm=cuts.tolist()))
    lines = ['# IXI bounded mechanism diagnostics', '', 'Validation pilot; 1/2 mm positional matching. Common-v2 geometry unchanged.', '',
             '|Experiment|Policy|Tolerance|Node recall|Node precision|Edge F1|Conditional recall|Branch F1|Curve F1|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values = ['NA' if r[k] is None else f'{r[k]:.4f}' for k in ['node_recall', 'node_precision', 'f1', 'conditional_recall', 'common_branch_f1', 'common_curve_f1']]
        lines.append(f"|{r['experiment']}|{r['policy']}|{r['tolerance_mm']}|"+'|'.join(values)+'|')
    lines += ['', 'B: confidence.json, discrimination.json, recovery.csv, probes.json and per-subject files under B/.',
              'Paired intervals: bootstrap.json. Long-branch recovery: length-strata.csv. See each experiment for full counts.',
              'D regularized versus hard arms isolates auxiliary objectives. Focal control changes the base objective and is a separate comparison.',
              'Oracle selection uses GT. Reused validation, one seed, cached training subset; no test-set claim.']
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    print(root/'results.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path)
    p.add_argument('--relation-source', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--submission', type=Path, default=Path('experiments/ixi_mechanisms/submission.json'))
    p.add_argument('--show', action='store_true', help='Print existing collected results without importing scientific packages or waiting.')
    args = p.parse_args()
    if args.relation_source:
        evaluate_relation(args.relation_source, args.output)
    else:
        root = args.root or Path(json.loads(args.submission.read_text())['output_root'])
        if args.show:
            result = root/'results.md'
            if not result.exists():
                raise SystemExit(f'Collection not complete; no waiting. Expected {result}')
            print(result.read_text())
        else:
            collect(root)
