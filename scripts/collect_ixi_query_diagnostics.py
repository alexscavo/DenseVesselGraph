#!/usr/bin/env python3
"""CPU-only P0 summaries and saved-prediction reproduction gate; no inference."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.evaluate_vessel_core import records
from scripts.ixi_query_diagnostics import summarize, nms


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def collect(root):
    manifest = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    arm = manifest['arm']
    if arm == 'parent':
        previous = Path(manifest['checkpoint']).parent/f"best-full-validation/validation-{manifest['parent_step']:06d}/predictions.json"
    else:
        previous = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_node_refinement_20261018')/arm/'validation-001000/predictions.json'
    selected = set(manifest['ids']['val'])
    checked = set(); max_coordinate_error = 0.; max_score_error = 0.; max_physical_error = 0.
    changes = []
    for r in records(previous):
        name = r['source_sample_id']
        if name not in selected:
            continue
        with np.load(root/'val'/(name+'.npz')) as c:
            old = {q: i for i, q in enumerate(r['query_ids'])}
            common = sorted(set(old) & set(c['retained']))
            before_nodes = np.asarray(r['nodes_dhw']).reshape(-1, 3)
            delta = c['nodes'][common]-before_nodes[[old[q] for q in common]]
            displacement = delta @ c['metric'].T
            physical_error = float(np.linalg.norm(displacement, axis=1).max(initial=0.))
            max_physical_error = max(max_physical_error, physical_error)
            error = np.max(np.abs(delta), initial=0.)
            max_coordinate_error = max(max_coordinate_error, float(error))
            before = {tuple(sorted((r['query_ids'][a], r['query_ids'][b]))) for a, b in r['edges']}
            after = {tuple(sorted(map(int, e))) for e in c['decoded_query_edges']}
            pair_scores = {tuple(map(int, e)): s for e, s in zip(c['edges'], c['edge_scores'])}
            scores = np.array([pair_scores[tuple(sorted((r['query_ids'][a], r['query_ids'][b])))] for a, b in r['edges']])
            max_score_error = max(max_score_error, float(np.max(np.abs(scores-np.array(r['edge_scores'])), initial=0.)))
            query_changes = sorted(set(old) ^ set(c['retained']))
            if query_changes or before != after:
                changes.append(dict(sample_id=name,
                    queries=[dict(query=int(q), current_score=float(c['node_scores'][q]),
                                  old_score=r['node_scores'][old[q]] if q in old else None) for q in query_changes],
                    added_edges=[dict(queries=e, current_score=float(pair_scores[e])) for e in sorted(after-before)],
                    removed_edges=[dict(queries=e, current_score=float(pair_scores[e])) for e in sorted(before-after)]))
        checked.add(name)
    # Full-validation expansion can exceed the prior refinement pilot's 128 patches.
    if not manifest['full_validation']:
        assert checked == selected, 'Saved-prediction reproduction missing selected validation IDs'
    else:
        assert checked, 'No overlap with saved comparator'
    # The earlier threshold diagnostic used the same batch-1 forward protocol.
    # Require exact per-patch positional counts for that primary frozen baseline.
    same_batch_reference_cases = None
    if arm == 'local_image':
        with gzip.open('experiments/ixi_threshold_calibration/full-1279574/pair-graphs.json.gz', 'rt') as f:
            prior = json.load(f)
        current = {(r['split'], r['sample_id']): r for r in read(root/'per-patch.csv')
                   if r['policy'] == 'baseline' and float(r['tolerance_mm']) == 2.}
        same_batch_reference_cases = 0
        for split, rr in prior.items():
            for r in rr:
                if (split, r['id']) not in current:
                    continue
                chosen = np.array(r['score']) > .5
                positive = np.array(r['positive'], bool)
                tp = int(positive[chosen].sum()); fp = int(chosen.sum())-tp
                expected = dict(nodes=r['nodes'], tp=tp, fp=fp, fn=r['target_edges']-tp, recoverable_edges=r['recoverable'])
                assert all(int(current[(split, r['id'])][k]) == v for k, v in expected.items()), r['id']
                same_batch_reference_cases += 1
        if not manifest['full_validation']:
            assert same_batch_reference_cases == sum(map(len, manifest['ids'].values()))
    save(root/'reproduction.json', dict(source=str(previous), checked=len(checked), exact_query_ids_and_edge_sets=not changes,
         max_normalized_coordinate_error=max_coordinate_error, max_physical_coordinate_error_mm=max_physical_error,
         max_edge_probability_error=max_score_error,
         historical_decision_changes=changes, same_batch_reference_cases=same_batch_reference_cases,
         historical_batch_note='Batch 1 here versus 4/32 in historical evaluator. Report all differences; no claim of exact historical reproduction. Batch size is a plausible cause, not isolated here. Same-forward cache/decoder parity is checked separately; local_image also requires exact per-patch counts against the previous batch-1 threshold diagnostic.'))

    nodes = read(root/'nodes.csv'); gt = read(root/'gt-recovery.csv'); edges = read(root/'decoded-edges.csv')
    node_groups = defaultdict(list)
    for r in nodes:
        node_groups[(r['split'], r['tolerance_mm'], r['population'], r['category'])].append(r)
    categories = []
    for (split, tolerance, population, category), group in sorted(node_groups.items()):
        score = np.array([float(r['score']) for r in group])
        denominator = sum(len(v) for (s, t, p, _), v in node_groups.items() if (s, t, p) == (split, tolerance, population))
        categories.append(dict(split=split, tolerance_mm=float(tolerance), population=population, category=category,
            count=len(group), fraction=len(group)/denominator, confidence_quantiles=np.quantile(score, [.1, .5, .9]).tolist()))
    gt_groups = defaultdict(list)
    for r in gt:
        gt_groups[(r['split'], r['tolerance_mm'], r['type'])].append(r)
    recovery = []
    for (split, tolerance, kind), group in sorted(gt_groups.items()):
        recovery.append(dict(split=split, tolerance_mm=float(tolerance), type=kind, count=len(group),
            retained_recovered=sum(r['retained_recovered'] == 'True' for r in group),
            raw_recovered=sum(r['raw_recovered'] == 'True' for r in group),
            anchors_with_two_nearby_retained=sum(int(r['nearby_retained']) >= 2 for r in group)))
    node_index = {(r['split'], r['sample_id'], r['tolerance_mm'], r['query']): r for r in nodes if r['population'] == 'retained'}
    subtypes = Counter()
    for r in edges:
        a = node_index[(r['split'], r['sample_id'], r['tolerance_mm'], r['query_a'])]['category']
        b = node_index[(r['split'], r['sample_id'], r['tolerance_mm'], r['query_b'])]['category']
        subtypes[(r['split'], r['tolerance_mm'], r['endpoint_type'], r['tp'], *sorted((a, b)))] += 1
    subtype_rows = [dict(split=k[0], tolerance_mm=float(k[1]), endpoint_type=k[2], tp=k[3] == 'True',
                         category_a=k[4], category_b=k[5], edges=v) for k, v in sorted(subtypes.items())]
    save(root/'node-summary.json', dict(categories=categories, gt_recovery=recovery))
    save(root/'edge-subtypes.json', subtype_rows)

    dataset = Path(manifest['dataset'])
    generation = json.loads((dataset/'generation_config.json').read_text())
    source_rows = read(Path(generation['source_root'])/'source_manifest.csv')
    sources = {r['patient_id']: r for r in source_rows}
    patch_rows = read(root/'per-patch.csv')
    by_site = defaultdict(list)
    for r in patch_rows:
        patient = r['subject'].rsplit('_', 1)[1]
        source = sources[patient]
        site = next((s for s in ('HH', 'Guys', 'IOP') if '-'+s in source['subject']), 'unknown')
        numbers = {k: float(v) for k, v in r.items() if k not in ('split', 'sample_id', 'subject', 'policy', 'degree_histogram')}
        by_site[(r['split'], r['tolerance_mm'], r['policy'], site)].append(numbers)
    save(root/'site-summary.json', [dict(split=k[0], tolerance_mm=float(k[1]), policy=k[2], site=k[3], **summarize(v))
                                   for k, v in sorted(by_site.items())])

    # Fixed training-derived arc-length quartiles and a 5 mm near-nonedge diagnostic.
    cuts = json.loads((dataset/'monitoring.json').read_text())['branch_bins']['gt_arc_quartile_cuts_mm']
    strata = {}
    for split, names in manifest['ids'].items():
        bins = [dict(targets=0, recovered_endpoints=0, direct=0) for _ in range(4)]
        near_total = near_selected = 0
        for name in names:
            with np.load(root/split/(name+'.npz')) as c, np.load(dataset/split/'spline'/(name+'.npz')) as g:
                mapping = {int(r['query']): int(r['matched_gt']) for key, r in node_index.items()
                           if key[:3] == (split, name, '2.0') and r['matched_gt'] != ''}
                inverse = {j: i for i, j in mapping.items()}
                lookup = {tuple(map(int, e)): bool(s) for e, s in zip(c['edges'], c['edge_selected'])}
                gt_links = {tuple(sorted(map(int, e))) for e in g['edges']}
                paths = g['full_paths'] @ g['metric'].T
                for (a, b), lo, hi in zip(g['edges'], g['path_offsets'][:-1], g['path_offsets'][1:]):
                    length = np.linalg.norm(np.diff(paths[lo:hi], axis=0), axis=1).sum()
                    row = bins[int(np.searchsorted(cuts, length, side='right'))]
                    row['targets'] += 1
                    if int(a) in inverse and int(b) in inverse:
                        row['recovered_endpoints'] += 1
                        row['direct'] += int(lookup[tuple(sorted((inverse[int(a)], inverse[int(b)])))])
                points = g['nodes'] @ g['metric'].T
                for offset, a in enumerate(sorted(mapping)):
                    for b in sorted(mapping)[offset+1:]:
                        u, v = mapping[a], mapping[b]
                        if tuple(sorted((u, v))) not in gt_links and np.linalg.norm(points[u]-points[v]) < 5.:
                            near_total += 1; near_selected += int(lookup[(a, b)])
        strata[split] = dict(arc_quartiles=bins, near_nonedge_candidates=near_total, near_nonedge_selected=near_selected,
                            near_nonedge_fpr=near_selected/near_total if near_total else None)
    save(root/'relation-strata.json', dict(arc_cuts_mm=cuts, matching_tolerance_mm=2., near_distance_mm=5.,
                                         cuts_source=str(dataset/'monitoring.json'), splits=strata))

    # Paired subject bootstrap: resample subjects, then pool each subject's patch counts.
    subjects = read(root/'per-subject.csv')
    val = [r for r in subjects if r['split'] == 'val' and float(r['tolerance_mm']) == 2.]
    subject_ids = sorted({r['subject'] for r in val})
    values = {(r['policy'], r['subject']): np.array([float(r[k]) for k in ('tp', 'fp', 'fn')]) for r in val}
    rng = np.random.default_rng(191031)
    draws = rng.integers(len(subject_ids), size=(1000, len(subject_ids)))
    f1 = lambda x: np.divide(2*x[..., 0], 2*x[..., 0]+x[..., 1]+x[..., 2], out=np.zeros(x.shape[:-1]), where=(2*x[..., 0]+x[..., 1]+x[..., 2]) > 0)
    baseline = np.stack([values[('baseline', s)] for s in subject_ids])
    boot = []
    for policy in ('oracle', 'nms_0p5mm', 'nms_1mm'):
        candidate = np.stack([values[(policy, s)] for s in subject_ids])
        delta = f1(candidate[draws].sum(1))-f1(baseline[draws].sum(1))
        boot.append(dict(policy=policy, subjects=len(subject_ids), resamples=1000,
                         paired_edge_f1_delta_95pct=np.quantile(delta, [.025, .975]).tolist()))
    save(root/'subject-bootstrap.json', boot)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    group = [r for r in categories if r['split'] == 'val' and r['tolerance_mm'] == 2. and r['population'] == 'retained']
    axes[0].barh([r['category'].replace('_', ' ') for r in group], [r['count'] for r in group])
    axes[0].set_title('Retained-query categories (2 mm)')
    group = [r for r in summary['comparison'] if r['split'] == 'val' and r['tolerance_mm'] == 2.]
    x = np.arange(len(group))
    axes[1].bar(x, [r['f1'] or 0 for r in group]); axes[1].set_xticks(x, [r['policy'] for r in group], rotation=30)
    axes[1].set_title('Positional edge F1')
    axes[2].bar(x-.18, [r['beta0_mae'] for r in group], .36, label='β0 MAE')
    axes[2].bar(x+.18, [r['beta1_mae'] for r in group], .36, label='β1 MAE')
    axes[2].set_xticks(x, [r['policy'] for r in group], rotation=30); axes[2].legend()
    fig.suptitle(arm+' — frozen checkpoint; oracle uses GT; validation development subset')
    fig.tight_layout(); fig.savefig(root/'overview.png', dpi=140); plt.close(fig)
    for name in manifest['ids']['val'][:3]:
        with np.load(root/'val'/(name+'.npz')) as c, np.load(dataset/'val/spline'/(name+'.npz')) as g:
            mapping = {int(r['query']): int(r['matched_gt']) for key, r in node_index.items()
                       if key[:3] == ('val', name, '2.0') and r['matched_gt'] != ''}
            physical = c['nodes'] @ c['metric'].T
            policies = dict(baseline=c['retained'], oracle=np.array(sorted(mapping), int),
                            nms_1mm=nms(physical, c['node_scores'], c['retained'], 1.))
            fig = plt.figure(figsize=(15, 5))
            paths = g['full_paths'] @ g['metric'].T
            for index, (policy, ids) in enumerate(policies.items(), 1):
                ax = fig.add_subplot(1, 3, index, projection='3d')
                for lo, hi in zip(g['path_offsets'][:-1], g['path_offsets'][1:]):
                    ax.plot(*paths[lo:hi].T, color='gray', alpha=.5, linewidth=1)
                for edge, curve in zip(c['decoded_query_edges'], c['decoded_curves']):
                    if not np.isin(edge, ids).all():
                        continue
                    count = sum(int(q) in mapping for q in edge)
                    ax.plot(*(curve @ c['metric'].T).T, color={2:'#168a45', 1:'#e69100', 0:'#cf2457'}[count], linewidth=2)
                ax.scatter(*physical[ids].T, s=10, color='blue')
                # Shared patch-local physical frame and camera across policies.
                corners = np.array([[a, b, d] for a in (0, 1) for b in (0, 1) for d in (0, 1)]) @ c['metric'].T
                for axis, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                    setter(corners[:, axis].min(), corners[:, axis].max())
                ax.set_box_aspect(np.ptp(corners, axis=0)); ax.view_init(elev=25, azim=35)
                ax.set_title(policy); ax.set_xlabel('mm'); ax.set_ylabel('mm'); ax.set_zlabel('mm')
            fig.suptitle(name+' — gray GT; green M–M; orange M–U; red U–U (baseline assignment)')
            fig.tight_layout(); fig.savefig(root/(name+'-3d.png'), dpi=140); plt.close(fig)
    print(json.dumps(dict(arm=arm, reproduced=len(checked), bootstrap=boot)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    collect(p.parse_args().root)
