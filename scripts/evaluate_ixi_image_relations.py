#!/usr/bin/env python3
"""CPU evaluation of frozen-node image-relation adapters and calibration."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from scripts.ixi_cached_mechanisms import heads, measure, summarize_run, save, table
from scripts.ixi_relation_matching_calibration import Geometry, select_threshold
from models.image_relation_adapter import ARMS, connection_points


def probability(margin):
    return 1./(1.+np.exp(-np.clip(margin, -700, 700)))


def ranking(labels, scores, mask):
    y, p = labels[mask], scores[mask]
    return dict(candidates=int(mask.sum()), positives=int(y.sum()), prevalence=float(y.mean()) if len(y) else None,
                ap=float(average_precision_score(y, p)) if y.any() else None)


def visualize(path, cache, record, gt, pairs, scores, labels, selected, arm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    metric = gt['metric']; physical = cache['nodes'] @ metric.T
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    points = gt['full_paths'] @ metric.T
    for a, b in zip(gt['path_offsets'][:-1], gt['path_offsets'][1:]):
        curve = points[a:b]; ax.plot(*curve.T, color='gray', alpha=.4, linewidth=1)
    for mask, color, label in [(selected & labels, 'green', 'TP'), (selected & ~labels, 'red', 'FP'), (~selected & labels, 'blue', 'FN')]:
        ids = np.flatnonzero(mask)[:200]
        for k, i in enumerate(ids):
            ax.plot(*physical[pairs[i]].T, color=color, alpha=.5, label=label if k == 0 else None)
    chosen = np.flatnonzero(labels)
    samples = np.empty((0, 3))
    if len(chosen) and arm != 'original' and arm != 'token_geometry':
        pair = pairs[chosen[0:1]]
        offsets = np.array(np.meshgrid(*[[-4., 0., 4.]]*3, indexing='ij')).reshape(3, -1).T
        samples = (cache['nodes'][pair].reshape(-1, 1, 3)+offsets[None]/cache['shape']).reshape(-1, 3)
        if arm in ('chord', 'corridor'):
            corridor = connection_points(torch.tensor(cache['nodes']), torch.tensor(pair), torch.tensor(metric)).numpy()
            context = corridor[:, :, 4] if arm == 'chord' else corridor
            samples = np.concatenate([samples, context.reshape(-1, 3)])
        samples = samples @ metric.T
        ax.scatter(*samples.T, s=8, color='orange', label='Actual sampling region', alpha=.5)
    ax.scatter(*physical[cache['retained']].T, s=10, color='black', alpha=.6)
    ax.set(title=f'{record["sample_id"]} — {arm}\nFixed 0.5; 2 mm positional labels; GT paths uncertain', xlabel='Physical x (mm)', ylabel='y (mm)', zlabel='z (mm)')
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    save(path.with_suffix('.json'), dict(sample_id=record['sample_id'], arm=arm,
        displayed_tp=min(200, int((selected & labels).sum())), displayed_fp=min(200, int((selected & ~labels).sum())),
        displayed_fn=min(200, int((~selected & labels).sum())), sampling_reference_pair=pairs[chosen[0]].tolist() if len(chosen) else None,
        sampling_points_mm=samples.tolist(), caveat='Displayed edges are straight pair links; metric curves use the frozen spline head. Colors are positional diagnostic labels.'))


def degree_distribution(group):
    histogram = np.zeros(max(len(r['degree_histogram']) for r in group), dtype=int)
    for row in group:
        histogram[:len(row['degree_histogram'])] += np.asarray(row['degree_histogram'], dtype=np.int64)
    return dict(patches=len(group), mean_nodes=float(histogram.sum()/len(group)),
        mean_edges=float(np.dot(histogram, np.arange(len(histogram)))/(2*len(group))),
        degree_histogram=histogram.tolist())


def evaluate(root, arm, smoke=False):
    cache_dir = root/('smoke-cache' if smoke else 'cache')
    records = json.loads((cache_dir/'records.json').read_text())
    directory = root/('smoke-evaluation' if smoke else 'evaluation')/arm; directory.mkdir(parents=True, exist_ok=False)
    with np.load(cache_dir/'common-pairs.npz') as f:
        common = {k: f[k] for k in f.files}
    offsets = common['offsets']; original = common['original_margin']
    record_split = np.array([r['split'] for r in records])
    split = np.repeat(record_split, np.diff(offsets)); subjects = np.repeat([r['subject'] for r in records], np.diff(offsets))
    sites = np.repeat([r['site'] for r in records], np.diff(offsets))
    train = split == 'train'; val = split == 'val'
    plan = json.loads((root/'plan.json').read_text())
    old_val = set(plan['previous_ids']['val'])
    prior_mask = np.repeat([r['split'] == 'val' and r['sample_id'] in old_val for r in records], np.diff(offsets))
    intermediates = []
    if arm == 'original':
        margin = original; updates = 0
    else:
        run = root/('smoke-runs' if smoke else 'runs')/arm
        summary = json.loads((run/'summary.json').read_text()); assert summary['completed']
        updates = summary['updates']
        for step in summary['checkpoints']:
            with np.load(run/f'scores-{step:06d}.npz') as f:
                np.testing.assert_array_equal(f['offsets'], offsets)
                np.testing.assert_array_equal(f['names'], [r['sample_id'] for r in records])
                np.testing.assert_array_equal(f['splits'], record_split)
                np.testing.assert_array_equal(f['original_margin'], original)
                candidate = f['margin'].copy()
            for tol in (1, 2):
                for which, mask in [('all_retained', np.ones(len(train), bool)), ('conditional', common[f'conditional_{tol}'])]:
                    for ss in ['train', 'val']:
                        intermediates.append(dict(step=step, split=ss, tolerance_mm=tol, population=which,
                            **ranking(common[f'labels_{tol}'], candidate, mask & (split == ss))))
            if step == updates: margin = candidate
    scores = probability(margin); baseline = probability(original)
    total = {s: sum(len(r['gt_arc_lengths']) for r in records if r['split'] == s) for s in ['train', 'val']}
    thresholds = {}; ap = []; strata = []; diagnostic = []
    cuts = json.loads((cache_dir/'strata.json').read_text())
    for tol in (1, 2):
        y = common[f'labels_{tol}']; original_selected = train & (baseline > .5)
        target = float(y[original_selected].mean()) if original_selected.any() else 0.
        thresholds[str(tol)] = dict(fixed_0p5=.5, precision_target=target,
            **select_threshold(scores[train], y[train], total['train'], target))
        for ss in ['train', 'val']:
            for population, mask in [('all_retained', split == ss), ('conditional', (split == ss) & common[f'conditional_{tol}'])]:
                ap.append(dict(arm=arm, split=ss, tolerance_mm=tol, population=population, **ranking(y, margin, mask)))
                if y[mask].any():
                    p, r, t = precision_recall_curve(y[mask], margin[mask])
                    np.savez_compressed(directory/f'{ss}-{tol}-{population}-PR.npz', precision=p, recall=r, margin_threshold=t)
                for sid in sorted(set(subjects[mask])):
                    ap.append(dict(arm=arm, split=ss, tolerance_mm=tol, population=population, subject=sid,
                                   **ranking(y, margin, mask & (subjects == sid))))
                for site in sorted(set(sites[mask])):
                    ap.append(dict(arm=arm, split=ss, tolerance_mm=tol, population=population, site=site,
                                   **ranking(y, margin, mask & (sites == site))))
                if ss == 'val':
                    for cohort, cohort_mask in [('previous_pilot', prior_mask), ('additional_validation', val & ~prior_mask)]:
                        ap.append(dict(arm=arm, split=ss, tolerance_mm=tol, population=population, cohort=cohort,
                                       **ranking(y, margin, mask & cohort_mask)))
            for variable, values, boundaries in [('GT_arc_mm', common[f'arc_{tol}'], cuts['training_arc_quartiles_mm']),
                    ('GT_endpoint_distance_mm', common[f'chord_{tol}'], cuts['training_chord_quartiles_mm']),
                    ('minimum_endpoint_confidence', common['endpoint_confidence'], [.75, .9, .95])]:
                bins = np.searchsorted(boundaries, values)
                for b in range(4):
                    # Arc/chord ranking uses positives in the bin plus the same
                    # conditional negatives, never relabeling other positives.
                    positive_bin = y & (bins == b)
                    mask = (split == ss) & common[f'conditional_{tol}'] & (~y | positive_bin)
                    strata.append(dict(split=ss, tolerance_mm=tol, variable=variable, bin=b+1,
                        definition='positives in bin + common conditional negatives', **ranking(y, margin, mask)))
        # Validation-only equal-precision readout: descriptive, never a deployment cutoff.
        reference = val & (baseline > .5); precision_target = float(y[reference].mean()) if reference.any() else 0.
        posthoc = select_threshold(scores[val], y[val], total['val'], precision_target)
        selected = val & (scores > posthoc['train_preserve_precision'])
        tp = int(y[selected].sum()); fp = int(selected.sum())-tp
        diagnostic.append(dict(tolerance_mm=tol, validation_only=True, target_original_validation_precision=precision_target,
            precision=tp/(tp+fp) if tp+fp else None, full_gt_recall=tp/total['val'] if total['val'] else None,
            tp=tp, fp=fp, caveat='Post-hoc validation PR diagnostic; not a train-calibrated policy.'))
    save(directory/'thresholds.json', thresholds); table(directory/'ranking.csv', ap)
    table(directory/'ranking-strata.csv', strata); save(directory/'validation-only-precision.json', diagnostic)
    save(directory/'intermediate-ranking.json', intermediates)
    # Small score arrays are retained for independent bootstrap/review, not pair features.
    np.savez_compressed(directory/'ranking-scores.npz', margin=margin, offsets=offsets)
    _, spline = heads(); rows, lengths = [], []; visual_subjects = []
    for record in records:
        if record['split'] == 'val' and record['subject'] not in visual_subjects and len(visual_subjects) < 3:
            visual_subjects.append(record['subject'])
    visual_done = set()
    for index, record in enumerate(records):
        with np.load(record['cache']) as f: c = {k: f[k] for k in f.files}
        with np.load(record['gt']) as f: gt = {k: f[k] for k in f.files}
        feature = torch.tensor(np.load(record['features']))
        geometry = Geometry(c, spline, feature)
        mask_path = Path(record['gt']).parent.parent/'seg'/(record['sample_id']+'_seg.nii.gz')
        mask = np.asarray(nib.load(mask_path).dataobj) > .5
        q = c['retained']; pairs = torch.combinations(torch.tensor(q), r=2).numpy().reshape(-1, 2)
        from scripts.ixi_image_relation_learning import pair_indices
        full_index = pair_indices(torch.tensor(pairs)).numpy(); lo, hi = offsets[index:index+2]
        assert hi-lo == len(pairs)
        patch_scores = scores[lo:hi]; full_scores = np.zeros(len(c['edges']), float); full_scores[full_index] = patch_scores
        for tol in (1, 2):
            mapping = {int(k): int(v) for k, v in record[f'mapping_{tol}'].items()}
            for policy in ['fixed_0p5', 'train_max_f1', 'train_preserve_precision']:
                threshold = thresholds[str(tol)][policy]; selected = np.zeros(len(c['edges']), bool)
                selected[full_index] = patch_scores > threshold
                graph = geometry.graph(q, selected)
                cc = dict(c, edge_scores=full_scores, edge_selected=selected)
                row, ll = measure(cc, graph, Path(record['gt']), gt, mask, tol, mapping)
                # Degree distributions retain all isolated nodes.
                degree = np.bincount(graph['edges'].ravel(), minlength=len(q))
                row['degree_histogram'] = np.bincount(degree).tolist()
                key = dict(split=record['split'], sample_id=record['sample_id'], subject=record['subject'],
                    site=record['site'], tolerance_mm=tol, policy=policy, threshold=threshold)
                rows.append(dict(**key, **row)); lengths.extend(dict(**key, **v) for v in ll)
                if tol == 2 and policy == 'fixed_0p5' and record['subject'] in visual_subjects and record['subject'] not in visual_done:
                    visualize(directory/(record['sample_id']+'-3d.png'), c, record, gt, pairs, patch_scores,
                        common['labels_2'][lo:hi], patch_scores > .5, arm)
                    visual_done.add(record['subject'])
        if (index+1) % 64 == 0: print(arm, 'CPU graphs', index+1, '/', len(records), flush=True)
    summarize_run(directory, rows, lengths, dict(arm=arm, updates=updates, smoke=smoke, records=len(records),
        source=str(cache_dir), thresholds=thresholds, compared_node_predictions='identical frozen local refinement', completed_evaluation=True))
    finish_reporting(directory, rows, lengths, cuts)


def finish_reporting(directory, rows, lengths, cuts):
    # Site graph summaries, in addition to the helper's subject summaries.
    from scripts.ixi_cached_mechanisms import aggregate
    site_rows = []
    for s, t, p, site in sorted({(r['split'], r['tolerance_mm'], r['policy'], r['site']) for r in rows}):
        site_rows.append(dict(split=s, tolerance_mm=t, policy=p, site=site,
            **aggregate([r for r in rows if (r['split'], r['tolerance_mm'], r['policy'], r['site']) == (s, t, p, site)])))
    table(directory/'per-site.csv', site_rows)
    distributions = []
    geometry_rows = []
    for s, t, p in sorted({(r['split'], r['tolerance_mm'], r['policy']) for r in rows}):
        group = [r for r in rows if (r['split'], r['tolerance_mm'], r['policy']) == (s, t, p)]
        distributions.append(dict(split=s, tolerance_mm=t, policy=p, **degree_distribution(group)))
        ll = [r for r in lengths if (r['split'], r['tolerance_mm'], r['policy']) == (s, t, p)]
        for quartile in range(4):
            subset = [r for r in ll if np.searchsorted(cuts['training_arc_quartiles_mm'], r['length_mm']) == quartile]
            recovered = sum(r['recovered'] for r in subset); selected = sum(r['selected'] for r in subset)
            geometry_rows.append(dict(split=s, tolerance_mm=t, policy=p, training_arc_quartile=quartile+1,
                gt_edges=len(subset), recovered_endpoints=recovered, selected_true_edges=selected,
                full_gt_recall=selected/len(subset) if subset else None,
                conditional_recall=selected/recovered if recovered else None))
    save(directory/'degree-distributions.json', distributions)
    table(directory/'length-recall.csv', geometry_rows)

    from scripts.ixi_cached_mechanisms import sha
    save(directory/'reporting-complete.json', dict(completed=True, rows=len(rows),
        source_sha256={name:sha(directory/name) for name in ['per-patch.json','positive-lengths.csv','ranking-scores.npz']},
        reporting_only=True, model_inference_rerun=False))


def resume_reporting(root, arm, smoke=False):
    """Finish only the saved-table stage; fail if upstream artifacts are partial."""
    import csv
    from collections import Counter
    directory=root/('smoke-evaluation' if smoke else 'evaluation')/arm
    cache=root/('smoke-cache' if smoke else 'cache')
    summary=json.loads((directory/'summary.json').read_text())
    records=json.loads((cache/'records.json').read_text())
    assert summary['completed'] and summary['records']==len(records)
    rows=json.loads((directory/'per-patch.json').read_text())
    expected=Counter((r['split'],r['sample_id'],tol,policy) for r in records for tol in [1,2]
        for policy in ['fixed_0p5','train_max_f1','train_preserve_precision'])
    found=Counter((r['split'],r['sample_id'],r['tolerance_mm'],r['policy']) for r in rows)
    assert found==expected, 'Saved graph rows are incomplete or belong to another population'
    lengths=list(csv.DictReader((directory/'positive-lengths.csv').open()))
    for r in lengths:
        r['length_mm']=float(r['length_mm']);r['tolerance_mm']=int(r['tolerance_mm'])
        for key in ['recovered','selected']:
            assert r[key] in ['True','False'], ('Invalid saved boolean',key,r[key])
            r[key]=r[key]=='True'
    cuts=json.loads((cache/'strata.json').read_text())
    finish_reporting(directory,rows,lengths,cuts)
    print('Completed reporting from saved tables:',arm,'rows',len(rows),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=['original']+ARMS, required=True); p.add_argument('--smoke', action='store_true')
    p.add_argument('--finish-reporting',action='store_true',help='Resume only from completed per-patch tables; no model inference or metric recomputation')
    a = p.parse_args(); torch.set_num_threads(1)
    (resume_reporting if a.finish_reporting else evaluate)(a.root,a.arm,a.smoke)
