#!/usr/bin/env python3
"""Collect CPU matching/calibration results, or print an existing report immediately."""
import argparse
import json
from pathlib import Path


def collect(root, smoke=False):
    import csv
    import hashlib
    import shutil
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from scripts.ixi_relation_matching_calibration import ARMS, CONDITIONS, SEED
    from scripts.ixi_cached_mechanisms import aggregate, save, table

    prepared = root/'prepared'; manifest = json.loads((prepared/'manifest.json').read_text())
    thresholds = json.loads((prepared/'thresholds.json').read_text())
    summaries, rows, ranks, rank_rows, graph_rows, strata_rows = {}, {}, {}, [], [], []
    subjects = sorted({n.rsplit('_', 1)[0] for n in manifest['ids']['val']})
    subject_index = {s: i for i, s in enumerate(subjects)}
    resamples = 40 if smoke else 1000
    draws = np.random.default_rng(SEED).integers(len(subjects), size=(resamples, len(subjects)))
    weights = np.array([np.bincount(draw, minlength=len(subjects)) for draw in draws])
    for experiment in ['A']+ARMS:
        output = root/'analyses'/experiment
        summaries[experiment] = json.loads((output/'summary.json').read_text())
        assert summaries[experiment]['completed'] and summaries[experiment]['ids'] == manifest['ids']
        rows[experiment] = json.loads((output/'per-patch.json').read_text())
        graph_rows.extend(dict(experiment=experiment, **r) for r in summaries[experiment]['comparison'])
        with (output/'ranking.csv').open() as f:
            rank_rows.extend(dict(experiment=experiment, **r) for r in csv.DictReader(f))
        with (output/'positive-lengths.csv').open() as f:
            strata_rows.extend(dict(experiment=experiment, **r) for r in csv.DictReader(f))
    table(root/'comparison.csv', graph_rows); table(root/'ranking.csv', rank_rows)

    def load_rank(experiment, condition, tolerance, population, split='val'):
        key = (experiment, condition, tolerance, population, split)
        if key not in ranks:
            path = root/'analyses'/experiment/'ranking'/f'{split}-{condition}-{tolerance}-{population}.npz'
            with np.load(path) as f:
                ranks[key] = {k: f[k] for k in f.files}
        return ranks[key]

    # Verify exact eligible populations and scores, not only candidate counts.
    for split in ['train', 'val']:
        for tolerance in (1, 2):
            for population in ['all_selected', 'conditional']:
                reference = load_rank('A', 'normal', tolerance, population, split)
                for i, arm in enumerate(ARMS):
                    current = load_rank(arm, arm, tolerance, population, split)
                    np.testing.assert_array_equal(reference['labels'], current['labels'])
                    np.testing.assert_array_equal(reference['subjects'], current['subjects'])
                    np.testing.assert_array_equal(reference['scores'][:, i], current['scores'][:, 0])
    base_rows = {(r['split'], r['sample_id'], r['tolerance_mm']): r for r in rows['A'] if r['policy'] == 'normal'}
    for r in rows['original']:
        if r['policy'] == 'fixed_0p5':
            before = base_rows[r['split'], r['sample_id'], r['tolerance_mm']]
            for key in ['nodes', 'node_tp', 'tp', 'fp', 'fn', 'beta0', 'beta1', 'common_curve_pred_samples', 'common_curve_supported_pred_samples', 'common_curve_supported_gt_samples']:
                assert r[key] == before[key], ('Original decoder parity', key, r['sample_id'])

    def ap_resamples(data, column):
        # Sort once. Subject multiplicities weight complete score-tie groups.
        score = data['scores'][:, column]; y = data['labels']
        if not len(y):
            return np.full(resamples, np.nan)
        order = np.argsort(-score, kind='stable'); p = score[order]; y = y[order]
        subject = np.array([subject_index[s] for s in data['subjects'][order]])
        end = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p)-1]
        result = []
        for multiplicity in weights:
            w = multiplicity[subject]; tp = np.cumsum(w*y)[end]; count = np.cumsum(w)[end]
            positives = tp[-1]
            precision = np.divide(tp, count, out=np.zeros(len(tp), float), where=count > 0)
            result.append(float(np.sum(np.diff(np.r_[0, tp])*precision)/positives) if positives else np.nan)
        return np.array(result)

    from scripts.ixi_relation_matching_calibration import weighted_ap
    from sklearn.metrics import average_precision_score
    # Check tie-weight semantics against sklearn on an actual subject resample.
    check = load_rank('original', 'original', 2, 'all_selected')
    w = weights[0][[subject_index[s] for s in check['subjects']]]
    if np.sum(w*check['labels']):
        expected = average_precision_score(check['labels'], check['scores'][:, 0], sample_weight=w)
        assert abs(weighted_ap(check['labels'], check['scores'][:, 0], w)-expected) < 1e-12

    ap_boot, ap_intervals = {}, []
    def interval(values):
        finite = values[np.isfinite(values)]
        return np.quantile(finite, [.025, .975]).tolist() if len(finite) else None
    for tolerance in (1, 2):
        for population in ['all_selected', 'conditional']:
            base = ap_resamples(load_rank('original', 'original', tolerance, population), 0)
            for arm in ARMS:
                data = load_rank(arm, arm, tolerance, population)
                result = ap_resamples(data, 0) if arm != 'original' else base
                ap_boot[arm, tolerance, population] = result
                ap_intervals.append(dict(experiment='B', arm=arm, tolerance_mm=tolerance, population=population,
                    subjects=len(subjects), resamples=resamples, ap_95pct=interval(result),
                    delta_vs_original_95pct=interval(result-base), usable_resamples=int(np.isfinite(result-base).sum())))
            for condition in CONDITIONS:
                data = load_rank('A', condition, tolerance, population)
                result = ap_resamples(data, 0)
                ap_intervals.append(dict(experiment='A', arm='original', condition=condition, tolerance_mm=tolerance,
                    population=population, subjects=len(subjects), resamples=resamples, ap_95pct=interval(result),
                    delta_vs_normal_95pct=interval(result-base), usable_resamples=int(np.isfinite(result-base).sum())))
    save(root/'ap-bootstrap.json', ap_intervals)

    fields = ['tp', 'fp', 'fn', 'node_tp', 'gt_nodes', 'common_curve_pred_samples', 'common_curve_gt_samples',
              'common_curve_supported_pred_samples', 'common_curve_supported_gt_samples', 'beta0_error', 'beta1_error']
    def graph_boot(experiment, policy, tolerance):
        rr = [r for r in rows[experiment] if r['split'] == 'val' and r['policy'] == policy and r['tolerance_mm'] == tolerance]
        matrix = np.array([[sum(r[f] for r in rr if r['subject'] == s) for f in fields]+[sum(r['subject'] == s for r in rr)] for s in subjects])
        summed = weights @ matrix
        tp, fp, fn, ntp, ngt, npred, ng, sp, sg, b0, b1, patches = summed.T
        precision = np.divide(sp, npred, out=np.ones(len(sp)), where=npred > 0)
        recall = np.divide(sg, ng, out=np.ones(len(sg)), where=ng > 0)
        return dict(f1=2*tp/(2*tp+fp+fn), node_recall=ntp/ngt,
                    common_curve_f1=np.divide(2*precision*recall, precision+recall, out=np.zeros(len(sp)), where=precision+recall > 0),
                    beta0_mae=b0/patches, beta1_mae=b1/patches)
    graph_intervals = []
    for tolerance in (1, 2):
        base = graph_boot('A', 'normal', tolerance)
        for condition in CONDITIONS[1:]:
            after = graph_boot('A', condition, tolerance)
            graph_intervals.append(dict(experiment='A', policy=condition, tolerance_mm=tolerance, baseline='normal',
                subjects=len(subjects), resamples=resamples, delta_95pct={k: interval(after[k]-base[k]) for k in base}))
        for policy in ['fixed_0p5', 'train_max_f1', 'train_preserve_precision']:
            base = graph_boot('original', policy, tolerance)
            for arm in ARMS[1:]:
                after = graph_boot(arm, policy, tolerance)
                graph_intervals.append(dict(experiment='B', arm=arm, policy=policy, tolerance_mm=tolerance, baseline='original same policy',
                    subjects=len(subjects), resamples=resamples, delta_95pct={k: interval(after[k]-base[k]) for k in base}))
    save(root/'graph-bootstrap.json', graph_intervals)

    cuts = json.loads((Path(manifest['previous'])/'summary.json').read_text())['arc_length_quartiles_mm']
    chord_train = [float(r['endpoint_distance_mm']) for r in strata_rows if r['experiment'] == 'original' and r['split'] == 'train' and r['policy'] == 'fixed_0p5' and int(r['tolerance_mm']) == 2]
    chord_cuts = np.quantile(chord_train, [.25, .5, .75]).tolist()
    save(root/'strata-definitions.json', dict(training_arc_length_quartiles_mm=cuts, training_gt_chord_quartiles_mm=chord_cuts))
    strata = []
    for key in sorted({(r['experiment'], r['split'], r['tolerance_mm'], r['policy']) for r in strata_rows}):
        rr = [r for r in strata_rows if (r['experiment'], r['split'], r['tolerance_mm'], r['policy']) == key]
        for measure_name in ['arc_length_mm', 'endpoint_distance_mm']:
            boundaries = cuts if measure_name == 'arc_length_mm' else chord_cuts
            for b in range(4):
                group = [r for r in rr if np.searchsorted(boundaries, float(r['length_mm' if measure_name == 'arc_length_mm' else 'endpoint_distance_mm'])) == b]
                strata.append(dict(experiment=key[0], split=key[1], tolerance_mm=key[2], policy=key[3], measure=measure_name, bin=b+1,
                    targets=len(group), recovered=sum(r['recovered'] == 'True' for r in group), correct=sum(r['selected'] == 'True' for r in group)))
    table(root/'length-distance-strata.csv', strata)

    with (prepared/'assignments.csv').open() as f:
        assignments = list(csv.DictReader(f))
    quality = []
    for split in ['train', 'val']:
        for outputs in ['refined', 'parent']:
            for subject in [None]+sorted({r['subject'] for r in assignments if r['split'] == split}):
                rr = [r for r in assignments if r['split'] == split and r['model_outputs'] == outputs and (subject is None or r['subject'] == subject)]
                distance = np.array([float(r['distance_mm']) for r in rr])
                quality.append(dict(model_outputs=outputs, split=split, subject=subject, assignments=len(rr), distance_quantiles_mm=np.quantile(distance, [.1, .5, .9, .99]).tolist(),
                    within_1mm=int((distance <= 1).sum()), within_2mm=int((distance <= 2).sum()),
                    confidence_rejected=sum(r['originally_retained'] == 'False' for r in rr)))
    save(root/'assignment-quality.json', quality)
    save(root/'summary.json', dict(completed=True, smoke=smoke, manifest=manifest, comparisons=graph_rows,
        ap_bootstrap=ap_intervals, graph_bootstrap=graph_intervals, assignment_quality=quality))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm in ARMS:
        p = root/'analyses'/arm/'ranking'/f'val-{arm}-2-all_selected-{arm}-PR.npz'
        if p.exists():
            with np.load(p) as f:
                axes[0].plot(f['recall'], f['precision'], label=arm)
        data = load_rank(arm, arm, 2, 'conditional')
        from sklearn.metrics import precision_recall_curve
        if data['labels'].any():
            precision, recall, _ = precision_recall_curve(data['labels'], data['scores'][:, 0])
            axes[1].plot(recall, precision, label=arm)
    for ax, title in zip(axes, ['All retained candidate pairs', 'Physically recovered endpoint pairs']):
        ax.set(xlabel='Candidate-positive recall', ylabel='Precision', title=title, xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2)
    if axes[1].lines:
        axes[1].legend(fontsize=7)
    else:
        axes[1].text(.1, .5, 'No eligible positives: PR/AP undefined')
    fig.suptitle('Validation PR diagnostics; these curves do not select thresholds')
    fig.savefig(root/'validation-PR.png', dpi=170); plt.close(fig)

    # Automatic interpretation states measured comparisons and keeps architecture choices provisional.
    def graph(experiment, policy, split='val', tolerance=2):
        return next(r for r in graph_rows if r['experiment'] == experiment and r['policy'] == policy and r['split'] == split and r['tolerance_mm'] == tolerance)
    def rank(experiment, condition, population, split='val', arm='original'):
        return next(r for r in rank_rows if r['experiment'] == experiment and r['condition'] == condition and r['population'] == population
                    and r['split'] == split and int(r['tolerance_mm']) == 2 and r['arm'] == arm and not r.get('subject'))
    def number(x):
        return 'NA' if x is None or x == '' else f'{float(x):.4f}'
    lines = ['## Computed results', '', f'Completed cache-only analysis; {len(manifest["ids"]["train"])} training / {len(manifest["ids"]["val"])} validation patches.',
        f'Paired subject bootstrap: {len(subjects)} validation subjects, {resamples} resamples. Tables below use 2 mm; CSVs contain both tolerances.', '',
        '### A: original scorer under each query selection', '',
        '|Selection|Node recall|Node precision|Recovered edges|Edge F1|Conditional recall|Branch F1|Curve F1|Cycle MAE|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        r = graph('A', condition)
        lines.append('|'+condition+'|'+'|'.join(number(r[k]) for k in ['node_recall', 'node_precision', 'recoverable_edges', 'f1', 'conditional_recall', 'common_branch_f1', 'common_curve_f1', 'beta1_mae'])+'|')
    lines += ['', '|Selection / population|Train AP|Validation AP|Validation candidates|Positives|Prevalence|', '|---|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS+['hungarian_training_labels', 'common_identity']:
        population = 'assignment_only' if condition == 'hungarian_training_labels' else 'conditional'
        train, val = rank('A', condition, population, 'train'), rank('A', condition, population)
        lines.append(f'|{condition} / {population}|{number(train["ap"])}|{number(val["ap"])}|{val["candidates"]}|{val["positives"]}|{number(val["prevalence"])}|')
    lines += ['', 'Hungarian training labels include distant assignments; their AP is not positional graph accuracy.',
        'Hungarian and normal AP generally condition on different pairs; common_identity uses identical endpoint identities across all selections.', '',
        '### B: training-calibrated operating points', '',
        '|Head|Policy|Threshold|Validation precision|Recall|Edge F1|Curve F1|Component MAE|Cycle MAE|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for policy in ['fixed_0p5', 'train_max_f1', 'train_preserve_precision']:
            r = graph(arm, policy); t = thresholds[f'{arm}/2'][policy]
            lines.append(f'|{arm}|{policy}|{t:.8f}|'+'|'.join(number(r[k]) for k in ['precision', 'recall', 'f1', 'common_curve_f1', 'beta0_mae', 'beta1_mae'])+'|')
    lines += ['', '|Head|Train conditional AP|Validation conditional AP|Train all-retained AP|Validation all-retained AP|', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        values = [rank(arm, arm, population, split, arm)['ap'] for population in ['conditional', 'all_selected'] for split in ['train', 'val']]
        lines.append('|'+arm+'|'+'|'.join(number(v) for v in values)+'|')
    lines += ['', '### Uncertainty for principal comparisons', '', '|Comparison|Population/policy|95% interval|', '|---|---|---|']
    for item in ap_intervals:
        if item['tolerance_mm'] == 2 and ((item['experiment'] == 'B' and item['arm'] != 'original') or (item['experiment'] == 'A' and item['condition'] == 'hungarian')):
            delta = item.get('delta_vs_original_95pct', item.get('delta_vs_normal_95pct'))
            lines.append(f'|AP: {item.get("condition",item["arm"])} vs original normal|{item["population"]}|{delta}|')
    for item in graph_intervals:
        if item['experiment'] == 'B' and item['tolerance_mm'] == 2 and item['policy'] == 'train_preserve_precision':
            lines.append(f'|Edge F1: {item["arm"]} vs original|common training-precision target|{item["delta_95pct"]["f1"]}|')
            lines.append(f'|Curve F1: {item["arm"]} vs original|common training-precision target|{item["delta_95pct"]["common_curve_f1"]}|')
    original = rank('original', 'original', 'conditional', arm='original')
    hungarian = rank('A', 'hungarian', 'conditional')
    train_h = rank('A', 'hungarian', 'conditional', 'train')
    strong_ranking = []
    strong_calibrated = []
    for arm in ARMS[1:]:
        aps = [r for r in ap_intervals if r['experiment'] == 'B' and r['arm'] == arm and r['tolerance_mm'] == 2]
        if all(r['delta_vs_original_95pct'] is not None and r['delta_vs_original_95pct'][0] > 0 for r in aps):
            strong_ranking.append(arm)
        ci = next(r for r in graph_intervals if r['experiment'] == 'B' and r['arm'] == arm and r['tolerance_mm'] == 2 and r['policy'] == 'train_preserve_precision')
        if all(ci['delta_95pct'][metric][0] > 0 for metric in ['f1', 'common_curve_f1']) and graph(arm, 'train_preserve_precision')['beta1_mae'] <= graph('original', 'train_preserve_precision')['beta1_mae']:
            strong_calibrated.append(arm)
    lines += ['', '### C: combined diagnosis and next experiment', '',
        f'1. Original conditional AP is {number(hungarian["ap"])} on localized Hungarian-selected validation pairs versus {number(original["ap"])} on normal pairs. Candidate differences and their paired AP intervals above prevent attributing this gap solely to matching.',
        f'2. Under localized Hungarian correspondence, training AP is {number(train_h["ap"])} versus validation {number(hungarian["ap"])}. Training is an in-sample fit measure, not an independent generalization estimate.',
        '3. Node selection and ranking remain distinct: compare recovered edges with conditional recall and AP in A. Distant Hungarian assignments are not counted as recovered endpoints; the rematching sensitivity row isolates correspondence from selected query identity.',
        f'4. Heads with positive paired AP intervals on both candidate populations: {strong_ranking or "none"}. Heads with positive edge-F1 and Curve-F1 intervals at the common training precision target and no larger point cycle error: {strong_calibrated or "none"}. These are descriptive comparisons, not a pre-registered selection rule. Threshold changes cannot improve AP; training precision need not transfer.',
        '5. Another mining/sampling change remains unsupported unless it survives ranking and calibrated connectivity comparisons. Positive intervals identify candidates for confirmation, not a replacement model or proof of anatomical topology. No random-negative auxiliary control was trained in experiment 7, so mining-specific causality remains unresolved.',
        '6. The next bounded experiment should first address the weakest measured link: if matching/calibration leaves poor localized-pair validation ranking, compare a small relation readout using additional local image evidence against an equal-capacity token/geometry control on the same pairs. This is a hypothesis to test, not proof that image features will help. Existing chord-image results were weak, so a new feature probe must change its context explicitly. Multiscale transformer attention is not justified by this diagnostic alone.', '',
        'Recommended priority: retain the original scorer unless a calibrated alternative shows a convincing validation gain without harmful topology tradeoffs; review these results before any further training.', '',
        f'Outputs: `{root}`. Detailed AP intervals, graph intervals, per-patch/per-subject tables, assignment distances, PR curves, length/distance strata and input hashes are preserved.',
        '![Validation precision–recall curves](validation-PR.png)', '']
    report = '\n'.join(lines)
    (root/'results.md').write_text(report)
    if not smoke:
        evidence = Path('docs/research/evidence/ixi_relation_matching_calibration_20261019'); evidence.mkdir(parents=True, exist_ok=True)
        for name in ['comparison.csv', 'ranking.csv', 'ap-bootstrap.json', 'graph-bootstrap.json', 'assignment-quality.json', 'length-distance-strata.csv', 'strata-definitions.json', 'summary.json', 'results.md', 'validation-PR.png']:
            shutil.copy2(root/name, evidence/name)
        shutil.copy2(prepared/'thresholds.json', evidence/'thresholds.json')
        doc = Path('docs/research/metrics/8-ixi-relation-matching-calibration.md')
        text = doc.read_text().split('## Computed results')[0]
        report = report.replace('](validation-PR.png)', '](../evidence/ixi_relation_matching_calibration_20261019/validation-PR.png)')
        doc.write_text(text+report)
        roadmap = Path('docs/research/IXI_graph_extraction_diagnostic_roadmap.md')
        with roadmap.open('a') as f:
            f.write('\n\n## Completed matching and calibration follow-up\n\nCache-only retrospective Hungarian and five-head training-calibrated evaluation completed; '
                    'see [report 8](metrics/8-ixi-relation-matching-calibration.md) for measured results, uncertainty and revised conditional priorities. '
                    'No further training or architecture change was launched. Earlier conclusions remain recorded above.\n')
    print(root/'results.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    root = args.root or Path(json.loads(Path('experiments/ixi_relation_matching_calibration/submission.json').read_text())['output_root'])
    if args.show:
        path = root/'results.md'
        if not path.exists():
            raise SystemExit(f'Results are not complete; no waiting. Expected {path}')
        print(path.read_text())
    else:
        collect(root, args.smoke)
