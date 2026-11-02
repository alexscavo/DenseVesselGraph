#!/usr/bin/env python3
"""Collect image-adapter evidence, or print an existing report without computation."""
import argparse
import json
from pathlib import Path


def collect(root, smoke=False):
    import csv
    import shutil
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from sklearn.metrics import average_precision_score
    from models.image_relation_adapter import ARMS
    from scripts.ixi_cached_mechanisms import table, save
    cache = root/('smoke-cache' if smoke else 'cache')
    records = json.loads((cache/'records.json').read_text())
    with np.load(cache/'common-pairs.npz') as f: common = {k: f[k] for k in f.files}
    offsets = common['offsets']; sizes = np.diff(offsets)
    split = np.repeat([r['split'] for r in records], sizes)
    subjects = np.repeat([r['subject'] for r in records], sizes)
    val = split == 'val'; unique = sorted(set(subjects[val])); lookup = {s: i for i, s in enumerate(unique)}
    resamples = 40 if smoke else 1000
    draws = np.random.default_rng(191032).integers(len(unique), size=(resamples, len(unique)))
    multiplicities = np.array([np.bincount(d, minlength=len(unique)) for d in draws])
    evaluations = root/('smoke-evaluation' if smoke else 'evaluation')
    runs = root/('smoke-runs' if smoke else 'runs')
    summaries, scores, point_ranking, graph_rows, raw_rows = {}, {}, [], [], {}
    arm_names = ['original']+ARMS
    for arm in arm_names:
        directory = evaluations/arm
        assert json.loads((directory/'reporting-complete.json').read_text())['completed'], ('Reporting incomplete',arm)
        summaries[arm] = json.loads((directory/'summary.json').read_text()); assert summaries[arm]['completed']
        assert summaries[arm]['records'] == len(records)
        with np.load(directory/'ranking-scores.npz') as f:
            np.testing.assert_array_equal(f['offsets'], offsets); scores[arm] = f['margin'].copy()
        with (directory/'ranking.csv').open() as f: point_ranking.extend(dict(arm=arm, **{k: v for k, v in r.items() if k != 'arm'}) for r in csv.DictReader(f))
        graph_rows.extend(dict(arm=arm, **r) for r in summaries[arm]['comparison'])
        raw_rows[arm] = json.loads((directory/'per-patch.json').read_text())
    states = [json.loads((runs/a/'summary.json').read_text()) for a in ARMS]
    assert all(s['completed'] and s['frozen_inputs_unchanged'] for s in states)
    assert len({s['batch_order_sha256'] for s in states}) == 1
    assert len({s['updates'] for s in states}) == 1
    if not smoke:
        assert len({s['parameters'] for s in states}) == 1
        logs = [json.loads((runs/a/'history.json').read_text()) for a in ARMS]
        assert all([(x['step'], x['candidates'], x['positives']) for x in h] == [(x['step'], x['candidates'], x['positives']) for x in logs[0]] for h in logs)
    intervals = []; graph_intervals = []
    def ci(values):
        good = values[np.isfinite(values)]
        return np.quantile(good, [.025, .975]).tolist() if len(good) else None
    def ap_boot(y, p, subject):
        if not len(y): return np.full(resamples, np.nan)
        order = np.argsort(-p, kind='stable'); y, p, subject = y[order], p[order], subject[order]
        ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p)-1]
        identifiers = np.array([lookup[s] for s in subject]); output = []
        for counts in multiplicities:
            w = counts[identifiers]; tp = np.cumsum(w*y)[ends]; n = np.cumsum(w)[ends]
            precision = np.divide(tp, n, out=np.zeros(len(tp), float), where=n > 0)
            output.append(float(np.sum(np.diff(np.r_[0, tp])*precision)/tp[-1]) if tp[-1] else np.nan)
        return np.array(output)
    for tol in (1, 2):
        y = common[f'labels_{tol}']
        for population, mask in [('all_retained', val), ('conditional', val & common[f'conditional_{tol}'])]:
            boot = {a: ap_boot(y[mask], scores[a][mask], subjects[mask]) for a in arm_names}
            for arm in arm_names:
                for reference in ['original', 'token_geometry']:
                    delta = boot[arm]-boot[reference]
                    intervals.append(dict(arm=arm, reference=reference, tolerance_mm=tol, population=population,
                        subjects=len(unique), resamples=resamples, ap_95pct=ci(boot[arm]), delta_ap_95pct=ci(delta),
                        usable_resamples=int(np.isfinite(delta).sum())))
    fields = ['tp', 'fp', 'fn', 'common_curve_pred_samples', 'common_curve_gt_samples',
              'common_curve_supported_pred_samples', 'common_curve_supported_gt_samples', 'beta0_error', 'beta1_error']
    def graph_boot(arm, tol, policy):
        rr = [r for r in raw_rows[arm] if r['split'] == 'val' and r['tolerance_mm'] == tol and r['policy'] == policy]
        matrix = np.array([[sum(r[f] for r in rr if r['subject'] == s) for f in fields]+[sum(r['subject'] == s for r in rr)] for s in unique])
        tp, fp, fn, n, ng, sp, sg, b0, b1, patches = (multiplicities @ matrix).T
        p = np.divide(sp, n, out=np.ones(len(sp)), where=n > 0); r = sg/ng
        return dict(edge_f1=2*tp/(2*tp+fp+fn), curve_f1=np.divide(2*p*r, p+r, out=np.zeros(len(p)), where=p+r > 0),
                    beta0_mae=b0/patches, beta1_mae=b1/patches)
    for tol in (1, 2):
        for policy in ['fixed_0p5', 'train_max_f1', 'train_preserve_precision']:
            boots = {a: graph_boot(a, tol, policy) for a in arm_names}
            for arm in ARMS:
                for reference in ['original', 'token_geometry']:
                    graph_intervals.append(dict(arm=arm, reference=reference, tolerance_mm=tol, policy=policy,
                        subjects=len(unique), resamples=resamples,
                        delta_95pct={k: ci(boots[arm][k]-boots[reference][k]) for k in boots[arm]}))
    table(root/('smoke-comparison.csv' if smoke else 'comparison.csv'), graph_rows)
    destination = root/'smoke-results' if smoke else root
    destination.mkdir(parents=True, exist_ok=True)
    table(destination/'ranking.csv', point_ranking); save(destination/'ap-bootstrap.json', intervals)
    save(destination/'graph-bootstrap.json', graph_intervals)
    save(destination/'summary.json', dict(completed=True, smoke=smoke, records=len(records), updates=states[0]['updates'],
        results=summaries, ap_bootstrap=intervals, graph_bootstrap=graph_intervals))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for arm in ARMS:
        history = json.loads((runs/arm/'history.json').read_text())
        axes[0].plot([h.get('step', i+1) for i, h in enumerate(history)], [h['loss'] for h in history], label=arm)
    axes[0].set(xlabel='Update', ylabel='Original edge-weighted focal loss', title='Training trajectory')
    axes[0].legend(fontsize=7)
    for arm in arm_names:
        path = evaluations/arm/'val-2-conditional-PR.npz'
        if path.exists():
            with np.load(path) as f: axes[1].plot(f['recall'], f['precision'], label=arm)
    axes[1].set(xlabel='Conditional-positive recall', ylabel='Precision', title='Validation conditional PR (2 mm)', xlim=(0, 1), ylim=(0, 1))
    if axes[1].lines: axes[1].legend(fontsize=7)
    fig.savefig(destination/'overview.png', dpi=170); plt.close(fig)

    def number(x): return 'NA' if x in [None, ''] else f'{float(x):.4f}'
    def ranking_row(arm, split, population):
        return next(r for r in point_ranking if r['arm'] == arm and r['split'] == split and int(r['tolerance_mm']) == 2 and r['population'] == population and not r.get('subject') and not r.get('site') and not r.get('cohort'))
    lines = ['## Completed results', '', f'Frozen dataset: {sum(r["split"] == "train" for r in records)} training / {sum(r["split"] == "val" for r in records)} validation patches; {len(unique)} validation subjects.',
        f'All adapters compared at update {states[0]["updates"]}. Original scorer has zero further updates.', '',
        '|Scorer|Train conditional AP|Validation conditional AP|Validation all-retained AP|', '|---|---:|---:|---:|']
    for arm in arm_names:
        vals = [ranking_row(arm, 'train', 'conditional')['ap'], ranking_row(arm, 'val', 'conditional')['ap'], ranking_row(arm, 'val', 'all_retained')['ap']]
        lines.append('|'+arm+'|'+'|'.join(number(v) for v in vals)+'|')
    lines += ['', '|Image arm|Reference|Conditional AP difference: paired 95% interval|All-retained AP difference|', '|---|---|---|---|']
    supported = []
    for arm in ARMS[1:]:
        wins = []
        for reference in ['original', 'token_geometry']:
            cc = next(r for r in intervals if r['arm'] == arm and r['reference'] == reference and r['tolerance_mm'] == 2 and r['population'] == 'conditional')
            aa = next(r for r in intervals if r['arm'] == arm and r['reference'] == reference and r['tolerance_mm'] == 2 and r['population'] == 'all_retained')
            lines.append(f'|{arm}|{reference}|{cc["delta_ap_95pct"]}|{aa["delta_ap_95pct"]}|')
            wins.append(cc['delta_ap_95pct'] is not None and cc['delta_ap_95pct'][0] > 0)
        if all(wins): supported.append(arm)
    lines += ['', '|Scorer|Policy|Precision|Full GT recall|Edge F1|Branch F1|Curve F1|Component MAE|Cycle MAE|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in graph_rows:
        if r['split'] == 'val' and r['tolerance_mm'] == 2:
            lines.append(f'|{r["arm"]}|{r["policy"]}|'+'|'.join(number(r[k]) for k in ['precision', 'recall', 'f1', 'common_branch_f1', 'common_curve_f1', 'beta0_mae', 'beta1_mae'])+'|')
    lines += ['', '### Interpretation and limitations', '',
        f'Image arms with positive paired conditional-AP intervals against BOTH the frozen original and token/geometry control: {supported or "none"}. This is exploratory evidence from one seed and reused validation subjects, with multiple comparisons and no multiplicity adjustment.',
        'Compare training/validation AP and predefined intermediate checkpoints to distinguish fitting from generalization. If token-only gains match image gains, added capacity/training remains sufficient to explain the result. Failure of these adapters rules out this particular sampling/training design, not all image-aware relation learning.',
        'Ranking AP is independent of the calibrated cutoff. Training-max-F1 and the common original training-precision policy use training labels only; their validation precision may differ. Validation-only equal-precision readouts are explicitly diagnostic. Fewer components and increased Curve F1 do not establish anatomical correctness when false cycles increase.',
        'Recommendation: if an image arm improves validation ranking against both controls and its calibrated graph errors are acceptable, confirm the lightweight adapter before a larger transformer change. If it only improves fitting, examine site/generalization and label reliability. If no adapter improves ranking, retain the original scorer and review the relation representation/supervision before another architecture change. No multiscale or mask-input experiment was launched.', '',
        'Supporting files: evaluation/*/{ranking.csv,ranking-strata.csv,per-subject.csv,per-site.csv,positive-lengths.csv,intermediate-ranking.json,validation-only-precision.json}; bootstrap JSONs; fixed 3D PNGs and sampling-point JSONs.',
        f'Full outputs: `{root}`.', '']
    (destination/'results.md').write_text('\n'.join(lines)+'\n')
    if not smoke:
        evidence = Path('docs/research/evidence/ixi_image_relation_learning_20261020'); evidence.mkdir(parents=True, exist_ok=True)
        for name in ['comparison.csv', 'ranking.csv', 'ap-bootstrap.json', 'graph-bootstrap.json', 'summary.json', 'results.md', 'overview.png']:
            shutil.copy2(root/name, evidence/name)
        shutil.copy2(root/'benchmark/budget.json', evidence/'budget.json')
        for arm in arm_names:
            target = evidence/arm; target.mkdir(exist_ok=True)
            for name in ['ranking.csv', 'ranking-strata.csv', 'per-subject.csv', 'per-site.csv', 'thresholds.json', 'intermediate-ranking.json', 'validation-only-precision.json', 'degree-distributions.json', 'length-recall.csv']:
                shutil.copy2(evaluations/arm/name, target/name)
            for path in (evaluations/arm).glob('*-3d.*'): shutil.copy2(path, target/path.name)
            if arm != 'original': shutil.copy2(runs/arm/'history.json', target/'history.json')
        doc = Path('docs/research/metrics/9-ixi-image-aware-relation-learning.md')
        prefix = doc.read_text().split('## Completed results')[0]
        doc.write_text(prefix+(root/'results.md').read_text()+'\n![Training and PR curves](../evidence/ixi_image_relation_learning_20261020/overview.png)\n')
        roadmap = Path('docs/research/IXI_graph_extraction_diagnostic_roadmap.md')
        with roadmap.open('a') as f: f.write('\n\nCompleted the larger image-aware relation-adapter batch; see [report 9](metrics/9-ixi-image-aware-relation-learning.md) for ranking/calibration results and conditional priorities. Previous findings remain recorded. No transformer/mask-input experiment was included.\n')
    print(destination/'results.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path); p.add_argument('--smoke', action='store_true'); p.add_argument('--show', action='store_true')
    a = p.parse_args(); root = a.root or Path(json.loads(Path('experiments/ixi_image_relation_learning/submission.json').read_text())['output_root'])
    if a.show:
        path = root/'results.md'
        if not path.exists(): raise SystemExit(f'Results pending; no waiting. Expected {path}')
        print(path.read_text())
    else: collect(root, a.smoke)
