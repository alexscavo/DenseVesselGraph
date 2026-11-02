"""CPU-only L08 evaluation/paired collection from immutable prediction caches."""
import argparse
import csv
import fcntl
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score
from scripts.node_first_curriculum import ARMS, MAIN, META, clean, plan, read, write
from scripts.node_verifier_metrics import measure, aggregate, association
from metrics.common_vessel_metrics import evaluate_common_vessels, aggregate_common_vessels
from training.evaluation.metrics import evaluate_graph, summarize_metrics


def cache_rows(folder):
    with zipfile.ZipFile(Path(folder) / 'caches.zip') as z:
        for rec in read(Path(folder) / 'records.json'):
            with np.load(io.BytesIO(z.read(rec['cache_key'])), allow_pickle=False) as arrays:
                yield rec, {k: arrays[k] for k in arrays.files}


def populations(c, tolerance=2., ids=None):
    ids = np.flatnonzero(c['scores'] > .5) if ids is None else np.asarray(ids)
    local = association(c['physical'][ids], c['truth'], tolerance)
    mapping = {int(ids[q]): int(t) for q, t in local.items()}
    pairs = c['pairs']
    active = np.isin(pairs, ids).all(1)
    eligible = np.array([int(a) in mapping and int(b) in mapping for a, b in pairs], bool)
    edges = {tuple(sorted(map(int, e))) for e in c['gt_edges']}
    labels = np.array([bool(ok) and tuple(sorted((mapping[int(a)], mapping[int(b)]))) in edges
                       for (a, b), ok in zip(pairs, eligible)], bool)
    return mapping, active, eligible, labels


def patch(c, rec, tolerance=2., edge_threshold=.5, ids=None):
    ids = np.flatnonzero(c['scores'] > .5) if ids is None else np.asarray(ids)
    r = measure(c, ids, tolerance, edge_threshold=edge_threshold)
    mapping, active, eligible, labels = populations(c, tolerance, ids)
    distances = [float(np.linalg.norm(c['physical'][q] - c['truth'][t])) for q, t in mapping.items()]
    h = np.linalg.norm(c['physical'][c['hungarian_source']] - c['truth'][c['hungarian_target']], axis=-1)
    selected = active & (c['relations'] > edge_threshold)
    r.update(raw_tp=len(association(c['physical'], c['truth'], tolerance)),
             localized_mm_sum=sum(distances), hungarian_mm_sum=float(h.sum()), hungarian_n=len(h),
             hungarian_localized=int((h <= tolerance).sum()),
             edge_fp=r['edges'] - r['edge_tp'], edge_fn=r['gt_edges'] - r['edge_tp'],
             false_edges_unmatched=int((selected & ~labels & ~eligible).sum()),
             subject=rec['subject'], sample=rec['sample'], split=rec['split'], tolerance=str(tolerance))
    return r, dict(retained=(c['relations'][active], labels[active]),
                   conditional=(c['relations'][eligible], labels[eligible]))


def summary(rows):
    r = aggregate(rows)
    r.update(raw_recall=r['raw_tp'] / r['gt_nodes'],
             localized_mm=r['localized_mm_sum'] / r['node_tp'] if r['node_tp'] else None,
             hungarian_mm=r['hungarian_mm_sum'] / r['hungarian_n'],
             hungarian_localized_fraction=r['hungarian_localized'] / r['hungarian_n'],
             node_count_bias=(r['nodes'] - r['gt_nodes']) / len(rows),
             edge_count_bias=(r['edges'] - r['gt_edges']) / len(rows))
    return r


def ap(scores, labels, weights=None):
    return float(average_precision_score(labels, scores, sample_weight=weights)) if labels.any() else None


def pooled(parts):
    scores = np.concatenate([v[0] for v in parts])
    labels = np.concatenate([v[1] for v in parts])
    return dict(ap=ap(scores, labels), candidates=len(scores), positives=int(labels.sum()),
                prevalence=float(labels.mean()) if len(labels) else None)


def common_summary(rows):
    # The existing aggregator expects metric fields only, not subject/sample IDs.
    return aggregate_common_vessels([{k: v for k, v in row.items()
                                     if k not in ['subject', 'sample']} for row in rows])


def evaluate_folder(folder):
    """Standard metrics, same saved nodes/edges and physical/common-v2 definitions."""
    folder = Path(folder)
    assert read(folder / 'complete.json')['completed']
    rows, common, native = [], {'train': [], 'val': []}, {'train': [], 'val': []}
    states = {s: [[], []] for s in ['train', 'val']}
    ranking = {s: {p: [] for p in ['retained', 'conditional']} for s in ['train', 'val']}
    for rec, c in cache_rows(folder):
        graph = {k.removeprefix('graph_'): v for k, v in c.items() if k.startswith('graph_')}
        graph['spline_family'] = 'bspline'
        common_row = evaluate_common_vessels(graph, rec['target_path'])
        common[rec['split']].append(dict(common_row, subject=rec['subject'], sample=rec['sample']))
        transform = np.eye(4)
        transform[:3, :3] = c['metric']
        metrics, n, e = evaluate_graph(graph, c['truth_normalized'], c['gt_edges'],
            protocol=plan()['config']['evaluation'].get('protocol', {}),
            world_transform=transform, return_detection_states=True)
        native[rec['split']].append(metrics)
        states[rec['split']][0].append(n)
        states[rec['split']][1].append(e)
        for tol in [1., 2.]:
            r, ranks = patch(c, rec, tol)
            rows.append(r)
            if tol == 2.:
                for p, pair in ranks.items():
                    ranking[rec['split']][p].append(pair)
    result = dict(physical={s: {str(t): summary([r for r in rows if r['split'] == s and r['tolerance'] == str(t)])
                                for t in [1., 2.]} for s in ['train', 'val']},
                  common={s: common_summary(common[s]) for s in common},
                  native={s: summarize_metrics(native[s], node_detection_states=states[s][0],
                                               edge_detection_states=states[s][1],
                                               iou_thresholds=plan()['config']['evaluation'].get('protocol', {}).get('iou_thresholds'))
                          for s in native},
                  ranking={s: {p: pooled(values) for p, values in groups.items()}
                           for s, groups in ranking.items()})
    records = read(folder / 'records.json')
    result['losses'] = {s: {k: float(np.mean([r['loss'][k] for r in records if r['split'] == s]))
                           for k in records[0]['loss']} for s in ['train', 'val']}
    write(folder / 'cpu-summary.json', result)
    write(folder / 'physical-rows.json', rows)
    write(folder / 'common-rows.json', common)
    per_subject = []
    for split in ['train', 'val']:
        for tol in ['1.0', '2.0']:
            for subject in sorted({r['subject'] for r in rows if r['split'] == split}):
                selected = [r for r in rows if r['split'] == split and r['subject'] == subject and r['tolerance'] == tol]
                per_subject.append(dict(subject=subject, split=split, tolerance=tol, metrics=summary(selected)))
    write(folder / 'per-subject.json', per_subject)
    with (folder / 'per-patch.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def assignment_stability(folder, previous):
    old = {rec['cache_key']: c for rec, c in cache_rows(previous)}
    rows = []
    for rec, c in cache_rows(folder):
        before = old[rec['cache_key']]
        a = dict(zip(before['hungarian_target'].tolist(), before['hungarian_source'].tolist()))
        b = dict(zip(c['hungarian_target'].tolist(), c['hungarian_source'].tolist()))
        assert a.keys() == b.keys()
        rows.append(dict(subject=rec['subject'], split=rec['split'], sample=rec['sample'],
                         assignments=len(a), changed=sum(a[k] != b[k] for k in a)))
    return rows


def evaluate_arm(stage, index):
    p = plan()
    seed, arm = p['stages'][stage]['seeds'][index // 2], ARMS[index % 2]
    out = Path(p['output_root']) / stage / str(seed) / arm
    assert read(out / 'complete.json')['completed']
    steps = p['stages'][stage]['updates']
    previous = out / 'initial-assignment'
    trajectories = {}
    for step in [steps // 10, steps // 5, steps // 2, steps]:
        folder = out / f'step-{step}'
        result = evaluate_folder(folder)
        stability = assignment_stability(folder, previous)
        write(folder / 'assignment-stability.json', stability)
        result['assignment_stability'] = {s: dict(
            changed=sum(r['changed'] for r in stability if r['split'] == s),
            total=sum(r['assignments'] for r in stability if r['split'] == s)) for s in ['train', 'val']}
        trajectories[str(step)] = result
        previous = folder
    write(out / 'evaluation-complete.json', dict(completed=True, stage=stage, seed=seed,
                                                arm=arm, trajectories=trajectories))
    # Checkpoint identity is captured without loading/retraining any model.
    import hashlib
    identities = {}
    for path in sorted(out.glob('model-*.pt')):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        identities[path.name] = dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())
    write(out / 'checkpoint-identities.json', identities)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for key in ['class', 'nodes', 'edges']:
        axes[0, 0].plot([r['step'] for r in history], [r[key] for r in history], label=key)
    axes[0, 0].set(title='Raw training losses', xlabel='update')
    axes[0, 0].legend()
    axes[0, 1].plot([r['step'] for r in history], [r['edge_multiplier'] for r in history])
    axes[0, 1].set(title='Edge multiplier', xlabel='update')
    for split in ['train', 'val']:
        keys = sorted(trajectories, key=int)
        for metric in ['node_f1', 'edge_f1']:
            axes[1, 0].plot([int(k) for k in keys], [trajectories[k]['physical'][split]['2.0'][metric] for k in keys], marker='o', label=split + ' ' + metric)
        axes[1, 1].plot([int(k) for k in keys], [trajectories[k]['losses'][split]['total'] for k in keys], marker='o', label=split)
    axes[1, 0].set(title='2mm F1 at fixed checkpoints', xlabel='update')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(title='Unscheduled joint evaluation loss', xlabel='update')
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(out / 'learning-curves.png', dpi=150)
    plt.close(fig)


def paired_bootstrap(a, b, metric, draws=1000):
    subjects = sorted({r['subject'] for r in a})
    assert {(r['sample'], r['subject']) for r in a} == {(r['sample'], r['subject']) for r in b}
    groups = [{s: [r for r in rows if r['subject'] == s] for s in subjects} for rows in [a, b]]
    rng = np.random.default_rng(plan()['bootstrap_seed'])
    values = []
    for _ in range(draws):
        chosen = rng.choice(subjects, len(subjects), replace=True)
        aa, bb = [[r for s in chosen for r in g[s]] for g in groups]
        values.append(metric(bb) - metric(aa))
    delta = metric(b) - metric(a)
    return dict(delta=delta, ci95=np.quantile(values, [.025, .975]).tolist(), subjects=len(subjects))


def common_rank_pairs(a, b, split='val'):
    """Identical eligible GT pairs/labels; learned query representatives may differ."""
    cached_a = {rec['cache_key']: c for rec, c in cache_rows(a)}
    output = []
    for rec, cb in cache_rows(b):
        if rec['split'] != split:
            continue
        ca = cached_a[rec['cache_key']]
        maps = [populations(c)[0] for c in [ca, cb]]
        inverse = [{t: q for q, t in mapping.items()} for mapping in maps]
        targets = sorted(set(inverse[0]) & set(inverse[1]))
        links = {tuple(sorted(map(int, e))) for e in ca['gt_edges']}
        labels, scores = [], [[], []]
        pair_maps = [{tuple(map(int, pair)): float(score) for pair, score in zip(c['pairs'], c['relations'])}
                     for c in [ca, cb]]
        for i, left in enumerate(targets):
            for right in targets[i + 1:]:
                labels.append((left, right) in links)
                for side in [0, 1]:
                    key = tuple(sorted([inverse[side][left], inverse[side][right]]))
                    scores[side].append(pair_maps[side][key])
        output.append(dict(subject=rec['subject'], sample=rec['sample'],
                           labels=np.array(labels, bool), a=np.array(scores[0]), b=np.array(scores[1])))
    labels = np.concatenate([r['labels'] for r in output])
    aa = np.concatenate([r['a'] for r in output])
    bb = np.concatenate([r['b'] for r in output])
    subjects = sorted({r['subject'] for r in output})
    subject_index = np.concatenate([np.full(len(r['labels']), subjects.index(r['subject']), int) for r in output])
    rng = np.random.default_rng(plan()['bootstrap_seed'])
    deltas = []
    if labels.any():
        for _ in range(plan()['subject_bootstrap_draws']):
            counts = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
            weights = counts[subject_index]
            if weights[labels].sum():
                deltas.append(ap(bb, labels, weights) - ap(aa, labels, weights))
    a_ap, b_ap = ap(aa, labels), ap(bb, labels)
    return dict(a=a_ap, b=b_ap, delta=b_ap - a_ap if a_ap is not None else None,
                ci95=np.quantile(deltas, [.025, .975]).tolist() if deltas else None,
                candidates=len(labels), positives=int(labels.sum()),
                prevalence=float(labels.mean()) if len(labels) else None,
                population='Intersection of default-retained, spatially recovered GT anchors at2mm in both arms; all unordered pairs')


def train_threshold(folder):
    scores, labels, gt_edges = [], [], 0
    for rec, c in cache_rows(folder):
        if rec['split'] != 'train':
            continue
        _, active, _, lab = populations(c)
        scores.extend(c['relations'][active])
        labels.extend(lab[active])
        gt_edges += len(c['gt_edges'])
    scores, labels = np.asarray(scores, np.float32), np.asarray(labels, bool)
    if not len(scores) or not labels.any():
        return 1.
    order = np.argsort(-scores, kind='stable')
    ss, yy = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.cumsum(yy)[ends]
    f1 = 2 * tp / (ends + 1 + gt_edges)
    chosen = int(ends[np.argmax(f1)])
    # Strict decoder '>' at float32; never split score ties or tune on val.
    return float(np.nextafter(ss[chosen], np.float32(-np.inf)))


def confidence_controls(a_folder, b_folder):
    aa = {rec['cache_key']: c for rec, c in cache_rows(a_folder)}
    cuts = [train_threshold(a_folder), train_threshold(b_folder)]
    calibrated = [[], []]
    count_rows = []
    for rec, cb in cache_rows(b_folder):
        if rec['split'] != 'val':
            continue
        ca = aa[rec['cache_key']]
        for side, c in enumerate([ca, cb]):
            calibrated[side].append(patch(c, rec, edge_threshold=cuts[side])[0])
        count = int((ca['scores'] > .5).sum())
        ids = np.sort(np.argsort(-cb['scores'], kind='stable')[:count])
        count_rows.append(patch(cb, rec, ids=ids)[0])
    return dict(train_max_f1_thresholds=dict(joint=cuts[0], node_first=cuts[1]),
                calibrated=dict(joint=summary(calibrated[0]), node_first=summary(calibrated[1])),
                matched_joint_predicted_node_count=summary(count_rows),
                limitations='Calibrated/matched-count physical graphs only; no learned Curve F1 fabricated for pairs without saved curves. Default Curve F1 remains primary.')


def collect(stage, auto_promote=False):
    p = plan()
    root = Path(p['output_root'])
    stage_spec = p['stages'][stage]
    final = stage_spec['updates']
    results = {}
    for seed in stage_spec['seeds']:
        folders = [root / stage / str(seed) / arm for arm in ARMS]
        completes = [read(f / 'complete.json') for f in folders]
        assert completes[0]['initial_state_sha256'] == completes[1]['initial_state_sha256']
        assert completes[0]['batch_chain'] == completes[1]['batch_chain'], 'Inputs/augmentation diverged'
        evaluations = [read(f / 'evaluation-complete.json') for f in folders]
        final_folders = [f / f'step-{final}' for f in folders]
        rr = [read(f / 'physical-rows.json') for f in final_folders]
        filtered = [[r for r in rows if r['split'] == 'val' and r['tolerance'] == '2.0'] for rows in rr]
        common = [read(f / 'common-rows.json')['val'] for f in final_folders]
        differences = {key: paired_bootstrap(*filtered, lambda rows, k=key: summary(rows)[k], p['subject_bootstrap_draws'])
                       for key in ['node_f1', 'edge_f1', 'raw_recall', 'conditional_recall']}
        differences.update({key: paired_bootstrap(*common, lambda rows, k=key: common_summary(rows)[k], p['subject_bootstrap_draws'])
                            for key in ['common_branch_f1', 'common_curve_f1']})
        results[str(seed)] = dict(completion=completes, evaluations=evaluations, paired=differences,
                                  common_candidate_ranking=common_rank_pairs(*final_folders),
                                  common_candidate_ranking_train=common_rank_pairs(*final_folders, split='train'),
                                  operating=confidence_controls(*final_folders))
    r = results[str(stage_spec['seeds'][0])]
    a, b = [e['trajectories'][str(final)] for e in r['evaluations']]
    af, bf = [e['physical']['val']['2.0'] for e in [a, b]]
    criteria = dict(edge_f1_delta=r['paired']['edge_f1']['delta'] >= .01,
                    edge_f1_subject_ci_positive=r['paired']['edge_f1']['ci95'][0] > 0,
                    common_candidate_ap=(r['common_candidate_ranking']['delta'] is not None and r['common_candidate_ranking']['delta'] >= .005),
                    branch_non_decreasing=r['paired']['common_branch_f1']['delta'] >= 0,
                    curve_non_decreasing=r['paired']['common_curve_f1']['delta'] >= 0,
                    true_edges_non_decreasing=bf['edge_tp'] >= af['edge_tp'],
                    training_calibrated_edge_f1=r['operating']['calibrated']['node_first']['edge_f1'] - r['operating']['calibrated']['joint']['edge_f1'] >= .005)
    positive = all(criteria.values())
    negative = r['paired']['edge_f1']['ci95'][1] < -.01
    decision = 'positive_pilot_full_confirmation_eligible' if positive else 'negative_stop' if negative else 'inconclusive_do_not_auto_expand'
    recommendation = 'Full100epoch matched confirmation and independent second seed are justified by the predefined positive pilot gate.' if positive else 'Stop this compressed pilot; retain the joint control.' if negative else 'No automatic expansion: inspect the learning-curve and representation evidence first.'
    if stage == 'pilot' and not positive and not negative:
        delta = r['common_candidate_ranking']['delta']
        if r['paired']['edge_f1']['delta'] > 0 and delta is not None and delta > 0:
            recommendation = 'A matched second-seed pilot is the smallest check of the positive but uncertain graph/ranking trend; do not alter the schedule or launch full training automatically.'
        elif max(af['edge_f1'], bf['edge_f1']) < .2 and all(
                e['trajectories'][str(final)]['physical']['val']['2.0']['edge_f1'] >
                e['trajectories'][str(final // 2)]['physical']['val']['2.0']['edge_f1']
                for e in r['evaluations']):
            recommendation = 'Both models remain weak and are improving: a longer matched learning-curve study is scientifically reasonable, but this compressed pilot does not establish a positive curriculum signal. Review before expanding.'
    if stage == 'full':
        consistent = all(e['paired']['edge_f1']['delta'] > 0 and
                         e['common_candidate_ranking']['delta'] is not None and
                         e['common_candidate_ranking']['delta'] > 0 and
                         e['paired']['common_branch_f1']['delta'] >= 0
                         for e in results.values())
        decision = 'full_two_seed_consistent_signal' if consistent else 'full_two_seed_negative_or_mixed'
        recommendation = 'Review the two-seed fixed-budget comparison; no additional schedules or combined architectures are automatically launched.'
    output = dict(stage=stage, results=results, decision=decision, promotion_checks=criteria,
                  recommended_next_action=recommendation,
                  interpretation='Order of supervision, not explicit separation of geometric/relational embedding subspaces.',
                  uncertainty='Paired whole-subject bootstrap; pilot single training seed and reused validation population.')
    write(root / stage / 'analysis.json', output)
    write(root / stage / 'decision.json', dict(decision=decision, checks=criteria, full_allowed=positive, stage=stage))
    lines = [f'## L08 {stage}: node-first versus joint scratch B0', '',
             f'Decision: **{decision}**. Same random state and full augmented input-chain hashes verified.', '', recommendation, '',
             '| Seed | Arm | Node F1 1/2mm | Edge F1 1/2mm | Raw/filtered recall2mm | Branch / Curve F1 | Conditional / retained AP |',
             '|---|---|---|---|---|---|---|']
    for seed, evidence in results.items():
        for arm, evaluation in zip(ARMS, evidence['evaluations']):
            e = evaluation['trajectories'][str(final)]
            one, two = e['physical']['val']['1.0'], e['physical']['val']['2.0']
            fmt = lambda x: 'n/a' if x is None else f'{x:.5f}'
            lines.append(f"|{seed}|{arm}|{one['node_f1']:.5f}/{two['node_f1']:.5f}|{one['edge_f1']:.5f}/{two['edge_f1']:.5f}|{two['raw_recall']:.5f}/{two['node_recall']:.5f}|{e['common']['val']['common_branch_f1']:.5f}/{e['common']['val']['common_curve_f1']:.5f}|{fmt(e['ranking']['val']['conditional']['ap'])}/{fmt(e['ranking']['val']['retained']['ap'])}|")
        lines += ['', f'### Learning-mechanism diagnostics, seed{seed}', '',
                  '| Update | Node F1 delta2mm | Raw recall delta2mm | Joint / curriculum assignment-change fraction |',
                  '|---:|---:|---:|---:|']
        for checkpoint in [final // 10, final // 5, final // 2, final]:
            ea, eb = [v['trajectories'][str(checkpoint)] for v in evidence['evaluations']]
            pa, pb = ea['physical']['val']['2.0'], eb['physical']['val']['2.0']
            changes = []
            for value in [ea, eb]:
                z = value.get('assignment_stability', {}).get('val', {})
                changes.append(f"{z['changed']/z['total']:.5f}" if z.get('total') else 'n/a')
            lines.append(f"|{checkpoint}|{pb['node_f1']-pa['node_f1']:+.5f}|{pb['raw_recall']-pa['raw_recall']:+.5f}|{' / '.join(changes)}|")
        for key in ['edge_f1', 'common_branch_f1', 'common_curve_f1']:
            item = evidence['paired'][key]
            lines.append(f"\nFinal{key} curriculum−joint: {item['delta']:+.5f}, paired subject95%CI[{item['ci95'][0]:+.5f},{item['ci95'][1]:+.5f}].")
        rank = evidence['common_candidate_ranking']
        lines.append(f"\nConditional AP on identical eligible GT pairs: joint{rank['a']}, curriculum{rank['b']}, delta{rank['delta']}, CI{rank['ci95']}; candidates{rank['candidates']}, positives{rank['positives']}. This controls pair labels/population, but does not isolate intrinsic token content from a changed learned query representative.")
        controls = evidence['operating']
        lines.append(f"\nTraining-only edge-calibrated valF1: joint{controls['calibrated']['joint']['edge_f1']:.5f}, curriculum{controls['calibrated']['node_first']['edge_f1']:.5f}. Curriculum at the joint model's predicted node count: node/edgeF1{controls['matched_joint_predicted_node_count']['node_f1']:.5f}/{controls['matched_joint_predicted_node_count']['edge_f1']:.5f}. Calibration/count diagnostics have no fabricated learned-curve score.")
        alignments = {}
        for arm in ARMS:
            source = root / stage / seed / arm / 'training.jsonl'
            values = []
            if source.exists():
                for line in source.read_text().splitlines():
                    value = json.loads(line)
                    alignment = value.get('gradient_alignment', {})
                    if value['step'] <= final // 5 and alignment and alignment['cosine'] is not None:
                        values.append(alignment['cosine'])
            alignments[arm] = dict(probes=len(values), median=float(np.median(values)) if values else None,
                                    negative_fraction=float(np.mean(np.asarray(values) < 0)) if values else None)
        evidence['early_gradient_alignment'] = alignments
        lines.append('\nEarly shared-decoder node/edge gradient cosine: `' + json.dumps(alignments) + '`. Negative cosine is a local gradient-conflict diagnostic; it does not alone establish harmful early edge supervision.')
    lines += ['', 'Node/edge losses, raw/filtered coverage, duplicates, Hungarian distances/churn, topology, native AP/AR and stage trajectories are retained in per-arm evaluation/cache files. Candidate prevalence and AP on the identical eligible GT-pair intersection are separate from each arm’s own conditional AP.', '',
              'The pilot uses20epochs (15640updates) and the unchanged100epoch LR horizon. Its node-only/ramp phases span2/2epochs; a promoted full run restarts both arms at random step0 with10/10epoch phases. The pilot does not establish final convergence. Detached K8 graph_only geometry trains throughout; only edge CE is scheduled. No test data.', '',
              'A negative early result is specific to the compressed pilot schedule. Inconclusive evidence does not automatically trigger a long run. Positive promotion is only a request for two-seed confirmation, not a journal efficacy claim. Cosine gradient measurements are diagnostic associations, not proof of harmful optimization interference.', '',
              f'Numerical evidence: `{root / stage}`. Retrieval: `python3 {META.parents[1]}/scripts/show_node_first_curriculum.py`.']
    text = '\n'.join(lines) + '\n'
    write(root / stage / 'analysis.json', output)
    (root / stage / 'report.md').write_text(text)
    publish(stage, text, decision)
    if stage == 'pilot' and auto_promote and positive:
        import subprocess
        attempt = subprocess.run([sys.executable, str(META.parents[1] / 'cluster/jean_zay/submit_node_first_curriculum.py'), '--stage', 'full'], capture_output=True, text=True)
        write(root / stage / 'promotion-attempt.json', dict(accepted=attempt.returncode == 0,
              returncode=attempt.returncode, stdout=attempt.stdout, stderr=attempt.stderr))
        if attempt.returncode:
            with (root / stage / 'report.md').open('a') as stream:
                stream.write('\nAutomatic full submission was not accepted cleanly; inspect promotion-attempt.json. Pilot measurements remain valid.\n')
    write(root / stage / 'collection-complete.json', dict(completed=True, decision=decision))
    shared = MAIN.parent / 'experiment_gpu_coordination'
    claim = shared / 'node-first-curriculum-20261028.json'
    if claim.exists():
        with (shared / 'submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = read(claim)
            if data.get('stage') == stage:
                data.update(state='completed; ' + decision, concurrent_h100=0,
                            results=str(root / stage / 'analysis.json'))
                write(claim, data)


def publish(stage, text, decision):
    marker = f'<!-- L08 RESULTS {stage} 20261028 -->'
    with (META / 'publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = MAIN / 'docs/research/fine_feature_architecture_study.md'
        before = doc.read_text()
        if marker not in before:
            assert doc.read_text() == before, 'Concurrent edit; results retained'
            doc.write_text(before + '\n\n' + marker + '\n' + text)
        registry = MAIN / 'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md'
        before = registry.read_text()
        after = []
        for line in before.splitlines():
            if line.startswith('| **L08** |'):
                line = f'| **L08** | Scratch B0 joint versus node-first edge curriculum | **{stage.upper()} DONE — measured; {decision}** | Same random weights,120queries, matched focal-node/CE-edge policy; see dated L08 results. Full confirmation only after predefined pilot gate. | [Scratch study](../docs/research/fine_feature_architecture_study.md); `{Path(plan()["output_root"]) / stage / "analysis.json"}` |'
            after.append(line)
        if marker not in before:
            after += ['', marker, text]
        assert registry.read_text() == before, 'Concurrent registry edit; results retained'
        registry.write_text('\n'.join(after) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['arm', 'collect', 'smoke'])
    parser.add_argument('--stage', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--auto-promote', action='store_true')
    args = parser.parse_args()
    if args.mode == 'arm':
        evaluate_arm(args.stage, args.index)
    elif args.mode == 'smoke':
        root = Path(plan()['output_root'])
        for arm in ARMS:
            evaluate_folder(root / 'smoke' / arm)
        write(root / 'cpu-smoke.json', dict(completed=True))
    else:
        collect(args.stage, args.auto_promote)


if __name__ == '__main__':
    main()
