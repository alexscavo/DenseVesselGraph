#!/usr/bin/env python3
"""A: raw-query oracle; B: confidence probes; C: inference-time refinement ablation."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from scripts.ixi_cached_mechanisms import (
    CACHES, SEED, identities, read_cache, target, heads, variant, decode, parity,
    measure, summarize_run, provenance, save, table,
)
from scripts.ixi_query_diagnostics import categorize
from scripts.diagnose_branch_relations import match_nodes


def graphs(args):
    ids = identities(args.smoke)
    _, spline = heads()
    rows, strata, checks = [], [], []
    references = {}
    for arm in ('local_image', 'baseline'):
        with (CACHES[arm]/'per-patch.csv').open() as f:
            references[arm] = {(r['split'], r['sample_id'], float(r['tolerance_mm']), r['policy']): r for r in csv.DictReader(f)}
    for split, names in ids.items():
        for name in names:
            path, gt, mask = target(split, name)
            local = read_cache('local_image', split, name)
            np.testing.assert_array_equal(mask.shape, local['shape'])
            checks.append(dict(split=split, sample_id=name, spline_max_error_mm=parity(local, spline)))
            if args.experiment == 'A':
                caches = {'normal': local}
            else:
                base = read_cache('baseline', split, name)
                # These two continuations never trained the backbone, decoder or relation head.
                np.testing.assert_array_equal(base['tokens'], local['tokens'])
                np.testing.assert_array_equal(base['pair_logits'], local['pair_logits'])
                checks.append(dict(split=split, sample_id=name, baseline_spline_max_error_mm=parity(base, spline)))
                caches = {'baseline_continuation': base, 'neither_local_residual': variant(local, False, False),
                          'coordinates_only': variant(local, True, False),
                          'classification_only': variant(local, False, True), 'both': local}
            for tolerance in (1., 2.):
                if args.experiment == 'A':
                    mappings = {k: categorize(local['nodes'], gt['nodes'], gt['metric'], mask, q, tolerance)[0]
                                for k, q in [('previous_oracle', local['retained']), ('raw_oracle', np.arange(256))]}
                    policies = [('normal', local, None, None)] + [(k, local, np.array(sorted(m), int), m) for k, m in mappings.items()]
                else:
                    policies = [(k, c, None, None) for k, c in caches.items()]
                for policy, c, q, mapping in policies:
                    graph = decode(c, spline, q)
                    r, lengths = measure(c, graph, path, gt, mask, tolerance, mapping)
                    reference_arm = 'baseline' if policy == 'baseline_continuation' else 'local_image'
                    if policy in ('normal', 'previous_oracle', 'baseline_continuation', 'both'):
                        old_policy = 'oracle' if policy == 'previous_oracle' else 'baseline'
                        old = references[reference_arm][split, name, tolerance, old_policy]
                        for field in ('nodes', 'node_tp', 'tp', 'fp', 'fn', 'recoverable_edges', 'beta0', 'beta1'):
                            assert r[field] == int(old[field]), (name, policy, field, r[field], old[field])
                    key = dict(split=split, sample_id=name, subject=name.rsplit('_', 1)[0], tolerance_mm=tolerance, policy=policy)
                    rows.append(dict(**key, **r)); strata.extend(dict(**key, **v) for v in lengths)
                    # Preserve actual query IDs, predicted coordinates, edges and geometry for replay.
                    graph_dir = args.output/'graphs'/split
                    graph_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(graph_dir/f'{name}-{policy}-{tolerance:g}mm.npz',
                                        **{k: v for k, v in graph.items() if isinstance(v, np.ndarray)})
            print(args.experiment, split, name, flush=True)
    save(args.output/'parity.json', checks)
    summarize_run(args.output, rows, strata, dict(experiment=args.experiment, **provenance(ids)))


def score_summary(scores, labels):
    result = dict(count=len(scores), matched=int(labels.sum()),
                  ap=float(average_precision_score(labels, scores)) if labels.any() else None,
                  auc=float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None)
    return result


def confidence(args):
    ids = identities(args.smoke)
    # Reuse exact labels/foreground distances established by report 6.
    with (CACHES['local_image']/'nodes.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['sample_id'] in set(ids[r['split']])]
    categories, discrimination, subjects = [], [], []
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['split'], r['population'], float(r['tolerance_mm']))].append(r)
    prdir = args.output/'pr'; prdir.mkdir()
    for (split, population, tolerance), group in grouped.items():
        scores = np.array([float(r['score']) for r in group])
        labels = np.array([r['category'] == 'matched' for r in group])
        key = dict(split=split, population=population, tolerance_mm=tolerance)
        discrimination.append(dict(**key, **score_summary(scores, labels)))
        p, recall, cutoffs = precision_recall_curve(labels, scores)
        np.savez_compressed(prdir/f'{split}-{population}-{tolerance:g}mm.npz', precision=p, recall=recall, thresholds=cutoffs)
        for category in sorted({r['category'] for r in group}):
            keep = np.array([r['category'] == category for r in group]); s = scores[keep]
            categories.append(dict(**key, category=category, count=len(s), above_0p9=int((s >= .9).sum()),
                below_or_equal_0p5=int((s <= .5).sum()), quantiles=np.quantile(s, [0, .1, .25, .5, .75, .9, 1]).tolist()))
            if category != 'matched':
                subset = keep | labels
                discrimination.append(dict(**key, negative_category=category, **score_summary(scores[subset], labels[subset])))
        for subject in sorted({r['subject'] for r in group}):
            keep = np.array([r['subject'] == subject for r in group])
            subjects.append(dict(**key, subject=subject, **score_summary(scores[keep], labels[keep]),
                                 high_confidence_duplicates=sum(r['subject'] == subject and r['category'] == 'duplicate_candidate' and float(r['score']) >= .9 for r in group)))
    # Actual GT recovery under fixed confidence cutoffs requires rematching, not fixed-label PR.
    recovery = []
    for split, names in ids.items():
        for name in names:
            c = read_cache('local_image', split, name)
            for tolerance in (1., 2.):
                raw_map = match_nodes(c['nodes'] @ c['metric'].T, c['gt_nodes'] @ c['metric'].T, tolerance)
                for threshold in (-1., .1, .25, .5, .75, .9, .95):
                    q = np.flatnonzero(c['node_scores'] > threshold)
                    mapping = match_nodes(c['nodes'][q] @ c['metric'].T, c['gt_nodes'] @ c['metric'].T, tolerance)
                    recovery.append(dict(split=split, sample_id=name, subject=name.rsplit('_', 1)[0], tolerance_mm=tolerance,
                        threshold=threshold, retained=len(q), recovered=len(mapping), gt_nodes=len(c['gt_nodes']),
                        raw_recovered=len(raw_map), raw_assigned_queries_rejected=int(sum(c['node_scores'][i] <= threshold for i in raw_map))))
    # Small linear readouts: no coordinates, GT distances or category labels as input features.
    probe_rows, probe_subjects = [], []
    for tolerance in (1., 2.):
        matrices = {}; classes = sorted({r['category'] for r in grouped[('train', 'raw', tolerance)]})
        for split, names in ids.items():
            index = {(r['sample_id'], int(r['query'])): r for r in grouped[(split, 'raw', tolerance)]}
            tokens, score, labels, subjects_for_rows = [], [], [], []
            for name in names:
                c = read_cache('local_image', split, name)
                tokens.append(c['tokens'][:256]); score.append((c['node_logits'][:, 1]-c['node_logits'][:, 0])[:, None])
                labels.extend(index[(name, q)]['category'] for q in range(256))
                subjects_for_rows.extend([name.rsplit('_', 1)[0]]*256)
            matrices[split] = (np.concatenate(tokens), np.concatenate(score), np.array(labels), np.array(subjects_for_rows))
        for features in ('score_only', 'tokens_only'):
            column = 1 if features == 'score_only' else 0
            scaler = StandardScaler().fit(matrices['train'][column])
            model = SGDClassifier(loss='log_loss', alpha=.001, max_iter=50 if args.smoke else 200,
                                  tol=1e-4, class_weight='balanced', random_state=SEED)
            model.fit(scaler.transform(matrices['train'][column]), matrices['train'][2])
            np.savez_compressed(args.output/f'probe-{features}-{tolerance:g}mm.npz', coefficient=model.coef_, intercept=model.intercept_,
                                mean=scaler.mean_, scale=scaler.scale_, classes=model.classes_)
            for split, arrays in matrices.items():
                prediction = model.predict(scaler.transform(arrays[column])); probabilities = model.predict_proba(scaler.transform(arrays[column]))
                key = dict(features=features, split=split, tolerance_mm=tolerance)
                probe_rows.append(dict(**key, balanced_accuracy=float(balanced_accuracy_score(arrays[2], prediction)),
                    fitting_iterations=int(model.n_iter_), reached_iteration_limit=bool(model.n_iter_ == model.max_iter),
                    classes=model.classes_.tolist(), confusion=confusion_matrix(arrays[2], prediction, labels=model.classes_).tolist(),
                    per_class_ap={label: float(average_precision_score(arrays[2] == label, probabilities[:, j])) if (arrays[2] == label).any() else None for j, label in enumerate(model.classes_)}))
                for subject in sorted(set(arrays[3])):
                    keep = arrays[3] == subject
                    probe_subjects.append(dict(**key, subject=subject, accuracy=float(np.mean(prediction[keep] == arrays[2][keep])),
                                               confusion=confusion_matrix(arrays[2][keep], prediction[keep], labels=model.classes_).tolist()))
    for name, data in [('confidence', categories), ('discrimination', discrimination), ('per-subject', subjects),
                       ('recovery', recovery), ('probes', probe_rows), ('probe-subjects', probe_subjects)]:
        save(args.output/(name+'.json'), data)
    table(args.output/'recovery.csv', recovery)
    save(args.output/'summary.json', dict(experiment='B', completed=True, **provenance(ids),
        caveat='Fixed GT-assigned category PR differs from rematched GT recovery. Token probes test linear predictability only; failure does not establish information absence.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['A', 'B', 'C'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(1); torch.manual_seed(SEED)
    args.output.mkdir(parents=True, exist_ok=False)
    confidence(args) if args.experiment == 'B' else graphs(args)
