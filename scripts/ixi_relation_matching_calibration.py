"""Cache-only IXI matching and calibration diagnostics; no detector inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from models.matcher import build_matcher
from models.spline import canonical_reverse
from scripts.diagnose_branch_relations import match_nodes
from scripts.ixi_cached_mechanisms import (
    CACHES, manifest, identities, read_cache, target, heads, measure, aggregate,
    summarize_run, save, table, sha,
)

SEED = 191032
ARMS = ['original', 'focal_control', 'regularized', 'hard_unmatched', 'hard_matched']
CONDITIONS = ['normal', 'spatial', 'hungarian', 'hungarian_rematched', 'parent_hungarian']
PREVIOUS = Path(json.loads(Path('experiments/ixi_mechanisms/submission.json').read_text())['output_root'])


def mapping_array(mapping, count=256):
    result = np.full(count, -1, dtype=np.int64)
    for q, g in mapping.items():
        result[q] = g
    return result


def pair_labels(pairs, mapping, gt_edges, selected):
    adjacency = np.zeros((max(int(mapping.max(initial=-1))+1, int(gt_edges.max(initial=-1))+1, 1),)*2, bool)
    for a, b in gt_edges:
        adjacency[a, b] = adjacency[b, a] = True
    g = mapping[pairs]
    eligible = (g >= 0).all(1)
    y = eligible & adjacency[g[:, 0].clip(0), g[:, 1].clip(0)]
    active = np.isin(pairs, selected).all(1)
    return active, eligible & active, y


def select_threshold(scores, y, total_gt_edges, precision_target):
    """Strict >, whole tied groups, highest threshold on objective ties."""
    if not len(scores):
        return dict(train_max_f1=1., train_preserve_precision=1., precision_target_attainable=False)
    order = np.argsort(-scores, kind='stable'); p = scores[order]; y = y[order]
    end = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p)-1]
    tp = np.cumsum(y)[end]; count = end+1
    cut = np.nextafter(p[end].astype(np.float64), -np.inf)
    best = int(np.argmax(2*tp/(count+total_gt_edges)))
    eligible = np.flatnonzero((tp/count >= precision_target) & (tp > 0))
    conservative = int(eligible[np.argmax(tp[eligible])]) if len(eligible) else None
    return dict(train_max_f1=float(cut[best]),
                train_preserve_precision=float(cut[conservative]) if conservative is not None else 1.,
                precision_target_attainable=conservative is not None)


def scores_from_logits(logits):
    margin = logits[..., 1].astype(np.float64)-logits[..., 0].astype(np.float64)
    return 1./(1.+np.exp(-margin))


def weighted_ap(labels, scores, weights):
    if not len(labels) or not np.sum(weights*labels):
        return float('nan')
    order = np.argsort(-scores, kind='stable'); p = scores[order]
    y, w = labels[order], weights[order]
    end = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p)-1]
    tp = np.cumsum(w*y)[end]; count = np.cumsum(w)[end]
    precision = np.divide(tp, count, out=np.zeros(len(tp), float), where=count > 0)
    return float(np.sum(np.diff(np.r_[0, tp])*precision)/tp[-1])


def prepare(root, smoke):
    root.mkdir(parents=True, exist_ok=False)
    ids = identities(smoke); cfg = manifest()['config']; matcher = build_matcher(cfg)
    assert cfg['model']['matcher'] == dict(type='hungarian', class_cost=2., node_cost=5.)
    assert manifest('parent')['config']['model']['matcher'] == cfg['model']['matcher']
    summaries = [json.loads((PREVIOUS/'D'/a/'summary.json').read_text()) for a in ARMS[1:]]
    assert len({s['initial_head_sha256'] for s in summaries}) == 1
    assert len({s['batch_order_sha256'] for s in summaries}) == 1
    assert all(s['ids'] == identities(False) and s['maximum_initial_logit_replay_error'] == 0 for s in summaries)
    assert all(s['checkpoint_sha256'] == manifest()['checkpoint_sha256'] for s in summaries)
    assert all(s['sources']['scripts/train_ixi_relation_mechanisms.py'] == sha('scripts/train_ixi_relation_mechanisms.py') for s in summaries)
    assert all(sha(PREVIOUS/'D'/a/'relation-head.pt') for a in ARMS[1:])
    errors, assignments, calibration_data = [], [], {t: [] for t in (1, 2)}
    inputs = []
    for split, names in ids.items():
        (root/split).mkdir()
        for name in names:
            c = read_cache('local_image', split, name)
            parent = read_cache('parent', split, name)
            np.testing.assert_array_equal(parent['tokens'], c['tokens'])
            np.testing.assert_array_equal(parent['pair_logits'], c['pair_logits'])
            path, gt, _ = target(split, name)
            np.testing.assert_array_equal(c['gt_nodes'], gt['nodes']); np.testing.assert_array_equal(c['gt_edges'], gt['edges'])
            np.testing.assert_array_equal(c['metric'], gt['metric'])
            assert len(gt['nodes']) <= 256
            np.testing.assert_array_equal(c['nodes'], c['boxes'][:, :3])
            np.testing.assert_array_equal(c['query_ids'], np.arange(256))
            np.testing.assert_array_equal(c['edges'], torch.combinations(torch.arange(256)).numpy())
            np.testing.assert_array_equal(np.flatnonzero(c['node_logits'].argmax(-1) == 1), c['retained'])
            source, truth = matcher({'pred_nodes': torch.tensor(c['boxes'])[None], 'pred_logits': torch.tensor(c['node_logits'])[None]},
                                    {'nodes': [torch.tensor(gt['nodes'])]})[0]
            h = dict(zip(source.tolist(), truth.tolist()))
            ps, pt = matcher({'pred_nodes': torch.tensor(parent['boxes'])[None], 'pred_logits': torch.tensor(parent['node_logits'])[None]},
                             {'nodes': [torch.tensor(gt['nodes'])]})[0]
            parent_h = dict(zip(ps.tolist(), pt.tolist()))
            parent_error = np.linalg.norm((parent['nodes'][ps]-gt['nodes'][pt]) @ gt['metric'].T, axis=1)
            # Independent cost reproduction checks inputs and the actual training rule.
            from scipy.optimize import linear_sum_assignment
            probability = torch.tensor(c['node_logits']).softmax(-1)[:, 1]
            cost = 5*torch.cdist(torch.tensor(c['boxes'][:, :3]), torch.tensor(gt['nodes']), p=1)-2*probability[:, None]
            ss, tt = linear_sum_assignment(cost.numpy())
            np.testing.assert_array_equal(ss, source); np.testing.assert_array_equal(tt, truth)
            physical_error = np.linalg.norm((c['nodes'][source]-gt['nodes'][truth]) @ gt['metric'].T, axis=1)
            for q, g, error in zip(source.tolist(), truth.tolist(), physical_error):
                assignments.append(dict(model_outputs='refined', split=split, sample_id=name, subject=name.rsplit('_', 1)[0], query=q, gt=g,
                    distance_mm=float(error), object_probability=float(probability[q]), originally_retained=bool(q in c['retained'])))
            values = dict(pairs=c['edges'], retained=c['retained'], hungarian_queries=np.array(sorted(h), int),
                          hungarian_assignment=mapping_array(h), hungarian_errors_mm=physical_error)
            values['parent_hungarian_queries'] = np.array(sorted(parent_h), int)
            for q, g, error in zip(ps.tolist(), pt.tolist(), parent_error):
                assignments.append(dict(model_outputs='parent', split=split, sample_id=name, subject=name.rsplit('_', 1)[0], query=q, gt=g,
                    distance_mm=float(error), object_probability=float(parent['node_scores'][q]), originally_retained=bool(q in parent['retained'])))
            logits = [c['pair_logits']]
            files = [CACHES['local_image']/split/(name+'.npz'), CACHES['parent']/split/(name+'.npz'), path, PREVIOUS/'features'/split/(name+'.npy')]
            for arm in ARMS[1:]:
                file = PREVIOUS/'D'/arm/split/(name+'.npz')
                with np.load(file) as f:
                    assert f['pair_logits'].shape == c['pair_logits'].shape
                    logits.append(f['pair_logits'])
                files.append(file)
            p = scores_from_logits(np.stack(logits))
            assert np.isfinite(p).all()
            np.testing.assert_array_equal(p > .5, np.stack(logits).argmax(-1) == 1)
            np.testing.assert_allclose(p[0], c['edge_scores'], atol=2e-6, rtol=2e-6)
            errors.append(float(np.max(np.abs(p[0]-c['edge_scores']))))
            values['scores'] = p
            for tolerance in (1, 2):
                def spatial(q):
                    return {int(q[i]): j for i, j in match_nodes(c['nodes'][q] @ gt['metric'].T, gt['nodes'] @ gt['metric'].T, tolerance).items()}
                normal = spatial(c['retained']); raw = spatial(np.arange(256))
                valid_h = {q: g for (q, g), e in zip(h.items(), physical_error) if e <= tolerance}
                rematched = spatial(np.array(sorted(h), int))
                maps = dict(normal=normal, spatial=raw, hungarian=valid_h, hungarian_rematched=rematched,
                            parent_hungarian={q: g for (q, g), e in zip(parent_h.items(), parent_error) if e <= tolerance})
                for condition, mapping in maps.items():
                    values[f'{condition}_map_{tolerance}'] = mapping_array(mapping)
                common = {q: g for q, g in normal.items() if raw.get(q) == g and valid_h.get(q) == g}
                values[f'common_map_{tolerance}'] = mapping_array(common)
                active, _, labels = pair_labels(c['edges'], mapping_array(normal), gt['edges'], c['retained'])
                if split == 'train':
                    calibration_data[tolerance].append((p[:, active], labels[active], len(gt['edges'])))
            np.savez_compressed(root/split/(name+'.npz'), **values)
            inputs.append(dict(split=split, sample_id=name, sources={str(f): sha(f) for f in files}))
            print('prepared', split, name, flush=True)
    thresholds = {}
    for tolerance, rr in calibration_data.items():
        p = np.concatenate([x[0] for x in rr], axis=1); y = np.concatenate([x[1] for x in rr]); total = sum(x[2] for x in rr)
        baseline = p[0] > .5
        target_precision = float(y[baseline].mean()) if baseline.any() else 0.
        for i, arm in enumerate(ARMS):
            thresholds[f'{arm}/{tolerance}'] = dict(fixed_0p5=.5, precision_target=target_precision,
                **select_threshold(p[i], y, total, target_precision))
    save(root/'thresholds.json', thresholds); table(root/'assignments.csv', assignments)
    save(root/'inputs.json', inputs)
    save(root/'manifest.json', dict(ids=ids, smoke=smoke, seed=SEED, previous=str(PREVIOUS),
        checkpoint=manifest()['checkpoint'], checkpoint_sha256=manifest()['checkpoint_sha256'],
        adapter_sha256=manifest()['adapter_sha256'], matcher=cfg['model']['matcher'],
        heads={a: sha(PREVIOUS/'D'/a/'relation-head.pt') for a in ARMS[1:]},
        score_replay_max_error=max(errors),
        pair_order='Original caches enumerate all 256 choose 2 lexicographically; D cache writer uses exactly that order. D files do not independently store pair IDs.',
        sources={str(p): sha(p) for p in [Path(__file__), Path('models/matcher.py'), Path('models/relationformer.py'),
            Path('training/losses/criterion.py'), Path('scripts/train_ixi_relation_mechanisms.py')]}))
    save(root/'summary.json', dict(completed=True, patches=sum(map(len, ids.values()))))


class Geometry:
    """Frozen spline replay only for uncached curves; never score relations."""
    def __init__(self, cache, spline, features):
        self.c, self.spline, self.features = cache, spline, features
        self.controls = {}
        for e, controls in zip(cache['decoded_query_edges'], cache['decoded_controls']):
            self.controls[tuple(sorted(e))] = controls

    def graph(self, selected, keep):
        c = self.c; pairs = c['edges'][keep]
        missing = np.array([e for e in pairs if tuple(e) not in self.controls], int).reshape(-1, 2)
        with torch.no_grad():
            for chunk in torch.tensor(missing, dtype=torch.long).split(256):
                if not len(chunk):
                    continue
                nodes = torch.tensor(c['nodes']); metric = torch.tensor(c['metric'])
                canonical = chunk.clone(); reverse = canonical_reverse(nodes[chunk] @ metric.T)
                canonical[reverse] = canonical[reverse][:, [1, 0]]
                tokens = torch.tensor(c['tokens']); rel = tokens[256:]
                feature = torch.cat([tokens[canonical[:, 0]], tokens[canonical[:, 1]], rel.flatten()[None].expand(len(chunk), -1)], -1)
                control, _ = self.spline(feature, nodes[canonical], metric, local_features=self.features, input_shape=tuple(c['shape']))
                for e, v in zip(chunk.numpy(), control.numpy()):
                    self.controls[tuple(e)] = v
        if len(pairs):
            reverse = canonical_reverse(torch.tensor(c['nodes'][pairs] @ c['metric'].T)).numpy()
            oriented = pairs.copy(); oriented[reverse] = oriented[reverse][:, [1, 0]]
        else:
            oriented = pairs
        inverse = {int(q): i for i, q in enumerate(selected)}
        edges = np.array([[inverse[int(a)], inverse[int(b)]] for a, b in oriented], int).reshape(-1, 2)
        controls = np.array([self.controls[tuple(e)] for e in pairs], np.float32).reshape(-1, self.spline.controls, 3)
        return dict(nodes=c['nodes'][selected], edges=edges, query_ids=selected, spline_controls=controls, spline_family=self.spline.family)


def selected_queries(data, condition, tolerance):
    if condition == 'normal':
        return data['retained']
    if condition == 'spatial':
        return np.flatnonzero(data[f'spatial_map_{tolerance}'] >= 0)
    if condition == 'parent_hungarian':
        return data['parent_hungarian_queries']
    return data['hungarian_queries']


def record_rank(groups, key, labels, scores, subject):
    groups.setdefault(key, []).append((labels, scores, np.full(len(labels), subject, dtype='U32')))


def write_rank(output, groups):
    rows = []
    directory = output/'ranking'; directory.mkdir()
    for key, chunks in groups.items():
        y = np.concatenate([x[0] for x in chunks]); scores = np.concatenate([x[1] for x in chunks], axis=0)
        subjects = np.concatenate([x[2] for x in chunks])
        np.savez_compressed(directory/('-'.join(map(str, key))+'.npz'), labels=y, scores=scores, subjects=subjects)
        for column in range(scores.shape[1]):
            arm = ARMS[column] if scores.shape[1] == 5 else key[1]
            def summary(mask):
                yy, pp = y[mask], scores[mask, column]
                return dict(candidates=int(mask.sum()), positives=int(yy.sum()), prevalence=float(yy.mean()) if len(yy) else None,
                            ap=float(average_precision_score(yy, pp)) if yy.any() else None)
            mask = np.ones(len(y), bool)
            rows.append(dict(split=key[0], condition=key[1], tolerance_mm=key[2], population=key[3], arm=arm, **summary(mask)))
            if y.any():
                precision, recall, thresholds = precision_recall_curve(y, scores[:, column])
                np.savez_compressed(directory/('-'.join(map(str, key))+f'-{arm}-PR.npz'), precision=precision, recall=recall, thresholds=thresholds)
            for s in sorted(set(subjects)):
                rows.append(dict(split=key[0], condition=key[1], tolerance_mm=key[2], population=key[3], arm=arm, subject=s, **summary(subjects == s)))
    table(output/'ranking.csv', rows)


def analyse(root, index):
    prepared = root/'prepared'; m = json.loads((prepared/'manifest.json').read_text())
    thresholds = json.loads((prepared/'thresholds.json').read_text())
    experiment = 'A' if index == 0 else ARMS[index-1]
    output = root/'analyses'/experiment; output.mkdir(parents=True, exist_ok=False)
    _, spline = heads(); rows, lengths, ranks = [], [], {}
    for split, names in m['ids'].items():
        for name in names:
            with np.load(prepared/split/(name+'.npz')) as f:
                d = {k: f[k] for k in f.files}
            c = read_cache('local_image', split, name); path, gt, mask = target(split, name)
            features = torch.tensor(np.load(PREVIOUS/'features'/split/(name+'.npy')))
            geometry = Geometry(c, spline, features)
            parent_cache = read_cache('parent', split, name) if index == 0 else None
            parent_geometry = Geometry(parent_cache, spline, features) if index == 0 else None
            np.testing.assert_array_equal(d['pairs'], c['edges'])
            subject = name.rsplit('_', 1)[0]
            for tolerance in (1, 2):
                if index == 0:
                    policies = [(condition, condition, .5, d['scores'][0]) for condition in CONDITIONS]
                    # Classifier adjacency targets used in training, explicitly NOT positional correctness.
                    all_h = d['hungarian_assignment']; hq = d['hungarian_queries']
                    _, eligible, y = pair_labels(d['pairs'], all_h, gt['edges'], hq)
                    record_rank(ranks, (split, 'hungarian_training_labels', tolerance, 'assignment_only'), y[eligible], d['scores'][:, eligible].T, subject)
                    common = d[f'common_map_{tolerance}']; q = np.flatnonzero(common >= 0)
                    _, eligible, y = pair_labels(d['pairs'], common, gt['edges'], q)
                    record_rank(ranks, (split, 'common_identity', tolerance, 'conditional'), y[eligible], d['scores'][:, eligible].T, subject)
                else:
                    arm = ARMS[index-1]; probabilities = d['scores'][index-1]
                    plan = thresholds[f'{arm}/{tolerance}']
                    policies = [(p, 'normal', plan[p], probabilities) for p in ['fixed_0p5', 'train_max_f1', 'train_preserve_precision']]
                for policy, condition, cutoff, scores in policies:
                    q = selected_queries(d, condition, tolerance)
                    mapping = d[f'{condition}_map_{tolerance}']
                    active, eligible, y = pair_labels(d['pairs'], mapping, gt['edges'], q)
                    keep = active & (scores > cutoff)
                    parent_condition = condition == 'parent_hungarian'
                    cc = dict(parent_cache if parent_condition else c, edge_scores=scores, edge_selected=scores > cutoff)
                    graph = (parent_geometry if parent_condition else geometry).graph(q, keep)
                    row, strata = measure(cc, graph, path, gt, mask, tolerance,
                        {int(i): int(mapping[i]) for i in np.flatnonzero(mapping >= 0)})
                    meta = dict(split=split, sample_id=name, subject=subject, tolerance_mm=tolerance, policy=policy, threshold=float(cutoff))
                    rows.append(dict(**meta, **row)); lengths.extend(dict(**meta, **v) for v in strata)
                    for r, edge in zip(lengths[-len(strata):] if strata else [], gt['edges']):
                        r['endpoint_distance_mm'] = float(np.linalg.norm((gt['nodes'][edge[0]]-gt['nodes'][edge[1]]) @ gt['metric'].T))
                    # Scores/labels do not depend on the policy; save each ranking population once.
                    if index == 0 or policy == 'fixed_0p5':
                        p = d['scores'].T if index == 0 else scores[:, None]
                        label = condition if index == 0 else arm
                        record_rank(ranks, (split, label, tolerance, 'all_selected'), y[active], p[active], subject)
                        record_rank(ranks, (split, label, tolerance, 'conditional'), y[eligible], p[eligible], subject)
            print(experiment, split, name, flush=True)
    summarize_run(output, rows, lengths, dict(experiment=experiment, ids=m['ids'], smoke=m['smoke']))
    write_rank(output, ranks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'analyse'])
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--index', type=int)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(SEED)
    if args.mode == 'prepare':
        prepare(args.root/'prepared', args.smoke)
    else:
        analyse(args.root, args.index)
