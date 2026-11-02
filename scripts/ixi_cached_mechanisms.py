"""Shared, cache-only decoding for the bounded IXI mechanism diagnostics."""
import csv
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from models.checkpoint import load_checkpoint_file
from models.spline import SplineHead
from training.evaluation.inference import infer_graphs
from metrics.common_vessel_metrics import evaluate_common_vessels, aggregate_common_vessels
from scripts.ixi_query_diagnostics import categorize, edge_accounting, summarize

CACHE_ROOT = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_query_diagnostics_20261019')
CACHES = {a: CACHE_ROOT/f'pilot-{j}-{a}' for a, j in
          [('parent', 1291324), ('baseline', 1291328), ('local_image', 1291329)]}
SEED = 191032


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def table(path, rows):
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(arm='local_image'):
    return json.loads((CACHES[arm]/'manifest.json').read_text())


def identities(smoke=False):
    m = manifest()
    assert not {n.rsplit('_', 1)[0] for n in m['ids']['train']} & {n.rsplit('_', 1)[0] for n in m['ids']['val']}
    for arm in CACHES:
        other = manifest(arm)
        assert other['ids'] == m['ids']
        assert other['checkpoint_sha256'] == m['checkpoint_sha256']
        s = json.loads((CACHES[arm]/'summary.json').read_text())
        assert s['cache_decode_parity'] and s['frozen_state_unchanged']
    return {s: names[:2] if smoke else names for s, names in m['ids'].items()}


def read_cache(arm, split, name):
    path = CACHES[arm]/split/(name+'.npz')
    with np.load(path) as f:
        cache = {k: f[k] for k in f.files}
    features = os.environ.get('IXI_MECHANISM_FEATURES')
    if features:
        cache['local_features'] = np.load(Path(features)/split/(name+'.npy'))
    cache['geometry_reference_nodes'] = cache['nodes']
    return cache


def target(split, name):
    source = json.loads((CACHES['local_image']/split/(name+'-source.json')).read_text())
    path = Path(source['gt'])
    assert sha(path) == source['gt_sha256'], 'GT changed since cached inference'
    with np.load(path) as f:
        gt = {k: f[k] for k in f.files}
    mask = np.asarray(nib.load(source['mask']).dataobj) > .5
    return path, gt, mask


def heads(device='cpu'):
    # Instantiate only two small heads, never the CNN/transformer.
    from models.relationformer import MLP
    m = manifest(); cfg = m['config']
    assert sha(m['checkpoint']) == m['checkpoint_sha256']
    state = load_checkpoint_file(m['checkpoint'])['net']
    width = cfg['model']['decoder']['hidden_dim']
    rel = cfg['model']['decoder']['relation_tokens']
    settings = cfg['model']['spline']
    assert not settings.get('curve_evidence', {}).get('enabled', False)
    relation = MLP(width*(2+rel), width, 2, 3)
    spline = SplineHead(width*(2+rel), width, settings.get('controls', 8),
                        settings.get('samples', 64), settings.get('residual_scale', 'mm'),
                        family=settings.get('family', 'bspline'), token_inputs=settings.get('token_inputs', 'full'),
                        local_geometry=settings.get('local_geometry'))
    for prefix, head in [('relation_embed.', relation), ('spline_head.', spline)]:
        head.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        head.to(device).eval().requires_grad_(False)
    return relation, spline


def decisions(logits, threshold=None):
    t = torch.as_tensor(logits)
    return (t.argmax(-1) == 1 if threshold is None else t.softmax(-1)[:, 1] > threshold).numpy()


def variant(cache, coordinates=True, classification=True):
    c = dict(cache)
    if not coordinates:
        c['nodes'] = cache['original_nodes'][:, :3]
        c['boxes'] = cache['original_nodes']
    if not classification:
        c['node_logits'] = cache['original_logits']
    c['node_scores'] = torch.as_tensor(c['node_logits']).softmax(-1)[:, 1].numpy()
    c['retained'] = np.flatnonzero(decisions(c['node_logits'], manifest()['config']['evaluation']['node_threshold']))
    return c


class CachedRelation:
    """Return immutable cached symmetric logits inside the standard decoder.

    Recover query identities from exact token rows; no GT becomes a relation input.
    Both orientation calls receive the already symmetrized cached value.
    """
    def __init__(self, cache):
        self.width = cache['tokens'].shape[1]
        self.lookup = {t.tobytes(): i for i, t in enumerate(cache['tokens'][:256])}
        assert len(self.lookup) == 256
        self.logits = np.zeros((256, 256, 2), np.float32)
        a, b = cache['edges'].T
        self.logits[a, b] = self.logits[b, a] = cache['pair_logits']

    def __call__(self, features):
        x = features.detach().cpu().numpy()
        a = [self.lookup[v[:self.width].tobytes()] for v in x]
        b = [self.lookup[v[self.width:2*self.width].tobytes()] for v in x]
        return torch.as_tensor(self.logits[a, b], device=features.device)


def decode(cache, spline, selected=None, reuse_geometry=True):
    cfg = manifest()['config']; logits = torch.as_tensor(cache['node_logits']).clone()
    threshold = cfg['evaluation']['node_threshold']
    if selected is not None:
        # GT assists node selection only; coordinates/tokens/relation logits stay fixed.
        logits[:] = torch.tensor([1., -1.]); logits[selected] = torch.tensor([-1., 1.])
        threshold = None
    graph = {k: v.numpy() if torch.is_tensor(v) else v for k, v in infer_graphs(
        torch.as_tensor(cache['tokens'])[None],
        {'pred_nodes': torch.as_tensor(cache['boxes'])[None], 'pred_logits': logits[None],
         **({'spline_local_features': torch.as_tensor(cache['local_features'])[None],
             'spline_input_shape': tuple(cache['shape'])} if spline is not None and spline.local_geometry is not None else {})},
        CachedRelation(cache), object_queries=256,
        relation_tokens=cfg['model']['decoder']['relation_tokens'],
        node_threshold=threshold, edge_threshold=cfg['evaluation']['edge_threshold'],
        spline_head=spline, spline_metrics=[torch.as_tensor(cache['metric'])])[0].items()}
    if spline is not None and reuse_geometry and np.array_equal(cache['nodes'], cache.get('geometry_reference_nodes')):
        old = {tuple(e): i for i, e in enumerate(cache['decoded_query_edges'])}
        for j, edge in enumerate(graph['query_ids'][graph['edges']]):
            i = old.get(tuple(edge))
            if i is not None:
                graph['spline_controls'][j] = cache['decoded_controls'][i]
                graph['spline_curves'][j] = cache['decoded_curves'][i]
    return graph


def parity(cache, spline):
    out = decode(cache, spline, reuse_geometry=False)
    np.testing.assert_array_equal(out['query_ids'], cache['retained'])
    np.testing.assert_array_equal(out['query_ids'][out['edges']], cache['decoded_query_edges'])
    # CPU head replay vs historical GPU head: log displacements, require <=0.001mm.
    delta = (out['spline_controls']-cache['decoded_controls']) @ cache['metric'].T
    error = float(np.linalg.norm(delta, axis=-1).max(initial=0))
    assert error < .001, ('Spline replay drift (mm)', error)
    return error


def measure(cache, graph, path, gt, mask, tolerance, mapping=None):
    q = graph['query_ids']
    found, categories, _, _ = categorize(cache['nodes'], gt['nodes'], gt['metric'], mask, q, tolerance)
    if mapping is None:
        mapping = found
    row, chosen, labels, types, _ = edge_accounting(q, cache['edges'], cache['edge_scores'],
        cache['edge_selected'], mapping, gt['edges'], len(gt['nodes']))
    expected = {tuple(sorted(e)) for e in q[graph['edges']]}
    assert expected == {tuple(e) for e in cache['edges'][chosen]}
    row.update(duplicate_candidates=sum(v == 'duplicate_candidate' for v in categories.values()),
               foreground_extras=sum(v == 'foreground_supported_extra' for v in categories.values()),
               off_vessel=sum(v == 'off_vessel_or_outside' for v in categories.values()))
    near = np.linalg.norm((cache['nodes'][q, None]-gt['nodes'][None]) @ gt['metric'].T, axis=-1) <= tolerance
    row['gt_with_multiple_nearby_queries'] = int((near.sum(0) > 1).sum())
    geo = evaluate_common_vessels(graph, path)
    row.update(geo)
    # Conditional strata on the same positional assignment; boundaries do not change labels.
    inverse = {g: p for p, g in mapping.items()}
    offsets = gt['path_offsets']; physical_paths = gt['full_paths'] @ gt['metric'].T
    lengths = [float(np.linalg.norm(np.diff(physical_paths[a:b], axis=0), axis=1).sum()) for a, b in zip(offsets[:-1], offsets[1:])]
    positives = []
    pair_index = {tuple(e): i for i, e in enumerate(cache['edges'])}
    for (a, b), length in zip(gt['edges'], lengths):
        recovered = a in inverse and b in inverse
        i = pair_index[tuple(sorted((inverse[a], inverse[b])))] if recovered else None
        positives.append(dict(length_mm=length, recovered=recovered, selected=bool(cache['edge_selected'][i]) if recovered else False))
    near_negative = []
    for i in np.flatnonzero((types == 'MM') & ~labels & np.isin(cache['edges'], q).all(1)):
        a, b = cache['edges'][i]
        length = np.linalg.norm((gt['nodes'][mapping[a]]-gt['nodes'][mapping[b]]) @ gt['metric'].T)
        if length <= 5.:
            near_negative.append(bool(chosen[i]))
    row['near_negative_candidates'] = len(near_negative)
    row['near_negative_selected'] = sum(near_negative)
    mm = (types == 'MM') & np.isin(cache['edges'], q).all(1)
    from sklearn.metrics import average_precision_score
    row['conditional_ap'] = float(average_precision_score(labels[mm], cache['edge_scores'][mm])) if labels[mm].any() else None
    return row, positives


def aggregate(rows):
    out = summarize(rows)
    out.update({k: sum(r[k] for r in rows) for k in ['duplicate_candidates', 'foreground_extras', 'off_vessel',
        'gt_with_multiple_nearby_queries', 'near_negative_candidates', 'near_negative_selected']})
    out.update(beta0_mean=float(np.mean([r['beta0'] for r in rows])), beta1_mean=float(np.mean([r['beta1'] for r in rows])))
    geo = aggregate_common_vessels([{k: v for k, v in r.items() if k.startswith('common_')} for r in rows])
    out.update({k: geo[k] for k in ['common_branch_f1', 'common_curve_f1']})
    return out


def summarize_run(output, rows, strata, metadata):
    comparison, subjects = [], []
    keys = sorted({(r['split'], r['tolerance_mm'], r['policy']) for r in rows})
    for split, tol, policy in keys:
        group = [r for r in rows if (r['split'], r['tolerance_mm'], r['policy']) == (split, tol, policy)]
        comparison.append(dict(split=split, tolerance_mm=tol, policy=policy, **aggregate(group)))
        for subject in sorted({r['subject'] for r in group}):
            subjects.append(dict(split=split, tolerance_mm=tol, policy=policy, subject=subject,
                                 **aggregate([r for r in group if r['subject'] == subject])))
    table(output/'per-patch.csv', rows); table(output/'comparison.csv', comparison)
    table(output/'per-subject.csv', subjects); table(output/'positive-lengths.csv', strata)
    save(output/'per-patch.json', rows)
    save(output/'summary.json', dict(**metadata, comparison=comparison, completed=True))


def provenance(ids):
    return dict(ids=ids, checkpoint=manifest()['checkpoint'], checkpoint_sha256=manifest()['checkpoint_sha256'],
                cache_manifests={a: dict(path=str(p/'manifest.json'), sha256=sha(p/'manifest.json'),
                                        adapter_sha256=manifest(a)['adapter_sha256']) for a, p in CACHES.items()},
                sources={str(p): sha(p) for p in Path('scripts').glob('*ixi*mechanism*.py')}, seed=SEED)
